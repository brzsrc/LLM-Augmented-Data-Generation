"""Eval-only re-evaluation of saved DDQN policies on a UNIFIED, real-anchored
FQE scaler + value-clip — so fixed and learned baselines are comparable ACROSS
training configs.

WHY THIS FILE EXISTS
--------------------
In `policy_utility.run_kfold`, the FQE estimator is re-fit per config with
  (1) a MinMaxScaler fit on that config's TRAINING data (real-not-test + synth), and
  (2) value-clip bounds vmin/vmax from that config's combined (real+synth) reward range.
Both depend on the synthetic data, so the SAME fixed policy (No message / Send a=1 /
Send a=2) evaluated on the SAME real held-out folds gets DIFFERENT FQE values across
configs. Those differences are estimation artifacts, not real value differences
(the fixed policy's true value on real dynamics is config-invariant).

This script removes that artifact by evaluating EVERY config's policies with ONE
shared, real-anchored evaluator:
  * FQE scaler  : per-fold MinMaxScaler fit on REAL-not-test avail=True states
                  (config-invariant, leakage-safe).
  * value-clip  : vmin/vmax from REAL reward range / (1-gamma)  (config-invariant).

It does NOT retrain the DDQN. It loads the saved per-fold checkpoints. Crucially,
each DDQN's greedy ACTIONS (the learned policy) are still computed with that DDQN's
OWN training scaler, so the policy stays faithful; only the FQE *value estimator*
is unified. (Policy-action computation and value estimation are decoupled.)

INPUTS
------
real_df : real data WITH derived features already added
          (run `data_loader.add_derived_features` first; must contain
          uid, send, reward, avail, study_day, slot + all cfg.STATE_FEATURES).
configs : list of dicts, one per training config:
    {
      "name"          : str,                # e.g. "real+LLM"
      "ddqn_dir"      : str,                # dir with ddqn_fold{f}_seed{s}.pt
      "synth_df"      : pd.DataFrame|None,  # the synth df this config trained with
                                            #   (derived features added); None=real-only
      "synth_only"    : bool,               # True if the DDQN trained on synth ONLY
                                            #   (real excluded from training). Default False.
    }
  The synth_df / synth_only fields are used ONLY to rebuild that config's DDQN
  *training* scaler so the argmax matches what was learned. Match these to how the
  config was actually trained.

OUTPUT
------
Writes `<out_dir>/unified_eval_summary.csv`, `<out_dir>/unified_paired_diff.csv`,
`<out_dir>/unified_boot_values.pkl`, and returns a results dict.

All hyperparameters (seed, n_folds, ddqn_seeds, fqe_seeds, fqe_iters, fqe_batch,
bootstrap_B, device, use_amp) read from `cfg.EVAL`; gamma from `cfg.GAMMA`;
state columns from `cfg.STATE_FEATURES` — exactly as in `run_kfold`. Run this with
the SAME cfg.EVAL["seed"] / n_folds you trained with, so make_folds reproduces the
same cross-fit partition (fold-k DDQN evaluated on fold-k held-out real patients).
"""
from __future__ import annotations
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src import config as cfg

# Reuse the EXACT networks / FQE / MDP helpers from the training pipeline.
from src.evaluation.policy_utility import (
    QNet,
    NUM_ACTIONS,
    K_POLICIES,
    POLICY_NAMES,
    DDQN_HIDDEN,
    build_transitions,
    to_arrays,
    filter_trans,
    get_initial_states_with_ids,
    make_folds,
    train_fqe_multi,
    per_patient_values_multi,
    policy_actions_for_states,
    bootstrap_over_patients,
    setup_determinism,
)


def _avail(trans):
    """Keep only decision points where the policy could choose (avail=True),
    mirroring one_fold. next_state pointers were already built across the full
    sequence in build_transitions, so temporal continuity is preserved."""
    return [t for t in trans if t.get("avail", True)]


def _fit_scaler(trans):
    s, *_ = to_arrays(trans)
    return MinMaxScaler().fit(s)


def _ddqn_training_scaler(real_df, synth_df, synth_only, test_pats):
    """Rebuild the per-fold scaler the DDQN was TRAINED with, for this config.

    Mirrors run_kfold: df = concat(real, synth); fold on real uids; synth pinned
    to train; train_pats = all_uids - test_pats; scaler fit on train avail=True.
    If synth_only=True, real is excluded from training entirely (train = synth)."""
    if synth_only:
        if synth_df is None:
            raise ValueError("synth_only=True but synth_df is None")
        df = synth_df
    elif synth_df is not None and len(synth_df) > 0:
        df = pd.concat([real_df, synth_df], ignore_index=True)
    else:
        df = real_df
    trans = build_transitions(df)
    all_pats = set(df["uid"].unique().tolist())
    train_pats = all_pats - set(test_pats)  # synth uids never overlap real test uids
    return _fit_scaler(_avail(filter_trans(trans, train_pats)))


def run_eval_unified(real_df: pd.DataFrame, configs: list, out_dir: str) -> dict:
    e = cfg.EVAL
    gamma = cfg.GAMMA
    dev = torch.device(e["device"])
    feats = list(cfg.STATE_FEATURES)
    state_dim = len(feats)
    use_amp = bool(e.get("use_amp", False)) and dev.type == "cuda"
    os.makedirs(out_dir, exist_ok=True)

    real_uids = set(real_df["uid"].unique().tolist())
    folds = make_folds(real_uids, e["n_folds"], e["seed"])

    # ---- UNIFIED value-clip from REAL reward range (config-invariant) ----
    r_real = real_df["reward"].astype(float).values
    vmin = float(r_real.min() / (1.0 - gamma))
    vmax = float(r_real.max() / (1.0 - gamma))
    print(f">>> unified clip from REAL reward [{r_real.min():.3f},{r_real.max():.3f}] "
          f"-> [{vmin:.2f},{vmax:.2f}] (gamma={gamma})")

    real_trans = build_transitions(real_df)  # full real sequence; avail kept

    # per-config -> {patient_id: value-vector over K policies}
    per_patient = {c["name"]: {} for c in configs}

    for fi, test_pats in enumerate(folds):
        # ---- UNIFIED FQE scaler: real-not-test avail=True (config-invariant) ----
        train_real = real_uids - test_pats
        uni_scaler = _fit_scaler(_avail(filter_trans(real_trans, train_real)))

        # ---- real held-out test set (same across all configs) ----
        test_trans = _avail(filter_trans(real_trans, test_pats))
        pids, init = get_initial_states_with_ids(test_trans)
        init_uni = uni_scaler.transform(init)            # init states in FQE (unified) space
        s_full, _, _, ns_full, _, _ = to_arrays(test_trans)

        for c in configs:
            name = c["name"]
            ddir = c["ddqn_dir"]
            synth_df = c.get("synth_df")
            synth_only = bool(c.get("synth_only", False))

            # ---- this config's OWN training scaler (faithful argmax) ----
            ddqn_scaler = _ddqn_training_scaler(real_df, synth_df, synth_only, test_pats)
            ns_ddqn = torch.tensor(ddqn_scaler.transform(ns_full),
                                   dtype=torch.float32, device=dev)
            init_ddqn = torch.tensor(ddqn_scaler.transform(init),
                                     dtype=torch.float32, device=dev)

            realizations = []
            for ds in range(e["ddqn_seeds"]):
                ckpt = os.path.join(ddir, f"ddqn_fold{fi}_seed{ds}.pt")
                qnet = QNet(state_dim, NUM_ACTIONS, DDQN_HIDDEN).to(dev)
                qnet.load_state_dict(torch.load(ckpt, map_location=dev))
                qnet.eval()

                # Policy actions: head0 = DDQN argmax (computed in the DDQN's OWN
                # scaler space -> faithful policy); heads 1/2/3 = fixed a0/a1/a2.
                pol_a_ns = policy_actions_for_states(qnet, ns_ddqn, K_POLICIES, dev)
                pol_a_init = policy_actions_for_states(qnet, init_ddqn, K_POLICIES, dev)

                for fs in range(e["fqe_seeds"]):
                    setup_determinism(e["seed"] + fi * 1000 + ds * 100 + fs + 1)
                    # FQE value estimator uses the UNIFIED scaler + UNIFIED clip.
                    qe = train_fqe_multi(
                        pol_a_ns, test_trans, uni_scaler, state_dim, K_POLICIES, dev,
                        n_iters=e["fqe_iters"], batch=e["fqe_batch"],
                        gamma=gamma, use_amp=use_amp, vmin=vmin, vmax=vmax,
                    )
                    vals = per_patient_values_multi(
                        qe, pol_a_init, init_uni, dev, vmin=vmin, vmax=vmax
                    )
                    realizations.append(vals)

            # per_patient_values_multi returns [K, n_test]; stack -> [n_real, K, n_test];
            # median over realizations -> [K, n_test]; take all-policies column per patient.
            med = np.median(np.stack(realizations, axis=0), axis=0)  # [K, n_test]
            for i, pid in enumerate(pids):
                per_patient[name][pid] = med[:, i].astype(float)

        print(f"    fold {fi} done ({len(test_pats)} test patients)")

    # ===================== aggregate / report / save =====================
    results = {}
    summary_rows = []
    boot_store = {}
    for c in configs:
        name = c["name"]
        pids = sorted(per_patient[name].keys())
        V = np.array([per_patient[name][p] for p in pids], dtype=float)  # [n, K]
        point = V.mean(axis=0)
        boot = bootstrap_over_patients(V, e["bootstrap_B"], e["seed"])
        boot_store[name] = {"per_patient_values": V, "patient_ids": pids,
                            "boot_matrix": boot, "policy_names": POLICY_NAMES,
                            "point_estimates": dict(zip(POLICY_NAMES, point.tolist()))}
        results[name] = boot_store[name]
        for k, nm in enumerate(POLICY_NAMES):
            lo = float(np.quantile(boot[:, k], 0.025))
            hi = float(np.quantile(boot[:, k], 0.975))
            summary_rows.append({"config": name, "policy": nm,
                                 "point_estimate": float(point[k]),
                                 "ci_low": lo, "ci_high": hi})

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(os.path.join(out_dir, "unified_eval_summary.csv"), index=False)

    # learned vs OWN best-fixed (paired over patients), now on the unified stick
    fixed_idx = [POLICY_NAMES.index(n) for n in ["No message", "Send a=1", "Send a=2"]]
    rng = np.random.default_rng(e["seed"])
    pair_rows = []
    print("\n>>> UNIFIED-scale results (mean over patients):")
    for c in configs:
        name = c["name"]
        V = boot_store[name]["per_patient_values"]
        pt = V.mean(0)
        bf = fixed_idx[int(np.argmax(pt[fixed_idx]))]
        d = V[:, 0] - V[:, bf]
        n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(e["bootstrap_B"])])
        lo, hi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        p_gt = float(np.mean(bs > 0))
        pair_rows.append({"config": name, "learned": float(pt[0]),
                          "best_fixed": POLICY_NAMES[bf], "best_fixed_value": float(pt[bf]),
                          "learned_minus_bestfixed": float(d.mean()),
                          "ci_low": lo, "ci_high": hi, "P_learned_gt_bestfixed": p_gt})
        print(f"    {name:12s} learned={pt[0]:6.2f}  "
              f"[No msg {pt[POLICY_NAMES.index('No message')]:.2f} | "
              f"a1 {pt[POLICY_NAMES.index('Send a=1')]:.2f} | "
              f"a2 {pt[POLICY_NAMES.index('Send a=2')]:.2f}]  "
              f"learned-best({POLICY_NAMES[bf].replace('Send ','')})={d.mean():+.2f} "
              f"P(L>best)={p_gt:.3f}")
    pd.DataFrame(pair_rows).to_csv(os.path.join(out_dir, "unified_paired_diff.csv"), index=False)

    with open(os.path.join(out_dir, "unified_boot_values.pkl"), "wb") as f:
        pickle.dump({"by_config": boot_store, "vmin": vmin, "vmax": vmax,
                     "policy_names": POLICY_NAMES}, f)

    # The fixed baselines SHOULD now be ~config-invariant. Print the spread as a check.
    print("\n>>> config-invariance check on fixed baselines (should be ~flat):")
    for nm in ["No message", "Send a=1", "Send a=2"]:
        ki = POLICY_NAMES.index(nm)
        vals = [boot_store[c["name"]]["per_patient_values"][:, ki].mean() for c in configs]
        print(f"    {nm:12s}: {[round(v,2) for v in vals]}  "
              f"(spread {max(vals)-min(vals):.2f})")

    print(f"\n✅ DONE. Outputs in: {out_dir}")
    return results


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Categorical encoders (only if column is still string)
    for col, mapping in cfg.ENCODERS.items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            unknown = sorted(set(df[col].astype(str).unique()) - set(mapping.keys()))
            if unknown:
                print(f"  WARNING [{col}] unknown values: {unknown}")
            df[col] = df[col].astype(str).map(mapping).fillna(
                mapping.get("other", -1)).astype(int)

    # 2. hour_sin / hour_cos from hr column
    if cfg.COL_HOUR in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df[cfg.COL_HOUR] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df[cfg.COL_HOUR] / 24.0)

    # 3. Dosage (gap-aware via study_day, uses PRIOR action only — no leakage)
    sort_cols = [cfg.COL_PATIENT_ID, cfg.COL_DAY, cfg.COL_SLOT]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    dosages = []
    for _, g in df.groupby(cfg.COL_PATIENT_ID, sort=False):
        d, prev_a, prev_day = 0.0, 0, None
        for a, day in zip(g[cfg.COL_ACTION].values, g[cfg.COL_DAY].values):
            if prev_day is not None and (day - prev_day) > 1:
                d, prev_a = 0.0, 0          # reset on >1 day gap
            d = 0.95 * d + (1 if prev_a > 0 else 0)
            dosages.append(d)
            prev_a = a
            prev_day = day
    df["dosage"] = dosages

    # 4. Reward = log(steps10 + 0.5)
    if cfg.COL_REWARD_SOURCE in df.columns:
        df["reward"] = np.log(df[cfg.COL_REWARD_SOURCE].astype(float)
                              + cfg.REWARD_LOG_OFFSET)

    return df


if __name__ == "__main__":
    # Example wiring — adjust paths and how you load/prepare the data.
    # IMPORTANT: run data_loader.add_derived_features on real_df and each synth_df
    # BEFORE passing them in (so hr_sin/hr_cos/dosage/reward + label-encoded
    # weather/temp/loc columns exist), exactly as you do before run_kfold.

    # Anchor all paths to this script's directory so it works regardless of cwd
    # (the module must be launched from monologue/ for `from src import ...` to
    # resolve, but the data lives under src/evaluation/runs/).
    runs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

    real_df = _add_derived_features(pd.read_csv(os.path.join(runs, "real_only/data_gen.csv")))
    # real_df['source_uid'] = real_df['uid']
    llm_df  = _add_derived_features(pd.read_csv(os.path.join(runs, "llm/synthetic_data.csv")))
    eb_df   = _add_derived_features(pd.read_csv(os.path.join(runs, "eb/synthetic_data.csv")))
    # eb_df['source_uid'] = eb_df['uid'] - 1000

    configs = [
        {"name": "real-only", "ddqn_dir": os.path.join(runs, "real_only/per_fold_ddqn"),
         "synth_df": None,   "synth_only": False},
        {"name": "real+EB",   "ddqn_dir": os.path.join(runs, "eb/abl_with_synth_a0/per_fold_ddqn"),
         "synth_df": eb_df,  "synth_only": False},
        {"name": "EB-only",   "ddqn_dir": os.path.join(runs, "eb/abl_synth_only_a0/per_fold_ddqn"),
         "synth_df": eb_df,  "synth_only": True},
        {"name": "real+LLM",  "ddqn_dir": os.path.join(runs, "llm/abl_with_synth_a0/per_fold_ddqn"),
         "synth_df": llm_df, "synth_only": False},
        {"name": "LLM-only",  "ddqn_dir": os.path.join(runs, "llm/abl_synth_only_a0/per_fold_ddqn"),
         "synth_df": llm_df, "synth_only": True},
    ]
    run_eval_unified(real_df, configs, out_dir=os.path.join(runs, "unified_eval"))
