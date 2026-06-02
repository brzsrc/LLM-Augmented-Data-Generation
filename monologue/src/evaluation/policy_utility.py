"""Policy Utility Evaluation — callable K-fold cross-fitted DDQN/FQE.

Inlined from monologue/evaluation/policy_utility_kfold.py so that pipeline.py
can invoke it as a Python function instead of subprocessing the CLI script.
The original CLI script is left untouched as a reference / fallback.

Differences from the CLI version:
  * No argparse / no main() — single entry point `run_kfold(df, out_dir, **kw)`.
  * No dual-GPU multiprocessing — single device only.
  * State columns come from `cfg.STATE_FEATURES` (10-dim incl. hour_sin/cos),
    NOT the CLI script's hard-coded 8-dim STATE_COLS.
  * Caller passes a DataFrame directly; the hard-coded `../data/data_eval.csv`
    read is gone.
  * Optional `debug_dump_csv=True` writes the input df to out_dir for inspection.

Everything else (SWA / multi-seed / CQL / paired-diff CI / power gate / value-clip
FQE) is identical to the source script.
"""
from __future__ import annotations
import json
import os

# CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first CUDA op for
# `torch.use_deterministic_algorithms` to work with cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pickle
import time
from collections import defaultdict
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from src import config as cfg


# ============================================================================
# Constants
# ============================================================================
NUM_ACTIONS = 3
DDQN_LR = 5.5e-5
DDQN_HIDDEN = 128

POLICY_NAMES = ["Original (cross-fit)", "No message", "Send a=1", "Send a=2"]
K_POLICIES = len(POLICY_NAMES)


# ============================================================================
# Determinism helper (Fix 2)
# ============================================================================
def setup_determinism(seed):
    """Lock all RNGs + force deterministic CUDA."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ============================================================================
# Data / MDP
# ============================================================================
def build_transitions(df):
    """Build (s, a, r, s', done, avail) records. State columns from cfg.STATE_FEATURES.

    `avail` is preserved per-transition so one_fold can filter the training set
    to decision points where the policy could actually choose an action
    (HeartSteps convention: policy only learns at avail=True). The next_state
    pointer still crosses avail=False rows, preserving temporal continuity.
    """
    df = df.sort_values(['uid', 'study_day', 'slot']).reset_index(drop=True)
    transitions = []
    has_avail = "avail" in df.columns
    for uid, g in df.groupby('uid', sort=False):
        g = g.reset_index(drop=True)
        S = g[cfg.STATE_FEATURES].values
        A = g["send"].values
        R = g["reward"].values
        AV = g["avail"].values if has_avail else None
        n = len(g)
        for t in range(n):
            ns = S[t + 1] if t < n - 1 else S[t]
            done = 0 if t < n - 1 else 1
            transitions.append({
                "patient_id": uid,
                "s": S[t], "a": int(A[t]), "r": float(R[t]),
                "ns": ns, "done": done,
                "avail": bool(AV[t]) if has_avail else True,
            })
    return transitions


def to_arrays(trans):
    s = np.array([t["s"] for t in trans], dtype=np.float32)
    a = np.array([t["a"] for t in trans], dtype=np.int64)
    r = np.array([t["r"] for t in trans], dtype=np.float32)
    ns = np.array([t["ns"] for t in trans], dtype=np.float32)
    d = np.array([t["done"] for t in trans], dtype=np.float32)
    pid = np.array([t["patient_id"] for t in trans])
    return s, a, r, ns, d, pid


def filter_trans(trans, pats):
    return [t for t in trans if t["patient_id"] in pats]


def get_initial_states_with_ids(trans):
    by_pid = defaultdict(list)
    for t in trans:
        by_pid[t["patient_id"]].append(t)
    pids = list(by_pid.keys())
    init = np.array([by_pid[p][0]["s"] for p in pids], dtype=np.float32)
    return pids, init


def make_folds(patients, n_folds, seed):
    rng = np.random.default_rng(seed)
    pats = list(patients)
    rng.shuffle(pats)
    return [set(pats[i::n_folds]) for i in range(n_folds)]


# ============================================================================
# Networks
# ============================================================================
class QNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class MultiQNet(nn.Module):
    def __init__(self, n_policies, state_dim, num_actions, hidden=64):
        super().__init__()
        K = n_policies
        self.W1 = nn.Parameter(torch.empty(K, state_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(K, 1, hidden))
        self.W2 = nn.Parameter(torch.empty(K, hidden, hidden))
        self.b2 = nn.Parameter(torch.zeros(K, 1, hidden))
        self.W3 = nn.Parameter(torch.empty(K, hidden, num_actions))
        self.b3 = nn.Parameter(torch.zeros(K, 1, num_actions))
        for W in [self.W1, self.W2, self.W3]:
            nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    def forward(self, x):
        h = torch.einsum("bd,kdh->kbh", x, self.W1) + self.b1
        h = F.relu(h)
        h = torch.einsum("kbh,khj->kbj", h, self.W2) + self.b2
        h = F.relu(h)
        out = torch.einsum("kbh,kha->kba", h, self.W3) + self.b3
        return out


# ============================================================================
# DDQN training — no early-stop, SWA over last N checkpoints  (Fix 3 + Fix 5)
# ============================================================================
def train_ddqn(train_trans, scaler, device,
               n_iters=80000, eval_every=6000,
               batch=512, lr=DDQN_LR, hidden=DDQN_HIDDEN,
               gamma=0.95, tau=1e-4, use_amp=False, verbose=False,
               swa_keep=5, cql_alpha=0.0):
    s, a, r, ns, d, _ = to_arrays(train_trans)
    s = scaler.transform(s); ns = scaler.transform(ns)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    ns = torch.tensor(ns, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)

    state_dim = s.shape[1]
    online = QNet(state_dim, NUM_ACTIONS, hidden).to(device)
    target = QNet(state_dim, NUM_ACTIONS, hidden).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)

    N = len(s)
    autocast = (use_amp and device.type == "cuda")
    ckpts = []

    for it in range(n_iters):
        idx = torch.randint(0, N, (batch,), device=device)
        sb, ab, rb, nsb, db = s[idx], a[idx], r[idx], ns[idx], d[idx]
        with torch.amp.autocast(device_type="cuda", enabled=autocast, dtype=torch.bfloat16):
            with torch.no_grad():
                next_a = online(nsb).argmax(dim=1, keepdim=True)
                next_q = target(nsb).gather(1, next_a).squeeze(1)
                y = rb + gamma * (1.0 - db) * next_q
            q_all_at_sb = online(sb)
            q_sa = q_all_at_sb.gather(1, ab.unsqueeze(1)).squeeze(1)
            loss_td = F.smooth_l1_loss(q_sa, y)
            if cql_alpha > 0.0:
                logsumexp_q = torch.logsumexp(q_all_at_sb, dim=1)
                loss_cql = (logsumexp_q - q_sa).mean()
                loss = loss_td + cql_alpha * loss_cql
            else:
                loss = loss_td
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * op.data)

        if (it + 1) % eval_every == 0:
            ckpts.append({k: v.detach().clone() for k, v in online.state_dict().items()})
            if len(ckpts) > swa_keep:
                ckpts.pop(0)
            if verbose:
                print(f"      [ddqn] iter {it+1:6d} loss={loss.item():.4f} "
                      f"(snapshot, kept={len(ckpts)})")

    if not ckpts:
        return online
    avg_state = {}
    for k in ckpts[0].keys():
        stacked = torch.stack([c[k].float() for c in ckpts])
        avg_state[k] = stacked.mean(dim=0).to(ckpts[0][k].dtype)
    online.load_state_dict(avg_state)
    return online


# ============================================================================
# FQE — value-clipped, multi-policy
# ============================================================================
def train_fqe_multi(policy_actions_at_ns, eval_trans, scaler, state_dim,
                    n_policies, device,
                    n_iters=20000, batch=512, lr=4e-3, gamma=0.95, tau=0.009,
                    hidden=64, use_amp=False, vmin=None, vmax=None):
    s, a, r, ns, d, _ = to_arrays(eval_trans)
    s = scaler.transform(s); ns = scaler.transform(ns)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    ns = torch.tensor(ns, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)

    online = MultiQNet(n_policies, state_dim, NUM_ACTIONS, hidden).to(device)
    target = MultiQNet(n_policies, state_dim, NUM_ACTIONS, hidden).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)

    N = len(s)
    autocast = (use_amp and device.type == "cuda")
    K = n_policies

    for it in range(n_iters):
        idx = torch.randint(0, N, (batch,), device=device)
        sb = s[idx]; ab = a[idx]; rb = r[idx]; nsb = ns[idx]; db = d[idx]
        ns_a_all = policy_actions_at_ns[:, idx]

        with torch.amp.autocast(device_type="cuda", enabled=autocast, dtype=torch.bfloat16):
            with torch.no_grad():
                tgt_q_all = target(nsb)
                next_q = tgt_q_all.gather(2, ns_a_all.unsqueeze(-1)).squeeze(-1)
                y = rb.unsqueeze(0) + gamma * (1.0 - db).unsqueeze(0) * next_q
                if vmin is not None:
                    y = y.clamp(vmin, vmax)
            q_all = online(sb)
            ab_exp = ab.unsqueeze(0).expand(K, -1).unsqueeze(-1)
            q_sa = q_all.gather(2, ab_exp).squeeze(-1)
            loss = F.smooth_l1_loss(q_sa, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * op.data)

    return online


def per_patient_values_multi(qnet_eval, policy_actions_at_init, init_states_scaled,
                             device, vmin=None, vmax=None):
    s = torch.tensor(init_states_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_all = qnet_eval(s)
        gathered = q_all.gather(2, policy_actions_at_init.unsqueeze(-1)).squeeze(-1)
        if vmin is not None:
            gathered = gathered.clamp(vmin, vmax)
    return gathered.cpu().numpy()


def policy_actions_for_states(qnet, states_tensor, K, device):
    with torch.no_grad():
        actions = torch.zeros(K, states_tensor.shape[0], dtype=torch.long, device=device)
        actions[0] = qnet(states_tensor).argmax(dim=1)
        actions[1] = 0
        actions[2] = 1
        actions[3] = 2
    return actions


# ============================================================================
# One fold: multi-seed DDQN x multi-seed FQE, median over all realisations
# ============================================================================
def one_fold(fold_idx, train_pats, test_pats, orig_trans, device,
             *, cql_alpha: float, vmin: float, vmax: float,
             save_dir=None):
    """Per-call params: cql_alpha (sweep), vmin/vmax (derived from df).
    All other hyperparameters read from cfg.EVAL directly — no fcfg dict."""
    e = cfg.EVAL
    setup_determinism(e["seed"] + fold_idx)
    use_amp = e["use_amp"] and device.type == "cuda"

    # Filter by patient_id THEN by avail=True. next_state pointers were
    # already computed against the full row sequence in build_transitions, so
    # temporal continuity is preserved even though we drop the avail=False
    # decision points from training/eval (HeartSteps: policy only learns at
    # decision points where it could actually choose an action).
    train_trans = [t for t in filter_trans(orig_trans, train_pats) if t.get("avail", True)]
    test_trans  = [t for t in filter_trans(orig_trans, test_pats)  if t.get("avail", True)]

    s_tr, _, _, _, _, _ = to_arrays(train_trans)
    scaler = MinMaxScaler().fit(s_tr)

    pids, init = get_initial_states_with_ids(test_trans)
    init_s = scaler.transform(init)
    init_t = torch.tensor(init_s, dtype=torch.float32, device=device)
    s_full, _, _, ns_full, _, _ = to_arrays(test_trans)
    ns_t = torch.tensor(scaler.transform(ns_full), dtype=torch.float32, device=device)

    K = K_POLICIES
    all_vals = []
    action_counts = [0, 0, 0]

    for ds in range(e["ddqn_seeds"]):
        torch.manual_seed(e["seed"] + fold_idx * 1000 + ds * 100)
        qnet = train_ddqn(train_trans, scaler, device,
                          n_iters=e["ddqn_iters"], eval_every=e["ddqn_eval_every"],
                          batch=e["ddqn_batch"], lr=DDQN_LR, hidden=DDQN_HIDDEN,
                          gamma=cfg.GAMMA, use_amp=use_amp, verbose=False,
                          swa_keep=e["ddqn_swa_keep"],
                          cql_alpha=cql_alpha)
        if save_dir:
            torch.save(qnet.state_dict(),
                       os.path.join(save_dir, f"ddqn_fold{fold_idx}_seed{ds}.pt"))

        with torch.no_grad():
            ts_t = torch.tensor(scaler.transform(s_full), dtype=torch.float32, device=device)
            ac = qnet(ts_t).argmax(dim=1).cpu().numpy()
        for k in range(NUM_ACTIONS):
            action_counts[k] += int((ac == k).sum())

        pol_a_ns = policy_actions_for_states(qnet, ns_t, K, device)
        pol_a_init = policy_actions_for_states(qnet, init_t, K, device)

        for fs in range(e["fqe_seeds"]):
            torch.manual_seed(e["seed"] + fold_idx * 1000 + ds * 100 + fs + 1)
            qe = train_fqe_multi(pol_a_ns, test_trans, scaler,
                                 len(cfg.STATE_FEATURES), K, device,
                                 n_iters=e["fqe_iters"], batch=e["fqe_batch"],
                                 gamma=cfg.GAMMA, use_amp=use_amp,
                                 vmin=vmin, vmax=vmax)
            vals = per_patient_values_multi(qe, pol_a_init, init_s, device,
                                            vmin=vmin, vmax=vmax)
            all_vals.append(vals)

    med = np.median(np.stack(all_vals, axis=0), axis=0)
    result = {pid: med[:, i].astype(float) for i, pid in enumerate(pids)}
    return result, action_counts


# ============================================================================
# Bootstrap over patients
# ============================================================================
def bootstrap_over_patients(V, B, seed):
    n, K = V.shape
    rng = np.random.default_rng(seed)
    boot = np.empty((B, K), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = V[idx].mean(axis=0)
    return boot


# ============================================================================
# Public API
# ============================================================================
def run_kfold(
    real_df: pd.DataFrame,
    synth_df: Optional[pd.DataFrame] = None,
    *,
    out_dir: str,
    cql_alpha: float,
) -> Dict:
    """Run K-fold cross-fitted DDQN/FQE evaluation.

    Concat + leakage tracking are internal:
      - synth_df=None  →  vanilla K-fold on real_df only
      - synth_df given →  concat(real, synth); fold ONLY on real uids,
                          synth pinned to train forever, never enters test
      NOTE: this version does NOT yet check borrowed_uids leakage — siblings
      whose borrowed peer lands in test can still leak. That's the next step.

    Both halves must contain: uid, send, reward, study_day, slot, plus all
    columns listed in `cfg.STATE_FEATURES`. (Run
    `data_loader.add_derived_features` first to produce
    hour_sin/hour_cos/dosage/reward.)

    All hyperparameters read from `cfg.EVAL` / `cfg.GAMMA` — edit them there,
    not in the call site. Only per-call args: `out_dir`, `cql_alpha`.

    Writes kfold_summary.csv / paired_diff.csv / action_distributions.json /
    boot_values.pkl under `out_dir`. Returns a dict with the same data
    (summary, paired_diff, action_distribution, boot_matrix, per_patient_values).
    """
    e = cfg.EVAL   # alias — all knobs live in cfg.EVAL, read directly below

    os.makedirs(out_dir, exist_ok=True)
    fold_dir = os.path.join(out_dir, "per_fold_ddqn")
    os.makedirs(fold_dir, exist_ok=True)
    
    # ----- Combine real + synth (if any) -----
    # Project to the columns run_kfold actually consumes BEFORE concat, so
    # extra real-only cols (resp) and synth-only cols (source_uid /
    # variant_type / archetype / borrowed_uids) don't leak in as NaN-filled
    # "ghost" columns that promote int dtypes to float64.
    keep_cols = list(cfg.STATE_FEATURES) + ["uid", "send", "reward", "avail"]
    real_uids = set(real_df["uid"].unique().tolist())
    
    if synth_df is not None and len(synth_df) > 0:
        df = pd.concat([real_df[keep_cols], synth_df[keep_cols]], ignore_index=True)
        print(f"[run_kfold] combined: {len(df)} rows, "
              f"real={len(real_uids)} uids, synth={synth_df['uid'].nunique()} uids")
    else:
        df = real_df[keep_cols]
        
    all_pats = set(df["uid"].unique().tolist())
    
    print(f"\n>>> run_kfold: {len(df)} rows "
        f"({len(real_uids)} real + {len(all_pats) - len(real_uids)} synth uids), "
        f"device={e['device']}, n_folds={e['n_folds']}, cql_alpha={cql_alpha}")
    
    # ----- Folds -----
    # Fold candidates: real_uids ∩ patients-in-df (synth never in test).
    folds = make_folds(real_uids, e["n_folds"], e["seed"])
    print(f"fold candidates: {len(real_uids)} real, "
          f"({len(all_pats) - len(real_uids)} synth pinned to train), "  
          f"fold sizes: {[len(f) for f in folds]}")

    # ----- Build MDP -----
    orig_trans = build_transitions(df)
    print(f"    transitions: {len(orig_trans)}")

    # ----- Value-clip bounds from reward range (scales with 1/(1-gamma)) -----
    r_all = df["reward"].astype(float).values
    vmin = float(r_all.min() / (1.0 - cfg.GAMMA))
    vmax = float(r_all.max() / (1.0 - cfg.GAMMA))
    print(f"    reward range [{r_all.min():.3f}, {r_all.max():.3f}] "
          f"-> FQE value clip [{vmin:.2f}, {vmax:.2f}] (gamma={cfg.GAMMA})")

    # ----- Run folds sequentially on single device -----
    t0 = time.time()
    results = {}
    action_counts = [0, 0, 0]
    dev = torch.device(e["device"])
    for fi, test_pats in enumerate(folds):
        train_pats = set(all_pats) - test_pats
        tf = time.time()
        res, counts = one_fold(fi, train_pats, test_pats, orig_trans, dev,
                                cql_alpha=cql_alpha, vmin=vmin, vmax=vmax,
                                save_dir=fold_dir)
        print(f"    fold {fi} done in {time.time()-tf:.1f}s "
              f"({len(res)} test patients)")
        results.update(res)
        for k in range(NUM_ACTIONS):
            action_counts[k] += counts[k]
    print(f"    all folds done in {time.time()-t0:.1f}s; "
          f"{len(results)} patients evaluated")

    # ----- Per-patient value matrix -----
    pids = sorted(results.keys())
    V = np.array([results[p] for p in pids], dtype=float)
    point = V.mean(axis=0)
    print("\n>>> Cross-fitted point estimates (mean over patients):")
    for name, v in zip(POLICY_NAMES, point):
        print(f"    {name:25s} V_hat = {v:.4f}")

    # ----- Bootstrap CI -----
    boot = bootstrap_over_patients(V, e["bootstrap_B"], e["seed"])
    summary_rows = []
    print("\n>>> Per-policy bootstrap CIs (B over patients):")
    for k, name in enumerate(POLICY_NAMES):
        vals = boot[:, k]
        Vhat = float(point[k])
        median = float(np.median(vals))
        errors = vals - Vhat
        ci_lo = float(Vhat - np.quantile(errors, 0.975))
        ci_hi = float(Vhat - np.quantile(errors, 0.025))
        pct_lo = float(np.quantile(vals, 0.025))
        pct_hi = float(np.quantile(vals, 0.975))
        summary_rows.append({"policy": name, "point_estimate": Vhat, "median": median,
                              "ci_low": ci_lo, "ci_high": ci_hi,
                              "pct_ci_low": pct_lo, "pct_ci_high": pct_hi})
        print(f"    {name:25s} Vhat={Vhat:7.3f} median={median:7.3f} "
              f"CI=[{ci_lo:7.3f}, {ci_hi:7.3f}]")
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "kfold_summary.csv"),
                                       index=False)

    # ----- Paired diff -----
    print("\n>>> Paired bootstrap differences (A vs B, diff = A - B):")
    order = sorted(range(K_POLICIES), key=lambda k: np.median(boot[:, k]), reverse=True)
    pair_rows = []
    for ii in range(len(order)):
        for jj in range(ii + 1, len(order)):
            ka, kb = order[ii], order[jj]
            d = boot[:, ka] - boot[:, kb]
            p_gt = float(np.mean(d > 0))
            lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
            crosses0 = bool(lo < 0.0 < hi)
            verdict = "no detectable diff" if crosses0 else f"{POLICY_NAMES[ka]} > {POLICY_NAMES[kb]}"
            pair_rows.append({"A": POLICY_NAMES[ka], "B": POLICY_NAMES[kb],
                               "P(A>B)": p_gt, "median_diff": float(np.median(d)),
                               "ci_low": lo, "ci_high": hi, "crosses_0": crosses0})
            print(f"    {POLICY_NAMES[ka]:20s} vs {POLICY_NAMES[kb]:20s} "
                  f"P={p_gt:.3f} diffCI=[{lo:+.2f},{hi:+.2f}] -> {verdict}")
    pd.DataFrame(pair_rows).to_csv(os.path.join(out_dir, "paired_diff.csv"),
                                    index=False)

    # ----- Power gate -----
    print("\n>>> Power check — can the pipeline separate 'Send a=1'?")
    k_a1 = POLICY_NAMES.index("Send a=1")
    separated = []
    for k, name in enumerate(POLICY_NAMES):
        if k == k_a1:
            continue
        d = boot[:, k] - boot[:, k_a1]
        lo, hi = np.quantile(d, 0.025), np.quantile(d, 0.975)
        ok = not (lo < 0 < hi)
        separated.append(ok)
        flag = "YES" if ok else "no"
        print(f"    {name:25s} > Send a=1 ? {flag}  (diffCI=[{lo:+.2f},{hi:+.2f}])")
    print(f"    => pipeline {'HAS' if any(separated) else 'does NOT yet have'} "
          f"power to separate the extreme policy.")

    # ----- Save raw + action distribution -----
    with open(os.path.join(out_dir, "boot_values.pkl"), "wb") as f:
        pickle.dump({"per_patient_values": V, "patient_ids": pids,
                     "boot_matrix": boot, "policy_names": POLICY_NAMES,
                     "point_estimates": dict(zip(POLICY_NAMES, point.tolist())),
                     "config": {"gamma": cfg.GAMMA,
                                "ddqn_seeds": e["ddqn_seeds"],
                                "fqe_seeds": e["fqe_seeds"],
                                "ddqn_swa_keep": e["ddqn_swa_keep"],
                                "cql_alpha": cql_alpha,
                                "amp": e["use_amp"], "seed": e["seed"],
                                "n_folds": e["n_folds"]}}, f)

    tot = sum(action_counts)
    dist = {f"a={k}": (action_counts[k] / tot if tot else 0.0)
            for k in range(NUM_ACTIONS)}
    print(f"\n>>> Cross-fitted DDQN action distribution "
          f"(averaged over {e['ddqn_seeds']} seeds): {dist}")
    with open(os.path.join(out_dir, "action_distributions.json"), "w") as f:
        json.dump({"original_cross_fit": dist,
                   "averaged_over_ddqn_seeds": e["ddqn_seeds"]}, f, indent=2)

    print(f"\n✅ DONE. Outputs in: {out_dir}")

    # ----- Return everything as a dict (no more parse_kfold_outputs) -----
    return {
        "summary": {row["policy"]: row for row in summary_rows},
        "paired_diff": pair_rows,
        "action_distribution": dist,
        "boot_matrix": boot,
        "per_patient_values": V,
        "patient_ids": pids,
        "point_estimates": dict(zip(POLICY_NAMES, point.tolist())),
    }
