"""Quality gates for synthetic data.

Five checks (run all; report per-gate pass/fail + diagnostics):
  1. distribution_gate  — mean reward by action within tolerance of real
  2. coverage_gate      — sparse (state,action) cells got more samples
  3. signal_gate        — per-slot action ranking preserved
  4. leakage_gate       — synth+real did not introduce new leakage
  5. consistency_gate   — each synth row's per-persona stats consistent with profile

Failures are surfaced; pipeline.py decides whether to reject vs warn.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.audit.leakage_detector import detect_leakage
from src import config as cfg


def distribution_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame, tol: float = 0.5) -> Tuple[bool, Dict]:
    """Synth's mean reward by action must be within `tol` of real."""
    action_col = cfg.COL_ACTION
    real_means = real_df.groupby(action_col)["reward"].mean()
    synth_means = synth_df.groupby(action_col)["reward"].mean()
    diffs = (synth_means - real_means).abs()
    worst = float(diffs.max()) if len(diffs) else 0.0
    ok = worst <= tol
    diag = {"real_means": real_means.round(3).to_dict(),
            "synth_means": synth_means.round(3).to_dict(),
            "max_abs_diff": round(worst, 3), "tolerance": tol}
    print(f"  [distribution_gate] {'✓ PASS' if ok else '✗ FAIL'}  "
          f"max |diff|={worst:.3f} (tol={tol})")
    return ok, diag


def coverage_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame, slot_col: str = cfg.COL_SLOT,
                  min_added_per_cell: int = 5) -> Tuple[bool, Dict]:
    """Sparse cells in real should gain ≥ min_added samples in real+synth combined."""
    action_col = cfg.COL_ACTION
    real_ct = pd.crosstab(real_df[slot_col], real_df[action_col])
    synth_ct = pd.crosstab(synth_df[slot_col], synth_df[action_col]).reindex(
        index=real_ct.index, columns=real_ct.columns, fill_value=0)
    # cells that were sparse (< 20) in real
    sparse = real_ct[real_ct < 20]
    added = synth_ct.where(real_ct < 20)
    n_sparse = int((real_ct < 20).sum().sum())
    n_filled = int(((real_ct < 20) & (synth_ct >= min_added_per_cell)).sum().sum())
    ratio = n_filled / max(n_sparse, 1)
    ok = ratio >= 0.5
    diag = {"n_real_sparse": n_sparse, "n_filled_by_synth": n_filled,
            "ratio_filled": round(ratio, 2),
            "min_added_threshold": min_added_per_cell}
    print(f"  [coverage_gate]    {'✓ PASS' if ok else '⚠ WARN'}  "
          f"{n_filled}/{n_sparse} sparse cells gained ≥ {min_added_per_cell} synth")
    return ok, diag


def signal_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame, slot_col: str = cfg.COL_SLOT) -> Tuple[bool, Dict]:
    """Per-slot action ranking by mean reward must match between real and synth."""
    action_col = cfg.COL_ACTION
    real = real_df.groupby([slot_col, action_col])["reward"].mean().unstack(action_col)
    synth = synth_df.groupby([slot_col, action_col])["reward"].mean().unstack(action_col)
    common_slots = real.index.intersection(synth.index)
    matches, total = 0, 0
    for slot in common_slots:
        if real.loc[slot].notna().sum() < 2 or synth.loc[slot].notna().sum() < 2:
            continue
        if real.loc[slot].idxmax() == synth.loc[slot].idxmax():
            matches += 1
        total += 1
    ratio = matches / max(total, 1)
    ok = ratio >= 0.6
    diag = {"matching_slots": matches, "total_slots": total,
            "match_ratio": round(ratio, 2)}
    print(f"  [signal_gate]      {'✓ PASS' if ok else '✗ FAIL'}  "
          f"per-slot best action matches in {matches}/{total} slots ({ratio:.0%})")
    return ok, diag


def leakage_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Combined data should not introduce NEW future-leakage columns
    (re-run leakage_detector on real+synth)."""
    combined = pd.concat([real_df, synth_df], ignore_index=True)
    reports = detect_leakage(combined)
    in_state_suspects = [r["column"] for r in reports
                         if r["leakage_suspect"] and r["column"] in cfg.STATE_FEATURES]
    ok = len(in_state_suspects) == 0
    print(f"  [leakage_gate]     {'✓ PASS' if ok else '✗ FAIL'}  "
          f"state-feature suspects: {in_state_suspects or 'none'}")
    return ok, {"in_state_suspects": in_state_suspects}


def temporal_correlation_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                                period: int = 7, tol: float = 0.20
                                ) -> Tuple[bool, Dict]:
    """Health-Gym style temporal fidelity check.

    For each patient, per variable pair (steps10, steps30pre), (steps10, dosage),
    (steps10, send): STL-decompose each variable's daily series, then Kendall's τ
    on the **trend** components. Average τ across patients per pair, then compare
    real vs synth.

      synth is faithful if  max_pair  |τ_real(pair) − τ_synth(pair)| ≤ tol

    Reference: Chen et al., "Health Gym: Synthetic Health Data...", and the
    paper's "Temporal Correlation" evaluation protocol.
    """
    try:
        from statsmodels.tsa.seasonal import STL
        from scipy.stats import kendalltau
    except ImportError:
        print("  [temporal_corr]   ⚠ SKIP  (statsmodels or scipy not installed)")
        return True, {"skipped": True}

    pairs = [("steps10", "steps30pre"), ("steps10", "dosage"),
             ("steps10", cfg.COL_ACTION)]

    def _avg_trend_tau(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        per_pair = {p: [] for p in pairs}
        pid = cfg.COL_PATIENT_ID if cfg.COL_PATIENT_ID in df.columns else "uid"
        if pid not in df.columns: return per_pair
        for uid, g in df.groupby(pid):
            for v1, v2 in pairs:
                if v1 not in g.columns or v2 not in g.columns: continue
                if cfg.COL_DAY not in g.columns: continue
                d1 = g.groupby(cfg.COL_DAY)[v1].mean().astype(float)
                d2 = g.groupby(cfg.COL_DAY)[v2].mean().astype(float)
                idx = d1.index.intersection(d2.index)
                if len(idx) < 2 * period: continue
                d1, d2 = d1.loc[idx].ffill().bfill().fillna(0), \
                          d2.loc[idx].ffill().bfill().fillna(0)
                try:
                    t1 = STL(d1, period=period, robust=True).fit().trend
                    t2 = STL(d2, period=period, robust=True).fit().trend
                except Exception:
                    continue
                tau, _ = kendalltau(t1.values, t2.values)
                if not np.isnan(tau): per_pair[(v1, v2)].append(tau)
        return {p: (float(np.mean(taus)) if taus else float("nan"))
                for p, taus in per_pair.items()}

    real_taus = _avg_trend_tau(real_df)
    synth_taus = _avg_trend_tau(synth_df)

    diffs = {}
    for p in pairs:
        r, s = real_taus.get(p), synth_taus.get(p)
        if r is None or s is None or np.isnan(r) or np.isnan(s): continue
        diffs[f"{p[0]}↔{p[1]}"] = round(abs(r - s), 3)
    max_diff = max(diffs.values()) if diffs else 0.0
    ok = max_diff <= tol
    diag = {
        "real_trend_taus":  {f"{p[0]}↔{p[1]}": round(v, 3)
                              for p, v in real_taus.items()
                              if v is not None and not np.isnan(v)},
        "synth_trend_taus": {f"{p[0]}↔{p[1]}": round(v, 3)
                              for p, v in synth_taus.items()
                              if v is not None and not np.isnan(v)},
        "abs_diff": diffs,
        "max_abs_diff": round(max_diff, 3),
        "tolerance": tol,
    }
    print(f"  [temporal_corr]    {'✓ PASS' if ok else '✗ FAIL'}  "
          f"max |Δτ_trend| = {max_diff:.3f} (tol={tol})")
    return ok, diag


def avail_consistency_gate(synth_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Structural HeartSteps rule: avail=False ⟹ send=0. Hard fail if violated."""
    if "avail" not in synth_df.columns or cfg.COL_ACTION not in synth_df.columns:
        return True, {"reason": "avail or send column missing — skipped"}
    bad = synth_df[(synth_df["avail"] == False) & (synth_df[cfg.COL_ACTION] > 0)]
    ok = len(bad) == 0
    diag = {"n_violations": int(len(bad)),
            "n_total": int(len(synth_df)),
            "violation_rate": round(len(bad) / max(len(synth_df), 1), 4)}
    print(f"  [avail_consistency] {'✓ PASS' if ok else '✗ FAIL'}  "
          f"{len(bad)}/{len(synth_df)} rows have avail=False AND send>0")
    return ok, diag


def consistency_gate(synth_df: pd.DataFrame, personas: List[Dict], tol_steps_factor: float = 3.0) -> Tuple[bool, Dict]:
    """Each synth user's empirical mean_steps10 should be within factor of profile target."""
    bad = []
    for p in personas:
        sub = synth_df[synth_df["uid"] == p["synth_uid"]]
        if len(sub) == 0: continue
        actual = sub["steps10"].mean()
        target = p.get("steps10_mean", actual)
        if target <= 0: continue
        ratio = actual / target
        if ratio > tol_steps_factor or ratio < 1.0 / tol_steps_factor:
            bad.append({"synth_uid": p["synth_uid"], "target": target,
                        "actual": round(actual, 1), "ratio": round(ratio, 2)})
    ok = len(bad) <= 0.1 * len(personas)   # allow ≤ 10% off
    print(f"  [consistency_gate] {'✓ PASS' if ok else '⚠ WARN'}  "
          f"{len(bad)}/{len(personas)} personas off-target by >{tol_steps_factor}x")
    return ok, {"off_target": bad}


def correlation_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                      tol_mae: float = 0.15) -> Tuple[bool, Dict]:
    """Inter-column correlation structure should match real.

    Per "Are LLMs Naturally Good at Synthetic Tabular Data Generation?" (2024),
    LLMs systematically distort column-pair correlations even when marginals
    look OK. Compute Pearson correlation matrices on the columns DDQN sees
    (cfg.STATE_FEATURES + send + steps10), then report Frobenius norm of the
    delta AND the off-diagonal mean absolute deviation.

    Passes if off-diagonal |Δρ| mean ≤ tol_mae (default 0.15).
    """
    cols = list(cfg.STATE_FEATURES) + [cfg.COL_ACTION, cfg.COL_REWARD_SOURCE]
    cols = [c for c in cols if c in real_df.columns and c in synth_df.columns]
    if len(cols) < 3:
        print("  [correlation_gate] ⚠ SKIP (need ≥3 numeric columns)")
        return True, {"skipped": True, "reason": "too few columns"}

    real_corr  = real_df[cols].astype(float).corr().fillna(0.0).values
    synth_corr = synth_df[cols].astype(float).corr().fillna(0.0).values
    diff = real_corr - synth_corr

    n = diff.shape[0]
    frobenius = float(np.linalg.norm(diff, ord="fro"))
    off_diag_mask = ~np.eye(n, dtype=bool)
    mae = float(np.abs(diff[off_diag_mask]).mean())

    # Find the 5 worst-distorted column pairs (upper triangle only)
    triu = np.triu_indices(n, k=1)
    pair_diffs = [(cols[i], cols[j],
                    float(real_corr[i, j]), float(synth_corr[i, j]),
                    float(diff[i, j]))
                   for i, j in zip(*triu)]
    pair_diffs.sort(key=lambda r: abs(r[4]), reverse=True)

    ok = mae <= tol_mae
    diag = {
        "n_columns":            n,
        "frobenius_norm":       round(frobenius, 3),
        "off_diag_mae":         round(mae, 3),
        "tolerance_mae":        tol_mae,
        "worst_5_pairs": [
            {"col1": p[0], "col2": p[1],
             "real_rho":  round(p[2], 3),
             "synth_rho": round(p[3], 3),
             "delta":     round(p[4], 3)}
            for p in pair_diffs[:5]
        ],
    }
    print(f"  [correlation_gate] {'✓ PASS' if ok else '✗ FAIL'}  "
          f"off-diag |Δρ| mean = {mae:.3f} (tol={tol_mae}), "
          f"Frobenius = {frobenius:.2f}")
    return ok, diag


def diversity_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                    n_pairs: int = 200, tol_ratio: float = 0.5,
                    seed: int = 42) -> Tuple[bool, Dict]:
    """Population-level diversity: synth users should be as DIFFERENT from each
    other as real users are.

    Per "LLM as user daily behavior data generator" (2025): we need a DUAL to
    `consistency_gate`. `consistency_gate` measures individual personality
    (each synth_uid matches its persona target); this measures population
    diversity (synth_uids are sufficiently different from each other).

    For both real_df and synth_df: sample n_pairs random uid-pairs, compute
    1-D Wasserstein distance between their steps10 distributions, average.
    Passes if synth's mean pairwise distance is in [tol_ratio, 1/tol_ratio]
    of real's. Synth/real ratio < tol_ratio → mode collapse (all synth too
    similar). Ratio > 1/tol_ratio → over-dispersed (variants too extreme).
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        print("  [diversity_gate]   ⚠ SKIP (scipy not installed)")
        return True, {"skipped": True}

    rng = np.random.default_rng(seed)

    def _mean_pairwise_wd(df: pd.DataFrame) -> float:
        users = sorted(df["uid"].unique())
        if len(users) < 2:
            return 0.0
        all_pairs = [(i, j) for i in range(len(users))
                              for j in range(i + 1, len(users))]
        n_take = min(n_pairs, len(all_pairs))
        chosen = rng.choice(len(all_pairs), n_take, replace=False)
        distances = []
        for k in chosen:
            i, j = all_pairs[k]
            xs = df.loc[df["uid"] == users[i], "steps10"].values
            ys = df.loc[df["uid"] == users[j], "steps10"].values
            if len(xs) >= 5 and len(ys) >= 5:
                distances.append(wasserstein_distance(xs, ys))
        return float(np.mean(distances)) if distances else 0.0

    real_div  = _mean_pairwise_wd(real_df)
    synth_div = _mean_pairwise_wd(synth_df)
    ratio = synth_div / max(real_div, 1e-6)

    ok = tol_ratio <= ratio <= (1.0 / tol_ratio)
    if ratio < tol_ratio:
        verdict = "synth too uniform (mode collapse)"
    elif ratio > 1.0 / tol_ratio:
        verdict = "synth too dispersed (over-perturbed variants?)"
    else:
        verdict = "diversity in range"

    diag = {
        "real_mean_pairwise_wasserstein":  round(real_div, 2),
        "synth_mean_pairwise_wasserstein": round(synth_div, 2),
        "synth_to_real_ratio":             round(ratio, 3),
        "acceptable_range":                [tol_ratio, round(1.0 / tol_ratio, 2)],
        "verdict":                          verdict,
        "n_pairs_sampled":                  n_pairs,
    }
    print(f"  [diversity_gate]   {'✓ PASS' if ok else '⚠ WARN'}  "
          f"synth/real Wasserstein ratio = {ratio:.2f} ({verdict})")
    return ok, diag


def run_all_gates(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                  personas: List[Dict]) -> Dict[str, bool]:
    print("\n[validation] Running quality gates ...")
    results = {}
    results["distribution"],       _ = distribution_gate(real_df, synth_df)
    results["coverage"],           _ = coverage_gate(real_df, synth_df)
    results["signal"],             _ = signal_gate(real_df, synth_df)
    results["leakage"],            _ = leakage_gate(real_df, synth_df)
    results["temporal_corr"],      _ = temporal_correlation_gate(real_df, synth_df)
    results["avail_consistency"],  _ = avail_consistency_gate(synth_df)
    results["consistency"],        _ = consistency_gate(synth_df, personas)
    # New gates added based on lit review:
    #   correlation: "Are LLMs Naturally Good at Synthetic Tabular Data Generation?" (2024)
    #   diversity:   "LLM as user daily behavior data generator" (2025)
    results["correlation"],        _ = correlation_gate(real_df, synth_df)
    results["diversity"],          _ = diversity_gate(real_df, synth_df)
    n_pass = sum(results.values())
    print(f"\n  Gates passed: {n_pass}/{len(results)}")
    return results
