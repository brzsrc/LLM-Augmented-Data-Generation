"""Replication-Fidelity gate suite — 8 gates that measure marginal/structural
similarity between synth and real.

PURPOSE: Diagnostic. These gates answer "does synth statistically resemble
real?" — necessary to prove synth isn't degenerate, but NOT sufficient for
augmentation goals (see gates_augmentation.py).

Paper backing per gate:
  R1  Esteban+ 2017 (RCGAN); Xu+ 2019 (CTGAN §4.1)
  R2  Standard tabular fidelity (corr matrix L1)
  R3  CTGAN §4 (column-conditional moment matching)
  R4  Kuo+ 2022 (Health Gym §4.2); Esteban+ 2017 §4.2
  R5  Zhao+ 2022 (CTAB-GAN+ DCR); Yoon+ 2019 (PATE-GAN privacy)
  R6  Standard schema validation
  R7  Tornqvist+ 2024 (coverage); Kuo+ 2024 (CAT score)
  R8  Ramdas+ 2017 (Wasserstein two-sample test)

Most R gates already exist in gates.py — this module RE-EXPORTS them under
the R-suite naming and adds the two missing ones (R3, R5). The intent is to
make the two-axis framework legible at the file level.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src import config as cfg
from src.validation.gates import (
    statistical_test_gate,            # → R1
    correlation_gate,                 # → R2
    temporal_correlation_gate,        # → R4
    avail_consistency_gate,           # → R6 (partial)
    boundary_coverage_gate,           # → R6 (partial)
    cat_score_gate,                   # → R7
    distributional_alignment_gate,    # → R8
    _to_jsonable,
)


# ============================================================================
# R3 — Conditional moments by action
# ----------------------------------------------------------------------------
# Paper: CTGAN (Xu+ 2019) §4 — "conditional column generation evaluation".
# Test: For each send level, compare mean / std / skew of steps10 between real
# and synth. Marginal mean (existing distribution_gate) only catches gross bias;
# moments expose shape mismatch within action arm.
# ============================================================================
def conditional_moments_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                               target: str = "steps10",
                               group: str | None = None,
                               tol_norm: float = 0.5) -> Tuple[bool, Dict]:
    """R3 — Per-group mean/std/skew of `target` should align within `tol_norm`
    standard deviations of the real moment."""
    from scipy import stats as _scstats
    group = group or cfg.COL_ACTION
    if target not in real_df.columns or group not in real_df.columns:
        return True, {"skipped": True, "reason": "missing columns"}

    per_group: Dict[str, Dict[str, float]] = {}
    worst_diff = 0.0
    for g, r_sub in real_df.groupby(group):
        s_sub = synth_df[synth_df[group] == g]
        if len(r_sub) < 20 or len(s_sub) < 20:
            continue
        r_v, s_v = r_sub[target].astype(float), s_sub[target].astype(float)
        r_mean, s_mean = float(r_v.mean()), float(s_v.mean())
        r_std,  s_std  = float(r_v.std()),  float(s_v.std())
        r_skew = float(_scstats.skew(r_v))  if len(r_v) >= 3 else 0.0
        s_skew = float(_scstats.skew(s_v))  if len(s_v) >= 3 else 0.0
        scale = r_std if r_std > 0 else 1.0
        dm = abs(s_mean - r_mean) / scale
        ds = abs(s_std  - r_std)  / scale
        dk = abs(s_skew - r_skew)
        worst_diff = max(worst_diff, dm, ds)
        per_group[str(g)] = {
            "n_real":  int(len(r_sub)), "n_synth": int(len(s_sub)),
            "real_mean":  round(r_mean, 2), "synth_mean":  round(s_mean, 2),
            "real_std":   round(r_std, 2),  "synth_std":   round(s_std, 2),
            "real_skew":  round(r_skew, 2), "synth_skew":  round(s_skew, 2),
            "mean_diff_norm": round(dm, 3),
            "std_diff_norm":  round(ds, 3),
            "skew_diff":      round(dk, 3),
        }

    ok = worst_diff <= tol_norm
    diag = {"per_group": per_group, "worst_diff_norm": round(worst_diff, 3),
            "tolerance_norm": tol_norm, "target": target, "group_by": group}
    print(f"  [R3 cond_moments]  {'✓ PASS' if ok else '✗ FAIL'}  "
          f"worst |Δmoment/σ| = {worst_diff:.3f} (tol={tol_norm})")
    return ok, diag


# ============================================================================
# R5 — Privacy / non-memorization (DCR + NNDR)
# ----------------------------------------------------------------------------
# Paper: CTAB-GAN+ (Zhao+ 2022) — Distance to Closest Record (DCR).
# Test: For each synth row, find nearest real row in feature space. DCR median
# should be MEANINGFULLY larger than the real-to-real nearest-neighbor baseline
# (NNDR) — otherwise synth is "memorizing" real records.
# ============================================================================
def privacy_dcr_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                      features: List[str] | None = None,
                      n_sample: int = 2000,
                      min_dcr_ratio: float = 0.5,
                      seed: int = 42) -> Tuple[bool, Dict]:
    """R5 — Distance to Closest Record. Synth-to-real median NN distance
    should be ≥ min_dcr_ratio × real-to-real median NN distance.

    ratio close to 1: synth is as "far from real" as real is from itself — good
    ratio < 0.5: synth is closer to real than real is to itself — memorization risk
    ratio >> 1: synth is way off-manifold (different bug, not privacy issue)
    """
    from sklearn.neighbors import NearestNeighbors
    features = features or [c for c in cfg.STATE_FEATURES
                              if c in real_df.columns and c in synth_df.columns]
    if len(features) < 2:
        return True, {"skipped": True, "reason": "too few features"}

    rng = np.random.default_rng(seed)
    real_arr  = real_df[features].astype(float).values
    synth_arr = synth_df[features].astype(float).values
    if len(real_arr) < 10 or len(synth_arr) < 10:
        return True, {"skipped": True, "reason": "too few rows"}

    # subsample for tractability
    ri = rng.choice(len(real_arr),  min(n_sample, len(real_arr)),  replace=False)
    si = rng.choice(len(synth_arr), min(n_sample, len(synth_arr)), replace=False)
    real_arr, synth_arr = real_arr[ri], synth_arr[si]

    # Normalize per feature so all dims contribute equally
    mu, sigma = real_arr.mean(0), real_arr.std(0) + 1e-9
    real_n  = (real_arr  - mu) / sigma
    synth_n = (synth_arr - mu) / sigma

    # NNDR baseline: real → real (k=2 to skip self at distance 0)
    nn_real = NearestNeighbors(n_neighbors=2).fit(real_n)
    d_rr, _ = nn_real.kneighbors(real_n)
    nndr_median = float(np.median(d_rr[:, 1]))

    # DCR: synth → real (k=1)
    nn_s2r = NearestNeighbors(n_neighbors=1).fit(real_n)
    d_sr, _ = nn_s2r.kneighbors(synth_n)
    dcr_median = float(np.median(d_sr[:, 0]))
    dcr_p5     = float(np.percentile(d_sr[:, 0], 5))  # closest tail

    ratio = dcr_median / max(nndr_median, 1e-9)
    ok = ratio >= min_dcr_ratio
    diag = {
        "nndr_median_real_to_real": round(nndr_median, 4),
        "dcr_median_synth_to_real": round(dcr_median, 4),
        "dcr_p5_synth_to_real":     round(dcr_p5, 4),
        "ratio_dcr_to_nndr":        round(ratio, 3),
        "min_ratio":                min_dcr_ratio,
        "n_features":               len(features),
        "n_real_sampled":           len(real_arr),
        "n_synth_sampled":          len(synth_arr),
    }
    print(f"  [R5 privacy_dcr]   {'✓ PASS' if ok else '⚠ WARN'}  "
          f"DCR/NNDR ratio = {ratio:.2f} (tol ≥ {min_dcr_ratio})")
    return ok, diag


# ============================================================================
# Runner
# ============================================================================
def run_replication_suite(real_df: pd.DataFrame, synth_df: pd.DataFrame
                            ) -> Dict:
    """Run all 8 Replication-Fidelity gates. Returns the standard
    {summary, gates} dict.

    R suite is DIAGNOSTIC ONLY — pass/fail is reported but the project's
    overall accept/reject decision should be driven by the augmentation
    suite (gates_augmentation.run_augmentation_suite).
    """
    print("\n[validation] Replication-Fidelity suite (R1-R8) ...")
    specs = [
        ("R1_univariate_marginal",     statistical_test_gate,         (real_df, synth_df)),
        ("R2_pairwise_correlation",    correlation_gate,              (real_df, synth_df)),
        ("R3_conditional_moments",     conditional_moments_gate,      (real_df, synth_df)),
        ("R4_temporal_dynamics",       temporal_correlation_gate,     (real_df, synth_df)),
        ("R5_privacy_dcr",             privacy_dcr_gate,              (real_df, synth_df)),
        ("R6_schema_integrity",        boundary_coverage_gate,        (real_df, synth_df)),
        ("R6b_avail_consistency",      avail_consistency_gate,        (synth_df,)),
        ("R7_support_coverage",        cat_score_gate,                (real_df, synth_df)),
        ("R8_distribution_shape",      distributional_alignment_gate, (real_df, synth_df)),
    ]
    gates_out: Dict[str, Dict] = {}
    for name, fn, args in specs:
        ok, diag = fn(*args)
        gates_out[name] = {"pass": bool(ok), "diag": _to_jsonable(diag)}

    n_pass  = sum(g["pass"] for g in gates_out.values())
    n_total = len(gates_out)
    failed  = [n for n, g in gates_out.items() if not g["pass"]]
    print(f"\n  R suite passed: {n_pass}/{n_total}  (diagnostic only)")
    return {
        "axis":     "replication",
        "summary":  {"n_pass": n_pass, "n_total": n_total, "failed": failed},
        "gates":    gates_out,
    }
