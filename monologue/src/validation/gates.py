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


def _to_jsonable(obj):
    """Recursively convert numpy/pandas/tuple types to JSON-safe primitives.

    - np.bool_ → bool;  np.integer → int;  np.floating → float (NaN/Inf → None)
    - np.ndarray / pd.Index → list;  pd.Series → dict
    - tuple/set → list
    - dict keys that are tuples → "a↔b" string join (e.g. ("steps10", "send"))
    """
    # Order matters: bool before integer (some numpy versions: np.bool_ < np.integer).
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, pd.Series):
        return _to_jsonable(obj.to_dict())
    if isinstance(obj, pd.Index):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                k = "↔".join(str(x) for x in k)
            elif not isinstance(k, (str, int, float, bool)):
                k = str(k)
            out[k] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    return obj


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


def coverage_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                  min_added_per_cell: int = 5,
                  sparse_threshold: int = 20) -> Tuple[bool, Dict]:
    """Sparse cells in real should gain ≥ min_added samples from synth.

    Checked on two finer-grained cell groupings (slot × send × loc × weather and
    slot × send × loc × temp): the original 5×3=15 slot×send grid is too coarse
    to expose actually-sparse cells in HeartSteps real data. Gate passes only
    if BOTH groupings have ≥50% of their sparse cells filled (or have no
    sparse cells at all)."""
    action_col = cfg.COL_ACTION
    slot_col = cfg.COL_SLOT
    groupings = [
        ("slot×send×loc×weather", [slot_col, action_col, "loc", "weather"]),
        ("slot×send×loc×temp",    [slot_col, action_col, "loc", "temp"]),
    ]

    per_grouping = {}
    all_ok = True
    for name, cols in groupings:
        missing = [c for c in cols if c not in real_df.columns or c not in synth_df.columns]
        if missing:
            per_grouping[name] = {"skipped": True, "missing_cols": missing}
            print(f"  [coverage_gate]    ⚠ SKIP  {name}: missing {missing}")
            continue

        real_ct = real_df.groupby(cols).size()
        synth_ct = synth_df.groupby(cols).size().reindex(real_ct.index, fill_value=0)

        sparse_mask = real_ct < sparse_threshold
        n_sparse = int(sparse_mask.sum())
        n_filled = int(((sparse_mask) & (synth_ct >= min_added_per_cell)).sum())

        if n_sparse == 0:
            ok = True                       # no sparse cells → trivially pass
            ratio = 1.0
        else:
            ratio = n_filled / n_sparse
            ok = ratio >= 0.5

        per_grouping[name] = {
            "n_real_sparse": n_sparse,
            "n_filled_by_synth": n_filled,
            "ratio_filled": round(ratio, 2),
            "ok": ok,
        }
        all_ok = all_ok and ok
        tag = "✓ PASS" if ok else "⚠ WARN"
        print(f"  [coverage_gate]    {tag}  {name}: "
              f"{n_filled}/{n_sparse} sparse cells gained ≥ {min_added_per_cell} synth")

    diag = {
        "min_added_threshold": min_added_per_cell,
        "sparse_threshold":    sparse_threshold,
        "per_grouping":        per_grouping,
    }
    return all_ok, diag


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


def hurdle_calibration_gate(synth_df: pd.DataFrame, personas: List[Dict],
                              tol_abs: float = 0.10,
                              max_off_frac: float = 0.10) -> Tuple[bool, Dict]:
    """Each synth user's empirical steps10 zero rate should match the persona's
    modeled overall zero rate (`steps10_zero_pct`).

    Once Python-side `compute_hurdle_p` is the sole source of zeros (LLM is
    schema-forced to positive), these two rates should closely agree. A large
    gap indicates one of:
      - LLM is still emitting 0 despite schema (prompt regression)
      - Python hurdle is mis-weighted, or signals diverge from the marginal
        target zero rate due to state distribution shift
      - Persona's `steps10_zero_pct` itself is mis-fit (rare; extractor bug)

    Pass if ≤ `max_off_frac` of personas have |empirical - target| > tol_abs.
    """
    bad = []
    n_eval = 0
    for p in personas:
        sub = synth_df[synth_df["uid"] == p["synth_uid"]]
        if len(sub) == 0:
            continue
        target = p.get("steps10_zero_pct")
        if target is None:
            continue
        n_eval += 1
        empirical = float((sub["steps10"] == 0).mean())
        diff = abs(empirical - float(target))
        if diff > tol_abs:
            bad.append({"synth_uid": p["synth_uid"],
                        "target":    round(float(target), 3),
                        "empirical": round(empirical, 3),
                        "abs_diff":  round(diff, 3)})
    denom = max(n_eval, 1)
    off_frac = len(bad) / denom
    ok = off_frac <= max_off_frac
    diag = {"n_evaluated":     n_eval,
            "n_off_calibration": len(bad),
            "off_calibration_frac": round(off_frac, 3),
            "tol_abs":          tol_abs,
            "max_off_frac":     max_off_frac,
            "off_users":        bad[:20]}   # cap diag size
    print(f"  [hurdle_calibration] {'✓ PASS' if ok else '⚠ WARN'}  "
          f"{len(bad)}/{n_eval} users have |emp-target zero rate| > {tol_abs}")
    return ok, diag


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


# ============================================================================
# Sim-LLM-style fidelity gates (Haas 2024, Master Thesis §4)
# ----------------------------------------------------------------------------
#   distributional_alignment_gate — per-column WD (continuous) + JSD (categorical)
#   boundary_coverage_gate        — synth values stay within real support
#   cat_score_gate                — real categories all appear in synth
#   statistical_test_gate         — hierarchical KS → Mann-Whitney / χ²
# ============================================================================
def _wd_cols_present(df: pd.DataFrame) -> List[str]:
    return [c for c in cfg.FIDELITY_WD_COLS if c in df.columns]


def _jsd_cols_present(df: pd.DataFrame) -> List[str]:
    return [c for c in cfg.FIDELITY_JSD_COLS if c in df.columns]


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Distance (sqrt of JS divergence), base-2 → [0, 1]."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12);    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask]))))
    js_div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(js_div, 0.0)))


def _aligned_freqs(real_col: pd.Series, synth_col: pd.Series
                    ) -> Tuple[np.ndarray, np.ndarray, List]:
    """Return (real_freq, synth_freq, categories) aligned over union of values."""
    cats = sorted(set(real_col.dropna().unique()) | set(synth_col.dropna().unique()),
                  key=lambda v: (str(type(v)), str(v)))
    r_counts = real_col.value_counts()
    s_counts = synth_col.value_counts()
    r = np.array([r_counts.get(c, 0) for c in cats], dtype=float)
    s = np.array([s_counts.get(c, 0) for c in cats], dtype=float)
    return r, s, cats


def distributional_alignment_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                                   tol_mean_wd_norm: float = 0.5,
                                   tol_mean_jsd: float = 0.3
                                   ) -> Tuple[bool, Dict]:
    """Per-column distributional alignment: WD for continuous, JSD for categorical.

    Continuous cols (FIDELITY_WD_COLS): 1-D Wasserstein distance normalized by
    the real column's std (so the threshold is in std-units, dataset-agnostic).
    Categorical cols (FIDELITY_JSD_COLS): JSD on frequency vectors, base-2 → [0, 1].

    Pass if mean(normalized WD) ≤ tol_mean_wd_norm AND mean(JSD) ≤ tol_mean_jsd.
    Reference: Haas 2024 §4.2.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        print("  [dist_align]       ⚠ SKIP (scipy not installed)")
        return True, {"skipped": True}

    wd_cols  = [c for c in _wd_cols_present(real_df) if c in synth_df.columns]
    jsd_cols = [c for c in _jsd_cols_present(real_df) if c in synth_df.columns]

    per_wd: Dict[str, Dict[str, float]] = {}
    for c in wd_cols:
        r = real_df[c].dropna().astype(float).values
        s = synth_df[c].dropna().astype(float).values
        if len(r) < 5 or len(s) < 5:
            continue
        wd = float(wasserstein_distance(r, s))
        std = float(np.std(r)) or 1.0
        per_wd[c] = {"wd": round(wd, 3),
                     "wd_norm": round(wd / std, 3),
                     "real_std": round(std, 3)}

    per_jsd: Dict[str, Dict[str, float]] = {}
    for c in jsd_cols:
        r, s, _ = _aligned_freqs(real_df[c], synth_df[c])
        if r.sum() == 0 or s.sum() == 0:
            continue
        per_jsd[c] = {"jsd": round(_jsd(r, s), 3),
                       "n_categories": int(len(r))}

    mean_wd_norm = (float(np.mean([v["wd_norm"] for v in per_wd.values()]))
                    if per_wd else 0.0)
    mean_jsd     = (float(np.mean([v["jsd"]      for v in per_jsd.values()]))
                    if per_jsd else 0.0)

    ok_wd  = mean_wd_norm <= tol_mean_wd_norm
    ok_jsd = mean_jsd     <= tol_mean_jsd
    ok = ok_wd and ok_jsd
    diag = {
        "per_column_wd":   per_wd,
        "per_column_jsd":  per_jsd,
        "mean_wd_norm":    round(mean_wd_norm, 3),
        "mean_jsd":        round(mean_jsd, 3),
        "tol_mean_wd_norm": tol_mean_wd_norm,
        "tol_mean_jsd":     tol_mean_jsd,
    }
    print(f"  [dist_align]       {'✓ PASS' if ok else '✗ FAIL'}  "
          f"mean WD(norm)={mean_wd_norm:.3f} (tol={tol_mean_wd_norm}), "
          f"mean JSD={mean_jsd:.3f} (tol={tol_mean_jsd})")
    return ok, diag


def boundary_coverage_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                            tol: float = 0.95) -> Tuple[bool, Dict]:
    """Synth values must stay within real support.

    For FIDELITY_WD_COLS: fraction of synth rows whose value is within
    [real_min, real_max] (no extrapolation). For FIDELITY_JSD_COLS: fraction
    of synth rows whose category exists in real. Pass if min(fraction) ≥ tol.
    Reference: Haas 2024 §4.3.
    """
    wd_cols  = [c for c in _wd_cols_present(real_df) if c in synth_df.columns]
    jsd_cols = [c for c in _jsd_cols_present(real_df) if c in synth_df.columns]

    per_col: Dict[str, Dict] = {}
    for c in wd_cols:
        r = real_df[c].dropna().astype(float)
        s = synth_df[c].dropna().astype(float)
        if len(r) == 0 or len(s) == 0:
            continue
        lo, hi = float(r.min()), float(r.max())
        in_range = ((s >= lo) & (s <= hi)).mean()
        per_col[c] = {"kind": "continuous",
                       "real_min": round(lo, 3), "real_max": round(hi, 3),
                       "frac_in_range": round(float(in_range), 4)}

    for c in jsd_cols:
        r_cats = set(real_df[c].dropna().unique())
        s = synth_df[c].dropna()
        if len(r_cats) == 0 or len(s) == 0:
            continue
        in_set = s.isin(r_cats).mean()
        per_col[c] = {"kind": "categorical",
                       "n_real_categories": int(len(r_cats)),
                       "frac_in_set": round(float(in_set), 4)}

    fracs = [d.get("frac_in_range", d.get("frac_in_set", 1.0))
             for d in per_col.values()]
    worst = float(min(fracs)) if fracs else 1.0
    worst_col = (min(per_col.items(),
                     key=lambda kv: kv[1].get("frac_in_range",
                                                kv[1].get("frac_in_set", 1.0)))[0]
                 if per_col else None)
    ok = worst >= tol
    diag = {
        "per_column":     per_col,
        "worst_col":      worst_col,
        "worst_fraction": round(worst, 4),
        "tolerance":      tol,
    }
    print(f"  [boundary_cov]     {'✓ PASS' if ok else '✗ FAIL'}  "
          f"min frac in-support = {worst:.3f} (worst: {worst_col}, tol={tol})")
    return ok, diag


def cat_score_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                    tol: float = 0.9) -> Tuple[bool, Dict]:
    """Category Coverage (CAT): fraction of real categories that appear in synth.

    Defined per FIDELITY_JSD_COLS column as |real_cats ∩ synth_cats| / |real_cats|.
    Pass if mean CAT across columns ≥ tol. Tracks missing categories per column.
    Reference: Haas 2024 §4.4 (CAT score).
    """
    cols = [c for c in _jsd_cols_present(real_df) if c in synth_df.columns]
    per_col: Dict[str, Dict] = {}
    for c in cols:
        r_cats = set(real_df[c].dropna().unique())
        s_cats = set(synth_df[c].dropna().unique())
        if len(r_cats) == 0:
            continue
        missing = r_cats - s_cats
        cat = (len(r_cats) - len(missing)) / len(r_cats)
        per_col[c] = {
            "n_real_categories":  int(len(r_cats)),
            "n_synth_categories": int(len(s_cats)),
            "n_missing":          int(len(missing)),
            "missing":            sorted([str(m) for m in missing])[:10],
            "cat_score":          round(cat, 3),
        }

    mean_cat = (float(np.mean([d["cat_score"] for d in per_col.values()]))
                if per_col else 1.0)
    ok = mean_cat >= tol
    diag = {"per_column": per_col,
            "mean_cat_score": round(mean_cat, 3),
            "tolerance": tol}
    print(f"  [cat_score]        {'✓ PASS' if ok else '✗ FAIL'}  "
          f"mean CAT = {mean_cat:.3f} (tol={tol})")
    return ok, diag


def statistical_test_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                           alpha: float = 0.05,
                           ks_effect_threshold: float = 0.1,
                           cramers_v_threshold: float = 0.1,
                           pass_threshold: float = 0.5
                           ) -> Tuple[bool, Dict]:
    """Hierarchical statistical tests per column, gated by effect size.

    Continuous (FIDELITY_WD_COLS): KS test; effect size = KS statistic itself
    (max CDF gap, in [0, 1]). If KS rejects, follow-up Mann-Whitney U for
    diagnostic on location shift.
    Categorical (FIDELITY_JSD_COLS): χ² on aligned 2×k frequency table;
    effect size = Cramér's V = sqrt(χ²/n) (since min(2-1, k-1)=1 always here).

    A column is REAL-DIFFERENT only if BOTH:
       p < alpha   (statistically distinguishable)
       AND  effect_size ≥ threshold   (difference is practically meaningful)

    This guards against the large-n trap where KS/χ² reject on trivial
    differences (e.g. Δ=1% on n=30k → p=0 but no one cares).

    A column "passes" if it is NOT real-different. Gate passes if the fraction
    of passing columns ≥ pass_threshold. Reference: Haas 2024 §4.5; effect-size
    convention from Cohen 1988 (small=0.1).
    """
    try:
        from scipy.stats import ks_2samp, mannwhitneyu, chi2_contingency
    except ImportError:
        print("  [stat_test]        ⚠ SKIP (scipy not installed)")
        return True, {"skipped": True}

    wd_cols  = [c for c in _wd_cols_present(real_df) if c in synth_df.columns]
    jsd_cols = [c for c in _jsd_cols_present(real_df) if c in synth_df.columns]

    per_col: Dict[str, Dict] = {}
    n_pass = 0
    n_total = 0

    for c in wd_cols:
        r = real_df[c].dropna().astype(float).values
        s = synth_df[c].dropna().astype(float).values
        if len(r) < 5 or len(s) < 5:
            continue
        ks_stat, ks_p = ks_2samp(r, s)
        ks_stat = float(ks_stat); ks_p = float(ks_p)
        large_effect = ks_stat >= ks_effect_threshold
        sig          = ks_p < alpha
        col_pass     = not (sig and large_effect)
        rec = {"kind":             "continuous",
                "ks_stat":          round(ks_stat, 3),
                "ks_p":             round(ks_p, 4),
                "effect_threshold": ks_effect_threshold,
                "significant":      bool(sig),
                "large_effect":     bool(large_effect),
                "passes":           bool(col_pass)}
        if sig:
            try:
                _, mw_p = mannwhitneyu(r, s, alternative="two-sided")
                rec["mw_p"] = round(float(mw_p), 4)
            except Exception:
                rec["mw_p"] = None
        per_col[c] = rec
        n_total += 1
        if col_pass:
            n_pass += 1

    for c in jsd_cols:
        r, s, cats = _aligned_freqs(real_df[c], synth_df[c])
        if r.sum() < 5 or s.sum() < 5 or len(cats) < 2:
            continue
        table = np.vstack([r, s])
        keep = table.sum(axis=0) > 0
        table = table[:, keep]
        try:
            chi2_stat, chi2_p, _, _ = chi2_contingency(table)
        except ValueError:
            continue
        chi2_stat = float(chi2_stat); chi2_p = float(chi2_p)
        n_total_obs = float(table.sum())
        # Cramér's V for 2×k: sqrt(χ²/n) (since min(2-1, k-1)=1).
        cramers_v   = float(np.sqrt(chi2_stat / max(n_total_obs, 1.0)))
        large_effect = cramers_v >= cramers_v_threshold
        sig          = chi2_p < alpha
        col_pass     = not (sig and large_effect)
        rec = {"kind":             "categorical",
                "chi2_stat":        round(chi2_stat, 2),
                "chi2_p":           round(chi2_p, 4),
                "cramers_v":        round(cramers_v, 3),
                "effect_threshold": cramers_v_threshold,
                "significant":      bool(sig),
                "large_effect":     bool(large_effect),
                "passes":           bool(col_pass),
                "n_categories":     int(len(cats))}
        per_col[c] = rec
        n_total += 1
        if col_pass:
            n_pass += 1

    pass_frac = (n_pass / n_total) if n_total else 1.0
    ok = pass_frac >= pass_threshold

    # Diagnostic: list columns that genuinely differ (significant + large effect)
    real_diff_cols = [c for c, rec in per_col.items() if not rec["passes"]]

    diag = {"per_column":            per_col,
            "alpha":                  alpha,
            "ks_effect_threshold":    ks_effect_threshold,
            "cramers_v_threshold":    cramers_v_threshold,
            "n_columns_tested":       n_total,
            "n_columns_passing":      n_pass,
            "pass_fraction":          round(pass_frac, 3),
            "pass_threshold":         pass_threshold,
            "real_different_cols":    real_diff_cols}
    print(f"  [stat_test]        {'✓ PASS' if ok else '✗ FAIL'}  "
          f"{n_pass}/{n_total} cols OK (sig+effect rule, "
          f"{pass_frac:.0%} ≥ {pass_threshold:.0%}); "
          f"real-diff: {real_diff_cols or 'none'}")
    return ok, diag


def run_all_gates(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                  personas: List[Dict]) -> Dict:
    """Run all gates; return {"summary": {...}, "gates": {name: {pass, diag}}}.

    The full per-gate diagnostic dict (per-column p-values, effect sizes,
    Wasserstein distances, JSD, worst correlation pairs, STL trend τ, etc.) is
    preserved under `gates[name]["diag"]` for downstream inspection.
    """
    print("\n[validation] Running quality gates ...")
    gate_specs = [
        ("distribution",       distribution_gate,             (real_df, synth_df)),
        ("coverage",           coverage_gate,                 (real_df, synth_df)),
        ("signal",             signal_gate,                   (real_df, synth_df)),
        ("leakage",            leakage_gate,                  (real_df, synth_df)),
        ("temporal_corr",      temporal_correlation_gate,     (real_df, synth_df)),
        ("avail_consistency",  avail_consistency_gate,        (synth_df,)),
        ("consistency",        consistency_gate,              (synth_df, personas)),
        # Python-side hurdle calibration: empirical zero rate per user ≈ target
        ("hurdle_calibration", hurdle_calibration_gate,       (synth_df, personas)),
        # Lit-review additions:
        #   correlation — "Are LLMs Naturally Good at Synthetic Tabular Data Generation?" (2024)
        #   diversity   — "LLM as user daily behavior data generator" (2025)
        ("correlation",        correlation_gate,              (real_df, synth_df)),
        ("diversity",          diversity_gate,                (real_df, synth_df)),
        # Sim-LLM fidelity gates (Haas 2024 §4):
        ("dist_align",         distributional_alignment_gate, (real_df, synth_df)),
        ("boundary_cov",       boundary_coverage_gate,        (real_df, synth_df)),
        ("cat_score",          cat_score_gate,                (real_df, synth_df)),
        ("stat_test",          statistical_test_gate,         (real_df, synth_df)),
    ]

    gates_out: Dict[str, Dict] = {}
    for name, fn, args in gate_specs:
        ok, diag = fn(*args)
        gates_out[name] = {"pass": bool(ok), "diag": _to_jsonable(diag)}

    n_pass  = sum(g["pass"] for g in gates_out.values())
    n_total = len(gates_out)
    failed  = [n for n, g in gates_out.items() if not g["pass"]]
    print(f"\n  Gates passed: {n_pass}/{n_total}")
    return {
        "summary": {"n_pass": n_pass, "n_total": n_total, "failed": failed},
        "gates":   gates_out,
    }
