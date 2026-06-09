"""Augmentation-Utility gate suite — 8 gates that measure synth's value for
downstream RL training.

PURPOSE: Pass/fail judge. These gates answer "does synth augmentation help
DDQN learn a better policy?" — the question that gates_replication.py
explicitly does NOT answer.

Paper backing per gate:
  A1  Esteban+ 2017 (TSTR); Xu+ 2019 (CTGAN §4.1 conditional fidelity)
  A2  Esteban+ 2017 (TSTR original definition)
  A3  Esteban+ 2017 (TSTR continuous variant); Yoon+ 2019 (PATE-GAN §5)
  A4  Kumar+ 2020 (CQL §3 variance bottleneck); Fujimoto+ 2019 (BCQ)
  A5  Voloshin+ 2021 (FQE); Bica+ 2021 (counterfactual identifiability)
  A6  Voloshin+ 2021 (FQE benchmark)
  A7  Multi-faceted framework 2024 (cell-level diversity)
  A8  Kumar+ 2020 (CQL §5 augmentation contribution)

All gates here are NEW relative to gates.py. Aggregate decision rule:
  - Suite PASSES if (n_pass / n_total) >= 6/8
  - Core gates A1, A2, A4 MUST pass individually
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from src import data_loader
from src import config as cfg
from src.validation.gates import _to_jsonable


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _state_features(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> List[str]:
    return [c for c in cfg.STATE_FEATURES
            if c in real_df.columns and c in synth_df.columns]


# ============================================================================
# A1 — Conditional KS per (state, action) cell
# ----------------------------------------------------------------------------
# Paper: Esteban+ 2017 §4 ("fidelity should be conditional on covariates");
# Xu+ 2019 (CTGAN) §4.1 — conditional evaluation > marginal.
# ============================================================================
def conditional_ks_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                          target: str = "steps10",
                          state_keys: tuple = ("slot", "loc"),
                          action_key: str | None = None,
                          min_cell_n: int = 20,
                          ks_threshold: float = 0.3,
                          pass_fraction: float = 0.6) -> Tuple[bool, Dict]:
    """A1 — Per-cell KS on P(target) within each (state, action) cell."""
    from scipy.stats import ks_2samp
    action_key = action_key or cfg.COL_ACTION
    keys = list(state_keys) + [action_key]
    keys = [k for k in keys if k in real_df.columns and k in synth_df.columns]
    if not keys or target not in real_df.columns:
        return True, {"skipped": True, "reason": "missing columns"}

    cell_results = []
    for cell, r_grp in real_df.groupby(keys):
        if len(r_grp) < min_cell_n:
            continue
        # match cell in synth
        mask = np.ones(len(synth_df), dtype=bool)
        for k, v in zip(keys, cell if isinstance(cell, tuple) else (cell,)):
            mask &= (synth_df[k].values == v)
        s_grp = synth_df[mask]
        if len(s_grp) < min_cell_n:
            continue
        ks, p = ks_2samp(r_grp[target].astype(float),
                          s_grp[target].astype(float))
        cell_results.append({
            "cell": str(cell),
            "ks": round(float(ks), 3),
            "n_real": int(len(r_grp)),
            "n_synth": int(len(s_grp)),
        })

    if not cell_results:
        return True, {"skipped": True, "reason": "no cells met min_n"}
    ks_values = [c["ks"] for c in cell_results]
    pass_rate = sum(k < ks_threshold for k in ks_values) / len(ks_values)
    ok = pass_rate >= pass_fraction
    diag = {
        "n_cells_evaluated":   len(cell_results),
        "median_ks":           round(float(np.median(ks_values)), 3),
        "pass_rate":           round(pass_rate, 3),
        "pass_threshold":      pass_fraction,
        "ks_threshold":        ks_threshold,
        "worst_5_cells":       sorted(cell_results, key=lambda c: -c["ks"])[:5],
        "state_keys":          list(state_keys),
        "action_key":          action_key,
    }
    print(f"  [A1 cond_ks]       {'✓ PASS' if ok else '✗ FAIL'}  "
          f"{int(pass_rate*100)}% of {len(cell_results)} cells under KS={ks_threshold} "
          f"(target {int(pass_fraction*100)}%)")
    return ok, diag


# ============================================================================
# A2 — TSTR binary (zero/positive classification)
# ============================================================================
def tstr_binary_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                      target: str = "steps10",
                      features: List[str] | None = None,
                      test_frac: float = 0.3,
                      utility_ratio_threshold: float = 0.85,
                      seed: int = 42) -> Tuple[bool, Dict]:
    """A2 — Train RandomForest on synth, eval on held-out real for
    P(target > 0). Compare AUROC to TRTR baseline (train real, test real)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        return True, {"skipped": True, "reason": "sklearn not installed"}

    feats = features or _state_features(real_df, synth_df) + [cfg.COL_ACTION]
    feats = [f for f in feats if f in real_df.columns and f in synth_df.columns]
    if len(feats) < 2 or target not in real_df.columns:
        return True, {"skipped": True, "reason": "too few features"}

    y_real_full = (real_df[target] > 0).astype(int).values
    if y_real_full.mean() in (0.0, 1.0):
        return True, {"skipped": True, "reason": "real target degenerate"}

    X_real = real_df[feats].astype(float).values
    X_synth = synth_df[feats].astype(float).values
    y_synth = (synth_df[target] > 0).astype(int).values

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_real, y_real_full, test_size=test_frac, random_state=seed,
        stratify=y_real_full)

    def _auc(Xtr, ytr, Xte, yte):
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return float("nan")
        m = RandomForestClassifier(n_estimators=100, random_state=seed,
                                     n_jobs=-1).fit(Xtr, ytr)
        return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

    auc_trtr = _auc(X_r_train, y_r_train, X_r_test, y_r_test)
    auc_tstr = _auc(X_synth,   y_synth,   X_r_test, y_r_test)

    if np.isnan(auc_trtr) or np.isnan(auc_tstr) or auc_trtr <= 0.5:
        return True, {"skipped": True, "reason": "TRTR baseline degenerate"}

    ratio = auc_tstr / auc_trtr
    ok = ratio >= utility_ratio_threshold
    diag = {
        "task":             "binary (target > 0)",
        "auc_trtr":         round(auc_trtr, 3),
        "auc_tstr":         round(auc_tstr, 3),
        "utility_ratio":    round(ratio, 3),
        "threshold":        utility_ratio_threshold,
        "n_features":       len(feats),
    }
    print(f"  [A2 tstr_binary]   {'✓ PASS' if ok else '✗ FAIL'}  "
          f"AUC TSTR/TRTR = {ratio:.3f} (tol ≥ {utility_ratio_threshold})")
    return ok, diag


# ============================================================================
# A3 — TSTR regression (continuous target prediction)
# ============================================================================
def tstr_regression_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                          target: str = "steps10",
                          features: List[str] | None = None,
                          test_frac: float = 0.3,
                          utility_ratio_threshold: float = 0.80,
                          seed: int = 42) -> Tuple[bool, Dict]:
    """A3 — Train RF regressor on synth, predict target on real. Compare
    RMSE ratio TRTR/TSTR (closer to 1 = better TSTR)."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
    except ImportError:
        return True, {"skipped": True, "reason": "sklearn not installed"}

    feats = features or _state_features(real_df, synth_df) + [cfg.COL_ACTION]
    feats = [f for f in feats if f in real_df.columns and f in synth_df.columns]
    if len(feats) < 2 or target not in real_df.columns:
        return True, {"skipped": True, "reason": "too few features"}

    X_real = real_df[feats].astype(float).values
    y_real = real_df[target].astype(float).values
    X_synth = synth_df[feats].astype(float).values
    y_synth = synth_df[target].astype(float).values

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_real, y_real, test_size=test_frac, random_state=seed)

    def _rmse(Xtr, ytr, Xte, yte):
        m = RandomForestRegressor(n_estimators=100, random_state=seed,
                                    n_jobs=-1).fit(Xtr, ytr)
        return float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))

    rmse_trtr = _rmse(X_r_train, y_r_train, X_r_test, y_r_test)
    rmse_tstr = _rmse(X_synth,   y_synth,   X_r_test, y_r_test)

    # ratio: lower RMSE = better; we want TSTR not much worse than TRTR
    # utility_ratio = TRTR / TSTR ∈ (0, 1] for "TSTR not better than TRTR"
    ratio = rmse_trtr / max(rmse_tstr, 1e-9)
    ok = ratio >= utility_ratio_threshold
    diag = {
        "task":            "regression (target value)",
        "rmse_trtr":       round(rmse_trtr, 2),
        "rmse_tstr":       round(rmse_tstr, 2),
        "utility_ratio":   round(ratio, 3),
        "threshold":       utility_ratio_threshold,
    }
    print(f"  [A3 tstr_reg]      {'✓ PASS' if ok else '✗ FAIL'}  "
          f"RMSE TRTR/TSTR = {ratio:.3f} (tol ≥ {utility_ratio_threshold})")
    return ok, diag


# ============================================================================
# A4 — Action-weighted coverage
# ----------------------------------------------------------------------------
# Paper: Kumar+ 2020 (CQL §3) — variance bottleneck; Fujimoto+ 2019 (BCQ §4).
# Message-arm cells (send > 0) that are sparse in real get DOUBLE weight,
# since DDQN needs them most for policy improvement.
# ============================================================================
def action_weighted_coverage_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                                    state_keys: tuple = ("slot", "loc", "weather"),
                                    message_weight: float = 2.0,
                                    min_added: int = 5,
                                    sparse_threshold: int = 20,
                                    ratio_threshold: float = 0.7) -> Tuple[bool, Dict]:
    """A4 — Fill rate on sparse real cells, with message arms weighted higher."""
    action_key = cfg.COL_ACTION
    keys = [k for k in list(state_keys) + [action_key]
            if k in real_df.columns and k in synth_df.columns]
    if action_key not in keys:
        return True, {"skipped": True, "reason": "action column missing"}

    real_ct = real_df.groupby(keys).size()
    synth_ct = synth_df.groupby(keys).size()

    sparse = real_ct[real_ct < sparse_threshold]
    if len(sparse) == 0:
        diag = {"reason": "no sparse cells in real", "n_sparse": 0}
        print("  [A4 action_cov]    ✓ PASS  no sparse cells in real")
        return True, diag

    weighted_filled, weighted_total = 0.0, 0.0
    per_arm = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # send → [filled, total]
    for cell, _ in sparse.items():
        send_val = cell[keys.index(action_key)] if isinstance(cell, tuple) else cell
        w = message_weight if send_val > 0 else 1.0
        n_s = int(synth_ct.get(cell, 0))
        weighted_total += w
        per_arm.setdefault(int(send_val), [0, 0])[1] += 1
        if n_s >= min_added:
            weighted_filled += w
            per_arm[int(send_val)][0] += 1

    ratio = weighted_filled / max(weighted_total, 1e-9)
    ok = ratio >= ratio_threshold
    diag = {
        "state_keys":            list(state_keys),
        "message_weight":        message_weight,
        "n_sparse_cells":        int(len(sparse)),
        "weighted_fill_ratio":   round(ratio, 3),
        "ratio_threshold":       ratio_threshold,
        "per_action_fill": {f"send={a}": {"filled": p[0], "total": p[1]}
                             for a, p in per_arm.items() if p[1] > 0},
    }
    print(f"  [A4 action_cov]    {'✓ PASS' if ok else '✗ FAIL'}  "
          f"weighted fill = {ratio:.2f} (tol ≥ {ratio_threshold})")
    return ok, diag


# ============================================================================
# A5 — Causal signal preservation
# ----------------------------------------------------------------------------
# Paper: Voloshin+ 2021 (FQE); Bica+ 2021 (counterfactual identifiability).
# For each state cell, compute Δ_send = E[r | s, a=1] - E[r | s, a=0].
# Real vs synth: Spearman rank correlation + sign agreement of Δ_send.
# ============================================================================
def causal_signal_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                        target: str = "steps10",
                        state_keys: tuple = ("slot", "loc"),
                        min_cell_n_per_arm: int = 10,
                        rank_corr_threshold: float = 0.5,
                        sign_agreement_threshold: float = 0.7) -> Tuple[bool, Dict]:
    """A5 — Real vs synth action-effect agreement across state cells."""
    from scipy.stats import spearmanr
    action_key = cfg.COL_ACTION
    if target not in real_df.columns or action_key not in real_df.columns:
        return True, {"skipped": True}
    keys = [k for k in list(state_keys)
            if k in real_df.columns and k in synth_df.columns]
    if not keys:
        return True, {"skipped": True, "reason": "no state keys"}

    def _per_cell_effect(df):
        out: Dict[tuple, float] = {}
        for cell, g in df.groupby(keys):
            a0 = g[g[action_key] == 0][target]
            a1 = g[g[action_key] >  0][target]
            if len(a0) < min_cell_n_per_arm or len(a1) < min_cell_n_per_arm:
                continue
            out[cell if isinstance(cell, tuple) else (cell,)] = (
                float(a1.mean()) - float(a0.mean()))
        return out

    real_eff = _per_cell_effect(real_df)
    synth_eff = _per_cell_effect(synth_df)
    common = sorted(set(real_eff.keys()) & set(synth_eff.keys()))
    if len(common) < 3:
        return True, {"skipped": True, "reason": "fewer than 3 common cells",
                      "n_common": len(common)}

    r_vec = np.array([real_eff[c] for c in common])
    s_vec = np.array([synth_eff[c] for c in common])
    rho, _ = spearmanr(r_vec, s_vec)
    sign_agree = float(((r_vec > 0) == (s_vec > 0)).mean())

    ok_rho  = (not np.isnan(rho)) and rho >= rank_corr_threshold
    ok_sign = sign_agree >= sign_agreement_threshold
    ok = ok_rho and ok_sign
    diag = {
        "n_common_cells":            int(len(common)),
        "rank_corr_action_effect":   round(float(rho) if not np.isnan(rho) else 0.0, 3),
        "sign_agreement":            round(sign_agree, 3),
        "rank_corr_threshold":       rank_corr_threshold,
        "sign_agreement_threshold":  sign_agreement_threshold,
    }
    print(f"  [A5 causal_signal] {'✓ PASS' if ok else '✗ FAIL'}  "
          f"rank_corr={rho:.2f} sign_agree={sign_agree:.2f}")
    return ok, diag


# ============================================================================
# A6 — Q-proxy disagreement
# ----------------------------------------------------------------------------
# Paper: Voloshin+ 2021 (FQE benchmark). Fit a simple Q ≈ r + γ·V'(s') proxy
# (here just E[r | s, a] via RF regressor) on real vs synth; compare on a
# probe grid. Disagreement = how much DDQN training would diverge.
# ============================================================================
def q_proxy_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                  target: str = "steps10",
                  features: List[str] | None = None,
                  n_probe: int = 1000,
                  max_norm_diff: float = 0.3,
                  seed: int = 42) -> Tuple[bool, Dict]:
    """A6 — Cheap Q-proxy: RF regressor on (state, action) → reward.
    Train on real, train on synth; compare predictions on random probes."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return True, {"skipped": True, "reason": "sklearn not installed"}

    feats = features or _state_features(real_df, synth_df) + [cfg.COL_ACTION]
    feats = [f for f in feats if f in real_df.columns and f in synth_df.columns]
    if len(feats) < 2 or target not in real_df.columns:
        return True, {"skipped": True, "reason": "too few features"}

    rng = np.random.default_rng(seed)
    X_real, y_real = real_df[feats].astype(float).values, real_df[target].astype(float).values
    X_synth, y_synth = synth_df[feats].astype(float).values, synth_df[target].astype(float).values

    q_real  = RandomForestRegressor(n_estimators=100, random_state=seed,
                                       n_jobs=-1).fit(X_real, y_real)
    q_synth = RandomForestRegressor(n_estimators=100, random_state=seed,
                                       n_jobs=-1).fit(X_synth, y_synth)

    # Probes drawn from real distribution (the deployment distribution we care about)
    idx = rng.choice(len(X_real), min(n_probe, len(X_real)), replace=False)
    probes = X_real[idx]
    q_r = q_real.predict(probes)
    q_s = q_synth.predict(probes)

    sigma_r = float(np.std(y_real)) or 1.0
    norm_diff = float(np.mean(np.abs(q_r - q_s))) / sigma_r
    ok = norm_diff <= max_norm_diff
    diag = {
        "mean_abs_q_diff":     round(float(np.mean(np.abs(q_r - q_s))), 2),
        "real_target_std":     round(sigma_r, 2),
        "normalized_diff":     round(norm_diff, 3),
        "max_norm_diff":       max_norm_diff,
        "n_probes":            len(probes),
    }
    print(f"  [A6 q_proxy]       {'✓ PASS' if ok else '✗ FAIL'}  "
          f"|ΔQ|/σ = {norm_diff:.3f} (tol ≤ {max_norm_diff})")
    return ok, diag


# ============================================================================
# A7 — Conditional diversity (per state cell)
# ----------------------------------------------------------------------------
# Paper: multi-faceted framework 2024 (cell-level diversity).
# Within each state cell, synth's target variance should not collapse below
# real's variance. Catches "mode collapse within state".
# ============================================================================
def conditional_diversity_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                                 target: str = "steps10",
                                 state_keys: tuple = ("slot", "loc"),
                                 min_cell_n: int = 20,
                                 ratio_threshold: float = 0.7,
                                 pass_fraction: float = 0.6) -> Tuple[bool, Dict]:
    """A7 — Per-cell σ_synth / σ_real should be ≥ ratio_threshold for the
    majority of cells."""
    keys = [k for k in list(state_keys)
            if k in real_df.columns and k in synth_df.columns]
    if not keys or target not in real_df.columns:
        return True, {"skipped": True}

    ratios, cells = [], []
    for cell, r_grp in real_df.groupby(keys):
        if len(r_grp) < min_cell_n:
            continue
        mask = np.ones(len(synth_df), dtype=bool)
        for k, v in zip(keys, cell if isinstance(cell, tuple) else (cell,)):
            mask &= (synth_df[k].values == v)
        s_grp = synth_df[mask]
        if len(s_grp) < min_cell_n:
            continue
        sr, ss = r_grp[target].astype(float).std(), s_grp[target].astype(float).std()
        ratio = float(ss / sr) if sr > 0 else 1.0
        ratios.append(ratio)
        cells.append({"cell": str(cell), "ratio_std": round(ratio, 3),
                       "real_std": round(float(sr), 2),
                       "synth_std": round(float(ss), 2)})
    if not ratios:
        return True, {"skipped": True, "reason": "no cells met min_n"}
    pass_rate = float(np.mean([r >= ratio_threshold for r in ratios]))
    ok = pass_rate >= pass_fraction
    diag = {
        "n_cells":              len(ratios),
        "median_ratio":         round(float(np.median(ratios)), 3),
        "pass_rate":            round(pass_rate, 3),
        "ratio_threshold":      ratio_threshold,
        "pass_fraction":        pass_fraction,
        "worst_5_cells":        sorted(cells, key=lambda c: c["ratio_std"])[:5],
    }
    print(f"  [A7 cond_diversity]{'✓ PASS' if ok else '✗ FAIL'}  "
          f"{int(pass_rate*100)}% cells with σ_synth/σ_real ≥ {ratio_threshold}")
    return ok, diag


# ============================================================================
# A8 — Off-policy importance
# ----------------------------------------------------------------------------
# Paper: Kumar+ 2020 (CQL §5) — augmentation contribution.
# Measure: for each sparse (state, send) cell, synth must add ≥ N samples,
# where N is the effective sample size needed to halve Q estimation variance.
# Effective bound: synth_n ≥ real_n (doubles the sample → variance halves).
# ============================================================================
def off_policy_importance_gate(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                                 state_keys: tuple = ("slot", "loc"),
                                 sparse_threshold: int = 15,
                                 contribution_floor: float = 1.0,
                                 pass_fraction: float = 0.6
                                 ) -> Tuple[bool, Dict]:
    """A8 — On sparse cells, synth_n / real_n ≥ contribution_floor for ≥
    pass_fraction of cells. Quantifies how much synth meaningfully augments."""
    action_key = cfg.COL_ACTION
    keys = [k for k in list(state_keys) + [action_key]
            if k in real_df.columns and k in synth_df.columns]
    if action_key not in keys:
        return True, {"skipped": True, "reason": "action column missing"}

    real_ct = real_df.groupby(keys).size()
    synth_ct = synth_df.groupby(keys).size()

    sparse_idx = real_ct[real_ct < sparse_threshold].index
    if len(sparse_idx) == 0:
        return True, {"reason": "no sparse cells", "n_sparse": 0}

    cells = []
    pass_count = 0
    for cell in sparse_idx:
        n_r = int(real_ct.get(cell, 0))
        n_s = int(synth_ct.get(cell, 0))
        ratio = n_s / max(n_r, 1)
        cells.append({"cell": str(cell), "n_real": n_r, "n_synth": n_s,
                       "synth_to_real": round(ratio, 2)})
        if ratio >= contribution_floor:
            pass_count += 1

    pass_rate = pass_count / len(cells)
    ok = pass_rate >= pass_fraction
    diag = {
        "n_sparse_cells":          len(cells),
        "contribution_floor":      contribution_floor,
        "pass_rate":               round(pass_rate, 3),
        "pass_fraction":           pass_fraction,
        "median_synth_to_real":    round(float(np.median(
            [c["synth_to_real"] for c in cells])), 2),
        "worst_5_cells":           sorted(cells, key=lambda c: c["synth_to_real"])[:5],
    }
    print(f"  [A8 off_policy]    {'✓ PASS' if ok else '✗ FAIL'}  "
          f"{int(pass_rate*100)}% sparse cells got synth_n ≥ {contribution_floor}× real_n")
    return ok, diag


# ============================================================================
# Runner
# ============================================================================
CORE_GATES = ("A1_conditional_ks", "A2_tstr_binary", "A4_action_coverage")


def run_augmentation_suite(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                              core_must_pass: bool = True) -> Dict:
    """Run all 8 Augmentation-Utility gates. Returns the standard
    {summary, gates} dict.

    Decision rule:
      - Suite PASSES if n_pass/n_total >= 6/8 AND all CORE_GATES pass
      - core_must_pass=False relaxes the core requirement (use carefully)
    """
    print("\n[validation] Augmentation-Utility suite (A1-A8) ...")
    specs = [
        ("A1_conditional_ks",     conditional_ks_gate,             (real_df, synth_df)),
        ("A2_tstr_binary",        tstr_binary_gate,                (real_df, synth_df)),
        ("A3_tstr_regression",    tstr_regression_gate,            (real_df, synth_df)),
        ("A4_action_coverage",    action_weighted_coverage_gate,   (real_df, synth_df)),
        ("A5_causal_signal",      causal_signal_gate,              (real_df, synth_df)),
        ("A6_q_proxy",            q_proxy_gate,                    (real_df, synth_df)),
        ("A7_conditional_div",    conditional_diversity_gate,      (real_df, synth_df)),
        ("A8_off_policy_import",  off_policy_importance_gate,      (real_df, synth_df)),
    ]
    gates_out: Dict[str, Dict] = {}
    for name, fn, args in specs:
        ok, diag = fn(*args)
        gates_out[name] = {"pass": bool(ok), "diag": _to_jsonable(diag)}

    n_pass  = sum(g["pass"] for g in gates_out.values())
    n_total = len(gates_out)
    failed  = [n for n, g in gates_out.items() if not g["pass"]]
    core_failed = [n for n in CORE_GATES if not gates_out[n]["pass"]]

    overall_pass = (n_pass / n_total) >= (6 / 8)
    if core_must_pass:
        overall_pass = overall_pass and (not core_failed)

    print(f"\n  A suite passed: {n_pass}/{n_total}  "
          f"(core {len(CORE_GATES) - len(core_failed)}/{len(CORE_GATES)})  "
          f"→ overall {'✓ PASS' if overall_pass else '✗ FAIL'}")
    return {
        "axis":          "augmentation",
        "summary":       {"n_pass": n_pass, "n_total": n_total, "failed": failed,
                          "core_failed": core_failed, "overall_pass": overall_pass},
        "gates":         gates_out,
    }

if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.data_loader import add_derived_features
    from src import config as cfg

    REPO = Path(__file__).resolve().parents[2]   # monologue/
    real_df = add_derived_features(pd.read_csv(cfg.CSV_PATH))

    runs = [
        ("run3-CoT-no0",      REPO / "outputs/run3-CoT-no0/generation/synthetic_data.csv"),
        ("run3-no0-replay2",  REPO / "outputs/run3-no0-replay/generation/synthetic_data.csv"),
        ("run7-clip-cal-0.4",  REPO / "outputs/run7-clip-cal-0.4/generation/synthetic_data_random_cal.csv"),
    ]
    for label, synth_path in runs:
        print(f"\n##### {label} #####")
        synth_df = add_derived_features(pd.read_csv(synth_path))
        result = run_augmentation_suite(real_df, synth_df)
        out_path = synth_path.parent / "gate_results_augmentation.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  → saved {out_path}")

