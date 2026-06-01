"""Honest oracle V̂ ceiling via cross-validated per-cell best action.

For increasing state granularities, computes:
  V_naive : best-action-per-cell mean reward × 1/(1-γ)   (overfit)
  V_cv    : same but train/test split per fold           (honest)

V_cv stabilises near the achievable ceiling. If V_cv barely exceeds
"best constant policy", state-dependent policies have little to gain in
this dataset.
"""
from __future__ import annotations
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src import config as cfg

def _bin(s, n=4):
    try:
        return pd.qcut(s, q=n, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=n)


def _naive_oracle(df, state_cols, action_col, reward_col, min_count):
    if not state_cols:
        m = df.groupby(action_col)[reward_col].mean()
        return float(m.max()), 1, 0
    stats = (df.groupby(state_cols + [action_col])[reward_col]
               .agg(["count", "mean"]).reset_index())
    eligible = stats[stats["count"] >= min_count]
    if len(eligible) == 0:
        return float("nan"), 0, 0
    best_df = (eligible.sort_values(state_cols + ["mean"])
                       .drop_duplicates(subset=state_cols, keep="last"))
    best = best_df.set_index(state_cols)["mean"]
    cell_freq = df.groupby(state_cols).size() / len(df)
    fallback = df.groupby(action_col)[reward_col].mean().max()
    n_sparse = 0
    rbar = 0.0
    for k, freq in cell_freq.items():
        if k in best.index:
            rbar += freq * best.loc[k]
        else:
            n_sparse += 1
            rbar += freq * fallback
    return float(rbar), int(len(cell_freq)), n_sparse


def _cv_oracle(df, state_cols, action_col, reward_col, min_count, n_splits=5, seed=42):
    if not state_cols:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scored, weights = [], []
        for tr, te in kf.split(df):
            best_a = df.iloc[tr].groupby(action_col)[reward_col].mean().idxmax()
            matched = df.iloc[te][df.iloc[te][action_col] == best_a]
            if len(matched):
                scored.append(matched[reward_col].mean()); weights.append(len(matched))
        return float(np.average(scored, weights=weights)) if scored else float("nan")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scored, weights = [], []
    fallback = df.groupby(action_col)[reward_col].mean().idxmax()
    for tr, te in kf.split(df):
        tr_df = df.iloc[tr]; te_df = df.iloc[te]
        stats = (tr_df.groupby(state_cols + [action_col])[reward_col]
                       .agg(["count", "mean"]).reset_index())
        eligible = stats[stats["count"] >= min_count]
        if len(eligible) == 0:
            continue
        pres_df = (eligible.sort_values(state_cols + ["mean"])
                            .drop_duplicates(subset=state_cols, keep="last"))
        pres = pres_df.set_index(state_cols)[action_col]
        te_p = te_df.merge(pres.rename("prescribed_a"),
                            left_on=state_cols, right_index=True, how="left")
        te_p["prescribed_a"] = te_p["prescribed_a"].fillna(fallback)
        matched = te_p[te_p[action_col] == te_p["prescribed_a"]]
        if len(matched):
            scored.append(matched[reward_col].mean()); weights.append(len(matched))
    return float(np.average(scored, weights=weights)) if scored else float("nan")


def estimate_ceiling(df: pd.DataFrame,
                     out_dir: str = None, min_count: int = 10,
                     n_bins: int = 4) -> pd.DataFrame:
    """Estimate honest V̂ ceiling at multiple state granularities."""
    df = df.copy()
    action_col = cfg.COL_ACTION
    gamma = cfg.GAMMA
    inv_disc = 1.0 / (1.0 - gamma)

    # bin continuous candidates
    for col in cfg.STATE_FEATURES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 8:
            df[f"{col}_bin"] = _bin(df[col], n_bins)

    slot = cfg.COL_SLOT
    # try increasingly fine state combos
    base_combos = [
        [],
        [slot],
        [slot, "weekday"] if "weekday" in df.columns else None,
        [slot, "dosage_bin"] if "dosage_bin" in df.columns else None,
        [slot, "steps30pre_bin"] if "steps30pre_bin" in df.columns else None,
        [slot, "weekday", "dosage_bin"] if "dosage_bin" in df.columns and "weekday" in df.columns else None,
        [slot, "weekday", "steps30pre_bin", "dosage_bin"]
            if all(c in df.columns for c in ["weekday","steps30pre_bin","dosage_bin"]) else None,
    ]
    combos = [c for c in base_combos if c is not None]

    rows = []
    print(f"\n[oracle_ceiling] γ={gamma}, multiplier=1/(1-γ)={inv_disc:.1f}, "
          f"min_count={min_count}")
    print(f"{'state':50s} {'#cells':>6s} {'r̄_naive':>9s} {'V_naive':>8s} "
          f"{'r̄_cv':>7s} {'V_cv':>7s}")
    print("-" * 90)
    for sc in combos:
        r_naive, n_cells, _ = _naive_oracle(df, sc, action_col, "reward", min_count)
        r_cv = _cv_oracle(df, sc, action_col, "reward", min_count)
        v_n = r_naive * inv_disc
        v_c = r_cv * inv_disc if not np.isnan(r_cv) else float("nan")
        label = "(no state)" if not sc else "+".join(sc)
        print(f"{label:50s} {n_cells:>6d} {r_naive:>9.3f} {v_n:>8.2f} "
              f"{r_cv:>7.3f} {v_c:>7.2f}")
        rows.append({"state_cols": "+".join(sc) if sc else "(none)",
                     "n_cells": n_cells, "r_naive": r_naive, "V_naive": v_n,
                     "r_cv": r_cv, "V_cv": v_c})

    df_out = pd.DataFrame(rows)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df_out.to_csv(os.path.join(out_dir, "oracle_ceiling.csv"), index=False)
    return df_out
