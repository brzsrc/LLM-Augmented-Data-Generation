"""Auto-detect future-leakage columns in proposed state features.

A state column leaks the current action if its values are structurally
constrained by the action — e.g. `resp > 1` only ever co-occurs with `send > 0`
in HeartSteps. Including such a column in DDQN state inflates the No-message
V̂ via OPE extrapolation.

Outputs:
  - prints a report
  - returns a list of suspect columns with diagnostic numbers
  - saves leakage_report.csv to out_dir
"""
from __future__ import annotations
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from src import config as cfg

def detect_leakage(df: pd.DataFrame,
                   out_dir: str = None,
                   min_zero_cells_threshold: int = 1,
                   entropy_threshold: float = 0.30) -> List[Dict]:
    """For each candidate state column, check structural co-occurrence with action.

    Heuristics:
      - zero_cells: count of (col_value, action) cells with 0 samples
      - cond_entropy: H(action | col); low = action determined by col → leakage
    """
    action_col = cfg.COL_ACTION
    candidates = cfg.STATE_FEATURES + cfg.FORBIDDEN_IN_STATE
    candidates = [c for c in candidates if c in df.columns]

    reports = []
    for col in candidates:
        # bin continuous columns into 4 quantiles for cross-tab
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            try:
                s_binned = pd.qcut(s, q=4, duplicates="drop")
            except Exception:
                s_binned = pd.cut(s, bins=4)
        else:
            s_binned = s

        ct = pd.crosstab(s_binned, df[action_col])
        n_cells = ct.shape[0] * ct.shape[1]
        zero_cells = int((ct == 0).sum().sum())

        # conditional entropy H(A|col)
        p_a_given_col = ct.div(ct.sum(axis=1), axis=0).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_a_given_each = -(p_a_given_col * np.log2(p_a_given_col.where(p_a_given_col > 0))).sum(axis=1).fillna(0)
        row_freq = ct.sum(axis=1) / ct.sum().sum()
        cond_ent = float((row_freq * h_a_given_each).sum())

        max_ent = np.log2(len(df[action_col].unique()))
        is_suspect = (zero_cells >= min_zero_cells_threshold
                      and cond_ent < entropy_threshold * max_ent)
        flag = (col in cfg.FORBIDDEN_IN_STATE) or is_suspect

        reports.append({
            "column": col,
            "n_cells": n_cells,
            "zero_cells": zero_cells,
            "cond_entropy": round(cond_ent, 3),
            "max_entropy": round(max_ent, 3),
            "declared_forbidden": col in cfg.FORBIDDEN_IN_STATE,
            "auto_flagged": is_suspect,
            "leakage_suspect": flag,
        })

    df_out = pd.DataFrame(reports)
    print("\n[leakage_detector] Report:")
    print(df_out.to_string(index=False))
    suspects = df_out[df_out["leakage_suspect"]]["column"].tolist()
    in_state = [c for c in suspects if c in cfg.STATE_FEATURES]
    if in_state:
        print(f"\n  ⚠️  WARNING: {len(in_state)} suspect columns ARE in STATE_COLS: "
              f"{in_state}")
        print(f"      Remove these from state_features in your dataset config.")
    else:
        print(f"\n  ✓ State features pass leakage check.")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df_out.to_csv(os.path.join(out_dir, "leakage_report.csv"), index=False)

    return reports
