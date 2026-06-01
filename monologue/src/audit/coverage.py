"""Per-(state, action) coverage audit.

For every state feature, cross-tab with action and flag cells below threshold.
These sparse cells are the "holes" that synthetic-data 'edge' variants should
oversample (mechanism B from our framework).

Outputs sparse_cells.csv ranked by smallest count.
"""
from __future__ import annotations
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from src import config as cfg

def _bin(s, n=4):
    try:
        return pd.qcut(s, q=n, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=n)


def audit_coverage(df: pd.DataFrame,
                   out_dir: str = None, min_count: int = 20,
                   n_bins: int = 4) -> Dict:
    """Cross-tab each state feature vs action; report sparse cells."""
    action_col = cfg.COL_ACTION
    state_cols = cfg.STATE_FEATURES

    # bin continuous state columns
    df = df.copy()
    binned_cols = []
    for c in state_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 8:
            df[f"{c}__bin"] = _bin(df[c], n_bins)
            binned_cols.append(f"{c}__bin")
        else:
            binned_cols.append(c)

    sparse_rows = []
    summary = []
    for col in binned_cols:
        ct = pd.crosstab(df[col], df[action_col])
        n_cells = ct.shape[0] * ct.shape[1]
        n_sparse = int((ct < min_count).sum().sum())
        summary.append({"feature": col, "n_cells": n_cells, "n_sparse": n_sparse,
                        "pct_sparse": round(100 * n_sparse / max(n_cells, 1), 1)})
        for fv in ct.index:
            for av in ct.columns:
                n = int(ct.loc[fv, av])
                if n < min_count:
                    sparse_rows.append({"feature": col, "feat_value": str(fv),
                                        "action": int(av), "count": n})

    sparse_df = pd.DataFrame(sparse_rows)
    if len(sparse_df):
        sparse_df = sparse_df.sort_values("count")
    summary_df = pd.DataFrame(summary)

    print("\n[coverage] Univariate sparseness (cell count < {} = sparse):".format(min_count))
    print(summary_df.to_string(index=False))
    print(f"\n  Total sparse cells: {len(sparse_df)}")
    if len(sparse_df):
        print(f"  Top 10 holes (smallest first):")
        print(sparse_df.head(10).to_string(index=False))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        sparse_df.to_csv(os.path.join(out_dir, "sparse_cells.csv"), index=False)
        summary_df.to_csv(os.path.join(out_dir, "coverage_summary.csv"), index=False)

    return {"summary": summary, "sparse_cells": sparse_rows}
