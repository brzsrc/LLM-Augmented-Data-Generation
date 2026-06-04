"""Evaluation orchestration: combine real + synth, run kfold, sweep ablations.

This is the only file pipeline.py needs to import for evaluation. It wraps
`policy_utility.run_kfold` (in-process, no subprocess) with:

  prepare_eval_df  — schema-normalise real + synth into one DataFrame
  run_ablation     — sweep real_only × cql_alphas, then real+synth × cql_alphas

Combines what used to be `kfold_runner.py` and `ablation.py`.
"""
from __future__ import annotations
import gc
import os
from typing import Dict, List, Optional

import pandas as pd

from src import config as cfg
from src import data_loader
from src.evaluation import policy_utility

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False



# ============================================================================
# Ablation sweep
# ============================================================================
def _extract_v(res: Dict) -> Dict:
    """Pull Original / No-msg V̂ from the run_kfold result dict."""
    summary = res.get("summary", {})
    orig = summary.get("Original (cross-fit)", {})
    nomsg = summary.get("No message", {})
    orig_pt = orig.get("point_estimate", float("nan"))
    nomsg_pt = nomsg.get("point_estimate", float("nan"))
    return {
        "orig_V":       round(orig_pt, 2),
        "orig_ci_low":  round(orig.get("ci_low", float("nan")), 2),
        "orig_ci_high": round(orig.get("ci_high", float("nan")), 2),
        "nomsg_V":      round(nomsg_pt, 2),
        "diff_to_nomsg": round(orig_pt - nomsg_pt, 2)
                          if (orig_pt == orig_pt and nomsg_pt == nomsg_pt)  # NaN check
                          else float("nan"),
    }


def _release_gpu():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_ablation(real_df: pd.DataFrame,
                 synth_df: Optional[pd.DataFrame],
                 out_root: str) -> pd.DataFrame:
    """Run real_only × cfg.EVAL['cql_alphas'] and (if synth_df given)
    real+synth × cfg.EVAL['cql_alphas'].

    All hyperparameters come from `cfg.EVAL` — edit there, not here.

    Writes `ablation_summary.csv` under `out_root` and returns the same dataframe.
    """
    os.makedirs(out_root, exist_ok=True)
    rows: List[Dict] = []
    cql_alphas = cfg.EVAL["cql_alphas"]

    has_synth = synth_df is not None and len(synth_df) > 0

    for a in cql_alphas:
        # Real only
        # out_dir = os.path.join(out_root, f"abl_real_only_a{a:g}")
        # res = policy_utility.run_kfold(real_df, synth_df=None,
        #                                 out_dir=out_dir, cql_alpha=a)
        # rows.append({"variant": f"real_only_α={a:g}", "cql_alpha": a,
        #               "synthetic": False, **_extract_v(res)})
        # _release_gpu()

        # Real + synth (leakage tracking is internal to run_kfold)
        if has_synth:
            out_dir = os.path.join(out_root, f"abl_with_synth_a{a:g}")
            res = policy_utility.run_kfold(real_df, synth_df=synth_df,
                                            out_dir=out_dir, cql_alpha=a)
            rows.append({"variant": f"real+synth_α={a:g}", "cql_alpha": a,
                          "synthetic": True, **_extract_v(res)})
            _release_gpu()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_root, "ablation_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n[runner] Ablation summary:")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    return df
