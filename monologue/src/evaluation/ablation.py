"""Ablation runner — compare DDQN trained on (real only) vs (real + synthetic).

Produces the paper-ready table:
    Variant                  Original V̂     vs baseline     status
    -------------------------------------------------------------
    Vanilla DDQN              ...            baseline        -
    + SWA + multi-seed        ...            +X.X            ...
    + CQL                     ...            +X.X            ...
    + Synthetic (this work)   ...            +X.X            ✓ ?
"""
from __future__ import annotations
import os
from typing import Dict, List
import pandas as pd

from src.evaluation.kfold_runner import prepare_eval_csv, run_kfold
from src import config as cfg


def run_ablation(real_df: pd.DataFrame,
                 synth_df: pd.DataFrame,
                 out_root: str,
                 monologue_dir: str,
                 data_eval_path: str = "../data/data_eval.csv",
                 cql_alphas: list = (0.0, 1.0)) -> pd.DataFrame:
    """Run several configs and tabulate Original V̂."""
    os.makedirs(out_root, exist_ok=True)
    abs_data_eval = os.path.join(monologue_dir, "evaluation", data_eval_path).replace("evaluation/..", "")
    # actually policy_utility_kfold.py reads ../data/data_eval.csv RELATIVE TO its cwd
    # which is monologue/evaluation. So target = monologue/data/data_eval.csv.
    target_csv = os.path.join(monologue_dir, "data", "data_eval.csv")

    results = []

    # ---- Baseline: real only, α=0 ----
    prepare_eval_csv(real_df, None, cfg, target_csv)
    for a in cql_alphas:
        out_dir = f"outputs/abl_real_only_a{a:g}"
        res = run_kfold(target_csv, out_dir, cfg,
                        cwd=os.path.join(monologue_dir, "evaluation"),
                        cql_alpha=a)
        results.append({"variant": f"real_only_α={a}", "cql_alpha": a,
                        "synthetic": False, **_extract_v(res)})

    # ---- Real + synthetic ----
    prepare_eval_csv(real_df, synth_df, cfg, target_csv)
    for a in cql_alphas:
        out_dir = f"outputs/abl_with_synth_a{a:g}"
        res = run_kfold(target_csv, out_dir, cfg,
                        cwd=os.path.join(monologue_dir, "evaluation"),
                        cql_alpha=a)
        results.append({"variant": f"real+synth_α={a}", "cql_alpha": a,
                        "synthetic": True, **_extract_v(res)})

    df = pd.DataFrame(results)
    out_csv = os.path.join(out_root, "ablation_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n[ablation] Summary:")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    return df


def _extract_v(kfold_results: Dict) -> Dict:
    """Pull Original / NoMsg V̂ from parsed outputs."""
    summ = kfold_results.get("summary", {})
    orig = summ.get("Original (cross-fit)", {})
    nomsg = summ.get("No message", {})
    return {
        "orig_V": round(orig.get("point_estimate", float("nan")), 2),
        "orig_ci_low": round(orig.get("ci_low", float("nan")), 2),
        "orig_ci_high": round(orig.get("ci_high", float("nan")), 2),
        "nomsg_V": round(nomsg.get("point_estimate", float("nan")), 2),
        "diff_to_nomsg": round(orig.get("point_estimate", float("nan"))
                               - nomsg.get("point_estimate", float("nan")), 2),
    }
