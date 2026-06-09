"""Run DDQN ablation on the three Bayesian-smoothing baseline variants.

For each variant in {replace, replace_4x, oversample}, load the Bayesian-
generated synthetic_data.csv and run the same ablation sweep that produced
run3's ablation_summary.csv. Outputs go under each variant's evaluation/
folder, matching the layout of monologue/outputs/run3-CoT-no0/.

Usage (from monologue/):
    python -m src.ablations.run_bayes_ablation
    python -m src.ablations.run_bayes_ablation --variant replace   # one only
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from src import data_loader
from src.evaluation import runner as eval_runner


# monologue/  (parents: ablations -> src -> monologue)
MONOLOGUE = Path(__file__).resolve().parents[2]

VARIANTS = (
    "replace_4x", "replace", "oversample",       # cell-EB baselines + volume control
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(MONOLOGUE / "outputs/bayes_baseline"))
    ap.add_argument("--variant", choices=VARIANTS + ("all",), default="all")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    print("[load] real:", end=" ")
    real = data_loader.add_derived_features(
        pd.read_csv(MONOLOGUE / "data/data_gen.csv"))
    print(f"{len(real)} rows")

    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for v in variants:
        synth_csv = out_root / v / "generation" / "synthetic_data.csv"
        if not synth_csv.exists():
            print(f"\n[skip] {v}: missing {synth_csv}")
            print(f"       run bayesian_smoothing_baseline.py first")
            continue

        print(f"\n{'=' * 72}")
        print(f"  Variant: {v}")
        print(f"{'=' * 72}")
        synth = data_loader.add_derived_features(pd.read_csv(synth_csv))
        print(f"[load] synth: {len(synth)} rows  (from {synth_csv})")

        eval_out = out_root / v / "evaluation"
        eval_out.mkdir(parents=True, exist_ok=True)
        eval_runner.run_ablation(real, synth, out_root=str(eval_out))

    # Aggregate comparison
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY: Bayesian baselines vs LLM (run3)")
    print(f"{'=' * 72}")
    rows = []
    for v in variants:
        f = out_root / v / "evaluation" / "ablation_summary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["variant_group"] = v
        rows.append(df)

    if rows:
        combined = pd.concat(rows, ignore_index=True)
        print(combined.to_string(index=False))
        out_combined = out_root / "ablation_summary_combined.csv"
        combined.to_csv(out_combined, index=False)
        print(f"\nSaved combined: {out_combined}")


if __name__ == "__main__":
    main()
