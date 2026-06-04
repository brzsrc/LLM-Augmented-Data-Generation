"""Reconstruct pre-calibration synth CSV from cot_reasoning.jsonl.

Old pipeline runs (run5, run6, run7) overwrote synthetic_data.csv in place
with the calibrated version. The raw LLM `value` per decision is preserved
in cot_reasoning.jsonl, so we can rebuild `synthetic_data_raw.csv` by
joining on (synth_uid, study_day, slot) and replacing only the steps10
column. All other columns (uid, source_uid, variant_type, archetype, hr,
context, etc.) are kept from the existing CSV.

Usage:
    python tests/recover_raw_synth.py run5-clip-cal run6-clip-cal-4key ...
"""
from __future__ import annotations
import json
import os
import sys

import pandas as pd


def recover(run_dir: str) -> None:
    gen = os.path.join(run_dir, "generation")
    csv_in  = os.path.join(gen, "synthetic_data.csv")     # calibrated (in place)
    jsonl   = os.path.join(gen, "cot_reasoning.jsonl")    # raw values
    csv_out = os.path.join(gen, "synthetic_data_raw.csv") # reconstructed pre-cal

    if not os.path.exists(jsonl):
        print(f"  [skip] {run_dir}: no cot_reasoning.jsonl")
        return

    print(f"\n=== {run_dir} ===")
    cal = pd.read_csv(csv_in)
    print(f"  cal csv:  {len(cal)} rows  cols={list(cal.columns)}")

    rows = []
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            rows.append((int(d["synth_uid"]), int(d["study_day"]),
                          int(d["slot"]), int(d["value"])))
    raw_vals = pd.DataFrame(rows,
                             columns=["uid", "study_day", "slot", "raw_steps10"])
    print(f"  jsonl:    {len(raw_vals)} rows")

    merged = cal.merge(raw_vals, on=["uid", "study_day", "slot"], how="left")
    n_missing = int(merged["raw_steps10"].isna().sum())
    if n_missing:
        print(f"  WARNING: {n_missing} rows didn't match jsonl — kept cal value")
        merged["raw_steps10"] = merged["raw_steps10"].fillna(merged["steps10"])

    raw = merged.copy()
    raw["steps10"] = merged["raw_steps10"].astype(int)
    raw = raw.drop(columns=["raw_steps10"])
    raw = raw[cal.columns]                                # match original order

    # Sanity: zero rate per send before vs after
    print(f"  zero-rate per send:")
    print(f"    cal:  {((cal.steps10==0).groupby(cal.send).mean()).round(3).to_dict()}")
    print(f"    raw:  {((raw.steps10==0).groupby(raw.send).mean()).round(3).to_dict()}")
    print(f"  mean steps10 per send:")
    print(f"    cal:  {(cal.groupby('send').steps10.mean()).round(1).to_dict()}")
    print(f"    raw:  {(raw.groupby('send').steps10.mean()).round(1).to_dict()}")

    raw.to_csv(csv_out, index=False)
    print(f"  saved → {csv_out}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    outputs_root = os.path.normpath(os.path.join(here, "..", "src", "outputs"))
    runs = sys.argv[1:] or ["run5-clip-cal", "run6-clip-cal-4key",
                              "run7-clip-cal-0.4"]
    for r in runs:
        recover(os.path.join(outputs_root, r))
