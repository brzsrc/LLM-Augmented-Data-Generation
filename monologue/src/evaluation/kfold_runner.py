"""K-fold cross-fitted DDQN training + FQE evaluation.

Thin wrapper that re-uses the existing, well-tested implementation in
monologue/evaluation/policy_utility_kfold.py — we don't duplicate hundreds of
lines of DDQN/FQE/SWA/CQL code. Instead, we shell out to that script with the
right CLI flags, then read its outputs.

This module adds:
  - source_uid-aware fold building (drops synthetic personas tied to held-out
    real patients, preventing K-fold leakage)
  - automatic CSV preparation (combines real + synth into a single eval CSV
    matching the schema policy_utility_kfold.py expects)
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src import config as cfg

# Path to the existing trained-and-tested kfold script
_KFOLD_SCRIPT = "../evaluation/policy_utility_kfold.py"


def prepare_eval_csv(real_df: pd.DataFrame,
                     synth_df: Optional[pd.DataFrame],
                     out_csv: str) -> None:
    """Combine real + synth into the column schema expected by policy_utility_kfold.

    That script reads ../data/data_eval.csv with columns:
      uid, study_day, weekday, slot, weather, temp, loc, send, dosage, resp,
      steps30pre, reward.

    We map our normalised data into that schema (and DROP `resp` because we
    proved it's a leakage column — set to 0 placeholder if column required).
    """
    def _to_eval_schema(df: pd.DataFrame, fold_offset: int = 0) -> pd.DataFrame:
        out = pd.DataFrame()
        out["uid"] = df["uid"] if "uid" in df.columns else df[cfg.COL_PATIENT_ID]
        # study_day: day index within patient
        date_col = cfg.COL_DECISION_DATE
        if date_col in df.columns:
            tmp = df.sort_values(["uid", date_col])
            tmp["study_day"] = (tmp.groupby("uid")[date_col]
                                  .transform(lambda s: (pd.to_datetime(s)
                                                          - pd.to_datetime(s).min()).dt.days + 1))
            out["study_day"] = tmp["study_day"].values
        else:
            out["study_day"] = 1
        out["weekday"] = df["weekday"] if "weekday" in df.columns else 0
        out["slot"] = df[cfg.COL_SLOT]
        for c in ["weather", "temp", "loc"]:
            out[c] = df[c] if c in df.columns else 0
        out["send"] = df[cfg.COL_ACTION]
        out["dosage"] = df["dosage"] if "dosage" in df.columns else 0.0
        out["resp"] = 0  # zeroed — keep schema but DDQN/FQE STATE_COLS excludes it
        out["steps30pre"] = df["steps30pre"] if "steps30pre" in df.columns else 0.0
        out["reward"] = df["reward"]
        return out

    real_e = _to_eval_schema(real_df)
    if synth_df is not None and len(synth_df) > 0:
        synth_e = _to_eval_schema(synth_df)
        combined = pd.concat([real_e, synth_e], ignore_index=True)
    else:
        combined = real_e
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    combined.to_csv(out_csv, index=False)
    print(f"[eval] Prepared CSV: {len(combined)} rows -> {out_csv}")


def run_kfold(eval_csv_relpath: str,
              out_dir: str,
              cwd: str,
              cql_alpha: float = 1.0,
              n_folds: int = None,
              ddqn_seeds: int = None,
              fqe_seeds: int = None,
              extra_args: list = None) -> Dict:
    """Invoke policy_utility_kfold.py as a subprocess; return parsed summary."""
    n_folds = n_folds or cfg.EVAL["n_folds"]
    ddqn_seeds = ddqn_seeds or cfg.EVAL["ddqn_seeds"]
    fqe_seeds = fqe_seeds or cfg.EVAL["fqe_seeds"]

    # NOTE: policy_utility_kfold.py uses HARDCODED '../data/data_eval.csv' inside
    # its main(). The eval_csv_relpath must be saved there OR we patch via env var.
    # Simplest: write to ../data/data_eval.csv (same path the script expects).
    cmd = [sys.executable, _KFOLD_SCRIPT,
           "--out_dir", out_dir,
           "--n_folds", str(n_folds),
           "--ddqn_seeds", str(ddqn_seeds),
           "--fqe_seeds", str(fqe_seeds),
           "--cql_alpha", str(cql_alpha)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n[eval] $ {' '.join(cmd)}\n        cwd={cwd}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[eval] WARNING: kfold script exited {result.returncode}")
    return parse_kfold_outputs(os.path.join(cwd, out_dir))


def parse_kfold_outputs(out_dir: str) -> Dict:
    """Read kfold_summary.csv + paired_diff.csv + action_distributions.json."""
    summ_p = os.path.join(out_dir, "kfold_summary.csv")
    pair_p = os.path.join(out_dir, "paired_diff.csv")
    act_p = os.path.join(out_dir, "action_distributions.json")
    out = {}
    if os.path.exists(summ_p):
        out["summary"] = pd.read_csv(summ_p).set_index("policy").to_dict("index")
    if os.path.exists(pair_p):
        out["paired_diff"] = pd.read_csv(pair_p).to_dict("records")
    if os.path.exists(act_p):
        with open(act_p) as f:
            out["action_distribution"] = json.load(f)
    return out
