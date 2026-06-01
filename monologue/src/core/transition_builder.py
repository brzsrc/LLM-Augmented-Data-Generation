"""Build MDP transitions (s, a, r, s', done) from a normalised dataframe.

The output is what `evaluation/kfold_runner.py` consumes for DDQN / FQE training.
Same logic as policy_utility_kfold.py:build_transitions but driven by config so
swapping datasets does not require code changes.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from src import config as cfg

def build_transitions(df: pd.DataFrame,
                      include_patients: Optional[Set] = None) -> List[Dict]:
    """Build per-patient time-sorted (s, a, r, s', done) records.

    Each transition dict has keys: patient_id, s, a, r, ns, done, source_uid?
    `source_uid` is preserved when present (synthetic data tagging for K-fold
    leakage prevention).
    """
    pid_col = cfg.COL_PATIENT_ID
    action_col = cfg.COL_ACTION
    slot_col = cfg.COL_SLOT
    date_col = cfg.COL_DECISION_DATE
    state_cols = cfg.STATE_FEATURES

    sort_cols = [pid_col]
    if date_col and date_col in df.columns:
        sort_cols.append(date_col)
    sort_cols.append(slot_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if include_patients is not None:
        df = df[df[pid_col].isin(include_patients)]

    # ensure all state cols are numeric
    for c in state_cols:
        if c not in df.columns:
            raise KeyError(f"State column '{c}' not in df. "
                           f"Did add_derived_features() run?")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"State column '{c}' is not numeric (dtype={df[c].dtype}). "
                            f"Did encoders/derived features cover it?")

    has_source = "source_uid" in df.columns
    transitions: List[Dict] = []
    for pid, g in df.groupby(pid_col, sort=False):
        g = g.reset_index(drop=True)
        S = g[state_cols].astype(float).values
        A = g[action_col].astype(int).values
        R = g["reward"].astype(float).values
        n = len(g)
        for t in range(n):
            ns = S[t + 1] if t < n - 1 else S[t]
            done = 0.0 if t < n - 1 else 1.0
            rec = {
                "patient_id": pid,
                "s": S[t], "a": int(A[t]), "r": float(R[t]),
                "ns": ns, "done": done,
            }
            if has_source:
                rec["source_uid"] = g["source_uid"].iloc[t]
            transitions.append(rec)
    return transitions


def to_arrays(trans: List[Dict]):
    s = np.array([t["s"] for t in trans], dtype=np.float32)
    a = np.array([t["a"] for t in trans], dtype=np.int64)
    r = np.array([t["r"] for t in trans], dtype=np.float32)
    ns = np.array([t["ns"] for t in trans], dtype=np.float32)
    d = np.array([t["done"] for t in trans], dtype=np.float32)
    pid = np.array([t["patient_id"] for t in trans])
    return s, a, r, ns, d, pid


def filter_by_patients(trans: List[Dict], pats: Set) -> List[Dict]:
    return [t for t in trans if t["patient_id"] in pats]


def filter_synth_by_source(trans: List[Dict], excluded_source_uids: Set) -> List[Dict]:
    """For K-fold leakage prevention: drop synthetic transitions whose source_uid
    is in the held-out test fold."""
    return [t for t in trans
            if "source_uid" not in t or t["source_uid"] not in excluded_source_uids]


def initial_states_with_ids(trans: List[Dict]):
    """One initial state per patient, aligned with patient ids."""
    by_pid = defaultdict(list)
    for t in trans:
        by_pid[t["patient_id"]].append(t)
    pids = list(by_pid.keys())
    init = np.array([by_pid[p][0]["s"] for p in pids], dtype=np.float32)
    return pids, init
