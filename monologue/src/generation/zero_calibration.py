"""Post-hoc zero-distribution calibration for LLM-generated trajectories.

Two-stage fix applied AFTER trajectory_sampler returns:

  1. clip_low_positives  — fold steps10 ∈ [1, THRESHOLD] into 0. LLM produces
     a roughly uniform tail in this range when uncertain; real data is sparse
     here (≤2% mass), so it's safer to read these as soft-zeros.

  2. calibrate_zeros     — per-cell, knock down LLM positives to 0 until each
     cell's P(steps10=0) matches the real-data target. Cell key is
     (slot, send, avail) by default; refine via ZERO_CAL_KEYS if a finer fit
     is needed and real has the row counts to support it.

Order matters: clip first, then calibrate. Clip already raises the zero rate
toward the target; the calibration step fills the remaining gap and locks the
per-cell marginal. Reversed order would over-zero (calibration hits target,
clip then pushes past it).
"""
from __future__ import annotations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src import config as cfg


def clip_low_positives(synth: pd.DataFrame,
                        threshold: int = cfg.CLIP_LOW_THRESHOLD) -> pd.DataFrame:
    """Fold steps10 ∈ [1, threshold] into 0. Returns a NEW dataframe."""
    out = synth.copy()
    mask = (out["steps10"] > 0) & (out["steps10"] <= threshold)
    n = int(mask.sum())
    out.loc[mask, "steps10"] = 0
    print(f"[zero-cal] clip ≤{threshold}: folded {n} rows "
          f"({n/len(out):.2%}) → 0")
    return out


def calibrate_zeros(synth: pd.DataFrame, real: pd.DataFrame,
                     keys: Sequence[str] = cfg.ZERO_CAL_KEYS,
                     seed: int = 42) -> pd.DataFrame:
    """Knock down LLM-positive rows to 0 so per-cell P(steps10=0) matches real.

    Per cell:
        need = round(p_zero_target * n_synth - n_synth_zero_now)
    Sample `need` rows uniformly from the cell's positive set and set them
    to 0. Cells where current zero count already meets target are left alone
    — this is a one-way operation (we never un-zero).

    Args:
        synth: synthetic trajectories (RAW string-categorical schema is fine;
            this function only touches `steps10`).
        real:  real trajectories used to estimate the per-cell target zero rate.
        keys:  tuple of column names defining a cell. Default = ZERO_CAL_KEYS.
        seed:  RNG seed for the within-cell row sampling.
    """
    keys = list(keys)
    rng = np.random.default_rng(seed)
    p_zero_tgt = (real.groupby(keys, observed=True)["steps10"]
                       .apply(lambda x: float((x == 0).mean()))
                       .rename("p_zero_tgt")
                       .reset_index())
    s = synth.merge(p_zero_tgt, on=keys, how="left")
    steps_out = s["steps10"].to_numpy().copy()

    n_cells, n_knocked = 0, 0
    n_skip_nan, n_skip_no_pos = 0, 0
    for _, g in s.groupby(keys, observed=True, sort=False):
        n_cells += 1
        tgt_val = g["p_zero_tgt"].iloc[0]
        if pd.isna(tgt_val):
            n_skip_nan += 1
            continue
        pos_idx = g.index[g["steps10"] > 0].to_numpy()
        if len(pos_idx) == 0:
            n_skip_no_pos += 1
            continue
        n = len(g)
        n_cur_zero = n - len(pos_idx)
        need = int(round(float(tgt_val) * n - n_cur_zero))
        if need <= 0:
            continue
        need = min(need, len(pos_idx))
        pick = rng.choice(pos_idx, size=need, replace=False)
        steps_out[pick] = 0
        n_knocked += need

    out = synth.copy()
    out["steps10"] = steps_out.astype(int)
    print(f"[zero-cal] per-cell calibration on {'+'.join(keys)}: "
          f"{n_cells} cells, knocked {n_knocked} pos→0, "
          f"skipped {n_skip_nan} no-target / {n_skip_no_pos} no-positives")
    return out


def apply(synth: pd.DataFrame, real: pd.DataFrame,
           threshold: int = cfg.CLIP_LOW_THRESHOLD,
           keys: Sequence[str] = cfg.ZERO_CAL_KEYS,
           seed: int = 42) -> pd.DataFrame:
    """Convenience: clip then calibrate, in the recommended order."""
    synth = clip_low_positives(synth, threshold=threshold)
    synth = calibrate_zeros(synth, real, keys=keys, seed=seed)
    return synth
