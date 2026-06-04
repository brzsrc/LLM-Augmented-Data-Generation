"""Post-hoc zero-distribution calibration for LLM-generated trajectories.

Two-stage fix applied AFTER trajectory_sampler returns:

  1. clip_low_positives  — fold steps10 ∈ [1, THRESHOLD] into 0. LLM produces
     a roughly uniform tail in this range when uncertain; real data is sparse
     here (≤2% mass), so it's safer to read these as soft-zeros.

  2. calibrate_zeros     — per-cell, knock down LLM positives to 0 until each
     cell's P(steps10=0) matches the real-data target. Cell key is governed
     by `cfg.ZERO_CAL_KEYS`; when "steps30pre_bin" is in the key tuple, a
     global quartile-edge bin (fit on real.steps30pre) is automatically
     attached to both frames before grouping and stripped from the output.
     Sampling strategy within a cell is governed by `cfg.ZERO_CAL_STRATEGY`:
     "low_s30_first" (default) zeros the lowest-steps30pre rows first,
     preserving the momentum correlation that real exhibits; "random"
     picks uniformly and is kept for ablation.

Order matters: clip first, then calibrate. Clip already raises the zero rate
toward the target; the calibration step fills the remaining gap and locks the
per-cell marginal. Reversed order would over-zero (calibration hits target,
clip then pushes past it).
"""
from __future__ import annotations
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from src import config as cfg


S30_BIN_COL = "steps30pre_bin"


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


def _attach_s30_bin(real: pd.DataFrame, synth: pd.DataFrame,
                     n_q: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """Compute global quartile edges from real.steps30pre and attach
    `steps30pre_bin` (string-categorical) to BOTH frames.

    Edges are padded with ±inf so synth values outside real's range still get
    a valid bin label (no NaN). qcut may collapse to fewer than `n_q` bins
    when steps30pre has many tied zeros — this is fine.
    """
    _, edges = pd.qcut(real["steps30pre"], q=n_q, duplicates="drop",
                        retbins=True)
    edges = list(edges)
    edges[0]  = -np.inf
    edges[-1] = np.inf
    real_out  = real.copy()
    synth_out = synth.copy()
    real_out[S30_BIN_COL]  = pd.cut(real_out["steps30pre"], bins=edges,
                                     include_lowest=True).astype(str)
    synth_out[S30_BIN_COL] = pd.cut(synth_out["steps30pre"], bins=edges,
                                     include_lowest=True).astype(str)
    print(f"[zero-cal] global steps30pre edges from real (n_q={n_q} → "
          f"{len(edges)-1} bins): {edges}")
    return real_out, synth_out, edges


def calibrate_zeros(synth: pd.DataFrame, real: pd.DataFrame,
                     keys: Sequence[str] = cfg.ZERO_CAL_KEYS,
                     n_q_s30: int = cfg.ZERO_CAL_S30_N_QUANTILES,
                     strategy: str = cfg.ZERO_CAL_STRATEGY,
                     seed: int = 42) -> pd.DataFrame:
    """Knock down LLM-positive rows to 0 so per-cell P(steps10=0) matches real.

    Per cell:
        need = round(p_zero_target * n_synth - n_synth_zero_now)
    Pick `need` rows from the cell's positive set (strategy below) and set
    them to 0. Cells where current zero count already meets target are left
    alone — this is a one-way operation (we never un-zero).

    Args:
        synth: synthetic trajectories (RAW string-categorical schema is fine).
        real:  real trajectories used to estimate per-cell target zero rate.
        keys:  tuple of column names defining a cell. If "steps30pre_bin" is
            present and not already a column, it is computed from real's
            global quartile edges and attached temporarily.
        n_q_s30: qcut quantile count for steps30pre binning (only used when
            keys includes "steps30pre_bin").
        strategy: which positive rows in a cell to zero.
            - "low_s30_first": sort the cell's positives by steps30pre asc
              (with tiny jitter for ties) and zero the lowest `need`. Keeps
              high-momentum rows positive so steps30pre→steps10 correlation
              survives.
            - "random": uniform draw over positives. Breaks the momentum
              correlation; kept for ablation comparisons.
        seed:  RNG seed for within-cell sampling / jitter.

    Returns:
        A NEW dataframe with the same schema as `synth` (any
        temporarily-attached `steps30pre_bin` column is stripped on the way
        out so downstream consumers see the original column set).
    """
    keys = list(keys)
    if strategy not in ("low_s30_first", "random"):
        raise ValueError(f"unknown strategy {strategy!r}; "
                         "expected 'low_s30_first' or 'random'")
    need_bin = (S30_BIN_COL in keys) and (S30_BIN_COL not in synth.columns)
    if need_bin:
        real, synth, _ = _attach_s30_bin(real, synth, n_q=n_q_s30)

    rng = np.random.default_rng(seed)
    p_zero_tgt = (real.groupby(keys, observed=True)["steps10"]
                       .apply(lambda x: float((x == 0).mean()))
                       .rename("p_zero_tgt")
                       .reset_index())
    s = synth.merge(p_zero_tgt, on=keys, how="left")
    steps_out = s["steps10"].to_numpy().copy()
    s30_arr   = s["steps30pre"].to_numpy()

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
        if strategy == "low_s30_first":
            # Sort positives by steps30pre ascending; tiny jitter breaks
            # ties randomly so e.g. the many steps30pre=0 rows don't always
            # get zeroed in index order.
            cell_s30 = s30_arr[pos_idx].astype(float)
            jitter   = rng.random(len(pos_idx)) * 1e-6
            order    = np.argsort(cell_s30 + jitter, kind="stable")
            pick     = pos_idx[order[:need]]
        else:                                     # "random"
            pick = rng.choice(pos_idx, size=need, replace=False)
        steps_out[pick] = 0
        n_knocked += need

    out = synth.copy()
    out["steps10"] = steps_out.astype(int)
    if need_bin:
        out = out.drop(columns=[S30_BIN_COL])
    print(f"[zero-cal] per-cell calibration on {'+'.join(keys)} "
          f"(strategy={strategy}): {n_cells} cells, knocked "
          f"{n_knocked} pos→0, skipped {n_skip_nan} no-target / "
          f"{n_skip_no_pos} no-positives")
    return out


def apply(synth: pd.DataFrame, real: pd.DataFrame,
           threshold: int = cfg.CLIP_LOW_THRESHOLD,
           keys: Sequence[str] = cfg.ZERO_CAL_KEYS,
           n_q_s30: int = cfg.ZERO_CAL_S30_N_QUANTILES,
           strategy: str = cfg.ZERO_CAL_STRATEGY,
           seed: int = 42) -> pd.DataFrame:
    """Convenience: clip then calibrate, in the recommended order."""
    synth = clip_low_positives(synth, threshold=threshold)
    synth = calibrate_zeros(synth, real, keys=keys, n_q_s30=n_q_s30,
                             strategy=strategy, seed=seed)
    return synth
