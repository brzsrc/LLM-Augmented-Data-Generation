"""Load data_gen.csv + add derived features.

The CSV is already pre-binned (string categoricals), pre-filtered to
avail=True, and has integer `hr`/`study_day`/`weekday`. So we just:
  1. encode the categoricals via cfg.ENCODERS
  2. compute hour_sin/cos from `hr`
  3. compute leakage-free dosage from past actions
  4. compute reward = log(steps10 + 0.5)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src import config as cfg


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(cfg.CSV_PATH)
    print(f"[loader] Loaded {cfg.CSV_PATH}: "
          f"{len(df)} rows, {df[cfg.COL_PATIENT_ID].nunique()} patients")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Categorical encoders (only if column is still string)
    for col, mapping in cfg.ENCODERS.items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            unknown = sorted(set(df[col].astype(str).unique()) - set(mapping.keys()))
            if unknown:
                print(f"  WARNING [{col}] unknown values: {unknown}")
            df[col] = df[col].astype(str).map(mapping).fillna(
                mapping.get("other", -1)).astype(int)

    # 2. hour_sin / hour_cos from hr column
    if cfg.COL_HOUR in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df[cfg.COL_HOUR] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df[cfg.COL_HOUR] / 24.0)

    # 3. Dosage (gap-aware via study_day, uses PRIOR action only — no leakage)
    sort_cols = [cfg.COL_PATIENT_ID, cfg.COL_DAY, cfg.COL_SLOT]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    dosages = []
    for _, g in df.groupby(cfg.COL_PATIENT_ID, sort=False):
        d, prev_a, prev_day = 0.0, 0, None
        for a, day in zip(g[cfg.COL_ACTION].values, g[cfg.COL_DAY].values):
            if prev_day is not None and (day - prev_day) > 1:
                d, prev_a = 0.0, 0          # reset on >1 day gap
            d = 0.95 * d + (1 if prev_a > 0 else 0)
            dosages.append(d)
            prev_a = a
            prev_day = day
    df["dosage"] = dosages

    # 4. Reward = log(steps10 + 0.5)
    if cfg.COL_REWARD_SOURCE in df.columns:
        df["reward"] = np.log(df[cfg.COL_REWARD_SOURCE].astype(float)
                              + cfg.REWARD_LOG_OFFSET)

    return df


def load() -> pd.DataFrame:
    df = load_raw()
    df = add_derived_features(df)
    print(f"[loader] Added: hour_sin/cos, dosage, reward")
    return df
