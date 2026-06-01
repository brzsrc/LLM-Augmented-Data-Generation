"""State scaler + encoding utilities.

Fits a MinMaxScaler on training transitions; reused at FQE evaluation time.
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.core.transition_builder import to_arrays


def fit_scaler(train_trans: List[Dict]) -> MinMaxScaler:
    s, _, _, _, _, _ = to_arrays(train_trans)
    return MinMaxScaler().fit(s)


def transform_states(scaler: MinMaxScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X.astype(np.float32))
