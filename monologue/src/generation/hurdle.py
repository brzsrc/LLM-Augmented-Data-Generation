"""Python-side Bernoulli hurdle for steps10 generation.

Computes P(steps10 = 0) for a single transition step. Used by
trajectory_sampler to short-circuit the LLM call when the user is likely
to be zero — the LLM only predicts the positive outcome, Python decides
whether the slot is zero at all.

Three signals are combined (missing-aware weighted average):
    bin_rate    P(steps10=0 | steps30pre bin, current avail)   instantaneous
    slot_rate   P(steps10=0 | slot, current avail)             instantaneous
    trans_rate  P(steps10=0 | prev steps10 = 0 vs > 0)         episodic

Why those three:
  - bin_rate and slot_rate condition on the CURRENT context (highly correlated
    with each other), so default weights downweight each to 0.5.
  - trans_rate conditions on the PREVIOUS slot's outcome (independent signal),
    so default weight is 1.0.

Sparse-cell handling is delegated to the persona extractor: if a cell wasn't
fit (n < threshold), the extractor should emit None / omit the key, NOT write
0.0. hurdle.py treats any non-None numeric value as valid.

Function signature matches the FLAT persona dict produced by
pipeline._persona_to_flat_dict, NOT the nested PersonaProfile dataclass.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from src import config as cfg


# bin / slot / trans. See module docstring for rationale.
DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.5, 0.5, 1.0)


def compute_hurdle_p(
    persona: Dict,
    current_state: Dict,
    episodic_history: Optional[List[Dict]] = None,
    *,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    sparse_threshold: int = 10,   # reserved for future extractor coupling
) -> float:
    """Combined P(steps10 = 0) for the current decision point.

    Args:
        persona: flat persona dict from pipeline._persona_to_flat_dict
        current_state: dict with keys avail, slot, steps30pre, ...
        episodic_history: trajectory generated so far for this synth user
            (list of dicts with at least a `steps10` key). The LAST entry is
            the immediately preceding slot.
        weights: (w_bin, w_slot, w_trans). Missing signals are dropped from
            both numerator and denominator (their weight is excluded too).
        sparse_threshold: placeholder. Currently unused — assumes the extractor
            has already nulled sparse cells. Wired up so callers can pass the
            same threshold both places once extractor honors it.

    Returns:
        p_zero in [0, 1]. 0.0 if all three signals are missing
        (fail-open default — hurdle never triggers).
    """
    history = episodic_history or []
    rates = (
        _bin_rate(persona, current_state),
        _slot_rate(persona, current_state),
        _transition_rate(persona, history),
    )
    valid = [(r, w) for r, w in zip(rates, weights) if r is not None]
    if not valid:
        return 0.0
    p = sum(r * w for r, w in valid) / sum(w for _, w in valid)
    return max(0.0, min(1.0, p))


def _bin_rate(persona: Dict, state: Dict) -> Optional[float]:
    """P(steps10=0 | steps30pre bin, current avail).

    Returns None when:
      - persona has no bin scheme (edges/labels missing)
      - persona has no hurdle table for the current avail
      - state has no steps30pre
      - the resolved bin label isn't a key in the table
    """
    avail = state.get("avail")
    table_key = ("steps10_avail_true_zero_pct_by_s30_bin"
                 if avail else "steps10_avail_false_zero_pct_by_s30_bin")
    table  = persona.get(table_key) or {}
    edges  = persona.get("steps30pre_bin_edges")  or []
    labels = persona.get("steps30pre_bin_labels") or []
    if not table or not edges or not labels:
        return None
    s30 = state.get("steps30pre")
    if s30 is None:
        return None
    label = cfg.s30_bin_label(s30, edges, labels)
    v = table.get(label)
    return float(v) if v is not None else None


def _slot_rate(persona: Dict, state: Dict) -> Optional[float]:
    """P(steps10=0 | slot, current avail).

    Tolerates both int and string slot keys — dataclass holds int but
    JSON-roundtrip yields string. Returns None when the table or slot
    entry is missing.
    """
    avail = state.get("avail")
    table_key = ("steps10_avail_true_per_slot_zero_pct"
                 if avail else "steps10_avail_false_per_slot_zero_pct")
    table = persona.get(table_key) or {}
    slot = state.get("slot")
    if slot is None or not table:
        return None
    v = table.get(slot)
    if v is None:
        v = table.get(str(slot))
    return float(v) if v is not None else None


def _transition_rate(persona: Dict, history: List[Dict]) -> Optional[float]:
    """P(steps10=0 | prev steps10) from momentum_pair_pct.

    Returns None when:
      - history is empty (first slot of trajectory)
      - prev entry has no `steps10` key
      - persona has no momentum_pair_pct stats
      - the relevant transition key isn't fitted
    """
    if not history:
        return None
    pair_pct = persona.get("steps10_momentum_pair_pct") or {}
    if not pair_pct:
        return None
    prev = history[-1].get("steps10")
    if prev is None:
        return None
    key = "zero_after_zero" if prev == 0 else "zero_after_nonzero"
    v = pair_pct.get(key)
    return float(v) if v is not None else None
