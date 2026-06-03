"""Unit tests for monologue/src/generation/hurdle.py.

Run from the monologue/ directory:
    python -m pytest tests/test_hurdle.py -v
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

# Allow `from src.generation.hurdle import ...` regardless of cwd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.generation.hurdle import (
    compute_hurdle_p,
    _bin_rate,
    _slot_rate,
    _transition_rate,
    DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _full_persona() -> dict:
    """Realistic persona dict matching the flat schema produced by
    pipeline._persona_to_flat_dict. Numbers loosely follow run4-stub uid 1000."""
    return {
        # Bin scheme
        "steps30pre_bin_edges":
            [0.0, 1.0, 81.0, 216.0, 396.0, 858.0, 1e12],
        "steps30pre_bin_labels":
            ["0", "1-80", "81-215", "216-395", "396-857", "858+"],
        # Hurdle by bin (avail=True only — sparse avail=False is realistic)
        "steps10_avail_true_zero_pct_by_s30_bin": {
            "0":       0.70,
            "1-80":    0.64,
            "81-215":  0.41,
            "216-395": 0.35,
            "396-857": 0.14,
            "858+":    0.09,
        },
        "steps10_avail_false_zero_pct_by_s30_bin": {},   # not fit
        # Per-slot zero rate
        "steps10_avail_true_per_slot_zero_pct":
            {1: 0.69, 2: 0.34, 3: 0.44, 4: 0.23, 5: 0.45},
        "steps10_avail_false_per_slot_zero_pct":
            {1: 0.80, 2: 0.50, 3: 0.50, 4: 0.33, 5: 0.60},
        # Transition pair pct
        "steps10_momentum_pair_pct": {
            "zero_after_zero":       0.44,
            "nonzero_after_zero":    0.56,
            "zero_after_nonzero":    0.30,
            "nonzero_after_nonzero": 0.70,
        },
    }


def _state(**overrides) -> dict:
    base = {
        "day": 5, "slot": 3, "hour": 12.5, "weekday": 2, "avail": True,
        "weather": "clear", "temp": "warm", "loc": "work", "steps30pre": 85,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------

class TestBinRate:
    def test_lookup_avail_true(self):
        p = _full_persona()
        # steps30pre=85 → bin "81-215" → 0.41
        assert _bin_rate(p, _state(steps30pre=85)) == pytest.approx(0.41)

    def test_lookup_zero_value(self):
        # steps30pre=0 → bin "0" → 0.70
        p = _full_persona()
        assert _bin_rate(p, _state(steps30pre=0)) == pytest.approx(0.70)

    def test_lookup_large_value_clamps_to_last_bin(self):
        p = _full_persona()
        assert _bin_rate(p, _state(steps30pre=9999)) == pytest.approx(0.09)

    def test_avail_false_table_empty_returns_none(self):
        p = _full_persona()
        assert _bin_rate(p, _state(avail=False, steps30pre=85)) is None

    def test_no_bin_scheme_returns_none(self):
        p = _full_persona()
        p["steps30pre_bin_edges"] = []
        assert _bin_rate(p, _state()) is None

    def test_missing_steps30pre_returns_none(self):
        p = _full_persona()
        s = _state()
        s.pop("steps30pre")
        assert _bin_rate(p, s) is None

    def test_missing_table_returns_none(self):
        p = _full_persona()
        p["steps10_avail_true_zero_pct_by_s30_bin"] = {}
        assert _bin_rate(p, _state()) is None

    def test_bin_label_not_in_table_returns_none(self):
        p = _full_persona()
        # Drop the entry for the bin our state maps into
        del p["steps10_avail_true_zero_pct_by_s30_bin"]["81-215"]
        assert _bin_rate(p, _state(steps30pre=85)) is None


class TestSlotRate:
    def test_lookup_int_key_avail_true(self):
        p = _full_persona()
        assert _slot_rate(p, _state(slot=3)) == pytest.approx(0.44)

    def test_lookup_int_key_avail_false(self):
        p = _full_persona()
        assert _slot_rate(p, _state(slot=3, avail=False)) == pytest.approx(0.50)

    def test_tolerates_string_keys(self):
        # Simulate JSON-roundtripped table where keys came back as strings.
        p = _full_persona()
        p["steps10_avail_true_per_slot_zero_pct"] = {
            "1": 0.69, "2": 0.34, "3": 0.44, "4": 0.23, "5": 0.45
        }
        assert _slot_rate(p, _state(slot=3)) == pytest.approx(0.44)

    def test_missing_slot_returns_none(self):
        p = _full_persona()
        s = _state()
        s.pop("slot")
        assert _slot_rate(p, s) is None

    def test_slot_not_in_table_returns_none(self):
        p = _full_persona()
        assert _slot_rate(p, _state(slot=99)) is None

    def test_missing_table_returns_none(self):
        p = _full_persona()
        p["steps10_avail_true_per_slot_zero_pct"] = {}
        assert _slot_rate(p, _state()) is None


class TestTransitionRate:
    def test_zero_after_zero(self):
        p = _full_persona()
        history = [{"slot": 1, "steps10": 0}, {"slot": 2, "steps10": 0}]
        assert _transition_rate(p, history) == pytest.approx(0.44)

    def test_zero_after_nonzero(self):
        p = _full_persona()
        history = [{"slot": 1, "steps10": 120}, {"slot": 2, "steps10": 240}]
        assert _transition_rate(p, history) == pytest.approx(0.30)

    def test_uses_last_entry_only(self):
        # Lots of zeros earlier but last is nonzero → use zero_after_nonzero
        p = _full_persona()
        history = [
            {"slot": 1, "steps10": 0},
            {"slot": 2, "steps10": 0},
            {"slot": 3, "steps10": 0},
            {"slot": 4, "steps10": 250},
        ]
        assert _transition_rate(p, history) == pytest.approx(0.30)

    def test_empty_history_returns_none(self):
        p = _full_persona()
        assert _transition_rate(p, []) is None

    def test_history_none_treated_as_empty(self):
        # _transition_rate is called from compute_hurdle_p which defaults
        # episodic_history=None → []; check the helper itself with empty list.
        p = _full_persona()
        assert _transition_rate(p, []) is None

    def test_prev_steps10_none_returns_none(self):
        p = _full_persona()
        history = [{"slot": 2, "steps10": None}]
        assert _transition_rate(p, history) is None

    def test_prev_entry_no_steps10_key_returns_none(self):
        p = _full_persona()
        history = [{"slot": 2}]                # malformed entry
        assert _transition_rate(p, history) is None

    def test_missing_pair_pct_returns_none(self):
        p = _full_persona()
        p["steps10_momentum_pair_pct"] = {}
        history = [{"slot": 2, "steps10": 0}]
        assert _transition_rate(p, history) is None

    def test_pair_pct_missing_specific_key_returns_none(self):
        p = _full_persona()
        del p["steps10_momentum_pair_pct"]["zero_after_zero"]
        history = [{"slot": 2, "steps10": 0}]
        assert _transition_rate(p, history) is None


# ---------------------------------------------------------------------------
# compute_hurdle_p — integration of the three signals
# ---------------------------------------------------------------------------

class TestComputeHurdleP:
    def test_all_three_signals_default_weights(self):
        """bin=0.41, slot=0.44, trans(prev=240)=0.30 ; weights (0.5,0.5,1.0)
        weighted-avg = (0.5*0.41 + 0.5*0.44 + 1.0*0.30) / 2.0 = 0.3525"""
        p = _full_persona()
        s = _state(steps30pre=85, slot=3, avail=True)
        h = [{"slot": 2, "steps10": 240}]
        p_zero = compute_hurdle_p(p, s, h)
        expected = (0.5*0.41 + 0.5*0.44 + 1.0*0.30) / (0.5+0.5+1.0)
        assert p_zero == pytest.approx(expected)

    def test_uniform_weights_equals_plain_mean(self):
        p = _full_persona()
        s = _state(steps30pre=85, slot=3, avail=True)
        h = [{"slot": 2, "steps10": 0}]
        p_zero = compute_hurdle_p(p, s, h, weights=(1.0, 1.0, 1.0))
        # bin=0.41, slot=0.44, trans(prev=0)=0.44 → mean=0.43
        assert p_zero == pytest.approx((0.41 + 0.44 + 0.44) / 3.0)

    def test_first_slot_drops_transition(self):
        """Empty history → trans=None → only bin + slot used."""
        p = _full_persona()
        s = _state(steps30pre=85, slot=3, avail=True)
        p_zero = compute_hurdle_p(p, s, episodic_history=[])
        # weights (0.5, 0.5) → plain mean of bin, slot
        assert p_zero == pytest.approx((0.41 + 0.44) / 2.0)

    def test_history_none_arg_treated_as_empty(self):
        p = _full_persona()
        s = _state(steps30pre=85, slot=3, avail=True)
        p_zero = compute_hurdle_p(p, s, None)
        assert p_zero == pytest.approx((0.41 + 0.44) / 2.0)

    def test_avail_false_missing_bin_table_uses_slot_and_trans(self):
        """avail=False has no bin table in this persona → bin=None.
        slot(avail=False, 3)=0.50, trans(prev=0)=0.44 → weighted (0.5, 1.0)"""
        p = _full_persona()
        s = _state(steps30pre=85, slot=3, avail=False)
        h = [{"slot": 2, "steps10": 0}]
        p_zero = compute_hurdle_p(p, s, h)
        expected = (0.5*0.50 + 1.0*0.44) / (0.5+1.0)
        assert p_zero == pytest.approx(expected)

    def test_all_missing_returns_zero_fail_open(self):
        """No persona stats and no history → return 0.0 (hurdle never triggers)."""
        p = {
            "steps30pre_bin_edges": [],
            "steps30pre_bin_labels": [],
            "steps10_avail_true_zero_pct_by_s30_bin": {},
            "steps10_avail_false_zero_pct_by_s30_bin": {},
            "steps10_avail_true_per_slot_zero_pct": {},
            "steps10_avail_false_per_slot_zero_pct": {},
            "steps10_momentum_pair_pct": {},
        }
        assert compute_hurdle_p(p, _state(), []) == 0.0

    def test_only_transition_signal(self):
        """Drop bin + slot data → transition is the only signal."""
        p = _full_persona()
        p["steps10_avail_true_zero_pct_by_s30_bin"] = {}
        p["steps10_avail_true_per_slot_zero_pct"]    = {}
        s = _state(slot=3, avail=True)
        h = [{"slot": 2, "steps10": 0}]                # zero_after_zero=0.44
        assert compute_hurdle_p(p, s, h) == pytest.approx(0.44)

    def test_result_in_unit_interval(self):
        """Even with degenerate inputs, output is clamped to [0, 1]."""
        p = _full_persona()
        # Force everything to extreme high value
        p["steps10_avail_true_zero_pct_by_s30_bin"]["81-215"] = 1.5  # bad input
        p_zero = compute_hurdle_p(p, _state(), [{"slot": 1, "steps10": 0}])
        assert 0.0 <= p_zero <= 1.0

    def test_default_weights_constant_is_what_docs_say(self):
        """Guard against silently changing the default weighting."""
        assert DEFAULT_WEIGHTS == (0.5, 0.5, 1.0)

    def test_weights_kwarg_only(self):
        """weights must be passed as a kwarg, not positional (the * forces it)."""
        p = _full_persona()
        with pytest.raises(TypeError):
            # 4 positional args — should fail because weights is kwarg-only
            compute_hurdle_p(p, _state(), [], (1.0, 1.0, 1.0))   # noqa


# ---------------------------------------------------------------------------
# Numerical smoke: hurdle should yield realistic rates on the run4-stub persona
# ---------------------------------------------------------------------------

class TestNumericalSmoke:
    def test_high_activity_low_s30pre_avail_true(self):
        """User just sitting still (s30pre=0) at slot 1 (low-activity slot)
        → should be HIGH p_zero."""
        p = _full_persona()
        s = _state(slot=1, steps30pre=0, avail=True)
        # bin "0" → 0.70 ; slot 1 → 0.69 ; no history
        p_zero = compute_hurdle_p(p, s, [])
        assert p_zero > 0.6

    def test_high_activity_burst_avail_true(self):
        """User just bursting (s30pre=900) at peak slot
        → should be LOW p_zero."""
        p = _full_persona()
        s = _state(slot=5, steps30pre=900, avail=True)   # bin "858+" → 0.09; slot 5 → 0.45
        p_zero = compute_hurdle_p(p, s, [{"slot": 4, "steps10": 600}])
        # bin 0.09 (w=0.5), slot 0.45 (w=0.5), trans=0.30 (w=1.0)
        # → (0.045 + 0.225 + 0.30) / 2.0 = 0.285
        assert 0.20 < p_zero < 0.35
