"""4-part persona profile dataclasses (Attitude removed).

PersonaProfile
├── Anchor              identity + variant tag + leakage tags
├── ContextProfile      A1: weather/temp marginal + within-day slot→slot transitions
├── LifestyleProfile    A2: loc + schedule + state-conditional avail predictors
├── ActivityProfile     B3: steps10/steps30pre + context-conditional distributions
└── CompliancePhases    B4: honeymoon/plateau/fatigue from STL trend on daily steps10

Attitude (B-old) removed: empirical analysis showed resp ⟂ steps10, so any
synth output never used it for DDQN downstream.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ============================================================================
@dataclass
class Anchor:
    source_uid: int
    archetype: str
    variant_type: str
    synth_uid: int
    slot_1_hour: int
    n_days: int
    borrowed_uids: List[int] = field(default_factory=list)


# ============================================================================
# A1: context
# ============================================================================
@dataclass
class ContextProfile:
    weather_dist: Dict[str, float]
    temp_dist: Dict[str, float]
    weather_transition: Dict[str, Dict[str, float]] = field(default_factory=dict)
    temp_transition: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ============================================================================
# A2: lifestyle
# ============================================================================
@dataclass
class LifestyleProfile:
    weekday_loc_dist: Dict[str, float]
    weekend_loc_dist: Dict[str, float]
    weekday_slot_loc: Dict[int, Dict[str, float]] = field(default_factory=dict)
    weekend_slot_loc: Dict[int, Dict[str, float]] = field(default_factory=dict)
    weekday_loc_transition: Dict[str, Dict[str, float]] = field(default_factory=dict)
    weekend_loc_transition: Dict[str, Dict[str, float]] = field(default_factory=dict)
    avail_rate: float = 0.9
    avail_by_slot: Dict[int, float] = field(default_factory=dict)
    avail_by_loc: Dict[str, float] = field(default_factory=dict)
    unavail_triggers: List[str] = field(default_factory=list)
    unusual_loc_events: List[str] = field(default_factory=list)


# ============================================================================
# B3: activity — 2 sub-profiles
#   activity.steps30pre  : prior-30-min activity (input feature)
#   activity.steps10     : next-10-min activity (target), 3 avail buckets +
#                          context-conditional marginals
# ============================================================================
@dataclass
class Steps30PreProfile:
    """Statistics of the 30-min step count preceding each decision point.
    Used by trajectory_sampler's hurdle-lognormal generator."""
    mean: float
    median: float
    zero_pct: float
    per_slot_mean: Dict[int, float] = field(default_factory=dict)
    per_slot_zero_pct: Dict[int, float] = field(default_factory=dict)
    # MLE-fit lognormal σ on non-zero steps30pre; fallback = 1.17 (data_gen median).
    sigma_log: float = 1.17


@dataclass
class Steps10BucketStats:
    """One availability bucket of steps10 stats. The avail=True bucket
    additionally carries per_slot_action_mean (the MRT-causal signal);
    avail=False keeps it None — when avail=False, send is structurally 0
    so per-action breakdown is degenerate."""
    mean: float
    median: float
    zero_pct: float
    per_slot_mean: Dict[int, float] = field(default_factory=dict)
    per_slot_zero_pct: Dict[int, float] = field(default_factory=dict)
    # {slot: {action: mean_steps10}} — only populated for the avail_true bucket.
    per_slot_action_mean: Optional[Dict[int, Dict[int, float]]] = None
    # ── Fields below support prompt zero_check + positive-only anchoring ──
    # P(steps10=0) bucketed by steps30pre quartile bin (per-persona qcut keys).
    zero_pct_by_s30_bin: Dict[str, float] = field(default_factory=dict)
    # P(steps10=0) per location for this avail bucket. avail=True vs avail=False
    # zero rates differ by 10-25pp per loc, so they MUST be kept separate.
    # Cells with <5 rows are dropped (per-user noise).
    per_loc_zero_pct: Dict[str, float] = field(default_factory=dict)
    # Same shape as per_slot_action_mean but filtered to steps10>0 rows.
    # Anchor used by LLM when zero_check decides "positive".
    per_slot_action_mean_positive: Optional[Dict[int, Dict[int, float]]] = None
    # {slot: {"25"|"50"|"75"|"95"|"99": int}} on steps10>0 only.
    # Surfaces the right-skewed tail so LLM doesn't snap to the mean.
    per_slot_positive_quantiles: Dict[int, Dict[str, int]] = field(default_factory=dict)
    # Flat per-user quantiles on steps10>0 (no slot split). Used by avail=False
    # branch where per-(user,slot) cells are too thin for stable quantiles.
    positive_quantiles_user: Dict[str, int] = field(default_factory=dict)


@dataclass
class Steps10AllBucket(Steps10BucketStats):
    """Marginal (avail=True ∪ avail=False) bucket — adds context-conditional
    distributions and aggregate scores computed on the union. These fields
    only make sense at the marginal level: avail-specific cells are too sparse
    for reliable conditional means."""
    by_loc:     Dict[str, float] = field(default_factory=dict)
    by_weather: Dict[str, float] = field(default_factory=dict)
    by_temp:    Dict[str, float] = field(default_factory=dict)
    # steps30pre binned (Q1/Q2/Q3/Q4) → steps10 mean
    by_steps30pre_bin: Dict[str, float] = field(default_factory=dict)
    # Pearson corr(steps30pre, steps10) — user-trait scalar
    momentum_score: float = 0.0
    # per-context std of steps10 — which dimension drives variation
    context_sensitivity: Dict[str, float] = field(default_factory=dict)
    # PICLe-style "real burst" anchors: e.g. "slot=3 + loc=work (mean=245, n=34)"
    high_activity_contexts: List[str] = field(default_factory=list)


@dataclass
class Steps10Profile:
    """Three avail-buckets. avail_true / avail_false carry only distributional
    stats; the `all` bucket additionally carries marginals (by_loc / by_weather
    / momentum_score / ...) since those are only computed on the union."""
    avail_true:  Steps10BucketStats     # message-eligible: per_slot_action_mean populated
    avail_false: Steps10BucketStats     # unreachable: action=0, per_slot_action_mean=None
    all:         Steps10AllBucket       # marginal + context-conditional + aggregates
    # Streak transition for zero_check: {"zero_after_zero": float,
    # "zero_after_nonzero": float}. Computed on the union (within-patient
    # consecutive transitions). Empty if not enough rows.
    momentum_pair_pct: Dict[str, float] = field(default_factory=dict)


@dataclass
class ActivityProfile:
    steps30pre: Steps30PreProfile
    steps10:    Steps10Profile


# ============================================================================
# B4: compliance phases (from STL trend on daily steps10)
# ============================================================================
@dataclass
class CompliancePhase:
    name: str
    day_range: Tuple[int, int]
    activity_mult: float


@dataclass
class CompliancePhases:
    phases: List[CompliancePhase]

    def get_phase(self, day: int) -> CompliancePhase:
        for p in self.phases:
            if p.day_range[0] <= day <= p.day_range[1]:
                return p
        return self.phases[-1]


# ============================================================================
@dataclass
class PersonaProfile:
    anchor: Anchor
    context: ContextProfile
    lifestyle: LifestyleProfile
    activity: ActivityProfile
    compliance: CompliancePhases

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_str(self) -> str:
        return (f"uid={self.anchor.synth_uid:<6d} arch={self.anchor.archetype:18s} "
                f"src=R{self.anchor.source_uid} slot_1_h={self.anchor.slot_1_hour} "
                f"steps10={self.activity.steps10.all.mean:.0f} "
                f"steps30pre={self.activity.steps30pre.mean:.0f} "
                f"momentum={self.activity.steps10.momentum_score:.2f} "
                f"n_days={self.anchor.n_days} "
                f"borrowed={self.anchor.borrowed_uids}")
