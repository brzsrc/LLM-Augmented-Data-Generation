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
    Used by trajectory_sampler's hurdle-lognormal generator.

    `bin_edges` and `bin_labels` define the PER-USER 6-bin partition of this
    user's steps30pre observations (bin 0 = exact zero, the remaining 5 are
    quantile-cut on positive values). Labels are literal numeric ranges and
    are referenced by Steps10BucketStats.zero_pct_by_s30_bin and
    Steps10AllBucket.by_steps30pre_bin.
    """
    mean: float
    median: float
    zero_pct: float
    per_slot_mean: Dict[int, float] = field(default_factory=dict)
    per_slot_zero_pct: Dict[int, float] = field(default_factory=dict)
    # MLE-fit lognormal σ on non-zero steps30pre; fallback = 1.17 (data_gen median).
    sigma_log: float = 1.17
    # Per-user 6-bin scheme used downstream for hurdle / conditional means.
    bin_edges:  List[float] = field(default_factory=list)
    bin_labels: List[str]   = field(default_factory=list)


@dataclass
class Steps10BucketStats:
    """One availability bucket of steps10 stats.

    Positive-only anchors (`*_positive`) hold means computed on rows where
    steps10>0, used by the LLM prompt's anchor_lookup step (after the hurdle
    gate decides non-zero). zero_pct_by_s30_bin is the per-bin hurdle signal.

    Mixed-zero fields (`mean`, `median`, `zero_pct`, `per_slot_mean`,
    `per_slot_zero_pct`) are preserved because downstream consumers
    (archetype classifier, consistency_gate, summary_str) depend on them.
    """
    mean: float
    median: float
    zero_pct: float
    per_slot_mean: Dict[int, float] = field(default_factory=dict)
    per_slot_zero_pct: Dict[int, float] = field(default_factory=dict)
    # Positive-only per-slot mean (steps10 > 0); both avail buckets.
    per_slot_mean_positive: Dict[int, float] = field(default_factory=dict)
    # {slot: {action: mean_steps10>0}} — only populated for the avail_true bucket.
    per_slot_action_mean_positive: Optional[Dict[int, Dict[int, float]]] = None
    # P(steps10=0 | steps30pre bin) — hurdle signal per this avail bucket.
    # Keys = cfg.STEPS30PRE_BINS["labels"]; cells with n<5 are omitted.
    zero_pct_by_s30_bin: Dict[str, float] = field(default_factory=dict)


@dataclass
class Steps10AllBucket(Steps10BucketStats):
    """Marginal (avail=True ∪ avail=False) bucket — adds context-conditional
    distributions and aggregate scores. Conditional means here are
    POSITIVE-ONLY (computed on steps10>0) so they're consistent with the
    positive anchor used by the LLM's non-zero reasoning path.
    """
    # Marginal positive-only mean (steps10>0 over all rows). Mixed-zero `mean`
    # is still in the base dataclass (used by archetype classifier).
    mean_positive: float = 0.0
    # Context-conditional means — POSITIVE-ONLY (steps10 > 0).
    by_loc_positive:     Dict[str, float] = field(default_factory=dict)
    by_weather_positive: Dict[str, float] = field(default_factory=dict)
    by_temp_positive:    Dict[str, float] = field(default_factory=dict)
    # steps30pre binned (per-user labels) → POSITIVE steps10 mean
    by_steps30pre_bin_positive: Dict[str, float] = field(default_factory=dict)
    # Pearson corr(steps30pre, steps10) on POSITIVE rows only.
    momentum_score_positive: float = 0.0
    # Consecutive-slot zero-state pair frequencies; e.g.
    #   {"zero_after_zero": 0.62, "nonzero_after_zero": 0.38,
    #    "zero_after_nonzero": 0.31, "nonzero_after_nonzero": 0.69}
    momentum_pair_pct: Dict[str, float] = field(default_factory=dict)
    # Mean steps10 on the RECOVERY slot (first non-zero after a zero streak),
    # binned by length of the preceding zero streak. Complements
    # momentum_pair_pct: that gives P(non-zero | prev), this gives the
    # CONDITIONAL MAGNITUDE on the recovery slot.
    # Bins: "1" / "2" / "3+" — matches the HeartSteps 5-slot/day structure
    # where a within-day streak can be at most 4 (slots 1-4 all zero, slot 5
    # non-zero). Streak resets at day boundary. Sparse buckets (n < threshold)
    # are omitted by the extractor.
    steps10_mean_after_zero_streak: Dict[str, float] = field(default_factory=dict)
    # per-context std of steps10 (derived from positive-only by_* dicts)
    context_sensitivity_positive: Dict[str, float] = field(default_factory=dict)
    # PICLe-style "real burst" anchors derived on POSITIVE rows only,
    # e.g. "slot=3 + loc=work (mean=245, n=34)"
    high_activity_contexts_positive: List[str] = field(default_factory=list)


@dataclass
class Steps10Profile:
    """Three avail-buckets. avail_true / avail_false carry only distributional
    stats; the `all` bucket additionally carries POSITIVE-ONLY marginals
    (by_loc_positive / by_weather_positive / momentum_score_positive / ...)
    since those are only computed on the union."""
    avail_true:  Steps10BucketStats     # message-eligible: per_slot_action_mean_positive populated
    avail_false: Steps10BucketStats     # unreachable: action=0, per_slot_action_mean_positive=None
    all:         Steps10AllBucket       # marginal + context-conditional + aggregates


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
                f"momentum={self.activity.steps10.all.momentum_score_positive:.2f} "
                f"n_days={self.anchor.n_days} "
                f"borrowed={self.anchor.borrowed_uids}")
