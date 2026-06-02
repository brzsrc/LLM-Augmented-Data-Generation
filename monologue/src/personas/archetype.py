"""Classify real personas into archetypes + build 3 variant profiles per source.

Variant strategy (attribute-level merge, per DeepPersona):
  twin   : ±10% perturbation on activity, identical schedule          → variance reduction
  sibling: ±30% activity, ±1h schedule, attribute-level peer borrow    → archetype generalisation
  edge   : extreme activity scaling + push schedule to archetype edge  → coverage filling

`borrowed_uids` is populated for sibling variants → enables K-fold leakage filter.
"""
from __future__ import annotations
import copy
from typing import Dict, List
import numpy as np

from src import config as cfg
from src.personas.schemas import (
    Anchor, ActivityProfile, CompliancePhases, CompliancePhase,
    LifestyleProfile, PersonaProfile,
)


# ============================================================================
# Step 1: classify into archetype (tier-based, first-match wins)
# ============================================================================
def classify(p: PersonaProfile) -> str:
    for name, spec in cfg.ARCHETYPES.items():
        try:
            if spec["test"](p):
                return name
        except Exception:
            continue
    return cfg.DEFAULT_ARCHETYPE


# ============================================================================
# Step 2: variant construction
# ============================================================================
def _refresh_compliance() -> CompliancePhases:
    """Build a fresh CompliancePhases from the single default in config."""
    spec_list = cfg.COMPLIANCE_PHASES["default"]
    return CompliancePhases(phases=[CompliancePhase(**spec) for spec in spec_list])


def _scale_per_slot(d: Dict[int, float], factor: float) -> Dict[int, float]:
    """Multiply every value in a {slot: float} dict by `factor`."""
    return {s: v * factor for s, v in (d or {}).items()}


def _scale_activity_means(activity, factor: float) -> None:
    """Apply a single mean-scaling factor consistently across every bucket.

    Lognormal shape (σ_log) is orthogonal to mean → not touched. zero rates
    are also untouched (they're sparsity properties, not magnitude)."""
    s10 = activity.steps10
    for bucket in (s10.avail_true, s10.avail_false, s10.all):
        bucket.mean = (bucket.mean or 0) * factor
        bucket.per_slot_mean = _scale_per_slot(bucket.per_slot_mean, factor)
    activity.steps30pre.mean = (activity.steps30pre.mean or 0) * factor
    activity.steps30pre.per_slot_mean = _scale_per_slot(
        activity.steps30pre.per_slot_mean, factor)


def _build_twin(source: PersonaProfile, rng: np.random.Generator) -> PersonaProfile:
    """Near-clone: ±10% activity, identical schedule.
    Single factor applied to all steps10 buckets + steps30pre so the
    downstream hurdle-lognormal sampler + prompt see consistent perturbation.
    σ_log shape stays as-is (orthogonal to mean)."""
    pert = cfg.VARIANT_PERTURBATIONS["twin"]
    new = copy.deepcopy(source)
    new.anchor.variant_type = "twin"
    factor = rng.uniform(*pert["mean_steps_scale"])
    _scale_activity_means(new.activity, factor)
    return new


def _build_sibling(source: PersonaProfile, peer: PersonaProfile,
                    rng: np.random.Generator) -> PersonaProfile:
    """Attribute-level merge: source spine, peer attributes at given fractions.

    With the new nested schema, activity borrow becomes a single bundle:
    whole steps10 (3 buckets + marginals) and/or whole steps30pre profile.
    Scale factor is still applied on top so the variant ≠ the peer.
    """
    pert = cfg.VARIANT_PERTURBATIONS["sibling"]
    borrow = pert["borrow_fractions"]
    new = copy.deepcopy(source)
    new.anchor.variant_type = "sibling"

    factor = rng.uniform(*pert["mean_steps_scale"])
    new.anchor.slot_1_hour = source.anchor.slot_1_hour + int(
        rng.choice(pert["slot_1_delta_choices"]))

    borrowed: List[int] = []
    if rng.random() < borrow["lifestyle"]:
        new.lifestyle.weekday_loc_dist = peer.lifestyle.weekday_loc_dist
        new.lifestyle.weekend_loc_dist = peer.lifestyle.weekend_loc_dist
        borrowed.append(peer.anchor.source_uid)
    if rng.random() < borrow["activity"]:
        # Whole-profile activity borrow — much cleaner than the old 10-field copy.
        new.activity.steps10    = copy.deepcopy(peer.activity.steps10)
        new.activity.steps30pre = copy.deepcopy(peer.activity.steps30pre)
        borrowed.append(peer.anchor.source_uid)
    if rng.random() < borrow["compliance"]:
        new.compliance = copy.deepcopy(peer.compliance)
        borrowed.append(peer.anchor.source_uid)

    # Apply mean perturbation AFTER any borrow so the result is "peer-flavored
    # but not peer-equal" (in the activity-borrow branch).
    _scale_activity_means(new.activity, factor)

    new.anchor.borrowed_uids = sorted(set(borrowed))
    return new


def _build_edge(source: PersonaProfile, archetype_bounds: dict,
                 rng: np.random.Generator) -> PersonaProfile:
    """Push to archetype boundary: extreme activity (0.5× or 2×) / schedule.
    σ_log unchanged (shape orthogonal to mean in lognormal)."""
    pert = cfg.VARIANT_PERTURBATIONS["edge"]
    new = copy.deepcopy(source)
    new.anchor.variant_type = "edge"

    factor = float(rng.choice(pert["mean_steps_scale_choices"]))
    _scale_activity_means(new.activity, factor)

    if pert.get("slot_1_to_extreme"):
        b = archetype_bounds.get("slot_1_hour", (10, 14))
        new.anchor.slot_1_hour = int(rng.choice([b[0], b[1]]))
    return new


# ============================================================================
# Step 3: build all variants from all sources
# ============================================================================
def _archetype_bounds_table() -> Dict[str, Dict]:
    """Hard-coded archetype slot_1_hour bounds (could be derived from data later)."""
    return {
        "high_activity":  {"slot_1_hour": (10, 14)},
        "low_activity":   {"slot_1_hour": (10, 14)},
        "morning_active": {"slot_1_hour": (10, 11)},
        "evening_active": {"slot_1_hour": (14, 16)},
        "standard":       {"slot_1_hour": (12, 13)},
    }


# synth uids start at SYNTH_UID_OFFSET so they're visually distinguishable from
# real uids in logs/CSV. Real uids are < 100 in our dataset (37 users), so 1000
# leaves plenty of gap. The (source_uid, variant_type, index) info is preserved
# in dedicated columns — uid itself is now opaque.
SYNTH_UID_OFFSET = 1000


def build_synth_personas(real_profiles: Dict[int, PersonaProfile],
                         seed: int = 42) -> List[PersonaProfile]:
    rng = np.random.default_rng(seed)
    bounds_table = _archetype_bounds_table()

    by_archetype: Dict[str, List[PersonaProfile]] = {}
    for p in real_profiles.values():
        by_archetype.setdefault(p.anchor.archetype, []).append(p)

    out: List[PersonaProfile] = []
    archetype_counts: Dict[str, int] = {}
    next_synth_uid = SYNTH_UID_OFFSET
    for uid, source in real_profiles.items():
        archetype = source.anchor.archetype
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        peers = [p for p in by_archetype[archetype] if p.anchor.source_uid != uid]

        for i in range(cfg.VARIANTS_PER_SOURCE["twin"]):
            v = _build_twin(source, rng)
            v.anchor.synth_uid = next_synth_uid
            next_synth_uid += 1
            out.append(v)

        if peers:
            for i in range(cfg.VARIANTS_PER_SOURCE["sibling"]):
                peer = peers[int(rng.integers(0, len(peers)))]
                v = _build_sibling(source, peer, rng)
                v.anchor.synth_uid = next_synth_uid
                next_synth_uid += 1
                out.append(v)

        for i in range(cfg.VARIANTS_PER_SOURCE["edge"]):
            v = _build_edge(source, bounds_table.get(archetype, {}), rng)
            v.anchor.synth_uid = next_synth_uid
            next_synth_uid += 1
            out.append(v)

    print(f"\n[archetype] Classified {len(real_profiles)} source patients:")
    for a, n in sorted(archetype_counts.items(), key=lambda x: -x[1]):
        print(f"  {a:22s}  {n} patients")
    print(f"  → {len(out)} synth persona variants total "
          f"(uid range: {SYNTH_UID_OFFSET}..{next_synth_uid - 1})")
    return out
