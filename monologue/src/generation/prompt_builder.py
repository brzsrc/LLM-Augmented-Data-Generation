"""Compose LLM prompts that condition trajectory generation on:
  - the persona profile (semantic memory, ℳ_sem)
  - data-derived state-conditional steps10 signals (replaces LLM prior)
  - the persona's archetype + variant type
  - cumulative episodic memory (already-generated steps, ℳ_epi)
  - explicit anti-bias clauses

Output: a (system, user) prompt pair ready for llm.generate_text().
"""
from __future__ import annotations
from typing import Dict, List, Optional

from src import config as cfg


def _format_persona(p: Dict, current_state: Optional[Dict] = None) -> str:
    """Survey-format persona traits (Pay-What-LLM-Wants 2025: Survey 4.0% vs
    Storytelling 0.2% accuracy → keep key:value, do NOT prose-ify).

    `current_state` is used to pick weekday vs weekend location distribution.
    If omitted, falls back to weekday distribution (legacy behavior)."""
    lines = [
        f"User type: {p.get('archetype', 'unknown')} (variant: {p.get('variant_type', '-')})",
        f"Schedule: first decision around hour {p.get('slot_1_hour', '?')}",
    ]
    mp = p.get("steps10_mean_positive")
    if mp is not None:
        lines.append(f"Baseline mean steps10 (steps10 non-zero rows): {int(mp)}")
    elif p.get("steps10_mean") is not None:
        lines.append(f"Baseline mean steps10: {int(p['steps10_mean'])}")

    lines.append(f"Peak slot: {p.get('peak_slot', '?')}")
    if p.get("steps10_momentum_score_positive") is not None:
        lines.append(
            f"Momentum (corr steps30pre→steps10, steps10 non-zero rows): "
            f"{p.get('steps10_momentum_score_positive'):.2f}")
    if p.get("steps10_context_sensitivity_positive"):
        cs = p.get("steps10_context_sensitivity_positive", {})
        parts = [f"{k}={v:.0f}" for k, v in cs.items()]
        lines.append(f"Context sensitivity — STEPS10 NON-ZERO ROWS ONLY (std of steps10 by ...): {', '.join(parts)}")
    
    # Common locations — pick weekday vs weekend dist based on current_state's
    # weekday field (0..6, weekend = 5/6). Falls back to weekday_loc_dist when
    # current_state is None.
    weekday = (current_state or {}).get("weekday")
    is_weekend = weekday is not None and int(weekday) >= 5
    loc_dist = (p.get("weekend_loc_dist") if is_weekend
                else p.get("weekday_loc_dist"))
    if not loc_dist:                                          # fallback safety
        loc_dist = p.get("weekday_loc_dist") or p.get("location_distribution")
    if loc_dist:
        top = sorted(loc_dist.items(), key=lambda x: -x[1])[:3]
        label = "weekend" if is_weekend else "weekday"
        lines.append(f"Common locations ({label}): " + ", ".join(
            f"{loc}({pct:.0%})" for loc, pct in top))
    return "\n".join(lines)


def _format_context_conditional(p: Dict) -> str:
    """A: per-context mean steps10 — values are POSITIVE-ONLY so they live on
    the same non-zero subspace as the anchor (used after the hurdle gate)."""
    blocks = []
    if p.get("steps10_by_loc_positive"):
        items = sorted(p["steps10_by_loc_positive"].items(), key=lambda x: -x[1])
        blocks.append("  by loc:        " + " | ".join(f"{k}={v:.0f}" for k, v in items))
    if p.get("steps10_by_weather_positive"):
        items = sorted(p["steps10_by_weather_positive"].items(), key=lambda x: -x[1])
        blocks.append("  by weather:    " + " | ".join(f"{k}={v:.0f}" for k, v in items))
    if p.get("steps10_by_temp_positive"):
        items = sorted(p["steps10_by_temp_positive"].items(), key=lambda x: -x[1])
        blocks.append("  by temp:       " + " | ".join(f"{k}={v:.0f}" for k, v in items))
    if p.get("steps10_by_steps30pre_bin_positive"):
        # Bins are ordered — preserve insertion order, don't sort by value
        items = list(p["steps10_by_steps30pre_bin_positive"].items())
        blocks.append("  by steps30pre: " + " | ".join(f"{k}={v:.0f}" for k, v in items))
    if not blocks:
        return ""
    return ("Context-conditional mean steps10 — STEPS10 NON-ZERO ROWS ONLY "
            "(real history, avail=True and avail=False):\n" + "\n".join(blocks))


def _format_unavail_baseline(p: Dict) -> str:
    """B: per-slot positive baseline when avail=False — used by no-message
    avail=False branch's anchor_lookup. Positive-only so the chain doesn't
    undershoot after the hurdle gate."""
    base = p.get("steps10_avail_false_per_slot_mean_positive") or {}
    if not base:
        return ""
    parts = [f"slot {s}:{v:.0f}" for s, v in sorted(base.items())]
    return ("When unreachable (avail=False), per-slot baseline — STEPS10 NON-ZERO ROWS ONLY:\n"
            "  " + " | ".join(parts))


def _format_current_s30_bin(persona: Dict, current_state: Dict) -> str:
    """Annotate the current steps30pre value with the persona's bin label,
    e.g. '  (bin "23-90")'. Returns empty string if persona has no bin scheme."""
    edges  = persona.get("steps30pre_bin_edges")  or []
    labels = persona.get("steps30pre_bin_labels") or []
    v = current_state.get("steps30pre")
    if v is None or not edges or not labels:
        return ""
    label = cfg.s30_bin_label(v, edges, labels)
    return f'  (bin "{label}")'


def _format_compliance_phase(p: Dict, day: int) -> str:
    """C: dynamic header line per Park 2023 Generative Agents — phase + day
    + activity multiplier as a single mutating cue."""
    phases = p.get("compliance_phases") or []
    if not phases:
        return ""
    for ph in phases:
        rng = ph.get("day_range") or [1, 999]
        lo, hi = rng[0], rng[1]
        if lo <= day <= hi:
            mult = ph.get("activity_mult", 1.0)
            return (f"Engagement phase: {ph.get('name', '?')} "
                    f"(day {day} / range {lo}-{hi}, activity ×{mult:.2f})")
    return ""


def _format_high_activity_anchors(p: Dict) -> str:
    """D: concrete (slot, loc) bursts from real data. Per PICLe 2024: 3-10
    ranked exemplars suffice; more doesn't help."""
    anchors = p.get("steps10_high_activity_contexts_positive") or []
    if not anchors:
        return ""
    return ("High-activity context anchors (real bursts):\n  "
            + "\n  ".join(anchors[:5]))


def _format_signal(per_slot_signal: Dict) -> str:
    """Per-slot mean steps10 by action — POSITIVE ROWS ONLY. Lives on the
    non-zero subspace so the anchor + multiplier chain doesn't undershoot
    after the hurdle gate forces zeros at the correct rate."""
    if not per_slot_signal:
        return "  (no per-slot positive signal extracted — use default)"
    lines = ["Per-slot mean steps10 by action — STEPS10 NON-ZERO ROWS ONLY (avail=True):"]
    for slot, by_act in sorted(per_slot_signal.items()):
        parts = [f"a={a}:{r:.0f}" for a, r in sorted(by_act.items())]
        lines.append(f"  slot {slot}: " + " | ".join(parts))
    return "\n".join(lines)


def _format_episodic(history: List[Dict],
                      persona: Optional[Dict] = None) -> str:
    """Recently generated (s, a, post_steps) — keeps trajectory self-consistent.

    When `persona` is provided and the tail of `history` contains ≥1 zero,
    append a "→ recovery context" line citing the typical recovery magnitude
    for the matching zero-streak bucket (steps10_mean_after_zero_streak).
    LLM's step 5 (episodic_check) references this line when present.
    """
    if not history:
        return "  (this is the first decision today)"
    lines = ["Recent history within this trajectory:"]
    for h in history[-5:]:
        lines.append(
            f"  day={h.get('day','?')} slot={h.get('slot','?')} "
            f"send={h.get('action','?')} → steps10={h.get('steps10','?')}"
        )

    # Tail-zero streak → recovery-magnitude hint
    if persona is not None:
        streak = 0
        for h in reversed(history):
            if h.get("steps10", 1) == 0:
                streak += 1
            else:
                break
        if streak >= 1:
            bucket = "1" if streak == 1 else ("2" if streak == 2 else "3+")
            table  = persona.get("steps10_mean_after_zero_streak") or {}
            recov  = table.get(bucket)
            if recov is not None:
                lines.append(
                    f"  → recovery context: {streak} prior zero(s); "
                    f"this user's typical recovery for bucket '{bucket}' "
                    f"= {int(recov)} steps10")
            else:
                lines.append(
                    f"  → recovery context: {streak} prior zero(s); "
                    f"bucket '{bucket}' not fit for this user — "
                    f"use anchor/baseline instead")
    return "\n".join(lines)



# ============================================================================
# No Chain-of-Thought (CoT) --- Outputs a interger steps10 prediction 
# ============================================================================

SYSTEM_TEMPLATE = """You are a behavior simulator for HeartSteps users. Generate one
transition (next steps10 count) given the user's profile, current state, and
the message action taken.

Output ONLY an integer steps count between 0 and 10000. No explanation.

Important rules — DO NOT violate:
  * Messages do NOT universally help. Some users respond negatively, especially
    at the wrong time of day.
  * Match the user's profile EXACTLY — do not invent traits not in the profile.
  * Be stochastic — even the same state can produce different outcomes.
  * When "Available for intervention: False", the user is in a state where no
    message could be sent (driving / sleeping / unreachable); send is always 0
    and steps10 should reflect the user's natural baseline at this slot.
"""


def build_step_prompt(persona: Dict,
                      current_state: Dict,
                      action: int,
                      episodic_history: List[Dict]) -> tuple[str, str]:
    """Build (system, user) prompt for one transition step.

    `current_state` :
    {
        "day": day, "slot": slot, "hour": round(actual_hour, 1),
        "weekday": weekday,
        "avail": avail,
        **ctx: weather/temp/loc/steps30pre for this decision point,
    }
    `action` is the send chosen (will be in the prompt).
    `episodic_history` is the trajectory so far for this synth user.
    """
    
    day = current_state.get("day", 1)
    # Build profile sections, skip empty ones so the prompt stays tight.
    profile_sections = [
        "## Synthetic User Profile",
        _format_persona(persona, current_state),
        _format_signal(persona.get("steps10_avail_true_per_slot_action_mean_positive", {})),
        _format_unavail_baseline(persona),         # B
        _format_context_conditional(persona),      # A
        _format_high_activity_anchors(persona),    # D
    ]
    profile_block = "\n\n".join(s for s in profile_sections if s)

    # Engagement-phase header (C) prepended inside "Current Decision Context"
    # since it depends on `day` — same persona has different phase across days.
    phase_line = _format_compliance_phase(persona, day)
    phase_header = (phase_line + "\n") if phase_line else ""

    user = f"""\
{profile_block}

## Current Decision Context
{phase_header}Day: {day}
Slot: {current_state.get('slot', '?')} (hour ≈ {current_state.get('hour', '?')})
Weekday: {current_state.get('weekday', '?')}
Weather: {current_state.get('weather', '?')}, Temp: {current_state.get('temp', '?')}
Location: {current_state.get('loc', '?')}
steps30pre (prior 30-min step count): {current_state.get('steps30pre', '?')}{_format_current_s30_bin(persona, current_state)}
Available for intervention: {current_state.get('avail', '?')}
Action just taken: send={action}  ({cfg.ACTION_NAMES.get(action, '?')})

## Episodic Memory (this trajectory so far)
{_format_episodic(episodic_history, persona)}

## Task

Predict this user's steps10 count for the NEXT 10 minutes after the decision.
Consider:
  - The per-slot action signal (what THIS user does at this slot per action)
  - Context-conditional means above (steps10 by loc/weather/temp/steps30pre bin)
  - If avail=False, use the unavailable-baseline section, not the per-slot signal
  - Engagement phase multiplier (honeymoon ↑, fatigue ↓)
  - High-activity anchors hint at when this user actually bursts
  - The user's archetype's response style
  - Be realistic: many slots have 0 steps; only some are active windows.

Output ONLY one integer between 0 and 10000."""

    return SYSTEM_TEMPLATE, user


# ============================================================================
# Chain-of-Thought (CoT) variant — branches on `send`:
#   send > 0:  _cot_message  — full 6-field CoT WITH phase-multiplier step
#   send = 0:  _cot_no_message — 6-field schema, phase_application = "N/A",
#                                anchor source depends on avail (per_slot_action
#                                vs unavail_baseline)
#
# Why split on send (not avail): phase multiplier is a "message-response
# amplifier". When no message is sent (regardless of avail), there's nothing
# to amplify → phase_application should be N/A. Cases avail=True∧send=0
# (MRT chose no-msg) and avail=False (no-msg forced) share the same prompt
# structure but pick different anchor cells (Y1 vs Y2).
#
# Output schema is the SAME 6-field cfg.STEPS_COT_JSON_SCHEMA for both
# branches — phase_application accepts string, so "N/A — no message" is
# valid. Refs: Tam 2024 reasoning-before-value, Wang 2024 Chain-of-Table,
# Xu 2024 PAFT, Sidorenko 2025.
# ============================================================================

SYSTEM_TEMPLATE_COT_MESSAGE = """You are a behavior simulator for HeartSteps users.
A MESSAGE has just been sent (send > 0). Reason through 5 calibration steps
and then output a final integer steps10 prediction.

You are predicting ONLY the positive outcome. Whether this slot has any
activity at all (steps10 = 0) is decided by an external mechanism BEFORE
you are called. Always output a positive integer ≥ 1.

OUTPUT FORMAT — a JSON object with EXACTLY these 6 keys, in this order:

  1. "anchor_lookup"     — look up per_slot_action_mean_positive[slot][send] from
                            the "Per-slot mean steps10 by action — STEPS10 NON-ZERO ROWS
                            ONLY (avail=True)" table. Quote the actual non-zero
                            baseline, e.g. "slot=3 a=2 → 320".
  2. "phase_application" — apply the engagement-phase multiplier numerically
                            to the anchor, e.g. "honeymoon ×2.20 → 320×2.20=704".
  3. "context_adjustment" — adjust for loc / weather / temp / steps30pre using the
                             NON-ZERO context-conditional means, e.g.
                             "loc=work (210<252 baseline) → ~-8%"， 
                             or "loc=home neutral; temp=warm +5%".
  4. "momentum_check"    — apply steps30pre × momentum coefficient drag on the
                            anchor, e.g. "steps30pre=85 in low bin, momentum 0.40
                            → drag down ~25%"， or "steps30pre=857 in high bin, momentum 0.30 → boost ~20%".
  5. "episodic_check"    — 1-2 sentence sanity check vs. recent history. If
                            a "→ recovery context" line appears under the
                            history, reference and also consider it (cite the streak length and
                            the typical recovery magnitude).
  6. "value"             — FINAL positive integer steps10 in [1, 10000].

CRITICAL rules:
  * The 5 reasoning fields MUST be filled FIRST. Do not commit `value`
    until you've written all 5.
  * Reference SPECIFIC numbers from the profile blocks; do not invent.
  * Do NOT output 0. The zero/non-zero decision is handled externally.
"""

SYSTEM_TEMPLATE_COT_NO_MESSAGE = """You are a behavior simulator for HeartSteps users.
NO MESSAGE has been sent (send = 0). Predict the user's natural baseline
steps10 at this state — phase multiplier does NOT apply (nothing to amplify).

You are predicting ONLY the positive outcome. Whether this slot has any
activity at all (steps10 = 0) is decided by an external mechanism BEFORE
you are called. Always output a positive integer ≥ 1.

OUTPUT FORMAT — a JSON object with EXACTLY these 6 keys, in this order:

  1. "anchor_lookup"     — look up the appropriate NON-ZERO baseline cell:
       - If Available=True (user reachable; MRT chose no-msg arm):
           cite per_slot_action_mean_positive[slot][a=0] from the "Per-slot mean
           steps10 by action — STEPS10 NON-ZERO ROWS ONLY" table, e.g.
           "slot=3 a=0 → 130".
       - If Available=False (user unreachable; send forced to 0):
           cite per_slot_mean_positive[slot] from the "When unreachable
           (avail=False), per-slot baseline — STEPS10 NON-ZERO ROWS ONLY" table, e.g.
           "unavail slot=3 → 811".
  2. "phase_application" — write "N/A — no message to amplify".
  3. "context_adjustment" — adjust the baseline for loc / weather / temp / steps30pre 
                             using the NON-ZERO context-conditional means, e.g. 
                             "loc=work (210<252 baseline) → ~-8%"， 
                             or "loc=home neutral; temp=warm +5%".
  4. "momentum_check"    — apply steps30pre × momentum coefficient drag on the
                            anchor, e.g. "steps30pre=85 in low bin, momentum 0.40
                            → drag down ~25%", or "steps30pre=857 in high bin, momentum 0.30 → boost ~20%".
  5. "episodic_check"    — 1-2 sentence sanity check vs. recent history. If
                            a "→ recovery context" line appears under the
                            history, reference and also consider it (cite the streak length and
                            the typical recovery magnitude).
  6. "value"             — FINAL positive integer steps10 in [1, 10000].

CRITICAL rules:
  * The 5 reasoning fields MUST be filled FIRST.
  * Pick the right baseline table based on Available=True/False.
  * Reference SPECIFIC numbers from the profile blocks; do not invent.
  * Do NOT output 0. The zero/non-zero decision is handled externally.
"""

_TASK_BLOCK_MESSAGE = """## Task
Reason through the 5 reasoning fields in the JSON schema above. Each field
MUST reference SPECIFIC numbers from the profile blocks above
(e.g. "slot=3 a=2 → 218", "honeymoon ×2.20", "steps30pre=85 in low bin").

Output a POSITIVE integer ≥ 1 (do NOT output 0).
Output ONLY the JSON object — no markdown fences, no extra text."""

_TASK_BLOCK_NO_MESSAGE = """## Task
No message was sent. Reason through the 5 reasoning fields with the NON-ZERO
BASELINE as the anchor. In `anchor_lookup`, explicitly state which table
you used:
  - "per_slot_action_mean_positive[slot][a=0] = X" if Available=True
  - "per_slot_mean_positive[slot] = X (unavail baseline)" if Available=False
In `phase_application`, write "N/A — no message".

Output a POSITIVE integer ≥ 1 (do NOT output 0).
Output ONLY the JSON object — no markdown fences, no extra text."""


def _cot_user_block(persona: Dict, current_state: Dict, action: int,
                     episodic_history: List[Dict], task_block: str) -> str:
    """Shared USER-side construction for both CoT branches.
    Differs only in the `task_block` text appended at the end."""
    day = current_state.get("day", 1)
    profile_sections = [
        "## Synthetic User Profile",
        _format_persona(persona, current_state),
        _format_signal(persona.get("steps10_avail_true_per_slot_action_mean_positive", {})),
        _format_unavail_baseline(persona),
        _format_context_conditional(persona),
        _format_high_activity_anchors(persona),
    ]
    profile_block = "\n\n".join(s for s in profile_sections if s)

    phase_line = _format_compliance_phase(persona, day)
    phase_header = (phase_line + "\n") if phase_line else ""

    return f"""\
{profile_block}

## Current Decision Context
{phase_header}Day: {day}
Slot: {current_state.get('slot', '?')} (hour ≈ {current_state.get('hour', '?')})
Weekday: {current_state.get('weekday', '?')}
Weather: {current_state.get('weather', '?')}, Temp: {current_state.get('temp', '?')}
Location: {current_state.get('loc', '?')}
steps30pre (prior 30-min step count): {current_state.get('steps30pre', '?')}{_format_current_s30_bin(persona, current_state)}
Available for intervention: {current_state.get('avail', '?')}
Action just taken: send={action}  ({cfg.ACTION_NAMES.get(action, '?')})

## Episodic Memory (this trajectory so far)
{_format_episodic(episodic_history, persona)}

{task_block}"""

def _cot_message(persona: Dict, current_state: Dict, action: int,
                  episodic_history: List[Dict]) -> tuple[str, str]:
    """Branch X: avail=True AND send > 0 — full CoT with phase multiplier."""
    user = _cot_user_block(persona, current_state, action,
                            episodic_history, _TASK_BLOCK_MESSAGE)
    return SYSTEM_TEMPLATE_COT_MESSAGE, user


def _cot_no_message(persona: Dict, current_state: Dict, action: int,
                     episodic_history: List[Dict]) -> tuple[str, str]:
    """Branch Y: send = 0 (Y1 avail=T or Y2 avail=F) — baseline CoT, no
    phase mult. Anchor source picked at runtime by LLM based on avail field
    in current_state (instructions live in SYSTEM_TEMPLATE_COT_NO_MESSAGE)."""
    user = _cot_user_block(persona, current_state, action,
                            episodic_history, _TASK_BLOCK_NO_MESSAGE)
    return SYSTEM_TEMPLATE_COT_NO_MESSAGE, user


def build_step_prompt_cot(persona: Dict,
                           current_state: Dict,
                           action: int,
                           episodic_history: List[Dict]) -> tuple[str, str]:
    """CoT/JSON variant of build_step_prompt — dispatches by `action`:
      action > 0 → message-effect CoT (with phase multiplier)
      action = 0 → baseline CoT (no phase multiplier; anchor source decided
                                 by avail flag, instructed in SYSTEM prompt)

    LLM output: JSON matching cfg.STEPS_COT_JSON_SCHEMA (same schema both
    branches; phase_application accepts "N/A — no message" string when no
    message was sent). Use Qwen{8,32}BLLM.judge_steps_cot to consume.
    """
    if action > 0:
        return _cot_message(persona, current_state, action, episodic_history)
    return _cot_no_message(persona, current_state, action, episodic_history)


# Back-compat: the original single SYSTEM_TEMPLATE_COT alias still resolves
# (in case something imported it elsewhere). Points to the MESSAGE branch
# since that was the historical default behavior.
SYSTEM_TEMPLATE_COT = SYSTEM_TEMPLATE_COT_MESSAGE