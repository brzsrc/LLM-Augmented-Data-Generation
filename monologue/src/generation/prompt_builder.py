"""Compose LLM prompts that condition trajectory generation on:
  - the persona profile (semantic memory, ℳ_sem)
  - data-derived state-conditional steps10 signals (replaces LLM prior)
  - the persona's archetype + variant type
  - cumulative episodic memory (already-generated steps, ℳ_epi)
  - explicit anti-bias clauses

Output: a (system, user) prompt pair ready for llm.generate_text().
"""
from __future__ import annotations
from typing import Dict, List

from src import config as cfg


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


def _format_persona(p: Dict) -> str:
    """Survey-format persona traits (Pay-What-LLM-Wants 2025: Survey 4.0% vs
    Storytelling 0.2% accuracy → keep key:value, do NOT prose-ify)."""
    lines = [
        f"User type: {p.get('archetype', 'unknown')} (variant: {p.get('variant_type', '-')})",
        f"Schedule: first decision around hour {p.get('slot_1_hour', '?')}",
        f"Baseline mean steps10: {int(p['steps10_mean']) if p.get('steps10_mean') is not None else '?'}",
    ]
    zp = p.get("steps10_zero_pct")
    if zp is not None:
        lines.append(f"Zero-step rate (steps10=0): {zp:.0%}")
    lines.append(f"Peak slot: {p.get('peak_slot', '?')}")
    if p.get("steps10_momentum_score") is not None:
        lines.append(
            f"Momentum (corr steps30pre→steps10): {p.get('steps10_momentum_score'):.2f}")
    if p.get("steps10_context_sensitivity"):
        cs = p.get("steps10_context_sensitivity", {})
        parts = [f"{k}={v:.0f}" for k, v in cs.items()]
        lines.append(f"Context sensitivity (std of steps10 by ...): {', '.join(parts)}")
    if p.get("location_distribution"):
        top = sorted(p["location_distribution"].items(),
                     key=lambda x: -x[1])[:3]
        lines.append("Common locations: " + ", ".join(
            f"{loc}({pct:.0%})" for loc, pct in top))
    return "\n".join(lines)


def _format_context_conditional(p: Dict) -> str:
    """A: per-context mean steps10 — gives the LLM the μ per dimension, not
    just the σ exposed by context_sensitivity. Most impactful addition per the
    Bias-Adjusted-LLM template + Survey>Storytelling result."""
    blocks = []
    if p.get("steps10_by_loc"):
        items = sorted(p["steps10_by_loc"].items(), key=lambda x: -x[1])
        blocks.append("  by loc:        " + " | ".join(f"{k}={v:.0f}" for k, v in items))
    if p.get("steps10_by_weather"):
        items = sorted(p["steps10_by_weather"].items(), key=lambda x: -x[1])
        blocks.append("  by weather:    " + " | ".join(f"{k}={v:.0f}" for k, v in items))
    if p.get("steps10_by_temp"):
        items = sorted(p["steps10_by_temp"].items(), key=lambda x: -x[1])
        blocks.append("  by temp:       " + " | ".join(f"{k}={v:.0f}" for k, v in items))
    if p.get("steps10_by_steps30pre_bin"):
        # Bins are ordered — preserve insertion order, don't sort by value
        items = list(p["steps10_by_steps30pre_bin"].items())
        blocks.append("  by steps30pre: " + " | ".join(f"{k}={v:.0f}" for k, v in items))
    if not blocks:
        return ""
    return "Context-conditional mean steps10 (from real history when avail=True and avail=False):\n" + "\n".join(blocks)


def _format_unavail_baseline(p: Dict) -> str:
    """B: per-slot baseline when avail=False. Critical because per_slot_signal
    is computed from avail=True rows only → using it for avail=False is wrong.
    """
    base = p.get("steps10_avail_false_per_slot_mean") or {}
    if not base:
        return ""
    parts = [f"slot {s}:{v:.0f}" for s, v in sorted(base.items())]
    return ("When unreachable (avail=False), per-slot baseline steps10:\n"
            "  " + " | ".join(parts))


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
    anchors = p.get("steps10_high_activity_contexts") or []
    if not anchors:
        return ""
    return ("High-activity context anchors (real bursts):\n  "
            + "\n  ".join(anchors[:5]))


def _format_signal(per_slot_signal: Dict) -> str:
    """Per-slot mean steps10 by action — embedded so LLM doesn't use its prior."""
    if not per_slot_signal:
        return "  (no per-slot signal extracted — use default)"
    lines = ["Per-slot mean steps10 by action (from this user's REAL history only when avail=True):"]
    for slot, by_act in sorted(per_slot_signal.items()):
        parts = [f"a={a}:{r:.0f}" for a, r in sorted(by_act.items())]
        lines.append(f"  slot {slot}: " + " | ".join(parts))
    return "\n".join(lines)


def _format_episodic(history: List[Dict]) -> str:
    """Recently generated (s, a, post_steps) — keeps trajectory self-consistent."""
    if not history:
        return "  (this is the first decision today)"
    lines = ["Recent history within this trajectory:"]
    for h in history[-5:]:
        lines.append(
            f"  day={h.get('day','?')} slot={h.get('slot','?')} "
            f"send={h.get('action','?')} → steps10={h.get('steps10','?')}"
        )
    return "\n".join(lines)


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
        _format_persona(persona),
        _format_signal(persona.get("steps10_avail_true_per_slot_action_mean", {})),
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
steps30pre (prior 30-min step count): {current_state.get('steps30pre', '?')}
Available for intervention: {current_state.get('avail', '?')}
Action just taken: send={action}  ({cfg.ACTION_NAMES.get(action, '?')})

## Episodic Memory (this trajectory so far)
{_format_episodic(episodic_history)}

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
