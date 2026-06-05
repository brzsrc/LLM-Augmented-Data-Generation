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
    
def _format_avail_baseline(steps10_avail_true_per_slot_action_mean: Dict) -> str:
    """Per-slot POSITIVE-only mean steps10 by action — anchor for the
    positive branch of zero_check. Computed from real rows where steps10 > 0,
    so the values represent E[steps10 | slot, action, steps10>0]. The
    bimodality (zero vs positive) is handled by zero_check separately."""
    if not steps10_avail_true_per_slot_action_mean:
        return "  (no per-slot positive signal extracted — use default)"
    lines = ["Per-slot mean steps10 by action — STEPS10>0 ROWS ONLY (avail=True):"]
    for slot, by_act in sorted(steps10_avail_true_per_slot_action_mean.items()):
        parts = [f"a={a}:{r:.0f}" for a, r in sorted(by_act.items())]
        lines.append(f"  slot {slot}: " + " | ".join(parts))
    return "\n".join(lines)



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


def _format_zero_baseline(persona: Dict, current_state: Dict) -> str:
    """Per-cell P(steps10=0) baselines for the zero_check reasoning step.

    Three signals (any subset may be empty depending on persona richness):
      - per-slot zero rate, picking avail_true vs avail_false table
      - per-steps30pre-bin zero rate (always from avail_true)
      - streak transition (P(zero | prev=0) vs P(zero | prev>0))
    """
    avail = bool(current_state.get("avail", True))
    table_key = ("steps10_avail_true_per_slot_zero_pct" if avail
                 else "steps10_avail_false_per_slot_zero_pct")
    per_slot = persona.get(table_key) or {}

    lines: List[str] = []
    if per_slot:
        parts = [f"slot {s}:{int(round(float(p) * 100))}%"
                 for s, p in sorted(per_slot.items(), key=lambda x: int(x[0]))]
        lines.append(f"  by slot (avail={avail}): " + " | ".join(parts))

    bin_table = persona.get("steps10_avail_true_zero_pct_by_s30_bin") or {}
    if bin_table:
        parts = [f"{k}:{int(round(float(p) * 100))}%"
                 for k, p in bin_table.items()]
        lines.append("  by steps30pre bin: " + " | ".join(parts))

    pair = persona.get("steps10_momentum_pair_pct") or {}
    za, zn = pair.get("zero_after_zero"), pair.get("zero_after_nonzero")
    if za is not None and zn is not None:
        lines.append(
            f"  streak effect: after prev=0 → {int(round(float(za)*100))}% zero; "
            f"after prev>0 → {int(round(float(zn)*100))}% zero")

    if not lines:
        return ""
    return ("Zero-rate baseline — fraction of windows with steps10=0:\n"
            + "\n".join(lines))


def _format_positive_quantiles(persona: Dict, current_state: Dict) -> str:
    """Per-slot positive-distribution quantiles (steps10 | steps10>0).

    Exposes the right-skewed tail so the LLM doesn't snap to the mean
    when building the positive value. Marks the current slot with an arrow.
    """
    avail = bool(current_state.get("avail", True))
    key = ("steps10_avail_true_per_slot_positive_quantiles" if avail
           else "steps10_avail_false_per_slot_positive_quantiles")
    qtab = persona.get(key) or {}
    if not qtab:
        return ""

    cur_slot = current_state.get("slot")
    try:
        cur_slot_i = int(cur_slot) if cur_slot is not None else None
    except (TypeError, ValueError):
        cur_slot_i = None

    lines = ["Positive-distribution quantiles (steps10 | steps10>0) — "
              "magnitude shape for value:"]
    for slot, q in sorted(qtab.items(), key=lambda x: int(x[0])):
        parts = [f"p{k}={int(v)}" for k, v in
                 sorted(q.items(), key=lambda x: int(x[0]))]
        marker = "  ← current slot" if int(slot) == cur_slot_i else ""
        lines.append(f"  slot {slot}: " + " | ".join(parts) + marker)
    lines.append("  → ~5% of positive windows EXCEED p95; ~1% EXCEED p99.")
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
A MESSAGE has just been sent (send > 0). Predict steps10 for this 10-minute window.

CRITICAL CALIBRATION FACT:
  Real HeartSteps data has steps10 = 0 in ~54% of windows.
  Most windows the user is sitting / sleeping / unreachable → 0 steps.
  Only ~46% of windows actually accumulate any steps.
  Your output MUST reflect this. Use zero_check (step 1) to decide 0 vs
  positive based on state cues, not by averaging out a positive guess.
  NOTE: per-action zero rates are essentially equal across send (a=0 52%,
  a=1 58%, a=2 55%). Sending a message does NOT meaningfully reduce zero
  probability — it modulates the POSITIVE magnitude when activity happens,
  not whether activity happens. Do not under-zero on send=1 or send=2.

OUTPUT FORMAT — a JSON object with EXACTLY these 7 keys, in this order:

  1. "zero_check"        — Decide whether THIS window's steps10 is 0 or >0.
                            Format the field as:
                              "decision=<zero|positive> — <slot baseline%>,
                               <s30 bin%>, <streak hint>"
                            Cite ALL THREE numbers from the "Zero-rate
                            baseline" block. Examples:
                              "decision=zero — slot=1 70%, s30=12 in '1-80' 64%,
                               prev=0 streak → 65% zero"
                              "decision=positive — slot=3 41%, s30=420 in '396-857'
                               14%, prev>0 → 42% zero"
                            Bias toward 'zero' when ALL THREE of:
                              (low steps30pre bin, morning slot,
                               prev=0 streak)

  2. "anchor_lookup"     — IF zero_check decision="positive": look up
                            per_slot_action_mean_positive[slot][send] from
                            the "Per-slot mean steps10 by action — STEPS10>0
                            ROWS ONLY" table AND the slot's positive quantiles.
                            Quote both, e.g.
                              "mean=218 (slot=3 a=2 positive); p50=180 p75=320
                               p95=720 — anchor is CENTER, real spans wide".
                            IF decision="zero": write "N/A — zero_check decided zero".

  3. "phase_application" — IF positive: apply engagement-phase multiplier to
                            the anchor, e.g. "honeymoon ×2.20 → 218×2.20=480".
                            IF zero: "N/A".

  4. "context_adjustment" — IF positive: adjust for loc / weather / temp using
                             the context-conditional means, e.g.
                             "loc=work (139<146 baseline) → ~-8%". IF zero: "N/A".

  5. "momentum_check"    — IF positive: adjust for steps30pre × momentum
                            coefficient, e.g. "steps30pre=420 high bin,
                            momentum 0.40 → boost ~30%". IF zero: "N/A".

  6. "episodic_check"    — 1-2 sentence sanity check vs. recent history,
                            regardless of zero_check decision.

  7. "value"             — FINAL integer steps10 in [0, 10000].
                            IF zero_check decision="zero": output 0.
                            ELSE: integer derived from steps 2-5. Real positive
                            distribution is right-skewed, but the BULK sits
                            LOW:
                              ~50% of positives < p50 (median ≈ 100)
                              ~25% in p25-p50 range
                              ~25% in p50-p75 range
                              ~ 6% in p75-p95 (mid-tail)
                              ~ 1% > p95 (true bursts)
                            DEFAULT to p25-p50 region. Move past p75 only when
                            ALL of (high s30, exercise/activity loc, peak slot
                            or high-activity anchor match) are present. Reach
                            p95+ ONLY for clear exercise events (~1% of cases).
                            DO NOT use p95/p99 as the typical anchor — they
                            are the rare-burst ceiling, not the default.
                            Real values [1, 17] are essentially noise (~1%
                            of positives); prefer 0 or ≥ 18.

CRITICAL rules:
  * zero_check MUST be done FIRST. Do not skip it to fill positive chain.
  * Reference SPECIFIC numbers from the profile blocks; do not invent.
  * Across many windows, your zero-decision rate should approach ~54%
    overall, modulated by state cues (morning ~70%, evening ~50%).
"""

SYSTEM_TEMPLATE_COT_NO_MESSAGE = """You are a behavior simulator for HeartSteps users.
NO MESSAGE has been sent (send = 0). Predict the user's natural baseline
steps10 at this state — phase multiplier does NOT apply (nothing to amplify).

CRITICAL CALIBRATION FACT:
  Real HeartSteps data has steps10 = 0 in ~54% of windows overall.
  For Available=True + send=0: ~57% zero.
  For Available=False: ~35% zero (unavailable often means
  commuting/exercising — actively moving).
  Note: send=0 just means no message was sent; it does NOT imply the
  user is in a high-activity state. Let state cues (slot, s30, streak)
  drive zero_check — don't default to "positive" just because no
  intervention happened.
  Use zero_check (step 1) to decide 0 vs positive based on state cues.

OUTPUT FORMAT — a JSON object with EXACTLY these 7 keys, in this order:

  1. "zero_check"        — Decide whether THIS window's steps10 is 0 or >0.
                            Same format / rules as the MESSAGE prompt. Cite
                            slot baseline (picking avail-correct row),
                            steps30pre bin rate, and streak hint. Examples:
                              "decision=zero — avail=True slot=2 48%, s30=15
                               in '1-80' 64%, prev=0 → 65% zero"
                              "decision=positive — avail=False slot=3 35%,
                               s30=520 in '396-857' 14%, prev>0 → 42% zero"
                            Bias toward 'zero' when ALL THREE of:
                              (low steps30pre bin, morning slot,
                               prev=0 streak)

  2. "anchor_lookup"     — IF zero_check decision="positive":
       - If Available=True (user reachable; MRT chose no-msg arm):
           cite per_slot_action_mean_positive[slot][a=0] AND positive
           quantiles for the slot, e.g.
             "mean=120 (slot=3 a=0 positive); p50=80 p95=480".
       - If Available=False (user unreachable; send forced to 0):
           cite per-slot unavail baseline AND quantiles, e.g.
             "unavail slot=3 → 548; p95=900".
                            IF decision="zero": "N/A — zero_check decided zero".

  3. "phase_application" — Write "N/A — no message to amplify".

  4. "context_adjustment" — IF positive: adjust the baseline for loc /
                             weather / temp using context-conditional means,
                             e.g. "loc=home neutral; temp=warm +5%".
                             IF zero: "N/A".

  5. "momentum_check"    — IF positive: adjust for steps30pre × momentum
                            coefficient. IF zero: "N/A".

  6. "episodic_check"    — 1-2 sentence sanity check vs. recent history.

  7. "value"             — FINAL integer steps10 in [0, 10000].
                            IF zero_check decision="zero": output 0.
                            ELSE: integer from steps 2-5. Real positive
                            distribution is right-skewed, but the BULK sits
                            LOW:
                              ~50% of positives < p50 (median ≈ 100)
                              ~25% in p25-p50, ~25% in p50-p75
                              ~ 6% in p75-p95 (mid-tail)
                              ~ 1% > p95 (true bursts)
                            DEFAULT to p25-p50 region. Move past p75 only
                            when ALL of (high s30, exercise loc / avail=False
                            commuting context, peak slot) are present. Reach
                            p95+ ONLY for clear exercise events (~1%).
                            DO NOT use p95/p99 as the typical anchor —
                            they are the rare-burst ceiling, not the default.
                            Prefer 0 or ≥ 18; values in [1, 17] are noise.

CRITICAL rules:
  * zero_check MUST be done FIRST.
  * Pick the right baseline table (avail_true positive vs avail_false unavail).
  * Reference SPECIFIC numbers from the profile blocks; do not invent.
"""


def _cot_user_block(persona: Dict, current_state: Dict, action: int,
                     episodic_history: List[Dict], task_block: str) -> str:
    """Shared USER-side construction for both CoT branches.
    Differs only in the `task_block` text appended at the end.

    Profile-block layout intentionally puts zero baseline FIRST and
    positive quantiles right after the per-slot×action anchor, so the
    LLM's eye-path matches the CoT order (zero_check → anchor_lookup → ...).
    Anchor source is per_slot_action_mean_positive (conditional on >0)
    because zero_check now owns the 0/positive split.
    """
    day = current_state.get("day", 1)
    profile_sections = [
        "## Synthetic User Profile",
        _format_persona(persona),
        _format_zero_baseline(persona, current_state),
        _format_avail_baseline(persona.get("steps10_avail_true_per_slot_action_mean_positive", {})),
        _format_positive_quantiles(persona, current_state),
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
steps30pre (prior 30-min step count): {current_state.get('steps30pre', '?')}
Available for intervention: {current_state.get('avail', '?')}
Action just taken: send={action}  ({cfg.ACTION_NAMES.get(action, '?')})

## Episodic Memory (this trajectory so far)
{_format_episodic(episodic_history)}

{task_block}"""


_TASK_BLOCK_MESSAGE = """## Task
Reason through the 6 reasoning fields (zero_check first, then the positive
chain) in the JSON schema above. Each field MUST reference SPECIFIC numbers
from the profile blocks above (e.g. "slot=3 a=2 positive → 218",
"honeymoon ×2.20", "steps30pre=85 in low bin", "slot=1 zero baseline 70%").

Remember: zero_check decides 0 vs positive based on the Zero-rate baseline
block. ~54% of windows are 0 in real data; your decisions across this
trajectory should reflect that overall.

Output ONLY the JSON object — no markdown fences, no extra text."""

_TASK_BLOCK_NO_MESSAGE = """## Task
No message was sent. First do zero_check using the Zero-rate baseline block
(avail-conditional: avail=True+send=0 ~57% zero; avail=False ~35% zero).
If decision="positive", reason through the BASELINE anchor chain:
  - "per_slot_action_positive[slot][a=0] = X" if Available=True
  - "unavail_baseline[slot] = X" if Available=False
In `phase_application`, write "N/A — no message".

Output ONLY the JSON object — no markdown fences, no extra text."""


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
