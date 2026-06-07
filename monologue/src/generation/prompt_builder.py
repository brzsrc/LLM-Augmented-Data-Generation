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


def _format_persona(p: Dict, current_state: Dict | None = None) -> str:
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
    # Pick weekday vs weekend loc dist by current state's weekday (0=Mon..6=Sun;
    # <5 weekday, >=5 weekend). Falls back to weekday if state missing.
    wd = (current_state or {}).get("weekday")
    try:
        kind = "weekend" if int(wd) >= 5 else "weekday"
    except (TypeError, ValueError):
        kind = "weekday"
    dist = p.get(f"{kind}_loc_dist") or {}
    if dist:
        top = sorted(dist.items(), key=lambda x: -x[1])[:3]
        lines.append(f"Common locations ({kind}): " + ", ".join(
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


def _format_zero_baseline(persona: Dict, current_state: Dict,
                            action: int | None = None) -> str:
    """Per-cell P(steps10=0) baselines for the zero_check reasoning step.

    Up to four signals (any subset may be empty depending on persona richness):
      - per-slot zero rate, picking avail_true vs avail_false table
      - per-steps30pre-bin zero rate (always from avail_true)
      - per-loc zero rate, picking avail_true vs avail_false table — current
        location marked with an arrow
      - per-action zero rate (avail=True only — avail=False has send forced 0).
        Current action marked with an arrow.
    """
    avail = bool(current_state.get("avail", True))
    slot_key = ("steps10_avail_true_per_slot_zero_pct" if avail
                else "steps10_avail_false_per_slot_zero_pct")
    per_slot = persona.get(slot_key) or {}

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

    loc_key = ("steps10_avail_true_per_loc_zero_pct" if avail
               else "steps10_avail_false_per_loc_zero_pct")
    per_loc = persona.get(loc_key) or {}
    if per_loc:
        cur_loc = current_state.get("loc")
        parts = []
        for loc, pct in sorted(per_loc.items(), key=lambda x: -float(x[1])):
            marker = " ← current" if loc == cur_loc else ""
            parts.append(f"{loc}:{int(round(float(pct)*100))}%{marker}")
        lines.append(f"  by loc (avail={avail}): " + " | ".join(parts))
    else:
        # Fallback: per-loc cells too thin → cite the bucket's overall zero rate
        # as a single number. Mostly hits avail=False users with sparse buckets
        # (~11% of personas). LLM still has SOMETHING to cite for the "loc rate".
        scalar_key = ("steps10_avail_true_zero_pct" if avail
                      else "steps10_avail_false_zero_pct")
        scalar = persona.get(scalar_key)
        if scalar is not None:
            lines.append(
                f"  by loc (avail={avail}): overall={int(round(float(scalar)*100))}% "
                f"(per-loc cells too thin to split)")

    # by send — avail=True branch only (avail=False has send forced 0).
    if avail:
        per_action = persona.get("steps10_avail_true_per_action_zero_pct") or {}
        if per_action:
            parts = []
            for a, pct in sorted(per_action.items(), key=lambda x: int(x[0])):
                marker = " ← current" if action is not None and int(a) == int(action) else ""
                parts.append(f"a={a}:{int(round(float(pct)*100))}%{marker}")
            lines.append("  by send (avail=True): " + " | ".join(parts))

    if not lines:
        return ""
    return ("Zero-rate baseline — fraction of windows with steps10=0:\n"
            + "\n".join(lines))


def _format_positive_quantiles(persona: Dict, current_state: Dict) -> str:
    """Per-slot positive-distribution quantiles (avail=True branch only).

    Primary: per-slot quantiles (≥20 positive rows per slot, ~38% of personas).
    Fallback: flat per-user quantiles pooled across slots (covers the remaining
    ~62% of personas where per-slot cells are too thin). Marks current slot
    when per-slot is available. For avail=False branch, use
    `_format_unavail_quantiles`.
    """
    qtab = persona.get("steps10_avail_true_per_slot_positive_quantiles") or {}
    if qtab:
        cur_slot = current_state.get("slot")
        try:
            cur_slot_i = int(cur_slot) if cur_slot is not None else None
        except (TypeError, ValueError):
            cur_slot_i = None

        lines = ["Positive-distribution quantiles (steps10 | steps10>0, avail=True) — "
                  "magnitude shape for value:"]
        for slot, q in sorted(qtab.items(), key=lambda x: int(x[0])):
            parts = [f"p{k}={int(v)}" for k, v in
                     sorted(q.items(), key=lambda x: int(x[0]))]
            marker = "  ← current slot" if int(slot) == cur_slot_i else ""
            lines.append(f"  slot {slot}: " + " | ".join(parts) + marker)
        lines.append("  → ~5% of positive windows EXCEED p95; ~1% EXCEED p99.")
        return "\n".join(lines)

    # Fallback: flat per-user quantiles (per-slot cells too thin for this user).
    q_user = persona.get("steps10_avail_true_positive_quantiles_user") or {}
    if not q_user:
        return ""
    parts = [f"p{k}={int(v)}" for k, v in sorted(q_user.items(), key=lambda x: int(x[0]))]
    return ("Positive-distribution quantiles (steps10 | steps10>0, avail=True, "
            "user-level across all slots — per-slot cells too thin):\n"
            f"  {' | '.join(parts)}\n"
            "  → ~5% of positive windows EXCEED p95; ~1% EXCEED p99.")


def _format_unavail_quantiles(persona: Dict) -> str:
    """Flat per-user positive quantiles for the avail=False branch.

    Primary: avail=False user-level quantiles (per-(user, slot) cells too thin).
    Fallback: avail=True user-level quantiles for the ~3% of users with <5
    positive avail=False rows. Caveat: avail=True magnitudes likely under-
    estimate the commute/exercise tail, but better than no anchor.
    """
    q = persona.get("steps10_avail_false_positive_quantiles_user") or {}
    if q:
        parts = [f"p{k}={int(v)}" for k, v in sorted(q.items(), key=lambda x: int(x[0]))]
        return ("Positive-distribution quantiles (steps10 | steps10>0, avail=False, "
                "user-level across all slots):\n"
                f"  {' | '.join(parts)}\n"
                "  → ~5% of unavail positive windows EXCEED p95; ~1% EXCEED p99.")

    # Fallback: avail=False positive rows too few → use avail=True quantiles.
    q_fb = persona.get("steps10_avail_true_positive_quantiles_user") or {}
    if not q_fb:
        return ""
    parts = [f"p{k}={int(v)}" for k, v in sorted(q_fb.items(), key=lambda x: int(x[0]))]
    return ("Positive-distribution quantiles (avail=False positive cells too thin; "
            "showing avail=True user-level as proxy — likely UNDERESTIMATES the "
            "commute/exercise tail):\n"
            f"  {' | '.join(parts)}\n"
            "  → ~5% of positive windows EXCEED p95; ~1% EXCEED p99.")


def _format_episodic(history: List[Dict]) -> str:
    """Recently generated (s, a, post_steps) — keeps trajectory self-consistent.
    Includes loc/weather/temp/avail so episodic_check can compare context.
    """
    if not history:
        return "  (this is the first decision today)"
    lines = ["Recent history within this trajectory:"]
    for h in history[-5:]:
        lines.append(
            f"  day={h.get('day','?')} slot={h.get('slot','?')} "
            f"loc={h.get('loc','?')} weather={h.get('weather','?')} "
            f"temp={h.get('temp','?')} avail={h.get('avail','?')} "
            f"send={h.get('action','?')} → steps10={h.get('steps10','?')}"
        )
    return "\n".join(lines)


_COMMON_HEADER_TPL = """You are a behavior simulator for HeartSteps users.
{branch_intro}

CRITICAL CALIBRATION FACT:
{calibration_block}

OUTPUT FORMAT — a JSON object with EXACTLY these 8 keys, in this order:

  1. "zero_check"        — Decide whether THIS window's steps10 is 0 or >0.
{zero_check_format_and_count}
{zero_check_examples}
                            P_zero computation (commit to a number, not just
                            a label):
                              1) mean = average of the cited rates
                              2) boost = +15pp if ≥3 rates >55%
                                        = -15pp if ≥3 rates <45%
                                        =   0   otherwise
                              3) P_zero = clip(mean + boost, 5, 95)
                            decision is a BERNOULLI DRAW weighted by P_zero,
                            NOT a hard threshold. For THIS call, sample:
                              decision = "zero"     with prob = P_zero/100
                                       = "positive" with prob = 1 − P_zero/100
                            Examples:
                              P_zero=80 → ~80% of similar calls say "zero",
                                          ~20% say "positive"
                              P_zero=50 → ~50/50 split — do NOT auto-pick
                                          "zero" just because it's the tie
                              P_zero=15 → ~15% "zero", ~85% "positive"
                            Across the trajectory, the empirical "zero" rate
                            must track the mean of P_zero. Borderline cases
                            (P_zero 40-60) MUST split — never collapse to one
                            side.

  2. "anchor_lookup"     — {anchor_lookup_body}

  3. "phase_application" — {phase_application}

  4. "context_adjustment" — IF positive: adjust for loc / weather / temp using
                             the context-conditional means, e.g.
                             "loc=work (139<146 baseline) → ~-8%". IF zero: "N/A".

  5. "momentum_check"    — IF positive: adjust for steps30pre × momentum
                            coefficient, e.g. "steps30pre=420 high bin,
                            momentum 0.40 → boost ~30%". IF zero: "N/A".

  6. "episodic_check"    — 1-2 sentence sanity check vs. recent history,
                            regardless of zero_check decision.

  7. "value_band"        — Quantile band the final value will land in. ONE
                            of: "zero", "<p25", "p25-p50", "p50-p75",
                            "p75-p95", ">p95".
                            IF zero_check decision="zero": output "zero".
                            ELSE: commit to the band BEFORE the integer.
                            Procedure:
                              a) Apply phase/context/momentum to the p50
                                 anchor from anchor_lookup.
                              b) Identify the band the adjusted value lies in.
                              c) VERIFY the band matches the branch default
                                 rule (see field 8). If signals are weak but
                                 the band is high, PULL BACK to the default
                                 region. If signals are strong, the higher
                                 band is justified.
                            Right-skew reference (population-level):
                              ~50% of positives < p50 (median)
                              ~25% in p25-p50 range
                              ~25% in p50-p75 range
                              ~ 20% in p75-p95 (mid-tail)
                              ~ 5% > p95 (true bursts)

  8. "value"             — FINAL integer steps10 in [0, 10000].
                            IF value_band="zero": output 0.
                            ELSE: integer consistent with the band declared
                            in field 7. Pick a number inside the band's
                            range as quoted in anchor_lookup (e.g. if
                            value_band="p50-p75" and anchor_lookup shows
                            "p25-p75 = 100-320", value must be in roughly
                            [p50, 320]; use p50 ≤ value ≤ p75 from the
                            anchor block).
{value_default_hint}
                            DO NOT use p95/p99 as the typical anchor — they
                            are the rare-burst ceiling, not the default.
                            Real values [1, 17] are essentially noise (~1%
                            of positives); prefer 0 or ≥ 18.

CRITICAL rules:
  * zero_check MUST be done FIRST. Do not skip it to fill positive chain.
  * Reference SPECIFIC numbers from the profile blocks; do not invent.
{extra_rule}"""


# Shared spec text — branches with vs without the per-action signal.
# avail=True branches (MESSAGE, NO_MSG_AVAIL_T) cite 4 numbers; avail=False
# (NO_MSG_AVAIL_F) cites 3, since send is structurally 0.
_ZERO_CHECK_FORMAT_4 = """                            Format the field as:
                              "decision=<zero|positive> (P_zero=<NN>%) —
                               <slot baseline%>, <s30 bin%>, <loc rate%>,
                               <send rate%>"
                            Cite ALL FOUR numbers from the "Zero-rate
                            baseline" block. Examples:"""

_ZERO_CHECK_FORMAT_3 = """                            Format the field as:
                              "decision=<zero|positive> (P_zero=<NN>%) —
                               <slot baseline%>, <s30 bin%>, <loc rate%>"
                            Cite ALL THREE numbers from the "Zero-rate
                            baseline" block. Examples:"""


# --- BRANCH BODY: MESSAGE (send > 0) ----------------------------------------
_BRANCH_BODY_MESSAGE = dict(
    branch_intro="A MESSAGE has just been sent (send > 0). Predict steps10 for this 10-minute window.",
    calibration_block="""  Within the randomized arm (Available=True, where MESSAGE applies),
  steps10 = 0 in ~56% of windows and >0 in ~44% of windows. Across many
  similar prompts, your decisions should split roughly 56/44 — neither
  over-zero nor over-positive. Use zero_check (step 1) to call each
  window based on the cited rates.
  NOTE: at the POPULATION level, per-action zero rates are nearly equal
  (a=0 57%, a=1 58%, a=2 55%) — sending doesn't broadly reduce P(zero).
  But the PER-USER "by send" row in the Zero-rate baseline can diverge for
  message-responsive users; treat that personalized rate as a real cue
  when it differs from the population baseline. Send still modulates the
  POSITIVE magnitude (handled in anchor_lookup), not just P(zero).""",
    zero_check_format_and_count=_ZERO_CHECK_FORMAT_4,
    zero_check_examples='''                              "decision=zero (P_zero=79%) — slot=1 70%,
                               s30=12 in '1-80' 64%, loc=home 62% zero,
                               a=1 58% zero"  (mean 63.5, 4/4>55 → +15)
                              "decision=positive (P_zero=22%) — slot=3 41%,
                               s30=420 in '396-857' 14%, loc=work 38% zero,
                               a=2 55% zero"  (mean 37, 3/4<45 → -15)''',
    anchor_lookup_body="""IF zero_check decision="positive":
                            Step 1 — Cite the slot's positive quantile bands
                              from the profile, e.g.
                                "slot=3 a=2 positive: p25=100, p50=180,
                                 p75=320, p95=720"
                            Step 2 — STOCHASTICALLY pick a band per the
                              right-skew weights (default; see value field
                              for signal-driven shifts):
                                25% <p25 | 25% p25-p50 | 25% p50-p75
                                | 20% p75-p95 | 5% >p95
                            Step 3 — Within the chosen band, pick a
                              representative point (band CENTER for neutral
                              signals; LOW edge for weak; HIGH edge for strong).
                              State as:
                                "sampled band=p50-p75, anchor=240 (mid)"
                            Step 4 — That anchor is the input to phase/ctx/
                              momentum adjustments below.
                            IF decision="zero": "N/A — zero_check decided zero".""",
    phase_application="""IF positive: apply engagement-phase multiplier to
                            the sampled anchor, e.g.
                            "honeymoon ×2.20 → 240×2.20=528".
                            IF zero: "N/A".""",
    value_default_hint="""                            Adjust the band-sampling weights from the default
                            25/25/25/20/5 based on signal strength:
                              • Strong upward (high s30 AND exercise/peak loc
                                AND/OR high-activity anchor match) → shift
                                ~15pp from <p50 to p75+.
                              • Weak (low s30 + sedentary loc) → shift ~10pp
                                from p50+ to <p25.
                              • Neutral → keep default weights.
                            NEVER force >p95 deterministically — keep <10%
                            mass even under strong signals (true bursts ~1%).
                            The sampled band BEFORE adjustments goes into
                            value_band; after adjustments value should sit
                            inside (or one band away from) that band.""",
    extra_rule="""  * Across many windows, your zero-decision rate should approach ~56%
    (avail=True overall), modulated by state cues. The split is roughly
    56/44 — do not collapse to 100% zero on borderline cases.""",
)

# --- BRANCH BODY: NO_MSG_AVAIL_T (send=0 ∧ avail=True) ----------------------
_BRANCH_BODY_NO_MSG_AVAIL_T = dict(
    branch_intro=("NO MESSAGE has been sent (send = 0) but the user is REACHABLE "
                  "(avail = True). MRT chose the no-message arm. Predict the user's "
                  "natural baseline at this state — phase multiplier does NOT apply."),
    calibration_block="""  For Available=True + send=0: ~57% of windows are 0, ~43% are positive.
  Across many similar prompts, your decisions should split roughly 57/43
  — call each window based on the cited slot / s30 / loc / send rates.
  send=0 is NOT evidence either way: it does not imply the user is high-
  activity, AND it does not imply zero. Let the cited probabilities decide.
  At the population level a=0 zero rate ≈ a=1/a=2 (all ~55-58%); the
  PER-USER `a=0` rate may diverge if this user behaves differently when
  unsent — treat that personalized signal as a real cue.""",
    zero_check_format_and_count=_ZERO_CHECK_FORMAT_4,
    zero_check_examples='''                              "decision=zero (P_zero=73%) — avail=True slot=2
                               48%, s30=15 in '1-80' 64%, avail=True loc=home
                               62% zero, a=0 57% zero"  (mean 57.75, 3/4>55 → +15)
                              "decision=positive (P_zero=17%) — avail=True slot=4
                               23%, s30=420 in '396-857' 14%, avail=True loc=work
                               38% zero, a=0 52% zero"  (mean 31.75, 3/4<45 → -15)''',
    anchor_lookup_body="""IF zero_check decision="positive":
                            Step 1 — Cite the slot's positive quantile bands
                              for a=0, e.g.
                                "slot=3 a=0 positive: p25=40, p50=80,
                                 p75=180, p95=480"
                            Step 2 — STOCHASTICALLY pick a band per the
                              right-skew weights (default; see value field
                              for signal-driven shifts):
                                25% <p25 | 25% p25-p50 | 25% p50-p75
                                | 20% p75-p95 | 5% >p95
                            Step 3 — Within the chosen band, pick a
                              representative point (CENTER neutral, LOW edge
                              weak, HIGH edge strong). State:
                                "sampled band=p25-p50, anchor=60 (mid)"
                            Step 4 — That anchor is the input to ctx/momentum
                              adjustments below.
                            IF decision="zero": "N/A — zero_check decided zero".""",
    phase_application='Write "N/A — no message to amplify".',
    value_default_hint="""                            Adjust the band-sampling weights from the default
                            25/25/25/20/5 based on signal strength:
                              • Strong upward (high s30 AND exercise/peak loc
                                AND/OR high-activity anchor match) → shift
                                ~15pp from <p50 to p75+.
                              • Weak (low s30 + sedentary loc) → shift ~10pp
                                from p50+ to <p25.
                              • Neutral → keep default weights.
                            send=0 itself is NOT a strong upward signal —
                            it's the natural baseline. NEVER force >p95
                            deterministically (~1% true bursts only).
                            Sampled band goes into value_band; after
                            adjustments value should sit inside (or one band
                            away from) that band.""",
    extra_rule="""  * Use the avail=True row of the zero-rate table; ignore avail=False.
  * Trajectory zero rate should approach ~57% — slot 3-5 with low-zero loc
    (e.g. work / activity) or high s30 should give POSITIVE more often than not.""",
)

# --- BRANCH BODY: NO_MSG_AVAIL_F (send=0 ∧ avail=False) ---------------------
_BRANCH_BODY_NO_MSG_AVAIL_F = dict(
    branch_intro=("The user is UNREACHABLE (avail = False); send is forced to 0. "
                  "This usually means commuting / driving / exercising / sleeping. "
                  "Predict the user's unavailable-context baseline — phase "
                  "multiplier does NOT apply."),
    calibration_block="""  For Available=False: ~35% of windows are 0, ~65% are POSITIVE — the
  MAJORITY of unavail windows have activity, because unavail correlates
  with movement (commute, walk, workout). Across many similar prompts,
  your decisions should split roughly 35/65 toward POSITIVE. DO NOT carry
  over the avail=True zero rate; use the avail=False row of the zero-rate
  baseline block.""",
    zero_check_format_and_count=_ZERO_CHECK_FORMAT_3,
    zero_check_examples='''                              "decision=zero (P_zero=60%) — avail=False slot=2
                               60%, s30=64%, avail=False loc=home 55% zero"
                               (mean 59.7, 2/3>55 → no boost)
                              "decision=positive (P_zero=5%) — avail=False slot=3
                               18%, s30=14%, avail=False loc=transit 24% zero"
                               (mean 18.7, 3/3<45 → -15, clipped to 5)''',
    anchor_lookup_body="""IF zero_check decision="positive":
                            Step 1 — Cite the user-level avail=False positive
                              quantile bands (per-(user,slot) avail=F cells
                              too thin to split by slot), AND cite the
                              per-slot UNAVAIL mean as a slot-peak signal:
                                "user avail=F positive: p25=60, p50=120,
                                 p75=280, p95=820; slot=3 unavail mean=548
                                 (slot peak — commute/exercise)"
                            Step 2 — STOCHASTICALLY pick a band per the
                              UNAVAIL-shifted weights (unavail skews higher
                              than avail=True; see value field for signal-
                              driven further shifts):
                                15% <p25 | 20% p25-p50 | 25% p50-p75
                                | 25% p75-p95 | 15% >p95
                            Step 3 — Within the chosen band, pick a
                              representative point (CENTER neutral; HIGH edge
                              if slot is a commute peak / slot mean > p75
                              of user; LOW edge if off-peak/sleep slot).
                              State:
                                "sampled band=p50-p75, anchor=200 (mid)"
                            Step 4 — That anchor is the input to ctx/momentum
                              adjustments below.
                            IF decision="zero": "N/A — zero_check decided zero".""",
    phase_application='Write "N/A — no message to amplify".',
    value_default_hint="""                            UNAVAIL default weights are pre-shifted right
                            (15/20/25/25/15). Further adjust based on signal:
                              • Slot is commute peak (slot mean > user p75) →
                                shift another ~10pp toward p75+.
                              • Off-peak/likely sleep (slot mean < user p25)
                                → shift ~10pp toward <p50.
                              • Otherwise → keep the UNAVAIL default weights.
                            Reach >p95 ONLY for clear commute/exercise
                            slot-peak combos (still <15% of mass).
                            Sampled band goes into value_band.""",
    extra_rule="""  * Use the avail=False row of the zero-rate table; ignore avail=True.
  * Trajectory zero rate should approach ~35% — MAJORITY of unavail
    windows are positive (commute/exercise).""",
)


SYSTEM_TEMPLATE_COT_MESSAGE             = _COMMON_HEADER_TPL.format(**_BRANCH_BODY_MESSAGE)
SYSTEM_TEMPLATE_COT_NO_MESSAGE_avail_true  = _COMMON_HEADER_TPL.format(**_BRANCH_BODY_NO_MSG_AVAIL_T)
SYSTEM_TEMPLATE_COT_NO_MESSAGE_avail_false = _COMMON_HEADER_TPL.format(**_BRANCH_BODY_NO_MSG_AVAIL_F)


# --- TASK blocks (appended to USER prompt) ----------------------------------
_TASK_BLOCK_MESSAGE = """## Task
Reason through the 6 reasoning fields (zero_check first, then the positive
chain) in the JSON schema above. Each field MUST reference SPECIFIC numbers
from the profile blocks above (e.g. "slot=3 a=2 positive → 218",
"honeymoon ×2.20", "steps30pre=85 in low bin", "slot=1 zero baseline 70%").

Remember: zero_check is a PROBABILISTIC call. Within avail=True the
split is ~56% zero / ~44% positive; your trajectory should approximate
that — DO NOT round every borderline case to zero.

Output ONLY the JSON object — no markdown fences, no extra text."""

_TASK_BLOCK_NO_MESSAGE_AVAIL_T = """## Task
No message was sent and the user is REACHABLE. First do zero_check using
the Zero-rate baseline (avail=True row). The split is ~57% zero / ~43%
positive — call each window using the cited slot / s30 / loc rates,
NOT a default. Across this trajectory your zero rate should APPROXIMATE
57%, not 90%+.

If decision="positive", reason through the BASELINE anchor chain:
  - anchor = per_slot_action_positive[slot][a=0]
  - widen / narrow using the avail=True positive quantiles for the slot
In `phase_application`, write "N/A — no message".

Output ONLY the JSON object — no markdown fences, no extra text."""

_TASK_BLOCK_NO_MESSAGE_AVAIL_F = """## Task
No message was sent and the user is UNREACHABLE (commute / exercise /
sleep). First do zero_check using the Zero-rate baseline (avail=False
row). The split is ~35% zero / ~65% positive — MOST unavail windows
are positive. Across this trajectory your zero rate should APPROXIMATE
35%, not 50%+. DO NOT carry over the avail=True zero rate.

If decision="positive", reason through the UNAVAIL anchor chain:
  - anchor = unavail_baseline[slot]   (per-slot mean when avail=False)
  - widen / narrow using the user-level avail=False positive quantiles
    (flat, no slot split — per-(user,slot) cells are too thin)
In `phase_application`, write "N/A — no message".

Output ONLY the JSON object — no markdown fences, no extra text."""


# --- USER profile-section assembly per branch -------------------------------
# Each branch sees a different combination of profile blocks. Sections that
# don't apply are dropped (not left empty) so the LLM's eye path is tight.
_BRANCH_MESSAGE        = "message"
_BRANCH_NO_MSG_AVAIL_T = "no_msg_avail_true"
_BRANCH_NO_MSG_AVAIL_F = "no_msg_avail_false"


def _cot_profile_sections(branch: str, persona: Dict, current_state: Dict,
                            action: int) -> List[str]:
    """Return the ordered list of profile-section strings for the given branch.

    MESSAGE / NO_MSG_AVAIL_T:
      use per-slot×action table + per-slot avail=True positive quantiles.
    NO_MSG_AVAIL_F:
      drop per-slot×action (no randomized action when unavail) and per-slot
      quantiles (data too thin); use unavail_baseline + flat user-level
      positive quantiles instead.
    """
    common_head = [
        "## Synthetic User Profile",
        _format_persona(persona, current_state),
        _format_zero_baseline(persona, current_state, action),
    ]
    common_tail = [
        _format_context_conditional(persona),
        _format_high_activity_anchors(persona),
    ]

    if branch == _BRANCH_NO_MSG_AVAIL_F:
        middle = [
            _format_unavail_baseline(persona),
            _format_unavail_quantiles(persona),
        ]
    else:
        # MESSAGE and NO_MSG_AVAIL_T share the same per-slot×action block;
        # the SYSTEM template tells the LLM which column (a=send vs a=0) to read.
        middle = [
            _format_avail_baseline(
                persona.get("steps10_avail_true_per_slot_action_mean_positive", {})),
            _format_positive_quantiles(persona, current_state),
        ]

    return common_head + middle + common_tail


def _cot_user_block(branch: str, persona: Dict, current_state: Dict, action: int,
                     episodic_history: List[Dict], task_block: str) -> str:
    """USER-side construction. `branch` selects which profile sections appear."""
    day = current_state.get("day", 1)
    profile_block = "\n\n".join(
        s for s in _cot_profile_sections(branch, persona, current_state, action) if s
    )

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


def _cot_message(persona: Dict, current_state: Dict, action: int,
                  episodic_history: List[Dict]) -> tuple[str, str]:
    user = _cot_user_block(_BRANCH_MESSAGE, persona, current_state, action,
                            episodic_history, _TASK_BLOCK_MESSAGE)
    return SYSTEM_TEMPLATE_COT_MESSAGE, user


def _cot_no_message_avail_true(persona: Dict, current_state: Dict, action: int,
                                 episodic_history: List[Dict]) -> tuple[str, str]:
    user = _cot_user_block(_BRANCH_NO_MSG_AVAIL_T, persona, current_state, action,
                            episodic_history, _TASK_BLOCK_NO_MESSAGE_AVAIL_T)
    return SYSTEM_TEMPLATE_COT_NO_MESSAGE_avail_true, user


def _cot_no_message_avail_false(persona: Dict, current_state: Dict, action: int,
                                  episodic_history: List[Dict]) -> tuple[str, str]:
    user = _cot_user_block(_BRANCH_NO_MSG_AVAIL_F, persona, current_state, action,
                            episodic_history, _TASK_BLOCK_NO_MESSAGE_AVAIL_F)
    return SYSTEM_TEMPLATE_COT_NO_MESSAGE_avail_false, user


def build_step_prompt_cot(persona: Dict,
                           current_state: Dict,
                           action: int,
                           episodic_history: List[Dict]) -> tuple[str, str]:
    """3-way dispatch over (action, avail):
      action > 0                  → MESSAGE
      action == 0 ∧ avail=True    → NO_MSG_AVAIL_T (MRT no-msg arm)
      action == 0 ∧ avail=False   → NO_MSG_AVAIL_F (unreachable baseline)
    """
    if action > 0:
        return _cot_message(persona, current_state, action, episodic_history)
    if bool(current_state.get("avail", True)):
        return _cot_no_message_avail_true(persona, current_state, action, episodic_history)
    return _cot_no_message_avail_false(persona, current_state, action, episodic_history)



