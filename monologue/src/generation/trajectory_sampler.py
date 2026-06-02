"""Per-synthetic-persona trajectory generator.

Decision order (mirrors the real HeartSteps protocol):
  1. Sample context (weather, temp, loc, steps30pre)
  2. **Predict avail** from state using the persona's avail_by_slot + avail_by_loc
  3. If avail=True  → randomize action ∈ {0, 1, 2} (MRT)
     If avail=False → action FORCED to 0 (HeartSteps structural rule)
  4. Call LLM to predict steps10
     - avail=False prompt is simpler: just "predict context-conditional steps10"
     - avail=True  prompt: full persona + per_slot_action_steps10 signal

Outputs the SAME schema as data_gen.csv (uid, study_day, weekday, hr, slot,
weather, temp, loc, avail, steps30pre, send, resp, steps10), plus source_uid /
variant_type / archetype columns.
"""
from __future__ import annotations
import os
from typing import Dict, List
import numpy as np
import pandas as pd

from src import config as cfg
from src.generation.prompt_builder import build_step_prompt


def _predict_avail(persona: Dict, slot: int, loc_str: str,
                    rng: np.random.Generator) -> bool:
    """Compute P(avail=True | slot, loc) by combining the two marginal predictors.

    Geometric mean of the per-slot and per-loc rates — gives reasonable
    smoothing without needing a full joint table.
    """
    p_slot = persona.get("avail_by_slot", {}).get(slot)
    p_loc = persona.get("avail_by_loc", {}).get(loc_str)
    default = persona.get("avail_rate", 0.85)
    candidates = [p for p in (p_slot, p_loc) if p is not None]
    p = float(np.prod(candidates) ** (1.0 / len(candidates))) if candidates else default
    p = max(0.05, min(0.99, p))
    return bool(rng.random() < p)


def _sample_from_dist(dist: Dict[str, float], rng: np.random.Generator,
                       fallback: str) -> str:
    """Categorical sample from a {str: prob} dict; returns the str key."""
    if not dist:
        return fallback
    keys = list(dist.keys())
    weights = np.array(list(dist.values()), dtype=float)
    if weights.sum() <= 0:
        return str(rng.choice(keys))
    weights = weights / weights.sum()
    return str(rng.choice(keys, p=weights))


def _bootstrap_context(persona: Dict, slot: int, weekday: int,
                        prev_ctx: Dict | None,
                        rng: np.random.Generator) -> Dict:
    """Sample weather/temp/loc/steps30pre — all driven by persona stats.

    - slot 1: marginal dist (weather_dist / temp_dist / weekday|weekend_loc_dist)
    - slot >1: transition from prev_ctx (weather_transition / temp_transition /
                 weekday|weekend_loc_transition)
    - loc uses weekday vs weekend variant.
    - steps30pre: hurdle-lognormal via persona's per-slot mean + zero_pct + σ_log.
    """
    is_weekend = weekday >= 5
    weather_dist  = persona.get("weather_dist") or {}
    weather_trans = persona.get("weather_transition") or {}
    temp_dist     = persona.get("temp_dist") or {}
    temp_trans    = persona.get("temp_transition") or {}
    if is_weekend:
        loc_dist  = persona.get("weekend_loc_dist") or persona.get("weekday_loc_dist") or {}
        loc_trans = persona.get("weekend_loc_transition") or persona.get("weekday_loc_transition") or {}
    else:
        loc_dist  = persona.get("weekday_loc_dist") or {}
        loc_trans = persona.get("weekday_loc_transition") or {}

    if slot == 1 or not prev_ctx:
        weather_str = _sample_from_dist(weather_dist, rng, "clear")
        temp_str    = _sample_from_dist(temp_dist,    rng, "mild")
        loc_str     = _sample_from_dist(loc_dist,     rng, "home")
    else:
        pw, pt, pl = prev_ctx["weather"], prev_ctx["temp"], prev_ctx["loc"]
        weather_str = _sample_from_dist(weather_trans.get(pw, {}), rng, pw)
        temp_str    = _sample_from_dist(temp_trans.get(pt, {}),    rng, pt)
        loc_str     = _sample_from_dist(loc_trans.get(pl, {}),     rng, pl)

    # No encoding: synth CSV keeps the same string categoricals as data_gen.csv.
    # data_loader.add_derived_features (called from pipeline before validation /
    # ablation) encodes both real and synth to ints in memory.
    steps30pre = _sample_steps30pre(persona, slot, rng)

    return {
        "weather": weather_str, "temp": temp_str, "loc": loc_str,
        "steps30pre": steps30pre,
    }


def _sample_steps30pre(persona: Dict, slot: int,
                        rng: np.random.Generator) -> int:
    """Hurdle-lognormal sampler for steps30pre.

    Stage 1: Bernoulli(p_zero) where p_zero = per_slot_steps30pre_zero_pct[slot]
    Stage 2: if non-zero, draw from Lognormal(μ_log, σ_log) with:
               σ_log = persona's MLE-fit shape (or SIGMA_LOG_DEFAULT fallback)
               μ_log = ln(μ_pos) - σ²/2     ← mean-preserving inversion
               μ_pos = per_slot_mean / (1 - p_zero)  ← non-zero conditional mean

    Picks lognormal over gamma based on empirical AIC on real data_gen.csv:
      - All 5 slots: lognormal wins (ΔAIC 30-200)
      - 76% of users: lognormal wins
      - Heavy tail accommodates real exercise bursts; gamma truncates too soon.
    """
    p_zero_map = persona.get("steps30pre_per_slot_zero_pct") or {}
    mean_map   = persona.get("steps30pre_per_slot_mean") or {}
    p_zero = float(p_zero_map.get(slot, p_zero_map.get(str(slot), 0.3)))
    mean   = float(mean_map.get(slot, mean_map.get(str(slot), 250.0)))

    # Stage 1: zero gate (clip away the degenerate p_zero=1 case)
    p_zero = min(p_zero, 0.99)
    if rng.random() < p_zero:
        return 0

    # Stage 2: positive draw from lognormal preserving the per-slot mean
    if mean <= 0:
        return 0
    sigma = float(persona.get("steps30pre_sigma_log") or cfg.SIGMA_LOG_DEFAULT)
    mu_pos = mean / (1.0 - p_zero)
    mu_log = np.log(mu_pos) - sigma * sigma / 2.0
    draw = float(rng.lognormal(mu_log, sigma))
    return int(round(np.clip(draw, 0.0, cfg.MAX_STEPS30PRE)))


def generate_one_trajectory(persona: Dict, llm, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_days = int(persona.get("n_days", 30))
    slot_1_hour = persona.get("slot_1_hour", 12) or 12

    records = []
    # dosage is NOT tracked here — empirically corr(dosage, steps10) ≈ -0.014
    # (0/37 users with |corr|>0.2), so it's noise in the prompt. The DDQN-side
    # dosage feature is recomputed by data_loader.add_derived_features after
    # generation, identically to how it's added to real data.
    episodic_history: List[Dict] = []

    for day in range(1, n_days + 1):
        weekday = (day - 1) % 7
        prev_ctx = None   # reset each day; transitions are within-day only
        for slot in range(1, 6):
            hour = (slot_1_hour + cfg.SLOT_HOUR_OFFSET[slot]) % 24
            hour_jitter = float(rng.uniform(-0.5, 0.5))
            actual_hour = (hour + hour_jitter) % 24

            ctx = _bootstrap_context(persona, slot, weekday, prev_ctx, rng)
            prev_ctx = ctx
            loc_str = ctx["loc"]

            # ---- 1) Predict avail from state (angle 2 + 3) ----
            avail = _predict_avail(persona, slot, loc_str, rng)

            # ---- 2) Action: MRT randomize if avail; forced 0 if not ----
            if avail:
                action = int(rng.choice([0, 1, 2]))
            else:
                action = 0   # structural HeartSteps rule

            #ctx: Dict with weather/temp/loc/steps30pre for this decision point — sampled
            current_state = {
                "day": day, "slot": slot, "hour": round(actual_hour, 1),
                "weekday": weekday,
                "avail": avail,        # surfaced to LLM (angle 6)
                **ctx,
            }

            # ---- 3) LLM predicts steps10 (simpler prompt when avail=False) ----
            sys_p, usr_p = build_step_prompt(persona, current_state, action,
                                              episodic_history)
            steps10 = llm.judge_steps(sys_p, usr_p)
            try:
                steps10 = max(0, min(10000, int(steps10)))
            except Exception:
                steps10 = 0

            # NOTE: no `resp` column in synth output. In real data resp is the
            # user's behavioural response to the sent message. In synth it
            # would be a deterministic heuristic on (action, steps10) — adding
            # nothing and risking confusion downstream. DDQN/OPE don't use it
            # (resp is in FORBIDDEN_IN_STATE). If a synth resp is ever needed
            # for compatibility, derive it from action + steps10 at that point.
            rec = {
                "uid":         persona["synth_uid"],
                "source_uid":  persona["source_uid"],
                "variant_type": persona["variant_type"],
                "archetype":   persona["archetype"],
                "study_day":   day,
                "weekday":     weekday,
                "hr":          int(round(actual_hour)),
                "slot":        slot,
                "weather":     ctx["weather"],
                "temp":        ctx["temp"],
                "loc":         ctx["loc"],
                "avail":       avail,
                "steps30pre":  int(ctx["steps30pre"]),
                "send":        action,
                "steps10":     int(steps10),
            }
            records.append(rec)

            episodic_history.append({"day": day, "slot": slot, "action": action,
                                      "steps10": steps10, "avail": avail})

    return pd.DataFrame(records)


def generate_all(personas: List[Dict], llm,
                 out_dir: str, seed: int = 42) -> pd.DataFrame:
    """Generate trajectories for every synthetic persona.
    Persona-driven: no real_df needed (steps30pre / context / availability all
    sampled from persona stats — see _bootstrap_context and _sample_steps30pre).
    """
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for i, p in enumerate(personas):
        df_one = generate_one_trajectory(p, llm, seed=seed + i)
        all_rows.append(df_one)
        if (i + 1) % 10 == 0 or (i + 1) == len(personas):
            print(f"  [generate] {i+1}/{len(personas)} personas "
                  f"({sum(len(d) for d in all_rows)} rows so far)")
    combined = pd.concat(all_rows, ignore_index=True)
    out_csv = os.path.join(out_dir, "synthetic_data.csv")
    combined.to_csv(out_csv, index=False)
    print(f"[generate] Saved {len(combined)} synth rows -> {out_csv}")
    return combined


# ============================================================================
# Batched / parallel variant for vLLM backends
# ============================================================================
def _batch_judge_steps(llm, batch: List[Dict]) -> List[int]:
    """Dispatch a batch of prompts to llm.batch_steps if available, else fall
    back to per-prompt llm.judge_steps (so StubLLM / any single-call backend
    still works without modification)."""
    if hasattr(llm, "batch_steps"):
        return llm.batch_steps(batch)
    return [llm.judge_steps(p["system"], p["user"]) for p in batch]


def generate_all_vllm(personas: List[Dict], llm,
                       out_dir: str, seed: int = 42) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    # ----- Per-persona state -----
    states = [
        {
            "persona":          p,
            "rng":              np.random.default_rng(seed + i),
            "records":          [],
            "episodic_history": [],
            "prev_ctx":         None,
            "n_days":           int(p.get("n_days", 30)),
            "slot_1_hour":      p.get("slot_1_hour", 12) or 12,
        }
        for i, p in enumerate(personas)
    ]
    max_n_days = max(s["n_days"] for s in states) if states else 0
    print(f"[gen_vllm] {len(states)} personas, max_n_days={max_n_days}, "
          f"max prompts/batch={len(states)}")

    total_prompts = 0
    for day in range(1, max_n_days + 1):
        weekday = (day - 1) % 7
        # Reset within-day prev_ctx for every persona at day start.
        for s in states:
            s["prev_ctx"] = None

        for slot in range(1, 6):
            # ---- Build prompts for all personas still active at this day ----
            batch: List[Dict] = []
            for idx, s in enumerate(states):
                if day > s["n_days"]:
                    continue
                p = s["persona"]
                rng = s["rng"]

                hour = (s["slot_1_hour"] + cfg.SLOT_HOUR_OFFSET[slot]) % 24
                hour_jitter = float(rng.uniform(-0.5, 0.5))
                actual_hour = (hour + hour_jitter) % 24

                ctx = _bootstrap_context(p, slot, weekday, s["prev_ctx"], rng)
                s["prev_ctx"] = ctx
                loc_str = ctx["loc"]

                avail = _predict_avail(p, slot, loc_str, rng)
                action = int(rng.choice([0, 1, 2])) if avail else 0

                current_state = {
                    "day": day, "slot": slot, "hour": round(actual_hour, 1),
                    "weekday": weekday,
                    "avail": avail,
                    **ctx,
                }
                sys_p, usr_p = build_step_prompt(
                    p, current_state, action, s["episodic_history"])

                batch.append({
                    "system":       sys_p,
                    "user":         usr_p,
                    "_idx":         idx,
                    "_ctx":         ctx,
                    "_action":      action,
                    "_avail":       avail,
                    "_actual_hour": actual_hour,
                })

            if not batch:
                continue

            # ---- One vLLM call for the whole batch ----
            steps10_list = _batch_judge_steps(llm, batch)
            total_prompts += len(batch)

            # ---- Distribute results back into each persona's state ----
            for item, steps10 in zip(batch, steps10_list):
                s = states[item["_idx"]]
                p = s["persona"]
                ctx = item["_ctx"]
                action = item["_action"]
                avail = item["_avail"]
                actual_hour = item["_actual_hour"]

                try:
                    steps10 = max(0, min(10000, int(steps10)))
                except Exception:
                    steps10 = 0

                s["records"].append({
                    "uid":          p["synth_uid"],
                    "source_uid":   p["source_uid"],
                    "variant_type": p["variant_type"],
                    "archetype":    p["archetype"],
                    "study_day":    day,
                    "weekday":      weekday,
                    "hr":           int(round(actual_hour)),
                    "slot":         slot,
                    "weather":      ctx["weather"],
                    "temp":         ctx["temp"],
                    "loc":          ctx["loc"],
                    "avail":        avail,
                    "steps30pre":   int(ctx["steps30pre"]),
                    "send":         action,
                    "steps10":      int(steps10),
                })
                s["episodic_history"].append({
                    "day":     day,
                    "slot":    slot,
                    "action":  action,
                    "steps10": steps10,
                    "avail":   avail,
                })

        if day % 5 == 0 or day == max_n_days:
            n_active = sum(1 for s in states if day <= s["n_days"])
            rows_so_far = sum(len(s["records"]) for s in states)
            print(f"  [gen_vllm] day {day}/{max_n_days}: "
                  f"{n_active} active personas, "
                  f"{total_prompts:,} prompts done, "
                  f"{rows_so_far:,} rows")

    # ----- Combine + save -----
    all_rows = []
    for s in states:
        all_rows.extend(s["records"])
    combined = pd.DataFrame(all_rows)
    out_csv = os.path.join(out_dir, "synthetic_data.csv")
    combined.to_csv(out_csv, index=False)
    print(f"[gen_vllm] Saved {len(combined)} synth rows ({total_prompts:,} LLM calls) "
          f"-> {out_csv}")
    return combined
