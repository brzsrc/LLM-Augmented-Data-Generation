"""
Two-layer behavioral-routine extraction (SensorPersona-style) for HeartSteps-like data.

LAYER 1 (intra-episode, deterministic / pandas):
    Aggregate each user-DAY into a compact "event card" with timestamp.
    A user-day is the natural episode window (analogous to SensorPersona's
    fixed-length window T). Numbers are computed here, NOT by the LLM.

LAYER 2 (inter-episode, LLM):
    Feed ALL of a user's day-cards to an LLM and ask it to promote ONLY
    patterns that recur across multiple days into a routine_profile, each
    grounded in supporting dates (traceable evidence E_p).

Usage:
    python persona_extract.py --csv data_cleaned.csv --out_dir ./out
    # add --call_llm to actually hit the Anthropic API (needs ANTHROPIC_API_KEY)
"""

import argparse, json, os
import numpy as np
import pandas as pd

# slot -> rough time-of-day label (HeartSteps: 5 user-set times ~2.5h apart)
SLOT_LABEL = {1: "morning_commute", 2: "midday", 3: "midafternoon",
              4: "evening_commute", 5: "post_dinner"}


# ----------------------------------------------------------------------------
# LAYER 1 — intra-episode: build one event card per user-day
# ----------------------------------------------------------------------------
def build_day_cards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = pd.to_datetime(df["decision_datetime"])
    df["decision_date"] = pd.to_datetime(df["decision_date"]).dt.date
    df["weekday"] = df["dt"].dt.weekday              # 0=Mon
    df["is_weekend"] = df["weekday"] >= 5
    df = df.sort_values(["uid", "dt"])

    rows = []
    for (uid, date), g in df.groupby(["uid", "decision_date"]):
        g = g.sort_values("decision_slot")
        # per-slot mini-trace within the day (the "intra-episode" content)
        slots = []
        for _, r in g.iterrows():
            slots.append({
                "slot": int(r["decision_slot"]),
                "tod": SLOT_LABEL.get(int(r["decision_slot"]), "unknown"),
                "loc": r["loc"],
                "steps30pre": _num(r["steps30pre"]),
                "sent": int(r["send"]) if pd.notna(r["send"]) else 0,
                "response": r["response"],
            })
        steps_pre = g["steps30pre"].dropna()
        rows.append({
            "uid": int(uid),
            "date": str(date),
            "weekday": int(g["weekday"].iloc[0]),
            "is_weekend": bool(g["is_weekend"].iloc[0]),
            "n_slots": int(len(g)),
            # day-level summary numbers (computed by pandas, not the LLM)
            "day_steps30pre_sum": _num(steps_pre.sum()),
            "day_steps30pre_mean": _num(steps_pre.mean()),
            "active_slots": [int(s["slot"]) for s in slots if (s["steps30pre"] or 0) >= 250],
            "dominant_loc": g["loc"].mode().iloc[0] if not g["loc"].mode().empty else None,
            "slots": slots,
        })
    return pd.DataFrame(rows)


def _num(x):
    if pd.isna(x):
        return None
    return round(float(x), 1)


# ----------------------------------------------------------------------------
# Compact textual rendering of one user's day-cards for the LLM prompt
# ----------------------------------------------------------------------------
def render_user_cards(cards_u: pd.DataFrame) -> str:
    lines = []
    for _, c in cards_u.sort_values("date").iterrows():
        wk = "WKND" if c["is_weekend"] else "WKDY"
        slot_str = " | ".join(
            f"s{s['slot']}({s['tod']},{s['loc']},pre={int(s['steps30pre']) if s['steps30pre'] is not None else 0})"
            for s in c["slots"]
        )
        lines.append(
            f"{c['date']} [{wk}] daySum={int(c['day_steps30pre_sum'] or 0)} "
            f"domLoc={c['dominant_loc']} :: {slot_str}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# LAYER 2 — inter-episode: the LLM prompt
# ----------------------------------------------------------------------------
ROUTINE_SCHEMA = {
    "uid": "int",
    "n_days_observed": "int",
    "routine_profile": {
        "active_time_type": "one of: morning_type | evening_type | midday_type | bimodal | flat | unclear",
        "weekday_weekend_difference": "short phrase, or 'none_detected'",
        "location_activity_tendency": "short phrase: where this user tends to be most active",
        "baseline_activity_level": "one of: sedentary | low | moderate | high",
    },
    "personas": [
        {
            "dimension": "physical | psychosocial",
            "statement": "one concise sentence describing a STABLE recurring pattern",
            "supporting_dates": ["YYYY-MM-DD", "..."],
            "n_occurrences": "int (>=3 to qualify as physical persona)",
            "confidence": "low | medium | high",
        }
    ],
    "notes": "any caveats (small sample, conflicting evidence, etc.)",
}

PROMPT_TEMPLATE = """You are a behavioral-pattern analyst. You will receive ALL daily activity \
records for ONE user from a physical-activity study. Each line is one DAY (an episode):
  DATE [WKDY/WKND] daySum=<sum of pre-decision 30-min step counts that day> domLoc=<most common location> :: per-slot traces
Each slot trace is s<slot>(<time_of_day>,<location>,pre=<steps in 30 min before that decision point>).

TASK: infer this user's STABLE behavioral routine — the "inter-episode" persona — \
by finding patterns that RECUR ACROSS MULTIPLE DAYS. This is aggregation across days, \
not description of any single day.

PROMOTION RULE (strict, from persona theory):
- A "physical" persona (routine, mobility, activity-timing pattern) may be stated ONLY if \
it recurs on at least 3 distinct days. List those dates as supporting_dates and set \
n_occurrences accordingly.
- A "psychosocial" persona (preference / tendency) may use weaker evidence but still needs \
>=2 supporting days, and must be marked confidence=low if thin.
- If a candidate pattern does NOT recur, DO NOT include it. Prefer fewer, well-supported \
personas over many speculative ones.
- Every persona MUST carry its supporting_dates (traceable evidence). No dates -> drop it.

Also fill routine_profile (active_time_type, weekday_weekend_difference, \
location_activity_tendency, baseline_activity_level). Use 'unclear'/'none_detected' when \
evidence is insufficient — do NOT guess.

Output ONLY valid JSON matching this schema (no markdown, no prose):
{schema}

USER {uid} — {n_days} days of records:
{records}
"""


def build_prompt(uid: int, cards_u: pd.DataFrame) -> str:
    return PROMPT_TEMPLATE.format(
        schema=json.dumps(ROUTINE_SCHEMA, ensure_ascii=False, indent=2),
        uid=uid,
        n_days=len(cards_u),
        records=render_user_cards(cards_u),
    )


# ----------------------------------------------------------------------------
# Optional: actually call the Anthropic API for inter-episode reasoning
# ----------------------------------------------------------------------------
def call_llm(prompt: str, model: str = "claude-sonnet-4-5") -> dict:
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model, max_tokens=2000, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_cleaned.csv")
    ap.add_argument("--out_dir", default="./out")
    ap.add_argument("--call_llm", action="store_true",
                    help="actually call the Anthropic API (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--train_only", action="store_true",
                    help="use only first 70%% of each user's days (leakage guard)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # ---- LAYER 1
    cards = build_day_cards(df)

    if args.train_only:  # keep early days only, per user (no future leakage)
        keep = []
        for uid, g in cards.groupby("uid"):
            g = g.sort_values("date")
            keep.append(g.head(int(np.ceil(len(g) * 0.7))))
        cards = pd.concat(keep, ignore_index=True)

    cards.to_json(os.path.join(args.out_dir, "layer1_day_cards.jsonl"),
                  orient="records", lines=True, force_ascii=False)
    print(f"[Layer 1] {len(cards)} day-cards across {cards.uid.nunique()} users "
          f"-> {args.out_dir}/layer1_day_cards.jsonl")

    # ---- LAYER 2 (prompts; optionally call the model)
    prompts, personas = {}, {}
    for uid, g in cards.groupby("uid"):
        p = build_prompt(int(uid), g)
        prompts[int(uid)] = p
        if args.call_llm:
            try:
                personas[int(uid)] = call_llm(p, args.model)
                print(f"[Layer 2] uid={uid}: {len(personas[int(uid)].get('personas', []))} personas")
            except Exception as e:
                personas[int(uid)] = {"error": str(e)}
                print(f"[Layer 2] uid={uid}: ERROR {e}")

    with open(os.path.join(args.out_dir, "layer2_prompts.json"), "w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"[Layer 2] wrote {len(prompts)} prompts -> {args.out_dir}/layer2_prompts.json")

    if args.call_llm:
        with open(os.path.join(args.out_dir, "layer2_personas.json"), "w") as f:
            json.dump(personas, f, ensure_ascii=False, indent=2)
        print(f"[Layer 2] wrote personas -> {args.out_dir}/layer2_personas.json")


if __name__ == "__main__":
    main()
