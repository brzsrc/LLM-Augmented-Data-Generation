# ================================================================
# 第三部分：Prompt 模板（不变，略）
# ================================================================

SYSTEM_SCORE = "You are a behavioral scoring system. Output ONLY a single digit (1-5). No explanation."
IMPORTANCE_SYSTEM = "You are a behavioral analysis system. Rate importance from 1-10. Output ONLY the number."
IMPORTANCE_USER = """Rate the importance of this event for understanding the participant's exercise behavior and notification response patterns.
1 = completely routine (sitting at desk as usual)
10 = extremely important (first time responding to a notification, or ignoring notifications for many days straight)

Event: {observation}

Output ONLY a single number (1-10)."""

REFLECTION_Q_PROMPT = """Below are the participant's recent behavior records:

{recent_memories}

List exactly 3 questions worth investigating to understand:
- Whether the participant's response to activity suggestions is changing
- The participant's daily behavioral patterns
- What factors may influence whether they respond to suggestions

List 3 questions, one per line. Be specific and evidence-based."""

REFLECTION_GEN_PROMPT = """Question: "{question}"

Relevant evidence:
{relevant_memories}

Write a 1-2 sentence high-level inference based on the evidence above. Be specific and cite patterns you observe."""

MOTIVATION_PROMPT = """## Participant Background
{seed_memory}

## Recent Reflections
{recent_reflections}

## Recent Behavior Records (newest first)
{recent_observations}

## Current State
Study day: {study_day}, time slot {slot}
Steps in prior 30min: {pre30_steps}
Suggestion sent: {action_desc}
Steps in following 30min: {post_steps}

## Task
On a 1-5 scale, rate this participant's current INTRINSIC MOTIVATION — the self-directed drive to walk, independent of external prompts.

1 = Very low. Participant only walks when prompted, or shows declining activity over days.
2 = Low. Mostly sedentary without prompts, occasional activity.
3 = Moderate. Some self-initiated walking but inconsistent.
4 = High. Regularly walks without prompts, stable or increasing trend.
5 = Very high. Consistently active regardless of suggestions, increasing self-initiated activity.

IMPORTANT: Compare steps at prompted vs unprompted decision points. If participant walks 200+ steps WITHOUT a suggestion, motivation is likely HIGH.

Output ONLY a single digit (1-5)."""

HABIT_PROMPT = """## Participant Background
{seed_memory}

## Recent Behavior Records (newest first)
{recent_observations}

## Behavioral Regularity Indicators
Steps in past 7 days by slot: {slot_pattern}
Day-to-day consistency: {consistency_desc}

## Current State
Study day: {study_day}

## Task
On a 1-5 scale, rate this participant's current HABIT STRENGTH — how automatic and regular their walking behavior has become.

1 = No habit. Walking is entirely effortful and irregular.
2 = Weak. Occasionally walks at similar times but mostly inconsistent.
3 = Forming. Some regular patterns emerging (e.g., walks at same slot most days).
4 = Moderate. Consistent patterns across multiple days, walks at predictable times.
5 = Strong. Highly regular, walks at same times daily regardless of context.

Output ONLY a single digit (1-5)."""

RECEPTIVITY_PROMPT = """## Participant Background
{seed_memory}

## Recent Reflections
{recent_reflections}

## Recent Behavior Records (newest first)
{recent_observations}

## Current State
Study day: {study_day}, time slot {slot}
Current dosage: {dosage:.2f}

## Task
On a 1-5 scale, how likely is this participant to open and respond to an activity suggestion RIGHT NOW by walking in the next 30 minutes?

1 = Very unlikely. Participant has a pattern of ignoring suggestions, or recent response rate is very low.
2 = Unlikely. Participant occasionally responds but mostly ignores.
3 = Uncertain. Not enough information, or response pattern is inconsistent.
4 = Likely. Participant has been responding positively to suggestions recently.
5 = Very likely. Participant actively responds and shows positive attitude.

Output ONLY a single digit (1-5)."""

SYS_STEPS = (
    "You are a behavioral simulation module predicting the actual step count "
    "of a specific user in the next 30 minutes. Different users behave very "
    "differently — anchor on this user's own history, not on the population. "
    "Output ONLY a non-negative integer (step count)."
)

PROMPT_STEPS = """{persona}

Current decision point:
- Day {study_day}, slot {slot} ({weekday_desc})
- Location: {location}
- Activity: {activity}
- Weather: {weather}, {temperature}C
- Steps in prior 30 min: {prior_30min_steps}
- Suggestion type: {action_desc}

Reference points for THIS user (from training data):
- This user at this slot + location: ~{user_bin_mean} steps  ← STRONGEST anchor (source: {user_bin_source})
- This user overall (averaged across all slots): ~{user_overall_mean} steps
- Population average at this slot + location: ~{pop_bin_mean} steps  ← only use as backup

This user's actual steps at the same slot in recent days (newest first):
{slot_history}

Recent observations and reflections:
{recent_obs}

Task: Predict how many steps THIS specific user will walk in the next 30 minutes.

Signal priorities:
1. The slot history above is the single most informative signal — it shows
   what THIS user actually does at THIS time of day. Zero in slot history
   often means zero now; high values often persist.
2. Prior 30-min steps ({prior_30min_steps}): high recent activity tends to
   continue for one more slot; zero often persists into the next slot.
3. Suggestion type: an active walking suggestion can boost a responsive user
   by 50-200 steps; sedentary/stand-up suggestions have smaller effects;
   judge responsiveness from the reflections and observations above.
4. The user_bin_mean is a stable baseline for THIS user in THIS context —
   most predictions should land within ±50% of it, but slot history and
   prior steps can pull you far above or below.

Do NOT default to the population average. Different users at the same
slot+location can differ by 10× in steps. Stay anchored to this user.

Output ONLY a single non-negative integer (predicted step count)."""


# ================================================================
# 兼容: 保留 PROMPT_ADJUSTMENT / SYS_ADJUSTMENT 让旧 simulator 仍能 import.
# 测试脚本已迁到 Plan B (PROMPT_STEPS), 不再使用这两个变量.
# Simulator 也建议尽快迁移; 这里留个 stub 是为了不阻塞测试运行.
# ================================================================
SYS_ADJUSTMENT = SYS_STEPS  # 兼容别名
PROMPT_ADJUSTMENT = PROMPT_STEPS  # 兼容别名 (旧 simulator 跑步骤会出错, 但能 import)
