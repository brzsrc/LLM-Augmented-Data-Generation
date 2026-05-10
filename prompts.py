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

SYS_ADJUSTMENT = "You are a behavioral simulation module. Output ONLY a single integer from -50 to 100."

PROMPT_ADJUSTMENT = """{persona}

Current decision point:
- Day {study_day}, slot {slot} ({weekday_desc})
- Location: {location}
- Activity: {activity}
- Weather: {weather}, {temperature}C
- Steps in prior 30 min: {prior_30min_steps}
- Suggestion type: {action_desc}

Reference points (real averages from training data):
- This user, same slot + location: ~{user_bin_mean} steps (source: {user_bin_source})
- Population average at this slot + location: ~{pop_bin_mean} steps
- This user, overall: ~{user_overall_mean} steps

This user's recent steps at the same slot (newest first):
{slot_history}

Other recent behavior:
{recent_obs}

Statistical baseline prediction for this row: {base_steps} steps.

The baseline above is a stochastic sample from the population distribution for
this context; it does not know who this user is or what they have been doing
recently. Compare the baseline to the reference points above to see if it is
likely too high or too low for THIS user in THIS context. Use the slot history
and recent behavior to refine your judgment.

Output an integer percentage adjustment from -50 (the user will walk much fewer
steps than the baseline predicts) to +100 (much more steps). 0 means the
baseline is already right.
Output ONLY a single integer."""

