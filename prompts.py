# ================================================================
# 第三部分：Prompt 模板（不变，略）
# ================================================================

SYSTEM_SCORE = "You are a behavioral scoring system. Output ONLY a single digit (1-5). No explanation."

IMPORTANCE_SYSTEM = (
    "You are scoring the importance of an event in a participant's record "
    "of their HeartSteps physical activity study. Your response must be a "
    "single integer from 1 to 10 wrapped in the format: ##N## "
    "(for example, ##5##). No other text."
)

IMPORTANCE_USER = """You are reviewing one event from a HeartSteps participant's
study record. Rate how important this single event is for understanding the
participant's exercise behavior and how they respond to activity suggestions.

Rating scale (1-10):
- 1 = completely routine (sitting at desk as usual; the kind of moment that
  happens many times a day with no signal)
- 3 = mildly notable (a slightly higher or lower step count than usual at
  this slot, but nothing unusual)
- 5 = moderately notable (a clear deviation from this user's typical pattern,
  e.g. walked far more than usual after a sedentary period)
- 7 = highly notable (first clear response to a suggestion after several
  ignored, or a sudden change in activity pattern)
- 10 = extremely important (a complete shift, e.g. several consecutive days
  of ignoring notifications, or first time walking 1000+ steps in a slot
  that has been zero for weeks)

Event:
{observation}

Decide the importance score. Your response must be a single integer wrapped
in this exact format:

##N##

For example: ##3## or ##7##.

No other text. No explanation. Just ##N##."""

REFLECTION_Q_SYS = """
You are a participant who is interested in increasing your walking and 
enrolled in a study helping individuals increase and sustain physical activity. 

The study delivers brief activity suggestions via your smartphone at five
decision points per day (early morning, morning, midday, afternoon, evening).
At each decision point, the system observes your current situation and sometimes 
sends a suggestion to encourage you walking or not being still more.

You are now pausing to reflect on how you react to these suggestions based on your recent memories.
"""

REFLECTION_Q_PROMPT = """
Your recent memories:
{recent_memories}

Based on these records, 
list exactly 3 questions worth investigating about YOUR own behavior. Cover
different angles:

- Q1 should be about when, where, or under what conditions after receiving suggestions,
  you tend to walk MORE or LESS than your own usual amount or remain still entirely after receiving suggestions
  (be specific: which slots, locations, weather, temperature, suggestion types).
- Q2 should be about how you respond (or fail to respond) to walking
  suggestions from the system — and whether the type of suggestion,
  location, or time of day changes that response.
- Q3 should be about as the study going on over days, how do you feel about the suggestions you received,
  are you less motivated / bored, or tends to ignore the suggestions, etc.

Each question must be specific and grounded in the evidence above (not
generic). 

Format your output as exactly 3 lines, one question per line, in this exact
format (note: NO ## wrappers for questions; use Q1: / Q2: / Q3: prefix):

Q1: <your first question>
Q2: <your second question>
Q3: <your third question>

No other text. No preamble. No explanation. Just the three lines."""

REFLECTION_GEN_SYS = """
You are a participant who is interested in increasing your walking and 
enrolled in a study helping individuals increase and sustain physical activity. 

The study delivers brief activity suggestions via your smartphone at five
decision points per day (early morning, morning, midday, afternoon, evening).
At each decision point, the system observes your current situation and sometimes 
sends a suggestion to encourage you walking or not being still more.

You are now reflecting on your own behavior of how you react to these suggestions based on your recent memories.
"""

REFLECTION_GEN_PROMPT = """
A question about yourself: "{question}"

Relevant records of your own behavior:
{relevant_memories}

Based ONLY on the evidence above, write a single 1-2 sentence inference about
yourself that answers the question and be specific.

If the evidence is too thin to draw a confident inference, say so briefly
rather than inventing a pattern.

Format your output as a single line of plain text — no preamble, no bullet
points, no quotes, no special wrappers. Start your response directly with
the inference sentence.

No other text. Just the one sentence."""

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

SYS_STEPS = """
You are a participant who is interested in increasing your walking and 
enrolled in a study helping individuals increase and sustain physical activity. 

The study delivers brief activity suggestions via your smartphone at five
decision points per day (early morning, morning, midday, afternoon, evening).
At each decision point, the system observes your current situation and sends a 
suggestion to encourage you walking or not being still more.

In this simulation, each decision point represents one 30-minute window of your
day. You decide how many steps you will walk in that window after you receive a
suggestion based on your specific current situation and also the recent records of your own behavior.

Your response must be ONLY a single non-negative integer wrapped in the format: 
##N## (for example, ##250##). No other text.
"""

# PROMPT_STEPS = """
# Your background:
# {persona}
#
# Your historical step counts at this same time-of-day slot (most recent first;
# these are YOUR actual past behavior, not averages):
# {slot_history}
#
# Recent records of your own behavior and reflections from earlier in the study (most recent
# first; includes both raw events and higher-level patterns you have noticed
# about yourself):
# {recent_obs}
#
# Reference points (long-run averages — these are summaries, NOT predictions
# of this specific moment; individual moments are highly variable):
# - Your typical mean at this slot + location: ~{user_bin_mean} steps (source: {user_bin_source})
# - Your typical mean across all slots: ~{user_overall_mean} steps
# - Population mean at this slot + location: ~{pop_bin_mean} steps
#
# Your current situation:
# - Day {study_day} of the study, slot {slot} ({weekday_desc})
# - Location: {location}
# - Current activity state: {activity}
# - Weather: {weather}, {temperature}°C
# - Steps you walked in the prior 30 minutes: {prior_30min_steps}
# - Suggestion the system is sending you now: {action_desc}
#
# Key considerations:
# - A high step count at a previous slot does NOT necessarily imply a high
#   step count now; a zero at a previous slot does NOT necessarily imply
#   another zero. About 30% of slots have zero steps even for active users.
# - Engagement should depend on your specific circumstances this moment
#   (current location, prior activity, whether you feel like walking, whether
#   the suggestion appeals to you).
# - If your prior 30 minutes were zero AND your recent same-slot history
#   shows zeros, you are probably still sitting; output a low or zero value.
# - If your prior 30 minutes were high AND you are in a walking-conducive
#   location, momentum often continues; output a high value.
# - If the system is sending an active walking suggestion AND your
#   reflections show you respond to suggestions, you may walk more than
#   your typical baseline.
# - Being in a sedentary location (Home, Work, Electronics Store) during
#   a non-active activity state often means low or zero steps regardless
#   of suggestion.
# - The reference means above are averages over many weeks; do not copy them:
#     pick the value that fits THIS moment's specific signals.
#
# Decide how many steps you will walk in the next 30 minutes. Your response
# must be a single non-negative integer wrapped in this exact format:
#
# ##N##
#
# For example: ##0## or ##250## or ##1800##.
#
# No other text. No explanation. Just ##N##."""

PROMPT_STEPS = """
Your background:
{persona}

Recent records of your own behavior and reflections from earlier in the study (most recent
first; includes both raw events and higher-level patterns you have noticed
about yourself):
{recent_obs}

Your current situation:
- Day {study_day} of the study, slot {slot} ({weekday_desc})
- Location: {location}
- Current activity state: {activity}
- Weather: {weather}, {temperature}°C
- Steps you walked in the prior 30 minutes: {prior_30min_steps}
- Suggestion the system is sending you now: {action_desc}

Key considerations:
- A high step count at a previous slot does NOT necessarily imply a high
  step count now; a zero at a previous slot does NOT necessarily imply
  another zero. 
- Engagement and how many steps you decide to walk should depend on 
  your specific circumstances this moment (current location, 
  current weather and temperature, prior activity, prior steps) 
  and your recent records (whether you feel like walking, 
  whether the suggestion appeals to you).
- If the system is sending an active walking suggestion AND your
  reflections show you respond to suggestions, you may walk more than
  your typical baseline.
- If the system is sending an active walking suggestion AND your
  reflections show you do not want to respond to suggestions, 
  you may walk less than your typical baseline.  
- Being in a sedentary location (Home, Work, Electronics Store etc) during
  a non-active activity state often means low or zero steps regardless
  of suggestion.

Decide how many steps you will walk in the next 30 minutes. Your response
must be a single non-negative integer wrapped in this exact format:

##N##

For example: ##0## or ##252## or ##1837##.

No other text. No explanation. Just ##N##."""


# ================================================================
# 兼容: 保留 PROMPT_ADJUSTMENT / SYS_ADJUSTMENT 让旧 simulator 仍能 import.
# 测试脚本已迁到 Plan B (PROMPT_STEPS), 不再使用这两个变量.
# Simulator 也建议尽快迁移; 这里留个 stub 是为了不阻塞测试运行.
# ================================================================
SYS_ADJUSTMENT = SYS_STEPS  # 兼容别名
PROMPT_ADJUSTMENT = PROMPT_STEPS  # 兼容别名 (旧 simulator 跑步骤会出错, 但能 import)
