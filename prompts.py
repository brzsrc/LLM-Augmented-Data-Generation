IMPORTANCE_SYSTEM = """
You are scoring the importance of an event in a participant's record 
of their HeartSteps physical activity study. Your response must be a 
single integer from 1 to 10 wrapped in the format: ##N## 
(for example, ##5##). No other text.
"""

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

IMPORTANCE_REASSESS_SYSTEM = """
You are re-scoring the importance of past events in a HeartSteps 
participant's record. You see ALL recent events at once and must judge 
each one's importance RELATIVE to the others. Your response must be a 
single line: ##N1,N2,N3,...## where each Ni is an integer 1-10. 
The number of integers must EXACTLY equal the number of events. 
No other text.
"""

IMPORTANCE_REASSESS_USER = """
Below are {n_events} events from a HeartSteps
participant's recent record. Re-score each event's importance on 1-10,
JUDGING THEM RELATIVE TO EACH OTHER.

An event is important if it carries signal for understanding the
participant's behavior or predicting their future activity. An event is
unimportant if it is routine and predictable from the participant's
baseline.

Scoring guidance:
- 1-3 = routine, predictable from baseline (e.g., zero steps in slot 1 at
  home; expected high steps at gym)
- 4-6 = some deviation from baseline, but not surprising
- 7-8 = clear signal: a meaningful response to a suggestion (positive or
  negative), an unexpected zero, or an unusual peak that breaks the
  participant's recent pattern
- 9-10 = key turning points: first occurrence of a pattern, a complete
  break from previous behavior, a sudden adoption or rejection of a
  suggested behavior, or an extreme outlier vs this user's history

Distribution constraint (REQUIRED):
- At least 50% of events should score 1-5.
- At most 20% of events should score 8+.
- At most 1-2 events should score 10.

Events (numbered, one per line):
{numbered_events}

Output EXACTLY {n_events} integers separated by commas, wrapped in ##...##.
Example for {n_events} events: ##3,2,7,1,5,4,2,6,3,8##

No preamble. No explanation. Just the wrapped list."""

REFLECTION_REASSESS_SYSTEM = (
    "You are re-evaluating past reflections (hypotheses/rules) about a "
    "HeartSteps participant. Each reflection was generated earlier based on "
    "limited evidence. Your job is to judge how well each reflection still "
    "holds up given the new observations that have come in since. Your "
    "response must be a single line: ##N1,N2,N3,...## where each Ni is an "
    "integer 1-10. The number of integers must EXACTLY equal the number of "
    "reflections. No other text."
)

REFLECTION_REASSESS_USER = """Below are {n_reflections} reflections that
were generated earlier about a HeartSteps participant. Then below them are
{n_new_obs} new observations collected SINCE the most recent reflection.

Re-score each reflection on 1-10 based on whether it still holds up given
the new evidence:

- 1-2 = INVALIDATED. The new observations directly contradict this
  reflection's claim. The reflection is now wrong or no longer applies.
- 3-4 = WEAKENED. New evidence shows mixed support. The reflection is
  partly right but should not be relied on as a rule.
- 5 = NEUTRAL. Not enough new evidence to evaluate, or the reflection
  describes a stable baseline pattern that neither gained nor lost
  support.
- 6-7 = CONFIRMED. New evidence supports the reflection's direction. It
  is a useful working hypothesis.
- 8-9 = STRONGLY CONFIRMED. Multiple new observations match this
  reflection's prediction. It captures a real pattern in this user.
- 10 = CRITICAL RULE. The reflection makes a specific prediction that has
  been confirmed multiple times and is highly distinctive of this user.

Distribution constraint (REQUIRED):
- Be willing to give low scores. At least 30% of reflections should
  score 5 or below if no new evidence supports them.
- Do not default to 7-8 for everything. Old reflections that nobody
  re-validated should drop.
- Reserve 10 for at most 1 reflection.

REFLECTIONS (numbered, oldest first):
{numbered_reflections}

NEW OBSERVATIONS since the last reflection (used as evidence for re-scoring):
{numbered_new_observations}

Output EXACTLY {n_reflections} integers separated by commas, wrapped in
##...##. Example for {n_reflections} reflections: ##7,3,8,2,5,9,4,1,6,5##

No preamble. No explanation. Just the wrapped list."""

REFLECTION_Q_SYS = """
You are a participant who is interested in increasing your walking and 
enrolled in a study helping individuals increase and sustain physical activity. 

The study delivers brief activity suggestions via your smartphone at five
decision points per day (early morning, morning, midday, afternoon, evening).
At each decision point, the system observes your current situation and sometimes 
sends a suggestion to encourage you walking or not being still more.

You are now pausing to reflect on how you react to these suggestions based on your recent memories.
"""

# REFLECTION_Q_PROMPT = """
# Recent records of your own behavior from earlier in the study (most recent first):
# {recent_obs}
#
# Recent reflections of your own behavior from earlier in the study:
# {recent_ref}
#
# Based on these records,
# list exactly 3 questions worth investigating about YOUR own behavior.
#
# ## Question design rules
# Each question must cover a DIFFERENT analytical angle. Pick 3 different
# angles from this list — do not pick two from the same angle:
#   (a) BASELINE: when do I walk more/less even without any intervention (when send=0)?
#   (b) IMMEDIATE RESPONSE: among slots where send>0, what conditions
#       predict whether I actually move vs stay still in the next 30 minutes?
#   (c) DOSE EFFECT: does receiving suggestions more frequently in recent
#       days change my behavior?
#   (d) CONTEXT × INTERVENTION: does the same suggestion type work
#       differently across locations / slots / weekday-weekend?
#   (e) CARRYOVER: does my activity in one slot predict my activity
#       in the next slot or the next day?
#   (f) STABILITY: which conditions show consistent behavior vs which
#       are highly variable?
#
# Each question must be specific and grounded in the evidence above (not
# generic).
#
# Format your output as exactly 3 lines, one question per line, in this exact
# format (note: NO ## wrappers for questions; use Q1: / Q2: / Q3: prefix):
#
# Q1: <your first question>
# Q2: <your second question>
# Q3: <your third question>
#
# No other text. No preamble. No explanation. Just the three lines."""

REFLECTION_Q_PROMPT = """
Recent records of your own behavior from earlier in the study (most recent first):
{recent_obs}

Based on these records, 
list exactly 3 questions worth investigating about YOUR own behavior. 

## Question design rules
Each question must cover a DIFFERENT analytical angle. Pick 3 different 
angles from this list — do not pick two from the same angle:
  (a) BASELINE: when do I walk more/less even without any intervention (when send=0)?
  (b) IMMEDIATE RESPONSE: among slots where send>0, what conditions 
      predict whether I actually move vs stay still in the next 30 minutes?
  (c) DOSE EFFECT: does receiving suggestions more frequently in recent 
      days change my behavior?
  (d) CONTEXT × INTERVENTION: does the same suggestion type work 
      differently across locations / slots / weekday-weekend?
  (e) CARRYOVER: does my activity in one slot predict my activity 
      in the next slot or the next day?
  (f) STABILITY: which conditions show consistent behavior vs which 
      are highly variable?

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

# REFLECTION_GEN_PROMPT = """
# A question about yourself: "{question}"
#
# Relevant recent records of your own behavior from earlier in the study (most recent first):
# {recent_obs}
#
# Relevant recent reflections of your own behavior from earlier in the study:
# {recent_ref}
#
# Based ONLY on the evidence above, write a single 1-2 sentence inference about
# yourself that answers the question and be specific.
#
# If the evidence is too thin to draw a confident inference, say so briefly
# rather than inventing a pattern.
#
# Format your output as a single line of plain text — no preamble, no bullet
# points, no quotes, no special wrappers. Start your response directly with
# the inference sentence.
#
# ## Output format Rule
# - A single sentence stating the conclusion.
# - Do NOT include reasoning steps, analysis, or "let me think". Just the conclusion.
# - No other text. Just the one sentence.
#
# ## Examples of BAD output (do NOT do this):
# - "Let me look at the data. First I notice that..."
# - "Okay, the user is asking about..."
# - A multi-paragraph analysis.
#
# Your single-sentence conclusion:
# """


REFLECTION_GEN_PROMPT = """
A question about yourself: "{question}"

Relevant recent records of your own behavior from earlier in the study (most recent first):
{recent_obs}

Based ONLY on the evidence above, write a single 1-2 sentence inference about
yourself that answers the question and be specific.

If the evidence is too thin to draw a confident inference, say so briefly
rather than inventing a pattern.

Format your output as a single line of plain text — no preamble, no bullet
points, no quotes, no special wrappers. Start your response directly with
the inference sentence.

## Output format Rule
- A single sentence stating the conclusion.
- Do NOT include reasoning steps, analysis, or "let me think". Just the conclusion.
- No other text. Just the one sentence.

## Examples of BAD output (do NOT do this):
- "Let me look at the data. First I notice that..."
- "Okay, the user is asking about..."
- A multi-paragraph analysis.

Your single-sentence conclusion:
"""



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

PROMPT_STEPS = """
Your background:
{persona}

Recent records of your own behavior from earlier in the study (most recent first):
{recent_obs}

Recent reflections of your own behavior from earlier in the study:
{recent_ref}

Your current situation:
- Day {study_day} of the study, slot {slot} ({weekday_desc})
- Location: {location}
- Current activity state: {activity}
- Weather: {weather}, {temperature}
- Steps you walked in the prior 30 minutes: {prior_30min_steps}
- Suggestion the system is sending you now: {action_desc}

Key considerations:
- A high step count at a previous slot does NOT necessarily imply a high
  step count now; a zero at a previous slot does NOT necessarily imply
  another zero. 
- Engagement and how many steps you decide to walk should depend on: 
  1. your specific circumstances this moment (current location, 
  current weather and temperature, prior activity, prior steps).
  2. Based on your recent records, whether you feel like walking under 
  this kind of circumstance patterns.
  3. your recent reflections on your behavior patterns 
  (e.g. whether the suggestion appeals to you under current circumstance)
- If the system is sending an active walking suggestion AND your
  reflections show you respond to suggestions, you may walk more than
  your typical baseline.
- If the system is sending an active walking suggestion AND your
  reflections show you do not want to respond to suggestions, 
  you may walk less than your typical baseline.  
- Being in a sedentary location (Home, Work, Electronics Store etc) during
  a non-active activity state might means low or zero steps regardless
  of suggestion.

Decide how many steps you will walk in the next 30 minutes. Your response
must be a single non-negative integer wrapped in this exact format:

##N##

For example: ##0## or ##252## or ##1837##.

No other text. No explanation. Just ##N##."""

