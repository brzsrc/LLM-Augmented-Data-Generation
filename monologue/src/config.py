"""Single-file config — replaces YAML.

All dataset-/archetype-/variant-specific constants live here as Python.
If you need a different dataset, edit the constants at the top.
"""
from __future__ import annotations
import os


# ============================================================================
# Dataset paths
# ============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

CSV_PATH = os.path.join(DATA_DIR, "data_gen.csv")


# ============================================================================
# Column mapping for data_gen.csv
# Schema: uid, study_day, weekday, hr, slot, weather, temp, loc, avail,
#         steps30pre, send, resp, steps10
# (No decision_datetime/date — `study_day` is the day index, `hr` is the hour.)
# ============================================================================
COL_PATIENT_ID = "uid"
COL_ACTION = "send"
COL_DAY = "study_day"           # integer day index (1..N)
COL_HOUR = "hr"                 # integer hour-of-day
COL_SLOT = "slot"               # 1..5
COL_WEEKDAY = "weekday"         # 0..6 (Mon=0)
COL_REWARD_SOURCE = "steps10"
COL_AVAIL = "avail"             # in data_gen this is always True (pre-filtered)

# Categorical encoders (string → int)
ENCODERS = {
    "weather":   {"clear": 0, "cloudy": 1, "bad": 2},
    "temp":      {"freezing": 0, "cold": 1, "cool": 2,
                  "mild": 3, "warm": 4, "hot": 5},
    "loc":       {"home": 0, "work": 1, "dining": 2, "shopping": 3,
                  "service": 4, "activity": 5, "other": 6},
}

# Reverse decoders (int → string) for prompt-ready output
DECODERS = {col: {v: k for k, v in m.items()} for col, m in ENCODERS.items()}

# Columns NEVER allowed in state (causal descendants of current action)
FORBIDDEN_IN_STATE = ["resp", "steps10"]


# ============================================================================
# Fidelity column groups — used by Sim-LLM-style validation gates
# ----------------------------------------------------------------------------
# WD cols are treated as continuous (Wasserstein on min-max-normalized values).
# JSD cols are treated as categorical (Jensen-Shannon Distance on freq vectors).
# Reference: Haas 2024 Sim-LLM Master Thesis §4 (Data Fidelity protocol).
# ============================================================================
FIDELITY_WD_COLS  = ["steps10", "steps30pre", "dosage",
                     "hour_sin", "hour_cos", "study_day", "hr"]
FIDELITY_JSD_COLS = ["weekday", "slot", "weather", "temp", "loc",
                     "send", "avail"]


# ============================================================================
# State features fed to DDQN
# ============================================================================
STATE_FEATURES = [
    "study_day", "weekday", "slot",
    "hour_sin", "hour_cos",
    "weather", "temp", "loc",
    "steps30pre", "dosage",
]


# ============================================================================
# Action space
# ============================================================================
ACTION_NAMES = {0: "no_message", 1: "anti_sedentary_suggestion", 2: "walking_suggestion"}


# ============================================================================
# RL / OPE
# ============================================================================
GAMMA = 0.95
REWARD_LOG_OFFSET = 0.5

EVAL = {
    # K-fold + bootstrap
    "n_folds":         3,
    "seed":            42,
    "bootstrap_B":     2000,
    # DDQN
    "ddqn_iters":      80000,
    "ddqn_eval_every": 6000,
    "ddqn_batch":      512,
    "ddqn_seeds":      3,
    "ddqn_swa_keep":   3,
    # FQE
    "fqe_iters":       20000,
    "fqe_batch":       512,
    "fqe_seeds":       5,
    # Device / AMP
    "device":          "cuda:0",
    "use_amp":         False,
    # Ablation sweep
    "cql_alphas":      (0.0, 1.0),
    # # Debug
    # "debug_dump_csv":  False,
}


# ============================================================================
# Archetype taxonomy — tier-based decision tree on slot_1_hour + steps10 + steps30pre
# ----------------------------------------------------------------------------
# Tier 1: high_activity   — mPED High cluster (~20%);  steps10 OR steps30pre extreme
# Tier 2: low_activity    — mPED Low cluster  (~34%);  sedentary
# Tier 3: morning_active  — moderate + early chronotype (M-type)
# Tier 4: evening_active  — moderate + late chronotype  (E-type, lit: most active)
# Tier 5: standard        — moderate + mid-day chronotype  (fallback)
#
# References:
#   - Fukuoka et al. 2018 (mPED Trial) — 3 baseline activity clusters
#   - Chronotype literature — M-type / N-type / E-type schedule clustering
# Membership tests are evaluated in dict-insertion order; first match wins.
# ============================================================================
def _is_high_activity(p):
    return ((p.activity.steps10.all.mean or 0) > 150
            or (p.activity.steps30pre.mean or 0) > 300)

def _is_low_activity(p):
    return ((p.activity.steps10.all.mean or 0) <= 60
            and (p.activity.steps10.all.zero_pct or 0) >= 0.55)

def _is_morning_active(p):
    return p.anchor.slot_1_hour <= 11

def _is_evening_active(p):
    return p.anchor.slot_1_hour >= 14

def _is_standard(p):
    return True   # catch-all


ARCHETYPES = {
    "high_activity":   {"label": "Highly Active (mPED High)",       "test": _is_high_activity},
    "low_activity":    {"label": "Sedentary (mPED Low)",            "test": _is_low_activity},
    "morning_active":  {"label": "Morning Chronotype (M-type)",     "test": _is_morning_active},
    "evening_active":  {"label": "Evening Chronotype (E-type)",     "test": _is_evening_active},
    "standard":        {"label": "Standard Mid-day Worker",         "test": _is_standard},
}
DEFAULT_ARCHETYPE = "standard"


# ============================================================================
# Variant rules — counts per source + perturbations (DeepPersona 5:3:2 inspired)
# ============================================================================
VARIANTS_PER_SOURCE = {"twin": 2, "sibling": 1, "edge": 1}

VARIANT_PERTURBATIONS = {
    "twin": {
        "mean_steps_scale": (0.9, 1.1),
        "slot_1_delta_choices": [0],
        # No attitude — only activity / lifestyle / compliance fields exist
        "borrow_fractions": {"lifestyle": 0.0,
                              "activity": 0.0, "compliance": 0.0},
        "oversample_sparse_cells": False,
    },
    "sibling": {
        "mean_steps_scale": (0.6, 1.4),
        "slot_1_delta_choices": [-1, 0, 1],
        "borrow_fractions": {"lifestyle": 0.30,
                              "activity": 0.60,   # attitude share absorbed here
                              "compliance": 0.30},
        "oversample_sparse_cells": False,
    },
    "edge": {
        "mean_steps_scale_choices": [0.5, 2.0],
        "slot_1_to_extreme": True,
        "borrow_fractions": {"lifestyle": 0.0,
                              "activity": 0.0, "compliance": 0.0},
        "oversample_sparse_cells": True,
    },
}


# ============================================================================
# B5: compliance phases by attitude type — multipliers per JITAI-Twins lit
# ============================================================================
# Single default — used as FALLBACK when per-user STL trend detection fails.
# When detection succeeds (extractor.py), this is NOT used; the personal
# day_range and activity_mult come from STL trend on daily steps10.
# Empirical data (data_gen.csv) shows honeymoon ~1.18× plateau (significant);
# fatigue not significant aggregate → safest default = single plateau phase.
COMPLIANCE_PHASES = {
    "default": [
        {"name": "plateau", "day_range": (1, 999), "activity_mult": 1.00},
    ],
}


# ============================================================================
# Slot → hour offset (median across HeartSteps users) — used by trajectory sampler
# ============================================================================
SLOT_HOUR_OFFSET = {1: 0, 2: 4, 3: 6, 4: 9, 5: 11}


# ============================================================================
# Transition smoothing
# ============================================================================
TRANSITION_LAPLACE_ALPHA = 0.01     # avoids zero-prob unseen transitions


# ============================================================================
# Hurdle-lognormal steps30pre sampler (trajectory_sampler)
# ----------------------------------------------------------------------------
# Empirical MLE fit across 37 real users (data_gen.csv, non-zero steps30pre):
#   median σ_log = 1.17  (p25=1.09, p75=1.37)
# Lognormal beat gamma in 76% of users + all 5 slots (ΔAIC > 30 per slot).
# ============================================================================
SIGMA_LOG_DEFAULT = 1.17      # fallback when per-user MLE fit fails (n<20)
MAX_STEPS30PRE    = 5000      # clip extreme lognormal tail draws


# ============================================================================
# Post-hoc zero-distribution calibration (zero_calibration.py)
# ----------------------------------------------------------------------------
# Applied after trajectory_sampler returns, before add_derived_features.
#
# CLIP_LOW_THRESHOLD: real data has < 2% of positives in (0, 5] and < 5% in
#   (0, 17]; LLM uniformly emits ~16% in (0, 17] as a low-confidence tail.
#   17 sits at the natural shoulder before real's 18+ activity mass.
#
# ZERO_CAL_KEYS: cell key for per-cell P(steps10=0) calibration. Trade-off:
#   - 3-key ("slot", "send", "avail"): 20 cells, stable target estimates,
#     loses (steps30pre) conditional structure → Q-network may overfit synth
#     state distribution. Distribution-gate max_abs_diff ≈ 0.141.
#   - 4-key (...+"steps30pre_bin"): 60 cells, conditional P(0|state) closer
#     to real → DDQN/FQE see cleaner per-state advantage; smallest real cell
#     has 24 rows (stable). Distribution-gate max_abs_diff ≈ 0.149.
#   When "steps30pre_bin" is in this tuple, zero_calibration internally
#   attaches the bin column using global quartile edges from real.steps30pre
#   (controlled by ZERO_CAL_S30_N_QUANTILES below).
#
# ZERO_CAL_S30_N_QUANTILES: qcut quantile count for steps30pre binning.
#   Empirical: with q=4, real.steps30pre collapses to 3 bins due to the
#   ~30% zero mass; that's the working setting.
# ============================================================================
CLIP_LOW_THRESHOLD       = 17
ZERO_CAL_KEYS            = ("slot", "send", "avail", "steps30pre_bin")
ZERO_CAL_S30_N_QUANTILES = 4


# ============================================================================
# Chain-of-Thought (CoT) JSON schema for the steps10 LLM call
# ----------------------------------------------------------------------------
# 6 reasoning fields BEFORE the integer `value`, motivated by:
#   - anchor_lookup     (Wang 2024 Chain-of-Table: force verbalize the lookup)
#   - phase_application (Tam 2024: reasoning-before-value, NOT after)
#   - context_adjustment (Sidorenko 2025: probability-driven prompting)
#   - momentum_check    (Xu 2024 PAFT: parent-first column dependency)
#   - episodic_check    (StructSynth 2025: temporal coherence)
#   - value             (final integer, schema-constrained 0-10000)
# Property order matters: vLLM/outlines emits keys in this order, which gates
# the LLM into reasoning before committing the answer.
# ============================================================================
STEPS_COT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "anchor_lookup":      {"type": "string", "maxLength": 250},
        "phase_application":  {"type": "string", "maxLength": 200},
        "context_adjustment": {"type": "string", "maxLength": 300},
        "momentum_check":     {"type": "string", "maxLength": 250},
        "episodic_check":     {"type": "string", "maxLength": 500},
        "value":              {"type": "integer", "minimum": 0, "maximum": 10000},
    },
    "required": [
        "anchor_lookup", "phase_application", "context_adjustment",
        "momentum_check", "episodic_check", "value",
    ],
    "additionalProperties": False,
}
