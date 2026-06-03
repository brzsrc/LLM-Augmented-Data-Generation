"""Extract 4-part PersonaProfile from a real patient's data (data_gen.csv).

Attitude (resp-based) was removed — empirical analysis showed resp ⟂ steps10
and DDQN doesn't use resp. Activity profile is now expanded with:
  - steps30pre statistics (mean / median / zero_pct, per-slot variants)
  - context-conditional steps10 distributions (by loc / weather / temp /
    steps30pre bin)
  - momentum_score_positive = corr(steps30pre, steps10) on positive rows
  - context_sensitivity_positive dict (per-dim std of conditional positive steps10)

Compliance phases B4: detected per-user from STL trend on daily steps10.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src import config as cfg
from src.personas.schemas import (
    Anchor, ContextProfile, LifestyleProfile,
    ActivityProfile, Steps10BucketStats, Steps10AllBucket,
    Steps10Profile, Steps30PreProfile,
    CompliancePhase, CompliancePhases, PersonaProfile,
)


# ============================================================================
# Helpers
# ============================================================================
def _norm_counts(s: pd.Series, decode: Optional[Dict[int, str]] = None) -> Dict[str, float]:
    vc = s.value_counts(normalize=True)
    out = {}
    for k, v in vc.items():
        key = decode.get(int(k), str(k)) if decode else str(k)
        out[key] = float(round(v, 3))
    return out


def _within_day_slot_transition(df: pd.DataFrame, col: str,
                                  all_states: list, decoder: dict,
                                  alpha: Optional[float] = None
                                  ) -> Dict[str, Dict[str, float]]:
    if alpha is None:
        alpha = cfg.TRANSITION_LAPLACE_ALPHA
    if len(df) == 0 or cfg.COL_DAY not in df.columns:
        return {}
    trans = {s: {t: alpha for t in all_states} for s in all_states}
    for _, day_g in df.sort_values([cfg.COL_DAY, cfg.COL_SLOT]).groupby(cfg.COL_DAY):
        vals = [decoder.get(int(x), str(x))
                for x in day_g.sort_values(cfg.COL_SLOT)[col].tolist()]
        for prev, curr in zip(vals[:-1], vals[1:]):
            if prev in trans and curr in trans[prev]:
                trans[prev][curr] += 1.0
    for prev in trans:
        tot = sum(trans[prev].values())
        trans[prev] = {t: float(round(v / tot, 3)) for t, v in trans[prev].items()}
    return trans


# ============================================================================
# STL trend extraction (Health Gym style)
# ============================================================================
def stl_trend(series: pd.Series, period: int = 7) -> np.ndarray:
    n = len(series)
    if n == 0:
        return np.array([])
    s = series.copy().astype(float).ffill().bfill().fillna(0.0)
    if n < 2 * period:
        return s.rolling(period, min_periods=1).mean().values
    try:
        from statsmodels.tsa.seasonal import STL
        return STL(s, period=period, robust=True).fit().trend.values
    except Exception:
        return s.rolling(period, min_periods=1).mean().values


# ============================================================================
# A1: context
# ============================================================================
def _extract_context(g: pd.DataFrame) -> ContextProfile:
    weather_states = list(cfg.ENCODERS["weather"].keys())
    temp_states = list(cfg.ENCODERS["temp"].keys())
    return ContextProfile(
        weather_dist=_norm_counts(g["weather"], cfg.DECODERS["weather"]),
        temp_dist=_norm_counts(g["temp"], cfg.DECODERS["temp"]),
        weather_transition=_within_day_slot_transition(
            g, "weather", weather_states, cfg.DECODERS["weather"]),
        temp_transition=_within_day_slot_transition(
            g, "temp", temp_states, cfg.DECODERS["temp"]),
    )


# ============================================================================
# A2: lifestyle
# ============================================================================
def _extract_lifestyle(g: pd.DataFrame) -> LifestyleProfile:
    loc_states = list(cfg.ENCODERS["loc"].keys())
    loc_decoder = cfg.DECODERS["loc"]

    weekday_df = g[g[cfg.COL_WEEKDAY] < 5] if cfg.COL_WEEKDAY in g.columns else g
    weekend_df = g[g[cfg.COL_WEEKDAY] >= 5] if cfg.COL_WEEKDAY in g.columns else g.iloc[0:0]

    weekday_loc_dist = _norm_counts(weekday_df["loc"], loc_decoder) if len(weekday_df) else {}
    weekend_loc_dist = _norm_counts(weekend_df["loc"], loc_decoder) if len(weekend_df) else {}

    weekday_slot_loc = {int(s): _norm_counts(sub["loc"], loc_decoder)
                         for s, sub in weekday_df.groupby(cfg.COL_SLOT)}
    weekend_slot_loc = {int(s): _norm_counts(sub["loc"], loc_decoder)
                         for s, sub in weekend_df.groupby(cfg.COL_SLOT)}

    def _build_loc_transition(df):
        return _within_day_slot_transition(df, "loc", loc_states, loc_decoder)

    avail_rate = float(g[cfg.COL_AVAIL].mean()) if cfg.COL_AVAIL in g.columns else 1.0
    avail_by_slot, avail_by_loc = {}, {}
    if cfg.COL_AVAIL in g.columns:
        for slot, sub in g.groupby(cfg.COL_SLOT):
            avail_by_slot[int(slot)] = float(round(sub[cfg.COL_AVAIL].mean(), 3))
        for loc_int, sub in g.groupby("loc"):
            loc_str = loc_decoder.get(int(loc_int), str(loc_int))
            avail_by_loc[loc_str] = float(round(sub[cfg.COL_AVAIL].mean(), 3))

    unavail_triggers = []
    if cfg.COL_AVAIL in g.columns and len(g):
        cell_stats = (g.groupby([cfg.COL_SLOT, "loc"])
                       .agg(n=("avail", "size"), avail=("avail", "mean")))
        for (slot, loc_int), row in cell_stats.iterrows():
            if row["n"] >= 5 and row["avail"] < avail_rate - 0.15:
                loc_str = loc_decoder.get(int(loc_int), str(loc_int))
                unavail_triggers.append(
                    f"slot={int(slot)} + loc={loc_str}: avail={row['avail']:.0%} "
                    f"(n={int(row['n'])})")
    unavail_triggers = unavail_triggers[:6]

    unusual = []
    full_loc = _norm_counts(g["loc"], loc_decoder) if "loc" in g.columns else {}
    avg = 1.0 / max(len(full_loc), 1)
    for loc, p in full_loc.items():
        if p > 3 * avg and loc not in ("home", "work"):
            unusual.append(f"{loc} {p*100:.0f}%")

    return LifestyleProfile(
        weekday_loc_dist=weekday_loc_dist,
        weekend_loc_dist=weekend_loc_dist,
        weekday_slot_loc=weekday_slot_loc,
        weekend_slot_loc=weekend_slot_loc,
        weekday_loc_transition=_build_loc_transition(weekday_df),
        weekend_loc_transition=_build_loc_transition(weekend_df),
        avail_rate=avail_rate,
        avail_by_slot=avail_by_slot,
        avail_by_loc=avail_by_loc,
        unavail_triggers=unavail_triggers,
        unusual_loc_events=unusual,
    )


# ============================================================================
# B3: activity — expanded with steps30pre + context-conditional steps10
# ============================================================================
def _per_slot_mean_positive(gg: pd.DataFrame) -> Dict[int, float]:
    pos = gg[gg[cfg.COL_REWARD_SOURCE] > 0]
    if len(pos) == 0:
        return {}
    psm = pos.groupby(cfg.COL_SLOT)[cfg.COL_REWARD_SOURCE].mean()
    return {int(k): float(round(v, 1))
            for k, v in psm.items() if pd.notna(v)}


def _per_slot_action_mean_positive(gg: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    pos = gg[gg[cfg.COL_REWARD_SOURCE] > 0]
    if len(pos) == 0:
        return {}
    psa = (pos.groupby([cfg.COL_SLOT, cfg.COL_ACTION])[cfg.COL_REWARD_SOURCE]
              .mean().unstack(cfg.COL_ACTION))
    return {int(slot): {int(a): float(round(v, 1))
                         for a, v in row.items() if pd.notna(v)}
            for slot, row in psa.iterrows()}


def _zero_pct_by_s30_bin(gg: pd.DataFrame, edges, labels,
                          min_n: int = 5) -> Dict[str, float]:
    """P(steps10=0 | steps30pre bin) for one avail bucket, using the persona's
    per-user `edges` / `labels`. Cells with n < min_n are omitted so the LLM
    never sees flaky signals."""
    if "steps30pre" not in gg.columns or len(gg) == 0 or not labels:
        return {}
    bins = pd.cut(gg["steps30pre"].astype(float),
                   bins=edges, labels=labels,
                   right=False, include_lowest=True)
    out: Dict[str, float] = {}
    for label in labels:
        mask = (bins == label)
        n = int(mask.sum())
        if n < min_n:
            continue
        z = float((gg.loc[mask, cfg.COL_REWARD_SOURCE] == 0).mean())
        out[label] = round(z, 3)
    return out


def _steps10_bucket(gg: pd.DataFrame, include_action: bool,
                     s30_edges, s30_labels) -> "Steps10BucketStats":
    """Compute the steps10 stats for one availability bucket.
    `include_action`=True populates per_slot_action_mean_positive (avail=True);
    avail=False leaves it None (action is structurally 0).
    `s30_edges`/`s30_labels` are the persona's per-user steps30pre bins."""
    if len(gg) == 0:
        return Steps10BucketStats(
            mean=0.0, median=0.0, zero_pct=0.0,
            per_slot_mean={}, per_slot_zero_pct={},
            per_slot_mean_positive={},
            per_slot_action_mean_positive={} if include_action else None,
            zero_pct_by_s30_bin={},
        )
    s = gg[cfg.COL_REWARD_SOURCE].astype(float)
    ps_mean = gg.groupby(cfg.COL_SLOT)[cfg.COL_REWARD_SOURCE].mean()
    ps_zero = (gg.assign(_z=lambda d: d[cfg.COL_REWARD_SOURCE] == 0)
                  .groupby(cfg.COL_SLOT)["_z"].mean())

    return Steps10BucketStats(
        mean=float(round(s.mean(), 1)),
        median=float(round(s.median(), 1)),
        zero_pct=float(round((s == 0).mean(), 3)),
        per_slot_mean={int(k): float(round(v, 1))
                        for k, v in ps_mean.items() if pd.notna(v)},
        per_slot_zero_pct={int(k): float(round(v, 3))
                            for k, v in ps_zero.items() if pd.notna(v)},
        per_slot_mean_positive=_per_slot_mean_positive(gg),
        per_slot_action_mean_positive=(_per_slot_action_mean_positive(gg)
                                         if include_action else None),
        zero_pct_by_s30_bin=_zero_pct_by_s30_bin(gg, s30_edges, s30_labels),
    )


def _momentum_pair_pct(g: pd.DataFrame) -> Dict[str, float]:
    """Consecutive-slot zero/non-zero pair frequencies within a single day.

    Returns 4 keys (each rate in [0,1]):
      zero_after_zero, nonzero_after_zero,
      zero_after_nonzero, nonzero_after_nonzero
    Empty dict if fewer than 5 ordered adjacent pairs of either kind.
    """
    if cfg.COL_DAY not in g.columns or len(g) < 2:
        return {}
    z_after_z = n_after_z = 0
    z_after_nz = n_after_nz = 0
    for _, day_g in g.sort_values([cfg.COL_DAY, cfg.COL_SLOT]).groupby(cfg.COL_DAY):
        vals = day_g.sort_values(cfg.COL_SLOT)[cfg.COL_REWARD_SOURCE].tolist()
        for prev, curr in zip(vals[:-1], vals[1:]):
            if prev == 0:
                n_after_z += 1
                if curr == 0:
                    z_after_z += 1
            else:
                n_after_nz += 1
                if curr == 0:
                    z_after_nz += 1
    out: Dict[str, float] = {}
    if n_after_z >= 5:
        r = z_after_z / n_after_z
        out["zero_after_zero"]    = round(r, 3)
        out["nonzero_after_zero"] = round(1.0 - r, 3)
    if n_after_nz >= 5:
        r = z_after_nz / n_after_nz
        out["zero_after_nonzero"]    = round(r, 3)
        out["nonzero_after_nonzero"] = round(1.0 - r, 3)
    return out


def _steps10_mean_after_zero_streak(g: pd.DataFrame,
                                     min_n: int = 10) -> Dict[str, float]:
    """Mean steps10 on the RECOVERY slot (first non-zero after a zero streak).

    For each day-local time series:
      - walk through slots in order; count consecutive zeros
      - when a non-zero slot is hit, attribute its value to the streak-length
        bucket of the preceding zeros
    Streak resets at day boundary (a new day starts with streak=0; the first
    slot of the day, if non-zero, does NOT count toward any bucket).

    Bins: "1" / "2" / "3+" — matches HeartSteps 5-slot/day structure where a
    within-day streak can be at most 4 (slots 1-4 all zero, slot 5 non-zero).
    Buckets with fewer than `min_n` recovery samples are omitted to prevent
    sparse-cell noise from polluting downstream prompts.
    """
    if cfg.COL_DAY not in g.columns or len(g) < 2:
        return {}
    buckets: Dict[str, list] = {"1": [], "2": [], "3+": []}
    for _, day_g in g.sort_values([cfg.COL_DAY, cfg.COL_SLOT]).groupby(cfg.COL_DAY):
        vals = day_g.sort_values(cfg.COL_SLOT)[cfg.COL_REWARD_SOURCE].tolist()
        streak = 0
        for v in vals:
            if v == 0:
                streak += 1
            else:
                if streak >= 1:
                    bucket = "1" if streak == 1 else ("2" if streak == 2 else "3+")
                    buckets[bucket].append(float(v))
                streak = 0
    out: Dict[str, float] = {}
    for k, vs in buckets.items():
        if len(vs) >= min_n:
            out[k] = round(float(np.mean(vs)), 1)
    return out


def _extract_activity(g: pd.DataFrame) -> ActivityProfile:
    steps = g[cfg.COL_REWARD_SOURCE].astype(float)
    pre30 = g["steps30pre"].astype(float) if "steps30pre" in g.columns else pd.Series([])

    # ── Per-user steps30pre bin scheme (used everywhere downstream) ──
    s30_edges, s30_labels = cfg.s30_bins_for_user(pre30.tolist())

    # ── 3 avail buckets for steps10 ──
    if cfg.COL_AVAIL in g.columns:
        g_T = g[g[cfg.COL_AVAIL]]
        g_F = g[~g[cfg.COL_AVAIL]]
    else:
        g_T = g
        g_F = g.iloc[0:0]

    avail_true_b  = _steps10_bucket(g_T, include_action=True,
                                      s30_edges=s30_edges, s30_labels=s30_labels)
    avail_false_b = _steps10_bucket(g_F, include_action=False,
                                      s30_edges=s30_edges, s30_labels=s30_labels)
    all_base      = _steps10_bucket(g,   include_action=False,
                                      s30_edges=s30_edges, s30_labels=s30_labels)

    # ── Context-conditional steps10 — POSITIVE ROWS ONLY ──
    # Rationale: with the new hurdle gate, anchors + multipliers must live on
    # the non-zero subspace, otherwise the chain undershoots after the gate
    # forces zeros at the correct rate.
    g_pos = g[g[cfg.COL_REWARD_SOURCE] > 0]

    def _mean_by(col, decoder=None):
        if col not in g_pos.columns or len(g_pos) == 0:
            return {}
        out = {}
        for val, sub in g_pos.groupby(col):
            key = decoder.get(int(val), str(val)) if decoder else str(val)
            out[key] = float(round(sub[cfg.COL_REWARD_SOURCE].mean(), 1))
        return out

    steps10_by_loc     = _mean_by("loc",     cfg.DECODERS["loc"])
    steps10_by_weather = _mean_by("weather", cfg.DECODERS["weather"])
    steps10_by_temp    = _mean_by("temp",    cfg.DECODERS["temp"])

    steps10_by_steps30pre_bin: Dict[str, float] = {}
    if "steps30pre" in g_pos.columns and len(g_pos) >= 5 and s30_labels:
        bin_series = pd.cut(g_pos["steps30pre"].astype(float),
                             bins=s30_edges, labels=s30_labels,
                             right=False, include_lowest=True)
        for label in s30_labels:
            mask = (bin_series == label)
            if int(mask.sum()) < 5:
                continue
            mu = float(g_pos.loc[mask, cfg.COL_REWARD_SOURCE].mean())
            steps10_by_steps30pre_bin[label] = float(round(mu, 1))

    # ── Aggregate scores (POSITIVE rows only) ──
    momentum_score = 0.0
    if len(g_pos) >= 5:
        pre30_pos  = g_pos["steps30pre"].astype(float) if "steps30pre" in g_pos.columns else pd.Series([])
        steps_pos  = g_pos[cfg.COL_REWARD_SOURCE].astype(float)
        if (len(pre30_pos) >= 5 and pre30_pos.std() > 0
                and steps_pos.std() > 0):
            momentum_score = float(round(pre30_pos.corr(steps_pos), 3))

    # ── Marginal positive mean (for prompt "Baseline mean steps10 (non-zero)") ──
    mean_positive = float(round(g_pos[cfg.COL_REWARD_SOURCE].mean(), 1)) \
                     if len(g_pos) else 0.0

    # ── Consecutive-slot zero-state pair frequencies (avail-agnostic) ──
    momentum_pair_pct = _momentum_pair_pct(g)

    # ── Recovery-magnitude by zero-streak length (POSITIVE on the recovery slot) ──
    mean_after_zero_streak = _steps10_mean_after_zero_streak(g)

    def _std_of(d):
        return float(round(np.std(list(d.values())), 1)) if len(d) else 0.0

    context_sensitivity = {
        "loc":        _std_of(steps10_by_loc),
        "weather":    _std_of(steps10_by_weather),
        "temp":       _std_of(steps10_by_temp),
        "steps30pre": _std_of(steps10_by_steps30pre_bin),
    }

    # ── High-activity (slot, loc) anchors — POSITIVE rows only ──
    # Mean threshold is computed on the positive subspace so the "burst" cells
    # are real motion bursts, not artifacts of zero-rate variation.
    high_ctx: List[str] = []
    if "loc" in g_pos.columns and len(g_pos):
        ct = g_pos.groupby([cfg.COL_SLOT, "loc"])[cfg.COL_REWARD_SOURCE].agg(["count", "mean"])
        if len(ct):
            overall_mean_pos = float(g_pos[cfg.COL_REWARD_SOURCE].mean())
            for (slot, loc_int), row in ct.iterrows():
                if row["count"] >= 5 and row["mean"] > 1.5 * overall_mean_pos:
                    loc_str = cfg.DECODERS["loc"].get(int(loc_int), str(loc_int))
                    high_ctx.append(f"slot={int(slot)} + loc={loc_str} "
                                    f"(mean={row['mean']:.0f}, n={int(row['count'])})")

    # ── steps30pre profile ──
    per_slot_pre_mean = (g.groupby(cfg.COL_SLOT)["steps30pre"].mean()
                          if "steps30pre" in g.columns else pd.Series([]))
    per_slot_pre_zero = (g.assign(_z=lambda d: d["steps30pre"] == 0)
                            .groupby(cfg.COL_SLOT)["_z"].mean()
                          if "steps30pre" in g.columns else pd.Series([]))

    # MLE σ_log on non-zero steps30pre (hurdle-lognormal shape; fallback 1.17)
    sigma_log = cfg.SIGMA_LOG_DEFAULT
    pos_pre = pre30[pre30 > 0].values if len(pre30) else np.array([])
    if len(pos_pre) >= 20:
        try:
            from scipy import stats as _scstats
            sigma_log = float(round(_scstats.lognorm.fit(pos_pre, floc=0)[0], 3))
        except Exception:
            pass

    steps30pre_profile = Steps30PreProfile(
        mean=float(round(pre30.mean(), 1))   if len(pre30) else 0.0,
        median=float(round(pre30.median(), 1)) if len(pre30) else 0.0,
        zero_pct=float(round((pre30 == 0).mean(), 3)) if len(pre30) else 0.0,
        per_slot_mean={int(k): float(round(v, 1))
                        for k, v in per_slot_pre_mean.items() if pd.notna(v)},
        per_slot_zero_pct={int(k): float(round(v, 3))
                            for k, v in per_slot_pre_zero.items() if pd.notna(v)},
        sigma_log=sigma_log,
        bin_edges=s30_edges,
        bin_labels=s30_labels,
    )

    # `all` bucket is richer (Steps10AllBucket) — carries the union-marginals.
    all_b = Steps10AllBucket(
        mean=all_base.mean,
        median=all_base.median,
        zero_pct=all_base.zero_pct,
        per_slot_mean=all_base.per_slot_mean,
        per_slot_zero_pct=all_base.per_slot_zero_pct,
        per_slot_mean_positive=all_base.per_slot_mean_positive,
        per_slot_action_mean_positive=all_base.per_slot_action_mean_positive,
        zero_pct_by_s30_bin=all_base.zero_pct_by_s30_bin,
        mean_positive=mean_positive,
        by_loc_positive=steps10_by_loc,
        by_weather_positive=steps10_by_weather,
        by_temp_positive=steps10_by_temp,
        by_steps30pre_bin_positive=steps10_by_steps30pre_bin,
        momentum_score_positive=momentum_score,
        momentum_pair_pct=momentum_pair_pct,
        steps10_mean_after_zero_streak=mean_after_zero_streak,
        context_sensitivity_positive=context_sensitivity,
        high_activity_contexts_positive=high_ctx[:5],
    )

    steps10_profile = Steps10Profile(
        avail_true=avail_true_b,
        avail_false=avail_false_b,
        all=all_b,
    )

    return ActivityProfile(
        steps30pre=steps30pre_profile,
        steps10=steps10_profile,
    )


# ============================================================================
# B4: compliance phases — from STL trend on daily steps10
# ============================================================================
_H_THRESH = 1.10
_F_THRESH = 0.90
_DIP_TOLERANCE = 2


def _detect_phase_boundaries(trend: np.ndarray, days: List[int]) -> Tuple[int, int]:
    if len(trend) == 0 or len(days) == 0:
        return 0, 999
    baseline = float(np.median(trend))
    if baseline <= 0:
        return 0, 999

    h_end, consec_below = 0, 0
    for i, day in enumerate(days):
        if trend[i] >= baseline * _H_THRESH:
            h_end = int(day)
            consec_below = 0
        else:
            consec_below += 1
            if consec_below > _DIP_TOLERANCE:
                break

    f_start, consec_above = 999, 0
    for i in range(len(days) - 1, -1, -1):
        if trend[i] <= baseline * _F_THRESH:
            f_start = int(days[i])
            consec_above = 0
        else:
            consec_above += 1
            if consec_above > _DIP_TOLERANCE:
                break

    if f_start <= h_end + 7:
        f_start = 999
    return h_end, f_start


def _compute_phase_mults(trend: np.ndarray, days: List[int],
                          h_end: int, f_start: int) -> Dict[str, float]:
    days_arr = np.array(days)
    h_mask = days_arr <= h_end if h_end > 0 else np.zeros_like(days_arr, dtype=bool)
    f_mask = days_arr >= f_start if f_start < 999 else np.zeros_like(days_arr, dtype=bool)
    p_mask = ~h_mask & ~f_mask

    p_mean = float(np.mean(trend[p_mask])) if p_mask.any() else float(np.mean(trend))
    if p_mean <= 0:
        return {"honeymoon": 1.0, "fatigue": 1.0}
    h_mean = float(np.mean(trend[h_mask])) if h_mask.any() else p_mean
    f_mean = float(np.mean(trend[f_mask])) if f_mask.any() else p_mean
    return {
        "honeymoon": round(h_mean / p_mean, 3),
        "fatigue":   round(f_mean / p_mean, 3),
    }


def _extract_compliance(g: pd.DataFrame) -> CompliancePhases:
    """Per-user CompliancePhases from STL trend on daily steps10. Falls back
    to config defaults if detection fails."""
    def _fallback():
        spec_list = cfg.COMPLIANCE_PHASES["default"]
        return CompliancePhases(phases=[CompliancePhase(**spec) for spec in spec_list])

    if cfg.COL_DAY not in g.columns:
        return _fallback()
    daily = g.groupby(cfg.COL_DAY)[cfg.COL_REWARD_SOURCE].mean().sort_index()
    if len(daily) < 7:
        return _fallback()

    trend = stl_trend(daily, period=7)
    days = daily.index.values.tolist()
    h_end, f_start = _detect_phase_boundaries(trend, days)

    if h_end == 0 and f_start == 999:
        return _fallback()

    mults = _compute_phase_mults(trend, days, h_end, f_start)
    phases: List[CompliancePhase] = []
    if h_end > 0:
        phases.append(CompliancePhase(
            name="honeymoon", day_range=(1, h_end),
            activity_mult=mults["honeymoon"],
        ))
    plateau_start = h_end + 1 if h_end > 0 else 1
    plateau_end = f_start - 1 if f_start < 999 else 999
    phases.append(CompliancePhase(
        name="plateau", day_range=(plateau_start, plateau_end),
        activity_mult=1.0,
    ))
    if f_start < 999:
        phases.append(CompliancePhase(
            name="fatigue", day_range=(f_start, 999),
            activity_mult=mults["fatigue"],
        ))
    return CompliancePhases(phases=phases)


# ============================================================================
# Entry
# ============================================================================
def extract_one(uid: int, g: pd.DataFrame) -> PersonaProfile:
    slot1 = g[g[cfg.COL_SLOT] == 1]
    slot_1_hour = int(slot1[cfg.COL_HOUR].median()) if (
        cfg.COL_HOUR in g.columns and len(slot1)) else 12
    n_days = int(g[cfg.COL_DAY].nunique()) if cfg.COL_DAY in g.columns else 30

    anchor = Anchor(
        source_uid=int(uid), archetype="", variant_type="source",
        synth_uid=int(uid),    # real persona's "uid in CSV" == its source_uid
        slot_1_hour=slot_1_hour, n_days=n_days,
    )
    return PersonaProfile(
        anchor=anchor,
        context=_extract_context(g),
        lifestyle=_extract_lifestyle(g),
        activity=_extract_activity(g),
        compliance=_extract_compliance(g),
    )


def extract_all(df: pd.DataFrame) -> Dict[int, PersonaProfile]:
    return {int(uid): extract_one(uid, g)
            for uid, g in df.groupby(cfg.COL_PATIENT_ID)}
