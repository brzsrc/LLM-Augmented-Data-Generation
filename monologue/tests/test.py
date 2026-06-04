"""Post-hoc zero-clip threshold sweep + gate validation.

Loads run3-CoT-no0's existing synth CSV, applies a hard clip
`steps10 <= THRESHOLD → 0` (default 9), then re-runs all validation gates
against real data and writes results to a sibling JSON so we can compare
against the run3 baseline (`outputs/run3-CoT-no0/generation/gate_results.json`).

Run from anywhere:
    python -m monologue.tests.test
or:
    cd monologue && python tests/test.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

# Make `src.*` importable regardless of CWD ----------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MONOLOGUE = os.path.dirname(HERE)
sys.path.insert(0, MONOLOGUE)

from src import data_loader                                # noqa: E402
from src.personas import extractor as persona_extractor    # noqa: E402
from src.personas import archetype as persona_archetype    # noqa: E402
from src.validation import gates as validation_gates       # noqa: E402
from src.pipeline import _persona_to_flat_dict             # noqa: E402


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
THRESHOLD = 17
# Cell key for per-cell zero-rate calibration. (slot, send, avail) was the
# coarse default; adding s30_bin tightens conditional P(0) estimation and
# helps DDQN learn cleaner per-state advantage.
# Verified empirically: with global quartile bins from real.steps30pre,
# all 60 cells in real have ≥24 rows (smallest 3 have 24-25), so the
# target estimates are stable.
CAL_KEYS = ["slot", "send", "avail", "s30_bin"]
S30_N_QUANTILES = 4   # qcut may collapse to 3 if steps30pre has many zeros
SYNTH_CSV = os.path.join(
    MONOLOGUE, "src/outputs/run3-CoT-no0/generation/synthetic_data.csv")
OUT_JSON = os.path.join(
    MONOLOGUE,
    f"src/outputs/run3-CoT-no0/generation/gate_results_clip{THRESHOLD}_cal.json")
BASELINE_JSON = os.path.join(
    MONOLOGUE, "src/outputs/run3-CoT-no0/generation/gate_results.json")
SEED = 42


def add_s30_bin(real: pd.DataFrame, synth: pd.DataFrame, n_q: int = S30_N_QUANTILES):
    """Compute global quartile edges from real.steps30pre, then label both
    real and synth rows with `s30_bin`. Edges are padded with ±inf so synth
    values outside real's range still get a bin (no NaN labels).

    Returns:
        (real_out, synth_out, edges_used)
    """
    _, edges = pd.qcut(real["steps30pre"], q=n_q, duplicates="drop",
                        retbins=True)
    edges = list(edges)
    edges[0]  = -np.inf
    edges[-1] = np.inf
    print(f"[s30_bin] global edges (from real, n_q={n_q} → "
          f"{len(edges)-1} bins): {edges}")

    real_out  = real.copy()
    synth_out = synth.copy()
    real_out["s30_bin"]  = pd.cut(real_out["steps30pre"],
                                   bins=edges, include_lowest=True).astype(str)
    synth_out["s30_bin"] = pd.cut(synth_out["steps30pre"],
                                   bins=edges, include_lowest=True).astype(str)
    return real_out, synth_out, edges


def clip_low_positives(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Fold steps10 ∈ [1, threshold] into 0. Returns a NEW dataframe."""
    out = df.copy()
    mask = (out["steps10"] > 0) & (out["steps10"] <= threshold)
    n = int(mask.sum())
    out.loc[mask, "steps10"] = 0
    print(f"[clip<={threshold}] folded {n} rows ({n/len(out):.2%}) → 0")
    return out


def calibrate_zeros(synth: pd.DataFrame, real: pd.DataFrame,
                     keys: list, seed: int = SEED) -> pd.DataFrame:
    """Knock down LLM-positive rows to 0 so per-cell P(steps10=0) matches real.

    Per cell:
        need = round(p_tgt * n_syn - n_cur_zero)
    Sample `need` rows uniformly from the cell's positive set and set them
    to 0. Cells where current zero count already meets target are left alone
    (we don't un-zero).

    `avail` is treated as a regular categorical key (pandas groupby handles
    bool fine).
    """
    rng = np.random.default_rng(seed)
    p_zero_tgt = (real.groupby(keys, observed=True)["steps10"]
                       .apply(lambda x: float((x == 0).mean()))
                       .rename("p_zero_tgt")
                       .reset_index())
    s = synth.merge(p_zero_tgt, on=keys, how="left")
    steps_out = s["steps10"].to_numpy().copy()

    n_cells, n_knocked, n_skipped_nan, n_skipped_no_pos = 0, 0, 0, 0
    for _, g in s.groupby(keys, observed=True, sort=False):
        n_cells += 1
        p_tgt = float(g["p_zero_tgt"].iloc[0]) if not pd.isna(g["p_zero_tgt"].iloc[0]) else np.nan
        if np.isnan(p_tgt):
            n_skipped_nan += 1
            continue
        pos_idx = g.index[g["steps10"] > 0].to_numpy()
        if len(pos_idx) == 0:
            n_skipped_no_pos += 1
            continue
        n = len(g)
        n_cur_zero = n - len(pos_idx)
        need = int(round(p_tgt * n - n_cur_zero))
        if need <= 0:
            continue
        need = min(need, len(pos_idx))
        pick = rng.choice(pos_idx, size=need, replace=False)
        steps_out[pick] = 0
        n_knocked += need

    out = synth.copy()
    out["steps10"] = steps_out.astype(int)
    print(f"[zero-cal] cells={n_cells}  knocked {n_knocked} pos→0  "
          f"(skipped {n_skipped_nan} NaN-target, "
          f"{n_skipped_no_pos} no-positives)")
    return out


def marginal_report(df: pd.DataFrame, label: str) -> None:
    """Print zero-rate + raw mean per send — checks ranking + zero-rate."""
    print(f"\n  --- {label} ---")
    z = (df.steps10 == 0).groupby(df.send).mean()
    m = df.groupby("send")["steps10"].mean()
    for s in sorted(df["send"].unique()):
        print(f"    send={s}: zero_rate={z[s]:.3f}  mean_steps10={m[s]:.2f}")


def diff_gate_summary(before_path: str, after: dict) -> None:
    """Pretty-print pass/fail diff against the baseline gate_results.json."""
    if not os.path.exists(before_path):
        print(f"  (no baseline at {before_path} — skip diff)")
        return
    with open(before_path) as f:
        before = json.load(f)
    print("\n  gate                  baseline   →   post-clip")
    print("  " + "-" * 50)
    all_keys = sorted(set(before.get("gates", {})) | set(after.get("gates", {})))
    for k in all_keys:
        b = before.get("gates", {}).get(k, {}).get("pass")
        a = after.get("gates",  {}).get(k, {}).get("pass")
        arrow = "  " if b == a else ("↑↑" if a and not b else "↓↓")
        bt = "PASS" if b else ("FAIL" if b is False else "—")
        at = "PASS" if a else ("FAIL" if a is False else "—")
        print(f"  {k:<22}{bt:<8} → {at:<8} {arrow}")


def main() -> None:
    print("=" * 72)
    print(f"CLIP-THRESHOLD VALIDATION  threshold={THRESHOLD}")
    print(f"  synth csv = {SYNTH_CSV}")
    print(f"  baseline  = {BASELINE_JSON}")
    print(f"  out json  = {OUT_JSON}")
    print("=" * 72)

    # ---- 1. Load real + add derived features (matches pipeline.stage1) -----
    print("\n[1/5] LOAD REAL")
    df_real = data_loader.load()
    print(f"  real: {len(df_real)} rows, {df_real['uid'].nunique()} uids")

    # ---- 2. Load synth CSV + clip + calibrate + add derived features -------
    print(f"\n[2/5] LOAD SYNTH + CLIP (≤{THRESHOLD} → 0) + ZERO-CAL")
    df_synth_raw = pd.read_csv(SYNTH_CSV)
    print(f"  synth: {len(df_synth_raw)} rows")
    marginal_report(df_synth_raw, "BEFORE clip")
    df_synth_clipped = clip_low_positives(df_synth_raw, THRESHOLD)
    marginal_report(df_synth_clipped, f"AFTER clip<={THRESHOLD}")
    # Attach global steps30pre bins to BOTH real and synth (consistent labels)
    # before per-cell calibration. Dropped from synth at the end so the
    # saved/validated frame has the same schema as before.
    df_real_b, df_synth_b, _ = add_s30_bin(df_real, df_synth_clipped)
    df_synth_cal_b = calibrate_zeros(df_synth_b, df_real_b,
                                       keys=CAL_KEYS, seed=SEED)
    df_synth_cal = df_synth_cal_b.drop(columns=["s30_bin"])
    marginal_report(df_synth_cal,
                     f"AFTER zero-cal on {'+'.join(CAL_KEYS)}")
    # Same encoding as pipeline does before validation
    df_synth = data_loader.add_derived_features(df_synth_cal)

    # ---- 3. Rebuild personas in the flat dict shape gates expect ------------
    print("\n[3/5] REBUILD PERSONAS (needed by consistency_gate)")
    real_profiles = persona_extractor.extract_all(df_real)
    for p in real_profiles.values():
        p.anchor.archetype = persona_archetype.classify(p)
    synth_personas = persona_archetype.build_synth_personas(real_profiles, seed=SEED)
    synth_personas_dicts = [_persona_to_flat_dict(p) for p in synth_personas]
    print(f"  rebuilt {len(synth_personas_dicts)} synth personas")

    # ---- 4. Run gates -------------------------------------------------------
    print("\n[4/5] RUN GATES")
    gate_results = validation_gates.run_all_gates(
        df_real, df_synth, synth_personas_dicts)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(gate_results, f, indent=2, default=str)
    print(f"  saved → {OUT_JSON}")

    # ---- 5. Compare against baseline ---------------------------------------
    print("\n[5/5] DIFF vs BASELINE")
    print(f"  summary: {gate_results['summary']}")
    diff_gate_summary(BASELINE_JSON, gate_results)

    # Quick targeted check: distribution gate diagnostics
    dist = gate_results.get("gates", {}).get("distribution", {})
    if dist:
        diag = dist.get("diag", {})
        rm = diag.get("real_means", {})
        sm = diag.get("synth_means", {})
        print("\n  distribution-gate per-send means (gate-scale):")
        for s in sorted(set(rm) | set(sm)):
            print(f"    send={s}: real={rm.get(s,'-')}  synth={sm.get(s,'-')}")
        print(f"  max_abs_diff = {diag.get('max_abs_diff')}  "
              f"tol = {diag.get('tolerance')}")


if __name__ == "__main__":
    main()
