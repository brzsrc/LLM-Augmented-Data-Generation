"""Analyze zero-rate structure in real data and compare against synth.

Goal: figure out WHERE the steps10 zero-rate is lost in synth (real ~54%
zero vs synth ~5%). Output conditional tables P(steps10=0 | slot, s30_bin,
avail) and the real-vs-synth gap, plus a markdown summary.

Usage:
  cd monologue
  python scripts/analyze_zero_rate.py
  python scripts/analyze_zero_rate.py --synth_run run3-CoT-no0
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `from src import ...` when run from monologue/
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src import data_loader, config as cfg   # noqa: E402


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
BIN_EDGES  = [0, 1, 50, 200, 500, 1500, np.inf]
BIN_LABELS = ["zero", "very_low", "low", "med", "high", "very_high"]


def add_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["s30_bin"] = pd.cut(df["steps30pre"], bins=BIN_EDGES,
                            labels=BIN_LABELS, right=False,
                            include_lowest=True)
    return df


# ---------------------------------------------------------------------------
# Wilson 95% CI for a proportion
# ---------------------------------------------------------------------------
def wilson_ci(n_pos: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = n_pos / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def zero_rate_table(df: pd.DataFrame, group_cols, min_n: int = 20) -> pd.DataFrame:
    """For each group, return (n, n_zero, zero_rate, ci_lo, ci_hi).

    Cells with n < min_n have NaN rate (still report n) so callers know to
    fall back to a coarser table.
    """
    rows = []
    for keys, g in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(g)
        n_zero = int((g["steps10"] == 0).sum())
        if n < min_n:
            rate = np.nan; lo = np.nan; hi = np.nan
        else:
            rate = n_zero / n
            lo, hi = wilson_ci(n_zero, n)
        rows.append({**dict(zip(group_cols, keys)),
                     "n": n, "n_zero": n_zero,
                     "zero_rate": rate,
                     "ci_lo":     lo,
                     "ci_hi":     hi})
    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    # Round for readability
    for c in ("zero_rate", "ci_lo", "ci_hi"):
        out[c] = out[c].round(3)
    return out


def df_to_md(df: pd.DataFrame) -> str:
    """Tiny markdown-table formatter (avoids the tabulate dependency)."""
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.values]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    body   = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, sep, body])


def gap_table(real_t: pd.DataFrame, synth_t: pd.DataFrame, key_cols) -> pd.DataFrame:
    """Merge real & synth tables on key_cols; return delta + per-side n."""
    r = real_t.rename(columns={"n": "n_real", "n_zero": "nz_real",
                                "zero_rate": "zr_real"})
    s = synth_t.rename(columns={"n": "n_synth", "n_zero": "nz_synth",
                                 "zero_rate": "zr_synth"})
    m = r[key_cols + ["n_real", "nz_real", "zr_real"]].merge(
        s[key_cols + ["n_synth", "nz_synth", "zr_synth"]],
        on=key_cols, how="outer")
    m["delta_zr"] = (m["zr_synth"] - m["zr_real"]).round(3)
    return m.sort_values(key_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persona stat cross-check
# ---------------------------------------------------------------------------
def aggregate_persona_per_slot_zero(profiles_path: Path) -> pd.DataFrame:
    """Average per_slot_zero_pct across uids, per avail branch.

    Persona schema (real_profiles.json):
      profiles[uid].activity.steps10.{avail_true|avail_false}.per_slot_zero_pct[slot]
    """
    if not profiles_path.exists():
        return pd.DataFrame()
    profiles = json.load(open(profiles_path))
    rows = []
    for uid, p in profiles.items():
        s10 = p.get("activity", {}).get("steps10", {})
        for branch, avail in (("avail_true", True), ("avail_false", False)):
            sub = s10.get(branch, {}) or {}
            ps = sub.get("per_slot_zero_pct", {}) or {}
            for slot, val in ps.items():
                rows.append({"uid": int(uid), "avail": avail,
                              "slot": int(slot),
                              "persona_zero_pct": val})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (df.groupby(["slot", "avail"])["persona_zero_pct"]
              .agg(["mean", "median", "count"])
              .reset_index()
              .rename(columns={"mean":   "persona_mean_zero_pct",
                                "median": "persona_median_zero_pct",
                                "count":  "n_uids"}))
    for c in ("persona_mean_zero_pct", "persona_median_zero_pct"):
        agg[c] = agg[c].round(3)
    return agg


# ---------------------------------------------------------------------------
# Hypothesis verdict for the markdown summary
# ---------------------------------------------------------------------------
def hypothesis_verdict(t4_gap: pd.DataFrame) -> str:
    """H1 uniform vs H2 concentrated."""
    g = t4_gap.dropna(subset=["zr_real", "zr_synth"])
    g = g[(g["n_real"] >= 20) & (g["n_synth"] >= 20)].copy()
    if len(g) < 5:
        return "(insufficient dense cells to judge H1 vs H2)"
    deltas = g["delta_zr"].abs()
    cv = float(deltas.std() / max(deltas.mean(), 1e-6))
    # H1 (uniform loss): low CV across cells, delta is consistent
    # H2 (concentrated): high CV; some cells lose ~all zeros, others little
    if cv < 0.35:
        return (f"**H1 (uniform loss)** — CV(|Δzero_rate|)={cv:.2f} is low, "
                "synth loses zeros across all conditions roughly equally. "
                "→ Option A (post-hoc Bernoulli overwrite) is enough.")
    else:
        worst = g.nlargest(5, "delta_zr", keep="first") if (g["delta_zr"] < 0).any() \
                 else g.nlargest(5, "delta_zr")
        cells = worst[["slot", "s30_bin", "avail", "zr_real",
                       "zr_synth", "delta_zr"]].to_dict("records")
        return (f"**H2 (concentrated loss)** — CV(|Δzero_rate|)={cv:.2f} is high. "
                f"Worst 5 cells:\n```\n{json.dumps(cells, indent=2, default=str)}\n```\n"
                "→ Option C (two-stage hurdle: Bernoulli + LLM for positives) is the right fix.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv",  default=str(ROOT / "data" / "data_gen.csv"))
    ap.add_argument("--synth_run", default="run3-CoT-no0",
                     help="run name under src/outputs/")
    ap.add_argument("--out_dir",   default=str(HERE / "zero_rate_analysis"))
    ap.add_argument("--min_n",     type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "src" / "outputs" / args.synth_run

    # ---- Load real + synth ----
    real  = add_bin(data_loader.add_derived_features(pd.read_csv(args.real_csv)))
    synth = add_bin(data_loader.add_derived_features(
        pd.read_csv(run_dir / "generation" / "synthetic_data.csv")))

    print(f"[load] real:  {len(real):,} rows from {args.real_csv}")
    print(f"[load] synth: {len(synth):,} rows from {args.synth_run}")
    print(f"[bin]  steps30pre bins: {BIN_LABELS}")
    print()

    # ---- Phase 1: real conditional tables ----
    print("[phase1] computing real conditional zero-rate tables ...")
    t1_real = zero_rate_table(real,  ["avail"],                              args.min_n)
    t2_real = zero_rate_table(real,  ["slot", "avail"],                      args.min_n)
    t3_real = zero_rate_table(real,  ["s30_bin", "avail"],                   args.min_n)
    t4_real = zero_rate_table(real,  ["slot", "s30_bin", "avail"],           args.min_n)

    t1_real.to_csv(out_dir / "T1_zero_by_avail_real.csv",            index=False)
    t2_real.to_csv(out_dir / "T2_zero_by_slot_avail_real.csv",       index=False)
    t3_real.to_csv(out_dir / "T3_zero_by_bin_avail_real.csv",        index=False)
    t4_real.to_csv(out_dir / "T4_zero_by_slot_bin_avail_real.csv",   index=False)

    n_dense  = int((t4_real["n"] >= args.min_n).sum())
    n_sparse = int(((t4_real["n"] > 0) & (t4_real["n"] < args.min_n)).sum())
    n_empty  = int(60 - len(t4_real))   # 5×6×2 = 60 theoretical cells
    print(f"  T4 cell coverage: {n_dense} dense, {n_sparse} sparse (n<{args.min_n}), "
          f"{n_empty} empty")

    # ---- Phase 1b: persona stat cross-check ----
    print("[phase1b] cross-checking persona per_slot_zero_pct ...")
    persona_agg = aggregate_persona_per_slot_zero(run_dir / "personas" / "real_profiles.json")
    if not persona_agg.empty:
        check = t2_real.merge(persona_agg, on=["slot", "avail"], how="left")
        check["delta_vs_persona"] = (check["zero_rate"]
                                      - check["persona_mean_zero_pct"]).round(3)
        check.to_csv(out_dir / "T2_persona_crosscheck.csv", index=False)
        max_drift = float(check["delta_vs_persona"].abs().max())
        print(f"  max |raw - persona_mean| = {max_drift:.3f} "
              f"({'OK' if max_drift < 0.05 else 'CHECK extractor.py'})")
    else:
        print("  (no real_profiles.json found — skipped)")
        check = None

    # ---- Phase 2: synth tables + gap ----
    print("[phase2] computing synth tables and real-vs-synth gap ...")
    t1_synth = zero_rate_table(synth, ["avail"],                             args.min_n)
    t2_synth = zero_rate_table(synth, ["slot", "avail"],                     args.min_n)
    t3_synth = zero_rate_table(synth, ["s30_bin", "avail"],                  args.min_n)
    t4_synth = zero_rate_table(synth, ["slot", "s30_bin", "avail"],          args.min_n)

    t1_gap = gap_table(t1_real, t1_synth, ["avail"])
    t2_gap = gap_table(t2_real, t2_synth, ["slot", "avail"])
    t3_gap = gap_table(t3_real, t3_synth, ["s30_bin", "avail"])
    t4_gap = gap_table(t4_real, t4_synth, ["slot", "s30_bin", "avail"])

    t1_gap.to_csv(out_dir / "T1_zero_gap.csv", index=False)
    t2_gap.to_csv(out_dir / "T2_zero_gap.csv", index=False)
    t3_gap.to_csv(out_dir / "T3_zero_gap.csv", index=False)
    t4_gap.to_csv(out_dir / "T4_zero_gap.csv", index=False)

    verdict = hypothesis_verdict(t4_gap)

    # ---- Summary markdown ----
    md = []
    md.append(f"# Zero-rate analysis — real vs `{args.synth_run}`\n")
    md.append(f"- real rows: {len(real):,}")
    md.append(f"- synth rows: {len(synth):,}")
    md.append(f"- steps30pre bins: `{BIN_LABELS}` (edges={BIN_EDGES})")
    md.append(f"- min_n per cell: {args.min_n}\n")

    md.append("## T1 — marginal zero rate (sanity check)\n")
    md.append(t1_gap.pipe(df_to_md))
    md.append("")

    md.append("## T2 — zero rate × slot × avail\n")
    md.append(t2_gap.pipe(df_to_md))
    md.append("")

    md.append("## T3 — zero rate × steps30pre bin × avail\n")
    md.append("Hurdle test: if bin=`zero` has near-100% zero in real but synth doesn't, "
              "the LLM is ignoring the 'no recent activity' signal.\n")
    md.append(t3_gap.pipe(df_to_md))
    md.append("")

    md.append(f"## T4 — full 5×6×2 grid: coverage {n_dense} dense / "
               f"{n_sparse} sparse / {n_empty} empty\n")
    md.append("(See `T4_zero_gap.csv` — too wide for inline display.)")
    md.append("")
    md.append("Top 10 cells with biggest synth-zero shortfall (delta_zr most negative):\n")
    worst = (t4_gap.dropna(subset=["zr_real", "zr_synth"])
                    .query(f"n_real >= {args.min_n} and n_synth >= {args.min_n}")
                    .sort_values("delta_zr")
                    .head(10))
    md.append(worst.pipe(df_to_md))
    md.append("")

    if check is not None:
        md.append("## Persona stat cross-check (T2 raw vs aggregated `per_slot_zero_pct`)\n")
        md.append(check.pipe(df_to_md))
        md.append("")

    md.append("## Hypothesis verdict\n")
    md.append(verdict)
    md.append("")

    (out_dir / "summary.md").write_text("\n".join(md))

    print()
    print(f"[done] outputs → {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
