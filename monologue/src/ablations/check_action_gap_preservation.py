"""Test whether a synthetic dataset preserves the per-cell action-gap signal
present in the real offline data.

The action gap is what DDQN's Bellman update consumes. For each state cell
`s = (slot, loc)`:

    Δ_a(s) = E[steps10 | s, send=a] - E[steps10 | s, send=0]    for a in {1, 2}

A synthetic dataset "preserves the signal" iff, on the cells where real has a
well-estimated Δ:
  (i)  the sign of Δ matches real's,
  (ii) the magnitude |Δ_synth| is at least half of |Δ_real|.

We also decompose `E[steps10 | s, a]` into  (1 − zero_rate) × non-zero mean,
since real's gap is carried almost entirely by the non-zero conditional mean,
not by the zero rate.

Usage (from monologue/):
    python -m src.ablations.check_action_gap_preservation \\
        --synth outputs/run3-CoT-no0/generation/synthetic_data.csv \\
        --label run3 \\
        --out   ../results_analysis/action_gap_run3.csv \\
        --avail-only
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.data_loader import add_derived_features


# monologue/  (parents: ablations -> src -> monologue)
MONOLOGUE = Path(__file__).resolve().parents[2]
REPO = MONOLOGUE.parent

STATE_KEYS = ("slot", "loc")
TARGET = "steps10"
ACTION = "send"
ACTIONS = (0, 1, 2)
MIN_N_PER_ARM = 10               # cells need this many samples per arm to count
SIGN_MAGNITUDE_FRACTION = 0.5    # |Δ_synth| ≥ this × |Δ_real|  ⇒ preserved


def per_cell_arm_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, arm): n, mean, zero_rate, nonzero_mean, nonzero_p90."""
    rows = []
    for cell, g in df.groupby(list(STATE_KEYS)):
        cell_str = str(cell)
        for a in ACTIONS:
            sub = g[g[ACTION] == a][TARGET]
            n = len(sub)
            if n < MIN_N_PER_ARM:
                continue
            nz = sub[sub > 0]
            rows.append({
                "cell": cell_str,
                "send": a,
                "n": n,
                "mean": float(sub.mean()),
                "zero_rate": float((sub == 0).mean()),
                "nonzero_mean": float(nz.mean()) if len(nz) else 0.0,
                "nonzero_p90":  float(nz.quantile(0.9)) if len(nz) else 0.0,
            })
    return pd.DataFrame(rows)


def per_cell_action_gaps(stats: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-cell stats and compute Δ_a1, Δ_a2 plus non-zero components."""
    cells_with_all_arms = (
        stats.groupby("cell")["send"].nunique() == len(ACTIONS)
    )
    cells_with_all_arms = cells_with_all_arms[cells_with_all_arms].index
    s = stats[stats["cell"].isin(cells_with_all_arms)]

    out = []
    for cell, g in s.groupby("cell"):
        g = g.set_index("send")
        out.append({
            "cell":            cell,
            "n_a0":            int(g.loc[0, "n"]),
            "n_a1":            int(g.loc[1, "n"]),
            "n_a2":            int(g.loc[2, "n"]),
            "mean_a0":         g.loc[0, "mean"],
            "mean_a1":         g.loc[1, "mean"],
            "mean_a2":         g.loc[2, "mean"],
            "delta_a1":        g.loc[1, "mean"] - g.loc[0, "mean"],
            "delta_a2":        g.loc[2, "mean"] - g.loc[0, "mean"],
            "nz_mean_a0":      g.loc[0, "nonzero_mean"],
            "nz_mean_a1":      g.loc[1, "nonzero_mean"],
            "nz_mean_a2":      g.loc[2, "nonzero_mean"],
            "nz_delta_a1":     g.loc[1, "nonzero_mean"] - g.loc[0, "nonzero_mean"],
            "nz_delta_a2":     g.loc[2, "nonzero_mean"] - g.loc[0, "nonzero_mean"],
            "zr_a0":           g.loc[0, "zero_rate"],
            "zr_a1":           g.loc[1, "zero_rate"],
            "zr_a2":           g.loc[2, "zero_rate"],
        })
    return pd.DataFrame(out).set_index("cell")


def preservation_verdict(real: pd.DataFrame, synth: pd.DataFrame,
                          label: str) -> Tuple[pd.DataFrame, dict]:
    """Join per-cell gaps from real and synth; flag preservation per cell."""
    common = real.index.intersection(synth.index)
    if len(common) == 0:
        raise ValueError("No common cells between real and synth.")

    j = pd.DataFrame(index=common)
    j["real_delta_a1"]  = real.loc[common, "delta_a1"]
    j["real_delta_a2"]  = real.loc[common, "delta_a2"]
    j["synth_delta_a1"] = synth.loc[common, "delta_a1"]
    j["synth_delta_a2"] = synth.loc[common, "delta_a2"]
    j["real_nz_delta_a1"]  = real.loc[common,  "nz_delta_a1"]
    j["synth_nz_delta_a1"] = synth.loc[common, "nz_delta_a1"]
    j["real_nz_delta_a2"]  = real.loc[common,  "nz_delta_a2"]
    j["synth_nz_delta_a2"] = synth.loc[common, "nz_delta_a2"]

    for a in (1, 2):
        rk, sk = f"real_delta_a{a}", f"synth_delta_a{a}"
        # Sign match (treat near-zero real Δ as ambiguous)
        sig_real = np.where(j[rk].abs() < 2, 0, np.sign(j[rk]))
        sig_syn  = np.sign(j[sk])
        j[f"sign_match_a{a}"]    = (sig_real != 0) & (sig_real == sig_syn)
        j[f"sign_ambig_a{a}"]    = sig_real == 0
        j[f"mag_ratio_a{a}"]     = j[sk].abs() / j[rk].abs().replace(0, np.nan)
        j[f"preserved_a{a}"]     = (
            j[f"sign_match_a{a}"]
            & (j[f"mag_ratio_a{a}"] >= SIGN_MAGNITUDE_FRACTION)
        )

    # Summary
    def _summ(a):
        rk, sk = f"real_delta_a{a}", f"synth_delta_a{a}"
        n_amb = int(j[f"sign_ambig_a{a}"].sum())
        n_eval = len(j) - n_amb
        sign_ok = int(j.loc[~j[f"sign_ambig_a{a}"], f"sign_match_a{a}"].sum())
        preserved = int(j[f"preserved_a{a}"].sum())
        rho, p_rho = spearmanr(j[rk], j[sk])
        # Magnitude preservation: |synth| vs |real|
        ratio = (j[sk].abs() / j[rk].abs().replace(0, np.nan)).dropna()
        med_mag_ratio = float(ratio.median())
        try:
            w, p_wil = wilcoxon(j[sk].abs() - j[rk].abs())
            wil = (float(w), float(p_wil))
        except ValueError:
            wil = (float("nan"), float("nan"))
        return {
            "n_cells_total":          len(j),
            "n_cells_ambig_real":     n_amb,
            "n_cells_evaluated":      n_eval,
            "sign_match_rate":        round(sign_ok / max(n_eval, 1), 3),
            "preserved_count":        preserved,
            "preserved_rate":         round(preserved / max(n_eval, 1), 3),
            "spearman_rho":           round(float(rho), 3),
            "spearman_p":             round(float(p_rho), 4),
            "median_magnitude_ratio": round(med_mag_ratio, 3),
            "wilcoxon_W":             round(wil[0], 1),
            "wilcoxon_p_magnitude":   round(wil[1], 4),
            "mean_abs_real":          round(float(j[rk].abs().mean()), 2),
            "mean_abs_synth":         round(float(j[sk].abs().mean()), 2),
        }

    summary = {
        "label":            label,
        "a1_vs_a0":         _summ(1),
        "a2_vs_a0":         _summ(2),
    }
    return j.round(2), summary


def verdict_string(summary: dict, key: str) -> str:
    """Two-axis verdict: magnitude (|Δ_synth| ≥ 0.8·|Δ_real|) + direction (≥ 70% sign match)."""
    s = summary[key]
    mag_ratio = s["mean_abs_synth"] / max(s["mean_abs_real"], 1e-9)
    mag_ok    = mag_ratio >= 0.80
    sign_ok   = s["sign_match_rate"] >= 0.70

    if mag_ok and sign_ok:
        tag = "✓ PRESERVED"
    elif mag_ok or sign_ok:
        tag = "~ PARTIAL"
    else:
        tag = "✗ LOST"

    return (f"{tag:14s}  "
            f"magnitude {s['mean_abs_synth']:5.1f} vs real {s['mean_abs_real']:5.1f} "
            f"(ratio {mag_ratio:.2f}, {'OK' if mag_ok else 'LOW'})  |  "
            f"sign-match {s['sign_match_rate']*100:.0f}% "
            f"({'OK' if sign_ok else 'LOW'})  |  "
            f"spearman ρ={s['spearman_rho']:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real",  default=str(MONOLOGUE / "data/data_gen.csv"))
    ap.add_argument("--synth", required=True,
                    help="Path to a synth CSV (e.g. run3 synthetic_data.csv)")
    ap.add_argument("--label", default="synth",
                    help="Short label for the synth dataset")
    ap.add_argument("--out",   default=None,
                    help="Optional: write per-cell table to this CSV")
    ap.add_argument("--avail-only", action="store_true",
                    help="Restrict both real and synth to avail=True rows "
                         "(eliminates the structural avail=False ⇒ send=0 bias).")
    args = ap.parse_args()

    print(f"[load] real:  {args.real}")
    real_df = add_derived_features(pd.read_csv(args.real))
    print(f"[load] synth: {args.synth}")
    synth_df = add_derived_features(pd.read_csv(args.synth))

    if args.avail_only:
        n_r0, n_s0 = len(real_df), len(synth_df)
        real_df  = real_df[real_df["avail"] == True].copy()
        synth_df = synth_df[synth_df["avail"] == True].copy()
        print(f"[filter] avail=True only: real {n_r0}→{len(real_df)}, "
              f"synth {n_s0}→{len(synth_df)}")

    print(f"\n[compute] per-cell arm stats ...")
    real_stats  = per_cell_arm_stats(real_df)
    synth_stats = per_cell_arm_stats(synth_df)

    print(f"[compute] per-cell action gaps ...")
    real_gaps  = per_cell_action_gaps(real_stats)
    synth_gaps = per_cell_action_gaps(synth_stats)
    print(f"  real:  {len(real_gaps)} cells with all 3 arms (n ≥ {MIN_N_PER_ARM})")
    print(f"  synth: {len(synth_gaps)} cells with all 3 arms")

    per_cell, summary = preservation_verdict(real_gaps, synth_gaps, args.label)

    # ---- Report ----
    print(f"\n{'='*72}")
    print(f"  Action-gap preservation: {args.label}  (vs real)")
    print(f"{'='*72}")
    print(f"  Cells with all 3 arms in BOTH datasets: {len(per_cell)}")
    print()
    print(f"  Δ(a=1 vs a=0):  {verdict_string(summary, 'a1_vs_a0')}")
    print(f"  Δ(a=2 vs a=0):  {verdict_string(summary, 'a2_vs_a0')}")
    print()
    print(f"  Non-zero mean component (where real's gap actually lives):")
    nzr_a1, p_nzr_a1 = spearmanr(per_cell["real_nz_delta_a1"],
                                  per_cell["synth_nz_delta_a1"])
    nzr_a2, p_nzr_a2 = spearmanr(per_cell["real_nz_delta_a2"],
                                  per_cell["synth_nz_delta_a2"])
    print(f"    Spearman ρ on nz_delta_a1: {nzr_a1:.3f}  (p={p_nzr_a1:.4f})")
    print(f"    Spearman ρ on nz_delta_a2: {nzr_a2:.3f}  (p={p_nzr_a2:.4f})")

    print(f"\n  Detail — Δ(a=1 vs a=0):")
    for k, v in summary["a1_vs_a0"].items():
        print(f"    {k:>28s}: {v}")
    print(f"\n  Detail — Δ(a=2 vs a=0):")
    for k, v in summary["a2_vs_a0"].items():
        print(f"    {k:>28s}: {v}")

    print(f"\n  Top 10 cells where synth most amplified |Δ_a1| (good for DDQN):")
    top = per_cell.copy()
    top["amp_a1"] = top["synth_delta_a1"].abs() - top["real_delta_a1"].abs()
    print(top.nlargest(10, "amp_a1")[
        ["real_delta_a1", "synth_delta_a1", "amp_a1"]
    ].to_string())

    print(f"\n  Top 10 cells where synth most compressed |Δ_a1| (bad for DDQN):")
    print(top.nsmallest(10, "amp_a1")[
        ["real_delta_a1", "synth_delta_a1", "amp_a1"]
    ].to_string())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_cell.to_csv(out_path)
        print(f"\n[save] per-cell table → {out_path}")

    # Exit code: 0 iff magnitude AND direction pass on BOTH action arms
    def _axes_ok(key):
        s = summary[key]
        mag_ok  = (s["mean_abs_synth"] / max(s["mean_abs_real"], 1e-9)) >= 0.80
        sign_ok = s["sign_match_rate"] >= 0.70
        return mag_ok, sign_ok

    a1_mag, a1_sign = _axes_ok("a1_vs_a0")
    a2_mag, a2_sign = _axes_ok("a2_vs_a0")
    overall = a1_mag and a1_sign and a2_mag and a2_sign
    print(f"\n  → Overall verdict: {'✓ SIGNAL PRESERVED' if overall else '✗ SIGNAL DEGRADED'}")
    print(f"     a=1 vs a=0:  magnitude {'✓' if a1_mag else '✗'}   direction {'✓' if a1_sign else '✗'}")
    print(f"     a=2 vs a=0:  magnitude {'✓' if a2_mag else '✗'}   direction {'✓' if a2_sign else '✗'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
