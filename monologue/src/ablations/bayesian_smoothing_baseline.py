"""Bayesian smoothing baseline for the LLM-augmentation ablation.

We test the hypothesis that LLM-generated synthetic data helps DDQN
through "conditional denoising" of the offline reward distribution. If
that mechanism is correct, then a simple non-LLM denoising procedure
(empirical-Bayes shrinkage per state-action cell) should reproduce most
of the DDQN improvement. If naive smoothing does NOT reproduce the gain,
the LLM is contributing something more than denoising.

Three baseline variants are generated; each writes a synthetic_data.csv
that the existing pipeline.evaluate stage can consume directly:

  A. replace        — real rows, reward replaced by per-cell EB posterior
                      draws (volume = real, denoised). Isolates pure denoising.
  B. replace_4x     — like A but oversampled 4x (volume ≈ run3, denoised).
                      Tests denoising + volume jointly.
  C. oversample     — real rows, original reward, replicated 4x (volume ≈ run3,
                      original noise). Isolates pure data-volume effect.

The cell is defined as (slot, loc, steps30pre_bin, send) — adding the
steps30pre activity bin (4 levels) gives DDQN a momentum signal that
the original (slot, loc, send) cell key could not encode. Shrinkage is
empirical Bayes:
    τ = σ²_within / σ²_between
    μ̂_c = (n_c · ȳ_c + τ · μ̂_global) / (n_c + τ)
    σ̂²_c = σ²_within / (n_c + τ)

Sparse cells shrink toward the global mean; dense cells stay near their
empirical mean. We operate on raw steps10 (not log-reward) so the output
CSV matches the schema of LLM-generated synthetic_data.csv.

Usage (from monologue/):
    python -m src.ablations.bayesian_smoothing_baseline \\
        --variant all \\
        --seed 42
"""
from __future__ import annotations
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


# monologue/  (parents: ablations -> src -> monologue)
MONOLOGUE = Path(__file__).resolve().parents[2]


CELL_KEYS = ("slot", "loc", "steps30pre_bin", "send")
TARGET = "steps10"

# steps30pre binning — captures user's recent activity state, which is
# a key DDQN state feature. Without it the cell EB baseline ignores
# the user's momentum signal entirely (see ablation discussion).
# 4 bins chosen to balance cell density (~14 rows/cell median) vs resolution:
#   0:     sedentary       (steps30pre == 0)          ~36% of rows
#   1-50:  barely moving                              ~12%
#   51-200: moving normally                           ~23%
#   201+:  actively walking                           ~29%
STEPS30PRE_BIN_EDGES  = [-0.01, 0.5, 50, 200, float("inf")]
STEPS30PRE_BIN_LABELS = [0, 1, 2, 3]


def _add_steps30pre_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'steps30pre_bin' column used as a cell-key dimension."""
    out = df.copy()
    out["steps30pre_bin"] = pd.cut(
        out["steps30pre"], bins=STEPS30PRE_BIN_EDGES,
        labels=STEPS30PRE_BIN_LABELS).astype(int)
    return out


@dataclass
class EBFit:
    """Empirical-Bayes hierarchical shrinkage on per-cell means."""
    per_cell_mean: pd.Series       # ȳ_c
    per_cell_n:    pd.Series       # n_c
    per_cell_var:  pd.Series       # s²_c (empirical within-cell variance)
    global_mean:   float           # μ̂
    var_within:    float           # σ̂²_within (pooled within-cell variance)
    var_between:   float           # σ̂²_between (variance of cell means)
    tau:           float           # shrinkage param σ²_w / σ²_b

    def posterior_mean(self, cell) -> float:
        n_c = self.per_cell_n.get(cell, 0)
        y_c = self.per_cell_mean.get(cell, self.global_mean)
        return (n_c * y_c + self.tau * self.global_mean) / (n_c + self.tau)

    def posterior_var(self, cell) -> float:
        n_c = self.per_cell_n.get(cell, 0)
        return self.var_within / (n_c + self.tau)

    def sample(self, cell, rng: np.random.Generator) -> float:
        """Draw a single synthetic value from N(posterior_mean, posterior_var).
        Clipped at 0 since steps10 cannot be negative."""
        mu  = self.posterior_mean(cell)
        var = self.posterior_var(cell)
        val = rng.normal(mu, np.sqrt(max(var, 1e-9)))
        return float(max(0.0, val))


def fit_eb(real_df: pd.DataFrame) -> EBFit:
    """Fit empirical-Bayes hierarchical model on (slot, loc, steps30pre_bin, send) cells."""
    real_df = _add_steps30pre_bin(real_df[real_df["avail"] == True])
    grp = real_df.groupby(list(CELL_KEYS))[TARGET]
    n_c     = grp.size()
    mean_c  = grp.mean()
    var_c   = grp.var(ddof=1).fillna(0.0)

    global_mean = float(real_df[TARGET].mean())
    # Pooled within-cell variance (weighted by n_c - 1)
    var_within = float(((n_c - 1) * var_c).sum() / max((n_c - 1).sum(), 1))
    # Between-cell variance (of the cell means themselves)
    var_between = float(np.var(mean_c.values, ddof=1))

    tau = var_within / max(var_between, 1e-9)

    print(f"  [EB] N cells: {len(mean_c)}    global mean ȳ = {global_mean:.2f}")
    print(f"  [EB] σ̂²_within = {var_within:.1f}    σ̂²_between = {var_between:.1f}")
    print(f"  [EB] τ = σ²_w / σ²_b = {tau:.2f}")
    print(f"  [EB] (large τ → strong shrinkage; small τ → trust empirical)")

    return EBFit(
        per_cell_mean=mean_c, per_cell_n=n_c, per_cell_var=var_c,
        global_mean=global_mean, var_within=var_within,
        var_between=var_between, tau=tau,
    )


def make_replace(real_df: pd.DataFrame, fit: EBFit,
                  rng: np.random.Generator) -> pd.DataFrame:
    """Variant A: same rows as real, but reward sampled from posterior.

    The cell key includes steps30pre_bin; we compute it here so the input
    real_df doesn't need to carry it. The synth output preserves the bin
    column too (harmless — DDQN's STATE_FEATURES does not include it)."""
    df = _add_steps30pre_bin(real_df)
    cells = list(zip(*[df[k].values for k in CELL_KEYS]))
    new_steps = np.array([fit.sample(c, rng) for c in cells])
    df[TARGET] = new_steps
    return df


def make_replace_oversample(real_df: pd.DataFrame, fit: EBFit, k: int,
                              rng: np.random.Generator) -> pd.DataFrame:
    """Variant B: real rows replicated k times, each with a fresh posterior draw."""
    out = []
    for _ in range(k):
        out.append(make_replace(real_df, fit, rng))
    return pd.concat(out, ignore_index=True)


def make_oversample_only(real_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Variant C: real rows replicated k times, original reward (no smoothing)."""
    return pd.concat([real_df] * k, ignore_index=True)


def write_synth(df: pd.DataFrame, out_root: Path, label: str):
    """Save synth in the layout expected by the pipeline's evaluate stage."""
    gen_dir = out_root / label / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)
    path = gen_dir / "synthetic_data.csv"
    df.to_csv(path, index=False)
    print(f"  [save] {label}: {len(df)} rows → {path}")
    return path


def sanity_report(real: pd.DataFrame, synth: pd.DataFrame, name: str):
    """Quick comparison of marginal stats to verify the synth looks right."""
    real = real[real["avail"] == True]
    synth = synth[synth["avail"] == True]
    print(f"\n  [sanity] {name}:")
    print(f"    rows: real={len(real)}  synth={len(synth)}")
    for s in [0, 1, 2]:
        r = real[real["send"] == s][TARGET]
        y = synth[synth["send"] == s][TARGET]
        print(f"    send={s}: real mean={r.mean():6.1f} std={r.std():5.1f}    "
              f"synth mean={y.mean():6.1f} std={y.std():5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real",  default=str(MONOLOGUE / "data/data_gen.csv"))
    ap.add_argument("--ref",   default=str(MONOLOGUE / "outputs/run3-CoT-no0/"
                                                       "generation/synthetic_data.csv"),
                    help="Reference LLM synth (used to set oversample multiplier).")
    ap.add_argument("--out_dir", default=str(MONOLOGUE / "outputs/bayes_baseline"))
    ap.add_argument("--variant", choices=["replace", "replace_4x", "oversample", "all"],
                    default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] real:  {args.real}")
    real = pd.read_csv(args.real)
    print(f"  real rows: {len(real)}, avail=True: {(real['avail']==True).sum()}")

    print(f"[load] ref (for K):  {args.ref}")
    ref = pd.read_csv(args.ref)
    k = max(1, round(len(ref) / len(real)))
    print(f"  ref synth rows: {len(ref)} → oversample multiplier k = {k}")

    print("\n[fit] empirical-Bayes shrinkage on (slot, loc, steps30pre_bin, send) cells:")
    fit = fit_eb(real)

    out_root = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    variants = (["replace", "replace_4x", "oversample"]
                if args.variant == "all" else [args.variant])

    for v in variants:
        print(f"\n[gen] variant '{v}':")
        if v == "replace":
            synth = make_replace(real, fit, rng)
        elif v == "replace_4x":
            synth = make_replace_oversample(real, fit, k, rng)
        elif v == "oversample":
            synth = make_oversample_only(real, k)

        sanity_report(real, synth, v)
        write_synth(synth, out_root, v)

    print(f"\n[done] All variants written under {out_root}/")
    print("\nNext: run DDQN evaluation on each variant via")
    print("  python -m src.ablations.run_bayes_ablation")


if __name__ == "__main__":
    main()
