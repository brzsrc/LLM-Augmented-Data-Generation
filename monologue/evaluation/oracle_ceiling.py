"""
Estimate the V_hat ceiling under increasingly fine state-conditional policies.

The oracle policy picks the empirical best action per state bucket.
Its expected per-step reward (r̄_oracle) and the implied V (= r̄ / (1-γ))
are computed at several state granularities, so you can see how V grows
with finer state — and where it plateaus due to small-sample noise.

Two estimates are reported per state combo:
  • naive    : pick best action per cell, weight by cell frequency.
               (Optimistic — overfits to noise in small cells.)
  • cv-honest: 5-fold over rows. On train fold pick best action per cell;
               on test fold compute mean reward where action matched the
               prescription. This is the unbiased ceiling.

USAGE
    cd monologue/evaluation
    python oracle_ceiling.py
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../data/data_eval.csv")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--n_bins", type=int, default=4)
    p.add_argument("--min_count", type=int, default=10,
                   help="Per-cell min count required to trust empirical best action.")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--out_dir", default="outputs/coverage_audit")
    return p.parse_args()


def bin_continuous(s, n_bins):
    try:
        return pd.qcut(s, q=n_bins, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=n_bins)


def naive_oracle(df, state_cols, action_col="send", reward_col="reward", min_count=10):
    """Pick best action per state cell; weight by cell frequency."""
    if not state_cols:
        means = df.groupby(action_col)[reward_col].mean()
        return float(means.max()), 1, 0

    # per-(cell, action) stats
    stats = (df.groupby(state_cols + [action_col])[reward_col]
               .agg(["count", "mean"]).reset_index())

    # only trust (cell, action) pairs with >= min_count samples
    eligible = stats[stats["count"] >= min_count].copy()

    # for each cell, pick the eligible action with highest mean
    if len(eligible) == 0:
        return float("nan"), 0, 0
    best_df = (eligible.sort_values(state_cols + ["mean"])
                       .drop_duplicates(subset=state_cols, keep="last"))
    best = best_df.set_index(state_cols)["mean"]

    # cell frequencies (over full df)
    cell_freq = df.groupby(state_cols).size() / len(df)

    # for cells that had no eligible action, fall back to overall best action's mean
    fallback = df.groupby(action_col)[reward_col].mean().max()
    n_sparse = 0
    contributions = []
    for cell_key, freq in cell_freq.items():
        if cell_key in best.index:
            contributions.append(freq * best.loc[cell_key])
        else:
            n_sparse += 1
            contributions.append(freq * fallback)
    r_bar = float(sum(contributions))
    return r_bar, int(len(cell_freq)), n_sparse


def crossval_oracle(df, state_cols, action_col="send", reward_col="reward",
                    min_count=10, n_splits=5, seed=42):
    """Honest oracle via k-fold:
       - Train fold: find best action per cell.
       - Test fold: average reward of test rows where actual action matches the
                    prescription (matching estimator of E[r | s, a*(s)]).
       Average across folds, weighted by # of matched test rows.
    """
    if not state_cols:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scored, weights = [], []
        for tr_idx, te_idx in kf.split(df):
            tr = df.iloc[tr_idx]; te = df.iloc[te_idx]
            best_a = tr.groupby(action_col)[reward_col].mean().idxmax()
            matched = te[te[action_col] == best_a]
            if len(matched):
                scored.append(matched[reward_col].mean())
                weights.append(len(matched))
        if not scored:
            return float("nan")
        return float(np.average(scored, weights=weights))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scored, weights = [], []
    fallback_a_overall = df.groupby(action_col)[reward_col].mean().idxmax()
    for tr_idx, te_idx in kf.split(df):
        tr = df.iloc[tr_idx]; te = df.iloc[te_idx].copy()

        # build prescription on train: cell -> best action (among eligible)
        stats = (tr.groupby(state_cols + [action_col])[reward_col]
                   .agg(["count", "mean"]).reset_index())
        eligible = stats[stats["count"] >= min_count]
        if len(eligible) == 0:
            continue
        prescription_df = (eligible.sort_values(state_cols + ["mean"])
                                  .drop_duplicates(subset=state_cols, keep="last"))
        prescription = prescription_df.set_index(state_cols)[action_col]

        # merge onto test
        te_p = te.merge(prescription.rename("prescribed_a"),
                        left_on=state_cols, right_index=True, how="left")
        # fallback for cells absent from train: use overall best
        te_p["prescribed_a"] = te_p["prescribed_a"].fillna(fallback_a_overall)

        matched = te_p[te_p[action_col] == te_p["prescribed_a"]]
        if len(matched):
            scored.append(matched[reward_col].mean())
            weights.append(len(matched))
    if not scored:
        return float("nan")
    return float(np.average(scored, weights=weights))


def hbar():
    print("=" * 78)


def main():
    args = get_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    # bin continuous features
    for col, n_bins in [("dosage", args.n_bins),
                        ("steps30pre", args.n_bins),
                        ("resp", args.n_bins)]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"{col}_bin"] = bin_continuous(df[col], n_bins)

    # state granularities to evaluate, from coarse to fine
    combos = [
        [],
        ['slot'],
        ['slot', 'weekday'],
        ['slot', 'dosage_bin'],
        ['slot', 'steps30pre_bin'],
        ['slot', 'weather'],
        ['slot', 'weekday', 'dosage_bin'],
        ['slot', 'weekday', 'steps30pre_bin'],
        ['slot', 'dosage_bin', 'steps30pre_bin'],
        ['slot', 'weekday', 'dosage_bin', 'steps30pre_bin'],
        ['slot', 'weekday', 'dosage_bin', 'steps30pre_bin', 'weather'],
    ]
    combos = [[c for c in cc if c in df.columns] for cc in combos]

    inv_disc = 1.0 / (1.0 - args.gamma)
    hbar()
    print(f"ORACLE-V̂ CEILING under increasingly fine state conditioning")
    print(f"  γ = {args.gamma}, multiplier 1/(1-γ) = {inv_disc:.1f}")
    print(f"  min_count per (cell, action) cell = {args.min_count}")
    print(f"  CV honest estimate uses {args.n_splits} folds")
    hbar()
    print(f"{'state_cols':50s} {'#cells':>7s} {'sparse':>7s} "
          f"{'r̄_naive':>9s} {'V_naive':>8s}  {'r̄_cv':>7s} {'V_cv':>7s}")
    print("-" * 110)
    rows = []
    for sc in combos:
        r_naive, n_cells, n_sparse = naive_oracle(df, sc, min_count=args.min_count)
        r_cv = crossval_oracle(df, sc, min_count=args.min_count, n_splits=args.n_splits)
        v_naive = r_naive * inv_disc
        v_cv = r_cv * inv_disc if not np.isnan(r_cv) else float("nan")
        label = "(no state — best constant)" if not sc else "+".join(sc)
        print(f"{label:50s} {n_cells:>7d} {n_sparse:>7d} "
              f"{r_naive:>9.3f} {v_naive:>8.2f}  "
              f"{r_cv:>7.3f} {v_cv:>7.2f}")
        rows.append({"state_cols": "+".join(sc) if sc else "(none)",
                     "n_cells": n_cells, "n_sparse": n_sparse,
                     "r_bar_naive": r_naive, "V_naive": v_naive,
                     "r_bar_cv": r_cv, "V_cv": v_cv})

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "oracle_ceiling.csv"),
                              index=False)
    hbar()
    print(f"  Saved -> {args.out_dir}/oracle_ceiling.csv")

    # interpretation
    print()
    print("INTERPRETATION:")
    print(f"  • r̄_naive grows with finer state -> classic overfit-to-noise pattern.")
    print(f"  • r̄_cv plateaus / flattens at the *honest* ceiling — that's the")
    print(f"    most you can realistically squeeze out by being state-smart.")
    print(f"  • If V_cv barely moves past slot-only, the data does not support")
    print(f"    much state-dependent advantage; ceiling ≈ best constant policy.")


if __name__ == "__main__":
    main()
