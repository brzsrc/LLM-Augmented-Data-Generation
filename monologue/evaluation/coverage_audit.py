"""
Coverage audit for HeartSteps real data
=======================================
Before generating synthetic data, this script answers:

  1. Which (state, action) combinations are under-represented in real data?
     -> these are the "holes" worth filling with synthetic.
  2. How strong is the state-conditional reward signal?
     -> tells you how much "smart sending" can exceed "best constant"
        (i.e. how far past 33-34 the ceiling can actually go).
  3. Does every patient see every action? Or are some actions missing
     entirely for some patients?

USAGE
-----
    cd monologue/evaluation
    python coverage_audit.py
    python coverage_audit.py --csv ../data/data_eval.csv --min_count 20

OUTPUTS (in --out_dir, default outputs/coverage_audit/)
    sparse_cells.csv         every (feature, value, action) cell below threshold
    audit_summary.csv        per-feature sparseness summary
    slot_weekday_send.csv    slot × weekday × send pivot table
    state_action_reward.csv  reward stats per (slot, send) — the signal map
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../data/data_eval.csv")
    p.add_argument("--out_dir", default="outputs/coverage_audit")
    p.add_argument("--min_count", type=int, default=20,
                   help="Cells with fewer than this many samples are flagged sparse.")
    p.add_argument("--n_bins", type=int, default=4,
                   help="Quantile bins for continuous features.")
    return p.parse_args()


def bin_continuous(s, n_bins):
    """Quantile-bin a numeric series; fall back to equal-width if duplicates."""
    try:
        return pd.qcut(s, q=n_bins, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=n_bins)


def hbar():
    print("=" * 78)


def audit_pair(df, feat, action_col, min_count):
    """Cross-tab feat × action. Returns (table, list of sparse cells)."""
    ct = pd.crosstab(df[feat], df[action_col])
    sparse = []
    for f_val in ct.index:
        for a_val in ct.columns:
            n = int(ct.loc[f_val, a_val])
            if n < min_count:
                sparse.append({"feature": feat, "feat_value": str(f_val),
                               "action": int(a_val), "count": n})
    return ct, sparse


def main():
    args = get_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    n_total = len(df)

    # ----- 1. Overall stats -----
    hbar()
    print(f"DATA: {args.csv}")
    hbar()
    print(f"  rows           : {n_total}")
    print(f"  patients       : {df['uid'].nunique()}")
    print(f"  reward range   : [{df['reward'].min():.3f}, {df['reward'].max():.3f}]  "
          f"mean={df['reward'].mean():.3f}")
    print(f"  action (send) distribution:")
    for a, p in df['send'].value_counts(normalize=True).sort_index().items():
        n = (df['send'] == a).sum()
        print(f"      send={a}  count={n:5d}  ({p*100:5.1f}%)")

    # bin continuous features so they can be cross-tabbed
    for col, n_bins in [("dosage", args.n_bins), ("steps30pre", args.n_bins),
                        ("resp", args.n_bins)]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"{col}_bin"] = bin_continuous(df[col], n_bins)

    # ----- 2. Univariate coverage: each feature × send -----
    hbar()
    print(f"UNIVARIATE coverage  (cells with count < {args.min_count} flagged sparse)")
    hbar()
    feats = ['slot', 'weekday', 'weather', 'temp', 'loc',
             'dosage_bin', 'steps30pre_bin', 'resp_bin']
    feats = [f for f in feats if f in df.columns]
    all_sparse = []
    summary = []
    for feat in feats:
        ct, sparse = audit_pair(df, feat, 'send', args.min_count)
        n_cells = ct.shape[0] * ct.shape[1]
        n_sparse = len(sparse)
        pct_sparse = n_sparse / n_cells * 100 if n_cells else 0
        print(f"\n>>> {feat}  ({n_sparse}/{n_cells} cells sparse, {pct_sparse:.1f}%)")
        print(ct.to_string())
        summary.append({"feature": feat, "n_cells": n_cells,
                        "n_sparse": n_sparse, "pct_sparse": pct_sparse})
        all_sparse.extend(sparse)

    # ----- 3. Bivariate: slot × weekday × send (the most actionable view) -----
    hbar()
    print("BIVARIATE coverage: slot × weekday × send")
    hbar()
    biv = df.groupby(['slot', 'weekday', 'send']).size().unstack('send', fill_value=0)
    print(biv.to_string())
    biv_n_sparse = (biv < args.min_count).sum().sum()
    biv_total = biv.size
    print(f"\n  {biv_n_sparse}/{biv_total} cells < {args.min_count} samples "
          f"({biv_n_sparse/biv_total*100:.1f}% sparse)")
    biv.to_csv(os.path.join(args.out_dir, "slot_weekday_send.csv"))

    # ----- 4. Top sparse cells -----
    hbar()
    print(f"TOP 25 HOLES — highest priority for synthetic generation")
    hbar()
    sparse_df = pd.DataFrame(all_sparse).sort_values("count").reset_index(drop=True)
    if len(sparse_df):
        print(sparse_df.head(25).to_string(index=False))
    else:
        print("  No sparse cells found at this threshold. Coverage is uniform.")
    sparse_df.to_csv(os.path.join(args.out_dir, "sparse_cells.csv"), index=False)

    # ----- 5. State-conditional reward signal (the ceiling diagnosis) -----
    hbar()
    print("STATE-CONDITIONAL REWARD: how much does reward vary by (state, action)?")
    print("  -> if reward varies strongly per state, 'smart sending' has room.")
    print("  -> if reward only varies by action, ceiling = best-constant policy.")
    hbar()
    sa_reward = df.groupby(['slot', 'send'])['reward'].agg(['count', 'mean']).round(3)
    print("\nMean reward per (slot, send):")
    print(sa_reward.to_string())
    sa_reward.to_csv(os.path.join(args.out_dir, "state_action_reward.csv"))

    # diagnostic: per-slot best action and per-slot delta
    print("\nPer-slot diagnosis: which action looks best, and by how much?")
    means_by_slot = df.groupby(['slot', 'send'])['reward'].mean().unstack('send')
    print(means_by_slot.round(3).to_string())
    best_a_per_slot = means_by_slot.idxmax(axis=1)
    max_minus_min = means_by_slot.max(axis=1) - means_by_slot.min(axis=1)
    print(f"\n  Best action per slot: {best_a_per_slot.to_dict()}")
    print(f"  Within-slot reward range (max−min): "
          f"{max_minus_min.round(3).to_dict()}")
    print(f"  Average within-slot spread: {max_minus_min.mean():.3f}")
    print(f"  -> if this avg > 0.3, state-dependent policy has real room above "
          f"the best-constant ceiling.")
    print(f"  -> if < 0.1, the data barely supports state-dependence; ceiling is "
          f"essentially the best constant.")

    # implied V ceiling per policy:
    print("\nImplied V (× 1/(1-γ) at γ=0.95 = × 20):")
    print("  Best constant policy ceiling:")
    g_const = df.groupby('send')['reward'].mean()
    for a, m in g_const.items():
        print(f"    Send a={a}: r̄={m:.3f}  → V ≈ {m*20:.2f}")
    print(f"  Optimal state-conditional policy ceiling (pick best action per slot):")
    # weighted by slot frequency
    slot_freq = df['slot'].value_counts(normalize=True).sort_index()
    opt_per_slot = means_by_slot.max(axis=1)
    # align indices in case some slots are missing
    common = slot_freq.index.intersection(opt_per_slot.index)
    weighted_max = (slot_freq.loc[common] * opt_per_slot.loc[common]).sum()
    print(f"    Weighted optimal r̄ = {weighted_max:.3f}  → V ≈ {weighted_max*20:.2f}")
    gain_over_best_constant = weighted_max - g_const.max()
    print(f"    Gain over best constant = +{gain_over_best_constant:.3f} in r̄  "
          f"= +{gain_over_best_constant*20:.2f} in V")
    print(f"    -> THIS IS THE REALISTIC CEILING for what synthetic data can buy you.")

    # ----- 6. Per-patient coverage -----
    hbar()
    print("PER-PATIENT action coverage (does every patient see every action?)")
    hbar()
    per_pat = df.groupby(['uid', 'send']).size().unstack('send', fill_value=0)
    actions = sorted(df['send'].unique())
    for a in actions:
        n_zero = int((per_pat[a] == 0).sum())
        n_lt5 = int((per_pat[a] < 5).sum())
        print(f"  send={a}: {n_zero}/{len(per_pat)} patients have 0 samples; "
              f"{n_lt5}/{len(per_pat)} have <5")
    print("\n  -> patients with 0 samples for an action will have unreliable Q-estimates "
          "for that action; consider per-patient synthetic to fill these.")
    per_pat.to_csv(os.path.join(args.out_dir, "per_patient_action_counts.csv"))

    # ----- 7. Save summary + recommendations -----
    hbar()
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "audit_summary.csv"),
                                 index=False)
    print(f"OUTPUTS written to {args.out_dir}/")
    for fn in ["sparse_cells.csv", "audit_summary.csv", "slot_weekday_send.csv",
               "state_action_reward.csv", "per_patient_action_counts.csv"]:
        print(f"  • {fn}")

    # ----- Final printable recommendations -----
    hbar()
    print("RECOMMENDATIONS for synthetic data design")
    hbar()
    total_sparse = sum(s["n_sparse"] for s in summary)
    if total_sparse:
        # top 3 sparsest features
        top_feat = sorted(summary, key=lambda x: -x["pct_sparse"])[:3]
        print(f"  • {total_sparse} univariate cells are sparse. Highest-priority features:")
        for s in top_feat:
            print(f"      - {s['feature']}: {s['n_sparse']}/{s['n_cells']} sparse "
                  f"({s['pct_sparse']:.1f}%)")
        print(f"  • For synthetic generation, oversample these feature values with "
              f"the under-represented actions.")
    else:
        print("  • No major coverage holes by univariate analysis. Focus on the "
              "bivariate (slot × weekday × send) view.")
    print(f"  • Realistic V ceiling from this data ≈ {weighted_max*20:.1f}  "
          f"(vs best constant {g_const.max()*20:.1f}).")
    print(f"  • The synthetic data's main job is to reduce variance so DDQN reliably "
          f"picks the best action per state — not to create new signal.")


if __name__ == "__main__":
    main()
