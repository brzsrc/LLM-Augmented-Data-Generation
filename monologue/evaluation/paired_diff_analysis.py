"""
Paired bootstrap difference analysis
====================================
Reads boot_values.pkl (produced by policy_utility_baseline.py) and reports, for
each policy pair, the *paired* bootstrap difference instead of a Wilcoxon test:

    diff_b = V_A(b) - V_B(b)        (paired by bootstrap index b)

For each ordered pair (A, B) it prints:
    P(A > B)           — fraction of bootstrap resamples where A beats B
    median diff        — typical size of the gap
    95% CI of diff     — [2.5%, 97.5%] percentile of the diffs
    crosses 0?         — if yes -> no detectable difference

Why this instead of Wilcoxon-on-bootstrap:
  Wilcoxon treats B (#bootstraps) as a sample size, so its p-value shrinks as
  you raise B (spurious). P(diff>0) and the diff-CI converge to stable values.

USAGE
-----
    python paired_diff_analysis.py --pkl outputs/baseline/boot_values.pkl
"""
import argparse
import pickle
import numpy as np
import pandas as pd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default="outputs/baseline/boot_values.pkl")
    p.add_argument("--out_csv", default=None,
                   help="Optional path to save the pairwise table as CSV.")
    return p.parse_args()


def paired_diff(a, b):
    """diff = a - b over bootstrap indices where both are non-nan."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    p_gt = float(np.mean(d > 0))
    med = float(np.median(d))
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    crosses0 = (lo < 0.0 < hi)
    return {"n": int(m.sum()), "P(A>B)": p_gt, "median_diff": med,
            "ci_low": lo, "ci_high": hi, "crosses_0": crosses0}


def main():
    args = get_args()
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    bv = data["boot_values"]
    names = list(bv.keys())

    print(f"\nLoaded {args.pkl}")
    print(f"Policies ({len(names)}), B={len(next(iter(bv.values())))}\n")
    print("Per-policy bootstrap medians:")
    for n in names:
        arr = np.asarray(bv[n], dtype=float)
        print(f"    {n:25s} median={np.nanmedian(arr):8.3f}  "
              f"valid={int((~np.isnan(arr)).sum())}")

    # All ordered pairs (A vs B), A higher-median first for readability.
    med = {n: np.nanmedian(np.asarray(bv[n], dtype=float)) for n in names}
    order = sorted(names, key=lambda n: med[n], reverse=True)

    rows = []
    print("\n" + "=" * 78)
    print("PAIRWISE PAIRED-DIFFERENCE  (A vs B,  diff = A - B)")
    print("=" * 78)
    for i, A in enumerate(order):
        for B in order[i + 1:]:
            r = paired_diff(bv[A], bv[B])
            verdict = ("NO detectable diff (CI crosses 0)" if r["crosses_0"]
                       else f"{A} reliably > {B}")
            print(f"\n  {A}  vs  {B}")
            print(f"    P({A.split()[0]}>{B.split()[0]}) = {r['P(A>B)']:.3f}"
                  f"   median diff = {r['median_diff']:+.3f}")
            print(f"    95% CI of diff = [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                  f"   ->  {verdict}")
            rows.append({"A": A, "B": B, **r})

    df = pd.DataFrame(rows)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved table -> {args.out_csv}")
    print()


if __name__ == "__main__":
    main()
