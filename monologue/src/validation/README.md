# Synthetic Data Validation — Two-Axis Framework

Synthetic data evaluation splits into two **independent** axes that can — and
empirically do — diverge:

| Axis | Question answered | Decision role | File |
|---|---|---|---|
| **Replication-Fidelity** | "Does synth look like real statistically?" | **Diagnostic only** | `gates_replication.py` |
| **Augmentation-Utility** | "Does synth augmentation help downstream RL?" | **Pass/fail judge** | `gates_augmentation.py` |

A synth dataset can fail the R suite (marginal distribution mismatch) yet
pass the A suite — and produce a better DDQN policy. See `run3-CoT-no0` vs
`run3-no0-replay` for an empirical case where this divergence is large.

---

## Replication-Fidelity Suite (R1–R8)

| Gate | What it measures | Paper |
|---|---|---|
| R1 `univariate_marginal` | Per-column KS / chi-square distribution alignment | Esteban+ 2017; Xu+ 2019 |
| R2 `pairwise_correlation` | Inter-column correlation structure (Frobenius L1) | Standard tabular fidelity |
| R3 `conditional_moments` | Per-action mean/std/skew of target | CTGAN §4 |
| R4 `temporal_dynamics` | STL trend / cycle Kendall τ per pair | Kuo+ 2022 (Health Gym §4.2) |
| R5 `privacy_dcr` | Distance to closest record vs real-to-real baseline | Zhao+ 2022 (CTAB-GAN+) |
| R6 `schema_integrity` | Boundary conformity + avail structural rule | Standard |
| R7 `support_coverage` | Category coverage (CAT) score | Tornqvist+ 2024; Kuo+ 2024 |
| R8 `distribution_shape` | Wasserstein-1 distance for continuous; JSD for categorical | Ramdas+ 2017 |

---

## Augmentation-Utility Suite (A1–A8)

Core gates (`A1`, `A2`, `A4`) MUST pass for suite to PASS overall.

| Gate | What it measures | Paper |
|---|---|---|
| **A1** `conditional_ks` | Per-(state, action) cell KS on target | Esteban+ 2017; Xu+ 2019 |
| **A2** `tstr_binary` | RF AUROC: train synth → predict P(target > 0) on real | Esteban+ 2017 (TSTR) |
| A3 `tstr_regression` | RF RMSE: train synth → predict target value on real | Esteban+ 2017 |
| **A4** `action_coverage` | Sparse cell fill rate, message arms weighted 2× | Kumar+ 2020 (CQL §3); Fujimoto+ 2019 (BCQ) |
| A5 `causal_signal` | Per-state action-effect rank correlation real vs synth | Voloshin+ 2021 (FQE); Bica+ 2021 |
| A6 `q_proxy` | RF Q-proxy disagreement real vs synth on real probes | Voloshin+ 2021 |
| A7 `conditional_diversity` | Per-cell σ_synth / σ_real (no cell-level collapse) | Multi-faceted framework 2024 |
| A8 `off_policy_importance` | Sparse cell synth_n / real_n contribution ratio | Kumar+ 2020 (CQL §5) |

---

## Usage

```python
import pandas as pd
from src.validation import (
    run_replication_suite,
    run_augmentation_suite,
    run_two_axis_validation,
)

real_df  = pd.read_csv("data/data_gen.csv")
synth_df = pd.read_csv("outputs/runX/generation/synthetic_data.csv")

# Independent suites:
rep = run_replication_suite(real_df, synth_df)
aug = run_augmentation_suite(real_df, synth_df)

# Combined runner with explicit accept/reject decision:
report = run_two_axis_validation(real_df, synth_df)
print(report["decision"])   # "accept" iff augmentation suite PASSES
```

Outputs follow the same `{summary, gates}` shape as the legacy
`run_all_gates` runner, so existing JSON consumers keep working.

---

## Empirical motivation

The Haas (2025) Sim-LLM thesis is a clear case of the fidelity-utility
divergence: 2/11 KS tests pass, mean WD=1.136, JSD=0.303 (R suite fails
broadly), yet policies trained on the synth significantly outperform
real-only baselines (p = 5.5×10⁻¹³). The thesis covers 6 of our 8 R gates
but 0 of our 8 A gates — exactly the gap the augmentation suite fills.

See `thesis report §4.x` for the full mapping table and discussion.

---

## Legacy

The original `gates.py` (13 gates) is still imported by `pipeline.py` via
the legacy `run_all_gates`. The new two-axis runners are additive and do
not replace it; both can run in parallel during transition.
