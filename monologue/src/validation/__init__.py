"""Synthetic data validation — two-axis framework.

Axes:
  - Replication-Fidelity (gates_replication.run_replication_suite) — diagnostic
  - Augmentation-Utility  (gates_augmentation.run_augmentation_suite) — judge

Legacy single-suite runner (gates.run_all_gates) remains for backward compat.
"""
from src.validation.gates import run_all_gates  # legacy
from src.validation.gates_replication import run_replication_suite
from src.validation.gates_augmentation import run_augmentation_suite


def run_two_axis_validation(real_df, synth_df, personas=None):
    """Run both suites; return a combined report.

    Decision rule: synth is acceptable iff the Augmentation suite passes.
    Replication suite is reported as diagnostic only.
    """
    rep = run_replication_suite(real_df, synth_df)
    aug = run_augmentation_suite(real_df, synth_df)
    return {
        "decision":      "accept" if aug["summary"]["overall_pass"] else "reject",
        "replication":   rep,
        "augmentation":  aug,
    }
