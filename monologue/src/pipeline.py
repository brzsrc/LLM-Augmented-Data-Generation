"""End-to-end orchestrator (YAML-free).

Stages:
  1. Load + preprocess data (driven by src/config.py constants)
  2. Audit:  leakage detector → coverage → oracle ceiling → signal extractor
  3. Persona extraction (5-part PersonaProfile per real patient)
  4. Archetype classification + twin/sibling/edge variant construction
  5. LLM generation (vLLM or stub)
  6. Validation: 5 quality gates
  7. Evaluation: ablation (real-only vs real+synth, with CQL sweep)

Usage:
    cd monologue
    python -m src.pipeline --stage audit
    python -m src.pipeline --backend stub --stage all
    python -m src.pipeline --backend vllm --stage all
(Evaluation hyperparameters — cql_alphas, n_folds, ddqn_iters, ... — live in
cfg.EVAL; edit `src/config.py` to change them.)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src import config as cfg
from src import data_loader
from src.audit import leakage_detector, coverage, oracle_ceiling, signal_extractor
from src.personas import extractor as persona_extractor
from src.personas import archetype as persona_archetype
from src.generation import trajectory_sampler
from src.generation.llm import StubLLM
from src.validation import gates as validation_gates
# NOTE: `src.evaluation.runner` is imported lazily inside stage 7 because it
# pulls in torch (needed only for the evaluate stage).


STAGES = ["audit", "personas", "generate", "validate", "evaluate", "all"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="src/outputs/run3")
    p.add_argument("--backend", choices=["qwen8b", 'qwen32b', "stub"], default="qwen32b")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_uids", type=int, default=None,
                   help="Smoke-test mode: keep only the first N unique uids "
                        "(sorted) from the loaded data. Default = all.")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_root, exist_ok=True)

    print("=" * 72)
    print(f"PIPELINE  stage={args.stage}  backend={args.backend}")
    print(f"  csv = {cfg.CSV_PATH}")
    print(f"  state features ({len(cfg.STATE_FEATURES)}): {cfg.STATE_FEATURES}")
    print("=" * 72)

    # ---- 1. Load -----
    print("\n[stage 1/7] LOAD DATA")
    df = data_loader.load()
    if args.max_uids is not None:
        keep = sorted(df["uid"].unique())[:args.max_uids]
        df = df[df["uid"].isin(keep)].reset_index(drop=True)
        print(f"[loader] --max_uids={args.max_uids}: kept uids {keep} "
              f"→ {len(df)} rows, {df['uid'].nunique()} patients")

    audit_out = os.path.join(args.out_root, "audit")
    persona_out = os.path.join(args.out_root, "personas")
    gen_out = os.path.join(args.out_root, "generation")
    eval_out = os.path.join(args.out_root, "evaluation")
    for d in (audit_out, persona_out, gen_out, eval_out):
        os.makedirs(d, exist_ok=True)

    # ---- 2. Audit -----
    if args.stage in ("audit", "all"):
        print("\n[stage 2/7] AUDIT")
        leakage_detector.detect_leakage(df, out_dir=audit_out)
        coverage.audit_coverage(df, out_dir=audit_out, min_count=20)
        oracle_ceiling.estimate_ceiling(df, out_dir=audit_out)
        signal = signal_extractor.extract_state_action_signal(df, out_dir=audit_out)
        with open(os.path.join(audit_out, "state_action_signal.json"), "w") as f:
            json.dump(signal, f, indent=2)

    # ---- 3 + 4. Personas + archetypes -----
    if args.stage in ("personas", "generate", "validate", "evaluate", "all"):
        print("\n[stage 3/7] EXTRACT PERSONAS")
        real_profiles = persona_extractor.extract_all(df)
        # Classify archetype upfront so real_profiles.json includes it
        for p in real_profiles.values():
            p.anchor.archetype = persona_archetype.classify(p)
        with open(os.path.join(persona_out, "real_profiles.json"), "w") as f:
            json.dump({uid: p.to_dict() for uid, p in real_profiles.items()},
                      f, indent=2, default=str, allow_nan=False)
        # Report archetype distribution
        from collections import Counter
        arch_counts = Counter(p.anchor.archetype for p in real_profiles.values())
        print(f"  Archetype distribution: {dict(arch_counts)}")

        print("\n[stage 4/7] CLASSIFY + BUILD VARIANTS")
        synth_personas = persona_archetype.build_synth_personas(real_profiles,
                                                                  seed=args.seed)
        with open(os.path.join(persona_out, "synth_personas.json"), "w") as f:
            json.dump([p.to_dict() for p in synth_personas], f, indent=2,
                       default=str, allow_nan=False)
    else:
        real_profiles = synth_personas = None

    # ---- 5. Generate -----
    synth_df = None
    if args.stage in ("generate", "validate", "evaluate", "all"):
        print("\n[stage 5/7] GENERATE")
        if args.backend == "stub":
            llm = StubLLM()
        elif args.backend == "qwen8b":
            from src.generation.llm import Qwen8BLLM  # lazy: vllm not installed on CPU/CI
            llm = Qwen8BLLM()
        elif args.backend == "qwen32b":
            from src.generation.llm import Qwen32BLLM  # lazy: vllm not installed on CPU/CI
            llm = Qwen32BLLM() 
        else:
            raise ValueError(f"Unknown LLM backend: {args.backend}")
        # convert dataclasses to dicts for downstream consumer
        synth_personas_dicts = [_persona_to_flat_dict(p) for p in synth_personas]
        synth_df = trajectory_sampler.generate_all_vllm(synth_personas_dicts, llm,
                                                    out_dir=gen_out, seed=args.seed)
        # synth CSV on disk keeps string categoricals (matches data_gen.csv).
        # In memory we encode them so validation / ablation see the same dtypes
        # as real `df` (which already went through add_derived_features at load).
        synth_df = data_loader.add_derived_features(synth_df)

    # ---- 6. Validate -----
    if args.stage in ("validate", "evaluate", "all"):
        print("\n[stage 6/7] VALIDATE")
        gate_results = validation_gates.run_all_gates(df, synth_df, synth_personas_dicts)
        with open(os.path.join(gen_out, "gate_results.json"), "w") as f:
            json.dump(gate_results, f, indent=2, default=str)
            
    # ---- 7. Evaluate -----
    if args.stage in ("evaluate", "all"):
        print("\n[stage 7/7] EVALUATE (ablation)")
        from src.evaluation import runner as eval_runner   # lazy: needs torch
        eval_runner.run_ablation(df, synth_df, out_root=eval_out)

    print(f"\n✅ DONE. Outputs under: {args.out_root}")


def _persona_to_flat_dict(persona):
    """Flatten PersonaProfile to the dict shape trajectory_sampler / prompt_builder
    expects. NO attitude fields (resp-based)."""
    return {
        "source_uid": persona.anchor.source_uid,
        "synth_uid": persona.anchor.synth_uid,
        "variant_type": persona.anchor.variant_type,
        "archetype": persona.anchor.archetype,
        "slot_1_hour": persona.anchor.slot_1_hour,
        "n_days": persona.anchor.n_days,
        "borrowed_uids": persona.anchor.borrowed_uids,
        # Availability (trajectory_sampler pre-decides avail per slot)
        "avail_rate": persona.lifestyle.avail_rate,
        "avail_by_slot": persona.lifestyle.avail_by_slot,
        "avail_by_loc": persona.lifestyle.avail_by_loc,
        "unavail_triggers": persona.lifestyle.unavail_triggers,
        "location_distribution": persona.lifestyle.weekday_loc_dist,
        # Context — slot-1 marginals + within-day slot→slot transitions
        "weather_dist": persona.context.weather_dist,
        "temp_dist": persona.context.temp_dist,
        "weather_transition": persona.context.weather_transition,
        "temp_transition": persona.context.temp_transition,
        # Lifestyle loc — weekday/weekend split for both dist and transition
        "weekday_loc_dist": persona.lifestyle.weekday_loc_dist,
        "weekend_loc_dist": persona.lifestyle.weekend_loc_dist,
        "weekday_loc_transition": persona.lifestyle.weekday_loc_transition,
        "weekend_loc_transition": persona.lifestyle.weekend_loc_transition,
        # Activity — keys mirror the nested schema path.
        # steps10.avail_true (message-eligible — carries per-action signal)
        "steps10_avail_true_mean":               persona.activity.steps10.avail_true.mean,
        "steps10_avail_true_zero_pct":           persona.activity.steps10.avail_true.zero_pct,
        "steps10_avail_true_per_slot_mean":      persona.activity.steps10.avail_true.per_slot_mean,
        "steps10_avail_true_per_slot_zero_pct":  persona.activity.steps10.avail_true.per_slot_zero_pct,
        "steps10_avail_true_per_slot_mean_positive":         persona.activity.steps10.avail_true.per_slot_mean_positive,
        "steps10_avail_true_per_slot_action_mean_positive":  persona.activity.steps10.avail_true.per_slot_action_mean_positive,
        "steps10_avail_true_zero_pct_by_s30_bin":            persona.activity.steps10.avail_true.zero_pct_by_s30_bin,
        # steps10.avail_false (unreachable baseline; no per-action — send forced=0)
        "steps10_avail_false_mean":          persona.activity.steps10.avail_false.mean,
        "steps10_avail_false_zero_pct":      persona.activity.steps10.avail_false.zero_pct,
        "steps10_avail_false_per_slot_mean": persona.activity.steps10.avail_false.per_slot_mean,
        "steps10_avail_false_per_slot_zero_pct":     persona.activity.steps10.avail_false.per_slot_zero_pct,
        "steps10_avail_false_per_slot_mean_positive":persona.activity.steps10.avail_false.per_slot_mean_positive,
        "steps10_avail_false_zero_pct_by_s30_bin":   persona.activity.steps10.avail_false.zero_pct_by_s30_bin,
        # steps10.all (marginal: avail=True+False union) — default view, no _all_ prefix
        "steps10_mean":          persona.activity.steps10.all.mean,
        "steps10_mean_positive": persona.activity.steps10.all.mean_positive,
        "steps10_median":        persona.activity.steps10.all.median,
        "steps10_zero_pct":      persona.activity.steps10.all.zero_pct,
        "steps10_per_slot_mean": persona.activity.steps10.all.per_slot_mean,
        "steps10_by_loc_positive":                  persona.activity.steps10.all.by_loc_positive,
        "steps10_by_weather_positive":              persona.activity.steps10.all.by_weather_positive,
        "steps10_by_temp_positive":                 persona.activity.steps10.all.by_temp_positive,
        "steps10_by_steps30pre_bin_positive":       persona.activity.steps10.all.by_steps30pre_bin_positive,
        "steps10_momentum_score_positive":          persona.activity.steps10.all.momentum_score_positive,
        "steps10_momentum_pair_pct":                persona.activity.steps10.all.momentum_pair_pct,
        "steps10_mean_after_zero_streak":           persona.activity.steps10.all.steps10_mean_after_zero_streak,
        "steps10_context_sensitivity_positive":     persona.activity.steps10.all.context_sensitivity_positive,
        "steps10_high_activity_contexts_positive":  persona.activity.steps10.all.high_activity_contexts_positive,
        # steps30pre profile (sampler inputs)
        "steps30pre_mean":              persona.activity.steps30pre.mean,
        "steps30pre_median":            persona.activity.steps30pre.median,
        "steps30pre_zero_pct":          persona.activity.steps30pre.zero_pct,
        "steps30pre_per_slot_mean":     persona.activity.steps30pre.per_slot_mean,
        "steps30pre_per_slot_zero_pct": persona.activity.steps30pre.per_slot_zero_pct,
        "steps30pre_sigma_log":         persona.activity.steps30pre.sigma_log,
        "steps30pre_bin_edges":         persona.activity.steps30pre.bin_edges,
        "steps30pre_bin_labels":        persona.activity.steps30pre.bin_labels,
        # Derived helpers
        "peak_slot": (max(persona.activity.steps10.all.per_slot_mean,
                           key=persona.activity.steps10.all.per_slot_mean.get)
                      if persona.activity.steps10.all.per_slot_mean else None),
        # Engagement-phase header (Park 2023): {name, day_range, activity_mult}
        # prompt_builder picks the phase matching current_state.day at decision time.
        "compliance_phases": [
            {"name": ph.name, "day_range": ph.day_range, "activity_mult": ph.activity_mult}
            for ph in (persona.compliance.phases if persona.compliance else [])
        ],
    }


if __name__ == "__main__":
    main()
