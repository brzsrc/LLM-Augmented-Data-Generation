# `src/` — LLM-Augmented Offline RL Framework

Audit-first synthetic data pipeline for offline RL on small healthcare datasets.

## ⚡ Two generation pipelines — which to use

The codebase contains **two** synthetic-data pipelines that coexist; pick based on
your goal:

| Pipeline | Entry point | When to use |
|---|---|---|
| 🆕 **Audit-first (new)** | `python -m src.pipeline` | Default. Produces synthetic data **conditioned on a per-persona archetype** with `source_uid` tagging, leakage detection, and 5 quality gates. Designed for K-fold cross-fit evaluation. |
| 🗂️ **MONOLOGUE legacy** | `python -m src.generation.step1_build_persona` → `step2_predict_rows_parallel` → `step3_merge_csv` | Original workflow that generates a **per-row monologue + reasoning + post30 step prediction**. Use when you specifically want the rich textual MONOLOGUE artifact, or to reproduce the existing `ablation_test.py` results. |

The two share infrastructure (`llm.py`, `common.py`, prompts/), so improvements to
LLM wrappers benefit both. They differ in *what gets generated*:

- **New pipeline** generates **trajectories** (steps10 per decision) anchored to
  archetype-derived personas — these go straight into DDQN training as
  augmented MDP transitions.
- **Legacy pipeline** generates **monologue + reasoning + post30** per row of
  the original CSV — a richer, more interpretable output, but heavier per row
  and not tagged for K-fold leakage prevention.

If unsure: start with `python -m src.pipeline --backend stub --stage audit`
and only fall back to the legacy steps if you need the per-row monologue text.

---


## Why this exists

Standard offline RL (DDQN/CQL) on small datasets (37 patients, 5663 transitions)
suffers from:
1. Q-extrapolation: vanilla DDQN picks OOD actions with virtually-high Q
2. Run-to-run instability: V̂ swings 10+ across runs with same seed
3. State leakage: features like `resp` causally depend on the current action
4. OPE bias: FQE overestimates constant policies that match data abundance

This framework addresses these by:
- **Audit-first**: detect leakage, measure coverage holes, estimate honest
  ceiling BEFORE any synthesis
- **Per-persona LLM generation** with archetype-aware variants (twin / sibling
  / edge) — each carrying `source_uid` for K-fold leakage prevention
- **Five quality gates** before training
- **Cross-fit evaluation** with paired-difference CIs (no Wilcoxon-on-bootstrap)

## Layout

```
src/
├── configs/                Dataset & archetype YAML (declarative)
│   ├── dataset_heartsteps.yaml
│   └── archetypes.yaml
├── core/                   Dataset-agnostic loaders
│   ├── data_loader.py      Reads config → adds derived features (hour, dosage)
│   ├── transition_builder.py  Builds MDP (s,a,r,s') with source_uid tags
│   └── state_encoder.py    MinMaxScaler wrapper
├── audit/                  All run BEFORE synthesis
│   ├── leakage_detector.py Auto-find future-leakage state columns
│   ├── coverage.py         (state, action) sparse-cell census
│   ├── oracle_ceiling.py   Honest V̂ ceiling via CV best-action
│   └── signal_extractor.py Per-slot-action reward → fed into prompts
├── personas/
│   ├── extractor.py        Per-patient behavioral fingerprint
│   └── archetype.py        Classify + 3 variant profiles per source
├── generation/
│   ├── llm.py              Qwen3BLLM / Qwen32BLLM vLLM wrappers (moved from monologue/)
│   ├── common.py           Shared helpers: UserState, prompt utilities (moved from monologue/)
│   ├── llm_provider.py     Backend factory: vLLM (production) or stub (CI)
│   ├── prompt_builder.py   Data-driven step prompts (NOT LLM defaults)
│   ├── trajectory_sampler.py    Per-persona, per-decision LLM calls (new pipeline)
│   ├── step1_build_persona.py   Legacy: persona MD generation (moved from monologue/)
│   ├── step2_predict_rows.py    Legacy: serial row predictor (moved from monologue/)
│   ├── step2_predict_rows_parallel.py  Legacy: parallel version (moved from monologue/)
│   └── step3_merge_csv.py       Legacy: merge predictions into CSV (moved from monologue/)
├── validation/
│   └── gates.py            5 quality gates (distribution / coverage / signal /
│                           leakage / consistency)
├── evaluation/
│   ├── kfold_runner.py     Wraps monologue/evaluation/policy_utility_kfold.py
│   ├── ablation.py         real-only vs real+synth, with CQL sweep
│   └── ablation_test.py    Legacy: MONOLOGUE / REASONING ablation (moved from monologue/)
└── pipeline.py             End-to-end orchestrator
```

## Quick start

### 1. Syntax / E2E sanity (Mac, no GPU, ~30 sec):
```bash
cd monologue
python -m src.pipeline \
    --config src/configs/dataset_heartsteps.yaml \
    --backend stub \
    --stage audit          # just the audit stage
```

### 2. Full pipeline with stub LLM (no real generation, but full mechanics):
```bash
python -m src.pipeline \
    --config src/configs/dataset_heartsteps.yaml \
    --backend stub --stage all
```

### 3. Production run with vLLM Qwen (autoDL, 2 GPUs):
```bash
python -m src.pipeline \
    --config src/configs/dataset_heartsteps.yaml \
    --backend vllm \
    --stage all \
    --cql_alphas 0.0,1.0,3.0
```

### 4. Single stage:
```bash
python -m src.pipeline --config ... --stage audit       # just audit
python -m src.pipeline --config ... --stage generate    # need stage 3 cached
```

## Outputs

```
src/outputs/run1/
├── audit/
│   ├── leakage_report.csv          # state cols flagged as leakage
│   ├── coverage_summary.csv        # per-feature sparseness
│   ├── sparse_cells.csv            # specific holes to fill
│   ├── oracle_ceiling.csv          # V̂ ceiling at increasing state granularity
│   └── state_action_signal.json    # per-slot-action reward (fed to prompts)
├── personas/
│   ├── real_profiles.json          # per-patient extracted profile
│   └── synth_personas.json         # 3 variants × N real = synth personas
├── generation/
│   ├── synthetic_data.csv          # generated trajectories
│   └── gate_results.json           # 5 quality gates pass/fail
└── evaluation/
    └── ablation_summary.csv        # real-only vs real+synth Original V̂
```

## How to extend to a new dataset

1. Write `configs/dataset_<your_dataset>.yaml` declaring:
   - columns mapping (patient_id, action, reward_source, etc.)
   - state features (and `forbidden_in_state` for known leakage cols)
   - persona attributes to extract
2. Optionally adapt `configs/archetypes.yaml` for new patient subtypes
3. Run pipeline — no code changes needed.

The only domain-aware files are `configs/*.yaml`. The rest is dataset-agnostic.

## Legacy pipeline (step1–step3)

The original MONOLOGUE-style pipeline has been moved under `src/generation/`
and can still be invoked as before, just with the new module path:

```bash
cd monologue
python -m src.generation.step1_build_persona --uids 1 11 37
python -m src.generation.step2_predict_rows_parallel --uids-file train_uids.json
python -m src.generation.step3_merge_csv --uids 1 11 37
```

The MONOLOGUE / REASONING ablation script is now
`src/evaluation/ablation_test.py`. See `src/README_legacy.md` for the original
workflow description.

## Key design decisions documented

| Choice | Why |
|--------|-----|
| `source_uid` tag on every synth row | Enables K-fold leakage prevention; never lose this tag |
| LLM generation per-transition (not per-trajectory) | Avoids compound error (Small Dataset Big Gains insight) |
| Prompt embeds per-slot data signal | Replaces LLM prior with empirical truth |
| Audit-first | Don't generate when oracle ceiling shows < 3 V̂ room |
| 3 variant types (twin/sibling/edge) | Balance variance reduction vs coverage filling |
| Wrapping (not rewriting) policy_utility_kfold.py | Don't duplicate working DDQN/FQE code |

## References

This framework borrows mechanisms from:
- **Patient-Zero** (arxiv 2509.11078) — hierarchical synthesis + dual-track memory + NLI verification
- **Small Dataset, Big Gains** (arxiv 2312.09844) — single-step augmentation + critic value monitoring
- **GuDA** (arxiv 2310.18247) — guided sampling + algorithm-agnostic interface
- **MORAL** (arxiv 2503.20285) — confidence-weighted training
- **CQL** (arxiv 2006.04779) — conservative Q-learning
- **HeartSteps** (PMC8439432) — domain-specific MRT design + dosage variable

Differentiation: combines audit-first + persona-level + cross-fit OPE in a way
no existing framework does, specifically for medical/behavioral offline RL.
