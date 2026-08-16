# FCUT-VLA Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible SmolVLA-based FedLIBERO-Fail pipeline that tests failure-conditioned transfer-utility ranking, personalized LoRA repair, and retention-gated global consolidation, while maintaining a submission-ready manuscript without fabricated results.

**Architecture:** The project separates benchmark manifests, client adapters, failure contexts, counterfactual utility labels, ranking models, repair/consolidation policies, evaluation, and manuscript generation. Mac-compatible synthetic and tiny-data tests gate every component before manually dispatched RTX 4090 jobs. Results are imported from immutable run manifests into tables and figures rather than copied manually.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face LeRobot/SmolVLA, LIBERO, PEFT LoRA, Hydra/OmegaConf or YAML configurations, pytest, NumPy, pandas, scipy, statsmodels, matplotlib/seaborn, Weights & Biases optional, LaTeX with BibTeX.

## Global Constraints

- Primary backbone: SmolVLA with official LeRobot/LIBERO integration.
- Training GPU: one RTX 4090 with 24 GB VRAM; all GPU commands must be resumable and explicitly supplied to the researcher.
- Local development: MacBook M2; every component requires CPU synthetic tests before a GPU job.
- Client knowledge: LoRA or low-rank deltas trained from an identical base checkpoint and identical target-module configuration.
- Task IDs and source-client identities must never be utility-ranker inputs.
- Raw client trajectories must not be uploaded to the simulated server interface.
- Repair is personalized first; global consolidation requires target-gain and retention gates.
- All counterfactual comparisons use identical target checkpoints and paired environment initial-state seeds.
- Main results require at least three independent training seeds and bootstrap 95% confidence intervals.
- Manuscript text must not contain invented numeric results; unresolved empirical statements use explicit bracketed result slots.
- SO-101 and reduced OpenVLA-OFT experiments begin only after SmolVLA Gates A-D pass.

---

## File map

- `pyproject.toml`: package metadata and development dependencies.
- `src/fcut_vla/types.py`: immutable identifiers and typed records shared across modules.
- `src/fcut_vla/config.py`: configuration loading and validation.
- `src/fcut_vla/benchmark/manifest.py`: task/client/stage split manifest and leakage checks.
- `src/fcut_vla/benchmark/failure_context.py`: construction and serialization of failure windows.
- `src/fcut_vla/adapters/bank.py`: adapter metadata bank and privacy-filtered descriptor interface.
- `src/fcut_vla/adapters/composition.py`: top-1 and sparse top-K LoRA composition.
- `src/fcut_vla/utility/counterfactual.py`: paired counterfactual utility-label computation.
- `src/fcut_vla/utility/ranker.py`: decomposed gain/risk/uncertainty ranker.
- `src/fcut_vla/repair/personalized.py`: personalized repair orchestration and abstention.
- `src/fcut_vla/repair/consolidation.py`: retention gate and reversible global promotion.
- `src/fcut_vla/evaluation/metrics.py`: retrieval, continual, recovery, and safety metrics.
- `src/fcut_vla/evaluation/statistics.py`: paired confidence intervals and hypothesis tests.
- `scripts/smoke_test.py`: Mac CPU end-to-end synthetic smoke test.
- `scripts/train_client_adapter.py`: RTX 4090 local-client SmolVLA LoRA training entry point.
- `scripts/generate_failures.py`: deterministic LIBERO rollout and failure-context generation.
- `scripts/build_utility_labels.py`: counterfactual candidate evaluation.
- `scripts/train_utility_ranker.py`: ranker training and validation.
- `scripts/run_repair_eval.py`: personalized repair and retention evaluation.
- `scripts/run_continual_experiment.py`: stage-wise federated continual orchestration.
- `scripts/export_paper_results.py`: verified tables/figures and result macros.
- `configs/`: frozen YAML configurations for smoke, development, main, and ablation runs.
- `manifests/fedlibero_fail_v1.yaml`: immutable task/client/stage split.
- `tests/`: unit and integration tests mirroring package modules.
- `paper/main.tex`: manuscript entry point.
- `paper/sections/*.tex`: focused manuscript sections.
- `paper/references.bib`: verified bibliography.
- `paper/tables/` and `paper/figures/`: generated artifacts only.
- `paper/RESULT_SLOTS.md`: registry of every unresolved empirical claim.
- `paper/draft.md`: readable initial full-paper draft before LaTeX conversion.

---

### Task 1: Manuscript skeleton and claim registry

**Files:**
- Create: `paper/draft.md`
- Create: `paper/RESULT_SLOTS.md`
- Create: `paper/references.bib`
- Create: `paper/figures/README.md`
- Create: `paper/tables/README.md`
- Test: `tests/test_paper_claims.py`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-08-16-fcut-vla-paper-design.md` and downloaded primary papers.
- Produces: stable result-slot identifiers such as `RESULT:RQ1_NDCG_MAIN` used by later result export.

- [ ] **Step 1: Write a failing manuscript-integrity test**

Create `tests/test_paper_claims.py` to assert that the draft contains all required sections, every empirical superiority sentence contains a `RESULT:` slot, prohibited novelty phrases are absent, and every citation key used in the draft exists in `paper/references.bib`.

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_paper_claims.py -v`

Expected: FAIL because the paper files do not exist.

- [ ] **Step 3: Write the complete result-free draft and bibliography**

Write title, abstract with explicit result slots, introduction, related work, problem formulation, method, FedLIBERO-Fail protocol, experiment plan, limitations, and conclusion. Register each slot with the producing experiment, metric, and acceptance criterion.

- [ ] **Step 4: Run manuscript-integrity tests**

Run: `python -m pytest tests/test_paper_claims.py -v`

Expected: PASS with no prohibited novelty phrases and no unregistered empirical claims.

- [ ] **Step 5: Commit manuscript skeleton**

Run:

```bash
git add paper tests/test_paper_claims.py
git commit -m "docs: draft FCUT-VLA manuscript and claim registry"
```

### Task 2: Typed project skeleton and validated experiment configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/fcut_vla/__init__.py`
- Create: `src/fcut_vla/types.py`
- Create: `src/fcut_vla/config.py`
- Create: `configs/smoke.yaml`
- Create: `configs/development.yaml`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `ClientStageId`, `FailureId`, `AdapterId`, `SeedSet`, and `load_config(path: Path) -> ExperimentConfig`.

- [ ] **Step 1: Write failing configuration tests**

Test valid loading, rejection of task IDs as ranker features, rejection of unequal paired seed sets, rejection of non-positive LoRA ranks, and stable serialization hashes.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL with missing `fcut_vla.config`.

- [ ] **Step 3: Implement the typed configuration boundary**

Use frozen dataclasses or Pydantic models. `ExperimentConfig` must expose `config_hash()` and must validate the Global Constraints above.

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit project skeleton**

```bash
git add pyproject.toml src configs tests/test_config.py
git commit -m "feat: add validated FCUT-VLA experiment configuration"
```

### Task 3: FedLIBERO-Fail manifest and leakage audit

**Files:**
- Create: `src/fcut_vla/benchmark/manifest.py`
- Create: `manifests/fedlibero_fail_v1.yaml`
- Create: `tests/benchmark/test_manifest.py`
- Create: `docs/fedlibero_fail_protocol.md`

**Interfaces:**
- Consumes: `ExperimentConfig`.
- Produces: `FedLiberoManifest`, `TaskStage`, `FailureStratum`, `validate_no_leakage(manifest) -> LeakageReport`.

- [ ] **Step 1: Write failing manifest tests**

Test four ordered stages per client, held-out task-skill combinations, held-out client assignments, disjoint initial-state seeds, presence of seen/compositional/no-match strata, and absence of client ownership in model features.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/benchmark/test_manifest.py -v`

Expected: FAIL with missing benchmark module.

- [ ] **Step 3: Implement manifest parsing and leakage validation**

Encode an explicit skill taxonomy and client-stage mapping. Fail closed when any train/test task-skill pair or initial-state seed overlaps illegally.

- [ ] **Step 4: Run manifest tests and print the frozen manifest hash**

Run: `python -m pytest tests/benchmark/test_manifest.py -v && python -m fcut_vla.benchmark.manifest manifests/fedlibero_fail_v1.yaml`

Expected: PASS and one SHA-256 manifest hash.

- [ ] **Step 5: Commit protocol and manifest**

```bash
git add src/fcut_vla/benchmark manifests docs/fedlibero_fail_protocol.md tests/benchmark
git commit -m "feat: define leakage-safe FedLIBERO-Fail protocol"
```

### Task 4: Failure context and privacy-filtered adapter bank

**Files:**
- Create: `src/fcut_vla/benchmark/failure_context.py`
- Create: `src/fcut_vla/adapters/__init__.py`
- Create: `src/fcut_vla/adapters/bank.py`
- Create: `tests/benchmark/test_failure_context.py`
- Create: `tests/adapters/test_bank.py`

**Interfaces:**
- Produces: `FailureContext.from_episode(...)`, `AdapterDescriptor`, `AdapterBank.register(...)`, `AdapterBank.query_candidates(...)`.

- [ ] **Step 1: Write failing temporal-window and privacy tests**

Test padding/masking, failure endpoint alignment, deterministic serialization, descriptor dimensions, and explicit rejection of raw images, raw trajectories, task IDs, and client IDs from ranker features.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/benchmark/test_failure_context.py tests/adapters/test_bank.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement failure contexts and adapter descriptors**

Use frozen records and content hashes. Store private artifact paths separately from descriptors returned by the simulated server.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/benchmark/test_failure_context.py tests/adapters/test_bank.py -v`

Expected: PASS.

- [ ] **Step 5: Commit interfaces**

```bash
git add src/fcut_vla/benchmark/failure_context.py src/fcut_vla/adapters tests/benchmark tests/adapters
git commit -m "feat: add failure contexts and private adapter bank"
```

### Task 5: Counterfactual utility labels and evaluation metrics

**Files:**
- Create: `src/fcut_vla/utility/__init__.py`
- Create: `src/fcut_vla/utility/counterfactual.py`
- Create: `src/fcut_vla/evaluation/__init__.py`
- Create: `src/fcut_vla/evaluation/metrics.py`
- Create: `src/fcut_vla/evaluation/statistics.py`
- Create: `tests/utility/test_counterfactual.py`
- Create: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Produces: `UtilityObservation`, `compute_utility(...)`, `ndcg_at_k(...)`, `selection_regret(...)`, `continual_metrics(transfer_matrix)`, and `paired_bootstrap_ci(...)`.

- [ ] **Step 1: Write failing utility and metric tests with hand-computed examples**

Include positive, neutral, harmful, tied-confidence, no-match, backward-transfer, forgetting, abstention, and paired-bootstrap cases.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/utility tests/evaluation -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement utility, ranking, continual, and statistical functions**

Ensure the cost and retention penalties have explicit normalization, and reject unpaired initial-state seed inputs.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/utility tests/evaluation -v`

Expected: PASS.

- [ ] **Step 5: Commit utility foundations**

```bash
git add src/fcut_vla/utility src/fcut_vla/evaluation tests/utility tests/evaluation
git commit -m "feat: compute counterfactual transfer utility and metrics"
```

### Task 6: Utility ranker, abstention, and sparse adapter composition

**Files:**
- Create: `src/fcut_vla/utility/ranker.py`
- Create: `src/fcut_vla/adapters/composition.py`
- Create: `src/fcut_vla/repair/__init__.py`
- Create: `src/fcut_vla/repair/personalized.py`
- Create: `tests/utility/test_ranker.py`
- Create: `tests/adapters/test_composition.py`
- Create: `tests/repair/test_personalized.py`

**Interfaces:**
- Produces: `UtilityRanker.forward(failure, descriptors) -> UtilityPrediction`, `select_with_abstention(...)`, `compose_lora(...)`, and `build_personalized_repair(...)`.

- [ ] **Step 1: Write failing ranker and repair tests**

Test permutation-equivariant candidate scoring, absence of client identity features, separated gain/risk/uncertainty heads, positive-lower-bound abstention, non-negative sparse coefficients, coefficient-sum bound, and deterministic top-1 behavior.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/utility/test_ranker.py tests/adapters/test_composition.py tests/repair/test_personalized.py -v`

Expected: FAIL with missing implementations.

- [ ] **Step 3: Implement the minimal cross-attention ranker and repair policy**

Use masks for variable candidate counts and return decomposed calibrated predictions. Keep top-1 and sparse top-K paths independently callable.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/utility/test_ranker.py tests/adapters/test_composition.py tests/repair/test_personalized.py -v`

Expected: PASS.

- [ ] **Step 5: Commit ranking and repair**

```bash
git add src/fcut_vla/utility/ranker.py src/fcut_vla/adapters/composition.py src/fcut_vla/repair tests/utility tests/adapters tests/repair
git commit -m "feat: rank client utility and compose personalized repairs"
```

### Task 7: Retention-gated consolidation and end-to-end CPU smoke test

**Files:**
- Create: `src/fcut_vla/repair/consolidation.py`
- Create: `tests/repair/test_consolidation.py`
- Create: `scripts/smoke_test.py`
- Create: `tests/integration/test_smoke_pipeline.py`

**Interfaces:**
- Produces: `ConsolidationDecision`, `evaluate_gate(...)`, `promote_with_rollback(...)`, and an end-to-end synthetic run manifest.

- [ ] **Step 1: Write failing consolidation and smoke tests**

Test gain lower-bound, average-risk upper-bound, worst-task bound, rollback preservation, no-match abstention, and deterministic end-to-end output hashes.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/repair/test_consolidation.py tests/integration/test_smoke_pipeline.py -v`

Expected: FAIL with missing consolidation and smoke entry point.

- [ ] **Step 3: Implement consolidation and synthetic orchestration**

The smoke run must generate three clients, positive/neutral/harmful synthetic adapters, one failure request, a ranked repair, gate decision, metrics JSON, and immutable run manifest without GPU dependencies.

- [ ] **Step 4: Run the full CPU test suite and smoke command**

Run: `python -m pytest -v && python scripts/smoke_test.py --config configs/smoke.yaml`

Expected: all tests PASS and `runs/smoke/<hash>/metrics.json` exists.

- [ ] **Step 5: Commit verified smoke pipeline**

```bash
git add src/fcut_vla/repair/consolidation.py scripts/smoke_test.py tests
git commit -m "feat: gate consolidation and verify CPU smoke pipeline"
```

### Task 8: SmolVLA/LIBERO GPU job entry points

Use an independent official LeRobot 0.6.1 checkout, LIBERO environment, deterministic evaluation, resume, and logging conventions documented in `docs/lerobot_runtime.md`. Do not import or invoke CausalVLA, and do not create a parallel simulator or checkpoint format.

**Files:**
- Create: `scripts/train_client_adapter.py`
- Create: `scripts/generate_failures.py`
- Create: `scripts/build_utility_labels.py`
- Create: `scripts/train_utility_ranker.py`
- Create: `scripts/run_repair_eval.py`
- Create: `scripts/run_continual_experiment.py`
- Create: `configs/gpu/development.yaml`
- Create: `configs/gpu/main.yaml`
- Create: `docs/gpu_commands.md`
- Create: `tests/integration/test_gpu_cli_dry_run.py`

**Interfaces:**
- Consumes: all Tasks 2-7 interfaces plus LeRobot and LIBERO.
- Produces: resumable run directories containing frozen config, git commit, manifest hash, checkpoints, paired seed list, metrics, stdout/stderr logs, and completion marker.

- [ ] **Step 1: Write failing dry-run CLI tests**

Test argument parsing, frozen hashes, resume behavior, refusal to overwrite completed runs, exact paired seed propagation, and command rendering without importing CUDA.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/integration/test_gpu_cli_dry_run.py -v`

Expected: FAIL because entry points do not exist.

- [ ] **Step 3: Implement resumable CLI entry points and command sheet**

Each command supports `--dry-run`, `--resume`, `--config`, and `--output-root`. `docs/gpu_commands.md` lists environment creation, dependency installation, smoke commands, Stage A-D jobs, expected artifacts, and failure recovery commands.

- [ ] **Step 4: Run all CLI dry runs on the Mac**

Run every documented command with `--dry-run` and verify the exact run graph without allocating GPU memory.

- [ ] **Step 5: Commit GPU orchestration**

```bash
git add scripts configs/gpu docs/gpu_commands.md tests/integration/test_gpu_cli_dry_run.py
git commit -m "feat: add resumable SmolVLA LIBERO GPU workflows"
```

### Task 9: Result export, LaTeX manuscript, and reproducibility checks

**Files:**
- Create: `scripts/export_paper_results.py`
- Create: `paper/main.tex`
- Create: `paper/sections/01_introduction.tex`
- Create: `paper/sections/02_related_work.tex`
- Create: `paper/sections/03_problem.tex`
- Create: `paper/sections/04_method.tex`
- Create: `paper/sections/05_protocol.tex`
- Create: `paper/sections/06_experiments.tex`
- Create: `paper/sections/07_limitations.tex`
- Create: `paper/sections/08_conclusion.tex`
- Create: `tests/test_result_export.py`
- Create: `tests/test_reproducibility_manifest.py`

**Interfaces:**
- Consumes: completed immutable run manifests and `paper/RESULT_SLOTS.md`.
- Produces: LaTeX macros, CSV tables, PDF figures, resolved-claim report, and compilable manuscript.

- [ ] **Step 1: Write failing result-export tests**

Test that unresolved slots cannot be silently replaced, run hashes are cited in exported artifacts, confidence intervals accompany main means, and manuscript compilation fails when a required main result is absent in release mode.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_result_export.py tests/test_reproducibility_manifest.py -v`

Expected: FAIL because exporter and LaTeX files do not exist.

- [ ] **Step 3: Implement exporter and convert the approved Markdown draft to LaTeX**

Draft mode renders visible `[RESULT:...]` boxes. Release mode requires all main-result slots and permits optional secondary/real-robot slots to remain explicitly omitted with corresponding scope text.

- [ ] **Step 4: Verify analysis and manuscript build**

Run: `python scripts/export_paper_results.py --mode draft --runs runs --paper paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper/main.tex`

Expected: exporter succeeds and `paper/main.pdf` builds with visible unresolved-result boxes only in draft mode.

- [ ] **Step 5: Commit reproducible paper pipeline**

```bash
git add scripts/export_paper_results.py paper tests/test_result_export.py tests/test_reproducibility_manifest.py
git commit -m "docs: build reproducible FCUT-VLA manuscript pipeline"
```

### Task 10: Gates A-D, secondary backbone, and SO-101 decision

**Files:**
- Create: `docs/gate_review_template.md`
- Create after each executed gate: `reports/gate_a.md`, `reports/gate_b.md`, `reports/gate_c.md`, `reports/gate_d.md`
- Modify after Gates A-D pass: `configs/gpu/openvla_subset.yaml`
- Modify after Gates A-D pass: `configs/real_robot/so101.yaml`

**Interfaces:**
- Consumes: verified metrics and run manifests.
- Produces: evidence-based proceed/pivot decisions and the exact approved scope for OpenVLA-OFT/SO-101.

- [ ] **Step 1: Create the gate-review template**

Require hypothesis, preregistered threshold, run hashes, paired statistical result, effect size, failure analysis, decision, and next authorized commands.

- [ ] **Step 2: Execute Gate A and record the decision**

Run documented SmolVLA reproducibility and failure-generation commands. Proceed only with stable non-trivial success and paired deterministic evaluation.

- [ ] **Step 3: Execute Gate B and record the decision**

Measure within-failure variance of candidate utility and verify meaningful positive, neutral, and harmful candidates.

- [ ] **Step 4: Execute Gate C and record the decision**

Compare full failure context against task-only and observation-only retrieval on held-out task-skill combinations.

- [ ] **Step 5: Execute Gate D and record the decision**

Evaluate recovery-retention Pareto frontiers for personalized repair and gated consolidation.

- [ ] **Step 6: Authorize or reject expansion**

Create OpenVLA-OFT subset and SO-101 configs only when Gates A-D pass. Otherwise apply the pivot rule from the design spec and update the title, claims, and manuscript scope.

- [ ] **Step 7: Commit gate reports and authorized scope**

```bash
git add reports docs/gate_review_template.md configs
git commit -m "docs: record FCUT-VLA evidence gates and expansion decision"
```
