# Failure-Aligned Repair Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one verified task-aligned continuation adapter from the failed
LIBERO adapter and execute an immutable, exactly paired baseline/candidate
counterfactual rollout without fabricating retention or paper metrics.

**Architecture:** Pure Python recipe and result contracts resolve task-matched
demonstration episodes and bind every input by canonical SHA-256. Thin CLIs
render or launch official LeRobot 0.6.1 training and reuse the authoritative
single-episode LIBERO recorder for each pair member. Utility-label construction
accepts only a completed eligible pair plus explicit paired retention results.

**Tech Stack:** Python 3.12, frozen dataclasses, JSON/JSONL, SHA-256, pytest,
LeRobot 0.6.1, PEFT, SmolVLA, hf-libero 0.1.4, CUDA/EGL on RTX 4090.

## Global Constraints

- Never import, invoke, or modify CausalVLA.
- The source policy is the verified failed PEFT adapter, not SmolVLA base.
- `failure_aligned_continuation_v1` trains only task-matched demonstration
  episodes through LeRobot's native `dataset.episodes` filter.
- Dataset, base policy, LeRobot, PEFT, and hf-libero identities are immutable.
- A candidate digest equal to the source adapter digest is invalid.
- Baseline and candidate rollouts use identical problem ID, seed,
  initial-state index/hash, episode budget, and environment hash.
- Raw images, simulator states, media, ownership, client IDs, and private paths
  never enter pair-result or utility-label artifacts.
- Retention values are never inferred from recovery rollouts. Missing paired
  retention results block utility-label emission.
- Completed artifacts are immutable and resume only after byte-level and
  semantic verification.
- Bounded smoke results are integration evidence, not paper results.

---

### Task 1: Freeze the failure-aligned continuation recipe

**Files:**
- Create: `src/fcut_vla/repair/__init__.py`
- Create: `src/fcut_vla/repair/continuation.py`
- Create: `tests/repair/test_continuation.py`

**Interfaces:**
- Consumes: dataset episode metadata as `Iterable[DatasetEpisode]`, the frozen
  failure instruction and identities, verified runtime identities, and
  continuation hyperparameters.
- Produces: `DatasetEpisode`, `ContinuationHyperparameters`,
  `ContinuationRecipe`, `resolve_task_episode_indices(...)`,
  `build_continuation_recipe(...)`, and `validate_continuation_recipe(...)`.

- [ ] **Step 1: Write failing deterministic resolution tests**

```python
def test_task_episode_resolution_is_exact_sorted_and_nonempty():
    episodes = (
        DatasetEpisode(8, ("put the bowl on the plate",)),
        DatasetEpisode(3, ("  Put  the BOWL on the plate ",)),
        DatasetEpisode(4, ("open the drawer",)),
    )
    assert resolve_task_episode_indices(
        episodes, "put the bowl on the plate"
    ) == (3, 8)


@pytest.mark.parametrize("episodes", [(), (DatasetEpisode(0, ("other",)),)])
def test_task_episode_resolution_rejects_no_match(episodes):
    with pytest.raises(ContinuationError, match="no dataset episodes"):
        resolve_task_episode_indices(episodes, "target instruction")
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
conda run -n fedvla python -m pytest -q tests/repair/test_continuation.py
```

Expected: collection fails because `fcut_vla.repair.continuation` does not
exist.

- [ ] **Step 3: Implement normalized task episode resolution**

```python
@dataclass(frozen=True, order=True)
class DatasetEpisode:
    index: int
    tasks: tuple[str, ...]


def resolve_task_episode_indices(
    episodes: Iterable[DatasetEpisode], instruction: str
) -> tuple[int, ...]:
    target = normalize_instruction(instruction)
    matched = sorted({
        episode.index
        for episode in episodes
        if any(normalize_instruction(task) == target for task in episode.tasks)
    })
    if not matched:
        raise ContinuationError("no dataset episodes match the failure instruction")
    if any(index < 0 for index in matched):
        raise ContinuationError("dataset episode indices must be non-negative")
    return tuple(matched)
```

Use the existing whitespace-collapse and case-fold semantics from task
resolution; expose a shared helper rather than creating a second normalization
definition.

- [ ] **Step 4: Write failing recipe identity and validation tests**

```python
def test_recipe_hash_binds_source_failure_dataset_runtime_and_hyperparameters():
    recipe = build_continuation_recipe(**valid_arguments())
    assert validate_continuation_recipe(recipe) == recipe
    changed = build_continuation_recipe(
        **{**valid_arguments(), "episode_indices": (3, 9)}
    )
    assert changed.content_hash != recipe.content_hash


def test_recipe_rejects_mutated_payload_with_old_hash():
    recipe = build_continuation_recipe(**valid_arguments())
    with pytest.raises(ContinuationError, match="content hash"):
        validate_continuation_recipe(replace(recipe, training_seed=999))
```

- [ ] **Step 5: Run the new tests and confirm RED**

Run: `conda run -n fedvla python -m pytest -q tests/repair/test_continuation.py`

Expected: failure because recipe types/functions are absent.

- [ ] **Step 6: Implement the canonical recipe**

```python
@dataclass(frozen=True)
class ContinuationHyperparameters:
    steps: int
    batch_size: int
    learning_rate: float
    training_seed: int


@dataclass(frozen=True)
class ContinuationRecipe:
    schema_version: int
    recipe_type: str
    source_run_hash: str
    source_adapter_sha256: str
    source_episode_sha256: str
    failure_hash: str
    task_alias: str
    problem_id: str
    instruction: str
    dataset_repo: str
    dataset_revision: str
    episode_indices: tuple[int, ...]
    episode_selection_sha256: str
    base_policy_repo: str
    base_policy_revision: str
    lerobot_commit: str
    peft_version: str
    hf_libero_version: str
    python_version: str
    torch_version: str
    hyperparameters: ContinuationHyperparameters
    source_target_modules: tuple[str, ...]
    content_hash: str
```

Require schema version 1, recipe type
`failure_aligned_continuation_v1`, sorted unique non-negative episode indices,
positive steps/batch size/learning rate, non-negative seed, non-empty immutable
revisions, and a recomputed canonical content hash.

- [ ] **Step 7: Run focused and full tests**

Run:

```bash
conda run -n fedvla python -m pytest -q tests/repair/test_continuation.py
conda run -n fedvla python -m pytest -q
git diff --check
```

Expected: all pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/fcut_vla/repair tests/repair/test_continuation.py
git commit -m "feat: freeze failure-aligned continuation recipes"
```

---

### Task 2: Render and verify continuation training

**Files:**
- Create: `src/fcut_vla/repair/training.py`
- Create: `scripts/build_repair_candidate.py`
- Create: `configs/gpu/smolvla_repair_smoke.yaml`
- Create: `tests/repair/test_training.py`
- Create: `tests/integration/test_build_repair_candidate_cli.py`
- Modify: `scripts/verify_adapter_run.py`

**Interfaces:**
- Consumes: a verified source adapter run, canonical failed episode, canonical
  failure shard, LeRobot dataset metadata, and smoke YAML.
- Produces: `ContinuationTrainingPlan`, `render_continuation_command(...)`,
  `verify_repair_candidate(...)`, `repair_recipe.json`, logs, checkpoint,
  `candidate_metadata.json`, and `COMPLETE`.

- [ ] **Step 1: Write failing command-rendering tests**

```python
def test_continuation_command_resumes_source_and_filters_exact_episodes(tmp_path):
    plan = training_plan_fixture(tmp_path, episode_indices=(3, 8))
    command = render_continuation_command(plan)
    assert command[:3] == (sys.executable, "-m", "lerobot.scripts.lerobot_train")
    assert f"--policy.path={plan.source_pretrained_dir}" in command
    assert "--resume=true" not in command
    assert "--dataset.episodes=[3,8]" in command
    assert f"--steps={plan.recipe.hyperparameters.steps}" in command
    assert str(plan.output_dir) in " ".join(command)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `conda run -n fedvla python -m pytest -q tests/repair/test_training.py`

Expected: missing training module/functions.

- [ ] **Step 3: Implement pure plan rendering**

`ContinuationTrainingPlan` must hold only validated paths and the immutable
recipe. Resolve the source's latest completed `pretrained_model` directory from
verified adapter metadata. Load that PEFT adapter through `--policy.path` so
LeRobot marks it trainable, but start a fresh optimizer and scheduler rather
than trusting source optimizer/RNG files that are not bound by the source
completion record. Bind the exact source adapter and policy configuration-file
digests plus a canonical filename-to-digest manifest for every serialized
preprocessor/postprocessor artifact (including normalizer state tensors) into
the recipe, then revalidate them before launch. Pin the separate tokenizer Hub
dependency to the immutable revision declared in the repair config and inject
that revision through the guarded training entry point. Render an argv tuple
without a shell. Set `MUJOCO_GL=egl`
only in the subprocess environment, not in the argv. The new run executes
exactly the recipe continuation-step count.

- [ ] **Step 4: Write failing candidate-verification tests**

```python
def test_candidate_verification_requires_new_digest_and_same_lora_contract(tmp_path):
    source, candidate = write_source_and_candidate_runs(tmp_path)
    report = verify_repair_candidate(source, candidate)
    assert report["candidate_adapter_sha256"] != report["source_adapter_sha256"]

    copy_source_weights_over_candidate(source, candidate)
    with pytest.raises(ValueError, match="distinct adapter digest"):
        verify_repair_candidate(source, candidate)


def test_candidate_verification_rejects_target_module_drift(tmp_path):
    source, candidate = write_source_and_candidate_runs(tmp_path)
    mutate_candidate_target_modules(candidate)
    with pytest.raises(ValueError, match="target contract"):
        verify_repair_candidate(source, candidate)
```

- [ ] **Step 5: Run candidate tests and confirm RED**

Run: `conda run -n fedvla python -m pytest -q tests/repair/test_training.py`

Expected: verification functions absent.

- [ ] **Step 6: Implement candidate verification and completion binding**

Reuse adapter file-format, parameter-count, target-module, and digest checks
from `verify_adapter_run.py` through imported pure helpers. Do not duplicate the
safetensors contract. Bind the canonical recipe digest, candidate adapter
digest, adapter-config digest, completed step, artifact byte size, source
adapter digest, and runtime provenance into `candidate_metadata.json` and the
final `COMPLETE` hash.

- [ ] **Step 7: Write failing CLI dry-run/resume tests**

```python
def test_repair_candidate_cli_dry_run_is_deterministic_and_does_not_train(tmp_path):
    first = run_cli(tmp_path, "--dry-run")
    second = run_cli(tmp_path, "--dry-run")
    assert first.stdout == second.stdout
    assert "dataset.episodes" in first.stdout
    assert not list((tmp_path / "output").glob("**/model.safetensors"))


def test_repair_candidate_cli_refuses_mismatched_existing_recipe(tmp_path):
    run_cli(tmp_path)
    mutate_recipe(tmp_path)
    assert run_cli(tmp_path, "--resume").returncode != 0
```

- [ ] **Step 8: Implement the CLI and smoke configuration**

The CLI accepts `--source-adapter-run`, `--episode`, `--failure-shard`,
`--config`, `--output-root`, `--dry-run`, and `--resume`. It derives task and
policy identities from verified artifacts; no free-form policy identity or
episode list is accepted. Dataset episode selection is resolved from downloaded
metadata before recipe creation. The smoke config uses a bounded continuation
budget and must not modify `configs/gpu/main.yaml`.

- [ ] **Step 9: Run focused, dry-run, and full verification**

```bash
conda run -n fedvla python -m pytest -q \
  tests/repair/test_training.py \
  tests/integration/test_build_repair_candidate_cli.py
conda run -n fedvla python scripts/build_repair_candidate.py --help
conda run -n fedvla python -m pytest -q
git diff --check
```

- [ ] **Step 10: Request code review and fix every Critical/Important issue**

Review must cover resume immutability, source-checkpoint semantics, task episode
selection, runtime pinning, and candidate digest/target-contract verification.

- [ ] **Step 11: Commit Task 2**

```bash
git add src/fcut_vla/repair/training.py scripts/build_repair_candidate.py \
  scripts/verify_adapter_run.py configs/gpu/smolvla_repair_smoke.yaml \
  tests/repair/test_training.py tests/integration/test_build_repair_candidate_cli.py
git commit -m "feat: train verified failure-aligned repair candidates"
```

---

### Task 3: Build an eligible exact counterfactual pair plan

**Files:**
- Create: `scripts/build_counterfactual_pair_plan.py`
- Create: `tests/integration/test_build_counterfactual_pair_plan_cli.py`
- Modify: `src/fcut_vla/libero/pairing.py`
- Modify: `tests/libero/test_pairing.py`

**Interfaces:**
- Consumes: verified source failed-adapter run, verified repair-candidate run,
  canonical failed episode/failure shard, and canonical environment contract.
- Produces: a `PairPlan` with `purpose="counterfactual_repair"`,
  `label_eligible=True`, source adapter as baseline, and distinct candidate as
  candidate.

- [ ] **Step 1: Write failing semantic pair-plan tests**

```python
def test_repair_plan_uses_failed_adapter_as_baseline_and_distinct_candidate():
    plan = build_repair_pair_plan(source_report, candidate_report, failure_fixture())
    pair = plan.pairs[0]
    assert pair.baseline.policy_revision == source_report["adapter_sha256"]
    assert pair.candidate.policy_revision == candidate_report["candidate_adapter_sha256"]
    assert pair.baseline.policy_revision != pair.candidate.policy_revision
    assert plan.purpose == "counterfactual_repair"
    assert plan.label_eligible


def test_repair_plan_rejects_base_policy_or_same_adapter_baseline():
    with pytest.raises(PairingError, match="distinct verified adapters"):
        build_repair_pair_plan(source_report, source_report, failure_fixture())
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `conda run -n fedvla python -m pytest -q tests/libero/test_pairing.py`

- [ ] **Step 3: Extend pair semantics minimally**

Add `build_repair_pair_plan(...)` as a pure helper. It must require the canonical
logical adapter repository for both members, revisions equal to their verified
artifact digests, candidate metadata bound to the source adapter/failure, and
different revisions. Keep `build_pair_plan(...)` generic for fixture tests.

- [ ] **Step 4: Write failing CLI artifact/resume tests**

```python
def test_counterfactual_pair_plan_cli_is_eligible_and_resume_safe(tmp_path):
    output = run_pair_plan_cli(valid_artifacts(tmp_path))
    plan = json.loads(output.read_text())
    assert plan["purpose"] == "counterfactual_repair"
    assert plan["label_eligible"] is True
    assert run_pair_plan_cli(valid_artifacts(tmp_path), "--resume").read_bytes() == output.read_bytes()
```

- [ ] **Step 5: Implement the eligible pair-plan CLI**

Accept only artifact paths and output/resume flags. Recompute Stage B, verify
both adapters, verify the candidate recipe, derive every policy/environment
identity, and write canonical JSON atomically. Reject an existing output unless
`--resume` revalidates identical bytes.

- [ ] **Step 6: Run focused/full tests and review**

```bash
conda run -n fedvla python -m pytest -q \
  tests/libero/test_pairing.py \
  tests/integration/test_build_counterfactual_pair_plan_cli.py
conda run -n fedvla python -m pytest -q
git diff --check
```

Request review of baseline identity, candidate binding, environment completeness,
and eligibility semantics; fix all Critical/Important findings.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/fcut_vla/libero/pairing.py scripts/build_counterfactual_pair_plan.py \
  tests/libero/test_pairing.py tests/integration/test_build_counterfactual_pair_plan_cli.py
git commit -m "feat: freeze eligible repair counterfactual plans"
```

---

### Task 4: Execute and validate paired LIBERO rollouts

**Files:**
- Create: `src/fcut_vla/libero/counterfactual_runner.py`
- Create: `scripts/run_counterfactual_pair.py`
- Create: `tests/libero/test_counterfactual_runner.py`
- Create: `tests/integration/test_run_counterfactual_pair_cli.py`
- Modify: `src/fcut_vla/libero/runtime.py`

**Interfaces:**
- Consumes: a validated eligible one-pair plan, verified source/candidate
  checkpoints, and the existing `record_lerobot_episode(...)` runtime.
- Produces: `PairedRecoveryResult`, `run_counterfactual_pair(...)`,
  `baseline_episode.json`, `candidate_episode.json`, `PAIR_RESULT.json`, and
  `COMPLETE`.

- [ ] **Step 1: Write failing fixture-runner tests**

```python
def test_pair_runner_records_both_members_with_identical_environment_identity(tmp_path):
    result = run_counterfactual_pair(plan_fixture(), fixture_executor(), tmp_path)
    assert result.baseline.key.seed == result.candidate.key.seed
    assert result.baseline.initial_state_hash == result.candidate.initial_state_hash
    assert result.baseline.problem_id == result.candidate.problem_id
    assert result.baseline.policy_revision != result.candidate.policy_revision


def test_pair_runner_closes_each_member_and_never_runs_candidate_after_baseline_error(tmp_path):
    executor = failing_executor(role="baseline")
    with pytest.raises(RuntimeError, match="baseline failed"):
        run_counterfactual_pair(plan_fixture(), executor, tmp_path)
    assert executor.closed_roles == ["baseline"]
    assert executor.started_roles == ["baseline"]
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `conda run -n fedvla python -m pytest -q tests/libero/test_counterfactual_runner.py`

- [ ] **Step 3: Implement the pure orchestration and result envelope**

```python
@dataclass(frozen=True)
class PairedRecoveryResult:
    schema_version: int
    pair_plan_hash: str
    pair_key: PairKey
    baseline_episode_sha256: str
    candidate_episode_sha256: str
    baseline_success: bool
    candidate_success: bool
    candidate_adapter_bytes: int
    candidate_load_and_rollout_seconds: float
    complete: bool
    content_hash: str
```

Validate both `EpisodeRecord`s against the selected pair before writing output.
Use monotonic timing with a frozen definition covering candidate policy load
through environment close. Do not include local checkpoint paths in the result.
Write member episode files first, fsync, then the result envelope, then the
completion hash.

- [ ] **Step 4: Write failing CLI dry-run and resume tests**

```python
def test_pair_cli_dry_run_renders_two_authoritative_member_runs(tmp_path):
    report = json.loads(run_cli(tmp_path, "--dry-run").stdout)
    assert [item["role"] for item in report["members"]] == ["baseline", "candidate"]
    assert report["members"][0]["seed"] == report["members"][1]["seed"]


def test_pair_cli_resume_rejects_changed_episode_bytes(tmp_path):
    run_cli(tmp_path)
    mutate_baseline_episode(tmp_path)
    assert run_cli(tmp_path, "--resume").returncode != 0
```

- [ ] **Step 5: Implement official runtime adapter and CLI**

Reuse `OfficialLeRobotBindings` and `record_lerobot_episode`; do not create a
second observation/action transformation. Construct a fresh environment/policy
per member and guarantee cleanup on construction or rollout failure. Validate
the installed hf-libero version and pinned LeRobot checkout before either run.

- [ ] **Step 6: Run focused/full tests and review**

```bash
conda run -n fedvla python -m pytest -q \
  tests/libero/test_counterfactual_runner.py \
  tests/integration/test_run_counterfactual_pair_cli.py
conda run -n fedvla python -m pytest -q
git diff --check
```

Review must cover paired identity, action safety, environment cleanup, timing
definition, privacy, output ordering, partial-failure behavior, and resume
tamper detection. Fix all Critical/Important findings.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/fcut_vla/libero/counterfactual_runner.py \
  src/fcut_vla/libero/runtime.py scripts/run_counterfactual_pair.py \
  tests/libero/test_counterfactual_runner.py \
  tests/integration/test_run_counterfactual_pair_cli.py
git commit -m "feat: execute exact paired LIBERO counterfactuals"
```

---

### Task 5: Enforce retention completeness and document server gates

**Files:**
- Create: `src/fcut_vla/utility/retention.py`
- Create: `tests/utility/test_retention.py`
- Modify: `src/fcut_vla/jobs/build_utility_labels.py`
- Modify: `tests/jobs/test_build_utility_labels.py`
- Modify: `docs/gpu_commands.md`

**Interfaces:**
- Consumes: validated `PairedRecoveryResult` records and explicit paired
  retention results keyed by the same pair-plan hash and adapter digest.
- Produces: `RetentionResult`, `validate_paired_retention(...)`,
  `build_pair_result(recovery, retention, descriptor) -> PairResult`, and a
  utility gate that refuses missing, unpaired, or identity-mismatched retention
  inputs.

- [ ] **Step 1: Write failing retention-gate tests**

```python
def test_utility_label_requires_explicit_complete_paired_retention():
    result, plan = valid_recovery_result_and_plan()
    with pytest.raises(ValueError, match="paired retention"):
        build_pair_result(
            result,
            retention=(),
            descriptor={"skill_prototype": [0.1], "reliability": [1.0]},
        )


def test_retention_rejects_different_seed_or_policy_identity():
    retention = retention_fixture(candidate_seed=102, baseline_seed=101)
    with pytest.raises(RetentionError, match="paired identity"):
        validate_paired_retention(retention)
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n fedvla python -m pytest -q \
  tests/utility/test_retention.py tests/jobs/test_build_utility_labels.py
```

- [ ] **Step 3: Implement the explicit retention contract**

`RetentionResult` freezes pair-plan hash, candidate adapter digest, retention
task problem ID, baseline/candidate policy revisions, paired seed list,
initial-state hashes, binary success vectors, completion, and content hash.
Require at least one retention task and one paired seed. Aggregate only validated
records into the existing `retention_before` and `retention_after` mappings.
`build_pair_result` converts the recovery booleans into one-element recovery
vectors, obtains communication bytes and frozen latency from the recovery
envelope, and obtains retention mappings only from validated `RetentionResult`
objects. Remove any public API that accepts caller-provided unbound retention
mappings. The returned `PairResult` then enters the existing
`build_utility_labels(..., pair_plans=...)` gate unchanged.

- [ ] **Step 4: Update server guide**

Document, in exact order:

1. pull and runtime version checks;
2. repair-candidate dry-run;
3. bounded RTX 4090 continuation training;
4. candidate verification commands and expected files;
5. eligible pair-plan construction;
6. paired rollout dry-run and execution;
7. result verification;
8. explicit statement that utility labels remain blocked until retention runs
   are available.

Use `export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:$PATH"` before any native build
or runtime command and use `hf-libero==0.1.4 --no-deps` if repair is necessary.

- [ ] **Step 5: Run all verification gates**

```bash
conda run -n fedvla python -m pytest -q \
  tests/utility/test_retention.py \
  tests/jobs/test_build_utility_labels.py
conda run -n fedvla python -m pytest -q
git diff --check
git status --short
```

- [ ] **Step 6: Request final code review**

The reviewer must inspect all changes since `8718063` for Critical/Important
issues in provenance, paired semantics, privacy, reproducibility, cleanup,
artifact immutability, retention gating, and truthfulness of documentation.
Fix every Critical/Important finding and rerun Step 5.

- [ ] **Step 7: Commit and push Task 5**

```bash
git add src/fcut_vla/utility/retention.py \
  src/fcut_vla/jobs/build_utility_labels.py \
  tests/utility/test_retention.py tests/jobs/test_build_utility_labels.py \
  docs/gpu_commands.md
git commit -m "feat: gate counterfactual utility on paired retention"
git push origin codex/fcut-vla-paper
```

## Final acceptance sequence

The implementation is ready for the user's RTX 4090 only after Tasks 1–5 pass
on Mac, the branch is clean and pushed, and final review has no unresolved
Critical/Important issue. The first server dispatch stops after candidate
training and verification; paired rollout is dispatched only after the user
returns the verified candidate report. Retention evaluation is dispatched only
after the paired recovery artifact verifies.
