# LIBERO Failure and Counterfactual Utility Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic Stage B failure artifacts and Stage C paired counterfactual utility labels by adapting the validated CausalVLA/LeRobot evaluator.

**Architecture:** Pure Python schema, resolution, pairing, and privacy modules operate without simulator dependencies and are tested on fixture evaluator output. Thin job modules consume immutable plans and invoke an evaluator adapter; the GPU adapter renders the existing CausalVLA evaluation command and records hashes, while a deterministic fixture adapter proves the complete Mac pipeline.

**Tech Stack:** Python 3.11+, dataclasses, JSON/JSON Lines, SHA-256, PyYAML, pytest, LeRobot 0.6.1, LIBERO, SmolVLA, CausalVLA evaluator.

## Global Constraints

- Reuse LeRobot 0.6.1 and the CausalVLA evaluator; do not create another simulator or checkpoint format.
- GPU rendering uses `MUJOCO_GL=egl`, synchronous LIBERO environments, and the wrist-image rename map.
- Every policy is identified by Hugging Face repository plus immutable revision under `phawitbinabik`.
- Baseline and candidate evaluation keys must match task, seed, initial-state index/hash, episode budget, and environment settings exactly.
- Raw images, simulator states, trajectories, task/client IDs, ownership, and private adapter paths never enter utility-ranker examples.
- Completed run artifacts are immutable; no missing rollout or metric may be replaced with a placeholder.

---

### Task 1: Resolve and freeze LIBERO task identities

**Files:**
- Create: `src/fcut_vla/libero/__init__.py`
- Create: `src/fcut_vla/libero/task_resolution.py`
- Create: `tests/libero/test_task_resolution.py`

**Interfaces:**
- Consumes: manifest task mappings with alias, suite, instruction, BDDL path, and initial-state path.
- Produces: `ResolvedTask`, `resolve_task(alias, instruction, installed_tasks)`, and `freeze_task_resolution(tasks, libero_commit)`.

- [ ] **Step 1: Write failing unique-resolution and hashing tests**

```python
def test_resolve_task_requires_exactly_one_normalized_instruction_match(tmp_path):
    installed = [InstalledTask("libero_spatial", "pick_up_mug", "Pick up the mug", bddl, states)]
    resolved = resolve_task("spatial.pick_mug", " pick up the mug ", installed)
    assert resolved.problem_id == "pick_up_mug"
    assert resolved.bddl_sha256 == sha256(bddl.read_bytes()).hexdigest()

@pytest.mark.parametrize("installed", [[], [task_a, task_b]])
def test_resolve_task_rejects_zero_or_multiple_matches(installed):
    with pytest.raises(TaskResolutionError):
        resolve_task("alias", "same instruction", installed)
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `.venv/bin/python -m pytest tests/libero/test_task_resolution.py -v`
Expected: FAIL because `fcut_vla.libero.task_resolution` does not exist.

- [ ] **Step 3: Implement immutable resolution records**

Use frozen dataclasses `InstalledTask` and `ResolvedTask`; normalize instructions with stripped, collapsed, case-folded whitespace; require one match within the requested suite; hash BDDL and initial-state bytes; serialize sorted records with canonical JSON. `freeze_task_resolution` includes `schema_version=1`, `libero_commit`, and a content SHA-256.

- [ ] **Step 4: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/libero/test_task_resolution.py -v && .venv/bin/python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fcut_vla/libero tests/libero/test_task_resolution.py
git commit -m "feat: freeze unique LIBERO task resolution"
```

### Task 2: Validate episode shards and extract failure contexts

**Files:**
- Create: `src/fcut_vla/libero/episode.py`
- Create: `src/fcut_vla/jobs/__init__.py`
- Create: `src/fcut_vla/jobs/generate_failures.py`
- Create: `tests/libero/test_episode.py`
- Create: `tests/jobs/test_generate_failures.py`

**Interfaces:**
- Consumes: `EpisodeRecord.from_mapping(raw)` and a canonical JSONL episode shard.
- Produces: `EpisodeKey`, `EpisodeRecord`, `validate_episode_shard(path)`, and `extract_failures(records, window_size)` returning existing `FailureContext` objects plus source hashes.

- [ ] **Step 1: Write failing episode and extraction tests**

```python
def test_failed_episode_extracts_one_context_but_success_is_excluded():
    failed = EpisodeRecord.from_mapping(failed_fixture)
    successful = EpisodeRecord.from_mapping(success_fixture)
    extracted = extract_failures([failed, successful], window_size=4)
    assert len(extracted) == 1
    assert extracted[0].context.mask == (False, False, True, True)
    assert extracted[0].source_episode_sha256 == failed.content_hash()

def test_shard_rejects_duplicate_keys_and_nonfinite_steps(tmp_path):
    with pytest.raises(EpisodeValidationError):
        validate_episode_shard(write_jsonl(tmp_path, [bad, bad]))
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `.venv/bin/python -m pytest tests/libero/test_episode.py tests/jobs/test_generate_failures.py -v`
Expected: FAIL because episode/job modules do not exist.

- [ ] **Step 3: Implement canonical episode records and extraction**

Require schema version, task alias/problem ID, seed, episode and initial-state indices/hash, policy repo/revision, instruction, terminal success, and non-empty ordered steps. Validate all numeric values as finite and all step dimensions as fixed. Hash canonical JSON excluding any claimed hash. Reject duplicate `EpisodeKey` values. Convert only `terminal_success=False` records through `FailureContext.from_episode`; output canonical JSONL sorted by episode key and write its SHA-256 sidecar before `SHARD.json`.

- [ ] **Step 4: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/libero/test_episode.py tests/jobs/test_generate_failures.py -v && .venv/bin/python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fcut_vla/libero/episode.py src/fcut_vla/jobs tests/libero/test_episode.py tests/jobs/test_generate_failures.py
git commit -m "feat: extract hashed failure contexts from LIBERO episodes"
```

### Task 3: Freeze exact counterfactual pair plans

**Files:**
- Create: `src/fcut_vla/libero/pairing.py`
- Create: `tests/libero/test_pairing.py`

**Interfaces:**
- Consumes: target failure hashes, candidate adapter digests, resolved tasks, paired seeds, initial-state records, episode budget, and environment settings.
- Produces: `PairKey`, `PairMember`, `CounterfactualPair`, `build_pair_plan(...)`, and `validate_completed_pair(pair, baseline, candidate)`.

- [ ] **Step 1: Write failing exact-pair tests**

```python
def test_pair_plan_preserves_every_seed_and_initial_state():
    plan = build_pair_plan(failures, adapters, seeds=(101, 102), states=states, episode_budget=2, environment=env)
    assert [pair.key.seed for pair in plan.pairs] == [101, 101, 102, 102]

def test_completed_pair_rejects_initial_state_or_revision_change():
    with pytest.raises(PairingError):
        validate_completed_pair(pair, baseline, replace(candidate, initial_state_hash="changed"))
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `.venv/bin/python -m pytest tests/libero/test_pairing.py -v`
Expected: FAIL because `fcut_vla.libero.pairing` does not exist.

- [ ] **Step 3: Implement pair-plan construction and validation**

Make `PairKey` contain failure hash, adapter digest, task problem ID, seed, initial-state index/hash, episode budget, and canonical environment hash. Sort by all fields. Baseline and candidate members differ only in policy repo/revision; validation compares the complete `PairKey`, rejects duplicate roles, and requires both members to be complete. The plan includes schema version, benchmark hash, policy revisions, and its own canonical content hash.

- [ ] **Step 4: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/libero/test_pairing.py -v && .venv/bin/python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fcut_vla/libero/pairing.py tests/libero/test_pairing.py
git commit -m "feat: freeze exact counterfactual rollout pairs"
```

### Task 4: Build privacy-safe utility labels

**Files:**
- Create: `src/fcut_vla/utility/privacy.py`
- Create: `src/fcut_vla/jobs/build_utility_labels.py`
- Create: `tests/utility/test_privacy.py`
- Create: `tests/jobs/test_build_utility_labels.py`

**Interfaces:**
- Consumes: validated `CounterfactualPair` results and existing `compute_utility(...)`.
- Produces: `assert_ranker_payload_safe(payload)`, `UtilityLabel`, `build_utility_labels(pair_results, ...)`, and `assign_failure_strata(labels_by_failure)`.

- [ ] **Step 1: Write failing privacy, utility, and no-match tests**

```python
@pytest.mark.parametrize("forbidden", ["client_id", "task_id", "raw_images", "private_path"])
def test_recursive_privacy_validator_rejects_forbidden_nested_key(forbidden):
    with pytest.raises(PrivacyBoundaryError):
        assert_ranker_payload_safe({"nested": [{forbidden: "secret"}]})

def test_labels_require_complete_pairs_and_assign_no_match():
    labels = build_utility_labels([neutral_pair, harmful_pair], normalization=normalization)
    assert assign_failure_strata(labels)[failure_hash] == "no_match"
    assert all(label.utility.utility_lower <= 0 for label in labels)
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `.venv/bin/python -m pytest tests/utility/test_privacy.py tests/jobs/test_build_utility_labels.py -v`
Expected: FAIL because privacy and job modules do not exist.

- [ ] **Step 3: Implement recursive boundary and label ledger**

Walk mappings and sequences recursively; compare case-folded keys to the union of adapter-bank forbidden fields plus `raw_simulator_state`, `raw_frames`, and `trajectory`. Validate each pair before computing utility. Store only failure hash, sanitized failure context, adapter digest, sanitized descriptor, utility fields, pair-plan hash, and masks. Assign `seen_task` or `compositional` from the frozen ledger only when any candidate has positive lower utility; otherwise assign `no_match`. Write canonical sorted JSONL atomically and hash it.

- [ ] **Step 4: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/utility/test_privacy.py tests/jobs/test_build_utility_labels.py -v && .venv/bin/python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fcut_vla/utility/privacy.py src/fcut_vla/jobs/build_utility_labels.py tests/utility/test_privacy.py tests/jobs/test_build_utility_labels.py
git commit -m "feat: build privacy-safe paired utility labels"
```

### Task 5: Integrate fixture evaluator, CausalVLA command adapter, and resumable jobs

**Files:**
- Create: `src/fcut_vla/libero/evaluator.py`
- Create: `tests/fixtures/libero/installed_tasks.json`
- Create: `tests/fixtures/libero/episodes.jsonl`
- Create: `tests/integration/test_failure_utility_backend.py`
- Modify: `scripts/_gpu_job.py`
- Modify: `configs/gpu/development.yaml`
- Modify: `configs/gpu/main.yaml`
- Modify: `docs/gpu_commands.md`

**Interfaces:**
- Consumes: Tasks 1–4 and the existing six GPU entry points.
- Produces: `FixtureEvaluator`, `CausalVLAEvaluator.render_command(...)`, executable Stage B/C job modules, immutable output tree, and removal of the Stage B/C execution guard only.

- [ ] **Step 1: Write failing fixture integration and command tests**

```python
def test_fixture_backend_is_byte_stable_and_resume_safe(tmp_path):
    first = run_failure_and_utility_fixture(tmp_path / "a")
    second = run_failure_and_utility_fixture(tmp_path / "b")
    assert first.failure_sha256 == second.failure_sha256
    assert first.label_sha256 == second.label_sha256
    assert first.labels[0]["stratum"] == "no_match"

def test_gpu_command_pins_causalvla_contract():
    command = CausalVLAEvaluator(config).render_command(plan)
    assert "MUJOCO_GL=egl" in command
    assert "--eval.use_async_envs=false" in command
    assert 'observation.images.wrist_image' in command
    assert config.policy_revision in command
```

- [ ] **Step 2: Run integration tests and verify failure**

Run: `.venv/bin/python -m pytest tests/integration/test_failure_utility_backend.py -v`
Expected: FAIL because evaluator adapters do not exist.

- [ ] **Step 3: Implement adapters and resumable stage execution**

`FixtureEvaluator` reads the committed JSONL fixture through the same episode parser as GPU data. `CausalVLAEvaluator` renders an argument list invoking the configured CausalVLA `scripts/eval_ood.py`, pinned policy repo/revision, LIBERO task, rename map, synchronous env, seed, episode count, and run-local output directory. Use `subprocess.run(argv, env={..., "MUJOCO_GL": "egl"})`, never `shell=True`. Validate `eval_info.json` and recorded shard hashes before marking a shard complete. Stage B writes task resolution then failures; Stage C writes pair plan then labels. On resume, validate all frozen hashes and skip only verified complete shards. Remove the execution guard for `generate_failures` and `build_utility_labels`; retain it for stages D–F.

- [ ] **Step 4: Run Mac integration, all dry runs, and full suite**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_failure_utility_backend.py -v
for script in train_client_adapter generate_failures build_utility_labels train_utility_ranker run_repair_eval run_continual_experiment; do
  .venv/bin/python scripts/$script.py --config configs/gpu/development.yaml --output-root /tmp/fcut-vla-preflight --dry-run --resume
done
.venv/bin/python -m pytest -q
git diff --check
```

Expected: integration and full suites PASS; all commands emit JSON plans without importing CUDA; no tracked run artifact is created.

- [ ] **Step 5: Update server guide with exact Stage B/C commands and outputs**

Document environment discovery for the local CausalVLA checkout, pinned revisions, preflight, real Stage B/C commands, expected hashes/files, resume, and transfer back to the Mac. State explicitly that stages D–F remain guarded until their backends are implemented.

- [ ] **Step 6: Commit and push**

```bash
git add src/fcut_vla/libero src/fcut_vla/jobs scripts/_gpu_job.py configs/gpu docs/gpu_commands.md tests
git commit -m "feat: collect LIBERO failures and paired utility labels"
git push
```
