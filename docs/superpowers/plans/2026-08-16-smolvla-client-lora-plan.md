# SmolVLA Client LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, validate, and hash native LeRobot PEFT/LoRA client adapters instead of full SmolVLA checkpoints.

**Architecture:** The existing immutable job runner renders LeRobot 0.6.1 native PEFT arguments from frozen YAML. A focused adapter-artifact module validates PEFT configuration, parameter counts, files, and digests before the runner writes `COMPLETE`; Mac and CUDA smoke configurations share every setting except device and experiment name.

**Tech Stack:** Python 3.12, official LeRobot 0.6.1, PyTorch 2.11, PEFT 0.18+, SmolVLA, YAML, safetensors, pytest, MPS, CUDA.

## Global Constraints

- Use native LeRobot `PeftConfig`; do not patch LeRobot or inject a second PEFT wrapper.
- Primary smoke uses LORA rank 8, alpha 16, native SmolVLA targets, and no modules saved in full.
- Every job runs with `PYTHONNOUSERSITE=1` and `PYTHONUNBUFFERED=1`.
- Full-model `model.safetensors` is not a valid client-adapter payload.
- `COMPLETE` is written only after adapter files, config, parameter ratio, digest, and reload metadata validate.
- All runtime paths and artifacts belong to FGKT-VLA; no other research project is referenced.

---

### Task 1: Render and freeze native LoRA training commands

**Files:**
- Modify: `scripts/_gpu_job.py`
- Modify: `configs/local/smolvla_smoke.yaml`
- Modify: `configs/gpu/smolvla_smoke.yaml`
- Modify: `configs/gpu/development.yaml`
- Modify: `configs/gpu/main.yaml`
- Modify: `tests/integration/test_gpu_cli_dry_run.py`

**Interfaces:**
- Consumes: YAML `lora.rank`, `lora.alpha`, and optional `lora.target_modules`.
- Produces: `render_peft_arguments(config) -> tuple[str, ...]` and a command containing native LeRobot PEFT flags.

- [ ] **Step 1: Write failing command/config tests**

```python
def test_training_command_enables_native_lora_without_explicit_primary_targets(tmp_path):
    result = run_local_smoke_dry_run(tmp_path)
    command = result["command"]
    assert "--peft.method_type=LORA" in command
    assert "--peft.r=8" in command
    assert "--peft.lora_alpha=16" in command
    assert "--peft.full_training_modules=[]" in command
    assert "--peft.target_modules" not in command

def test_invalid_lora_rank_or_alpha_is_rejected(tmp_path):
    config = smoke_config(tmp_path, rank=0, alpha=16)
    with pytest.raises(ValueError, match="rank"):
        render_peft_arguments(config)
```

- [ ] **Step 2: Run tests and verify behavioral failure**

Run: `conda run -n fedvla python -m pytest tests/integration/test_gpu_cli_dry_run.py -v`
Expected: FAIL because commands have no `--peft.*` arguments.

- [ ] **Step 3: Implement validated PEFT argument rendering**

Parse rank/alpha as positive integers. Render method, rank, alpha, and an empty full-training-module list. Omit target modules when the YAML value is empty or `native_default`; for an ablation, sort and JSON-encode a non-empty explicit list. Add the PEFT configuration fields to the immutable manifest identity, so any adapter-setting change changes the run hash.

- [ ] **Step 4: Run focused and full tests**

Run: `conda run -n fedvla python -m pytest tests/integration/test_gpu_cli_dry_run.py -v && conda run -n fedvla python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_gpu_job.py configs tests/integration/test_gpu_cli_dry_run.py
git commit -m "feat: render native SmolVLA LoRA training jobs"
```

### Task 2: Validate and describe PEFT adapter artifacts

**Files:**
- Create: `src/fcut_vla/adapters/artifact.py`
- Create: `tests/adapters/test_artifact.py`
- Modify: `scripts/_gpu_job.py`

**Interfaces:**
- Consumes: final `pretrained_model` directory, expected rank/alpha, total/trainable counts parsed from training output, completed step, and target names.
- Produces: `AdapterArtifact`, `validate_adapter_checkpoint(...) -> AdapterArtifact`, and `write_adapter_metadata(path, artifact)`.

- [ ] **Step 1: Write failing artifact validation tests**

```python
def test_valid_lora_artifact_records_digest_and_safe_metadata(tmp_path):
    checkpoint = write_adapter_fixture(tmp_path, r=8, alpha=16, peft_type="LORA")
    artifact = validate_adapter_checkpoint(
        checkpoint, expected_rank=8, expected_alpha=16,
        trainable_parameters=4_000_000, total_parameters=450_000_000,
        completed_step=10, resolved_targets=("model.expert.q_proj",),
    )
    assert artifact.trainable_ratio < 0.10
    assert artifact.adapter_sha256 == sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
    assert "client_id" not in artifact.to_json()

def test_full_model_only_checkpoint_is_rejected(tmp_path):
    write_full_model_fixture(tmp_path)
    with pytest.raises(AdapterArtifactError, match="adapter_model.safetensors"):
        validate_adapter_checkpoint(tmp_path, **expected)
```

Also test empty adapter files, wrong PEFT type/rank/alpha, non-empty modules-to-save, zero trainable parameters, trainable ratio at or above 10%, missing targets, wrong completed step, and non-finite counts.

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `conda run -n fedvla python -m pytest tests/adapters/test_artifact.py -v`
Expected: FAIL because `fcut_vla.adapters.artifact` does not exist.

- [ ] **Step 3: Implement immutable adapter validation**

Read canonical JSON from `adapter_config.json`; accept only `peft_type=LORA`, expected `r`/`lora_alpha`, and absent/empty `modules_to_save`. Require non-empty `adapter_model.safetensors`, positive integer parameter counts, ratio below 0.10, exact requested step, and non-empty sorted resolved targets. Hash adapter bytes with SHA-256. Serialize only technical metadata and reject forbidden adapter-bank identity keys recursively.

- [ ] **Step 4: Integrate the runner completion gate**

After LeRobot exits, locate the latest numeric checkpoint, parse its `train_config.json` and training log parameter counts, validate the adapter, write `adapter_metadata.json`, and only then write `COMPLETE`. A PEFT job must never call the old full-model validator.

- [ ] **Step 5: Run focused and full tests**

Run: `conda run -n fedvla python -m pytest tests/adapters/test_artifact.py tests/integration/test_gpu_cli_dry_run.py -v && conda run -n fedvla python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fcut_vla/adapters/artifact.py scripts/_gpu_job.py tests/adapters/test_artifact.py tests/integration/test_gpu_cli_dry_run.py
git commit -m "feat: validate private LoRA adapter artifacts"
```

### Task 3: Run independent Mac LoRA smoke and prepare CUDA handoff

**Files:**
- Create: `scripts/verify_adapter_run.py`
- Create: `tests/integration/test_verify_adapter_run.py`
- Modify: `docs/gpu_commands.md`
- Modify: `docs/lerobot_runtime.md`

**Interfaces:**
- Consumes: a completed immutable LoRA run directory.
- Produces: `verify_adapter_run.py --run-dir PATH`, a machine-readable verification report, and the exact RTX 4090 smoke command.

- [ ] **Step 1: Write failing run-verification tests**

```python
def test_verify_run_requires_complete_and_matching_adapter_metadata(tmp_path):
    run = write_completed_adapter_run(tmp_path)
    result = subprocess.run([sys.executable, "scripts/verify_adapter_run.py", "--run-dir", str(run)], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["valid"] is True

def test_verify_run_detects_adapter_tampering(tmp_path):
    run = write_completed_adapter_run(tmp_path)
    adapter_path(run).write_bytes(b"changed")
    result = run_verifier(run)
    assert result.returncode != 0
    assert "sha256" in result.stderr.lower()
```

- [ ] **Step 2: Run tests and verify missing-entry-point failure**

Run: `conda run -n fedvla python -m pytest tests/integration/test_verify_adapter_run.py -v`
Expected: FAIL because the verifier does not exist.

- [ ] **Step 3: Implement verifier and update commands**

Validate frozen config/hash, manifest/hash, `COMPLETE`, latest step, adapter metadata, adapter digest, PEFT values, target list, and parameter ratio. Print canonical JSON containing run hash, step, adapter digest/bytes, trainable/total counts and ratio, resolved-target count, device, and `valid=true`. Document Python 3.12, independent official LeRobot install with `[smolvla,dataset,peft]`, user-site isolation, Mac smoke, CUDA smoke, log monitoring, and artifact verification.

- [ ] **Step 4: Install PEFT extra and run the Mac MPS LoRA smoke**

Run:

```bash
conda run -n fedvla python -m pip install -e './.deps/lerobot[smolvla,dataset,peft]'
conda run -n fedvla python scripts/train_client_adapter.py --config configs/local/smolvla_smoke.yaml --output-root runs/lora-smoke
conda run -n fedvla python scripts/verify_adapter_run.py --run-dir "$(find runs/lora-smoke -name COMPLETE -exec dirname {} \; | head -1)"
```

Expected: training reaches 10/10, verification returns `valid=true`, trainable ratio is below 0.10, and adapter PEFT files exist.

- [ ] **Step 5: Run all tests and CUDA dry-run**

Run:

```bash
conda run -n fedvla python -m pytest -q
conda run -n fedvla python scripts/train_client_adapter.py --config configs/gpu/smolvla_smoke.yaml --output-root /tmp/fgkt-lora-cuda --dry-run
git diff --check
```

Expected: all tests PASS and the CUDA command matches the Mac command except device, experiment name, and output path.

- [ ] **Step 6: Commit and push**

```bash
git add scripts/verify_adapter_run.py tests/integration/test_verify_adapter_run.py docs/gpu_commands.md docs/lerobot_runtime.md
git commit -m "test: verify SmolVLA LoRA smoke artifacts"
git push
```
