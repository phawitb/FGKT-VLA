# RTX 4090 execution guide

This is an independent FGKT-VLA workflow using official LeRobot 0.6.1,
`lerobot/libero_spatial_image`, `MUJOCO_GL=egl`, paired evaluation seeds, and
Hugging Face namespace `phawitbinabik`. It does not import or invoke CausalVLA.
Tokens remain in the user credential store and are never written here.

## 1. Server environment

```bash
git clone https://github.com/phawitb/FGKT-VLA.git
cd FGKT-VLA
git switch codex/fcut-vla-paper

conda create -y -n fedvla python=3.12 pip
conda activate fedvla
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Install an independent official LeRobot checkout.
git clone --branch v0.6.1 --depth 1 https://github.com/huggingface/lerobot.git .deps/lerobot
PYTHONNOUSERSITE=1 python -m pip install -e './.deps/lerobot[smolvla,dataset,peft]'

hf auth whoami
nvidia-smi
```

If the server has a dedicated Hugging Face cache, set `HF_HOME` to an explicit
server path. Do not copy the Mac keychain token into scripts or YAML files.

## 2. Mac or server preflight

The following commands allocate no GPU memory. Each creates a deterministic
run directory with `frozen_config.yaml`, `run_manifest.json`, `command.sh`, and
`logs/`.

```bash
python scripts/train_client_adapter.py --config configs/gpu/development.yaml --output-root runs --dry-run
python scripts/generate_failures.py --config configs/gpu/development.yaml --output-root runs --dry-run
python scripts/build_utility_labels.py --config configs/gpu/development.yaml --output-root runs --dry-run
python scripts/train_utility_ranker.py --config configs/gpu/development.yaml --output-root runs --dry-run
python scripts/run_repair_eval.py --config configs/gpu/development.yaml --output-root runs --dry-run
python scripts/run_continual_experiment.py --config configs/gpu/development.yaml --output-root runs --dry-run
```

Use a fresh output root for another preflight, or add `--resume` to reopen an
incomplete planned run. A directory containing `COMPLETE` is immutable and is
never overwritten.

## 3. RTX 4090 smoke test

Inspect `command.sh` from the dry run, then launch through the entry point:

```bash
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 python scripts/train_client_adapter.py \
  --config configs/gpu/smolvla_smoke.yaml \
  --output-root runs/lora-smoke
```

This runs the same bounded 10-step training contract that passed on the Mac,
changing only `mps` to `cuda`. The rendered command loads
`lerobot/smolvla_base`, derives input/output features from the LIBERO dataset,
and trains native LeRobot LoRA rank 8 / alpha 16. Both Hub inputs are pinned to
immutable revisions, and runtime provenance is checked against LeRobot commit
`7e241bd6` plus PEFT `0.20.0`. FGKT-VLA resolves the base revision to a local
immutable Hub snapshot before LeRobot reads its config or weights, and rejects
a dirty `.deps/lerobot` checkout. It does not upload to the Hub.

Monitor and verify the final adapter:

```bash
LOG=$(find runs/lora-smoke/fgkt-vla-gpu-smoke -name stderr.log | head -1)
tail -f "$LOG"

RUN=$(find runs/lora-smoke/fgkt-vla-gpu-smoke -name COMPLETE -exec dirname {} \; | head -1)
PYTHONNOUSERSITE=1 python scripts/verify_adapter_run.py --run-dir "$RUN"
```

Do not proceed unless verification prints `"valid":true`, the completed step is
10, the trainable ratio is below 0.10, and the checkpoint contains
`adapter_config.json` plus `adapter_model.safetensors` rather than a full-model
`model.safetensors`.

## 4. Install LIBERO and run one bounded rollout

Training does not require the `libero` extra. Install it only after adapter
verification. On Ubuntu, LeRobot issue 3397 requires CMake below version 4 and
building both EGL probe packages without pip build isolation:

```bash
python -m pip uninstall -y egl-probe hf-egl-probe robomimic hf-libero

# TLJH may put a stale ~/.local/bin/cmake before the active conda environment.
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:$PATH"
hash -r
PYTHONNOUSERSITE=1 python -m pip install --force-reinstall 'cmake>=3.29,<4'
test "$(command -v cmake)" = "$CONDA_PREFIX/bin/cmake"
cmake --version

CMAKE_POLICY_VERSION_MINIMUM=3.5 PYTHONNOUSERSITE=1 \
python -m pip install --no-build-isolation --no-cache-dir \
  --no-binary egl_probe,hf_egl_probe \
  egl-probe==1.0.2 hf-egl-probe==1.0.2

CMAKE_POLICY_VERSION_MINIMUM=3.5 PYTHONNOUSERSITE=1 \
python -m pip install -e './.deps/lerobot[libero]'

python -m pip check
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONNOUSERSITE=1 python - <<'PY'
import egl_probe
import libero
import mujoco
print("LIBERO/EGL imports: OK")
PY
```

Render and inspect the exact official LeRobot command first. The default is
deliberately bounded to LIBERO-Spatial task 0, seed 101, and one episode:

```bash
RUN=runs/lora-smoke-final/fgkt-vla-gpu-smoke/train_client_adapter/e05aed2369daaf8e

PYTHONNOUSERSITE=1 python scripts/evaluate_libero_smoke.py \
  --adapter-run "$RUN" \
  --output-dir runs/libero-eval-smoke/spatial-task0-seed101 \
  --suite libero_spatial --task-id 0 --seed 101 --episodes 1 --dry-run

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 python scripts/evaluate_libero_smoke.py \
  --adapter-run "$RUN" \
  --output-dir runs/libero-eval-smoke/spatial-task0-seed101 \
  --suite libero_spatial --task-id 0 --seed 101 --episodes 1
```

The runner refuses an existing output directory and accepts the checkpoint
only after `verify_adapter_run.py` succeeds. It reports the official
`eval_info.json` success result on a zero-to-one scale.

After that evaluator smoke succeeds, record one authoritative sanitized Stage
B episode. This uses the same official LeRobot environment, PEFT policy, and
processors, but stores only fixed numeric image moments, the eight-value policy
state, executed actions, action summaries, rewards, and terminal flags:

```bash
EPISODE=runs/libero-failure-smoke/spatial-task0-seed101/episode.json

PYTHONNOUSERSITE=1 python scripts/record_libero_episode.py \
  --adapter-run "$RUN" --output "$EPISODE" \
  --suite libero_spatial --task-id 0 --seed 101 \
  --episode-index 0 --initial-state-index 0 --dry-run

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
python scripts/record_libero_episode.py \
  --adapter-run "$RUN" --output "$EPISODE" \
  --suite libero_spatial --task-id 0 --seed 101 \
  --episode-index 0 --initial-state-index 0
```

The command refuses an existing episode file, derives the canonical policy
namespace and task alias internally, hashes the exact selected initial-state
array, validates seven finite action values before every simulator step, and
forces a failed terminal record at the frozen task horizon. `image_moments_v1`
is a pipeline smoke extractor, not the learned visual representation intended
for final paper experiments.

Finalize the recorded episode into an immutable Stage B failure shard without
using the GPU again:

```bash
PYTHONNOUSERSITE=1 python scripts/build_failure_shard.py \
  --episode "$EPISODE" \
  --output-dir runs/libero-failure-smoke/spatial-task0-seed101/failure-shard \
  --window-size 16
```

The 16-step smoke window covers 0.8 seconds at LIBERO's fixed 20 Hz. The
command writes `failures.jsonl`, `failures.sha256`, and `SHARD.json`. Re-running
requires `--resume`, which revalidates the exact episode content, window size,
and artifact bytes rather than overwriting them.

## 5. Stages B-D safety status

The bounded single-adapter LIBERO evaluator above is executable. Full failure
generation, paired counterfactual collection, ranker training, repair
evaluation, and continual orchestration still support **dry-run only**. Their
immutable CLI contracts and privacy-safe paired-label primitives are frozen,
but the full rollout recorder backend must be implemented and tested before
removing those execution guards. This is intentional: the pipeline must not
emit placeholder rollouts or fabricated paper metrics.

The development configuration is for integration checks. Do not launch
`configs/gpu/main.yaml` until development runs produce paired artifacts and
pass retention-gate verification.

## 6. Resume and failure recovery

```bash
# Resume an incomplete stage without changing its frozen identity.
python scripts/train_client_adapter.py \
  --config configs/gpu/development.yaml \
  --output-root runs \
  --resume

# Inspect logs and the exact rendered command.
find runs/fgkt-vla-development -name command.sh -o -name stderr.log -o -name stdout.log
```

Never delete or edit an existing completed run. Change the configuration to
obtain a new hash. Preserve the frozen config, manifest, paired seed lists,
logs, checkpoints, metrics, and `COMPLETE` marker together when transferring a
run back from the GPU server.
