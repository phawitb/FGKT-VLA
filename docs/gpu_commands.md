# RTX 4090 execution guide

This workflow reuses the LeRobot/LIBERO conventions validated in CausalVLA:
LeRobot 0.6.1, `lerobot/libero_spatial_image`, `MUJOCO_GL=egl`, paired evaluation
seeds, and Hugging Face namespace `phawitbinabik`. Tokens must remain in the
user credential store and must never be written to this repository.

## 1. Server environment

```bash
git clone https://github.com/phawitb/FGKT-VLA.git
cd FGKT-VLA
git switch codex/fcut-vla-paper

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Reuse the same LeRobot checkout/version as CausalVLA.
git clone https://github.com/huggingface/lerobot.git ../lerobot
git -C ../lerobot checkout v0.6.1
python -m pip install -e '../lerobot[smolvla,libero]'

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

## 3. Stage A: client adapter training

Inspect `command.sh` from the dry run, then launch through the entry point:

```bash
python scripts/train_client_adapter.py \
  --config configs/gpu/development.yaml \
  --output-root runs
```

This launches `lerobot-train` with SmolVLA, the public LIBERO image dataset,
Hub upload under `phawitbinabik/fcut-vla-dev-client-adapter`, EGL rendering,
and a run-local checkpoint directory.

## 4. Stages B-D safety status

Failure generation, counterfactual-label construction, ranker training, repair
evaluation, and continual orchestration currently support **dry-run only**.
Their immutable CLI contracts are frozen, but their LeRobot rollout backends
must be implemented and tested before removing the execution guard. This is
intentional: the pipeline must not emit placeholder rollouts or fabricated
paper metrics.

The development configuration is for integration checks. Do not launch
`configs/gpu/main.yaml` until development runs produce paired artifacts and
pass retention-gate verification.

## 5. Resume and failure recovery

```bash
# Resume an incomplete stage without changing its frozen identity.
python scripts/train_client_adapter.py \
  --config configs/gpu/development.yaml \
  --output-root runs \
  --resume

# Inspect logs and the exact rendered command.
find runs/fcut-vla-development -name command.sh -o -name stderr.log -o -name stdout.log
```

Never delete or edit an existing completed run. Change the configuration to
obtain a new hash. Preserve the frozen config, manifest, paired seed lists,
logs, checkpoints, metrics, and `COMPLETE` marker together when transferring a
run back from the GPU server.
