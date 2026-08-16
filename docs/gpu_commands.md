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
python -m pip install -e './.deps/lerobot[smolvla,dataset,libero]'

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
python scripts/train_client_adapter.py \
  --config configs/gpu/smolvla_smoke.yaml \
  --output-root runs
```

This runs the same bounded 10-step training contract that passed on the Mac,
changing only `mps` to `cuda`. It does not upload to the Hub. Verify the step-10
checkpoint before running `configs/gpu/development.yaml`.

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
find runs/fgkt-vla-development -name command.sh -o -name stderr.log -o -name stdout.log
```

Never delete or edit an existing completed run. Change the configuration to
obtain a new hash. Preserve the frozen config, manifest, paired seed lists,
logs, checkpoints, metrics, and `COMPLETE` marker together when transferring a
run back from the GPU server.
