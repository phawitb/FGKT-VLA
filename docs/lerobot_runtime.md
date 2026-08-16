# Independent LeRobot runtime

FGKT-VLA uses the official Hugging Face LeRobot repository at tag `v0.6.1`.
It has no runtime, source, checkpoint, configuration, or filesystem dependency
on CausalVLA or any other local research project.

## Local environment

```bash
conda create -n fedvla python=3.12 pip
conda activate fedvla
git clone --branch v0.6.1 --depth 1 \
  https://github.com/huggingface/lerobot.git .deps/lerobot
python -m pip install -e '.[dev]'
PYTHONNOUSERSITE=1 python -m pip install -e './.deps/lerobot[smolvla,dataset,peft]'
```

`.deps/` and `runs/` are ignored. All experiment code, configurations,
manifests, logs, and checkpoint validation belong to FGKT-VLA.

## Validated Mac smoke

The independent smoke uses `lerobot/libero_spatial_image`, MPS, batch size 2,
10 optimizer steps, seed 11, and checkpoints at steps 5 and 10. It performs no
environment evaluation and no Hub upload.

```bash
conda activate fedvla
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 python scripts/train_client_adapter.py \
  --config configs/local/smolvla_smoke.yaml \
  --output-root runs/lora-smoke
```

The command loads the official `lerobot/smolvla_base` weights and lets LeRobot
derive the visual/action feature schema from `lerobot/libero_spatial_image`.
The base config and weights come from a revision-pinned local Hub snapshot; the
adapter records the public base repo plus commit SHA for portable reload.
The validated Mac run completed 10/10 steps with 371,328 trainable parameters
out of 450,417,504 (0.0824%), 37 resolved LoRA targets, and a 1,496,448-byte
adapter payload.

The run is successful only when LeRobot exits with zero, `COMPLETE` exists,
and this command returns `"valid":true`:

```bash
RUN=$(find runs/lora-smoke/fgkt-vla-local-smoke -name COMPLETE -exec dirname {} \; | head -1)
PYTHONNOUSERSITE=1 python scripts/verify_adapter_run.py --run-dir "$RUN"
```

The verifier checks the frozen-config and benchmark hashes, immutable Hub and
runtime revisions, artifact-bound completion marker, latest training step,
PEFT rank/alpha/type, recomputed parameter ratio and target list, byte count,
and SHA-256 digest. A valid client payload contains `adapter_config.json` and
`adapter_model.safetensors`; a full-model-only checkpoint is rejected.
