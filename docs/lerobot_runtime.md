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
python -m pip install -e './.deps/lerobot[smolvla,dataset]'
```

`.deps/` and `runs/` are ignored. All experiment code, configurations,
manifests, logs, and checkpoint validation belong to FGKT-VLA.

## Validated Mac smoke

The independent smoke uses `lerobot/libero_spatial_image`, MPS, batch size 2,
10 optimizer steps, seed 11, and checkpoints at steps 5 and 10. It performs no
environment evaluation and no Hub upload.

```bash
conda activate fedvla
python scripts/train_client_adapter.py \
  --config configs/local/smolvla_smoke.yaml \
  --output-root runs/local-smoke
```

The run is successful only when LeRobot exits with zero, `COMPLETE` exists,
and the final pretrained model contains both `config.json` and
`model.safetensors`.
