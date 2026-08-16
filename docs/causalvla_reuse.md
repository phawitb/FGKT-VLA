# CausalVLA Integration Reuse

FCUT-VLA reuses the validated LIBERO/SmolVLA operational workflow from `/Users/phawit/Projects/CausalVLA` rather than creating a second training stack.

## Reusable contracts

- LeRobot version: 0.6.1.
- Primary training dataset family: `lerobot/libero_*_image`.
- Local smoke test: `lerobot_train`, SmolVLA, MPS, `num_workers=0`, absolute output path.
- CUDA evaluation: `MUJOCO_GL=egl`, deterministic seed, synchronous environment mode, and `eval_info.json` as the completion artifact.
- Observation rename used by CausalVLA evaluation: `observation.images.image2` to `observation.images.wrist_image`.
- Completed jobs are detected by a non-empty `eval_info.json`; output directories are never silently overwritten.
- Hub artifacts are pinned by revision rather than floating model names.

## FCUT-specific changes

- Do not patch SmolVLA with CausalVLA policy variants for the first FCUT baseline; load the official SmolVLA policy and add LoRA through a separate adapter boundary.
- Replace one monolithic dataset with client-stage filtered views defined by the frozen FedLIBERO-Fail manifest.
- Extend evaluation output with episode seed, task alias resolution, failure step, failure-context hash, adapter ID, base-checkpoint revision, and retention-task results.
- Preserve CausalVLA's resumable job and logging pattern for training, failure generation, and repair evaluation.
- Reuse the installed LIBERO task registry to resolve manifest aliases. Alias resolution must be one-to-one and recorded before any GPU job.

## Source files reviewed

- `/Users/phawit/Projects/CausalVLA/PIPELINE.md`
- `/Users/phawit/Projects/CausalVLA/worklog/phase1.md`
- `/Users/phawit/Projects/CausalVLA/scripts/run_eval_gpu.sh`
- `/Users/phawit/Projects/CausalVLA/scripts/summarize_eval.py`
- `/Users/phawit/Projects/CausalVLA/setup_colab.sh`

