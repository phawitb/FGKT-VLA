# SmolVLA Client LoRA Design

## Scope

Replace the full-fine-tuning client smoke path with native LeRobot 0.6.1
PEFT/LoRA training. This design covers client adapter creation, validation,
metadata, and Mac/RTX 4090 smoke tests. Adapter ranking, composition, LIBERO
failure rollouts, and continual promotion consume the resulting adapters but
are not changed here.

## Independence boundary

FGKT-VLA invokes official LeRobot 0.6.1 from its private `.deps/lerobot`
checkout. It does not import, invoke, or read runtime artifacts from another
research project. Every job disables Python user-site packages and writes only
inside its hashed FGKT-VLA run directory.

## Training method

Client training uses LeRobot's native `PeftConfig` and SmolVLA
`wrap_with_peft` path:

- method: `LORA`;
- rank: 8 for smoke/development unless an ablation overrides it;
- alpha: 16, giving scaling `alpha/rank = 2`;
- target modules: SmolVLA's native default target expression;
- modules saved in full: empty list;
- base: official SmolVLA configuration with pretrained VLM weights;
- smoke: batch size 2, 10 optimizer steps, seed 11, checkpoints at 5 and 10;
- smoke Hub upload: disabled;
- local device: MPS; server device: CUDA.

The native target expression adapts expert attention `q_proj/v_proj` plus
SmolVLA state, action, and time projections. The exact resolved target module
names are recorded in the run metadata. A change in target expression, resolved
targets, rank, alpha, base revision, or training data produces a different run
identity.

## Artifact contract

Each client checkpoint must contain PEFT-native adapter files, including
`adapter_config.json` and `adapter_model.safetensors`, plus the LeRobot
preprocessor/postprocessor and training state needed for resume. The run
manifest records:

- base policy repository and immutable revision;
- LeRobot commit and PEFT version;
- dataset repository and immutable revision when available;
- rank, alpha, target expression, resolved target names, and modules to save;
- total and trainable parameter counts;
- adapter byte size and SHA-256;
- seed, completed step, device type, config hash, manifest hash, and git commit.

The client adapter bank stores only the adapter digest, private path, and
identity-free descriptor. It does not expose client ownership or private paths
to the utility ranker.

## Validation gates

A LoRA checkpoint is complete only when all of the following hold:

1. LeRobot exits with code zero and reaches the requested optimizer step.
2. `adapter_config.json` and `adapter_model.safetensors` exist and are non-empty.
3. The adapter config reports PEFT type `LORA`, rank 8, alpha 16, and no
   full-training modules for the smoke configuration.
4. At least one LoRA parameter is trainable and every non-adapter parameter is
   frozen, except an explicitly declared module-to-save (none in smoke).
5. Trainable parameters are less than 10% of total policy parameters.
6. Resolved target modules are non-empty and match the frozen target contract.
7. Reloading the base plus adapter reproduces the saved trainable state keys.
8. `COMPLETE` is written only after these checks pass.

A full-model `model.safetensors` is not accepted as the client adapter payload.
The infrastructure full-fine-tuning smoke remains preserved as evidence but is
not used in adapter-transfer experiments.

## Configuration and command behavior

LoRA settings live under the existing `lora` YAML mapping. The job renderer
maps them to native LeRobot arguments:

```text
--peft.method_type=LORA
--peft.r=8
--peft.lora_alpha=16
--peft.full_training_modules=[]
```

The target-module argument is omitted for the primary method so SmolVLA's
version-pinned native default is used. Ablations may set an explicit target
list, which must be serialized deterministically and changes the run hash.

## Testing

Mac unit tests first verify command rendering, parameter-count validation,
adapter-file validation, config validation, rejection of full-model-only
checkpoints, and identity-free metadata. A 10-step MPS smoke then verifies
native PEFT wrapping, backward/optimizer execution, checkpoint serialization,
and adapter reload. The RTX 4090 smoke uses the same configuration except for
device and output root. Development training is blocked until both smokes pass.

## Acceptance criteria

- Mac MPS and RTX 4090 CUDA each complete the same 10-step native LoRA smoke;
- trainable parameters are positive and below 10% of total parameters;
- final checkpoints contain validated PEFT adapter artifacts rather than a
  client-sized full-model payload;
- adapter SHA-256 and resolved target modules are recorded reproducibly;
- no CausalVLA source, path, checkpoint, or configuration is referenced;
- the full test suite passes with completed-run immutability preserved.
