# Failure-Aligned Repair Candidate Design

**Date:** 2026-08-18  
**Status:** Approved design, pending implementation plan  
**Scope:** One bounded LIBERO-Spatial Stage C backend smoke before scaling to the
full FGKT-VLA counterfactual adapter bank.

## Objective

Produce a repair candidate that is genuinely distinct from the adapter that
generated a recorded failure, then evaluate the two policies under an exact
paired LIBERO intervention. The resulting artifact may enter utility-label
construction only when all training and rollout identities are verified.

This stage validates the counterfactual backend. It is not a paper result and
does not establish the effectiveness of the final failure-guided transfer
method.

## Chosen intervention

The failed adapter is the baseline policy. The candidate starts from that exact
verified PEFT checkpoint and receives a bounded continuation update using only
demonstration episodes whose task identity matches the failed episode. This is
called `failure_aligned_continuation_v1`.

The first smoke uses the existing failed LIBERO-Spatial task-0 episode, a small
fixed training budget, and a deterministic list of matching demonstration
episode indices. Candidate training must produce a new adapter digest. A copied
checkpoint, unchanged digest, weight perturbation, or adapter independently
trained from the SmolVLA base is not an eligible repair candidate.

## Immutable repair recipe

The recipe binds:

- source failed-adapter run hash and verified adapter SHA-256;
- source failure hash and source episode hash;
- LIBERO suite, problem ID, canonical instruction, and task alias;
- dataset repository and immutable revision;
- the sorted demonstration episode indices and their canonical selection hash;
- base policy repository and immutable revision inherited through the source
  adapter;
- LeRobot commit, PEFT version, hf-libero version, Python version, and PyTorch
  version;
- optimizer, learning rate, batch size, seed, continuation step count, LoRA
  target contract, and output format;
- schema version and canonical recipe content hash.

Task matching is resolved from dataset metadata, not inferred from array order
or user-entered episode indices. Exactly one dataset task must match the frozen
failure instruction after the existing normalization rules. An empty or
ambiguous match stops before training.

## Training semantics

Training loads the verified failed adapter as the initial policy and preserves
its pinned underlying SmolVLA base. Only existing PEFT trainable parameters may
change. The selected task demonstrations are passed through LeRobot's native
episode filtering; no private failure images or simulator states are added to
the training dataset.

The smoke configuration is intentionally small and separate from paper
configurations. It records pre- and post-training adapter digests, parameter
counts, resolved target modules, exact command, logs, checkpoints, and a
completion manifest. Completion requires a distinct output digest and the same
structural adapter contract as the source checkpoint.

Resume is identity-based. It validates the recipe and every completed artifact;
it never overwrites or silently reuses a directory created from different
inputs.

## Pair-plan semantics

The label-producing plan uses `purpose=counterfactual_repair` and therefore
`label_eligible=true`. Its baseline is the failed adapter, not the original
SmolVLA base. Its candidate is the verified continuation adapter. The plan
binds both policy identities and requires different adapter digests.

Each `PairKey` freezes:

- failure hash and candidate adapter digest;
- problem ID;
- evaluation seed;
- initial-state index and exact initial-state hash;
- episode budget;
- canonical environment hash.

The baseline and candidate members share the complete `PairKey`; only their
policy repository/revision may differ. For the first smoke, both policies are
local verified adapters represented by the canonical logical adapter repository
and their immutable weight digests.

## Paired rollout execution

The runner evaluates one pair synchronously with official LeRobot/LIBERO APIs:

1. Validate the repair recipe, source adapter, candidate adapter, failure shard,
   and pair plan before creating a simulator.
2. Run the baseline with the frozen seed and selected initial state.
3. Close the environment and release policy resources.
4. Recreate the same canonical environment and run the candidate with the exact
   same seed and initial state.
5. Validate both sanitized `EpisodeRecord` artifacts and their source
   identities.
6. Write a canonical, immutable paired-result envelope and completion marker.

The runner fails closed if task, seed, initial-state hash, horizon, environment,
policy digest, or completion status differs. It does not substitute missing
metrics and does not reuse the earlier failure episode as the new baseline
rollout.

## Utility result

For the bounded smoke, recovery observations are paired binary successes from
the newly executed baseline and candidate episodes. Communication cost is the
verified candidate adapter byte size. Latency is measured for candidate loading
and rollout with the measurement definition frozen in the result envelope.
Retention is not fabricated: utility-label emission remains blocked until a
small, explicit retention task set is executed with both policies under paired
seeds. The paired rollout backend may complete before the utility label does.

The utility builder must locate exactly one matching `PairKey` in the validated
plan. Caller-supplied eligibility, unrelated failures, unrelated adapter
digests, self-replay plans, incomplete pairs, and missing retention results are
rejected.

## Privacy boundary

Authoritative Stage C artifacts contain only canonical identities, sanitized
numeric episode fields, scalar metrics, hashes, adapter descriptors, timing,
and byte counts. Raw images, raw simulator states, private filesystem paths,
client ownership, task ownership, and trajectory media never enter the utility
ranker ledger.

## Interfaces

Planned entry points:

- `scripts/build_repair_candidate.py`: resolve the matching dataset episodes,
  freeze/render the repair recipe, launch or resume continuation training, and
  verify the resulting adapter.
- `scripts/build_counterfactual_pair_plan.py`: construct an eligible plan from
  a verified failed adapter and a distinct verified candidate.
- `scripts/run_counterfactual_pair.py`: execute and validate the paired LIBERO
  rollouts, producing sanitized episode files and a paired-result envelope.

Pure helpers will own recipe hashing, dataset-task episode resolution, candidate
verification, and result-envelope validation so Mac fixtures can cover the full
contract without CUDA or LIBERO.

## Test strategy

Tests must prove:

- normalized task matching resolves one deterministic episode list and rejects
  zero or ambiguous matches;
- recipe hashes change for any source adapter, failure, dataset revision,
  episode selection, hyperparameter, or runtime change;
- candidate verification rejects identical source/candidate digests, changed
  LoRA target contracts, and incomplete checkpoints;
- pair plans reject a SmolVLA-base baseline for repair purpose and reject
  baseline/candidate identity reuse;
- paired execution uses the same seed and initial-state hash, closes resources
  after either failure, and resumes only byte-identical completed output;
- utility construction remains blocked without complete paired retention
  results;
- command rendering is deterministic on Mac, and the full repository test suite
  passes before a server command is issued.

## Acceptance gates

The Mac gate is deterministic command rendering, fixture-backed artifact
generation, negative provenance tests, and the full test suite. The RTX 4090
gate is one verified continuation checkpoint with a new digest followed by one
complete paired LIBERO rollout. Only after those pass may the retention smoke
and first eligible utility label be implemented.

No success-rate, recovery, retention, or utility number from this bounded smoke
may be presented as a paper result.
