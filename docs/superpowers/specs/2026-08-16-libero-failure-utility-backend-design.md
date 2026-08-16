# LIBERO Failure and Counterfactual Utility Backend Design

## Scope

Implement the executable backend for FCUT-VLA stages B and C: deterministic
LIBERO failure collection and paired counterfactual utility-label generation.
The backend integrates official LeRobot and LIBERO directly inside FGKT-VLA,
without importing another research project or creating another simulator/checkpoint format. Utility-ranker training, personalized repair
evaluation, continual orchestration, and real-robot execution remain outside
this design.

## Reused evaluation contract

The GPU backend uses the following pinned FGKT-VLA conventions:

- LeRobot 0.6.1 and its SmolVLA policy loader;
- synchronous LIBERO vector environments;
- `MUJOCO_GL=egl` on the RTX 4090 server;
- `observation.images.image2` renamed to
  `observation.images.wrist_image`;
- explicit Hugging Face policy revisions under `phawitbinabik`;
- deterministic seeds and complete `eval_info.json` outputs;
- no parallel simulator, checkpoint serializer, or dataset implementation.

The FGKT-VLA evaluator is implemented in this repository using official LeRobot
and LIBERO APIs with an episode recorder hook. The hook observes evaluator inputs and outputs but does not
change actions, resets, termination, success computation, or environment
ordering.

## Components

### Task resolver

The resolver maps each frozen FedLIBERO-Fail task alias to exactly one
installed LIBERO problem identifier. Zero matches and multiple matches are hard
errors. For each resolved task, the run manifest records the alias, problem
identifier, LIBERO commit, BDDL file SHA-256, and initial-state file SHA-256.
Resolution occurs before GPU rollout begins, so a partial run cannot hide an
ambiguous benchmark definition.

### Episode recorder

For every episode, the recorder writes an append-only JSON Lines stream under
`rollouts/<policy-revision>/<seed>/episodes.jsonl`. Each record contains:

- schema version, task alias, resolved problem identifier, seed, episode index,
  initial-state index and initial-state hash;
- policy repository and immutable revision;
- instruction and terminal success;
- ordered steps containing sanitized visual features, proprioception, executed
  action, action-distribution statistics, reward, done flag, and success flag;
- episode-content SHA-256.

Raw RGB frames and simulator states are optional private artifacts and are
never serialized into ranker inputs. A completed seed shard receives a sidecar
hash and completion marker. Resume skips only shards whose content hash and
metadata validate.

### Failure extractor

Only terminally unsuccessful episodes produce a `FailureContext`. The failure
step is the final valid decision step unless the evaluator supplies an earlier
explicit failure event. The extractor uses the frozen context window size,
left-pads short contexts, and validates finite, fixed-dimensional step fields.
Outputs are stored in `failures/failure_contexts.jsonl` with source episode
hashes. Successful episodes, incomplete episodes, or records lacking the
required step fields are rejected rather than silently converted.

### Paired counterfactual runner

For every target failure and candidate adapter, the runner evaluates baseline
and candidate policies on identical tuples of task, evaluation seed,
initial-state index, initial-state hash, episode budget, and environment
settings. Pair construction is deterministic and written before execution.
Missing members, duplicate members, seed changes, initial-state changes, or
policy revisions that do not match the frozen plan are hard errors.

Candidate evaluation produces recovery outcomes for the failed target and
retention outcomes for the frozen prior-task set. Communication bytes and
latency are measured and retained separately from task performance.

### Utility-label builder

The builder consumes only complete paired-evaluation records. It calls the
existing `compute_utility` implementation to produce recovery gain, retention
risk, normalized cost, utility, and a paired-bootstrap confidence interval.
The output is an append-only `utility_labels/labels.jsonl` ledger keyed by
failure-content hash, candidate-adapter digest, benchmark-manifest hash, and
pair-plan hash.

`no_match` is assigned only after all eligible candidates for a failure have
been evaluated and none has a positive utility lower confidence bound. It is
an evaluation label and is never exposed as a utility-ranker feature.

## Artifact layout

```text
runs/<experiment>/<stage>/<run-hash>/
  frozen_config.yaml
  run_manifest.json
  command.sh
  task_resolution.json
  pair_plan.json
  rollouts/<policy-revision>/<seed>/episodes.jsonl
  rollouts/<policy-revision>/<seed>/SHARD.json
  failures/failure_contexts.jsonl
  paired_eval/<failure-hash>/<adapter-digest>/<seed>.json
  utility_labels/labels.jsonl
  logs/stdout.log
  logs/stderr.log
  COMPLETE
```

`COMPLETE` is written last. Existing completed runs are immutable. Resume may
continue only an incomplete run whose frozen config, git commit, benchmark
manifest, task resolution, pair plan, policy revisions, and existing shard
hashes match.

## Privacy and leakage boundary

Raw images, raw simulator states, raw trajectories, task identifiers, client
identifiers, source-client ownership, and private adapter paths may exist only
inside rollout and evaluation ledgers. Utility-ranker examples contain the
sanitized `FailureContext`, identity-free `AdapterDescriptor`, numeric utility
targets, masks, and content hashes. A schema validator rejects forbidden keys
recursively before an example is written.

## Error handling

The backend fails before or during execution for ambiguous tasks, unavailable
policy revisions, mismatched paired seeds, changed initial states, malformed or
non-finite steps, duplicate episode keys, incomplete paired evaluations,
artifact hash mismatches, and attempts to modify completed runs. Failures leave
logs and incomplete shards in place for diagnosis. They do not write
`COMPLETE`, utility labels, or synthetic substitute metrics.

## Testing strategy

Mac tests use fixture evaluator outputs and never import MuJoCo, CUDA, LeRobot,
or SmolVLA. Unit tests cover unique task resolution, episode schema validation,
successful-episode exclusion, context-window padding, recursive privacy checks,
exact pair keys, paired-seed preservation, initial-state mismatch rejection,
utility computation, `no_match`, shard hashes, immutable completion, and resume.

An integration test runs stages B and C against a deterministic fake evaluator,
then verifies byte-stable artifacts across two output roots. A GPU preflight
test checks the rendered FGKT-VLA/LeRobot command, pinned policy revision,
rename map, EGL, synchronous environments, and output locations without
allocating GPU memory. Real GPU execution is enabled only after all fixture and
preflight tests pass.

## Acceptance criteria

- A failed fixture episode becomes one valid, hash-addressed failure context;
- a successful fixture episode never becomes a failure;
- baseline and candidate records cannot be paired unless their complete pair
  keys match;
- repeated fixture runs produce identical task resolution, pair plan, failure
  context, label, and manifest hashes;
- utility labels contain no forbidden identity or raw-data fields;
- a fully evaluated failure with no positive lower bound is labeled
  `no_match`;
- incomplete or corrupted runs can resume safely, while completed runs cannot
  be overwritten;
- the GPU command uses the FGKT-VLA evaluation entry point and explicitly
  pins all environment- and policy-defining inputs.
