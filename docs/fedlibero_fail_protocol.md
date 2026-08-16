# FedLIBERO-Fail v1 Protocol

FedLIBERO-Fail is a logical continual-federated split over LIBERO task suites. Version 1 defines five clients, four ordered stages per client, explicit skill annotations, and train/validation/test partitions that are disjoint by complete task-skill combination, client-stage-task assignment, and environment initial-state seed.

## Important integration rule

Task entries are stable research aliases with a suite and natural-language instruction. Before GPU execution, `generate_failures.py` must resolve every alias to an installed LIBERO problem identifier and record that identifier, LIBERO commit, BDDL hash, and initial-state file hash in the immutable run manifest. An unresolved or multiply matched alias is a hard error.

## Leakage boundary

The utility ranker may consume failure context, target context, skill prototypes, update sketches, reliability, and embodiment compatibility. It may not consume task IDs, source-client IDs, client ownership, or assignment keys. These identifiers exist only in the evaluation ledger.

## Failure strata

- `seen_task`: an exact evaluated task has a source adapter trained on the same task.
- `compositional`: no exact source exists, but component skills occur across multiple source adapters.
- `no_match`: paired counterfactual evaluation finds no candidate with a positive lower utility confidence bound.

Strata labels are assigned from the frozen training ledger and realized counterfactual utilities. They are evaluation labels, not ranker inputs.

## Versioning

Any change to tasks, skills, assignments, splits, seeds, or ranker features creates a new manifest version and hash. Results from different hashes may not be pooled as one experiment.
