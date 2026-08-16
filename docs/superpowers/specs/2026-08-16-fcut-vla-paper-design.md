# FCUT-VLA Paper Design Specification

Date: 2026-08-16

Status: Design approved through the experimental-protocol section; remaining technical decisions delegated by the researcher.

## 1. Research objective

Develop and evaluate a publishable method for safe, personalized repair of continually trained federated vision-language-action policies. When a deployed global policy fails at a target robot, the system must use the failed rollout as context to rank transferable knowledge from remote clients by expected recovery benefit, repair the target client first, and consolidate the repair globally only when retained capabilities are not materially degraded.

The target standard is a Q1 robotics/ML journal or a top robotics/ML conference. The work must not invent experimental results and must separate proposed claims from empirically established claims.

## 2. Working title and terminology

Preferred title:

**Failure-Conditioned Transfer Utility for Safe Continual Federated Vision-Language-Action Learning**

Preferred acronym: **FCUT-VLA**.

The acronym FGKT-VLA may be retained internally but is not recommended for submission because “knowledge transfer” is too broad and overlaps FedWeIT/FedSeIT. “Transfer utility” names the measurable object introduced by this work.

Key terms:

- **Failure context:** a bounded temporal window containing observations, proprioception, language instruction, proposed/executed actions, and outcome surrounding a failed rollout.
- **Transferable object:** a client LoRA adapter or low-rank parameter delta trained from a common base checkpoint.
- **Transfer utility:** counterfactual recovery gain after applying a transferable object, penalized by retention regression and transfer cost.
- **Personalized repair:** an update applied only to the client that experienced the failure.
- **Safe consolidation:** promotion of a validated personalized repair into the shared global adapter after retention checks.

## 3. Research gap

### 3.1 What prior work already covers

- FedWeIT and FedSeIT already perform weighted/selective inter-client transfer in federated continual learning.
- FedSeIT already represents client tasks, computes domain similarity, and selects top-K historical task parameters.
- FedSaC already combines similarity and complementarity for personalized client collaboration.
- FedVLA already uses expert-selection similarity for layer-wise federated aggregation.
- ForgeVLA already uses task prototypes, contrastive planning, and update-aware adaptive aggregation.
- Stellar VLA and CLARE already route task/observation features to reusable experts or adapters in continual VLA.
- Failing Forward, RoboFAC, REFLECT, RACER, and related work already use failures for negative guidance, diagnosis, explanation, or recovery.

### 3.2 Missing capability

Existing work does not jointly address the following measurable problem:

1. A deployed VLA produces a failed rollout at a target federated client.
2. Several remote clients expose potentially useful but heterogeneous adapters without sharing raw trajectories.
3. The server or target client must estimate the *counterfactual post-transfer effect* of each adapter for this particular failure.
4. The repair should improve fresh rollouts from the failure neighborhood rather than merely retry the same state.
5. The repair must not materially damage retained or unrelated capabilities.
6. Only validated repairs may affect the global model.

The gap is therefore **failure-conditioned transfer-utility estimation with retention-constrained personalized repair**, not selective transfer in general.

### 3.3 Claims explicitly prohibited

The paper must not claim that:

- it is the first selective inter-client transfer method;
- prior federated VLA methods only use data-size weighting;
- failures have not been used to train or recover VLA policies;
- task/skill routing for continual VLA is new;
- semantic similarity is equivalent to transferable utility;
- privacy follows merely from not sharing raw data.

## 4. Research questions and hypotheses

### RQ1: Predictability

Can a failed rollout predict which remote client adapter will provide positive recovery utility better than task identity, language similarity, observation similarity, task-domain similarity, or update similarity?

**H1:** Failure-conditioned ranking achieves higher NDCG, Spearman correlation, Recall@K, and lower selection regret than non-failure retrieval baselines.

### RQ2: Repair effectiveness

Does utility-ranked personalized adapter repair improve held-out task success more than FedAvg-style aggregation, random retrieval, and similarity-based retrieval?

**H2:** Utility-ranked repair improves success on fresh failure-neighborhood rollouts while querying fewer clients.

### RQ3: Retention safety

Does retention-gated consolidation reduce negative transfer and forgetting compared with immediately merging successful local repairs into the global model?

**H3:** Gated consolidation maintains higher backward transfer and lower unrelated-task regression at comparable target recovery.

### RQ4: Generalization

Does the utility estimator generalize to unseen task-skill compositions, client assignments, and failure initializations?

**H4:** Performance remains above task-only retrieval when task identities and client-task ownership patterns are held out.

### RQ5: Practicality

Can the method operate with LoRA-sized communication and acceptable repair latency on a single RTX 4090 training server?

**H5:** Sparse top-K adapter retrieval provides a better recovery/communication Pareto frontier than full aggregation.

## 5. System model

There are N robot clients and one coordinating server. At continual stage r, client i observes private dataset D_i^r and receives a shared SmolVLA base plus a global adapter. Each participating client trains a local LoRA adapter A_i^r. Raw observations and trajectories remain local.

The initial experiments use simulation clients rather than physically separate machines. The separation is logical and enforced at the data-loader, artifact, and metadata levels.

When target client q experiences a failure, it constructs failure context c_f and requests a repair. Candidate source clients return a privacy-filtered descriptor h_i and make their adapter A_i available through the server. The ranker estimates utility, selects top-K sources, and creates a personalized adapter A_q^repair. The global adapter remains unchanged until the repair passes a consolidation evaluation.

Threat scope:

- Honest-but-curious server.
- Honest clients in the main paper.
- No formal differential-privacy guarantee in the first version.
- Descriptor leakage is measured empirically and discussed as a limitation.
- Malicious-client robustness is an optional stress test, not a main contribution.

## 6. Backbone decision

### 6.1 Primary backbone

Use SmolVLA with the official LeRobot/LIBERO implementation and checkpoint.

Reasons:

- Feasible for repeated LoRA training and counterfactual adapter evaluation on one RTX 4090.
- Existing LIBERO integration.
- Action chunking supports temporal failure representations.
- Natural path to later SO-101 validation.
- Enables multiple clients, seeds, and ablations that would be impractical with a 7B backbone.

### 6.2 Secondary validation

After all primary hypotheses pass, reproduce only the core ranking and personalized-repair result using OpenVLA-OFT on a reduced subset: three clients, four tasks, one or two seeds. This is a robustness check, not a prerequisite for the initial go/no-go decision.

### 6.3 Real robot

SO-101 experiments begin only after LIBERO Stage 1 and Stage 2 meet the go/no-go criteria. Real-robot validation uses SmolVLA and a small task suite; it confirms mechanism transfer rather than attempting a new benchmark leaderboard.

## 7. Transferable knowledge representation

### 7.1 Adapter bank

For every client-stage pair, store:

- LoRA weights A_i^r;
- client and stage identifiers used only for bookkeeping, never ranker input;
- skill prototype p_i^r;
- update sketch d_i^r;
- reliability record rho_i^r;
- embodiment descriptor e_i, constant in initial LIBERO experiments;
- training-data count and adapter norm for calibration.

Adapters must share the same base checkpoint, target modules, rank, scaling convention, and action normalization.

### 7.2 Client descriptor

The descriptor h_i concatenates:

1. Skill prototype: pooled frozen-policy features from successful local trajectories.
2. Update sketch: random projection or layer-wise statistics of the LoRA delta, avoiding full parameter input to the ranker.
3. Reliability: exponentially weighted historical recovery gain, uncertainty, and retention regression.
4. Compatibility metadata: action dimension, observation modalities, and embodiment identifier.

Raw task IDs, raw images, raw trajectories, and human-readable client ownership are excluded from h_i.

## 8. Failure representation

Use a temporal window of W steps ending at failure detection. Each step contains frozen SmolVLA visual-language features, proprioception, executed action, predicted action distribution statistics, and temporal position. A temporal transformer or lightweight sequence encoder produces z_f.

Failure detection in the main simulation experiments uses the environment terminal success signal and episode timeout. Failure *diagnosis* is optional auxiliary supervision. The main method must work without manually assigned failure categories.

Input variants for ablation:

- task instruction only;
- initial observation only;
- terminal observation only;
- successful trajectory context;
- binary failure outcome plus task instruction;
- pre-failure window without task instruction;
- full failure context;
- full context plus oracle failure category.

## 9. Counterfactual utility target

For failure context c_f and candidate adapter A_i, apply A_i to the same target base and evaluate on a fresh failure-neighborhood set E_f. Define:

u_i(c_f) = DeltaS_i(c_f) - lambda_ret R_i - lambda_cost C_i,

where:

- DeltaS_i is the change in success probability or empirical success rate on E_f;
- R_i is the mean positive performance drop over retained/unrelated tasks;
- C_i is normalized communication plus adaptation cost.

For low-rollout settings, utility labels include a confidence interval. Ambiguous pairs whose confidence intervals overlap substantially are used with a soft/listwise ranking target rather than forced hard ordering.

The counterfactual label set must be generated by applying every candidate independently to an identical target checkpoint and evaluating paired initial-state seeds.

## 10. Utility ranker

### 10.1 Architecture

The preferred ranker uses cross-attention:

- Query tokens: failure representation z_f and target-client context.
- Candidate tokens: client descriptor h_i.
- Output: predicted recovery gain, predicted retention risk, predicted uncertainty, and scalar utility.

The decomposed prediction is preferred over a single score because it supports calibration and interpretable consolidation decisions.

### 10.2 Objective

Use a composite loss:

L_rank = L_listwise + alpha L_gain + beta L_risk + gamma L_uncertainty.

- L_listwise: ListNet/ListMLE or differentiable NDCG surrogate over candidate adapters.
- L_gain: Huber regression on DeltaS.
- L_risk: Huber regression on retention regression.
- L_uncertainty: Gaussian negative log likelihood or calibrated quantile loss.

Task IDs and source-client identities are excluded. Training examples are grouped by failure request so the loss compares candidates for the same failure.

### 10.3 Selection and abstention

Select top-K candidates only when the lower confidence bound of predicted utility is positive. If all candidates are non-positive or uncertain, abstain and retain the current target model. Report abstention accuracy and harmful-transfer avoidance.

## 11. Personalized repair

The default repair is a sparse mixture of the selected source LoRA deltas:

A_q^repair = A_q + sum_{i in K} alpha_i A_i,

subject to alpha_i >= 0, sum alpha_i <= tau, and a norm/trust-region bound.

Mixture coefficients are initialized from normalized predicted utilities and refined on a small local validation or recovery buffer when available. The first implementation must also support top-1 direct adapter composition because it provides a clean causal baseline.

The repair modifies only target client q. It is evaluated on fresh rollouts rather than the exact failure state used to request repair.

## 12. Retention-gated consolidation

A personalized repair is eligible for global consolidation only if:

- lower confidence bound of target recovery gain exceeds delta_gain;
- upper confidence bound of mean retention regression is below delta_ret;
- worst-task regression is below delta_worst;
- adapter norm and update conflict remain inside calibrated bounds.

The global update uses a conservative coefficient eta_cons and preserves the previous global adapter for rollback. Consolidation is evaluated against:

- no consolidation;
- immediate unconditional merge;
- data-size-weighted merge;
- utility-weighted merge without a retention gate;
- oracle gate based on complete evaluation.

## 13. FedLIBERO-Fail protocol

### 13.1 Development split

Use five logical clients and LIBERO-10 for pipeline smoke tests. This split is not used to support the final continual-learning claim.

### 13.2 Main split

Construct client streams from LIBERO-Long, LIBERO-Goal, and LIBERO-Spatial. Each client receives four sequential stages. Streams must create:

- overlapping motor skills across non-identical tasks;
- variation in objects, spatial relations, and goals;
- clients with different competence on similar skills;
- compositional tasks at later stages;
- at least one useful, one neutral, and one harmful candidate for a substantial fraction of failures.

The final task assignment is generated from an explicit skill taxonomy and frozen before method evaluation.

### 13.3 Split integrity

Partition by task-skill combination, client assignment, and initial-state seed rather than randomly splitting episodes. Maintain:

- training combinations;
- validation combinations;
- held-out task-skill compositions;
- held-out client-task assignments;
- held-out initial-state seeds.

### 13.4 Failure strata

- Seen-task failure: an exact task has appeared somewhere in the federation.
- Compositional failure: the exact task is new but component skills appeared separately.
- No-match/OOD failure: no candidate offers positive utility.

No-match cases are required to evaluate abstention and harmful-transfer avoidance.

### 13.5 Continual loop

At each stage:

1. Broadcast the shared base and global adapter.
2. Train local client LoRA adapters.
3. Perform the normal federated baseline update.
4. Evaluate the complete task-stage transfer matrix.
5. Generate deployment rollouts on held-out initializations.
6. Collect failure contexts.
7. Generate counterfactual candidate-utility labels.
8. Train or evaluate the utility ranker without split leakage.
9. Perform personalized repair.
10. Run target and retention evaluations.
11. Apply the consolidation policy.
12. Re-evaluate the transfer matrix.

## 14. Baselines

### 14.1 Federated optimization

- Local-only continual learning.
- Centralized multitask oracle.
- FedAvg.
- FedProx.
- FedAvg plus replay.
- FedVLA-style expert-selection aggregation approximation on the same backbone.
- ForgeVLA-style update-conflict weighting approximation on the same backbone.

### 14.2 Selective inter-client transfer

- Random top-K.
- Data-size top-K.
- Task-instruction similarity.
- Observation similarity.
- FedSeIT-style task-prototype similarity.
- FedSaC-style similarity/complementarity score.
- LoRA cosine similarity and gradient/update alignment.
- Historical reliability only.
- Oracle counterfactual utility.

FedWeIT/FedSeIT should be reproduced at the mechanism level using LoRA adapters if their original architectures cannot be fairly ported to SmolVLA. The paper must clearly label faithful reproduction, adaptation, and conceptual comparison.

### 14.3 Continual VLA and routing

- Sequential fine-tuning.
- Experience replay.
- Semantic/phase-aware replay approximation inspired by PHASER.
- CLARE-style reconstruction-error routing.
- Task/skill routing approximation inspired by Stellar VLA.

### 14.4 Failure handling

- No repair.
- Retry without parameter update.
- Direct local recovery fine-tuning.
- Failure-negative guidance approximation inspired by Failing Forward where compatible.
- Failure representation plus random source.
- Source retrieval without failure representation.

## 15. Ablations

Mandatory ablations:

1. Remove failed trajectory; use task instruction only.
2. Remove task instruction; use failed trajectory only.
3. Replace failed trajectory with successful trajectory.
4. Remove update sketch.
5. Remove skill prototype.
6. Remove reliability.
7. Predict one utility score instead of decomposed gain/risk.
8. Remove uncertainty and abstention.
9. Top-1 versus top-K sparse mixture.
10. Remove retention penalty from utility.
11. Remove consolidation gate.
12. Personalized versus immediate global repair.
13. Current-client adapter candidates versus historical client-stage bank.
14. Seen task versus compositional versus no-match failures.
15. Task-ID available versus prohibited.

## 16. Metrics

### Retrieval quality

- Spearman rank correlation.
- NDCG@K.
- Recall@K for positive-utility adapters.
- Top-K regret relative to oracle.
- Harmful-selection rate.
- Abstention AUROC and coverage-risk curve.

### Robot performance

- Success rate before and after repair.
- Failure recovery gain on fresh initial states.
- Number of rollouts required for repair.
- Worst-client and worst-task success.

### Continual learning

- Final average success.
- Backward transfer.
- Forward transfer.
- Average and worst-task forgetting.
- Area under the continual success curve.

### Safety and efficiency

- Unrelated-task regression.
- Consolidation acceptance/rejection accuracy.
- Communication bytes per successful repair.
- Ranker and repair latency.
- Adapter-bank storage.

## 17. Statistical design

- Smoke tests: one seed and 10-20 rollouts per task-condition.
- Main experiments: at least three independent training seeds.
- Evaluation: 30-50 rollouts per task-condition where computationally feasible.
- Pair methods on identical environment initial-state seeds.
- Report bootstrap 95% confidence intervals.
- Use paired permutation or Wilcoxon signed-rank tests for success differences across matched seeds/tasks; use mixed-effects logistic regression when sufficient rollout-level data are available.
- Correct families of ablation comparisons using Holm's procedure.
- Report effect sizes, not only p-values.

## 18. Compute plan

### MacBook M2

- Dataset and task-stream validation.
- Synthetic/unit tests for adapter bank, ranker, and utility calculations.
- Tiny checkpoint and one-client smoke tests.
- Analysis, plotting, manuscript generation.
- SO-101 control and eventual real-robot inference where feasible.

### RTX 4090 server

- SmolVLA LoRA training.
- LIBERO rollout generation.
- Counterfactual adapter evaluation.
- Utility-ranker training.
- Main seeds and ablations.
- Reduced OpenVLA-OFT validation only after primary success.

Commands must be packaged as reproducible scripts with configuration files and resumable checkpoints because the researcher will execute GPU jobs manually.

## 19. Go/no-go stages

### Gate A: backbone reproducibility

Proceed only if the SmolVLA LIBERO checkpoint and local LoRA fine-tuning reproduce stable non-trivial success and failures can be collected deterministically enough for paired evaluation.

### Gate B: utility exists

Proceed only if candidate adapters show meaningful positive, neutral, and negative utility variation for the same failure contexts. If nearly all adapters behave identically, retrieval is not a meaningful problem.

### Gate C: failure adds information

Proceed only if failure-conditioned ranking significantly outperforms task-only and observation-only retrieval on held-out combinations.

### Gate D: repair is safe

Proceed only if personalized repair provides recovery gain with bounded unrelated-task regression and the retention gate improves the recovery-retention frontier.

### Gate E: paper expansion

Only after Gates A-D pass should the study add full continual streams, OpenVLA-OFT, and SO-101 validation.

Pivot rules:

- If Gate B fails, redesign client streams to create heterogeneous competence rather than semantic task ownership.
- If Gate C fails, rename the method context-conditioned transfer utility and treat failure as an optional feature.
- If Gate D fails globally but succeeds locally, publish personalized repair without global consolidation.

## 20. Expected paper contributions

Before results, contributions are phrased as designs rather than achievements:

1. Formulation of failure-conditioned transfer-utility estimation for continual federated VLA.
2. A privacy-filtered utility ranker that predicts recovery gain, retention risk, and uncertainty for remote client adapters.
3. A personalized repair and retention-gated consolidation mechanism with abstention.
4. FedLIBERO-Fail, a controlled evaluation protocol with counterfactual client-utility labels, compositional failures, no-match cases, and continual transfer metrics.

Empirical superiority claims are added only after complete experiments.

## 21. Paper structure

1. Abstract: problem, missing capability, method, protocol, verified results only.
2. Introduction: failure at deployment, why similarity is insufficient, counterfactual utility insight, contributions.
3. Related Work: federated VLA; federated continual/selective transfer; continual VLA routing; failure-aware adaptation.
4. Problem Formulation: client streams, failure request, utility, personalized repair, consolidation constraint.
5. Method: adapter bank, failure encoder, decomposed utility ranker, abstention, sparse repair, retention gate.
6. FedLIBERO-Fail Protocol: stream construction, counterfactual labels, splits, failure strata.
7. Experiments: retrieval, repair, retention, generalization, efficiency, secondary backbone.
8. Real-Robot Validation: included only if LIBERO gates pass.
9. Limitations and Broader Impacts: simulated federation, privacy leakage, evaluation cost, failure detector assumptions, negative-client risks.
10. Conclusion.

## 22. Minimum evidence for a Q1/top-tier submission

The submission should not proceed with only a final success-rate table. Minimum evidence includes:

- a counterfactual adapter-utility dataset or benchmark protocol;
- direct ranking evaluation against strong retrieval signals;
- a demonstration that failure context adds information beyond task identity;
- personalized repair and safe consolidation evaluated separately;
- at least three seeds and confidence intervals;
- compositional and no-match cases;
- communication/latency analysis;
- either a second backbone or SO-101 validation, preferably both if resources permit;
- transparent limitations and reproducible commands/configuration.

## 23. Deliverables after design approval

1. Research implementation plan with smoke-test-first milestones.
2. GPU command sheet for RTX 4090 execution.
3. FedLIBERO-Fail task taxonomy and split manifest.
4. Method code and tests.
5. Analysis notebooks/scripts.
6. Manuscript draft with placeholders only for actual numeric results.

