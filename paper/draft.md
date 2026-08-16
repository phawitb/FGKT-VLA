# Failure-Conditioned Transfer Utility for Safe Continual Federated Vision-Language-Action Learning

## Draft status

This manuscript is a method-and-protocol draft. Statements marked `RESULT:*` are preregistered result slots and must be replaced only by verified outputs from immutable experiment manifests. No empirical superiority is claimed at this stage.

# Abstract

Federated vision-language-action (VLA) learning allows robot fleets to improve a shared policy without centralizing raw interaction data, but continual deployment creates a difficult asymmetry: a failed robot needs a specific capability, whereas conventional aggregation integrates updates according to dataset size, representation similarity, expert activity, or update compatibility. These signals describe clients globally but do not directly estimate whether a particular remote update will repair a particular failure. We formulate **failure-conditioned transfer utility**, the expected counterfactual recovery benefit of transferring a client adapter for a failed rollout, penalized by regression on retained capabilities and transfer cost. We introduce FCUT-VLA, a framework that (i) represents a failed temporal context and privacy-filtered client capabilities, (ii) predicts decomposed recovery gain, retention risk, and uncertainty for each candidate adapter, (iii) performs abstaining, sparse personalized repair, and (iv) promotes a repair to the global model only after a retention gate. We further introduce FedLIBERO-Fail, a continual federated evaluation protocol with paired counterfactual adapter labels, held-out task-skill compositions, and explicit no-match failures. Experiments with SmolVLA are designed to test whether failure-conditioned ranking improves retrieval over task-, observation-, parameter-, and domain-similarity baselines (`RESULT:RQ1_NDCG_MAIN`), whether personalized repair increases held-out recovery success (`RESULT:RQ2_RECOVERY_MAIN`), and whether gated consolidation reduces unrelated-task regression (`RESULT:RQ3_RETENTION_MAIN`).

# 1. Introduction

Vision-language-action models unify visual perception, language-conditioned reasoning, and continuous robot control. Their scale and diversity increasingly motivate learning from fleets rather than from a single centrally collected dataset. Federated learning offers a natural systems abstraction: robots retain local observations and trajectories, train locally, and communicate parameter updates to a coordinating server. Recent federated VLA systems demonstrate that multimodal representations, mixture-of-experts activity, and update geometry can improve aggregation under heterogeneous tasks [@fedvla2025;@forgevla2026].

Deployment, however, exposes a question that ordinary aggregation does not answer. Suppose a shared policy fails to align a gripper, loses an object during transport, or completes the wrong subgoal. Other robots may possess adapters trained on useful experience, but their relevance cannot be inferred reliably from dataset size alone. More subtly, the most semantically similar client is not necessarily the most useful source: a client may have seen the same object yet learned a brittle controller, while a different task may have produced a robust transferable grasp. Transferring the wrong update can also repair the target behavior while degrading previously retained capabilities.

Selective inter-client transfer is not new. FedWeIT decomposes global, local, and task-adaptive parameters and learns attention over foreign task knowledge [@fedweit2020]. FedSeIT explicitly represents historical client tasks and retrieves top-ranked task-adaptive parameters using domain overlap [@fedseit2022]. Continual VLA methods likewise route tasks or observations to experts and adapters [@stellar2025;@clare2026]. Meanwhile, failure-aware robot learning uses failed experience for diagnosis, negative guidance, replanning, and recovery-policy training [@reflect2023;@racer2024;@robofac2025;@failingforward2026]. What remains unresolved is the intersection of these problems: **can a failed rollout predict the post-transfer effect of remote client knowledge, and can that prediction support a repair that is both personalized and retention-safe?**

We argue that client retrieval should target *utility*, not similarity. For a failure context and candidate adapter, utility is defined by the counterfactual change in recovery success after applying that adapter, minus its measured regression on retained tasks and its transfer cost. This target makes three distinctions explicit. First, semantic relevance and causal benefit are different objects. Second, repair and global aggregation are different decisions. Third, a system should be allowed to abstain when no available client offers positive utility.

We propose FCUT-VLA around this formulation. Each source client contributes a LoRA adapter and a privacy-filtered descriptor containing a skill prototype, update sketch, compatibility metadata, and historical reliability. A temporal failure encoder summarizes the pre-failure rollout without requiring an oracle failure label. A cross-attention ranker predicts recovery gain, retention risk, and uncertainty for every candidate. Candidates with a non-positive lower confidence bound are rejected; positive candidates are composed sparsely into a target-specific adapter. A separate retention gate determines whether a successful local repair is safe to consolidate into the global adapter.

The paper makes four design contributions:

1. We formulate failure-conditioned transfer-utility estimation for continual federated VLA, distinguishing measurable post-transfer benefit from task or representation similarity.
2. We introduce a decomposed and uncertainty-aware client-utility ranker with abstention and sparse personalized adapter repair.
3. We introduce retention-gated consolidation, separating immediate target repair from conservative global knowledge promotion.
4. We define FedLIBERO-Fail, a leakage-controlled protocol that exposes counterfactual candidate utilities, compositional failures, no-match cases, and continual retention effects.

# 2. Related Work

## 2.1 Federated robot and VLA learning

FedAvg optimizes a shared model by averaging local updates in proportion to client data [@fedavg2017]. This objective is vulnerable to client drift under heterogeneous data, motivating proximal, clustered, personalized, and similarity-aware alternatives. FLAME provides a federated robot-manipulation benchmark spanning tasks and environment variations [@flame2025]. FedVLA specializes federated learning to multimodal action policies through instruction-oriented scene parsing, dual-gated experts, and aggregation weights derived from expert-selection similarity [@fedvla2025]. ForgeVLA addresses clients whose robot logs lack language annotations and combines local contrastive planning with task prototypes and adaptive aggregation designed to reduce cross-client update conflict [@forgevla2026]. These methods establish that federated VLA aggregation can exploit more than dataset size. FCUT-VLA instead conditions the transfer decision on an observed deployment failure and predicts adapter-specific post-transfer outcomes.

## 2.2 Federated continual and selective transfer

FedWeIT introduced federated continual learning with sparse task-adaptive parameters and learned attention over foreign knowledge [@fedweit2020]. FedSeIT further represents client task domains and retrieves top-K historical task parameters [@fedseit2022]. FedSaC constructs personalized cooperation graphs using both model similarity and feature-subspace complementarity [@fedsac2024]. These works are direct antecedents: they show that selective transfer and complementarity are established concepts. FCUT-VLA differs in its query and supervision. Its query is a failed rollout rather than a current task dataset, and its target is paired counterfactual recovery utility rather than domain overlap or collaboration similarity.

## 2.3 Continual VLA learning and routing

LIBERO formalizes lifelong robot learning through sequential task suites and transfer matrices [@libero2023]. Stellar VLA learns an evolving task-skill knowledge space and routes inputs through experts while using limited replay [@stellar2025]. CLARE expands lightweight adapters and selects them from observation features using discriminator reconstruction error [@clare2026]. CRL-VLA constrains continual reinforcement learning through old-task value consistency and policy divergence [@crlvla2026]. PHASER develops semantic, phase-aware replay for continual VLA [@phaser2026]. FCUT-VLA treats these as continual and routing baselines. It asks whether failure context predicts a useful *remote update* beyond observation-only routing and whether repair can be consolidated without increasing forgetting.

## 2.4 Failure-aware robot learning

REFLECT summarizes multimodal experience for language-based failure explanation and replanning [@reflect2023]. RACER augments imitation data with richly described failure-recovery trajectories [@racer2024]. RoboFAC introduces structured failure analysis and correction supervision [@robofac2025]. Failing Forward trains success and failure action generators and uses failed rollouts as negative sampling guidance [@failingforward2026]. These methods establish the value of failure experience but do not rank federated client adapters by their counterfactual repair effect. FCUT-VLA does not claim to replace diagnosis or recovery-policy learning; it studies failure as a query for remote capability transfer.

# 3. Problem Formulation

## 3.1 Continual federated VLA

Consider clients \(\mathcal C=\{1,\ldots,N\}\) and continual stages \(r=1,\ldots,R\). At stage \(r\), client \(i\) receives private demonstrations \(\mathcal D_i^r\) for a task distribution that may differ from other clients and previous stages. All clients share a pretrained VLA base \(\theta_0\). The server maintains a global low-rank adapter \(A_g^r\), while a participating client trains a local adapter \(A_i^r\) from the common initialization.

The deployed target policy at client \(q\) is

\[
\pi_q^r(a_t\mid o_{\leq t},\ell)=
\pi_{\theta_0,A_g^r,A_q^r}(a_t\mid o_{\leq t},\ell),
\]

where \(o_t\) includes images and proprioception and \(\ell\) is the language instruction. The base remains fixed in the primary study so that client adapters share a compatible parameter coordinate system.

## 3.2 Failure request

A deployment episode produces trajectory

\[
\tau=(o_1,a_1,\ldots,o_T,a_T,y),
\]

with terminal success \(y\in\{0,1\}\). When \(y=0\), client \(q\) constructs failure context

\[
c_f=\big(\ell,\{o_t,p_t,a_t,\xi_t\}_{t=T-W+1}^{T},m\big),
\]

where \(p_t\) is proprioception, \(\xi_t\) contains policy uncertainty statistics, \(W\) is the context window, and \(m\) is a temporal mask. Oracle failure categories are excluded from the default input.

## 3.3 Counterfactual transfer utility

Let \(A_i\) be a candidate source adapter and \(\mathcal E_f\) a held-out set of fresh initial states drawn from the local neighborhood of the failed task. Applying each candidate independently to the same target checkpoint yields recovery gain

\[
\Delta S_i(c_f)=
S(\theta_q\oplus A_i;\mathcal E_f)-S(\theta_q;\mathcal E_f),
\]

where \(\oplus\) denotes a controlled adapter composition, not unrestricted model averaging. Let \(\mathcal U_f\) be retained tasks unrelated to the target failure. Retention risk is

\[
R_i(c_f)=\frac{1}{|\mathcal U_f|}
\sum_{k\in\mathcal U_f}
\max\!\left(0,S_k(\theta_q)-S_k(\theta_q\oplus A_i)\right).
\]

We define utility

\[
u_i(c_f)=\Delta S_i(c_f)-\lambda_{\mathrm{ret}}R_i(c_f)
-\lambda_{\mathrm{cost}}C_i,
\]

where \(C_i\) normalizes communication, adapter storage, and repair latency. The learning problem is to estimate the ranking induced by \(u_i(c_f)\) without evaluating every candidate online.

## 3.4 Safe repair objective

The system selects a sparse candidate set \(K_f\), constructs personalized repair \(A_q^{\mathrm{rep}}\), and may abstain. A repair becomes eligible for global consolidation only if the lower confidence bound of target gain exceeds \(\delta_{\mathrm{gain}}\), the upper confidence bound of mean retention risk remains below \(\delta_{\mathrm{ret}}\), and worst-task regression remains below \(\delta_{\mathrm{worst}}\).

# 4. Method

## 4.1 Client capability bank

Each client-stage adapter is registered with descriptor

\[
h_i=[p_i;d_i;\rho_i;e_i;n_i],
\]

where \(p_i\) is a pooled prototype from frozen VLA features over successful local trajectories, \(d_i\) is a compact random projection or layer-wise sketch of the LoRA delta, \(\rho_i\) records calibrated historical gain and retention behavior, \(e_i\) encodes embodiment compatibility, and \(n_i\) describes data scale and update norm. Raw trajectories, raw images, task IDs, source-client IDs, and human-readable ownership metadata are excluded from ranker inputs.

The distinction between prototype and update sketch is important. A prototype approximates what a client experienced; an update sketch approximates how its learned intervention changes the policy. Their combination allows the ranker to distinguish semantic match from parameter-space compatibility.

## 4.2 Temporal failure encoder

Frozen SmolVLA visual-language tokens are extracted for the last \(W\) execution steps and combined with proprioception, executed actions, action-distribution statistics, and relative time embeddings. A lightweight temporal transformer returns failure tokens \(Z_f\) and pooled query \(z_f\). Masked temporal attention supports early termination and variable-length episodes.

The default detector uses environment success and timeout signals. This isolates the central question—source utility—from the separate problem of online failure detection. Failure-type supervision is used only for an oracle analysis and an auxiliary representation ablation.

## 4.3 Decomposed transfer-utility ranker

For every candidate \(i\), cross-attention combines \(Z_f\), target context, and \(h_i\). Three prediction heads output expected recovery gain \(\hat g_i\), retention risk \(\hat r_i\), and uncertainty \(\hat\sigma_i\). Predicted scalar utility is

\[
\hat u_i=\hat g_i-\lambda_{\mathrm{ret}}\hat r_i
-\lambda_{\mathrm{cost}}C_i.
\]

The training objective is

\[
\mathcal L=
\mathcal L_{\mathrm{list}}
+\alpha\,\mathrm{Huber}(\hat g,g)
+\beta\,\mathrm{Huber}(\hat r,r)
+\gamma\,\mathcal L_{\mathrm{unc}},
\]

where \(\mathcal L_{\mathrm{list}}\) is a listwise ranking loss grouped by failure request. Confidence-overlapping candidate pairs are not forced into arbitrary hard orderings.

## 4.4 Abstaining source selection

The system selects candidates whose lower utility confidence bound is positive:

\[
K_f=\operatorname{TopK}\left\{i:
\hat u_i-\kappa\hat\sigma_i>0\right\}.
\]

If \(K_f=\varnothing\), the system abstains. No-match failures are therefore first-class evaluation cases rather than errors removed during data construction.

## 4.5 Sparse personalized repair

For selected adapters, the target constructs

\[
A_q^{\mathrm{rep}}=A_q+\sum_{i\in K_f}\alpha_i A_i,
\quad \alpha_i\geq0,\quad \sum_i\alpha_i\leq\tau.
\]

Coefficients are initialized from normalized positive lower confidence bounds. The top-1 transfer is evaluated separately as the cleanest causal test; sparse top-K composition tests whether complementary skills improve compositional failures. A norm bound and optional local recovery buffer prevent uncontrolled delta magnitude.

## 4.6 Retention-gated consolidation

Personalized repair and global promotion are distinct operations. After repair, paired target and retention rollouts produce confidence intervals. A gate accepts global promotion only when all preregistered gain, average-risk, worst-task, and update-norm conditions pass. Accepted repairs are merged conservatively,

\[
A_g^{r+1}=(1-\eta_c)A_g^r+\eta_c A_q^{\mathrm{rep}},
\]

while retaining \(A_g^r\) for rollback. This gate is compared with immediate merge, utility-weighted merge without retention checks, no consolidation, and an oracle gate.

# 5. FedLIBERO-Fail Protocol

## 5.1 Motivation

A static split in which each client owns two LIBERO tasks creates heterogeneous federated multitask learning but not a sufficient continual-learning protocol. It also makes client retrieval nearly equivalent to Task-ID lookup. FedLIBERO-Fail instead defines a client-task-stage tensor with overlapping skills, held-out compositions, varying client competence, and paired counterfactual adapter evaluation.

## 5.2 Streams and splits

The development split uses five logical clients and LIBERO-10 for engineering checks only. The main split draws four sequential stages per client from LIBERO-Long, LIBERO-Goal, and LIBERO-Spatial. A frozen skill taxonomy identifies grasp, transport, articulation, spatial-relation, placement, and long-horizon composition attributes. Exact task-skill combinations, client assignments, and initial-state seeds are partitioned across train, validation, and test.

The test set includes:

- **Seen-task failures:** an exact task occurred at another client.
- **Compositional failures:** component skills appeared separately but the exact task did not.
- **No-match failures:** every candidate has non-positive or statistically uncertain utility.

## 5.3 Counterfactual label generation

For each failure request, every candidate adapter is applied independently to the same target checkpoint. Candidate policies are evaluated on identical fresh initial-state seeds for target and retained tasks. This produces paired estimates of gain, risk, cost, and confidence. Episode-level random splits are prohibited; all examples derived from one task-skill combination and client assignment remain in a single split.

## 5.4 Continual evaluation

After every client stage, the experiment records the complete task-stage transfer matrix. Repair is evaluated before and after personalized transfer and, separately, before and after global consolidation. The protocol thus distinguishes retrieval quality, immediate recovery, backward transfer, and accumulated forgetting.

# 6. Experimental Design

## 6.1 Research questions

**RQ1—Predictability.** Does failure context rank candidate adapters better than task instruction, initial or terminal observation, task prototypes, LoRA similarity, update alignment, historical reliability, and reconstruction-error routing? Main metrics are Spearman correlation, NDCG@K, Recall@K, selection regret, and harmful-selection rate. The preregistered main comparison is `RESULT:RQ1_NDCG_MAIN`.

**RQ2—Repair.** Does utility-ranked personalized repair improve held-out success over no repair, retry, random top-K, domain-similarity retrieval, direct recovery fine-tuning, and generic federated aggregation? The main paired recovery result is `RESULT:RQ2_RECOVERY_MAIN`.

**RQ3—Retention.** Does retention-gated promotion improve the recovery-retention Pareto frontier relative to immediate global merging and ungated utility merging? The main result is `RESULT:RQ3_RETENTION_MAIN`.

**RQ4—Generalization.** Does the ranker transfer to held-out task-skill compositions and client assignments with Task ID removed? The main result is `RESULT:RQ4_COMPOSITIONAL_MAIN`.

**RQ5—Efficiency.** Does sparse retrieval reduce bytes and repair latency per recovered failure relative to full aggregation? The main result is `RESULT:RQ5_EFFICIENCY_MAIN`.

## 6.2 Baselines

Federated baselines include local-only training, centralized multitask training as an oracle upper bound, FedAvg, FedProx, FedAvg with replay, a FedVLA-inspired expert-activity weighting, and a ForgeVLA-inspired update-conflict weighting. Selective-transfer baselines include random, data-size, task-language, observation, FedSeIT-style prototype similarity, FedSaC-style similarity/complementarity, LoRA cosine, update alignment, reliability-only, and oracle counterfactual utility. Continual/routing baselines include sequential fine-tuning, experience replay, a PHASER-inspired semantic replay approximation, CLARE-style reconstruction routing, and a Stellar-inspired task-skill router. Adaptations of prior mechanisms are labeled explicitly rather than presented as exact reproductions.

## 6.3 Ablations

We remove, independently, failure trajectories, task instructions, update sketches, skill prototypes, reliability, decomposed gain/risk prediction, uncertainty, abstention, retention penalty, and the consolidation gate. We compare successful versus failed trajectory queries; top-1 versus top-K composition; personalized versus immediate-global repair; current versus historical adapter banks; task-identified versus task-hidden inputs; and seen, compositional, and no-match strata.

## 6.4 Statistics

Smoke tests use one seed. Main experiments use at least three independent training seeds and paired environment initial-state seeds, with 30–50 evaluation rollouts per task-condition when feasible. We report bootstrap 95% confidence intervals and effect sizes. Paired permutation or Wilcoxon tests compare matched task-seed aggregates; mixed-effects logistic regression is used when rollout-level sample size supports it. Holm correction controls families of ablation comparisons.

## 6.5 Compute stages

MacBook M2 testing covers manifests, synthetic adapter utilities, ranking, gating, and dry-run command graphs. RTX 4090 stages are gated: (A) reproduce SmolVLA LIBERO behavior, (B) verify useful within-failure utility variation, (C) show that failure context adds information beyond non-failure signals, and (D) establish safe personalized repair. Only after A–D pass do we run full continual streams, a reduced OpenVLA-OFT validation (`RESULT:BACKBONE_TRANSFER`), and SO-101 experiments (`RESULT:SO101_VALIDATION`).

# 7. Limitations and Broader Impact

The initial federation is simulated within a shared compute environment and therefore does not reproduce device dropouts, network failures, or physical-client security boundaries. Although raw trajectories remain local by design, capability prototypes and update sketches may leak properties of client data; FCUT-VLA does not provide a formal differential-privacy guarantee. Counterfactual utility labeling is expensive because every candidate must be evaluated during benchmark construction, although online deployment evaluates only selected candidates. Utility is environment- and checkpoint-dependent, so stale descriptors may require recalibration. Environment success signals simplify failure detection and do not address open-world failure-monitor errors. Finally, adapter composition assumes a shared base and compatible action space; cross-embodiment transfer requires additional alignment.

Positive transfer can improve data efficiency and reduce repeated collection of sensitive robot experience. Conversely, an incorrect or malicious adapter could cause unsafe actions. Abstention, personalized deployment, retention checks, rollback, and conservative global promotion reduce but do not eliminate this risk. Real-robot evaluation must use workspace limits, emergency stops, low-speed initial trials, and human supervision.

# 8. Conclusion

This work reframes client selection after robot failure as counterfactual transfer-utility estimation. FCUT-VLA is designed to predict which remote adapter will repair a specific failed rollout, quantify the accompanying retention risk, abstain when no source is beneficial, and separate personalized repair from global consolidation. FedLIBERO-Fail makes these decisions measurable through paired candidate evaluation, leakage-controlled continual streams, compositional tasks, and no-match cases. The central empirical question is deliberately falsifiable: failure context must predict realized transfer utility better than task, observation, representation, and update similarity. Verified conclusions will be inserted only after the preregistered experimental gates are completed.

