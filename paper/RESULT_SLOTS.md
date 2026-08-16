# FCUT-VLA Result Slot Registry

Each slot may be resolved only from an immutable run manifest containing the git commit, configuration hash, FedLIBERO-Fail manifest hash, seed list, and metrics artifact.

| Slot | Producing experiment | Metric | Required interpretation |
|---|---|---|---|
| `RESULT:RQ1_NDCG_MAIN` | Held-out failure-context ranking | NDCG@K, Spearman, regret | Full failure context versus strongest non-failure retrieval baseline |
| `RESULT:RQ2_RECOVERY_MAIN` | Personalized repair on paired fresh states | Success-rate difference and 95% CI | Utility-ranked repair versus strongest feasible repair baseline |
| `RESULT:RQ3_RETENTION_MAIN` | Global consolidation comparison | Target gain, mean and worst-task regression | Gated versus immediate and ungated merging |
| `RESULT:RQ4_COMPOSITIONAL_MAIN` | Held-out task-skill/client split | NDCG@K and recovery gain | Generalization without Task ID or ownership leakage |
| `RESULT:RQ5_EFFICIENCY_MAIN` | Communication/latency sweep | Bytes, latency, recovered failures | Recovery-efficiency Pareto comparison |
| `RESULT:BACKBONE_TRANSFER` | Reduced OpenVLA-OFT subset | Ranking and repair effect sizes | Architecture robustness after SmolVLA Gates A–D |
| `RESULT:SO101_VALIDATION` | Supervised SO-101 task subset | Recovery success and safety events | Real-robot mechanism validation after simulation gates |

Slots must remain visible when evidence is absent. A negative or null result resolves a slot as faithfully as a positive result.

