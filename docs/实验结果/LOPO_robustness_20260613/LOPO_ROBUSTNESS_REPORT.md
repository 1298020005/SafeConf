# LOPO robustness audit

## Decision

- PertMean pre_model_task_only positive partial rho: 6/7
- pre_model_task_only gate: `strong_pre_model_task_risk_evidence`
- PertMean LODOxLOPO full positive partial rho: 7/7
- LODOxLOPO gate: `cross_dataset_cross_predictor_signal`
- Control1NN datasets below 50% near-identical to ContextSim: 5/7
- Control1NN role: `mechanistically_close_secondary_predictor`

## Interpretation boundary

These experiments test task-risk transfer among retrieval-based predictors. They do not establish universal transfer to deep predictors such as GEARS or CPA.

`pre_model_task_only` does not use the target predictor output, but it still requires fold-local historical perturbation effects and the target context control profile.
