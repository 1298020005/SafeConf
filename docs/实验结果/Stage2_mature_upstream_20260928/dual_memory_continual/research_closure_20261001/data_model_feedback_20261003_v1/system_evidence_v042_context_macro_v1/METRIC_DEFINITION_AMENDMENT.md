# Context-macro primary metric amendment

The registered v0.4.2 contract names context-macro Utility@20 as the primary metric. For each target/context stratum, `k=ceil(0.2*n)` and task IDs break score ties; the reported primary value is the equal-weight mean over valid strata. The earlier pooled-task U20 remains in `secondary_pooled_u20` for diagnostic comparison and is not used for adoption decisions.

All rows use the same 212-task current holdout and 152 perturbation-gene clusters. Feedback H1_F1 and XGB F1 are reused from the frozen `feedback_v1` fixed evaluation at 50% feedback; truth hashes are checked before scoring. No new learner fit or truth read occurs here.
