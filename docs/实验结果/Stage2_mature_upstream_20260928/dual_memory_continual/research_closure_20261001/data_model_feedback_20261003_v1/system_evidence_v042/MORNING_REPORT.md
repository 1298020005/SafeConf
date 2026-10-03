# SafeConf v0.4.2 execution report

## Actual work

- Assembled the current-contract Source, Target and full-system evidence into one audit package.
- Verified 212 tasks / 152 perturbation genes share one current truth contract.
- Verified rule and supervised dispatch, OOF requirement, separate score and label CDF audits, and unique rankings.

## Current decision

- Default: `PublicRule`.
- At the fixed 20% review budget, PublicRule finds 22/43 true highest-error tasks versus 4/43 for Amplitude and lowers the remaining mean error by 0.0020566 (7.07% relative to the Amplitude remainder).
- Source Ridge: point gain exists in the external Source audit, but the paired gene-cluster interval crosses zero and the current complete ranking is below PublicRule; conditional only.
- Target Ridge at 50%: point gain is small and its paired interval crosses zero; conditional only.
- Legacy PertEMA: excluded from current same-task metrics after truth-contract mismatch audit.

## Next action

SAMS remains in the existing protected training process (`TRAINING_RUNNING`). When it terminates, complete its registered generation/competence/cross-family postprocess and update this evidence package. Papers and PDF remain paused.

## Metric correction

The registered primary is context-macro U20. On the same current holdout, PublicRule is 0.8377, TargetRidge50 is 0.8459 (paired 95% CI for delta vs PublicRule: -0.0865 to +0.0449), and fixed H1_F1 plus Public is 0.8399 (CI -0.0939 to +0.0545). The default therefore remains PublicRule. The prior pooled-task U20 is retained only as a secondary diagnostic.
