# Fixed Source DEV component diagnosis

This diagnostic separates the Public feature version and Source-error training budget in the completed rejected replay. It makes exactly two additional fixed HGB fits. It publishes no candidate, selects no winner and changes no formal replay files or serving configuration. No Orion or McFaline data are opened.

## Predeclared design

P1 is the initial 1,047-record Replogle Public bank; P2 includes the 961 added Nadig records. Both use the existing Manual cell-count-weighted prior and its genuinely reconstructed version-specific features. E1 uses initial Source-error folds 2–3 (1,372 predictor/task records, 225 genes); E2 adds fold 4 (2,038 records, 336 genes). The error budget alone determines the training-only grouped CDF, preprocessing fit and original mean-one gene-cluster weights.

The original P1E1 and P2E2 models are reused and reproduce their saved predictions byte for byte. P1E2 and P2E1 are the two additional fits. The zero-fit drift condition evaluates the original P1E1 model on P2 features; it is a diagnostic interface mismatch and is never served. All use the same 13 features, HGB parameters and seed 20260930. Models and all five conditions' predictions were frozen before DEV scoring.

The complete comparison cohort remains 1,699 biological tasks/3,398 predictor-task records. Only the fixed old-anchor fold 0 and new-task gate fold 1 are scored. Their 227 distinct gene clusters are resampled jointly across both roles, four contexts, two existing architectures and all conditions, using 5,000 draws. Every planned draw is valid. GAT and Exphormer are two predictors within one TxPert family. This is conditional, retrospective Source DEV evidence; the initial Source-error scope already contains both historical studies.

## U20 point estimates

| Public / Error condition | Old anchor | New-task gate | Additional fit |
|---|---:|---:|---:|
| P1E1, original baseline | 0.7372 | 0.7523 | 0 |
| P1E2, Error budget change | 0.7420 | 0.7292 | 1 |
| P2E1, Public feature change | 0.7453 | 0.7064 | 1 |
| P2E2, original rejected candidate | 0.7483 | 0.7343 | 0 |
| Original v1 model on P2 features, zero-fit drift | 0.7331 | 0.7478 | 0 |

All 80 role × predictor × context × condition rows, including U20, Spearman, AURC and high-risk miss rate, are in [STRATA.csv](STRATA.csv). [MACRO.csv](MACRO.csv) gives the ten equal-context-by-predictor macro rows and all seven original metrics. Every condition is retained.

## Fixed paired U20 contrasts on the new-task gate

| Contrast | Difference | Joint gene-bootstrap 95% interval |
|---|---:|---:|
| Error change at P1: P1E2 − P1E1 | −0.0230 | −0.0414 to 0.0490 |
| Public feature/refit change at E1: P2E1 − P1E1 | −0.0459 | −0.1210 to 0.0193 |
| Interaction: P2E2 − P1E2 − P2E1 + P1E1 | 0.0509 | −0.0272 to 0.1148 |
| Coupled change: P2E2 − P1E1 | −0.0180 | −0.0620 to 0.0513 |
| Frozen-model interface drift | −0.0045 | −0.0563 to 0.0402 |

The new-task point loss is larger for Public-feature rebuilding plus shared-core refitting at the fixed initial error budget than for Error-budget growth at fixed initial Public features. The positive interaction offsets much of the two separate point losses. Frozen-model interface drift is smaller. Old-anchor point U20 improves under either refitted single-component update and the coupled update.

All paired U20 intervals include zero. These results do not establish a unique cause or a robust disadvantage. The Public factor changes the Manual feature distribution and the shared-core fit; it does not retrain a Public Biology Learner. The Error factor changes training records, the CDF reference, preprocessing and absolute weight sum together, so it does not isolate supervision information from row/regularization effects. No model-specific residual adapter is trained. [PAIRED_CONTRASTS.csv](PAIRED_CONTRASTS.csv) retains every fixed contrast for every metric and both roles.

## Execution and preservation

The two fits, saving and reload took 0.3113 and 0.2886 seconds respectively; the full diagnostic and 5,000 draws took 6.97 seconds with one numerical thread and zero GPU. [FIT_COSTS.csv](FIT_COSTS.csv) is separate from the formal replay cost ledger. The original serving version remains risk-v1, the original v2 remains rejected, and no quality gate or publication operation is called by this diagnostic.

[REGISTRATION.json](REGISTRATION.json) fixes the conditions and contrasts before fits/scoring and binds all original formal input/code pins, features, models and Error revisions. [PREDICTION_FREEZE.json](PREDICTION_FREEZE.json) and [STATUS.json](STATUS.json) record score preservation and completion. Models, complete scores and shared bootstrap counts/draws remain in the new isolated server runtime directory. The diagnosis script refuses existing output roots and cannot write inside the formal replay directories.
