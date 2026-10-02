# One fixed Ridge output repair: SEEN retrospective analysis

## Why and what changed

Saved-model diagnostics found that hard clipping maps every target score to the same endpoint in all 15 scGPT→GEARS folds. The registered repair applies the same fixed, strictly increasing map to both Ridge variants and both model directions:

`bounded_score = 0.5 + arctan(raw_ridge_score - 0.5) / pi`

This retains the Ridge ordering while bounding scores. It is **not** percentile calibration and does not correct distribution shift. No parameter, feature, target, split, prediction task, model, or hyperparameter was refitted/selected. Source-side outputs use the same map for audit. Original 120 model files, original predictions, primary tables and input hashes are verified unchanged.

This is one post-primary **SEEN technical score-repair** version, not external confirmation. All six upstream assets still fail the same competence gate. The repair must not upgrade their qualification.

## Same-task results

| direction | scope | method | utility20 | spearman | aurc |
|---|---|---|---|---|---|
| GEARS_to_scGPT | all_test | Magnitude | 0.092242 | -0.001088 | 0.051880 |
| GEARS_to_scGPT | heldout_context | Magnitude | 0.108732 | 0.052207 | 0.051951 |
| GEARS_to_scGPT | all_test | PredictionRidge | -0.023455 | -0.003755 | 0.052154 |
| GEARS_to_scGPT | heldout_context | PredictionRidge | 0.127844 | 0.077919 | 0.052566 |
| GEARS_to_scGPT | all_test | PredictionRidge_ArctanRepair | -0.023455 | -0.003755 | 0.052154 |
| GEARS_to_scGPT | heldout_context | PredictionRidge_ArctanRepair | 0.127844 | 0.077919 | 0.052566 |
| GEARS_to_scGPT | all_test | PublicRidge | 0.601367 | 0.386123 | 0.049448 |
| GEARS_to_scGPT | heldout_context | PublicRidge | 0.706859 | 0.569094 | 0.048818 |
| GEARS_to_scGPT | all_test | PublicRidge_ArctanRepair | 0.601367 | 0.385924 | 0.049447 |
| GEARS_to_scGPT | heldout_context | PublicRidge_ArctanRepair | 0.706859 | 0.568646 | 0.048815 |
| GEARS_to_scGPT | all_test | WeightedHistoryDistance | 0.890185 | 0.707412 | 0.047106 |
| GEARS_to_scGPT | heldout_context | WeightedHistoryDistance | 0.884100 | 0.769495 | 0.047092 |
| scGPT_to_GEARS | all_test | Magnitude | 0.332735 | 0.251751 | 0.054323 |
| scGPT_to_GEARS | heldout_context | Magnitude | 0.378869 | 0.300987 | 0.055824 |
| scGPT_to_GEARS | all_test | PredictionRidge | -0.245554 | nan | 0.054475 |
| scGPT_to_GEARS | heldout_context | PredictionRidge | -0.056218 | nan | 0.057751 |
| scGPT_to_GEARS | all_test | PredictionRidge_ArctanRepair | 0.060699 | -0.002125 | 0.057642 |
| scGPT_to_GEARS | heldout_context | PredictionRidge_ArctanRepair | 0.067335 | 0.016897 | 0.058978 |
| scGPT_to_GEARS | all_test | PublicRidge | -0.245554 | nan | 0.054475 |
| scGPT_to_GEARS | heldout_context | PublicRidge | -0.056218 | nan | 0.057751 |
| scGPT_to_GEARS | all_test | PublicRidge_ArctanRepair | 0.098116 | 0.000731 | 0.057520 |
| scGPT_to_GEARS | heldout_context | PublicRidge_ArctanRepair | 0.099943 | 0.026620 | 0.058707 |
| scGPT_to_GEARS | all_test | WeightedHistoryDistance | 0.879437 | 0.760489 | 0.051883 |
| scGPT_to_GEARS | heldout_context | WeightedHistoryDistance | 0.887906 | 0.807961 | 0.052587 |

## 5000 paired gene-bootstrap comparisons

| direction | scope | method | baseline | delta_utility20 | ci95_lower | ci95_upper |
|---|---|---|---|---|---|---|
| GEARS_to_scGPT | all_test | PredictionRidge_ArctanRepair | PredictionRidge | 0.000000 | 0.000000 | 0.000000 |
| GEARS_to_scGPT | heldout_context | PredictionRidge_ArctanRepair | PredictionRidge | 0.000000 | 0.000000 | 0.000000 |
| GEARS_to_scGPT | all_test | PredictionRidge_ArctanRepair | Magnitude | -0.115697 | -0.215904 | 0.029346 |
| GEARS_to_scGPT | heldout_context | PredictionRidge_ArctanRepair | Magnitude | 0.019111 | -0.128912 | 0.203318 |
| GEARS_to_scGPT | all_test | PredictionRidge_ArctanRepair | WeightedHistoryDistance | -0.831079 | -0.936448 | -0.680335 |
| GEARS_to_scGPT | heldout_context | PredictionRidge_ArctanRepair | WeightedHistoryDistance | -0.730785 | -0.884219 | -0.613807 |
| GEARS_to_scGPT | all_test | PublicRidge_ArctanRepair | PublicRidge | 0.000000 | 0.000000 | 0.000000 |
| GEARS_to_scGPT | heldout_context | PublicRidge_ArctanRepair | PublicRidge | 0.000000 | -0.000000 | 0.000000 |
| GEARS_to_scGPT | all_test | PublicRidge_ArctanRepair | Magnitude | 0.509125 | 0.348384 | 0.732623 |
| GEARS_to_scGPT | heldout_context | PublicRidge_ArctanRepair | Magnitude | 0.598126 | 0.443081 | 0.777550 |
| GEARS_to_scGPT | all_test | PublicRidge_ArctanRepair | WeightedHistoryDistance | -0.091451 | -0.146283 | 0.008387 |
| GEARS_to_scGPT | heldout_context | PublicRidge_ArctanRepair | WeightedHistoryDistance | -0.060769 | -0.145905 | -0.000811 |
| scGPT_to_GEARS | all_test | PredictionRidge_ArctanRepair | PredictionRidge | 0.306254 | 0.196211 | 0.478628 |
| scGPT_to_GEARS | heldout_context | PredictionRidge_ArctanRepair | PredictionRidge | 0.123554 | 0.008461 | 0.342436 |
| scGPT_to_GEARS | all_test | PredictionRidge_ArctanRepair | Magnitude | -0.272036 | -0.395807 | -0.156952 |
| scGPT_to_GEARS | heldout_context | PredictionRidge_ArctanRepair | Magnitude | -0.311534 | -0.402626 | -0.176191 |
| scGPT_to_GEARS | all_test | PredictionRidge_ArctanRepair | WeightedHistoryDistance | -0.765768 | -0.872956 | -0.640871 |
| scGPT_to_GEARS | heldout_context | PredictionRidge_ArctanRepair | WeightedHistoryDistance | -0.798420 | -0.900349 | -0.658176 |
| scGPT_to_GEARS | all_test | PublicRidge_ArctanRepair | PublicRidge | 0.343671 | 0.229889 | 0.485174 |
| scGPT_to_GEARS | heldout_context | PublicRidge_ArctanRepair | PublicRidge | 0.156161 | 0.028426 | 0.348401 |
| scGPT_to_GEARS | all_test | PublicRidge_ArctanRepair | Magnitude | -0.234619 | -0.392239 | -0.117248 |
| scGPT_to_GEARS | heldout_context | PublicRidge_ArctanRepair | Magnitude | -0.278926 | -0.419592 | -0.130025 |
| scGPT_to_GEARS | all_test | PublicRidge_ArctanRepair | WeightedHistoryDistance | -0.738376 | -0.870158 | -0.609399 |
| scGPT_to_GEARS | heldout_context | PublicRidge_ArctanRepair | WeightedHistoryDistance | -0.744538 | -0.896116 | -0.630596 |

## Remaining boundary and stopping rule

The map can only recover ordering lost by clipping; it cannot make incorrectly ordered raw extrapolations correct, align conditional source/target error distributions, or calibrate the target model's risk percentiles. A ranking benefit over clipped output alone does not establish superiority to Magnitude or WeightedHistoryDistance. History comparisons use exactly the common covered subset. No additional map, temperature, target or calibration will be searched in this branch.

## Reproduce

`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/scripts/run_safeconf_frangieh_cross_family_v1.py --score-repair`

This command requires the saved 120-fit primary run, reads only its fitted models/features, and refuses to overwrite completed repair predictions. New model fits: **0**.


## Decision after this single repair

- scGPT→GEARS heldout-context PredictionRidge U20: −0.056218 → 0.067335; PublicRidge: −0.056218 → 0.099943. Recovered-order improvements over original clipping are +0.123554 (95% CI [0.008461, 0.342436]) and +0.156161 ([0.028426, 0.348401]).
- Neither repaired score approaches Magnitude=0.378869. Their differences to Magnitude are −0.311534 and −0.278926, both intervals entirely below zero. PublicRidge's repaired AURC is 0.058707 versus original 0.057751: improved top-tail ranking does not mean uniformly improved selective risk.
- GEARS→scGPT U20 is unchanged: PredictionRidge=0.127844 and PublicRidge=0.706859. This direction's main weakness was not ranking destruction by clipping.
- Paired against WeightedHistoryDistance on the identical history-covered set, repaired PublicRidge differences remain −0.060769 for GEARS→scGPT (CI [−0.145905, −0.000811]) and −0.744538 for scGPT→GEARS (CI [−0.896116, −0.630596]).
- **Stop this repair branch.** The mathematical output defect is corrected in this separate version, but the raw transferred Ridge ordering still fails strong controls. Keep the original HGB/simple-public comparisons as primary; no further calibration, feature, target or map search. No upstream competence claim changes.
