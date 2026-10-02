# Frangieh native512: retrospective scGPT ↔ GEARS risk transfer

This is a new **SEEN retrospective** analysis of already-opened upstream prediction caches. Old upstream TEST is re-registered as risk-development data, with five gene-grouped risk folds inside each original checkpoint fold. It does not inherit the original confirmation claim. Both directions use the same hash assignment; each source learner receives only that source model's own errors and prediction features. Target errors are evaluation-only for that direction. No E108 ensemble-error/disagreement risk score is reused.

## Contract and interpretation

- Original folds remain separate. 279 original TEST rows / 189 genes per fold; all three upstream folds overlap biologically and are synchronized in gene bootstrap.
- Public history contains only original upstream TRAIN/VAL experiments of the same perturbation in another context. No old TEST truth can become another query's history. Public is support weighted, requires no learner/CDF; quality is unavailable and is not invented from cell counts.
- Native512 is a separate within-study output contract. No common-axis zero filling, no claim of unchanged Source3285 parameter transfer.
- Same task truth is shared across the two model caches and checked numerically. Public features are built before realized errors attach. Input/feature identity fields never enter regressors.
- Training uses equal total weight per gene and a source-only, context-specific training ECDF; predictions are clipped to [0,1] following the existing rank-risk implementation. No target error rank is needed at inference.
- Source errors are from original upstream TEST, unused for upstream fit/early stopping. However most target genes were upstream-train-exposed in another context: this is held-out task plus unseen risk-training gene transfer, not all-genes-unseen upstream prediction.
- One fixed training seed; no hyperparameter search. All 120 model fits are saved. Three original folds are not three independent studies.
- all_test and heldout_context primary summaries macro-average original-fold × risk-fold strata. Other context results are additionally pooled over OOF risk predictions within an original fold. n<20 makes U20 NA.
- Direct history baselines are NA without history. Their pairwise bootstrap uses the common covered subset; compare coverage before interpreting their utility next to all-task scores.
- Native upstream competence uses the best of zero, TRAIN-global mean and TRAIN-context mean, selected using TRAIN gene-OOF, then evaluated on original validation. The validation was previously used for neural early stopping; it is not independent confirmation. All models remain in the report even if they fail competence.

## Main risk results

| direction | scope | method | utility20 | spearman | aurc | valid_strata | planned_strata |
|---|---|---|---|---|---|---|---|
| GEARS_to_scGPT | all_test | DirectHistoryDistance | 0.668385 | 0.520415 | 0.047710 | 15 | 15 |
| GEARS_to_scGPT | heldout_context | DirectHistoryDistance | 0.691551 | 0.631621 | 0.047561 | 15 | 15 |
| GEARS_to_scGPT | all_test | Magnitude | 0.092242 | -0.001088 | 0.051880 | 15 | 15 |
| GEARS_to_scGPT | heldout_context | Magnitude | 0.108732 | 0.052207 | 0.051951 | 15 | 15 |
| GEARS_to_scGPT | all_test | NegativeHistorySupport | -0.022087 | 0.128440 | 0.051679 | 15 | 15 |
| GEARS_to_scGPT | heldout_context | NegativeHistorySupport | 0.110402 | 0.323956 | 0.051553 | 15 | 15 |
| GEARS_to_scGPT | all_test | PredictionHGB | -0.108006 | -0.007312 | 0.051812 | 15 | 15 |
| GEARS_to_scGPT | heldout_context | PredictionHGB | -0.050475 | 0.033884 | 0.052512 | 15 | 15 |
| GEARS_to_scGPT | all_test | PredictionRidge | -0.023455 | -0.003755 | 0.052154 | 15 | 15 |
| GEARS_to_scGPT | heldout_context | PredictionRidge | 0.127844 | 0.077919 | 0.052566 | 15 | 15 |
| GEARS_to_scGPT | all_test | PublicHGB | 0.552565 | 0.494517 | 0.048434 | 15 | 15 |
| GEARS_to_scGPT | heldout_context | PublicHGB | 0.701422 | 0.661671 | 0.047976 | 15 | 15 |
| GEARS_to_scGPT | all_test | PublicRidge | 0.601367 | 0.386123 | 0.049448 | 15 | 15 |
| GEARS_to_scGPT | heldout_context | PublicRidge | 0.706859 | 0.569094 | 0.048818 | 15 | 15 |
| GEARS_to_scGPT | all_test | WeightedHistoryDistance | 0.890185 | 0.707412 | 0.047106 | 15 | 15 |
| GEARS_to_scGPT | heldout_context | WeightedHistoryDistance | 0.884100 | 0.769495 | 0.047092 | 15 | 15 |
| scGPT_to_GEARS | all_test | DirectHistoryDistance | 0.628866 | 0.479033 | 0.053054 | 15 | 15 |
| scGPT_to_GEARS | heldout_context | DirectHistoryDistance | 0.764028 | 0.658406 | 0.053074 | 15 | 15 |
| scGPT_to_GEARS | all_test | Magnitude | 0.332735 | 0.251751 | 0.054323 | 15 | 15 |
| scGPT_to_GEARS | heldout_context | Magnitude | 0.378869 | 0.300987 | 0.055824 | 15 | 15 |
| scGPT_to_GEARS | all_test | NegativeHistorySupport | -0.135812 | -0.066304 | 0.057376 | 15 | 15 |
| scGPT_to_GEARS | heldout_context | NegativeHistorySupport | 0.063291 | 0.239083 | 0.057968 | 15 | 15 |
| scGPT_to_GEARS | all_test | PredictionHGB | -0.104170 | -0.061711 | 0.056025 | 15 | 15 |
| scGPT_to_GEARS | heldout_context | PredictionHGB | -0.041587 | -0.072575 | 0.058662 | 15 | 15 |
| scGPT_to_GEARS | all_test | PredictionRidge | -0.245554 | nan | 0.054475 | 15 | 15 |
| scGPT_to_GEARS | heldout_context | PredictionRidge | -0.056218 | nan | 0.057751 | 15 | 15 |
| scGPT_to_GEARS | all_test | PublicHGB | 0.641323 | 0.444496 | 0.052698 | 15 | 15 |
| scGPT_to_GEARS | heldout_context | PublicHGB | 0.740099 | 0.623161 | 0.053537 | 15 | 15 |
| scGPT_to_GEARS | all_test | PublicRidge | -0.245554 | nan | 0.054475 | 15 | 15 |
| scGPT_to_GEARS | heldout_context | PublicRidge | -0.056218 | nan | 0.057751 | 15 | 15 |
| scGPT_to_GEARS | all_test | WeightedHistoryDistance | 0.879437 | 0.760489 | 0.051883 | 15 | 15 |
| scGPT_to_GEARS | heldout_context | WeightedHistoryDistance | 0.887906 | 0.807961 | 0.052587 | 15 | 15 |

## Paired gene bootstrap (5000 draws)

| direction | scope | method | baseline | delta_utility20 | ci95_lower | ci95_upper | n_paired_gene_clusters |
|---|---|---|---|---|---|---|---|
| GEARS_to_scGPT | all_test | PublicHGB | Magnitude | 0.460323 | 0.316803 | 0.704281 | 189 |
| GEARS_to_scGPT | heldout_context | PublicHGB | Magnitude | 0.592689 | 0.439170 | 0.771305 | 189 |
| GEARS_to_scGPT | all_test | PublicHGB | PredictionHGB | 0.660570 | 0.524863 | 0.856127 | 189 |
| GEARS_to_scGPT | heldout_context | PublicHGB | PredictionHGB | 0.751897 | 0.601259 | 0.901153 | 189 |
| GEARS_to_scGPT | all_test | PublicHGB | NegativeHistorySupport | 0.574652 | 0.397069 | 0.830098 | 189 |
| GEARS_to_scGPT | heldout_context | PublicHGB | NegativeHistorySupport | 0.591020 | 0.365823 | 0.806414 | 189 |
| GEARS_to_scGPT | all_test | PublicRidge | Magnitude | 0.509125 | 0.348384 | 0.732623 | 189 |
| GEARS_to_scGPT | heldout_context | PublicRidge | Magnitude | 0.598126 | 0.443081 | 0.777550 | 189 |
| GEARS_to_scGPT | all_test | PublicHGB | WeightedHistoryDistance | -0.108237 | -0.174712 | 0.009203 | 189 |
| GEARS_to_scGPT | heldout_context | PublicHGB | WeightedHistoryDistance | -0.058246 | -0.144611 | 0.015472 | 189 |
| GEARS_to_scGPT | all_test | PublicHGB | DirectHistoryDistance | 0.113563 | 0.011157 | 0.320851 | 189 |
| GEARS_to_scGPT | heldout_context | PublicHGB | DirectHistoryDistance | 0.134302 | 0.012416 | 0.274454 | 189 |
| scGPT_to_GEARS | all_test | PublicHGB | Magnitude | 0.308587 | 0.079136 | 0.440430 | 189 |
| scGPT_to_GEARS | heldout_context | PublicHGB | Magnitude | 0.361230 | 0.124624 | 0.497324 | 189 |
| scGPT_to_GEARS | all_test | PublicHGB | PredictionHGB | 0.745492 | 0.594412 | 0.867971 | 189 |
| scGPT_to_GEARS | heldout_context | PublicHGB | PredictionHGB | 0.781686 | 0.613053 | 0.900945 | 189 |
| scGPT_to_GEARS | all_test | PublicHGB | NegativeHistorySupport | 0.777134 | 0.593145 | 0.953413 | 189 |
| scGPT_to_GEARS | heldout_context | PublicHGB | NegativeHistorySupport | 0.676808 | 0.431232 | 0.819355 | 189 |
| scGPT_to_GEARS | all_test | PublicRidge | Magnitude | -0.578290 | -0.761937 | -0.468196 | 189 |
| scGPT_to_GEARS | heldout_context | PublicRidge | Magnitude | -0.435088 | -0.662316 | -0.275980 | 189 |
| scGPT_to_GEARS | all_test | PublicHGB | WeightedHistoryDistance | -0.161288 | -0.254823 | -0.060439 | 189 |
| scGPT_to_GEARS | heldout_context | PublicHGB | WeightedHistoryDistance | -0.123904 | -0.240957 | -0.055226 | 189 |
| scGPT_to_GEARS | all_test | PublicHGB | DirectHistoryDistance | 0.089283 | -0.042151 | 0.323512 | 189 |
| scGPT_to_GEARS | heldout_context | PublicHGB | DirectHistoryDistance | -0.000025 | -0.110090 | 0.179838 | 189 |

## Reproduction

`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/scripts/run_safeconf_frangieh_cross_family_v1.py`

Existing completed predictions are protected from rerunning; `--report-only` recalculates statistics without fitting. Models and feature matrices are in the runtime directory recorded in CONFIG/ledger; compressed per-task predictions and all provenance/statistics are attached here.


## Completed competence audit and saved-model diagnosis

All six upstream/fold assets failed the registered competence gate: relative macro RMSE disadvantages to TRAIN-OOF-selected context mean are +8.13% to +14.10%. This run is retained as native-axis cross-family stress/sensitivity evidence; risk-ranking gains do not convert these upstreams into primary qualified validation. UPSTREAM_COMPETENCE.csv retains all estimates and uncertainty.

PublicHGB improves over prediction-only and magnitude in both directions. On the identical history-covered cohort, its heldout-context differences to WeightedHistoryDistance are GEARS→scGPT −0.058246 (95% CI [−0.144611, +0.015472]) and scGPT→GEARS −0.123904 ([−0.240957, −0.055226]). The current source-risk learner therefore does not establish an increment over this strong simple public comparator.

The saved-model diagnostic required zero refits. scGPT→GEARS PredictionRidge and PublicRidge each produced a single clipped score in all 15 risk-fold evaluations; raw predictions fell entirely below zero or above one. RISK_SCORE_RANGE_DIAGNOSTIC.csv records the ranges and feature-magnitude shift. This is a concrete transfer/clip failure boundary. No unclipped metric, calibration sweep, alternative target, or outcome-driven repair was run.
