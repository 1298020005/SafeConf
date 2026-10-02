# Supplementary methods and results

## S1. Exact information budgets

The strict McFaline feedback experiment uses a fixed 331-record, 228-gene feedback pool and a disjoint 212-record, 152-gene evaluation pool. Percentages apply to eligible feedback gene clusters, not all target tasks. The Public+Target ledger records:

| Budget | Opened error records | Opened gene clusters | Extra validation errors for target-risk CDF | Validation records used for upstream calibration | Validation biological records available to Public |
|---|---:|---:|---:|---:|---:|
| 10% | 37 | 23 | 0 | 542 | 542 |
| 25% | 87 | 57 | 0 | 542 | 542 |
| 50% | 164 | 114 | 0 | 542 | 542 |
| 75% | 248 | 171 | 0 | 542 | 542 |
| 100% | 331 | 228 | 0 | 542 | 542 |

The biological validation effects are not labeled as model errors in the Public builder, but they are still target-study information and remain visible in this ledger. The upstream calibration and risk-CDF uses are different operations. Earlier feedback runs with validation-error CDFs are not pooled with this strict table.

For each budget, permitted records are selected first, the CDF is fitted to their errors only, and target learners use the resulting labels. A small-budget empirical CDF may have few unique levels. That is a property of the information condition, not grounds to silently use the full pool's CDF. If a group lacks ranking information, the failure state is recorded. The main evaluation uses realized task error, not a test-derived rank label for training.

## S2. PertEMA adaptation detail

The official estimator factory is pinned at commit `43c09a32e23d0ee2ae5dfbab21b2deeab27f1803`. It uses XGBoost with 300 estimators, learning rate 0.05, depth 6, subsample 0.8, colsample-by-tree 0.8, squared-error objective, and histogram tree construction. The SafeConf adapter replaces the original CD4-specific features with each registered information set and uses the budget-only task-RMSE rank target. Training has equal biological-cluster weights and fixed seeds.

The full-feedback first-seed results are:

| Same-information adaptation | Raw U20 | Group-OOF isotonic U20 |
|---|---:|---:|
| Prediction inputs | 0.0233 | 0.0278 |
| Public inputs | 0.7560 | 0.7447 |
| Public inputs plus Shared score | 0.7704 | 0.7704 |

An additional adaptation includes prediction-time native control statistics, state summaries and embeddings. Its raw U20 is 0.2173 without Public, 0.7371 with Public, and 0.6755 with the Shared score added. These are distinct information conditions and are not an equal-input ranking of learners. Isotonic calibration can create ties and change ranking statistics; all raw and calibrated outputs are retained. No conformal interval coverage is claimed because an independent conformal calibration set was not allocated in this adaptation.

## S3. Public distance identity and edge cases

The weighted-distance equality in the main paper follows by expanding each difference around the same weighted mean. The cross term vanishes because the weighted centered historical effects sum to zero. The result requires the same weights, gene coordinates, and mask in the mean and dispersion terms.

- For one historical record, its dispersion is zero and the weighted rule equals direct distance. This is not a confidence guarantee.
- For no eligible record, historical rules have no defined score. Risk learners use the explicit missing-history representation, and covered-cohort comparisons report their coverage.
- For repeated copies of one biological effect, record multiplicity is not treated as an independent replicate count.
- Conflicts in the original eligible set are retained independently of a concentrated learned weight distribution.
- Reconstruction and ranking optimize different objects. A reconstruction oracle is therefore not a task-risk oracle.

## S4. Model exposure and nested preprocessing

The outer risk split groups biological perturbations and synchronizes all predictions of the same true task. The inner public split constructs features for risk fitting without fitting the public learner to its own query outcome. Fitted preprocessors, including any public PCA and missing-value transforms, follow the same scope.

Membership must also be checked at the historical experimental-record level: an evaluation experiment cannot enter the learned training representation indirectly as a history of another query when the contract excludes it. Similarly, a source teacher checkpoint may have seen a biological response during upstream training even if its risk label is held out. The strengthened execution-branch table `risk_followup_v1/TEACHER_CONTEXT_PROVENANCE.csv` checks 32 E201/E205 checkpoint records, with zero target treated training and zero target perturbed access, no current inference context in source training contexts, and matching final checkpoint hashes. Source risk fitting uses only that context's source errors on outer training genes; its CDF does not include other contexts or target errors. Public models still share the outer gene fold and may use permitted other-context records. This contract supports rotating-context evaluation, not an unrun independent-study experiment. Old matrices retain their original development/seen scope.

## S5. Frangieh native512 stress experiment

The completed analysis uses 567 unique biological tasks and 189 genes. Three original upstream folds contain 279 test rows each, which produces repeated biological uses rather than 837 independent tasks. Five gene-grouped risk folds within each original upstream fold yield fifteen evaluation strata. The GEARS-to-scGPT and scGPT-to-GEARS directions are synchronized by gene hash.

History contains only original upstream training/validation records of the same perturbation in another context. It does not draw on the old test truths now used for risk-development outcomes. The supported held-out-context cohort contains 471 tasks, and historical-rule differences are recomputed on those same tasks. The evidence tables expose full-cohort and support-restricted counts separately.

All six original model/fold assets fail validation noninferiority against a mean baseline selected using training-gene out-of-fold results. Their relative macro error gaps range from 8.13% to 14.10%. The original validation set had been used for upstream early stopping, so this capability analysis is retrospective as well. The result is retained as an independent-family stress example, not an admission of these upstreams to qualified main validation.

The run performs 120 small risk fits, no upstream training, no data download, and zero GPU-hours. Its recorded wall time is 23.08 seconds and CPU time is 43.02 seconds. These describe one cached-data run, not a general runtime guarantee.

## S6. Zero-refit Ridge score repair

The separate RidgeScoreRepair version transforms every raw Ridge score as `0.5 + arctan(z - 0.5) / pi` in both directions and for both feature variants. It does not estimate a target percentile, change features, refit a model, or search a calibration map. All 120 primary model files and the original predictions and input hashes remain unchanged.

| Held-out-context direction | Original Prediction Ridge U20 | Repaired Prediction Ridge U20 | Original Public Ridge U20 | Repaired Public Ridge U20 | Magnitude U20 |
|---|---:|---:|---:|---:|---:|
| GEARS to scGPT | 0.1278 | 0.1278 | 0.7069 | 0.7069 | 0.1087 |
| scGPT to GEARS | -0.0562 | 0.0673 | -0.0562 | 0.0999 | 0.3789 |

For scGPT-to-GEARS, the repair-minus-original differences are 0.123554 [0.008461, 0.342436] for Prediction Ridge and 0.156161 [0.028426, 0.348401] for Public Ridge. These intervals use 5,000 paired gene-bootstrap draws of the same held-out tasks. Repaired Public Ridge versus weighted history uses the identical history-covered tasks and gives -0.060769 [-0.145905, -0.000811] for GEARS-to-scGPT and -0.744538 [-0.896116, -0.630596] for scGPT-to-GEARS. The correction recovers ordering destroyed by clipping but leaves poor raw transfer ordering unresolved. Public Ridge AURC increases to 0.058707 in scGPT-to-GEARS, from 0.057751 before repair. All six competence gates remain failed. This technical branch is complete and no additional map is searched.

## S7. Complete five-fold public-reference comparison

Four biological builders share the registered query/history information, outer gene folds, reconstruction loss and full 2,840-gene effect output. B0 is the support mean; B1 is the existing per-history HGB; B2 is pointwise neural scoring; B3 adds nonlinear collection context. The neural seeds are 20260930, 20261001 and 20261002. B0, B1 and fixed simple rules are deterministic seed-zero entries. Each seed's metric is computed within context-by-fold first; seed metrics are then averaged and the strata are macro-averaged. Neural seed replication does not multiply the number of biological units.

All arms contain the same 1,808 Source tasks/575 genes and 542 McFaline tasks/377 genes, with no missing planned tasks. Source contributes twenty strata per upstream and McFaline fifteen. `publicset_common_task_coverage.csv` records exact task membership. `publicset_macro_metric_intervals.csv`, `publicset_paired_comparisons.csv` and `publicset_strong_simple_comparisons.csv` provide all point estimates, percentile intervals, safety changes and practical-gate fields. The corresponding stratum tables retain both successful and unsuccessful strata; the seed table retains every fit.

| Biological reconstruction, macro RMSE | Source | McFaline |
|---|---:|---:|
| B0 support mean | 0.062596 | 0.029723 |
| B1 per-history HGB | 0.061179 | 0.029755 |
| B2 pointwise | 0.059746 | 0.029640 |
| B3 set-conditioned | 0.059739 | 0.029639 |
| Nearest control | 0.066622 | 0.041910 |
| Same-context support | 0.062596 | 0.032065 |
| Same-condition support | 0.062596 | 0.035243 |

The same-context and same-condition matching rules fall back to the complete allowed history when no match exists. Control distance is calculated at prediction time and ties use experiment identifier. These rules use no evaluated query outcome to select their subset. In Source, same-context fallback equals B0 because eligible history is from other contexts. In McFaline, same-context histories from other conditions exist and produce a distinct strong rule. Its historical-reader U20 is 0.872175, compared with B2 0.869870 and B3 0.864788. Paired B2/B3-minus-same-context intervals include zero; neither supplies the minimum practical increment.

The fixed-reader replacement decision is `STOP_PUBLICSET_ARCHITECTURE_RETAIN_B1`: a candidate must pass every registered baseline comparison in each upstream/domain view, and B3 must also pass its structure comparison with B2. This result is restricted to the historical reader R_H. It does not exclude utility from an independently assessed nested supervised reader.

### S7.1 Exact bootstrap and endpoint semantics

There are 5,000 paired cluster draws, bootstrap seed 20261002. Source draws sample all 575 genes and reuse the same integer multiplicities across both upstream predictors, folds, reference builders and fitting seeds. McFaline's 377 genes are sampled independently. Multiplicity is implemented as repeated task copies, including ties and partial top-k blocks; it is not converted to a continuous sample weight with different rounding. Sixty literal-repeat checks cover U20, AURC, miss rate, retained errors and biological metrics, with maximum tolerance 2e-12.

For U20 and retained-error cutoffs, task counts use ceil. AURC is the mean of cumulative mean realized errors sorted from low predicted risk, without normalizing by full-cohort mean error. High-risk miss rate is one minus the fraction of the oracle top20 task copies captured in the predicted top20. A bootstrap macro draw with fewer than 80% of planned valid strata is unavailable. Primary intervals are percentiles of valid paired differences, conditional on the fitted models. Each baseline and safety condition remains visible; absence of a significant difference alone does not satisfy a practical benefit requirement.

### S7.2 Alignment and constrained reconstruction

`publicset_mc_data_alignment_comparisons.csv` contrasts the same B0 rule using old equal-guide effects and the cell-weighted bank. On the 542-task fold-based cohort, historical-reader U20 increases by 0.159165 [0.041750, 0.214427], with 93.3% nonnegative strata and a passed macro safety gate. This is a data-contract effect, separately versioned from the earlier 543-task alignment diagnostic in Table 7. It is not credited to B2 or B3.

The constrained reconstruction diagnostic covers every query: 1,808 Source and 542 McFaline records. It optimizes a simplex coefficient a under exactly the permitted weights w=0.5a+0.5support using the evaluated biological response. Every solver succeeds. Source has one to three histories per query (mean 2.620); McFaline has seven to fourteen (mean 12.500). Mean per-query support reconstruction MSE versus constrained best MSE is 0.004353 versus 0.003829 on Source and 0.000918 versus 0.000896 on McFaline. These task means are descriptive and are not the context-fold macro RMSE in the preceding table. The result is an unattainable outcome-informed reconstruction diagnostic, not a feasible risk method or an upper bound on U20.

## S8. Registered fully nested readers and three source-label conditions

[[REGISTERED_READER_SUPPLEMENT]]

## S9. Computation and version binding

The resumed four-arm run completes in 599.11 seconds, executes 45 new neural fitting stages in 422.89 seconds and reuses 38 checked neural-model records from the interrupted full run and repaired prototype. These are continuation costs, not a claim about training all arms from scratch. The fitting and reuse ledgers preserve the selected epochs, query hashes, unique PCA-history hashes and effective parameters. PublicSet performs no upstream predictor retraining and no new McFaline test-array reads. The fixed matching references use no fit and require 15.59 seconds for the cached-data comparison. The paired statistics take 72.26 seconds on four declared CPU threads.

Every result file copied into this package is bound in `evidence/RESULT_MANIFEST.json` by original path, row count where applicable, and SHA-256. The public fitting manifest additionally binds the aligned gene axis, query tables, biological effect arrays and public-memory manifests. Manuscript rebuilding and numeric verification consume only these completed records; neither starts experimental fitting nor reads evaluation truth arrays.
