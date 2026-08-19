# Phase 5a-1 Methods Draft

Date: 2026-06-16
Status: Methods-only draft for Claude review

This draft is written from the frozen Evidence-to-Claim package. It does not add
new experiments, does not modify frozen protocol v0.2, and does not replace the
older manuscript drafts. It is intended as a clean Methods source for the current
task-risk / prediction-triage manuscript line.

## Methods

### Study design and claim scope

SafeConf was evaluated as a task-level reliability scoring protocol for
single-cell perturbation effect predictions. The central question was whether
features available before reading held-out prediction errors can rank prediction
error for held-out perturbation tasks, and whether this ranking retains signal
after controlling for effect magnitude. The primary frozen score was evaluated
without modifying the target prediction models and without requiring
model-specific uncertainty outputs.

All primary analyses used frozen output artifacts from completed SafeConf runs.
The scoring protocol was not re-fit on test errors. Held-out true effects and
true prediction errors were used only for evaluation. Learned residual-risk
models and external benchmark associations were treated as secondary analyses
or robustness analyses and were not used to modify frozen v0.2.

The allowed claim scope is task-risk scoring and prediction triage. The analyses
should not be described as a complete proof of predictor-agnostic performance,
unseen-axis out-of-distribution generalization, or full external validation of
all external benchmark methods.

### Main analysis population

The main formal analysis used seven single-cell perturbation datasets represented
by completed SafeConf run directories:

- CuiHacohen2023;
- Frangieh;
- LaraAstiasoHuntly2023 ex vivo;
- LaraAstiasoHuntly2023 in vivo;
- McFarlandTsherniak2020;
- SantinhaPlatt2023;
- SrivatsanTrapnell2020 sci-Plex3.

Each run contained prediction records, confidence features, confidence scores,
and effect-vector arrays. The corrected seven-dataset audit used explicit
drop-blank handling for small numbers of blank perturbation labels in Lara ex
vivo, Lara in vivo, and Santinha. The original run directories were not modified.
After this correction, all seven datasets were usable in the main formal table.
Santinha was assigned to the gene-main / CRISPR-cas9 family rather than the
chemical family. McFarland was retained as a frozen-protocol failure boundary.

Two scoring families were used as-run:

- `gene_main`, for gene perturbation or stimulation-style datasets;
- `chem_robust`, for drug-label/as-run chemical datasets.

This family assignment controlled which frozen v0.2 formula was applied; it was
not tuned to held-out performance.

### Perturbation tasks and prediction records

Each prediction record represents a model prediction for a task defined by a
cellular context, perturbation label, fold, split, and predictor. Effect vectors
were defined as perturbation effects relative to matched control cells within
the source dataset. Each record retained keys linking the predicted effect,
true effect, and target control mean arrays. The evaluation target was
`true_error_rmse`, the root-mean-squared error between the predicted effect
vector and the corresponding true effect vector.

The formal audit operated on the existing `PREDICTION_RECORDS.csv`,
`CONFIDENCE_FEATURES.csv`, `CONFIDENCE_SCORES.csv`, and associated NPZ arrays
from each run. These tables separate score construction from evaluation labels:
confidence features and frozen v0.2 scores do not require true test errors,
whereas `true_error_rmse` and true-effect magnitudes are introduced only in the
evaluation stage.

### Cross-validation split and leakage constraints

Evaluation used 5-fold cross-validation with a held-out `(context,
perturbation)` pair split. In each fold, test-set `(context, perturbation)`
pairs did not appear in the training set, but both the context and the
perturbation appeared separately in other training pairs. Folds were stratified
by perturbation label to balance perturbation representation across folds.

Feature computation used only fold-local training statistics under the L4
constraint. Test-set true effects and errors were never used for scoring. The
frozen scorer used fold-local training rows to define robust z-score reference
statistics, and learned robustness analyses used train/validation reference rows
for feature normalization while holding out test labels for evaluation.

### Reference predictors

The primary SafeConf artifacts were built around two retrieval-style reference
predictors.

`V0StrongBaseline` estimated a target effect using the mean effect of the same
perturbation in training tasks, falling back to the global training mean when
needed. When the target context appeared in training, this perturbation prior
was blended with a context-level mean effect using an 85/15 perturbation/context
blend.

`ContextSimilarityBaseline` used training tasks with the same perturbation and
weighted their effect vectors by cosine similarity between the target control
profile and source control profiles. If no same-perturbation source was
available, it fell back to a global mean. This predictor also produced
context-similarity information that was used as a SafeConf feature.

Because both predictors are retrieval-style baselines, conclusions about
task-risk transfer among them should not be generalized to arbitrary deep
learning predictors without qualification. The external benchmark association
analysis below provides aggregate method-error evidence on shared perturbations,
not full vector-level SafeConf scoring of those methods.

### SafeConf confidence features

Frozen v0.2 used three primary features:

- `context_similarity_max`, the strongest source-context similarity available
  for a target task;
- `perturbation_support_count`, the amount of fold-visible same-perturbation
  support, transformed as `log1p(support)`;
- `model_disagreement_rmse`, the RMSE disagreement between the two reference
  predictors.

Additional leakage-safe features were retained for learned-risk robustness
analyses, including context-similarity summaries, perturbation effect stability
and variance, historical residual risk, model-disagreement cosine, OOD-distance
features, and prediction-output magnitude features. These learned analyses used
quantile-normalized feature versions constructed within dataset/fold/predictor
groups from train/validation reference rows only.

### Frozen v0.2 scoring protocol

The primary frozen score was `protocol_v0_2_family_confidence`. For each
`(dataset, fold, predictor)` group, each feature was robustly z-scored using
the fold-local training rows as the reference distribution. The z-score used
the training median and interquartile range, with standard-deviation and unit
scale fallbacks for degenerate reference distributions. Test rows were scored
using the training reference statistics; held-out true effects and held-out
errors were not used for scoring.

For `gene_main` datasets, frozen v0.2 confidence was:

```text
z(context_similarity_max)
+ z(log1p(perturbation_support_count))
- z(model_disagreement_rmse)
```

For `chem_robust` datasets, the context term was omitted:

```text
z(log1p(perturbation_support_count))
- z(model_disagreement_rmse)
```

Higher values indicate higher confidence. For error-ranking analyses,
confidence was converted to a risk axis by negating the confidence score, so
higher risk corresponds to higher expected error.

### Primary evaluation metrics

The primary rank metric was direction-aligned Spearman correlation between the
SafeConf risk axis and `true_error_rmse` on held-out test records. For confidence
scores, the raw Spearman sign was inverted so that positive aligned rho means
that higher inferred risk corresponds to higher error.

To assess effect-magnitude confounding, the audit computed two additional
quantities:

- a magnitude-only baseline, defined as the Spearman correlation between true
  effect L2 norm and prediction error;
- a magnitude-controlled partial Spearman correlation, computed by rank
  residualizing both the SafeConf risk axis and prediction error against true
  effect L2 norm and correlating the residuals.

Risk-coverage summaries were also computed. AURC was calculated by sorting
predictions from low risk to high risk and averaging retained-set error across
coverage thresholds. The oracle AURC sorted predictions by true error, and
excess AURC was defined as scorer AURC minus oracle AURC. Lower AURC and lower
excess AURC indicate better selective prediction behavior.

Formal main-table confidence intervals used 1000 bootstrap resamples. Bootstrap
resampling was stratified by fold in the formal audit. The figure-ready tables
report aligned rho, magnitude-controlled partial rho, the magnitude-only
baseline, risk-coverage summaries, AURC, and excess AURC.

### Magnitude-residual calibration analysis

Because effect magnitude is a strong baseline predictor of error, a
fold-safe magnitude-residual analysis was run as a secondary defense against
magnitude-only explanations. This analysis used dataset-local LOPO training with
nested, group-cross-fitted magnitude calibration. In the isotonic-calibration
version, a magnitude model estimated expected error from true effect magnitude,
and a learned residual-risk model predicted remaining error structure from
SafeConf features. The combined predicted error was defined as magnitude-based
expected error plus the learned predicted residual.

This analysis is not the frozen v0.2 protocol. It tests whether SafeConf-derived
features contain residual signal beyond effect magnitude. The primary outputs
were residual partial rho and the AURC improvement of combined predicted error
over magnitude-only ranking. Bootstrap confidence intervals used 1000 resamples
over task clusters.

### Learned task-risk robustness analyses

Learned robustness analyses used the fold-safe LOPO feature matrix with a third
unseen predictor, `PertMeanPredictor`, which predicts each task using a
same-perturbation mean effect across training contexts. The learned model was
trained on errors from `V0StrongBaseline` and `ContextSimilarityBaseline` and
evaluated on the third predictor's held-out test errors. This design tests
whether task-risk signals transfer within a family of retrieval-style
predictors.

Six feature groups were pre-registered for ablation:

- context;
- support;
- historical;
- disagreement;
- OOD;
- prediction output.

HistGradientBoosting and ElasticNet configurations were used in stability
analyses. Seed sensitivity used ten seeds, although the HistGradientBoosting
configuration was close to deterministic; configuration sensitivity was
therefore treated as more informative than seed variance. These learned analyses
are secondary and do not change frozen v0.2.

### Negative controls

Permutation and nuisance-control analyses were used to check whether learned
risk signals could be explained by artifacts unrelated to task risk. The
negative controls included target permutation, feature permutation, and
missingness-only diagnostics. A useful observed model was required to exceed
the permutation null, whereas the missingness-only diagnostic was expected not
to pass. Empirical p-values and FDR-adjusted summaries were reported in the
E3 negative-control tables.

### External benchmark method-error association

The E8b analysis tested whether frozen SafeConf perturbation-level risk was
associated with official scPerturBench aggregate method errors on shared
biological datasets. This was pre-registered as an external benchmark
method-error association, not a full external validation and not vector-level
SafeConf scoring of all benchmark methods.

The primary E8b analysis used Frangieh, because 74 of 74 SafeConf perturbation
labels matched scPerturBench perturbations exactly. Frozen v0.2 gene-main scores
were computed using the real scorer. Scores were aggregated by taking medians
across V0/ContextSim within `(fold, context, perturbation)`, then across folds
within `(context, perturbation)`, then across contexts to obtain one risk value
per perturbation. Risk was defined as negative frozen confidence.

The primary benchmark metric was scPerturBench MSE at DEG=5000. For each
`(method, perturbation)` pair, available benchmark seeds were aggregated by the
median. Per-method Spearman correlations were computed between SafeConf risk and
benchmark error. Across-method evidence was summarized by the median Spearman
across methods. Perturbation-cluster bootstrap with 1000 resamples was used to
obtain a confidence interval for the across-method median. A 200-permutation
shuffled-risk null was used as a negative control.

E8b also reported a sample-size baseline,
`sample_size_risk = -log1p(Nstimulated)`, and a post hoc partial analysis
controlling log-transformed stimulated-cell count. This sample-size analysis is
a diagnostic caveat, not the pre-registered E8b gate. Sensitivity analyses
included alternative DEG cutoffs, Pearson distance, and a sciplex3 drug
sensitivity analysis using an explicit exact/alias drug mapping. The sciplex3
analysis was excluded from the primary E8b gate.

### Prediction triage and high-error retrieval

Prediction triage analyses used the existing B1 bad-prediction retrieval table.
For each dataset, score, and top-risk fraction, the analysis selected the
highest-risk predictions and measured enrichment for high-error predictions
relative to random selection. The primary practical-value display used the top
10% risk threshold and macro-averaged enrichment across the seven real datasets.

Strategies included random selection, predicted magnitude, frozen v0.2,
cross-dataset LODO risk, within-dataset risk, and a non-deployable oracle
magnitude diagnostic. The within-dataset risk is a reference upper bound rather
than the deployable frozen protocol. The oracle uses evaluation information and
is shown only as a ceiling reference. Per-dataset top-10% enrichment was plotted
in the main Fig 5 heatmap, and the full top 5% / 10% / 20% threshold heatmap was
retained as a supplementary figure.

The intended interpretation is practical triage: how efficiently a risk score
prioritizes predictions likely to be wrong under a limited validation budget.
Macro-averaged frozen v0.2 enrichment should be described as comparable to
magnitude-only at top 10%, with complementary per-dataset strengths, not as
uniform superiority over magnitude.

### Reproducibility and source tables

All figure-ready tables and plots in the current evidence package were generated
from frozen result tables without re-running large experiments. Source files,
source commits, generated figure-ready CSVs, and draft figure outputs are listed
in `REPRODUCIBILITY_MANIFEST.md`. The plotting scripts are
`build_figure_ready_tables.py` and `plot_phase4c_figures.py`.

The current Methods draft is tied to the following evidence package:

```text
docs/实验结果/Evidence_to_Claim_20260615/
```

The key source-of-truth files for manuscript claims are:

- `SAFE_CONF_EVIDENCE_TO_CLAIM_MATRIX.md`;
- `REPRODUCIBILITY_MANIFEST.md`;
- `figure_ready_tables/`;
- `figures/`;
- `PHASE5A0_COST_EFFECTIVENESS_REPORT.md`.

## Boundaries To Preserve In Results And Discussion

- Do not write that frozen SafeConf succeeds on 7/7 datasets; McFarland remains
  a frozen-protocol failure boundary.
- Do not write that frozen SafeConf generally outperforms magnitude-only;
  magnitude is a strong baseline and often has higher rank correlation.
- Do not use E2 learned residual calibration as if it were frozen v0.2.
- Do not write that E8b validates 27 architectures or constitutes complete
  external validation.
- Do not write that Fig 5 shows frozen v0.2 uniformly matches magnitude
  per-dataset; the correct statement is macro-averaged comparability with
  complementary per-dataset strengths.
