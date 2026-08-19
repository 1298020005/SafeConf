# Phase 5a-2 Results Draft

Date: 2026-06-16
Status: Results-only draft for Claude review

This draft follows the current Evidence-to-Claim package. It is intentionally
claim-bounded: no new experiments, no frozen v0.2 modification, no claim of
complete predictor-agnostic validation, and no claim of uniform superiority over
effect magnitude.

## Results

### 2.1 Task identity dominates error structure across reference predictors

SafeConf is motivated by a simple empirical observation: for the current
retrieval-style reference predictors, prediction error is largely a property of
the task rather than of the specific predictor. We paired the held-out errors of
`V0StrongBaseline` and `ContextSimilarityBaseline` for matched tasks across the
seven main datasets. Across 4,584 paired held-out tasks, the two predictors'
errors were highly correlated (Spearman rho = 0.973). A two-way variance
decomposition attributed 93.6% of error variance to task identity, 0.1% to
predictor identity, and 6.3% to residual variation (Fig. 1).

This result does not imply that predictor choice is irrelevant, nor that all
deep learning predictors must share the same error structure. Rather, it shows
that in the current benchmark, tasks that are difficult for one retrieval-style
predictor tend to be difficult for the other. This observation makes task-level
risk scoring a meaningful target: before deciding which predictions to validate
experimentally, one can ask whether the task itself is likely to be unreliable.

### 2.2 Frozen v0.2 ranks prediction error in most main datasets, with McFarland as a failure boundary

We next evaluated the frozen SafeConf v0.2 score on seven formal main datasets
using 5-fold held-out `(context, perturbation)` pair splits. In six of seven
datasets, the direction-aligned Spearman correlation between frozen risk and
held-out RMSE was positive. Magnitude-controlled partial rho was also positive
in six of seven datasets (Fig. 2): Cui (partial rho = 0.328, 95% CI
[0.293, 0.362]), Frangieh (0.474 [0.430, 0.510]), Lara ex vivo (0.443
[0.376, 0.506]), Lara in vivo (0.357 [0.290, 0.424]), Santinha (0.212
[0.129, 0.297]), and Srivatsan sci-Plex3 (0.629 [0.595, 0.660]).

McFarland was the exception. Frozen v0.2 had aligned rho = -0.086 and
magnitude-controlled partial rho = -0.061 (95% CI [-0.100, -0.023]), and was
therefore retained as a failure boundary rather than removed from the analysis.
Santinha was retained as weak positive CRISPR evidence after ontology correction
to `gene_main`, but its effect size was small and should not be interpreted as
a strong positive dataset.

The magnitude-only baseline was strong across datasets, and often stronger than
the frozen score in rank-correlation terms. This makes Fig. 2 a primary rank
signal result, not a claim that frozen v0.2 generally outperforms magnitude.
The next analyses therefore focus on whether SafeConf-derived features retain
information beyond effect magnitude.

### 2.3 Learned residual-risk calibration detects signal beyond effect magnitude

To test whether SafeConf features contain residual information after accounting
for effect magnitude, we ran a fold-safe magnitude-residual calibration analysis.
This analysis is not the frozen v0.2 protocol; it is a learned extension used to
interrogate the magnitude-bias criticism. The magnitude component estimated
expected error from true effect magnitude, and the residual-risk model used
SafeConf-derived features to predict remaining error structure.

The residual-risk signal was positive in all seven datasets, with 95% bootstrap
confidence intervals above zero: Cui (partial rho = 0.600 [0.555, 0.644]),
Frangieh (0.232 [0.132, 0.327]), Lara ex vivo (0.638 [0.556, 0.700]),
Lara in vivo (0.615 [0.519, 0.688]), McFarland (0.226 [0.173, 0.281]),
Santinha (0.291 [0.175, 0.409]), and Srivatsan sci-Plex3 (0.154
[0.065, 0.241]) (Fig. 3A).

The combined predicted error, defined as magnitude-based expected error plus
learned predicted residual, also improved AURC over magnitude-only ranking in
all seven datasets. The AURC improvement confidence interval was above zero in
every dataset, satisfying the preregistered 4/7 gate. These results support the
bounded claim that a learned residual-risk model captures signal beyond
magnitude alone; they should not be read as frozen v0.2 itself surpassing
magnitude.

### 2.4 Learned task-risk models expose signals missed by the frozen formula

The frozen formula was deliberately simple and fixed before these robustness
analyses. We therefore asked whether a learned task-risk model could recover
risk signal missed by the frozen heuristic. In fold-safe LOPO analyses using a
third predictor (`PertMeanPredictor`), the learned full-feature model produced
positive magnitude-controlled partial rho in all seven datasets. McFarland was
the most informative case: frozen v0.2 failed on McFarland (partial rho =
-0.061), whereas the learned LOPO risk model produced partial rho = 0.331
(Fig. 3B).

This does not rescue frozen v0.2, and it does not change McFarland's status as a
frozen-protocol failure boundary. Instead, it shows that risk signal exists in
that dataset but is not captured by the current frozen feature combination.
Feature-group ablations further showed that useful risk sources were
dataset-dependent. In LOPO mode, dropping context, support, or prediction-output
features produced CI-positive degradation in 5/7, 3/7, and 3/7 datasets,
respectively. In LODOxLOPO mode, prediction-output and disagreement groups each
had 4/7 CI-positive degradation, whereas no single feature group dominated all
settings. This supports a cautious interpretation: task-risk signal is
learnable, but its feature basis varies across datasets.

### 2.5 External benchmark method-error association on Frangieh

We next tested whether frozen SafeConf task risk was associated with official
scPerturBench aggregate method errors on a shared biological dataset. This E8b
analysis used Frangieh because all 74 perturbations matched exactly between the
SafeConf and scPerturBench tables. The primary benchmark endpoint was MSE at
DEG=5000. For each method, we correlated perturbation-level SafeConf risk with
the method's per-perturbation benchmark error.

The primary Frangieh analysis passed its preregistered gate. Across 15 methods,
the median Spearman rho was 0.584 with perturbation-bootstrap 95% CI
[0.393, 0.726], and 14/15 methods had positive rho (Fig. 4A). A shuffled-risk
null was centered near zero (median 0.007, 95% range [-0.232, 0.231];
one-sided empirical p = 1/201).

This association was not uniform across all methods: 12/15 methods clustered at
rho approximately 0.55-0.62, whereas baseMLP and scFoundation were near zero
and CPA was weaker. The association was also sensitive to the error metric and
gene set. It was strongest for MSE at DEG=5000 and did not generalize uniformly
to all alternative metrics.

Sample size was an important caveat. A sample-size baseline based on
`Nstimulated` had median rho = 0.764, stronger than frozen SafeConf risk. After
post hoc control for log-transformed stimulated-cell count, the median partial
rho remained positive but smaller (0.335, 95% CI [0.047, 0.538]) (Fig. 4B).
Thus E8b supports an external benchmark method-error association on shared
biological tasks, not a full external validation of SafeConf on all benchmark
methods.

### 2.6 Negative controls and robustness checks reduce artifact-only explanations

Several controls were used to test whether learned risk associations could be
explained by label permutation, feature permutation, or missingness artifacts.
In E3, the observed model exceeded both shuffled-target and jointly shuffled
feature nulls in all seven datasets after FDR adjustment. By contrast, the
missingness-only diagnostic did not pass the negative-control gate: 0/7 datasets
were FDR-significant for missingness-only risk, and adding missingness produced
CI-positive improvement in only 2/7 datasets.

Model stability analyses also supported the robustness of the learned
association. Across ten seeds, all seven datasets had positive learned LOPO
partial rho. Because the HistGradientBoosting configuration was nearly
deterministic, configuration sensitivity was treated as more informative than
seed variance; all seven datasets had at least 3/6 positive configurations.

These controls do not eliminate all possible confounding. They do, however,
argue against the simplest artifact-only explanations: that the signal is
created by random labels, shuffled feature-risk mapping, or a missingness
signature alone.

### 2.7 Prediction triage translates risk scores into validation priorities

Finally, we asked whether risk scores can prioritize predictions for a limited
experimental validation budget. Using the existing bad-prediction retrieval
table, we selected the top-risk predictions and measured enrichment for
high-error predictions relative to random selection. At the top 10% risk
threshold, frozen SafeConf identified 3.35x more high-error predictions than
random selection, comparable to the magnitude-only baseline in macro average
(3.30x). Cross-dataset LODO risk achieved 2.31x enrichment, whereas
within-dataset risk reached 5.36x and the non-deployable oracle reference
reached 8.21x (Fig. 5A).

The macro average hides strong per-dataset heterogeneity. Frozen scoring
outperformed magnitude on datasets where effect magnitude poorly predicted
error, most notably Lara ex vivo (7.80x versus 0.46x at top 10%). Conversely,
magnitude dominated on datasets with strong magnitude-error coupling, including
Frangieh (8.08x versus 2.90x) and Srivatsan sci-Plex3 (6.27x versus 4.95x)
(Fig. 5B). McFarland again illustrated the failure boundary: frozen v0.2 was
below random at top 10% (0.90x), whereas magnitude reached 2.57x.

These results support prediction triage as a practical interpretation of
SafeConf: the score can help rank which predictions are most worth checking
under a limited validation budget. The correct claim is not that frozen v0.2
uniformly matches or beats magnitude, but that frozen scoring and magnitude
capture complementary per-dataset risk dimensions.

## Results Boundary Notes For Revision

- Do not convert "6/7 positive with McFarland failure boundary" into "7/7
  success".
- Do not describe E2 as the frozen v0.2 result.
- Do not describe E8b as validating 27 architectures.
- Do not write that frozen v0.2 is generally better than magnitude.
- Keep Fig. 5 language as macro-averaged comparability plus per-dataset
  complementarity.
