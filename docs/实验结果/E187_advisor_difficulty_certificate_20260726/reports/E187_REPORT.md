# E187 | Advisor-defined difficulty ladder certificate audit

## Status

This is a retrospective re-analysis of prediction records frozen in E98, E100,
E103, E87, and E89. It does not alter any split, predictor, target, or revealed
truth. The lower certificate uses prediction vectors only; truth is used solely
to evaluate tightness and violations.

## Within-study Cartesian settings

The audit contains 8,196 task instances from four data sets,
four training-submatrix fractions, and four test settings. The two-member family
lower bound had 0 family-RMS
violations and 0 worst-member
violations.

At the 100% training-submatrix level:

- Random pair: macro median tightness 0.328 (95% cluster bootstrap interval 0.315–0.342).
- Unseen context: macro median tightness 0.260 (95% cluster bootstrap interval 0.252–0.269).
- Unseen perturbation: macro median tightness 0.175 (95% cluster bootstrap interval 0.163–0.183).
- Double unseen: macro median tightness 0.148 (95% cluster bootstrap interval 0.133–0.158).

The double-unseen setting was the least informative on average. This is a
tightness result, not a validity failure: the deterministic lower inequality
remained exact in every setting.

## Direct cross-dataset transfer

- sciPlex3_to_OpenProblems: 553 tasks, zero lower-bound violations, median tightness 0.703; no target calibration was available, so no cross-dataset upper coverage claim was made.
- sciPlex3_to_sciPlex4: 28 tasks, zero lower-bound violations, median tightness 0.641; no target calibration was available, so no cross-dataset upper coverage claim was made.

The high cross-dataset tightness arose because the two transferred predictors
often failed by different amounts. It certifies that at least one family member
has substantial error; it does not identify the failed member and does not make
small disagreement safe.

## Relation to the advisor's questions

1. The scored error object is family RMS and family worst-member RMSE, not an
   unspecified model error.
2. Prediction magnitude and model distance are calculated from prediction
   vectors. Target perturbed expression is absent from the score.
3. Random-pair, unseen-context, unseen-perturbation, double-unseen, and
   25%/50%/75%/100% training-submatrix settings are all represented.
4. Direct dataset transfer is retained as a boundary analysis. Without target
   calibration, only the deterministic lower certificate is claimed.

## Interpretation limits

- The Cartesian genetic and cytokine experiments use two inductive reference
  predictors, not the five-seed scGPT and GEARS family used in the main
  confirmation studies.
- The same biological task may recur across training fractions and folds.
  Counts are task instances, not independent experiments.
- Bootstrap intervals resample perturbation identities within each data set and
  average data-set medians; they do not treat folds as independent studies.
- No conformal upper bound is transported directly across data sets.

## Reproducible outputs

- `tables/E187_CARTESIAN_TASK_CERTIFICATES.csv`
- `tables/E187_CARTESIAN_SETTING_SUMMARY.csv`
- `tables/E187_MACRO_BOOTSTRAP.csv`
- `tables/E187_CROSS_DATASET_TASK_CERTIFICATES.csv`
- `tables/E187_CROSS_DATASET_SUMMARY.csv`
- `tables/INPUT_HASHES.csv`
- `figures/Figure_E187_difficulty_ladder.*`
