# Fixed Orion failure: descriptive feature audit

This audit used the original13 model features from the actual3616 Source OOF risk-fit records (575 genes), the completed fixed232-query primary identities, their frozen Manual/Learned features and prediction-time scores. Source and target feature identities/hashes match their existing registries. It made zero fits, predictions or new bootstrap draws and read no raw expression, effect/error-label vector or TEST error parquet. Existing aggregate seven-metric summaries were copied as outcome context. Independent code/input-metadata review passed.

## Observed feature ranges

| Fixed feature | Source reference | HCT116 (107 queries) | HEK293T (125 queries) |
| --- | --- | --- | --- |
| Predicted magnitude | min .003140; median .046464; q01 .008294 | median .002169;75.7% below Source min;100% below q01 | median .003108;52.8% below min;98.4% outside q01–q99 |
| Learned prediction/prior cosine | min .159451; median .923289 | median .215381;30.8% below min | median .153982;53.6% below min |
| Manual prediction/prior cosine | min .174198; median .951844 | median .238770;33.6% below min | median .149531;56.0% below min |
| Log history support | min3.0445,max8.2268;median5.3799 | median5.2523;0% outside min/max | median5.2832;0% outside min/max |

Prior magnitude is inside its corresponding Source min/max for about98% of each context. Prior uncertainty and history conflict are inside Source min/max for every fixed query. Manual/Learned effective-source counts exceed Source maxima in about7–14% of the queries, and prediction/prior RMSE exceeds maxima in about10–15%. There are no missing/nonfinite values in the audited13 Source or target features. Full finite-only quantiles/min/max/missingness for every feature, plus both Source predictors separately, are in `FEATURE_DISTRIBUTIONS.csv`; every context's strict range and percentile-tail rates are in `TARGET_OUTSIDE_SOURCE_RANGES.csv`.

These differences describe the saved feature representation. They do not establish a cause of failure, a statistically significant shift or a justification for target calibration. The audit uses unweighted record-level descriptive Source quantiles; records/context/gene are not treated as independent experimental replication.

## Frozen score ties

P-only HGB has6 distinct exact scores in HCT116 and3 in HEK293T. Its largest tied groups contain80/107 and123/125 queries; the fixed20% selection boundary cuts those groups. Learned HGB remains much more differentiated:102/107 and114/125 distinct scores, with10/107 and20/125 rows involved in ties. Its20% boundary is untied in both contexts. Learned historical distance has no exact ties. Thus the P-only tie pattern does not by itself explain the Learned HGB primary comparison.

Negative Source support has100/107 and118/125 distinct scores and an untied20% boundary. Its observed negative utility point estimates coexist with support values lying inside the historical training range. Saved Ridge support coefficients are negative in both references; these are historical model-structure facts, not evidence about the Target relationship or the HGB's causal behavior. All tie definitions use exact float64 equality and frozen query-ID ordering, never labels or rounded thresholds.

## Existing outcome evidence and limits

The already completed U20 results are preserved without recalculation: Learned HGB versus learned historical distance is .160524 versus .236258 in HCT116 and .249146 versus .193517 in HEK293T; macro .204835 versus .214887. Negative support U20 is -.189015/-.123552 across the two contexts, macro -.156284. Its stored confidence intervals include zero. The primary paired interval also includes zero, as reported in the fixed evaluation; this audit does not change the failure/safety rule or claim a stable universal disadvantage.

The co-occurrence of low prediction-only magnitudes, lower prediction/prior alignment, historical-support sign reversal and different ranking granularity is a descriptive boundary of the frozen transfer system. It does not isolate which component caused the outcome, establish a rescue method or establish that a new Public/risk learner would help. No threshold, parameter, Source CDF, query cohort or model is changed. The actual seven-model Source version is the one in manifestb74; no claim that this was a newly fitted V2 learner is made.
