# Actual Source prediction-scale stress — complete, diagnostic only

The fixed Source DEV experiment completed once, exit0, in9.998462s,10.335635CPU seconds, peakRSS567353344bytes. It reused1808tasks/575genes, two existing architectures and four contexts; all three scales1/.1/.05 and five scoring rules are reported. Six frozen HGB batch predictions were executed, with zero fits, upstream calls, downloads, CDF fits, Target/Orion accesses or bootstrap draws. The formal upstream attempt count remains2.

All conditions were hashsealed before Source truth numeric access. Baseline features, both fitted-risk scores and Source RMSE exactly match original bytes at s=1. The original101artifacts plus registry and executing dependencies retain their before/after hashes. Final risk models were trained on these same Source rows: cached OOF Public priors do **not** make this held-out risk evaluation.

| Existing architecture | Scale | P-only HGB U20 | Manual HGB U20 | Same-prior weighted distance U20 |
|---|---:|---:|---:|---:|
| GAT |1|0.798533|0.827282|0.589296|
| GAT |.1|0.426607|0.722050|0.747305|
| GAT |.05|0.008382|0.733208|0.762013|
| Exphormer |1|0.799634|0.820810|0.588347|
| Exphormer |.1|0.368840|0.728436|0.745511|
| Exphormer |.05|-0.000895|0.741802|0.762112|

These are equal-four-context descriptive point estimates, not independent transfer or confidence intervals. Scaled predictions change the actual RMSE ordering and U20 denominator; comparing U20 across scales does not isolate a fixed-target model degradation.

## Independent interpretation and stop

- At both lower scales, Manual HGB−distance becomes negative in K562/RPE1/HepG2 for both predictors; Jurkat retains a positive contrast. All24strata are retained.
- The final Public HGB remains informative:403–550unique scores, maximum ties2–5, risk-order Spearman versus original.890–.923. Its macro Spearman with changed errors remains approximately.709–.716. At all four scaled macro comparisons, error@10 remains better than weighted distance. A U20 reversal is not deterioration of every metric.
- P-only HGB is strongly compressed at scale.05:5–8distinct scores per stratum and macroU20 nearzero. The frozen tree representation is sensitive to this artificial low-output regime.
- Error rankings themselves change, Spearman.746–.960 relative to scale1. The final HGB loss and improvement of the distance comparator both contribute to the relative reversal. This does not identify Orion's failure cause or justify the proposed ECDF anchor.
- The coarse directional check is met, but no repair is activated. Stop scale variants, new bootstrap and attempts to rescue this same hypothesis. Current scientific publication readiness remains unproven.

## Actual technical correction

Before any real run, independent review rejected v1 because pandas macro mean skipped undefined metrics while reporting four contexts. The exact rejected code snapshot and FAIL receipt remain in `../source_counterfactual_scale_stress_v1/`. v2 requires all four contexts per metric and reports finite-context counts; constant-risk strata are retained. This is a reporting correction, not a change to models, features, scales or metrics. v2 independently passed static review before Root authorization.

The authoritative small receipts are `RESULT_MANIFEST.json`, `INDEPENDENT_SCIENTIFIC_REVIEW_AND_STOP.json` and `ACTUAL_ACCOUNTING.json`. All120stratum and30macro rows are appended to the master table with explicit in-sample counterfactual roles, not mixed into primary transfer results. Dense feature/score/error arrays remain in `/home/yyf/runtime_artifacts/safeconf_research_20261001/source_counterfactual_scale_stress_20261002_v2`.
