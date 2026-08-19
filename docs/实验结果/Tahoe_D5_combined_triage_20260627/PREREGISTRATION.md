# Tahoe D5: fixed SafeConf + magnitude triage preregistration

Date: 2026-06-27

## Question

On the frozen Tahoe D1 test records, does a fixed combination of SafeConf risk and predicted-effect magnitude improve top-10% high-error retrieval over magnitude alone?

This is a practical triage sensitivity analysis. It does not modify frozen v0.2 and is not a new external dataset.

## Frozen input

```text
/home/yyf/safeconf_runtime/outputs/tahoe_d3_triage_20260625/tables/TAHOE_D3_SCORE_AUDIT.csv
```

The table was produced from the D1 300-shard held-out-pair test records before D5 was designed.

## Scores

Within each `(fold_id, predictor_name)` test batch, convert both scores to percentile ranks:

```text
safeconf_rank  = percentile_rank(safeconf_full)
magnitude_rank = percentile_rank(predicted_magnitude)
```

Primary score, frozen before calculation:

```text
combined_equal = 0.50 * safeconf_rank + 0.50 * magnitude_rank
```

Sensitivity scores, not used for the primary gate:

```text
combined_magnitude75 = 0.25 * safeconf_rank + 0.75 * magnitude_rank
combined_safeconf75  = 0.75 * safeconf_rank + 0.25 * magnitude_rank
```

No weight is selected using test errors. Percentile ranking uses prediction-side values only.

## Metrics

- Primary: top-10% high-error enrichment.
- Secondary: top-5% and top-20% enrichment; aligned Spearman rho.
- Comparators: SafeConf alone and magnitude alone.

## Uncertainty

- 1,000 bootstrap draws.
- Unit: `task_key`; both predictor rows for a task are resampled together.
- Seed: 5201.

## Gate

- `PASS_ADDS_VALUE`: combined-equal minus magnitude 95% CI lower bound > 0.
- `PASS_USEFUL_NOT_BETTER`: combined-equal enrichment CI lower bound > 1, but its difference from magnitude crosses 0 or is below 0.
- `FAIL`: combined-equal enrichment CI lower bound <= 1.

Only `PASS_ADDS_VALUE` supports the claim that the fixed combination improves over magnitude on Tahoe.

## Claim boundaries

- This is a post-D3, preregistered sensitivity analysis.
- It cannot be described as the frozen v0.2 score itself outperforming magnitude.
- It cannot establish universal complementarity across datasets or predictors.
- If the combination does not beat magnitude, the result is retained as a negative boundary.
