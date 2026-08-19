# E1-E4 preregistered audit report

This report records gate outcomes without changing frozen protocol v0.2.

## Decisions

### E1

```json
{
  "passing_groups": [
    "context",
    "support",
    "prediction_output"
  ],
  "n_passing_groups": 3,
  "gate": "pass",
  "gate_note": "Unadjusted paired bootstrap CIs are an across-dataset consistency gate, not multiplicity-adjusted inference."
}
```

### E2

```json
{
  "isotonic_partial_ci_positive_datasets": 7,
  "isotonic_aurc_improvement_ci_positive_datasets": 7,
  "partial_gate": "pass",
  "aurc_gate": "pass",
  "overall_gate": "pass"
}
```

### E3

```json
{
  "observed_vs_target_null_fdr_significant_datasets": 7,
  "observed_vs_feature_null_fdr_significant_datasets": 7,
  "missingness_only_fdr_significant_datasets": 0,
  "full_plus_missingness_ci_positive_datasets": 2,
  "target_null_gate": "pass",
  "feature_null_gate": "pass",
  "missingness_gate": "pass"
}
```

### E4

```json
{
  "datasets_with_at_least_9_of_10_positive_seeds": 7,
  "datasets_with_at_least_3_of_6_positive_configs": 7,
  "seed_gate": "pass",
  "config_gate": "pass",
  "note": "HistGBT is close to deterministic; configuration sensitivity is more informative than seed sensitivity."
}
```

## E3 gate correction

A useful observed model should exceed the permutation null, so its one-sided empirical p-value should be small. The supplied `p > 0.10` direction was reversed before execution. Missingness-only remains a negative control and should not be significant.
