# Stage completion checklist v2.1

## Metrics used for gates
```json
{
  "test_records": 4458,
  "leak_counts": {
    "pair_leak": 0,
    "pert_missing": 0,
    "context_missing": 0
  },
  "kcc_simple_combined_aligned_rho": NaN,
  "kcc_gate_applicable": false,
  "simple_combined_median_aligned_rho": 0.7097720175132364,
  "max_learned_minus_disagreement": 0.1594087119829174,
  "n_genes_ok": false
}
```

| Gate | Status | Evidence |
|---|---|---|
| G1 | PASS | tables/CONFIDENCE_EVAL_SUMMARY.csv level=dataset |
| G2 | PASS | test_records=4458 |
| G3 | PASS | leak_counts={'pair_leak': 0, 'pert_missing': 0, 'context_missing': 0} |
| G4 | PASS | MVP_V2_1_REPORT.md starts with per-dataset table |
| G5 | PASS | historical_residual_risk in CONFIDENCE_EVAL_SUMMARY.csv |
| G6 | PASS | tables/LEARNED_RISK_FOLD_STATUS.csv |
| G7 | SKIPPED_NOT_APPLICABLE | KaggleCrossCell not present in this blind single-dataset run |
| G8 | PASS | simple_combined median aligned rho=0.7098 |
| G9 | PASS | max learned-disagreement=0.1594 |
| G10 | FAIL | tables/DATASET_TASK_SUMMARY.csv n_genes=5000 |
| G11 | PASS | zip created after checklist and scripts copied |
