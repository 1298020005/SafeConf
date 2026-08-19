# Stage completion checklist v2.1

## Metrics used for gates
```json
{
  "test_records": 308,
  "leak_counts": {
    "pair_leak": 0,
    "pert_missing": 0,
    "context_missing": 0
  },
  "kcc_simple_combined_aligned_rho": NaN,
  "kcc_gate_applicable": false,
  "simple_combined_median_aligned_rho": 0.6187467434960859,
  "max_learned_minus_disagreement": -0.4590612185611061,
  "n_genes_ok": true
}
```

| Gate | Status | Evidence |
|---|---|---|
| G1 | FAIL | tables/CONFIDENCE_EVAL_SUMMARY.csv level=dataset |
| G2 | PASS | test_records=308 |
| G3 | PASS | leak_counts={'pair_leak': 0, 'pert_missing': 0, 'context_missing': 0} |
| G4 | PASS | MVP_V2_1_REPORT.md starts with per-dataset table |
| G5 | PASS | historical_residual_risk in CONFIDENCE_EVAL_SUMMARY.csv |
| G6 | PASS | tables/LEARNED_RISK_FOLD_STATUS.csv |
| G7 | SKIPPED_NOT_APPLICABLE | KaggleCrossCell not present in this blind single-dataset run |
| G8 | PASS | simple_combined median aligned rho=0.6187 |
| G9 | FAIL | max learned-disagreement=-0.4591 |
| G10 | PASS | tables/DATASET_TASK_SUMMARY.csv n_genes=5000 |
| G11 | PASS | zip created after checklist and scripts/run_confidence_mvp_v2_1.sh copied |
