# Stage completion checklist v2.1

## Metrics used for gates
```json
{
  "test_records": 154,
  "leak_counts": {
    "pair_leak": 0,
    "pert_missing": 0,
    "context_missing": 0
  },
  "kcc_simple_combined_aligned_rho": 0.1208231819745679,
  "kcc_gate_applicable": true,
  "simple_combined_median_aligned_rho": 0.4323276495117311,
  "max_learned_minus_disagreement": 0.0306147326669629,
  "n_genes_ok": true
}
```

| Gate | Status | Evidence |
|---|---|---|
| G1 | PASS | tables/CONFIDENCE_EVAL_SUMMARY.csv level=dataset |
| G2 | PASS | test_records=154 |
| G3 | PASS | leak_counts={'pair_leak': 0, 'pert_missing': 0, 'context_missing': 0} |
| G4 | PASS | MVP_V2_1_REPORT.md starts with per-dataset table |
| G5 | PASS | historical_residual_risk in CONFIDENCE_EVAL_SUMMARY.csv |
| G6 | PASS | tables/LEARNED_RISK_FOLD_STATUS.csv |
| G7 | FAIL | KCC simple_combined aligned rho=0.1208 |
| G8 | PASS | simple_combined median aligned rho=0.4323 |
| G9 | FAIL | max learned-disagreement=0.0306 |
| G10 | PASS | tables/DATASET_TASK_SUMMARY.csv n_genes=5000 |
| G11 | PASS | zip created after checklist and scripts/run_confidence_mvp_v2_1.sh copied |
