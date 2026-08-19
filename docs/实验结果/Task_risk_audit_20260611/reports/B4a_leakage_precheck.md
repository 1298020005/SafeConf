# B4a leakage precheck

Source row-level score table: `/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_reliability_model_corrected_20260610/tables/RELIABILITY_ALL_SCORES.csv`
Code root: `/home/yyf/proj_task_risk_audit_20260611/code/20260426_154505_perturb_transport_final_push`

## Decision: PASS

No blocking leakage was detected by schema, row-level, and static-code checks.

## Checks

### PASS - B1 source table schema

- Evidence: Rows=210904, columns=14, missing=none
- Action: Stop B1 if required columns are missing.

### PASS - B1 score direction map

- Evidence: Observed score types={'oracle_magnitude_diagnostic': 'risk', 'predicted_magnitude': 'risk', 'protocol_v0_2_family_confidence': 'confidence', 'random': 'risk', 'safeconf_lodo_linear_risk': 'risk', 'safeconf_lodo_risk': 'risk', 'safeconf_perdataset_risk': 'risk'}; mismatches=none
- Action: Confidence scores are flipped to risk axis in B1; risk scores are not flipped.

### PASS - B1 target score availability

- Evidence: Available test scores=['oracle_magnitude_diagnostic', 'predicted_magnitude', 'protocol_v0_2_family_confidence', 'random', 'safeconf_lodo_linear_risk', 'safeconf_lodo_risk', 'safeconf_perdataset_risk']; missing=none
- Action: Missing scores will be marked missing_score in B1 rather than hand-filled.

### PASS - LODO output rows are held-out test rows

- Evidence: safeconf_lodo_risk rows=9168, non-test splits=none, heldout_dataset_matches_dataset=True
- Action: If this fails, do not interpret LODO risk until score construction is repaired.

### PASS - Oracle magnitude is diagnostic only

- Evidence: oracle rows=45850; true_effect_l2_norm column present=True
- Action: B1 may report oracle as a non-deployable ceiling, never as a SafeConf method.

### PASS - LODO code path excludes held-out dataset from training

- Evidence: Static check looked for dataset != held train selection, held test selection, and train-only target ranking.
- Action: If WARN, manually inspect run_safeconf_reliability_model.py before relying on B1.

### PASS - Feature normalization reference is train/val scoped

- Evidence: Static check looked for train/val reference splits and empirical CDF mapping through reference rows.
- Action: If WARN, manually inspect normalize.py before relying on B1.

### PASS - Forbidden label columns are registered

- Evidence: Forbidden entries found in schema include=['failure_label', 'normalized_rmse', 'score_value', 'true_effect', 'true_effect_abs_mean', 'true_effect_key', 'true_effect_l2_norm', 'true_error_cosine', 'true_error_rmse']
- Action: These columns may be evaluation labels/diagnostics, not model features.

### PASS - LOPO code path trains on existing predictors only

- Evidence: Static check looked for V0/ContextSim train source and third-predictor test target.
- Action: If WARN, manually inspect run_lopo_third_predictor.py before relying on LOPO claims.

## Boundary

This is a precheck, not a full formal proof. It is designed to decide whether B1 can run safely. A full reproducibility and leakage lock remains B4b.
