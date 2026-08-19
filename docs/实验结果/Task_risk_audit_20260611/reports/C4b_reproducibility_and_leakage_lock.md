# C4b reproducibility and leakage lock

## Git state

- Branch: `exp/task-risk-audit-20260611`
- HEAD: `abdff201444a9ca42b88b7d3c9ec5373c94abe5f`

```text
## exp/task-risk-audit-20260611...origin/exp/task-risk-audit-20260611
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/RUN_STATUS_c1.json"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/RUN_STATUS_c2.json"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/RUN_STATUS_c3.json"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/RUN_STATUS_c4b.json"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/reports/C1_gears_existing_record_dedup_audit.md"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/reports/C2_task_risk_feature_explanation.md"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/reports/C3_scope_boundary_interpretation.md"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/reports/C4b_reproducibility_and_leakage_lock.md"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/tables/C1_gears_all_prediction_records_index.csv"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/tables/C1_gears_bad_prediction_retrieval.csv"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/tables/C1_gears_canonical_prediction_records.csv"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/tables/C1_gears_canonical_score_rows.csv"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/tables/C1_gears_duplicate_audit.csv"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/tables/C2_task_risk_feature_high_low_contrast.csv"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/tables/C2_task_risk_feature_quartiles.csv"
?? "docs/04-\345\256\236\351\252\214\347\273\223\346\236\234/Task_risk_audit_20260611/tables/C3_scope_boundary_table.csv"
```

## Recent commits

```text
abdff20 (HEAD -> exp/task-risk-audit-20260611, origin/exp/task-risk-audit-20260611) fix: clarify GEARS retrieval scope in followup audit
859bc35 exp: add overnight task-risk followup audits
3256819 docs: summarize B4a B1 and GEARS inventory outcomes
a156763 docs: add GEARS feasibility inventory for task-risk audit
3f587ae fix: keep GEARS empty inventory tables readable
8b889c0 fix: separate GEARS checkpoints from data resources
75ebbe7 exp: add B1 bad prediction retrieval audit
2f1eab7 fix: strengthen B1 retrieval interpretation report
5ed94eb docs: add B4a leakage precheck for task-risk audit
434a5d5 fix: reduce false warning in B4a leakage check
```

## Inputs kept on server

- Row-level reliability scores: `/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_reliability_model_corrected_20260610/tables/RELIABILITY_ALL_SCORES.csv`
- Corrected feature root: `/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609`

Row-level outputs are not copied into Git. Git stores compact CSV summaries and Markdown reports only.

## Report availability

- B4a_leakage_precheck.md: present
- B1_bad_prediction_retrieval_interpretation.md: present
- B1_5_gears_feasibility_inventory.md: present
- C1_gears_existing_record_dedup_audit.md: present
- C2_task_risk_feature_explanation.md: present
- C3_scope_boundary_interpretation.md: present
- D1_lara_exvivo_lodo_failure_diagnostic.md: present

## Leakage status

B4a passed before B1 was interpreted. C4b does not replace a full external reproducibility package, but it locks the local evidence chain for this branch.
