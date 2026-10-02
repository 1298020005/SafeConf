# Existing Orion validation risk-control preparation

**Cached inputs are technically available. No new numerical preparation or experiment was run.** A separate registered SEEN supplement is required before reusing competence labels for risk learning.

| Input | Cached availability |
|---|---|
| Eligible C validation | **2790 tasks /1661genes**, HCT1161348 andHEK293T1442. Every identity exactly joins the sealed query order and frozen eligibility. |
| P6 | Complete14716-query float64 Parquet; all2790eligible validation row identities are present. |
| Manual/Learned P+PUBLIC | Both exact13-column Parquets already cached, with full3285 prior arrays. No recomputation or Public fit is needed. |
| Public-supported validation | **224tasks /141genes**:111HCT116,113HEK293T. The other2566 have explicit no-history/public-risk-unsupported status. Cached row presence is not full Public support. |
| Shared scores | Six original Source risk score columns already cached. P-only scores cover allvalidation; Manual/Learned shared-history scores are unsupported outside those224. Root must lock the exact shared column and missing-support/cohort policy. |
| Endpoint/model identity | Both contexts have the exact ordered Source3285 Ensembl/symbol axis. Source7model/current registry hashes match; expanded bank remains8053a611… under0b6b extension. |
| Gene-disjoint evaluation | Validation has zero gene/symbol overlap with all frozen TEST identities and zero gene overlap with fixed232primary tasks/144genes. |

Minimal authorized preparation is a row-index projection/join of the already frozen P/PUBLIC tables, prechosen shared-score column and cached `VALIDATION_TASK_ERRORS.csv:model_rmse`. No raw biology, prior vector materialization, published-LM prediction call or Public refit is required. Feature rows align with `PRETRUTH_RISKS.tsv` /comparison identities, which exactly match concatenated HCT116 thenHEK293T `QUERY_IDENTITIES.tsv`.

The fixed proposed control families are Target-only(P6), Public+Target(P6+PUBLIC7), and Shared+Target(add frozen shared score), Ridge/HGB, budgets10/25/50/75/100 and budget-TRAIN-only CDF. This audit does not choose a reference, shared column, missing-data policy, seed or winner, and does not activate any fits.

Original6b8 scientific contract and `C_VALIDATION_ACCESS_LEDGER.json` restrict the actual validation receipt to competence and forbid risk/CDF fitting. Their bytes and truthful historical meaning must remain unchanged. A separate supplement must explicitly authorize feedback-label reuse, freeze the new budget/split/features/parameters/predictions, and label the analysis SEEN/retrospective. The completed original13 methods and232primary cohort remain fixed; this is not new confirmation.

Evidence:

- [Expanded seal](/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_public_bank_extension_20261002_v4_partitioned/expanded_risk_seal/RISK_SEAL_MANIFEST.json), [cached P6](/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_public_bank_extension_20261002_v4_partitioned/expanded_risk_seal/P_ONLY_FEATURES.parquet), [Manual features](/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_public_bank_extension_20261002_v4_partitioned/expanded_risk_seal/Manual_P_PUBLIC_FEATURES.parquet), [Learned features](/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_public_bank_extension_20261002_v4_partitioned/expanded_risk_seal/Learned_P_PUBLIC_FEATURES.parquet).
- [Frozen comparison scores](/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_public_bank_extension_20261002_v4_partitioned/expanded_comparison/PRETRUTH_COMPARISON_SCORES.parquet).
- [Existing competence error table](/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_postfit_followthrough_20261002_v4_decimal_tokens/competence/VALIDATION_TASK_ERRORS.csv) and [original C-use ledger](/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_postfit_followthrough_20261002_v4_decimal_tokens/competence/C_VALIDATION_ACCESS_LEDGER.json).
- [Exact readiness bindings and identity checks](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/existing_validation_risk_control_v1/PREPARATION_READINESS.json).

Read scope: identifiers/status strings, gene-axis metadata, Parquet schemas, two prior-array headers and small manifests. Zero numeric feature/score/error values, raw expressions, upstream calls, fits or bootstraps were used.
