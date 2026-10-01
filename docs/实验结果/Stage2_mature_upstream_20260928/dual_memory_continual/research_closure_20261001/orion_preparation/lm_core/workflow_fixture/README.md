# Blind published LM workflow

`tools/scripts/run_safeconf_orion_lm_blind.R` fits and reloads the unchanged published numerical core. The official `solve_y_axb` source, both TRAIN processed-X PCA calls, PCA10, bilateral ridge0.1, seed1, TRAIN intercept and additive own-context NTC mean remain fixed. It executes no original expression loader or observed-versus-predicted report. Models and predictions are saved before any held-out truth is provided.

The executed example here is entirely synthetic: two contexts, 38,606 readout genes, 24 TRAIN conditions including ctrl, two cells per condition, four held-out query identities and a 3,285 gene endpoint. Query expression and query truth are not generated. No real Orion model attempt has started. The first numerical-core fixture remains in the parent directory; this workflow additionally tests the actual CLI, serialized models, output hashes and inference without TRAIN files.

## Input formats

Fit uses a TSV manifest with one row per context and these columns:

`context_id, train_x, baseline, conditions, queries, full_gene_axis, output_gene_axis, receipt`

Paths are to prepared inputs. Contexts run serially. `train_x` is a numeric genes-by-conditions RDS matrix or a TSV/TSV.gz with `gene_id` as its first column. `baseline` is a named numeric RDS vector or a TSV with `gene_id, mean_cell_logCP4000`. TRAIN values and the NTC mean must be **equal-cell means of log1p(4000 × cell UMI / official full-library total)**. The full denominator precedes gene projection. The wrapper checks the receipt, axis, roles, nonnegative finite values and exact equality of the ctrl column and NTC mean; raw-count aggregation belongs to the separately authorized reader.

`full_gene_axis` has `gene_id, gene_token_id, gene_symbol`, exactly 38,606 unique Ensembl IDs and unique tokens in fixed readout order. Duplicate symbols stay in the readout matrix. TRAIN and query target symbols must map uniquely to their declared Ensembl IDs. Missing or ambiguous target tasks are excluded in metadata before fitting; the wrapper rejects such tasks if supplied. It never chooses the first duplicate. The actual metadata ambiguity includes TRAIN MKKS/SPATA13 and VALIDATION ELFN2; these labels came from the parent metadata audit, not a new expression read by this workflow.

`output_gene_axis` has `gene_id, gene_symbol` and optionally `source_index`: exactly 3,285 unique symbols and Ensembl IDs in the frozen Source order. Projection occurs after the full gene space has supplied embeddings; targets outside the endpoint still have TRAIN-derived embeddings.

`conditions` has `condition_id, target_gene_symbol, context_id, role, n_cells`, in matrix-column order. Perturbed condition IDs are uniquely mapped Ensembl IDs and their role is TRAIN. Exactly one `ctrl` column has role TRAIN_CONTROL_SOURCE_SCOPE and the same context. Positive integer cell counts are metadata, not independent biological replicate counts.

`queries` contains exactly `query_id, target_gene_id, target_gene_symbol, context_id, role`. Roles are VALIDATION or TEST, and targets cannot appear among TRAIN condition IDs. This file contains identities only; an expression/truth column is rejected from its header. No query matrix is accepted by the API.

`receipt` is a key/value TSV. Required fields are:

- `schema=safeconf_orion_lm_blind_inputs_v1`, `dataset_kind=SYNTHETIC` or `AUTHORIZED_TRAIN`, matching `context_id`, and `estimand_id=mean_cell_log1p_cp4000_v1`.
- `allowed_expression_roles=TRAIN,TRAIN_CONTROL_SOURCE_SCOPE`, `contains_only_authorized_training_rows=TRUE`, `query_metadata_only=TRUE`.
- SHA256 fields `train_x_sha256, baseline_sha256, conditions_sha256, queries_sha256, full_gene_axis_sha256, output_gene_axis_sha256`, each checked against actual files before numeric loading.
- Actual TRAIN input additionally requires `upstream_attempt_id`, `private_expression_values_decoded=FALSE`, and `expression_permit_path/sha256`, `method_contract_path/sha256`, `access_audit_path/sha256`. The wrapper verifies all three artifact hashes. The authorized reader is responsible for the permit and private numeric access audit; the receipt does not grant new raw-expression access.

Prediction mode uses `context_id, model_bundle, model_sha256, queries, queries_sha256, output_gene_axis, output_gene_axis_sha256`. It verifies hashes and the saved provenance and locked defaults. TRAIN pseudobulk and NTC input files are unnecessary in this mode.

## Outputs and competence

Each fitted context saves MODEL.rds, full GENE_EMBEDDING.rds and PERTURBATION_EMBEDDING.rds, RIDGE_COEFFICIENTS.rds, the TRAIN intercept and NTC mean, input identities, treated/delta endpoint predictions in RDS and TSV.gz, and ARTIFACT_HASHES.tsv. RUN_MANIFEST.tsv records model and prediction hashes. Model reload must reproduce treated predictions exactly; this fixture also verifies byte-identical treated and delta RDS files from the separate prediction CLI.

SIMPLE_BASELINES.rds saves zero effect and the **unweighted mean of TRAIN perturbation-task effects**, excluding ctrl, separately for each context. BASELINE_TRAIN_GROUPED_OOF_ERRORS.tsv evaluates zero and leave-one-gene-out TRAIN task means on the endpoint; BASELINE_SELECTION.tsv chooses the lower TRAIN macro RMSE, with a deterministic zero-effect tie break. Each condition is one uniquely mapped gene, so leaving out its column leaves out the complete gene cluster. Validation and SafeConf errors never enter this selection.

The prospective rule is the parent `orion_preparation/ORION_METHOD_CONTRACT.json`, SHA256 `6b8a26939c8c6134dbf2e8294ee7905fccc2122b570b046f3183a3d2f5caba97`. On legally released VALIDATION effect vectors, use task-level RMSE over the frozen Source3,285 endpoint against the selected simple baseline. The fixed relative macro error margin is +2%; at least 60% of registered context strata must be noninferior. There are only HCT116 and HEK293T, so both must satisfy that fraction; these are not three biological states. Use 5,000 paired gene-cluster bootstraps, jointly resampling both contexts, seed20260929. A lower95CI of the paired context-macro relative gap above +0.02 is stable disadvantage. Eligible validation tasks require frozen QC count≥30, unique target mapping and finite predictions. This synthetic proof establishes no validation competence. Validation is competence-only; it does not fit prediction scaling or SafeConf calibration.

The runtime is a numerical-core reproduction: R4.4.1 and irlba2.3.5.1 match the recorded lock; Matrix1.7.6 differs from Matrix1.7.0. Full paper environment reproduction is false. Full Orion CPU time and peak memory remain unmeasured; contexts are processed serially and no GPU is used.

## Reproduce

Use `/home/yyf/.conda/envs/safeconf-orion-lm-20261002/bin/Rscript --vanilla tools/scripts/run_safeconf_orion_lm_blind.R --mode fit --manifest FIT_MANIFEST.tsv --output-dir NEW_OUTPUT_DIR` from the repository. The input manifests here refer only to synthetic assets. Output directories must be new or empty, so the CLI cannot silently replace frozen predictions. `GENERATE_AND_VERIFY_WORKFLOW.R` creates the synthetic inputs and runs both modes; its output root must have no existing `inputs` directory.
