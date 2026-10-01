# Authorized biology to blind R inputs

`tools/scripts/bridge_safeconf_orion_lm_inputs_agent.py` connects the guarded biology loader to `run_safeconf_orion_lm_blind.R`. It prepares inputs and receipts; the bridge does not invoke fitting. This preparation has opened no real expression and started no actual upstream attempt.

The bridge validates BIOLOGY_MANIFEST.json, scientific-contract and permit hashes, current loader/reader implementations, every biology artifact hash, exact full38,606/endpoint3,285 axes, unique Ensembl targets, per-role matrix-row masks, measured versus metadata cell counts and own-context TRAIN NTC row identities. It selects uniquely mapped TRAIN targets with at least 30 cells, followed by the context's control row. It indexes only these FULL_MEANS.npy rows. VALIDATION means are unnecessary for preparing fit inputs.

Python writes one condition's 38,606 float64 values at a time from the read-only NPY mapping; it releases resident mapping pages every 32 rows. The output byte budget is checked before numeric loading. No whole TRAIN submatrix or full cell-by-gene matrix is allocated in Python. The R runtime still needs the complete gene-by-condition double matrix and the published PCA/ridge workspaces; actual throughput and peak R memory remain unmeasured.

## Exact binary format

`TRAIN_X.f64le` is little-endian IEEE float64 in condition-major, gene-minor order. Python's condition-by-gene C-order bytes are already R's gene-by-condition column-order bytes; no numeric transpose, truncation or rounding occurs. `.f64le.meta.tsv` binds schema, dtype, byte order, layout, n_genes, n_conditions, expected byte count, binary SHA256, and both sidecar-axis file paths and hashes. The R receipt separately binds the sidecar SHA256. R checks shape, size, payload/axis hashes and axis uniqueness, then calls readBin and sets dimensions in place to avoid a second matrix allocation. The original `tools/safeconf_continual/orion_linear.R` bytes are unchanged.

The bridge saves a fixed full gene-axis TSV, endpoint-axis TSV, TRAIN conditions, own-context NTC mean at 17-digit precision, query identities, per-context input receipts, FIT_MANIFEST.tsv, BRIDGE_ACCESS_AUDIT.json, BRIDGE_MANIFEST.json and ARTIFACT_HASHES.json. Receipts bind every input plus the source biology, scientific contract, expression permit and source/bridge access audit.

## Query identities and future permit

`plan-queries` uses frozen GENE_SPLIT and the static unique-symbol-to-Ensembl mapping. It creates identities for **every mapped VALIDATION/TEST split target crossed with HCT116 and HEK293T**. It does not check TEST row presence, calculate TEST counts/distributions or build TEST numerical metadata features. Missing/ambiguous target symbols are omitted using the static full-axis mapping only. A separate VALIDATION_ELIGIBILITY.tsv uses permitted VALIDATION QC metadata with count≥30. TEST's original fixed count≥30 rule is applied only during eventual evaluation after prediction/parameter hash freezing and authorized test opening.

The immutable real metadata query scope is under `/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_lm_bridge_20261002_v1/query_scope_metadata_only/`; its source hashes and rule are in QUERY_SCOPE_MANIFEST.json. It contains identities, not truth or numerical TEST covariates.

Before actual aggregation, the final technical permit must bind `implementation_bindings.bridge` and `implementation_bindings.lm_cli` to their exact source paths/SHA256, plus `lm_query_scope.path/sha256`. BIOLOGY_MANIFEST.permit_sha256 must exactly match that permit. Changing the permit after aggregation cannot silently relabel the old biology artifact. The loader's immutable scientific dependency checks and specialized private numeric conversion guarantee remain required. The production integration approval and the registered future attempt ID are also required. No synthetic fixture grants production permission.

The future full pipeline v2 permit is now frozen as `orion_preparation/ORION_FULL_PIPELINE_ACCESS_PERMIT.json`, SHA256 `82a57a5c5ebd7937325d4b84c0659e291405f114869dd41540e29bef67eae25a`. The independent audit in V2_PERMIT_BINDING_AUDIT.json passed all24 file binding checks, including the unchanged scientific contract and the sole exact reader technical override. Future full aggregation and bridge receipts use v2. The already running bounded real TRAIN pilot retains its original v1 permit; this bridge preparation neither executes nor relabels that pilot. V2 does not authorize TEST numeric values or permanent test opening.

## Verified synthetic integration

Large arrays, models and predictions are runtime-only under `/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_lm_bridge_20261002_v1/synthetic_full_v1/`. The fixture reuses verified synthetic CP4000 cell means, with each of two cell profiles repeated15 to provide a 30-cell TRAIN metadata count. Its aggregate layout matches the biology loader; it does not claim another raw-loader expression execution. VALIDATION rows contain forbidden synthetic numeric sentinels; TEST has split identities but no cell rows.

Binary and RDS fits use the same canonical metadata and locked PCA10/ridge0.1/seed1. All 20 matrix/component/prediction comparisons passed with maximum difference0 and tolerance1e-12. The separate reload produced byte-identical treated and delta RDS predictions while TRAIN binaries and NTC files were unavailable. VALIDATION sentinels were excluded, and all TEST identity queries received predictions without any TEST counts. Two format guards reject wrong shape/byte counts and corrupted payload hashes before readBin loads numeric values.

`EXPORT_RDS_FIXTURE.R`, `BUILD_AND_VERIFY_SYNTHETIC_BRIDGE.py`, `VERIFY_BINARY_PARITY.R` and `VERIFY_BINARY_GUARDS.R` reproduce this proof in new runtime output directories. BRIDGE_STATUS.json records current code hashes and links the runtime reports. Matrix1.7.6 still differs from the paper's1.7.0 lock; this is a verified mathematical-core runtime, not a complete paper environment.
