# Explicit pretruth Public-bank extension

The new adapter reuses the seven frozen Source models, their preprocessors/CDFs, original inference functions and 50:50 Learned weights. It makes **zero fits**. Retrieval uses a separately registered bank version and does not inherit the original bank's confirmation status. Original science `6b8a2693…` and original math/reader/evaluator/coordinator bytes remain unchanged.

The completed bank is `public_source_history_expanded_20261002_v2/PUBLIC_BANK_MANIFEST.json`, SHA `8053a6118bc2e3cace841e1498cee3c1366ce5b49e9c79328e7cd34a180b6607`. It uses expansion contract `0b6b9b86…`, read contract `d7f8f9d6…` and sorted-CSR builder `9114a69e…`. Its 5,365 records contain the original 2,008-record prefix plus 3,357 historical records for 1,490 targets absent from the old eligible580 set. The old575 risk-training target histories remain unchanged. Actual Source bank validation and independent bank review are stored separately under `historical_bank_v1` and `historical_bank_independent_actual_review_v1`.

## Production entry points

| File | SHA256 |
| --- | --- |
| `tools/safeconf_continual/orion_public_bank_extension.py` | `9857b96b98460346bce334313c1a212466c85dc450c07f16e5d934e1ef045454` |
| `tools/scripts/seal_safeconf_orion_public_bank_extension_agent.py` | `81ca3f44f147880b19fc9b8225ac8c63572e9a44dfc78e1d6d2208b3770b0414` |
| `tools/scripts/run_safeconf_orion_public_bank_extension_chain_agent.py` | `2df89381e10075ccad73c3072dba610a77159257f80c8adf5162a1bc63ef1dff` |

`load_bank` verifies exact registered contracts, seven model bindings, Source3285 axis, prefix metadata/dtype/vector bits, Source575, absent-only append eligibility, physical-unit uniqueness, fixed legacy-cache compatibility and completed Source-only access audit. It reads aggregate bank inputs; it does not reopen raw Source X. Asset provenance is bound through the immutable bank/read contract and the completed builder audit.

The sealer calls the unchanged original `source_core`, `load_inputs` and `infer`. It checks all registered VAL/TEST queries whose targets belong to old580: scores/statuses, numerical feature bits, priors, pair features, transfer predictions and weights. It also requires the primary intersection of all13 fixed candidates—including `NegativeSourceHistorySupport = -Manual.log_history_support`—to contain at least100 independent target-gene clusters **in the union** and at least20 tasks in each original context. Metadata QC identities are not numerical risk features. The final comparison repeats this check on the actual frozen Parquet scores.

Original2008 scores are retained in `ORIGINAL2008_FIXED_PRETRUTH_RISKS.tsv`, separate from the expanded primary. The existing original followthrough package is preserved as a separate fixed archive. Its unsupported Public-score NaNs are never added as extra comparators that would shrink the expanded intersection.

## Exact waiting and pretruth preparation

`PRETRUTH_QUEUE_RECIPE.json` gives fixed paths, pins, waiting statuses, registration schema and exact launch argument arrays. The static recipe describes only pretruth stages. After Root authorized the reviewed implementation, the separate operational controller `WAIT_PRETRUTH_EXTENSION.py` was copied into the isolated runtime and started as PID424742. Its initial status is `WAITING_ACTUAL_ORIGINAL_COMPETENCE_AND_SEAL`; `RUNTIME_QUEUE_START_RECEIPT.json` binds the controller, registration, PID/log/status and initial closed-TEST state. The controller waits for:

`/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_postfit_followthrough_20261002_v2/STATUS.json`

Only `VALIDATION_QUALIFIED_AND_RISK_PREDICTIONS_FROZEN_TEST_TRUTH_STILL_CLOSED` permits progression. Actual competence must separately be PASS. Failed/incomplete/nonqualified/schema-uncertain competence stops the queue with TEST closed. Polling may occur every30seconds for at most16hours. Materialize the actual ready `FROZEN_PRETEST_COMPARISON.json` SHA/size into the new immutable `SEAL_REGISTRATION.json`; no placeholder may enter production. Queue activation also requires the final adapter review PASS.

The two commands in the recipe produce `expanded_risk_seal` and `expanded_comparison` only. Stop there for actual seal/comparison review. No first-open claim or TEST reader is invoked by those stages. No existing Source, LM, biology, original seal or frozen method file is overwritten.

## Once-only TEST chain

The new outer operation schema is `safeconf_orion_extended_bank_postseal_test_operation_v1`. It carries every original operation field and a `public_bank_extension` object with:

- `schema`: `safeconf_orion_public_bank_extension_operation_v1`;
- exact `extension_contract`, `bank_manifest`, `comparison_receipt` bindings;
- the complete `implementation_bindings` map returned by `chain.implementation_bindings()`;
- immutable `root_and_independent_review` binding.

The final review schema is `safeconf_orion_public_bank_extension_final_review_v1`, status `APPROVED_FOR_ONCE_ONLY_REGISTERED_TEST`, with `root_approved=true`, `independent_review_passed=true`, exact bank/contract/comparison/implementation bindings and a hashed independent receipt whose status is PASS. Root creates this only after actual artifacts and the reviewed implementation match. Actual original competence/full LM/source/candidate/metadata/role/resource gates remain enforced by unchanged852.

`authorize --receipt OUTER_OPERATION.json` validates readiness using a temporary base-schema file deleted in `finally`. It leaves **no persistent delegation and no once-only claim**. The old CLI rejects the outer schema. `evaluate-test --receipt OUTER_OPERATION.json --output EXACT_REGISTERED_OUTPUT` checks the output before claiming, reserves the canonical immutable `ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json`, then creates a persistent delegation binding that claim and calls unchanged852. Failure never releases the claim. No automatic retry or second opening is permitted.

The truth output adds `EXTENSION_TRUTH_RECEIPT.json` alongside the original reader receipt. `evaluate-risk --comparison EXPANDED_COMPARISON --extension-truth-receipt EXTENSION_TRUTH_RECEIPT.json --output NEW_EVALUATION` checks the extension chain, review, canonical claim, exact delegation and original completion/semantic gates **before error-byte hashing**, then delegates statistics to unchangeddec. Its separate evaluation receipt carries the bank/extension identities. The original CLI alone does not validate an extension.

## Serialization diagnosis and tests

The generated first attempt exposed a technical parse mismatch: dec reads TSV with the default float parser, while852 uses `float_precision='round_trip'` and exact score equality. Differences were at most `5.551115123125783e-17`; the old guard correctly rejected before reading TEST. `FAILED_GENERATED_DEFAULT_PARSE_DIAGNOSIS.json` preserves the diagnosis. The new comparison adapter restores all12 existing numerical candidates from the sealed CSV using the unchanged guarded-reader parser, preserves fixed negative support, verifies exact Parquet reload and records divergence counts/magnitudes. Formulas, rounding/tolerances, method IDs and cohorts are unchanged. Original dec/852 files are untouched.

`GENERATED_FIXTURE_PROOF_FINAL.json` records15 cases, including nonfinite13th support rejection. `GENERATED_END_TO_END_PROOF_FINAL.json` records a generated full chain with Source3285, original2008 rows/580 targets/575 training targets,204 queries,13 candidates and6060 selected synthetic cells. It covers exact legacy invariance, all13 finite coverage, wrong-chain rejection, transient authorization, wrong-output rejection before claim, claim validation, generic COMPLETE rejection before error hashes, unchanged private-reader exclusions, the original evaluator and duplicate claim rejection. Seven generated deterministic predictors are never fitted. Only fixture-local synthetic science/root constants and four bootstrap draws are substituted; production5000 draws and all production code bytes remain fixed. Temporary numerical fixtures are removed. Earlier proofs are preserved as diagnostic history and are not the final launch pins.

**No actual Orion predictions were scored by this adapter and no actual TEST was opened during implementation or fixture verification.** Real competence, LM fit and original risk seal were still pending at queue start. The controller may run the authorized pretruth stages later and stops for Root review; it never invokes a TEST operation. It introduces no Public Biology Learner retraining, risk retraining, target CDF fitting, residual adaptation or posttruth selection.
