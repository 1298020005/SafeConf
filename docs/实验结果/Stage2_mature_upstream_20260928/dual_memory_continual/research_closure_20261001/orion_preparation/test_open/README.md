# Guarded future Orion TEST truth reader

**Implementation ready; real TEST truth remains unopened by this agent.**
The standalone reader is
`tools/scripts/evaluate_safeconf_orion_guarded_test_agent.py`, SHA256
`852fcd764b567a63f04c4a5539a118c8a715f664bf534184018010ac269087b2`.
It uses a separate postseal TEST authorization path and never changes
`biology.LEGAL_ROLES`, relabels TEST as TRAIN, or changes SCI6b8, the existing
TRAIN/VALIDATION permit, loader, reader or model code.

`INTERFACE.json` specifies the exact future operation receipt. The root creates
that read-only receipt only after the actual final competence PASS, complete
LM fits/predictions and all-candidate Source risk/comparison seals exist.

## API and outputs

`authorize_test(receipt_path)` verifies every prerequisite before any raw object
checksum, footer or numeric page read. It refuses absent/failed/nonfinite final
competence evidence, changed or incomplete full LM parameter/hash seals, missing
Source risk/prior/config artifacts, unfrozen comparison scores or a stale evaluator implementation, changed endpoint
or registered object identities, and a role registry that does not preserve TEST
metadata SEEN / truth CLOSED. It binds the exact code implementations and the
same frozen scientific contract. The original structural-reader technical
amendment must be copied into `technical_dependency_overrides`; only that
previously amended dependency may differ from the original scientific hash.

`evaluate_test(receipt_path, output)` then reads exact immutable metadata TEST
masks, verifies complete raw object hashes, and reuses the fixed v2 byte-selective
reader. It retains one compressed numeric column at a time, spools allowed
tokens, and allocates endpoint-only sums. Its default endpoint sum limit is
512 MiB; it creates no full38606-by-task aggregate. It uses the unchanged official
full-library total for per-cell CP4000/log1p and subtracts frozen own-context
TRAIN NTC endpoint means.

The operation must explicitly require finite/nonnegative integer UMI within the
original 1e-6 tolerance, known unique tokens, and **exact** raw full-library sum
equality to the official total. Any violation aborts the entire output; it cannot
change the denominator, choose surviving cells, adjust thresholds or select cells
using risk/errors. Per-file deadlines and frozen memory/admission budgets apply.

All exact TEST rows are processed before applying the fixed task count >=30.
Counts/QC and missing-task coverage are computed only after legal opening. Tasks
below30 are listed in `TEST_TASK_QC.tsv` and are not silently substituted. Eligible
task errors compare their Source3285 mean effects to already frozen LM DELTA
predictions. No fitting, target CDF, comparator selection or partition change
occurs.

Committed outputs are read-only:

- `TEST_TASK_ERRORS.parquet`: exact fields `query_id,target_gene_id,target_gene_symbol,context_id,role,n_cells,true_error_rmse`.
- `TRUTH_READER_RECEIPT.json`: status COMPLETE, exact operation/code/scientific/
  object/endpoint/metadata/competence/risk/comparison/LM parameter and prediction
  bindings, generated task-error hash, semantic rule and full iterator proofs.
- `TEST_TASK_QC.tsv` and `TEST_ENDPOINT_EFFECTS.npy`: post-open task coverage and
  endpoint effects, with no full-axis matrix.
- `ARTIFACT_HASHES.json`: final output hashes.

Failures retain an explicitly incomplete directory and
`ABORTED_TEST_OPEN.json`; no complete error artifact is committed. That ledger
distinguishes materialized token rows from expression rows and makes no whole-file
completion claim. Private/unselected feature-count/page distributions are not
written into successful traces.

## Synthetic verification

`SYNTHETIC_PROOF.json` records16 passing generated-fixture checks. The positive
case includes forbidden TRAIN/VALIDATION/DEV numeric sentinels, exact CP4000 means,
two different own TRAIN controls, and30-versus29 cell tasks. All118 TEST cells are
opened before task filtering, demonstrating that the30-cell threshold is applied
after legal opening. Negative cases cover absent authorization, wrong code,
missing/failed/malformed competence, changed full LM parameters, missing risk/prior
seals, unfrozen comparisons, changed semantic rules, already-open truth, and
unknown/fractional/inconsistent TEST measurements. Missing prerequisite cases
prove rejection before raw checksum and numeric reader calls. Exact2% competence
margin comparison preserves the original ratio rule without an epsilon.

The generated roots are restricted to this subtree's `synthetic` directory;
setting `synthetic_only=true` cannot authorize an actual Orion raw root.

```bash
python tools/scripts/evaluate_safeconf_orion_guarded_test_agent.py validate --receipt IMMUTABLE_POSTSEAL_TEST_ACCESS.json
python tools/scripts/evaluate_safeconf_orion_guarded_test_agent.py evaluate --receipt IMMUTABLE_POSTSEAL_TEST_ACCESS.json --output EXACT_NEW_RECEIPTED_OUTPUT
```

These commands are for the later root-issued legal operation. Implementation
testing has not executed them with real TEST data.
