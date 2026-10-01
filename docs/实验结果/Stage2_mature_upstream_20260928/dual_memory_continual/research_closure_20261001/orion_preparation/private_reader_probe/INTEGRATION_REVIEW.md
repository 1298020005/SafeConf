# Independent strict-reader integration review

**PASS under the frozen TRAIN/NTC library-sum policy. No real Orion expression or
token pages were opened during this review. A separate production permit remains
required.**

Reviewed implementation hashes:

- Loader: `cdf261ffcae854f49d469437492a12b7f835aae7b720bbe565044d8a89369e8b`
- Reader: `3e7c506f62d384073f2c20af383a371a2a23511c306d2ac046bd72c691f41312`

`INTEGRATION_REVIEW.json` records all 20 passing independent cases. The test runner
and generated fixtures are outside the repository under
`/home/yyf/data/safeconf_orion_frozen40_20261002/private_reader_probe_20261002_v1/`.
The runner captures the loader hash before fixture execution and requires the same
hash afterward, so these results identify the reviewed implementation exactly.

## Verified behavior

The expected normalization was calculated independently by reading **generated
synthetic** numeric columns with Arrow, normalizing each retained cell using its
official full-library total, treating sparse absent known genes as zero, and then
averaging across retained cells. The loader's full means, own-context TRAIN NTC
effects, cell counts and Source endpoint projection match these expected arrays.
Unequal library totals and sparse control rows distinguish this calculation from
normalizing a count sum or dividing each gene by its nonzero-cell count.

The negative cases cover unknown/duplicate token IDs, negative/nonfinite/fractional
UMI values, null lists/items, differing token/expression lengths, missing TRAIN
full-library counts, and empty measurement lists with positive metadata total.
All reject without committing a final output directory.

Physical metadata row permutations, row-metadata hash changes, reader code-hash
changes, requested roles exceeding the permit, a TEST numeric permission flag,
and raw size/hash identity changes reject before the numeric reader is
called. Authorization binds the immutable scientific contract, metadata roots,
role/context lists, exact loader/reader implementations and top-level metadata
artifacts; per-file row metadata and whole raw-file identity are checked before
numeric conversion. TEST/DEV sentinels are present in the generated fixtures and
never reach typed conversion.

A generated full axis with two entries sharing one target symbol preserves both
unique Ensembl/token entries for upstream biology. The ambiguous target task is
omitted using metadata before numeric conversion; its deliberately forbidden
numeric sentinel remains unopened. The fixed endpoint still requires exact unique
symbol/Ensembl/token identity. A separate real **gene-mapping metadata-only** audit
found 38,606 unique Ensembl and token IDs (contiguous 0..38,605) and 22 duplicate
symbol surplus rows, confirming why symbol-first mapping would be invalid.

An injected expression-iterator failure after two authorized rows leaves partial
arrays only in an `.incomplete.*` staging directory. The final output directory
and `BIOLOGY_MANIFEST.json` are absent. Both token and expression iterators must
exhaust exactly the permitted physical rows before successful directory rename.

The loader initializes memmaps in 32-row blocks, writes final means/effects in
32-row blocks, and flushes/drops aggregate mapping pages every 512 cells during
aggregation. It spools tokens before opening the expression iterator, retaining
one compressed numeric column chunk at a time. These memory changes were reviewed
in code; this review does not claim a multi-GiB stress-test result.

## Explicit policy limit

Raw UMI sum versus official full-library total is checked, and gap diagnostics
are recorded, **only for TRAIN and TRAIN_CONTROL_SOURCE_SCOPE**. VALIDATION receives
the generic type/token/list/nonempty/finite/nonnegative/integer checks and uses
the unchanged official denominator. It has no library-sum calculation,
distribution diagnostic or consistency gate. The independent policy test confirms
that a deliberately truncated synthetic VALIDATION list does not trigger the
TRAIN-only gate and that no VALIDATION gap record is emitted.

The earlier reader integration version committed fractional counts, incomplete
TRAIN counts and empty positive-total measurements. Those failures were fixed
and the complete independent suite was rerun on the exact final hash above.
VALIDATION sum rejection was removed from the expected checks to honor the
corrected frozen policy; this policy limit is preserved explicitly.

## Resource evidence and production conditions

`READER_ADMISSION_PROOF.json` verifies chunk admission before physical numeric
read, page/entry admission before decompression/structural decoding, and zero
numeric access for an all-false mask. Default ceilings are 1 GiB compressed
chunk, 128 MiB uncompressed page and 16 Mi structural entries. The loader also
enforces its frozen spool, group matrix, resident-memory and per-file time limits.
The frozen metadata's largest shard has 37,402 rows, below its 50,000-row limit.

Synthetic SNAPPY timing at 1,000 rows and 80% authorization gives:

| Fixed values per cell per column | Both-column time | Scenario over 1,104,270 metadata cells |
| --- | --- | --- |
| 2,000 PLAIN | 5.565 seconds | 1.707 hours |
| 2,000 dictionary | 6.167 seconds | 1.892 hours |
| 5,000 PLAIN | 13.906 seconds | 4.266 hours |
| 5,000 dictionary | 15.407 seconds | 4.726 hours |

These are reader-only synthetic workload scenarios. Actual numerical `num_values`
or private feature distributions were not inspected. Hashing, token spooling,
aggregation and output costs are additional. A 600-second file budget can stop a
large shard if those costs exceed the ceiling; the loader must leave incomplete
staging and report the failure, never alter normalization or privacy guarantees.

Before any real numeric operation, the final production permit must bind both
reviewed implementation hashes, the reader technical amendment, unchanged pinned
decoder dependencies, the immutable method/metadata hashes, exact authorized
contexts/roles and TRAIN/NTC-only library-sum scope. Any implementation change
invalidates this review's exact-hash result until reviewed again.
