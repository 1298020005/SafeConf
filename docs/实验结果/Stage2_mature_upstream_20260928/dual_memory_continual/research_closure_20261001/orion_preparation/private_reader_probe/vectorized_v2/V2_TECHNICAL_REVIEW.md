# Private numeric reader v2 technical repair

**PASS. The separate v2 reader is more than 20 times faster in paired synthetic
5,000-value workloads, while preserving selection before numeric conversion.**

Frozen implementations:

- Numeric backend:
  `636455cb30fc0e8c8ce6ac92db82756eff7508322e92862cf23ffad861d2e435`
- Integration wrapper:
  `6135469b10d5e200f4654a53f2a750fc9cc4c356037ce6abdfddd95030b45d1f`

The original reader `3e7c506f…`, original scientific loader `cdf261ff…`, original
20-case integration report and active old staging were preserved. Read-only code
snapshots of both new implementations are stored outside the repository under
the existing reader probe runtime's `implementation_snapshots` directory.

## Conversion boundary

Version1 performed a Python loop and scalar unpack for every numeric value.
Version2 calculates physical row routing from structural repetition/definition
levels in arrays, applies the immutable metadata row mask, and gathers only the
authorized eight-byte entries. Raw PLAIN payloads and raw dictionaries receive
**uint8 byte views only**. Advanced indexing creates an authorized byte copy;
INT64/DOUBLE views are applied only to that copy. Numeric dictionaries and whole
numeric pages are never typed before filtering.

One Python iteration remains per physical row segment. The generator preserves
original physical row indices, Python int/float lists, null lists, empty lists,
null items and list continuation across pages. Valid empty V2 data pages advance
no row and perform no typed conversion. Original schema, codec, page/entry/chunk
admission and fail-closed behavior remain. Unknown encodings get no fallback.

## Evidence

- All 12 original PLAIN/dictionary × V1/V2 × NONE/ZSTD/SNAPPY privacy fixtures
  match independent synthetic Arrow expected rows. Forbidden private numeric
  sentinels do not reach conversion, and every materialized row is authorized.
- All 20 existing independent normalization, role/hash binding, null/missing
  measurements, unknown tokens, ambiguity, private-sentinel and atomic staging
  checks pass with the v2 backend injected **in the synthetic test process only**.
- Twelve additional valid Parquet V1 files split authorized and private rows
  across data pages, including a null item, null list and empty list. Arrow reads
  these generated files to the expected full synthetic lists; v2 returns the
  exact authorized rows across every continuation. Both INT64 and DOUBLE, PLAIN
  and dictionary values, and NONE/ZSTD/SNAPPY codecs are covered.
- The integration wrapper's own synthetic proof gives exact equality for all six
  aggregate arrays between v1 and v2 and completes the 128-cell TRAIN/control
  pilot with private and unselected same-role sentinels excluded.

Paired timing uses the same generated SNAPPY file, 1,000 cells, 5,000 values per
cell per numeric column, 80% authorization and both INT64/DOUBLE passes. Both
versions convert exactly 8 million authorized values with sentinel guards:

| Numeric encoding | Version1 | Version2 | Speedup |
| --- | --- | --- | --- |
| Dictionary | 15.158 seconds | 0.738 seconds | 20.55 times |
| PLAIN | 13.881 seconds | 0.682 seconds | 20.37 times |

These measurements establish decoder speed on generated data. They exclude
production disk contention, token spooling, aggregate updates and final output
costs. No real numerical pages, private value counts, feature distributions or
Source quality profiles were read for this repair.

## Integration review

The separate wrapper preserves the original authorization and scientific
functions, and additionally requires an immutable permit binding its exact hash
and the actual v2 backend hash before any raw access. Dispatch changes function
objects only in that operating-system process and restores them afterward; it
does not modify original source files or another active process.

New outputs remain in an explicitly uncommitted versioned directory until the
actual wrapper/backend identity audit is attached and artifact hashes refreshed.
Committed manifests retain the original guard hashes with their roles explained
and separately identify the numeric parser that actually executed. Committed
traces omit private skipped-value/page counts, preserving opaque byte/budget
provenance and selected conversion counters only. The per-file signal deadline
covers hashing/reading and private trailing structural pages, including intervals
where the generator yields no authorized row.

No normalization, mapping, QC, scale, model, target truth or algorithm rule was
changed. Real execution requires a separate new frozen permit and operation
receipt for these exact code hashes and a new output/runtime version. This review
does not authorize reuse or overwrite of the earlier incomplete staging.
