# Strict private numeric Parquet reader feasibility

**A specialized reader is feasible. Ordinary PyArrow Scanner and fastparquet
`row_filter` do not supply the required guarantee. This probe does not grant a
production expression permit or open any real Orion expression/token columns.**

`tools/scripts/probe_safeconf_private_parquet_agent.py` passed 12 synthetic cases:
Parquet DataPage V1/V2, PLAIN/RLE_DICTIONARY, and NONE/ZSTD/SNAPPY. Each fixture has
600 rows in one mixed TRAIN/VALIDATION/TEST rowgroup. All 300 TRAIN rows match the
known synthetic expected lists, including null lists, empty lists and null items.
Each case skips 1,200 private numeric values across the two numeric columns and
converts 936 authorized values. Private numeric sentinels are rejected at the
only typed numeric conversion boundary if they ever reach it. See
`SYNTHETIC_PROOF.json` for fixture hashes, page counts and counters.

Three fail-closed probes also pass: an unsupported FLOAT list leaf and GZIP codec
are rejected before any numeric chunk read; a corrupted first PLAIN payload
length is rejected before any typed numeric conversion. Dictionary routing IDs
are checked against the dictionary byte length, and repeated private dictionary
references bypass the conversion boundary. Authorized dictionary lookups have
their own counter.

The negative proof wraps fastparquet's `core.read_plain` and raises before numeric
conversion if its input contains the synthetic private expression sentinel.
Calling `to_pandas(..., row_filter=mask)` triggers this guard for both PLAIN and
dictionary fixtures. Full-page PLAIN conversion and dictionary conversion occur
before row selection. This shows why ordinary `row_filter` cannot be substituted.

The adapter reuses fastparquet 2026.9.0's footer/schema parser, page headers,
definition/repetition level readers and compression decoders. Numeric payloads
and dictionaries stay opaque byte/uint8 buffers. It parses structural list levels
and dictionary routing IDs, increments original physical row indices, and applies
the independently obtained metadata mask **before** unpacking any int64/double or
creating a row list. Private rows never get a result-vector entry. Dictionary
lookup converts only the bytes for a dictionary ID referenced by an authorized
row; the numeric dictionary is never converted wholesale. Unknown page types,
numeric encodings, schemas or codecs raise errors without decoder fallback.

This guarantee allows copying and decompressing opaque bytes that also encode
private values. It does not promise that private byte ranges are never read, that
private structural levels are never parsed, or that no decompressed byte buffer
contains private bytes. It promises that private numeric expression/token values
and private numeric row vectors are never materialized. An authorization requiring
zero private byte access would need physically separate publisher data.

## Real footer verification

`REAL_FOOTER_ONLY_SCHEMA.json` records a 7,096-byte footer-only read of the complete
`data/HCT116_Batch1.parquet`. No numeric column pages or dictionaries were read.
The footer confirms one rowgroup, optional LIST with optional INT64/DOUBLE leaves,
PLAIN/RLE/RLE_DICTIONARY encodings, and **SNAPPY** compression. RLE here includes
structural levels; numeric value RLE outside dictionary encoding is not supported.
The two compressed numeric chunks are about 223 MB and 103 MB. No `.statistics`
property was requested and no footer min/max numeric byte strings were converted.

The existing metadata cache contains all 40 immutable file identities; its mixed
roles confirm that the existing strict Scanner refusal must remain until an
authorized strict adapter is integrated. Footer checks cannot identify whether
actual data pages are V1 or V2; both are covered synthetically.

## Integration work that remains

1. Bind the exact file/LFS hash, metadata hash, physical row order, context, row
   role mask and separate frozen method contract/permit before any expression
   operation. Never derive the mask from expression values.
2. Restrict TRAIN and authorized context-local controls explicitly, preserving
   DEV/VALIDATION/TEST exclusion and all existing preparation-policy checks.
3. Consume `iter_selected_numeric_lists` into aggregation and training buffers.
   It yields `(original_row_index, list_or_None)` and retains only the current
   authorized row list. `selected_numeric_lists` is a collecting wrapper for the
   synthetic fixtures. Both buffer one full compressed column chunk, one opaque
   dictionary, one decompressed page and structural arrays. Concurrent token and
   expression iterators retain both compressed chunks (about 327 MB for the
   checked shard), plus dictionaries/pages/current rows. Set
   `trace['numeric_materialization_rows']=None` for counters without an audit list
   per value. The technical memory amendment now enforces default 1 GiB
   `max_compressed_chunk_bytes` before reads, 128 MiB
   `max_page_uncompressed_bytes` before decompression and 16 Mi
   `max_page_level_entries` before structural decoding. Callers may supply
   stricter positive integer limits in `trace`; these are separate admission
   limits rather than a combined process RSS guarantee. Both old and amended
   reader implementations are immutable snapshots and the final access permit
   must bind the audited amended hash.
4. Pin fastparquet/cramjam and review the conversion boundary/codec paths before
   enabling real expression reads. Keep unknown formats fail-closed.

Iterator outputs must be staged until both iterators exhaust successfully: a
malformed later page can raise after earlier authorized rows have been yielded.
The metadata cache exposes `original_row_index`, `original_row_group` and
`row_role`; the caller must prove exact `0..N-1` order and all group IDs zero.
Reject mismatched token/expression yielded indices or list lengths. Null/empty
lists and nullable items must have an explicit aggregation policy.

Supported numeric encodings are PLAIN, PLAIN_DICTIONARY and RLE_DICTIONARY with
PLAIN dictionaries; supported structural levels use RLE hybrid streams. Supported
compression is UNCOMPRESSED, ZSTD and SNAPPY. Numeric BYTE_STREAM_SPLIT,
DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, direct numeric RLE,
BIT_PACKED levels, unknown pages/codecs, multiple dictionary pages, multiple
rowgroups and other schemas are rejected without fallback.

No independent custodian contract, server mirror authorization, expression
permit, production training attempt or policy downgrade is created here.

## Reproduce

The isolated `/home/yyf/data/safeconf_orion_frozen40_20261002/private_reader_probe_20261002_v1/python_deps`
contains fastparquet 2026.9.0 and cramjam 2.13.0;
PyArrow 24.0.0 and NumPy are already installed. Code packages were downloaded,
but no real dataset content was downloaded by this probe. The 12 original privacy-proof fixture
files together are below 1 MB and live under that external runtime. Larger locally
generated performance fixtures also live there; they were not downloaded. Source
snapshots also live only in the external runtime. `DEPENDENCY_IDENTITY.json`
records installed versions and hashes of decoder sources/binaries and the full
runtime dependency manifest.

```bash
python tools/scripts/probe_safeconf_private_parquet_agent.py
```

## Primary source evidence

- [PyArrow Scanner documentation](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html)
  says residual predicates filter already loaded RecordBatches. Projection and
  statistics pushdown do not guarantee private numeric values remain unparsed.
- [Arrow 24.0.0 Parquet scan source](https://github.com/apache/arrow/blob/apache-arrow-24.0.0/cpp/src/arrow/dataset/file_parquet.cc#L576-L607)
  filters rowgroups, infers the projection, then obtains RecordBatches for projected
  columns. There is no metadata row mask passed into this Parquet batch reader.
- [fastparquet page reader source](https://github.com/dask/fastparquet/blob/main/fastparquet/core.py)
  reads numeric PLAIN values before applying `row_filter`, and typed numeric
  dictionaries are decoded as soon as dictionary pages are encountered. A local
  copy of that source is saved beside this report; the installed release itself
  was exercised by the negative proof.
- [Apache Arrow late materialization discussion](https://arrow.apache.org/blog/2022/12/26/querying-parquet-with-millisecond-latency/)
  describes the separate Rust Parquet reader. Its existence does not confer the
  same guarantee on PyArrow's C++ dataset Scanner. We did not install or verify a
  Rust reader in this probe; no Rust toolchain was present.
