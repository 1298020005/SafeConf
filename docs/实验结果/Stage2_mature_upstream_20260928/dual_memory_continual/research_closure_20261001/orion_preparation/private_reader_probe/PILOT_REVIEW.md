# Separate TRAIN semantic pilot review

**PASS for the exact scope below. This review opened generated synthetic numeric
data only and does not create a production operation receipt.**

- Pilot script SHA256:
  `8d6dbe17f1a98097968e408deae209bb168de7e4292a73da7e1287122cb1cf19`
- Existing loader SHA256:
  `cdf261ffcae854f49d469437492a12b7f835aae7b720bbe565044d8a89369e8b`
- Fixed reader SHA256:
  `3e7c506f62d384073f2c20af383a371a2a23511c306d2ac046bd72c691f41312`
- Existing production access permit SHA256:
  `e3a897cb1e6a4f2aabdcd4fff33c121be6e8de57d89842e4aa8fa2cf375ad293`

The scope is the first 64 exact TRAIN rows and first 64 exact
TRAIN_CONTROL_SOURCE_SCOPE rows in immutable metadata physical order from
`data/HCT116_Batch1.parquet`. The selection is fixed before raw access and cannot
adapt to expression quality. It excludes VALIDATION, TEST, DEV and every
unselected cell, including cells sharing an authorized role.

Code review confirms that existing production authorization and the additional
immutable pilot operation receipt are validated before any whole raw-file hash
read or numeric reader call. The operation receipt binds the pilot's exact script
hash, existing permit/scientific contract/loader/reader hashes, fixed source file
size/hash, row-metadata hash, exact selected row indices and mask hash. A full raw
SHA check then precedes numeric conversion. The pilot validates exact INT64 token
and DOUBLE expression schema, finite nonnegative integer UMI values, list identity
and full-library count consistency for the selected TRAIN/control cells.

Seven independent synthetic cases pass, recorded in `PILOT_REVIEW.json`:

1. Exact 64+64 selection, private and unselected same-role numeric sentinels never
   converted, both iterators exhausted, only TRAIN/control count-gap diagnostics,
   no vector persistence and no model or prediction calls.
2. Wrong pilot-script hash rejects before whole raw SHA or numeric reader calls.
3. A receipt requesting VALIDATION rejects before raw access.
4. Changing the per-role count rejects before raw access.
5. Wrong existing permit hash rejects before raw access.
6. A same-size corrupted raw file rejects on whole-file SHA before numeric reads.
7. An injected late numeric iterator failure writes `FAILED` with explicit partial
   proof and false exhaustion flags; it does not claim whole-file completion.

Success is reported only after `zip_longest` consumes both iterators to EOF and
the yielded rows equal the exact mask. This covers trailing unselected rows
structurally while their numeric values remain opaque. A failed operation reports
partial proof explicitly, including failures before numeric access.

An initial output issue persisted reader skipped-value counters for unselected
rows. The final reviewed script sanitizes both success and failure traces to
opaque byte/budget provenance and selected-row conversion counters. Private
skipped-value and page-count distributions are absent from saved pilot results.

The pilot persists a JSON diagnostic result only. It saves no numeric cell vectors,
normalization products, fitted models or predictions. It applies a 600-second
deadline, 6 GiB address-space ceiling and existing reader admission limits.

The previous reader implementation and 20-case integration report were preserved
unchanged. Real pilot execution must use the additional read-only operation
receipt binding the exact pilot hash above; this audit itself is synthetic only.
