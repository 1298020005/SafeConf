# E168 metadata freeze report

- Source shape from allowlisted metadata: 278,684 pseudobulk rows × 18,129 genes.
- Decoded expression values: **0**; decoded forbidden-column values: **0**.
- Guide types observed: non-targeting, targeting.
- Eligible targets in the label-free assay-available universe before hash selection: 5,510; selected: 200; column-unseen: 40.
- The universe requires at least two design-matched guides with identity rows in all 12 donor-state combinations. This is availability selection and is disclosed; test targeting `n_cells`, expression, DE, guide efficacy and `keep_*` values were not used.
- Final test manifest: 600 tasks in 3 state-specific ranking batches; test donor `CE0010866`. The three batches share one donor and the same target genes; they are not independent biological replicates.
- Donor roles: CE0010866=test, CE0006864=train, CE0008162=train, CE0008678=validation.
- Relevant row-access manifest: 15,818 HDF5 rows. Any row absent from this manifest is default-deny.
- Local source-byte state during this freeze: `partial_download_unparsed` (33,520,546,743 bytes). The freeze script did not open the local H5AD. Byte download/hash do not decode X; test targeting X remains sealed until the pretruth snapshot is committed to both remotes.
