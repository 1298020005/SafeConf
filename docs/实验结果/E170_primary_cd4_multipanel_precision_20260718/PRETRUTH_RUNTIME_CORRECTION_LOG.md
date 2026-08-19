# E170 pretruth runtime correction log

## 2026-07-18｜code freeze `0db0f709a635352744b3c502735fb9e345fa9016`

The first parallel P01/P02 F2 build attempts both failed closed and produced no immutable `F2_pretruth` directory.

### P01

- Failure: `IntegrityError: Train-only NTC coexpression graph is missing target genes`.
- Location: train-NTC coexpression construction, before any P01 targeting rows were consumed.
- Rows reached: registered NTC control rows only. Test targeting X and column-unseen X were not read.
- Cause: after constant-gene correlations were converted from NaN to zero, NumPy's tied top-k indices could omit the target's own index. The implementation comment and preregistered rule both required a self edge, but the candidate list did not force it.
- Mechanical correction: append the target's own index when the tied top-k list omits it. No threshold, target, model, endpoint, or truth-dependent choice changed. A constant-profile regression test was added.

### P02

- Failure: `KeyError: 'checksum_crc64nvme_base64'` while writing the access attestation.
- Location: after registered pretruth controls/train/validation rows had been processed, before the staging directory was promoted.
- Rows permitted by the builder at this stage: NTC controls, seen-target train rows, and seen-target validation rows. Test targeting X and column-unseen X remained excluded by the frozen phase mask.
- Cause: the E170 source lock stores `official_crc64nvme_base64`, while a later E168 helper line still expected the compatibility name `checksum_crc64nvme_base64`.
- Mechanical correction: translate the immutable E170 source-lock keys once in the in-memory frozen state so all downstream checks receive the same checksum value. Source bytes and hashes are unchanged.

Both staging directories were automatically removed by the fail-closed exception handlers. The corrections must be committed and pushed to GitHub and Gitee before any retry. P01–P04 selection, the 160/40 splits, six model seeds, SafeConf formula, gates, comparators, and joint statistical plan remain unchanged.

## 2026-07-18｜pretruth launcher environment

The first P01/P02 model-launch commands used the server's base Python 3.12 and both stopped at graph construction with `ModuleNotFoundError: No module named 'torch_geometric'`. No model epoch ran and no release/staging output was created. The model runner only opened the already isolated F2 bundles; it has no path to F3 or source test truth.

The rerun uses the same validated environment recorded by E168: `/home/yyf/.conda/envs/scgpt_env/bin/python` (Python 3.9.25, PyTorch 2.1.2+cu118, torch-geometric 2.6.1). This is an execution-environment correction, not a model or analysis change.

## 2026-07-18｜P01/P02 release-directory promotion

P01 and P02 completed all six model runs and wrote their gate snapshots, then both stopped at the final `os.replace` because the shared parent directory `pretruth_release/` did not yet exist. Before any promotion, every file named by each snapshot was rehashed, the file allowlist was checked, and all G2/G3/G4/synthetic certificate row counts were verified. Both snapshots were already formal FAIL results; no byte was edited and no model was rerun. The missing parent directory was created and the two complete staging directories were atomically renamed to `pretruth_release/P01` and `pretruth_release/P02`. P03/P04 then used the existing parent and completed the same code path normally.
