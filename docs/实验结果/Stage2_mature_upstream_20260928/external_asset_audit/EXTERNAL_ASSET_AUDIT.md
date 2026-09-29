# SafeConf external asset audit

This audit is metadata-only. No h5ad expression matrix, prediction error, or test truth was opened.

## Independent decision

**Select McFalineFigueroa23 as the sole same-modality external candidate, conditional on metadata verification and the preregistered upstream competence gate.** It is the only current genetic candidate that is both untouched by SafeConf method design and large enough on official metadata. OP3 remains a chemical enhancement and will not replace the main genetic confirmation.

The two upstream slots are provisionally assigned to **PerturBench LatentAdditive** and **PerturBench DecoderOnly**. This selection was made from official configuration maturity, distinct architecture, reproducibility, and cost before any SafeConf result exists.

## Version and integrity findings

- PerturBench commit: `c84038bc1ea409aa54f3832cfa6f34f5059adf0c` (2026-08-10T16:01:41+01:00).
- Current Hugging Face file tree reports McFaline data size `5549946093` bytes and LFS SHA-256 `34db710ad850b5b5fd478d123f1c10ae3d5aaa013fce6061be8216ea27aed215`.
- Current split tar LFS SHA-256 `4d5e674afeedbf0395c6b6a5c68b5b2f9d49fb031edc4a2949123d984cd5393f` matches the downloaded tar.
- The current Croissant SHA fields are stale for both McFaline artifacts; file-tree LFS OIDs and downloaded bytes are the integrity authority.
- Full split has `878229` rows / `878229` unique cell IDs: train `738901`, val `70300`, test `69028`.
- Metadata-only inspection of the downloaded H5AD found `878229` cells and `15009` genes. The registered split contains 528/377/380 perturbation clusters and 6631/542/543 treated tasks in train/validation/test, respectively. These observed counts supersede inconsistent prose metadata for this exact file version.
- All 543 registered test tasks have eligible same-perturbation history in train before any test expression or truth is read; median train-history support is 998 cells across 13 biological states. This establishes history feasibility only and does not reveal confirmation outcomes.
- The checked-in base `mcfaline23.yaml` incorrectly overrides the split with `jiang24_split.csv`; official experiment YAMLs override it back to the correct McFaline split. Formal runs must provide the explicit split path and record its SHA.
- Current data download: `5549946093/5549946093` bytes (`complete`).
- Current Croissant license field: `https://creativecommons.org/licenses/by-nc/4.0/deed.en`.

## Candidate disposition

| Candidate | Modality | SafeConf-seen | Formal role | Decision |
|---|---|---:|---|---|
| McFalineFigueroa23 | genetic | False | main confirmation | SELECT_PENDING_METADATA_AND_UPSTREAM_GATE |
| Frangieh21 | genetic | True | secondary/ineligible | REJECT_SEEN |
| Norman19 | genetic | True | secondary/ineligible | REJECT_SEEN_AND_ONE_CONTEXT |
| Jiang24 | genetic | True | secondary/ineligible | REJECT_UPSTREAM_GATE_ALREADY_FAILED |
| OP3 | chemical | False | secondary/ineligible | KEEP_CHEMICAL_ENHANCEMENT_ONLY |


## Compute decision

- No official McFaline checkpoints were found locally, so validation competence requires training from official configs.
- Metadata and dataloader preflight do not consume an upstream attempt.
- The first formal validation prediction from each candidate consumes one of the two weekly slots.
- Test evaluation remains disabled until both the upstream competence decision and the frozen external SafeConf configuration are committed.
- If both candidates fail the upstream gate, no third upstream will be trained this week.
- The first dual preflight exposed a DataLoader process-memory failure before any GPU batch. The single registered engineering repair sets `num_workers=0` while preserving the official batch size and scientific configuration.
