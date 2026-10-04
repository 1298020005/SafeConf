# PerturbMap bounded prototype decision

The K562-to-Orion route was fitted only on Orion TRAIN target-background anchors. Validation predictions were loaded after the mapping fit and validation errors were used only by the scorer. Permanent TEST truth was not read. Six reference rules are reported separately for reconstruction and risk ranking.

## Current decision

The existing same-background PublicRule remains the default. This prototype is a background-transfer development result, not an automatic promotion.

## Risk readout on covered validation tasks

- `Amplitude`: n=53, U20=-0.2161, AURC=0.03992, Spearman=-0.2705
- `ProjectedCopy`: n=53, U20=0.5432, AURC=0.03400, Spearman=0.1355
- `RawCopy`: n=53, U20=0.6521, AURC=0.03260, Spearman=0.2571
- `RecipientMean`: n=53, U20=0.2915, AURC=0.03377, Spearman=0.2808
- `RidgeTransport`: n=53, U20=0.4175, AURC=0.03358, Spearman=0.2563
- `ScalarAffine`: n=53, U20=0.7084, AURC=0.03388, Spearman=0.2092
- `ShuffledPairTransport`: n=53, U20=0.4096, AURC=0.03469, Spearman=0.2008

The pooled covered readout has 53 tasks (30 HCT116 and 23 HEK293T). RawCopy is stronger than Amplitude on this small covered set, while the learned transport variants are context dependent and do not pass the current default-replacement gate. The next valid use is as a conditional background-transfer candidate after an independent route or larger held-out task set.
