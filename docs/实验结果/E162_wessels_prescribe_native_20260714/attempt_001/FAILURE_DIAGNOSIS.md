# E162 attempt_001 failure diagnosis

- Failure time: 2026-07-15 15:40:22 +08:00.
- Boundary: seed 3407 completed five warmup epochs, but no main epoch completed; no validation label-only query, test-label query, test expression, test truth, or test endpoint was accessed.
- Cause: upstream `NaturalPosteriorNetworkLightningModule.save_hyperparameters()` populated both Lightning `hparams` and `_hparams_initial`. The E162 cleaner removed `model`/`adata` only from `hparams`; TensorBoard therefore attempted to serialize the full 32.8-million-parameter model from `hparams_initial` into `hparams.yaml`.
- Evidence: accidental YAML size 139,539,689 bytes; SHA256 `9cb38b8f18c5a97c333176cc8447d19aa2b6f68bb2b91958cb8f8610095ed46b`.
- Correction: purge `model` and `adata` from both Lightning hyperparameter stores before Trainer/logger construction, and regression-test that the resulting YAML is below 100 KB. The corrected smoke test produced a 289-byte YAML.
- Reuse policy: attempt_001 is terminal and is not resumed. Corrected code/contract bytes require a new Git commit and append-only attempt_002.
