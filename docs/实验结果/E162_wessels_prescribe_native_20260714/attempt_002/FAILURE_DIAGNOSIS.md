# E162 attempt_002 validation-gate result

## Boundary and disposition

- All three native PRESCRIBE seeds (3407/3408/3409) completed five warmup epochs and four main epochs; each stopped under the frozen native `val/loss`, `min_delta=1e-3`, `patience=3` rule.
- All three best checkpoints were locked before any label-only forward. Strict state reconstruction gave exactly zero maximum difference for prediction, raw log probability, epistemic uncertainty and aleatoric uncertainty.
- The frozen main validation non-degeneracy gate failed. `TEST_LABEL_QUERY_EVENT.json` does not exist; no test label forward, test expression, test truth or test endpoint was accessed. E163 remains sealed.
- This exact fingerprint is terminal and must not be retrained or rescued by changing seed, checkpoint, jitter, clamp, threshold or score formula.

## Observed validation-only result

| seed | best val/loss | raw log-prob unique / 24 | raw log-prob SD | predicted PCA10 unique / 24 | any coordinate SD > 1e-6 | locked checkpoint SHA256 |
|---:|---:|---:|---:|---:|---|---|
| 3407 | 29.26584243774414 | 24 | 74899.2092177116 | 1 | no | `8ed092669f88f4c26e8f470ac33f0edd12941b25ecadf85d775cc9dea23a6460` |
| 3408 | 29.26584243774414 | 24 | 10671.25324525789 | 1 | no | `4ce4f59271890f25cde8257538299aa43164cd8a98a3a69e4173632b410da4cd` |
| 3409 | 29.26584243774414 | 24 | 32509.49682227867 | 1 | no | `2702bd4d836e5a6621ea49f6179b54e0c7c2e32fe536ff7e139556051d1a67fb` |

All eight fixed native-graph versus label-only forward comparisons had maximum absolute difference 0 for every audited field. The failure is therefore not an adapter mismatch: the flow `raw_log_prob` is condition-specific, but the posterior MAP prediction, official combined confidence and reconstructed predicted magnitude are constant across the 24 validation tasks.

## Scientific interpretation

This arm does not support using native PRESCRIBE as a non-degenerate Wessels perturbation predictor. It does show, on development data only, that the flow-density output retains task variation even when the decoder prediction collapses. Any raw-log-probability-only follow-up must be declared as a new validation-informed arm and frozen before the first test-label forward; it cannot be presented as a successful E162 confirmatory result.
