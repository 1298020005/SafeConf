# C1 GEARS existing-record dedup audit

No GEARS training or inference was run. This audit deduplicates existing GEARS prediction-record CSVs and evaluates only available per-row scores.

## Canonical source rule

- Frangieh uses `gears_frangieh_formal_eval_20260606`.
- Adamson/Dixit/Norman use `gears_confidence_eval_formal`.
- Smoke and run03 artifacts are indexed as duplicate/provenance evidence, not mixed into the canonical table.

## Counts

- All GEARS record rows indexed: 192
- Canonical GEARS records: 116
- Canonical records by dataset: {'adamson': 21, 'dixit': 3, 'frangieh': 62, 'norman': 30}
- Duplicate keys with non-identical RMSE across artifacts: 76
- Canonical score rows by score: {'gears_prediction_magnitude_risk': 62, 'gears_uncertainty_confidence': 62}
- Scoreable datasets for per-row retrieval: ['frangieh']
- Score rows by dataset and score: [{'dataset_name': 'frangieh', 'score_name': 'gears_prediction_magnitude_risk', 'n': 62}, {'dataset_name': 'frangieh', 'score_name': 'gears_uncertainty_confidence', 'n': 62}]

## Retrieval summary

- Macro top-10 enrichment, `gears_prediction_magnitude_risk`: 8.857
- Macro top-10 enrichment, `gears_uncertainty_confidence`: 0.000

Important: the retrieval summary currently reflects scoreable Frangieh rows only. Adamson/Dixit/Norman are present in the canonical record table, but per-row magnitude scores were not recovered from available arrays in this audit.

Additional boundary: the score with strong retrieval is `gears_prediction_magnitude_risk`, which is a prediction-magnitude diagnostic for GEARS outputs. It is not a frozen v0.2, LODO, or per-dataset SafeConf score applied to GEARS predictions. `gears_uncertainty_confidence` does not currently retrieve bad predictions.

## Interpretation

Existing GEARS records are sufficient for a small registered dedup/provenance audit and for a Frangieh-only magnitude diagnostic. They are not yet evidence that SafeConf scores GEARS prediction risk, because no frozen v0.2 / LODO / per-dataset SafeConf score has been evaluated on GEARS predictions.
