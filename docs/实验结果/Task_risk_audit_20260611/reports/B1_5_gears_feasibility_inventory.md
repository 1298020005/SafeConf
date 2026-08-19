# B1.5 GEARS feasibility inventory

This step is inventory only. No GEARS training or inference was run.

## Current GEARS artifacts

- GEARS CSV artifacts found: 40
- GEARS prediction-record rows found across matching CSVs, not deduplicated: 374
- Unique `(dataset_name, record_id)` GEARS prediction rows across artifacts: 116
- Largest single GEARS prediction-record artifact: 62 rows
- Datasets in prediction records: adamson, dixit, frangieh, norman
- Trained checkpoint candidates found: 0
- GEARS data/split resources found: 17

## Recommendation

Do not train yet. First run a registered deduplication/audit of existing GEARS prediction records, because the inventory suggests 100+ unique GEARS rows may already exist across artifacts.

## Interpretation boundary

- Existing Frangieh GEARS records are useful as a preliminary probe, not a strong deep-model validation.
- `gears_prediction_magnitude_risk` is a magnitude diagnostic for GEARS outputs; it should not be described as SafeConf learned risk.
- Data resources such as `cell_graphs.pkl` and split files are not trained checkpoints.
- If a compatible checkpoint and target split are confirmed, the next GEARS step should be a separate registered run.
