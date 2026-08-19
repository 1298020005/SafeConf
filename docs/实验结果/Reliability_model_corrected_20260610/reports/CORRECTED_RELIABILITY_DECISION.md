# Corrected reliability-model decision

Date: 2026-06-10

## Source of truth

This is the canonical compact export for the learned SafeConf reliability
experiments. It uses the corrected drop-blank seven-main inputs:

`outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/`

The older `safeconf_reliability_model_v2_20260609`,
`safeconf_lopo_third_predictor_20260609`, and
`safeconf_external_validation_20260609` outputs remain historical artifacts.
They must not be used as the final numeric source because they mixed older
`phase1_main` inputs with the corrected audit.

## Corrections applied

- Lara exvivo, Lara invivo, and Santinha use explicit drop-blank inputs.
- Reliability loading now validates the PredictionRecord contract with legacy
  compatibility enabled.
- Training error ranks are computed from train/validation rows only.
- The reproducer uses the project `scgpt_env` Python instead of system Python.
- Corrected runs write to new directories and do not overwrite 2026-06-09 runs.
- Shifrut is copied to a separate corrected input and 20 blank-perturbation
  records are removed; the original run is unchanged.
- External validation now fails closed if any requested input fails validation,
  and writes internal/external input-status tables.

## Run status

- LODO: 7 datasets, 45,850 records, 1,000 task-cluster bootstraps, status `ok`.
- LOPO PertMean: 7 datasets, 4,584 third-predictor test rows, status `ok`.
- LOPO ControlKNN: 7 datasets, 4,584 third-predictor test rows, status `ok`.
- External validation: 7 internal inputs, 4 external inputs, 206 external test
  rows, status `ok`.

## LODO result

The learned reliability model was trained without the held-out dataset. All
seven magnitude-controlled partial correlations are positive.

| dataset | n | aligned rho | partial rho | AURC reduction % | reduction CI |
|---|---:|---:|---:|---:|---:|
| CuiHacohen2023 | 2506 | 0.375 | 0.239 | 19.83 | [15.35, 23.75] |
| Frangieh | 1266 | 0.541 | 0.206 | 13.71 | [11.97, 15.43] |
| Lara exvivo | 646 | 0.452 | 0.488 | 34.57 | [28.81, 39.81] |
| Lara invivo | 750 | 0.421 | 0.407 | 24.68 | [12.16, 34.36] |
| McFarland | 2326 | 0.248 | 0.162 | 14.83 | [10.71, 18.74] |
| Santinha | 546 | 0.128 | 0.295 | 4.34 | [-2.72, 10.39] |
| Srivatsan sciplex3 | 1128 | 0.501 | 0.594 | 21.10 | [17.79, 24.40] |

Interpretation:

- Dataset-transfer signal remains positive in 7/7 datasets.
- McFarland is positive for the learned LODO layer, while it remains a failure
  of the frozen v0.2 rule. These are two different claims and must not be
  collapsed.
- Santinha has positive partial rho, but its AURC-reduction interval crosses
  zero. It remains weak/supportive evidence.

## LOPO result

The reliability model was trained on V0StrongBaseline and ContextSimBaseline,
then applied to an unseen third predictor.

| third predictor | positive datasets | mean partial rho | McFarland partial rho |
|---|---:|---:|---:|
| PertMeanPredictor | 7/7 | 0.630 | 0.360 |
| ControlKNNPredictor | 7/7 | 0.606 | 0.303 |

Interpretation:

- The signal is reproduced with two different third predictors.
- LOPO shares dataset/task-level features with the training predictors,
  including V0/ContextSim disagreement. It supports predictor transfer, but is
  not evidence of a reliability model that is independent of all existing
  predictor information.

## External validation

The model was trained on the corrected seven-main inputs and applied without
error-label refitting to four external studies.

| dataset | n | aligned rho | partial rho | AURC reduction % | reduction CI |
|---|---:|---:|---:|---:|---:|
| KaggleCrossPatient | 80 | 0.428 | 0.074 | 15.77 | [5.92, 25.96] |
| ShifrutMarson2018 | 80 | 0.323 | 0.246 | 9.52 | [-0.21, 17.63] |
| XieHon2017 | 30 | 0.384 | 0.558 | 15.41 | [-10.98, 30.39] |
| crossPatient | 16 | 0.037 | 0.223 | 3.19 | [-8.96, 12.43] |

Interpretation:

- Magnitude-controlled partial rho is positive in 4/4 external datasets.
- Only KaggleCrossPatient has an AURC-reduction interval that clearly excludes
  zero in this corrected run.
- Shifrut, XieHon, and crossPatient are supportive but statistically uncertain;
  they must not be described as individually conclusive.

## Final decision

The corrected evidence supports retaining the learned reliability model as an
additive SafeConf method layer:

- LODO is positive in 7/7 datasets.
- Two LOPO probes are positive in 7/7 datasets.
- External partial rho is positive in 4/4 datasets.

The evidence does not justify saying that every external validation is
statistically conclusive. Frozen v0.2 remains the interpretable preregistered
baseline; the learned model is a separate transfer/calibration layer.

## Compact files

- `tables/RELIABILITY_BASELINE_LADDER.csv`
- `tables/RELIABILITY_WITHIN_MAGNITUDE_STRATUM.csv`
- `tables/LOPO_PERTMEAN_RESULT.csv`
- `tables/LOPO_CONTROLKNN_RESULT.csv`
- `tables/EXTERNAL_VALIDATION_RESULT.csv`
- `tables/EXTERNAL_INPUT_STATUS.csv`
- `tables/SHIFRUT_DROP_BLANK_LOG.csv`

Full row-level outputs remain on the server under the corrected 2026-06-10
output directories and are intentionally excluded from Git.
