# Dual-Memory SafeConf paper result tables (live)

This file is generated from committed DEV/SEEN artifacts and is updated after
the repaired error-adaptation run and McFaline confirmation. Values are not
selected after inspecting a sealed test outcome.

## Table 1 — public biology reconstruction

| method | RMSE | cosine | effective sources |
|---|---:|---:|---:|
| Uniform | 0.064002 | 0.527729 | 2.620 |
| Cell-weighted | 0.063678 | 0.525004 | 2.219 |
| Learned HGB | 0.062924 | 0.509135 | 1.667 |
| **Learned HGB + fixed support regularisation** | **0.062113** | **0.529424** | **2.118** |

## Table 2 — shared risk on TxPert

| upstream | HGB prediction-only | HGB manual public | HGB repaired public |
|---|---:|---:|---:|
| Exphormer | 0.770472 | 0.789953 | **0.803162** |
| GAT | 0.774242 | 0.788090 | **0.797582** |

## Table 3 — cross-architecture transfer

The corresponding cross-architecture output is in
`txpert_public_biology_repaired/CROSS_ARCHITECTURE_SUMMARY.csv`.

## Table 4 — model feedback

The complete paired budget table is in
`error_adaptation_repaired/FEEDBACK_BOOTSTRAP.csv` and the feedback curve is
in `error_adaptation_repaired/FEEDBACK_CURVE_AUC.csv`.

The table distinguishes the official PertEMA audit from the
`PertEMA_style_HGB_P_proxy`; the proxy is not presented as the official
implementation.
