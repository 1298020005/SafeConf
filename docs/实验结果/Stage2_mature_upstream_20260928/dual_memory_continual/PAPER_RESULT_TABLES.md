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

The stricter comparison uses no target-predictor error labels and excludes
source rows from the queried biological fold.

| risk-label source | unseen target predictor | target magnitude | transferred repaired-public HGB | delta U20 | 95% cluster CI |
|---|---|---:|---:|---:|---:|
| Exphormer | GAT | 0.768216 | 0.794104 | +0.027936 | [-0.008901, 0.072414] |
| GAT | Exphormer | 0.758726 | 0.795157 | +0.037366 | [0.002665, 0.082318] |

All predictions and bootstrap draws are in
`cross_predictor_zero_label_transfer/`.

## Table 4 — model feedback

The complete paired budget table is in
`error_adaptation_repaired/FEEDBACK_BOOTSTRAP.csv` and the feedback curve is
in `error_adaptation_repaired/FEEDBACK_CURVE_AUC.csv`.

The table distinguishes the official PertEMA audit from the
`PertEMA_style_HGB_P_proxy`; the proxy is not presented as the official
implementation.


Current repaired Shared Core feedback AUC:

| upstream | Shared Core U20 AUC | Residual HGB U20 AUC | decision |
|---|---:|---:|---|
| Exphormer | 0.803162 | 0.790206 | no stable U20 gain |
| GAT | 0.797582 | 0.792046 | no stable U20 gain |

This is retained as a boundary result. The adapter is not promoted to the
main performance claim.

## Table 5 — Public Memory growth

Five nested reveal orders were fixed by memory identity hashes. The table
reports the mean increment over prediction-only across all five orders.

| memory revealed | task coverage | Exphormer ΔU20 | GAT ΔU20 | biological-effect RMSE |
|---:|---:|---:|---:|---:|
| 10% | 0.2413 | +0.0010 | -0.0089 | 0.074699 |
| 25% | 0.5158 | -0.0002 | -0.0066 | 0.074003 |
| 50% | 0.8135 | +0.0027 | +0.0038 | 0.070342 |
| 75% | 0.9525 | +0.0031 | +0.0092 | 0.066048 |
| 100% | 1.0000 | +0.0297 | +0.0294 | 0.062113 |

This is a coverage-threshold result. It supports memory accumulation together
with release/rollback checks; it does not support a claim that each update is
monotonically beneficial.

## Figure 1 — public memory and feedback boundary

`figures/DUAL_MEMORY_PUBLIC_ERROR_RESULTS.svg` contains the publication draft:
the left panel is the repaired Public Memory comparison and the right panel is
the model-error feedback curve. The right panel is deliberately shown as a
boundary result rather than a claimed continual-learning gain.

## Figure 2 — Public Memory growth

`figures/PUBLIC_MEMORY_GROWTH.svg` shows the U20 increment, biological-effect
reconstruction and eligible-history coverage across the five reveal fractions.
