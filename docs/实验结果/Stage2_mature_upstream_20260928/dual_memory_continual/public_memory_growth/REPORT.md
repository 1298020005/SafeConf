# Public Memory growth

## Contract

- Evidence role: DEV/SEEN; McFaline test truth remained sealed.
- Five nested identity-hash reveal orders were fixed independently of outcomes.
- At every memory fraction and outer fold, both the biological transfer learner and Shared Risk Core were refit from the revealed memory only.
- All five orders are reported; no best order was selected.

## Main result

The result supports a **coverage-threshold** interpretation rather than a claim that every memory update monotonically improves risk. Biological reconstruction improves consistently as the bank grows, but risk gains are unstable at 10--25% memory, begin to turn positive near 50%, and become clearly positive in both architectures at full coverage. At 100% memory, mean U20 gains over prediction-only are `+0.029741` for Exphormer and `+0.029440` for GAT; Spearman gains are `+0.021188` and `+0.019981`, while AURC decreases (improves) by `0.000357` and `0.000335`.

The 100% run is a full numerical rerun from serialized artifacts. Its point estimates differ slightly from the original registered full-memory run because HGB tree thresholds react to sub-1e-7 round-trip differences in reconstructed prior features. Both runs preserve the same positive conclusion. The registered full-memory results remain the primary static comparison; this experiment is used for the growth pattern.

## Risk increment over prediction-only

| fraction | upstream | mean_delta_u20 | worst_delta_u20 | best_delta_u20 | mean_delta_spearman | worst_delta_spearman | mean_delta_aurc | worst_delta_aurc |
|---|---|---|---|---|---|---|---|---|
| 0.100000 | TxPert_Exphormer | 0.001001 | -0.001854 | 0.008271 | -0.004439 | -0.009553 | 0.000062 | 0.000114 |
| 0.100000 | TxPert_GAT | -0.008943 | -0.012957 | -0.003013 | -0.002996 | -0.006924 | 0.000044 | 0.000107 |
| 0.250000 | TxPert_Exphormer | -0.000202 | -0.011336 | 0.005397 | -0.003694 | -0.010816 | 0.000061 | 0.000188 |
| 0.250000 | TxPert_GAT | -0.006552 | -0.018748 | 0.003500 | -0.003330 | -0.009081 | 0.000063 | 0.000162 |
| 0.500000 | TxPert_Exphormer | 0.002744 | -0.005433 | 0.009930 | 0.001889 | -0.004758 | 0.000018 | 0.000246 |
| 0.500000 | TxPert_GAT | 0.003751 | -0.003771 | 0.009186 | 0.000976 | -0.005335 | 0.000035 | 0.000199 |
| 0.750000 | TxPert_Exphormer | 0.003066 | -0.013747 | 0.014446 | 0.005706 | 0.001429 | -0.000036 | 0.000086 |
| 0.750000 | TxPert_GAT | 0.009150 | -0.002445 | 0.027269 | 0.004835 | 0.000285 | -0.000045 | 0.000053 |
| 1.000000 | TxPert_Exphormer | 0.029741 | 0.029741 | 0.029741 | 0.021188 | 0.021188 | -0.000357 | -0.000357 |
| 1.000000 | TxPert_GAT | 0.029440 | 0.029440 | 0.029440 | 0.019981 | 0.019981 | -0.000335 | -0.000335 |

## Biological reconstruction and coverage

| fraction | mean_task_coverage | worst_task_coverage | mean_effect_rmse | worst_effect_rmse | mean_effect_cosine | worst_effect_cosine | mean_effective_sources |
|---|---|---|---|---|---|---|---|
| 0.100000 | 0.241261 | 0.237832 | 0.074699 | 0.076285 | 0.439538 | 0.427437 | 1.066521 |
| 0.250000 | 0.515819 | 0.511615 | 0.074003 | 0.074543 | 0.458786 | 0.451685 | 1.182534 |
| 0.500000 | 0.813496 | 0.799779 | 0.070342 | 0.070930 | 0.475619 | 0.472720 | 1.409370 |
| 0.750000 | 0.952544 | 0.946903 | 0.066048 | 0.066218 | 0.502377 | 0.499801 | 1.727655 |
| 1.000000 | 1.000000 | 1.000000 | 0.062113 | 0.062113 | 0.529424 | 0.529424 | 2.118432 |

## Decision

Public Memory growth is retained as a main systems result, with the explicit boundary that sparse memory can be neutral or harmful. The release/rollback gate is therefore necessary: new memory is stored, but a newly trained risk model is published only after anchor-task non-degradation checks pass. The evidence does not support promising improvement after every incremental update.
