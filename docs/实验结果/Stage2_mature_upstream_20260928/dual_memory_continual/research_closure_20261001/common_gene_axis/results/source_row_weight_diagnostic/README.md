# Source training-row and absolute-weight diagnostic

This fixed diagnostic uses the released common 2,840-gene Source feature cache and 543 McFaline query tasks. It reads no Orion data, adds no upstream predictor, changes no frozen method, and selects no target winner. The information-free copies are algorithmic training-copy controls. They provide no additional independent labels.

## Execution and identity

The original seed is 20260930. Both Manual and Learned use the original 13 numerical features, training-only grouped error CDFs, and HGB parameters: 200 iterations, learning rate 0.05, depth 3, minimum leaf size 20, L2 10. All six original unique-source/pool cases reproduced archived scores before any new controls were fitted; maximum absolute score difference was 9.71445146547012e-17. The 12 fixed table fits and bootstrap completed in 12.53 seconds. Every saved model gives byte-identical predictions after reload.

Input and code hashes are in [REGISTRATION.json](REGISTRATION.json). Per-model hashes, original record identities, training-copy identities and Source gene cluster hashes are in [INFORMATION_WEIGHT_LEDGER.csv](INFORMATION_WEIGHT_LEDGER.csv). Models, query predictions and joint bootstrap draws remain under `/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/source_row_weight_diagnostic/`.

## Record and weight accounting

All scenarios have 575 Source gene clusters and 1,808 biological task IDs. A record here is an existing predictor/task pair, not an independent biological cluster.

| Scenario | Original unique predictor/task records | Training rows | Predictors | Sum of training weights |
|---|---:|---:|---:|---:|
| GAT unique | 1,808 | 1,808 | 1 | 1,808 |
| Exphormer unique | 1,808 | 1,808 | 1 | 1,808 |
| GAT identical copies ×2 | 1,808 | 3,616 | 1 | 3,616 |
| Exphormer identical copies ×2 | 1,808 | 3,616 | 1 | 3,616 |
| Full pool, original weights | 3,616 | 3,616 | 2 | 3,616 |
| Full pool, single-source weight sum | 3,616 | 3,616 | 2 | 1,808 |

`cluster_weights` normalizes to mean one. Thus doubling rows doubles the weighted data-loss term while keeping the configured L2 value fixed. The matched pool multiplies the original pool weights by 0.5. Repeated rows receive copied CDF labels; CDF fitting is restricted to original unique Source records. The original and duplicated preprocessors have exactly equal medians, centers and scales in all four cases.

Minimum observed leaf size was 20 training rows in every scenario. For identical copies ×2 this can be only 10 original predictor/task records. Absolute weight matching does not match row counts, leaf-size constraints or histogram construction. These controls do not identify each of those mechanisms separately.

## Equal-context macro U20

Evaluation uses all 543 released tasks, 380 gene clusters and three actual McFaline contexts. U20 is the original `research.metrics` definition; larger is better. Each bootstrap draw resamples complete gene blocks jointly across all contexts and all six scenarios. The bootstrap uses 5,000 draws, seed 20260930, equal-context macro averaging and linear percentile 95% intervals; every planned draw was valid. The compact U20 calculation is checked against the original metric for every point estimate.

| Scenario | Manual | Learned |
|---|---:|---:|
| GAT unique | 0.7034 | 0.6092 |
| Exphormer unique | 0.6784 | 0.5704 |
| GAT copies ×2 | 0.6096 | 0.5883 |
| Exphormer copies ×2 | 0.4630 | 0.6302 |
| Full pool, original weights | 0.6488 | 0.6953 |
| Full pool, matched weight sum | 0.5752 | 0.7097 |

All context metrics and other original metrics are retained in [STRATA.csv](STRATA.csv) and [MACRO.csv](MACRO.csv). All 14 fixed paired contrasts are in [PAIRED_GENE_U20.csv](PAIRED_GENE_U20.csv); positive differences favor the first scenario.

## Interpretation

Information-free duplication changes the learned risk function and can change U20 substantially. It does not produce a uniform improvement. Manual Exphormer copies reduce macro U20 by 0.2154 (95% interval −0.3086 to −0.0740). Learned GAT copies differ by −0.0209 (−0.0654 to 0.0458), and Learned Exphormer copies by 0.0598 (−0.0137 to 0.1681), relative to their unique originals.

The Learned full pool exceeds unique GAT by 0.0860 (0.0115 to 0.1575) and unique Exphormer by 0.1248 (0.0381 to 0.2127). Matching the pool's absolute weight sum gives a difference of 0.0145 (−0.0474 to 0.0540) relative to the original pool, with a macro U20 point estimate of 0.7097. Therefore the doubled weight sum does not account for the observed Learned pooled point advantage in this diagnostic. The Manual pool has no corresponding U20 advantage.

The result supports reporting the Learned pooling gain together with its row/weight controls and the existing equal-record control. It does not establish that every increase in Source rows adds useful information, and it does not isolate L2, leaf-size and histogram effects. This is an already-SEEN diagnostic with one fixed seed and descriptive paired intervals; it does not change Orion fitting, the registered method or any target selection.

## Run

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yyf/.venvs/safeconf-research-20261001/bin/python \
  tools/scripts/run_safeconf_source_row_weight_diagnostic_agent.py
```

Existing diagnostic output directories are refused, so this command cannot overwrite the completed result. [STATUS.json](STATUS.json) records completion, preserved original outputs and server artifact hashes.
