# SafeConf Memory Alignment Contract v1

## Scope

This contract controls biological-effect comparisons used by the model-independent Public Memory. Model-specific realised errors never enter this bank.

## Required identity

Every vector records its study, context, perturbation, condition, control source, gene-space ID, normalization and effect-contract ID. Direct vector comparison is allowed only when effect definitions are compatible.

## Alignment rules

1. Missing genes are masked, never filled with zero.
2. Identical contracts use the registered gene order.
3. Cross-study comparison requires at least 2,000 common genes and at least 70% coverage of both registered axes.
4. Compatible cross-study effects use rank, cosine and direction comparisons; raw RMSE is not pooled across incompatible scales.
5. HVG selection, PCA, scaling and imputation are fitted inside the outer-train partition.
6. A current target task, its replicate group and target-derived preprocessing cannot enter its eligible history.

## Current contracts

- `E201_log1p_matched_batch_delta_v1`: source perturbation mean minus source-context batch-matched control on the registered 3,352-gene GEARS axis.
- McFaline contracts remain unregistered until train/validation-only alignment and guide/plate reproducibility audits finish. The sealed test partition cannot influence that registration.

## Fail-closed behavior

When alignment cannot be proven, the item remains auditable metadata but supplies no effect-vector similarity or content feature.
