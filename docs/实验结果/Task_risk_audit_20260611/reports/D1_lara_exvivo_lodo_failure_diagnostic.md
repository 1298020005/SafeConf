# D1 Lara_exvivo LODO failure diagnostic

This note records a correction raised during independent review: `LaraAstiasoHuntly2023_exvivo` is a clear failure case for the LODO transfer risk score in top-k bad-prediction retrieval.

## Main finding

`safeconf_lodo_risk` is below random at the most important tail levels:

```text
top 5%  enrichment = 0.593
top 10% enrichment = 0.612
top 20% enrichment = 1.797
```

This means the LODO transfer score does not successfully rank the worst Lara_exvivo predictions into the top-risk tail. It is not enough to say “LODO is above random in 6/7 datasets” without naming this failure.

## Why this is not a frozen-v0.2 failure

The same dataset behaves very differently for the frozen and per-dataset scores:

```text
frozen v0.2 top 10% enrichment        = 7.798
per-dataset risk top 10% enrichment   = 7.186
predicted magnitude top 10% enrichment = 0.459
LODO transfer top 10% enrichment       = 0.612
```

So the correct interpretation is:

> Lara_exvivo is a LODO transfer-layer failure, not a dataset where all SafeConf-style scoring fails.

The frozen v0.2 family score and the per-dataset risk layer both retrieve bad predictions strongly here, while the leave-one-dataset-out transfer layer and predicted magnitude do not.

## Practical implication

This failure should be reported as a boundary:

- Do not hide Lara_exvivo behind macro averages.
- Do not present LODO transfer as uniformly reliable at top-k retrieval.
- Separate three claims:
  - frozen v0.2 can be strong on Lara_exvivo;
  - per-dataset learning can also be strong on Lara_exvivo;
  - cross-dataset LODO transfer fails in the top-risk tail on Lara_exvivo.

## Next diagnostic

The next highest-value check is small and targeted:

1. Compare the `safeconf_lodo_risk` score distribution against `true_error_rmse` in Lara_exvivo.
2. Compare frozen v0.2 features against LODO-learned risk features for the same records.
3. Test whether the LODO failure is caused by magnitude dependence, exvivo-specific distribution shift, or task-tail ranking instability.

Until this is done, Lara_exvivo should stay in the boundary table as a named LODO failure case.
