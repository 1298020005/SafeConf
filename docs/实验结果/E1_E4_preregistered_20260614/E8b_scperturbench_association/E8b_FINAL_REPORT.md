# E8b scPerturBench aggregate-error association

## Claim boundary

This is an external benchmark method-error association on shared biological
datasets. It is not vector-level SafeConf validation of the benchmark methods.

## Frangieh primary

- Gate: **PASS**
- Methods: 15
- Perturbations: 74
- Across-method median Spearman: 0.5836 (95% perturbation-bootstrap CI 0.3935 to 0.7261)
- Positive methods: 14/15 (93.3%)
- Shuffled-risk null median: 0.0072; empirical one-sided p=0.004975
- Sample-size baseline median Spearman: 0.7637

## Post hoc sample-size diagnostic

The sample-size baseline is stronger than the frozen score, so the primary
association must not be presented as free of cell-count confounding.
- Median partial Spearman after controlling log(Nstimulated): 0.3349 (post hoc 95% perturbation-bootstrap CI 0.0469 to 0.5378)

## Frangieh sensitivity

```text
 dataset           metric  DEG  n_methods  n_perturbations  median_spearman  pct_methods_positive
Frangieh              mse   20         15               74         0.009005              0.600000
Frangieh              mse   50         15               74         0.083526              0.800000
Frangieh              mse  100         15               74         0.212129              0.866667
Frangieh pearson_distance 5000         15               74        -0.426500              0.200000
```

The association weakens for small DEG panels and reverses for
pearson_distance at DEG=5000. The E8b conclusion is therefore metric- and
gene-panel-specific, not universal across benchmark metrics.

The context-similarity-only component is undefined after perturbation-level
aggregation because it is constant across the 74 aligned Frangieh tasks.

## sciplex3 sensitivity

- Uses only 60 exact/parenthetical-alias drug mappings.
- All 15 manual proposals are excluded.
- Benchmark errors are aggregated across available seeds and then four doses.

```text
analysis  n_methods  median_spearman  pct_methods_positive
    A549          9         0.457650              1.000000
    K562          9         0.468992              0.777778
    MCF7          9         0.424638              0.888889
  pooled          9         0.383516              0.777778
```

## Reproducibility

- Frozen protocol v0.2 was not modified.
- Primary bootstrap B=1000; shuffled-risk permutations=200; seed=5201.
- Raw scPerturBench aggregate CSV files remain outside Git.
