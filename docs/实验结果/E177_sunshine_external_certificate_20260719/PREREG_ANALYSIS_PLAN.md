# E177 external certificate preregistration

## Purpose

E176 confirmed the risk certificate on a same-study multi-donor Primary CD4 setting. E177 moves to an independent public processed single-cell perturbation dataset. The goal is narrow: test whether the certificate-style conclusion still behaves sensibly outside the Primary CD4 line, without claiming wet-lab validation or deployment readiness.

## Frozen data boundary

The source is used only as a processed expression matrix. The metadata freeze reads `obs` and `var_names` in backed mode and does not decode `X` or layers. The selected population is exact single-gene CRISPRi labels with at least 20 exact-match cells overall, present in the expression matrix and scGPT vocabulary, and at least 3 cells in each of the eight `gem_group` technical groups.

`gem_group` is treated only as a technical repeat label. It is not a donor, patient, biological context, or independent study by itself.

## Frozen split

Selected targets: 144. Split by target identity before expression access:

- train targets: 54 (432 tasks)
- validation targets: 10 (80 tasks)
- calibration targets: 30 (240 tasks)
- final evaluation targets: 50 (400 tasks)

The split is balanced across target cell-count quartiles and sorted by salted SHA-256 identities. Calibration and evaluation target expression is not available to model training or the pretruth gate.

## Model and gate

Use scGPT and GEARS with seeds 3407, 3408, 3409, 3410, 3411. Before calibration truth is opened, the run must pass a truth-blind stability gate on train/validation/query predictions. The deployed object is the five-seed family mean for each predictor.

The primary certificate keeps the E176 conclusion style:

- deterministic lower bound: RMSE(p1, p2) / 2 for the two-predictor mean and max errors
- split conformal upper bound: target-level clusters, one cluster contains all eight technical groups of the same perturbation
- target coverage: 90 percent

Ranking metrics are diagnostics only. A failure to beat predicted magnitude is reported directly and does not get repaired after evaluation truth is visible.
