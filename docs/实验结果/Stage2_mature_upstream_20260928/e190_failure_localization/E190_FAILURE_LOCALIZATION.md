# E190 SafeConf-v4 failure localization

This is post-hoc diagnosis on a SEEN development asset. It does not modify frozen v4.

## Frozen pipeline by fold

| fold | raw P U20 | raw PQ U20 | p slope | pq slope | weight | V2 U20 | classification |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.1633 | 0.4475 | -0.0631 | 0.9536 | 0.8800 | 0.4475 | Calibration abnormality |
| 1 | -0.3361 | 0.3418 | 0.7971 | 0.1727 | 0.0007 | -0.3361 | Gate misrouting |
| 2 | 0.0348 | 0.1822 | 0.1872 | 0.7340 | 0.9994 | 0.1822 | No registered abnormality |
| 3 | -0.0843 | 0.0141 | 0.2493 | 0.2798 | 0.9291 | 0.0021 | No registered abnormality |
| 4 | -0.1421 | -0.1794 | 0.0801 | 0.8092 | 0.9998 | -0.1794 | History branch underperformance; Gate misrouting |

## Diagnostic calibration variants

| calibration | raw P | raw PQ | calibrated P | calibrated PQ | V2 | V2 rho |
|---|---:|---:|---:|---:|---:|---:|
| frozen_linear | -0.049121 | 0.365209 | -0.037673 | 0.027636 | -0.011379 | 0.202962 |
| identity | -0.049121 | 0.365209 | -0.049121 | 0.365209 | 0.311265 | 0.253041 |
| positive_affine | -0.049121 | 0.365209 | -0.037673 | 0.027636 | -0.011379 | 0.202477 |
| isotonic | -0.049121 | 0.365209 | -0.015624 | 0.371898 | 0.328744 | 0.247498 |

## Paired gene-cluster bootstrap for diagnostic variants

| comparison | delta U20 | 95% CI |
|---|---:|---:|
| identity_vs_frozen | +0.322644 | [-0.053303, +0.554285] |
| isotonic_vs_frozen | +0.340123 | [-0.089148, +0.600832] |
| positive_affine_vs_frozen | +0.000000 | [+0.000000, +0.000000] |
| identity_vs_V1 | +0.007778 | [-0.280327, +0.237622] |
| isotonic_vs_V1 | +0.025256 | [-0.270997, +0.276682] |

## Decision

- Calibration abnormality occurs in 1/5 folds.
- History branch underperformance occurs in 1/5 folds.
- Gate misrouting occurs in 2/5 folds.
- Frozen fold-specific affine calibration reduces pooled history-branch U20 from 0.365209 to 0.027636.
- Identity and isotonic calibration recover pooled V2 U20 to 0.311265 and 0.328744, respectively.
- Cluster-count downsampling is authorized only when gate misrouting appears in at least 3/5 folds.
- Downsampling decision: **STOP**.
- Distribution summaries are diagnostic and are not used as an automatic causal verdict.

The calibration variants are explanatory controls. Their performance cannot replace the frozen v4 result.
