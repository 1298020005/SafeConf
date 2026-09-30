# McFaline cold-start pre-truth seal

The external upstream, public learner, risk learners, task manifest, and all
543 test risk predictions were frozen before any test treated expression was
read.

## Validation-only diagnostic

| method | Utility@20 | Spearman |
|---|---:|---:|
| Universal-P HGB | 0.540615 | 0.495507 |
| Manual Public HGB | 0.740915 | 0.806857 |
| Learned Public HGB | 0.738144 | 0.801553 |
| Learned Public Ridge | 0.647788 | 0.670412 |

The primary external method remains `ZeroLabelSharedHGB`, selected from the
TxPert development evidence before this validation diagnostic. It uses no
McFaline upstream-error labels. `ValidationAdapted_LearnedPublic_HGB` is a
predeclared secondary held-out-test method. The slightly higher validation
score of manual weighting does not change the frozen primary method.

## Integrity

- Test metadata and matched control expression were used to construct legal
  prediction inputs and the treated-state effect contract.
- Test treated expression was not read.
- `SEALED_TEST_RISK_PREDICTIONS.csv.gz` and `FINAL_METHOD_CONFIG.json` were
  hashed before the confirmation evaluator was created or run.
- Public biological transfer may use McFaline validation biological effects;
  the primary risk core's error labels come only from TxPert.
