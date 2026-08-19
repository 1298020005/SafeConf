# C3 SafeConf scope and boundary table

This table is a defensive map: what can be claimed, what must be limited, and what should stay supplementary.

## Independent-review correction

Claude's latest review is substantively correct on three points, and this report has been updated accordingly:

- `LaraAstiasoHuntly2023_exvivo` must be named as a LODO top-k retrieval failure.
- `predicted_magnitude` is stronger than LODO in most datasets, so LODO should not be framed as a universal improvement over magnitude.
- Current GEARS evidence has no frozen v0.2 / LODO / per-dataset SafeConf score on GEARS predictions; it is only a provenance/dedup audit plus a Frangieh-only magnitude diagnostic.

## Boundary table

```text
                     scope                       status                                                                    safe_claim                                                                                                  caveat
          Leakage precheck                       strong                        B1 can be interpreted as a prechecked retrieval audit.                                                         Full reproducibility lock still belongs to C4b.
Simple predictor task risk         strong_with_boundary                     Task difficulty dominates among tested simple predictors.                                                Do not generalize this statement to all deep predictors.
  Bad-prediction retrieval      useful_but_not_dominant         SafeConf LODO enriches bad predictions above random in most datasets.                       Predicted magnitude is stronger at macro top10 and must remain a main comparator.
Magnitude baseline vs LODO magnitude_stronger_than_lodo                         LODO has above-random cross-dataset screening value. Incremental value over deployable magnitude is limited and must be framed as transfer/calibration value rather than universal dominance.
                 McFarland      failure_rescue_boundary Frozen v0.2 fails; learned LODO provides partial rescue and useful retrieval.                                                       Never present McFarland as a frozen v0.2 success.
                  Santinha                weak_positive                                                 Some retrieval value remains.                                                            Keep as weak/supportive, not a headline win.
         Lara exvivo LODO        lodo_retrieval_failure       Frozen and per-dataset scoring can retrieve bad Lara_exvivo predictions strongly.      LODO transfer fails in the top-risk tail here; report separately rather than hiding behind 6/7 macro wording.
               Lara invivo predictor_difference_caution                                                   Risk retrieval is positive.                                  A1 showed larger V0/ContextSim difference here, so discuss separately.
                     GEARS       no_safeconf_gears_score_yet             Existing GEARS records support provenance/dedup and magnitude diagnostics only.    Current work has not shown frozen v0.2 / LODO / per-dataset SafeConf scores on GEARS predictions.
          External small-n              supportive_only                              External validation is directionally supportive.                                             Small-n and uncertain AURC intervals prevent strong claims.
```

## Most cautious wording

The safest current wording is:

> SafeConf is a task-risk scoring protocol validated across multiple simple predictors. Its LODO transfer risk score is above random for bad-prediction retrieval in 6/7 datasets, with macro top-10 enrichment of 2.313x. This supports practical screening value. However, three boundaries must be reported clearly: deployable predicted magnitude is stronger than LODO in most datasets, Lara_exvivo is a LODO top-k failure, and GEARS currently has no SafeConf score evaluation.

## Lara_exvivo correction

For `LaraAstiasoHuntly2023_exvivo`, LODO is below random at the most important top-k levels:

```text
safeconf_lodo_risk top 5%  enrichment = 0.593
safeconf_lodo_risk top 10% enrichment = 0.612
safeconf_lodo_risk top 20% enrichment = 1.797
```

But frozen v0.2 and per-dataset risk are strong on the same dataset:

```text
frozen v0.2 top 10% enrichment      = 7.798
per-dataset risk top 10% enrichment = 7.186
```

So this is a LODO transfer-layer failure, not a failure of every SafeConf-style score on Lara_exvivo.

## Magnitude comparison

At top-10 retrieval:

```text
predicted_magnitude macro enrichment = 3.300
safeconf_lodo_risk macro enrichment  = 2.313
```

`predicted_magnitude` is stronger in 5/7 datasets. LODO only meaningfully beats magnitude in Santinha. In Lara_exvivo, LODO is numerically above predicted magnitude, but both are below random at top 10%, so it should not be counted as a real LODO win.

## GEARS correction

C1 does not currently show that SafeConf scores GEARS risk. It only shows:

- existing GEARS rows can be deduplicated and audited;
- scoreable retrieval rows are Frangieh-only;
- `gears_prediction_magnitude_risk` is strong, but this is a magnitude diagnostic;
- `gears_uncertainty_confidence` has top-10 enrichment of 0.000.

Therefore GEARS should stay as a preliminary provenance/dedup and magnitude-diagnostic probe until frozen v0.2 / LODO / per-dataset SafeConf scores are actually evaluated on GEARS prediction records.
