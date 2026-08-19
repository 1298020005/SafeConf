# GEARS alignment audit

## Verdict

`FAIL_FOR_DIRECT_DISAGREEMENT`.

GEARS can be reported as adapter feasibility / supplement, but current outputs should **not** be directly subtracted from V0/ContextSim predicted effects to compute model disagreement.

## Key findings

- Main Frangieh records: 6330 records, 211 perturbations.
- GEARS records: 62 records, 58 clean perturbations.
- Perturbation overlap: 58 / 58 GEARS perturbations are present in main Frangieh.
- Main contexts: Co-culture, Control, IFNγ.
- GEARS context: GEARS_single_heldout.
- Main prediction array dimension: 5000 genes.
- GEARS prediction array dimension: 3000 genes.
- Source h5ad gene overlap: 3000 genes, but selected gene order for prediction arrays is not exported.

## Interpretation

Perturbation names mostly align, but context and prediction-space contracts do not.

The biggest blocker is not gene overlap alone. The bigger issue is that GEARS records are `GEARS_single_heldout`, while the main SafeConf Frangieh task uses biological contexts such as `Co-culture`, `Control`, and `IFNγ`. Therefore a direct `GEARS vs V0/ContextSim` disagreement would mix different task definitions.

## Recommendation

Do not spend major effort forcing alignment in this sprint.

Use GEARS as:

1. adapter feasibility evidence;
2. a negative/weak native uncertainty result;
3. motivation for external confidence scoring.

If GEARS disagreement is revisited later, rebuild the contract from scratch: same task definition, same context, same held-out split, same selected gene panel and gene order.
