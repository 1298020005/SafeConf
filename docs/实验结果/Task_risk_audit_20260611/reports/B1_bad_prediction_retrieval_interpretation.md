# B1 bad-prediction retrieval

## One-line decision

SafeConf can prioritize bad-prediction review in most datasets, but report dataset-level heterogeneity.

## Main top-10% result

- Datasets where `safeconf_lodo_risk` top 10% enrichment is above random: 6/7
- Datasets where `safeconf_lodo_risk` top 10% enrichment is at least 2x random: 3/7
- Macro mean top 5% enrichment, `safeconf_lodo_risk`: 2.656
- Macro mean top 10% enrichment, `safeconf_lodo_risk`: 2.313
- Macro mean top 20% enrichment, `safeconf_lodo_risk`: 2.030
- Macro mean top 10% enrichment, `predicted_magnitude`: 3.300
- Macro mean top 10% enrichment, `safeconf_perdataset_risk`: 5.357
- Macro mean top 10% enrichment, oracle true magnitude diagnostic: 8.211
- Magnitude caveat: The deployable predicted-magnitude baseline is stronger than LODO at top 10%, so the current claim should be practical enrichment above random, not dominance over magnitude.
- Dataset-level caveat: `predicted_magnitude` is stronger than LODO in 5/7 datasets. LODO only meaningfully beats magnitude in Santinha. In Lara_exvivo, LODO is numerically above predicted magnitude at top 10%, but both are below random, so this should not be counted as a real LODO win.

## Required caveats

- All top-k thresholds are computed within each dataset, not pooled across datasets.
- `oracle_magnitude_diagnostic` uses true-effect magnitude and is not deployable; it is shown only as an upper diagnostic reference.
- Confidence scores are flipped onto a risk axis before ranking.
- The macro row is an average of per-dataset metrics; interpret the per-dataset rows first.

## Special datasets

- McFarlandTsherniak2020: top-10 enrichment lodo=2.356, predicted_magnitude=2.571, frozen_v02=0.900
- SantinhaPlatt2023: top-10 enrichment lodo=1.444, predicted_magnitude=0.902, frozen_v02=1.083
- LaraAstiasoHuntly2023_exvivo: top-10 enrichment lodo=0.612, predicted_magnitude=0.459, frozen_v02=7.798, perdataset=7.186. This is a LODO transfer-layer failure, not a frozen-v0.2 failure.
- LaraAstiasoHuntly2023_invivo: top-10 enrichment lodo=1.733, predicted_magnitude=2.267, frozen_v02=2.533

## How to use this table

If `safeconf_lodo_risk` has enrichment above random, it means the highest-risk predictions are enriched for the truly worst errors. That is the practical answer to: 'what is task-risk scoring useful for?'
