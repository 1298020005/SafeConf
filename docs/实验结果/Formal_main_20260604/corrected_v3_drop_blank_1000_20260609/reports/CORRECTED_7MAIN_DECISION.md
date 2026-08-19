# Corrected 7-main formal audit decision

Date: 2026-06-09

This run uses explicit drop-blank handling for small numbers of blank perturbation labels in Lara exvivo, Lara invivo, and Santinha. Original run directories were not modified.

## Run status

- Input runs: 7
- Usable runs: 7
- Test score rows: 125846
- Bootstrap: 1000
- Main score: `protocol_v0_2_family_confidence`

## Main interpretation

- Lara exvivo and Lara invivo are restored as valid `gene_main` evidence after explicit blank-label handling.
- Santinha is corrected to `gene_main` / CRISPR-cas9. Its frozen protocol result is weak positive and should not be oversold.
- McFarland remains a frozen-protocol failure boundary.
- AURC and excess-AURC are included as formal risk-coverage metrics.

## Main table

| dataset | family | n | aligned rho | partial rho | RC80 improve % | excess AURC |
|---|---:|---:|---:|---:|---:|---:|
| CuiHacohen2023 | gene_main | 2506 | 0.445 | 0.328 | 21.58 | 0.0425 |
| Frangieh | gene_main | 1266 | 0.583 | 0.474 | 5.03 | 0.0042 |
| LaraAstiasoHuntly2023_exvivo | gene_main | 646 | 0.563 | 0.443 | 56.19 | 0.1627 |
| LaraAstiasoHuntly2023_invivo | gene_main | 750 | 0.394 | 0.357 | 12.51 | 2.0194 |
| McFarlandTsherniak2020 | chem_robust | 2326 | -0.086 | -0.061 | 3.95 | 0.9268 |
| SantinhaPlatt2023 | gene_main | 546 | 0.152 | 0.212 | -1.04 | 0.1329 |
| SrivatsanTrapnell2020_sciplex3 | chem_robust | 1128 | 0.428 | 0.629 | 15.10 | 0.0107 |

## Related files

- `tables/FORMAL_MAIN_TABLE.csv`
- `tables/FORMAL_SCORE_SUMMARY.csv`
- `tables/OLD_DOCS_VS_DROP_BLANK_1000_MAIN_COMPARISON.csv`
- `../safeconf_formal_main_v3_drop_blank_probe_20260609/tables/SENTINEL_VS_DROP_BLANK_MAIN_COMPARISON.csv`
- `../safeconf_formal_main_v3_drop_blank_inputs_20260609/DROP_BLANK_LOG.csv`

