# Requested information-versus-row contrasts

These fixed contrasts use the already saved 5,000 joint gene bootstrap draws. No new fits, CDFs, predictions or bootstrap draws were performed. The original six-case reproduction proof, diagnostic models and Source/Orion frozen parameters remain unchanged. The extraction script records its code and exact input hashes in [SUPPLEMENT_REGISTRATION.json](SUPPLEMENT_REGISTRATION.json).

Positive macro U20 differences favor the first scenario. The full original-weight pool and duplicate controls have the same 3,616 training rows, total weight 3,616 and HGB settings. The matched pool and unique controls have total weight 1,808, with 3,616 versus 1,808 training rows.

| Difference | Manual ΔU20 (95% interval) | Learned ΔU20 (95% interval) |
|---|---:|---:|
| Full standard pool − GAT copies ×2 | 0.0393 (−0.0700, 0.1309) | 0.1069 (0.0113, 0.1681) |
| Full standard pool − Exphormer copies ×2 | 0.1858 (0.0399, 0.2650) | 0.0650 (−0.0182, 0.1341) |
| Matched-weight pool − GAT unique | −0.1283 (−0.2009, 0.0146) | 0.1005 (0.0220, 0.1609) |
| Matched-weight pool − Exphormer unique | −0.1032 (−0.1803, 0.0312) | 0.1393 (0.0455, 0.2205) |

The Learned pool's observed gain is not reproduced by duplicating GAT records, and its advantage over both unique controls persists with matched total weight in this fixed diagnostic. Its contrast with duplicated Exphormer remains uncertain. Manual gives a different pattern.

These results do not prove distinct biological knowledge as the unique mechanism. Duplication can change empirical quantile bins and the original-record meaning of `min_samples_leaf`; weight matching alone leaves those row-dependent effects and the mixed-predictor feature distribution different. The diagnostic reports these limitations and selects no target method or winner. Exact results are in [REQUESTED_PAIRED_GENE_U20.csv](REQUESTED_PAIRED_GENE_U20.csv).
