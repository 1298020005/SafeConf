# Source biological label control: actual saved results

The ten fits, twenty saved models and all fixed-count statistics completed. The original process exited with code 1 only while formatting this summary because `tabulate` was absent. Its FAILED receipt and exact runner are preserved. This renderer uses saved summaries only: zero new fits, scores, statistics or RNG.

| Direction | BioNorm − OldRank U20 | Nominal paired 95% CI | Valid draws |
| --- | ---: | --- | ---: |
| GAT_to_Exphormer | -0.056636 | [-0.094668, -0.025689] | 5000 / 5000 |
| Exphormer_to_GAT | -0.052161 | [-0.086591, -0.019170] | 5000 / 5000 |

| Direction | Method | utility20 | spearman | aurc | high_risk_miss_rate | error_at_10 | error_at_20 | error_at_50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exphormer_to_GAT | BioNorm_HGB | 0.730973 | 0.730567 | 0.047858 | 0.386544 | 0.037299 | 0.040503 | 0.048404 |
| Exphormer_to_GAT | LearnedWeightedHistoryDistance | 0.593518 | 0.591451 | 0.052043 | 0.507915 | 0.045960 | 0.045825 | 0.050494 |
| Exphormer_to_GAT | Magnitude | 0.748808 | 0.736428 | 0.047795 | 0.369342 | 0.037510 | 0.040944 | 0.048375 |
| Exphormer_to_GAT | OldRank_HGB | 0.783134 | 0.752031 | 0.047775 | 0.346693 | 0.037837 | 0.040498 | 0.048206 |
| GAT_to_Exphormer | BioNorm_HGB | 0.727108 | 0.725227 | 0.047801 | 0.394579 | 0.037473 | 0.040446 | 0.048026 |
| GAT_to_Exphormer | LearnedWeightedHistoryDistance | 0.592336 | 0.595199 | 0.051616 | 0.493667 | 0.044932 | 0.045251 | 0.050260 |
| GAT_to_Exphormer | Magnitude | 0.746253 | 0.723165 | 0.047896 | 0.373728 | 0.038402 | 0.041082 | 0.048350 |
| GAT_to_Exphormer | OldRank_HGB | 0.783743 | 0.740778 | 0.047688 | 0.356272 | 0.037849 | 0.040508 | 0.048056 |

Both primary macro contrasts favor old predictor-error supervision over this biological-magnitude supervision control. The estimate is conditional on the same prediction/Public inputs, learner and Source family. It does not remove prediction information, establish value beyond every biological proxy, identify a root cause, or establish external transfer. Secondary metrics are retained; the control has lower macro error_at_10 in both directions, so the advantage is not universal across endpoints.

The Source population is the existing 1808 biological tasks / 575 gene clusters. Both directions share biological truth. The 14464 norm-supervised training rows are repeated fit-row instances across ten fits, not independent new experiments. The new control uses zero realized predictor-error training labels; the ten reused original models retain their historical error-label training. Cached Source prediction inputs remain in both arms; only new upstream calls are zero.

Counts are the exact saved 5000 × 575 Source gene multiplicities, generation seed 20261002; this is not a reproduction of the original main bootstrap seed. All contexts, both contrast signs and all seven metrics remain in the saved CSVs. Nominal paired intervals describe Source DEV/SEEN sampling conditional on these fitted models; no fresh confirmation, formal-method promotion or old external champion selection is claimed.

Runtime predictions, models and draw arrays stay server-side in `/home/yyf/runtime_artifacts/safeconf_research_20261001/source_biology_label_control_v1`. The immutable original execution cost was 16.752635 seconds wall, 16.740402 seconds CPU, peak RSS 566534144 bytes; ten new fits and ten model reuses. The original terminal code is 1; this report-only recovery terminal code is 0.
