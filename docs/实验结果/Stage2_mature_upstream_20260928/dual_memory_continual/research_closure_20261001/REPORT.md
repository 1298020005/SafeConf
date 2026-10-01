# October research closure: measured results

These analyses use DEV/SEEN data. McFaline test truth was already opened in September. The original frozen method, sealed predictions and confirmation remain unchanged. These results do not constitute another pristine confirmation.

## Implemented isolation and budget rules

- Five outer gene-cluster folds are identical for all TxPert upstreams.
- For each outer fold, the public transfer learner generates risk-training references through four inner, gene-disjoint folds. The risk-query references come from a full outer-training public learner.
- Universal evidence is recomputed from each upstream's actual aligned effect vectors. Model/family/dataset IDs are used only in split and label bookkeeping.
- Source labels use training-only, per-study/output-contract/model/version/context mid-rank CDFs. Source scaling refits each CDF on that budget's allowed records.
- Complete compatible cluster-label blocks are shuffled together, preserving joint upstream/context layouts. Actual movement exceeds 99% in all runs.
- Strict feedback reuses the previous hash-defined 228-cluster pool / 152-cluster permanent evaluation set. No C validation errors enter the new feedback CDF. Validation biology and upstream calibration uses remain disclosed.
- The existing residual adapter's kappa is selected inside feedback-only, gene-disjoint folds; each inner CDF is refit there. Permanent evaluation errors do not select kappa.

## Main point estimates

| line | method | utility20 | spearman | aurc |
|---|---|---|---|---|
| Exphormer_to_GAT | Learned_WeightedHistoryDistance | 0.605786 | 0.599227 | 0.053083 |
| Exphormer_to_GAT | Learned_hgb | 0.793723 | 0.753558 | 0.048675 |
| Exphormer_to_GAT | Magnitude | 0.768216 | 0.741977 | 0.048831 |
| Exphormer_to_GAT | Manual_WeightedHistoryDistance | 0.589739 | 0.566887 | 0.053591 |
| Exphormer_to_GAT | Manual_hgb | 0.784847 | 0.744611 | 0.048874 |
| Exphormer_to_GAT | Prediction_hgb | 0.762670 | 0.734398 | 0.049132 |
| GAT_to_Exphormer | Learned_WeightedHistoryDistance | 0.626475 | 0.603109 | 0.052629 |
| GAT_to_Exphormer | Learned_hgb | 0.795018 | 0.749545 | 0.048570 |
| GAT_to_Exphormer | Magnitude | 0.758726 | 0.727634 | 0.048927 |
| GAT_to_Exphormer | Manual_WeightedHistoryDistance | 0.565115 | 0.570039 | 0.053188 |
| GAT_to_Exphormer | Manual_hgb | 0.786514 | 0.737704 | 0.048815 |
| GAT_to_Exphormer | Prediction_hgb | 0.764206 | 0.723861 | 0.048996 |
| TxPert_to_McFaline | Learned_WeightedHistoryDistance | 0.614206 | 0.639245 | 0.019812 |
| TxPert_to_McFaline | Learned_hgb | 0.551331 | 0.510874 | 0.020515 |
| TxPert_to_McFaline | Magnitude | -0.141916 | -0.118309 | 0.023005 |
| TxPert_to_McFaline | Manual_WeightedHistoryDistance | 0.604652 | 0.628957 | 0.019845 |
| TxPert_to_McFaline | Manual_hgb | 0.583540 | 0.271321 | 0.021314 |
| TxPert_to_McFaline | Prediction_hgb | 0.042859 | -0.011514 | 0.022245 |

## Paired uncertainty: source learning versus strong simple history

Actual point differences and bootstrap means are separate columns. Every comparison has 5000 paired biological-cluster draws. The same cluster draw applies to every context in a dataset; context results are macro-aggregated.

| line | method_a | method_b | delta_utility20 | ci95_lower | ci95_upper |
|---|---|---|---|---|---|
| Exphormer_to_GAT | Manual_hgb | Manual_WeightedHistoryDistance | 0.195109 | 0.151362 | 0.270840 |
| Exphormer_to_GAT | Learned_hgb | Learned_WeightedHistoryDistance | 0.187937 | 0.127291 | 0.247440 |
| GAT_to_Exphormer | Manual_hgb | Manual_WeightedHistoryDistance | 0.221400 | 0.142230 | 0.266120 |
| GAT_to_Exphormer | Learned_hgb | Learned_WeightedHistoryDistance | 0.168543 | 0.117654 | 0.232377 |
| TxPert_to_McFaline | Manual_hgb | Manual_WeightedHistoryDistance | -0.021111 | -0.149550 | 0.031432 |
| TxPert_to_McFaline | Learned_hgb | Learned_WeightedHistoryDistance | -0.062875 | -0.124213 | 0.046060 |

## Strict feedback curve

The shared model is the October nested-source baseline, not a renamed September frozen estimate. The zero-feedback target learners are unfitted and absent. A constant-score small-budget HGB has undefined Spearman; its task-ID tie ordering must not be interpreted as useful learning.

| budget | method | utility20 | spearman | aurc |
|---|---|---|---|---|
| 0.100000 | PublicTarget_HGB | 0.034479 | nan | 0.022493 |
| 0.250000 | PublicTarget_HGB | 0.351332 | 0.583077 | 0.020156 |
| 0.500000 | PublicTarget_HGB | 0.685280 | 0.646762 | 0.020136 |
| 0.750000 | PublicTarget_HGB | 0.676234 | 0.661742 | 0.019989 |
| 1.000000 | PublicTarget_HGB | 0.722281 | 0.690677 | 0.019908 |
| 0.100000 | ResidualHGB | 0.598697 | 0.516347 | 0.020709 |
| 0.250000 | ResidualHGB | 0.560590 | 0.627691 | 0.020097 |
| 0.500000 | ResidualHGB | 0.738655 | 0.663320 | 0.020108 |
| 0.750000 | ResidualHGB | 0.747965 | 0.666395 | 0.019998 |
| 1.000000 | ResidualHGB | 0.774432 | 0.704037 | 0.019886 |
| 0.000000 | Shared | 0.598697 | 0.516347 | 0.020709 |
| 0.100000 | Shared | 0.598697 | 0.516347 | 0.020709 |
| 0.250000 | Shared | 0.598697 | 0.516347 | 0.020709 |
| 0.500000 | Shared | 0.598697 | 0.516347 | 0.020709 |
| 0.750000 | Shared | 0.598697 | 0.516347 | 0.020709 |
| 1.000000 | Shared | 0.598697 | 0.516347 | 0.020709 |
| 0.100000 | SharedTarget_HGB | 0.034479 | nan | 0.022493 |
| 0.250000 | SharedTarget_HGB | 0.340544 | 0.587718 | 0.020149 |
| 0.500000 | SharedTarget_HGB | 0.696690 | 0.649011 | 0.020132 |
| 0.750000 | SharedTarget_HGB | 0.675864 | 0.670362 | 0.019977 |
| 1.000000 | SharedTarget_HGB | 0.681707 | 0.693147 | 0.019898 |
| 0.100000 | TargetOnly_HGB | 0.034479 | nan | 0.022493 |
| 0.250000 | TargetOnly_HGB | 0.149738 | 0.074980 | 0.022107 |
| 0.500000 | TargetOnly_HGB | 0.227365 | 0.069158 | 0.021996 |
| 0.750000 | TargetOnly_HGB | 0.118892 | 0.016979 | 0.022180 |
| 1.000000 | TargetOnly_HGB | 0.099192 | 0.040772 | 0.022168 |
| 0.100000 | TargetOnly_Ridge | 0.173697 | 0.173597 | 0.021810 |
| 0.250000 | TargetOnly_Ridge | 0.258461 | 0.176424 | 0.021857 |
| 0.500000 | TargetOnly_Ridge | 0.076946 | 0.097695 | 0.021992 |
| 0.750000 | TargetOnly_Ridge | 0.036134 | 0.073586 | 0.022143 |
| 1.000000 | TargetOnly_Ridge | 0.220497 | 0.121770 | 0.021981 |

## Research lead's interpretation

1. Public biological information is strongly useful on the external McFaline task set. Learned weighted historical distance obtains U20 0.614206 versus magnitude -0.141916; the paired difference is 0.756122, CI [0.611099, 0.905596]. This remains an information comparison, not a new confirmed algorithm claim.
2. True source supervision beats the same-feature shuffled-label learner. It also improves over strong historical distance on both TxPert architecture-transfer lines. This does not imply universal source-learning superiority: on McFaline, learned HGB is 0.551331 versus historical distance 0.614206; the difference -0.062875 has CI [-0.124213, 0.046060]. The external comparison does not establish that HGB adds value beyond the simple rule.
3. More supervision is not monotonically better. The five-order source curves and equal-record/full-record controls are preserved. Source diversity has method-dependent effects; it cannot be reduced to “more models always help.”
4. Sufficient target feedback plus Public features can improve risk ordering. The strict P-only target learner remains weak, while Public+Target and residual adaptation become useful. A source score added as a target feature is not uniformly beneficial; fusion should not be declared the default winner.
5. The target prediction magnitude median is 0.0059 versus source 0.0471; all target values fall outside the source 1st–99th percentile range for four prediction-amplitude features. This is an observed input shift, not proof of the root cause or an automatic failure classification. The frozen architecture is unchanged.
6. The current defensible narrative is conditional reuse of biological references and source risk supervision, followed by target feedback under an explicit budget. Neither learned retrieval superiority, safe no-history fallback, nor universally improved performance with more sources is established.

## Artifact and run status

The 16-method matrix, source scaling/diversity and strict feedback curves are complete. Original September assets are hash-checked. Official PertEMA, additional mechanism controls, Public growth/temperature, full feedback uncertainty and manuscript closure remain unfinished; no full-project completion is claimed.

Figures are in `figures/`; per-task predictions, audits, source ranges and complete results are in this directory. `EXECUTION_STATUS.json` names completed and outstanding work. The run-scope budget ledger was replaced by a method-level ledger so magnitude and direct-distance baselines do not incorrectly inherit source-error budgets.
