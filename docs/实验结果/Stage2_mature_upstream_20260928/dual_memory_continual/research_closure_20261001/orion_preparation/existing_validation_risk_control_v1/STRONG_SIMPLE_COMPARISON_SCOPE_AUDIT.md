# SEEN validation-control statistical completeness audit

Scope: existing code, manifests, identities and cached summary tables only. No model/prediction/error arrays, fits, RNG calls or raw expression reads were used in this audit.

## Completed comparison

The saved control contains all30fixed models plus3anchors:693method/context/metric rows and1680paired rows (80pairs ×3contexts including macro ×7metrics), all with5000valid draws. On the fixed232tasks/144genes, full-budget Ridge U20 is TargetOnly0.029795, PublicTarget0.279075 and SharedTarget0.243019. Public−Target is0.249280,95%CI[-0.067474,0.476093]; Shared−Public is−0.036056,CI[-0.040850,0.056696]. These point patterns do not establish incremental Public or Shared value. Positive standalone U20 intervals do not establish superiority to another method.

The80pairs cover LearnedWeightedDistance and frozenLearnedSourceHGB, but omit the three DirectRMSE rules and ManualWeightedDistance. The existing LearnedDirectRMSE point≈0.2671 is therefore not yet paired with the new controls. Marginal intervals cannot substitute for that paired comparison.

## Reusing fixed bootstrap draws

Original `cohort_statistics` obtains genes by first-occurrence `unique()` and retains draws only in memory; the actual original EVALUATION_MANIFEST/ARTIFACT_HASHES/output directory does not persist draw arrays. The control uses canonical sorted Ensembl genes and saves `BOOTSTRAP_GENE_INDICES.npy` (SHA be533a498413bf284e187faf2d8c3d103949f865ba9ef0765d71ef7925b9479c) plus all33methods' metric draws. The actual144-gene orders differ at144positions (old orderSHA 27851c2a912af5b951940eb54270c99830756bd1c11f5a8302a6049464f90e1c, newSHA 6082795644a9a888245efcf146119cead9ade85ee30ed84bce62153864ce0ac7). Identical seed20260929 does not make the old and new CIs paired.

The stated zero-fit supplement is statistically coherent: calculate only Uniform/Manual/LearnedDirectRMSE and ManualWeightedDistance metrics on the existing control's exact5000indices/144gene ordering, keeping the same232rows, context blocks, query-ID tie rules and unchanged dec8 `metric_values`. LearnedWeightedDistance already has matching saved draws. Compute each of the30target methods minus each of the5fixed rules per draw, preserving sign as target−rule. This gives150baseline contrasts:30existingWeighted comparisons plus120additional comparisons. Preserve undefined metrics and finite-draw counts; reproduce existingWeighted contrasts exactly. Show all5rules; do not choose a best baseline after outcomes. This is completion of the SEEN descriptive comparisons, with no new resampling, learner or Source variant.

## What this control represents

Each fitted target-risk model pools validation records from HCT116 andHEK293T. Their underlying published-LM checkpoint hashes differ; the budget-only CDFs are stratified by context/version, while the risk learner itself is pooled. SharedTarget predicts rank risk with frozenSharedscore as one feature; it does not fit `rank−shared` residuals or a checkpoint-specific shrinkage adapter. Thus it is a target-study pooled risk-control result for two context-specific model versions, not evidence for single-checkpoint ErrorAdapter personalization or version-isolated ErrorMemory updates. The two upper models remain one published family attempt; the30downstream risk fits are not30independent upstream families. Original Source parameters and13primary results remain unchanged; all conclusions remain SEEN/conditional on fixed trained models and this cohort.
