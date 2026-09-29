# Dual-Memory Failure and Repair Log

## DM-PUBLIC-001 — learned retrieval over-concentrates

- **Status before repair:** registered; repair result unseen.
- **Evidence:** On TxPert DEV/SEEN, `LearnedHGB` reduced biological reconstruction RMSE from `0.063678` (cell-weighted manual aggregation) to `0.062924`, but reduced cosine from `0.525004` to `0.509135`. Its mean effective source count was `1.667`, compared with `2.222` for cell-weighted aggregation. In the downstream shared risk model, learned retrieval improved macro Utility@20 over the manual prior by only `0.003440` for Exphormer and `0.002419` for GAT; both paired gene-cluster bootstrap intervals included zero.
- **Hypothesis:** the transfer-error learner has a small RMSE signal, but its softmax weighting is too concentrated and discards directionally stable information carried by the support-weighted aggregate.
- **Single permitted repair:** add `LearnedHGBRegularized`, whose retrieval weights are the fixed arithmetic mean of the outer-fold OOF learned weights and cell-count weights: `w_reg = 0.5*w_learned + 0.5*w_cells`, followed by normalization. The coefficient is fixed before viewing repair results and is not tuned.
- **Expected result:** retain the learned prior's RMSE benefit while restoring effect cosine and effective source count; downstream Utility@20 should be non-inferior to both `LearnedHGB` and `CellWeighted` across GAT and Exphormer.
- **Stopping rule:** if the repaired prior does not improve both effect reconstruction stability and downstream risk consistently, stop modifying public retrieval. Retain the strongest simple/manual prior and report that learned retrieval was not supported.
- **Evidence role:** DEV/SEEN only. No McFaline test truth may be accessed.

### Repair result (completed)

`LearnedHGBRegularized` achieved biological transfer RMSE `0.062113`, cosine `0.529424`, and effective sources `2.118`. It improved the downstream Shared HGB U20 over the manual prior by `+0.010540` (Exphormer) and `+0.009741` (GAT). The paired intervals for the manual comparison include zero, so this is retained as the preregistered repaired candidate and described as a positive but not definitive increment. No second retrieval repair is permitted.

## Execution-only incidents

- The first public-biology report write failed because optional `tabulate` was absent after all calculations completed. Replaced `DataFrame.to_markdown()` with a dependency-free Markdown formatter; no data, split, feature, model, or metric changed.
- The Public Memory growth run completed all 25 registered order-by-fraction evaluations, then its Markdown serialization hit the same absent optional `tabulate` dependency. `REPORT.md` was regenerated from the already written CSV artifacts with the dependency-free formatter; no numerical experiment was rerun or changed.
- The first McFaline LatentAdditive validation-only invocation failed before loading data because Hydra parsed the `=` characters in Lightning's checkpoint filename as override grammar. The immutable checkpoint is now exposed through a hash-recorded `model.ckpt` symlink. This changes only command serialization; candidate, checkpoint bytes, validation partition, model configuration, and sealed test status remain unchanged.
- LatentAdditive generated all 109 official validation prediction chunks, after which PerturBench's `on_test_end` attempted to merge roughly 32 GB of dense chunks plus the reference object in memory and was terminated before metric serialization. The already complete predictions were evaluated by the precommitted bounded-memory 512-gene competence audit; no prediction was regenerated and test remained sealed.
- The first competence-audit invocation treated the versioned `gene_ids.json` object as a bare list and failed before scanning expression. The parser was corrected to read its `gene_ids` member. A second execution was interrupted after 20 prediction chunks when scattered HDF5 column reads proved needlessly slow; the implementation now reads each dense chunk sequentially and subsets in memory. Neither incident produced or exposed a metric.
- The initial TxPert source table contained 5,238 target-to-source eligibility rows but only 2,008 physically distinct public experiments. The persistent bank now stores 2,008 experiment entities and a separate 5,238-edge eligibility relation. This corrected entity duplication before any reported Dual-Memory experiment.

## McFaline upstream competence decision

- **LatentAdditive:** failed without repair. Validation effect RMSE was `0.024689` versus `0.022033` for the strongest simple baseline (relative gap `+12.0557%`, cluster CI `[+11.3422%, +12.7948%]`, 0/3 non-inferior strata).
- **Raw DecoderOnly:** narrowly failed the strict point rule. RMSE was `0.022477`, relative gap `+2.0145%`, cluster CI `[+1.3771%, +2.6443%]`, and 2/3 strata were non-inferior.
- **Single registered Decoder repair:** perturbation-cluster OOF convex shrinkage achieved RMSE `0.021841`, improving on the baseline by `0.8734%` with cluster CI `[0.6601%, 1.0882%]` in the favorable direction and 3/3 non-inferior strata. Fold Decoder weights were `0.50/0.25/0.25/0.25/0.25`; the all-validation test-time weight is `0.25`.
- **Decision:** select validation-calibrated DecoderOnly as the sole external upstream. Selection used upstream competence only; no SafeConf score or test truth was available.
