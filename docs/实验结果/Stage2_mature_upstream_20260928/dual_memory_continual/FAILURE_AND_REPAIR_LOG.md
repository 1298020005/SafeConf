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
- The initial TxPert source table contained 5,238 target-to-source eligibility rows but only 2,008 physically distinct public experiments. The persistent bank now stores 2,008 experiment entities and a separate 5,238-edge eligibility relation. This corrected entity duplication before any reported Dual-Memory experiment.
