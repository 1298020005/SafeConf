# Fixed Source prediction uncertainty

These are SEEN statistics conditional on the saved fitted models and fixed orders/seeds. They do not change the frozen Source models or Orion method.

- Evaluation uses **543 tasks, 380 biological gene clusters**, jointly across a172 (189), t98g (177), and u87mg (177). Every original context passes n≥20 before resampling; all 5000 U20 draws are valid.
- The same gene multiplicities are used for every budget, order, seed, reference, and diversity method. Curves average the metrics of five fixed orders × three fixed seeds; duplicate prediction rows are never pooled into a larger biological cohort. Seeds add no independent observations.
- All **150 scaling cells and 30 diversity cells**, all seven metrics, and every fixed order point are retained. Original point estimates reproduce within 1.67e−16. Intervals are nominal paired percentile 95% intervals.
- Learned full-budget U20 is 0.695253, versus the fixed-order mean 0.355536 at 10% Source supervision: Δ+0.339716, CI [0.223499, 0.390742]. Manual Δ100%−10% is +0.205204, CI [0.112537, 0.277519]. Manual intermediate means are nonmonotonic; these results do not support universal improvement with each additional budget increment.
- Learned full-pool U20 exceeds GAT-only by +0.086029, CI [0.011476, 0.157517]; Exphormer-only by +0.124845, CI [0.038087, 0.212687]; equal-record pooling by +0.128708, CI [0.021106, 0.179955]; and separate-risk averaging by +0.102166, CI [0.018369, 0.164578]. Every Manual diversity-pair U20 interval crosses zero. All pairs are reported; no source winner is selected.
- Single-source/equal-record fits use 1808 predictor-task error records; full pooling and separate averaging use 3616 records from the same 1808 physical biological tasks and 575 Source gene clusters. GAT and Exphormer are two architectures within the TxPert family. Pooling records does not double independent biological measurements or original studies.
- **CDF resolution changes with budget.** The existing 448-group CDF audit fits only allowed Source training errors. At 10% budget, individual model/context groups contain 37–58 errors (nominal midrank resolution 1/58–1/37). Growth changes supervision, fitting, and label resolution together; it is not an isolated causal estimate of adding records at constant labels. No target CDF or new error labels are fitted here.
- These paired Source increments do not establish an external advantage over historical support/distance, biological-model-specific error estimation, or zero-history safety. The registered prospective Orion comparison is still separate and incomplete.

Actual computation: 19.442 seconds wall, 76.800 CPU seconds, 378456 KiB peak RSS; zero fits, upstream calls, new error labels, or Orion reads. Independent review covers 724 helper cases and 392 whole-pipeline point/contrast checks.

Primary artifacts: `SCALING_CURVE_INTERVALS.csv`, `SCALING_PAIRED_BUDGET_CHANGES.csv`, `DIVERSITY_INTERVALS.csv`, `DIVERSITY_ALL_PAIRED_CONTRASTS.csv`, `ALL_FIXED_ORDER_SEED_POINTS.csv`, `SCALING_FIXED_ORDER_RANGE.csv`, `POINT_REPRODUCTION.csv`, and `INDEPENDENT_REVIEW/FINAL_REVIEW.json`.
