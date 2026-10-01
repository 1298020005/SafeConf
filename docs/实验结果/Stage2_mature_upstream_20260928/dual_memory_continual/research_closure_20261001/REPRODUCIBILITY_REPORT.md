# Reproduction entry

Run from the repository root, with the registered September prediction/public-memory assets present:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -u tools/scripts/run_safeconf_research_closure.py --phase all
```

This rebuilds nested Public features, all sixteen primary methods, five cluster-label nulls, source coverage/diversity, strict feedback curves, 54 paired comparisons with 5000 draws each, and the three scientific figures. It does not claim to execute the pending official PertEMA or Public-temperature experiments.

Intermediate numerical features are under `/home/yyf/runtime_artifacts/safeconf_research_20261001`. The registered primary contexts are macro-aggregated after pooling outer OOF predictions; outer-fold context results are separately available. All three fixed risk seeds and all five growth orders are retained. They are not independent datasets.

The strict feedback evaluation uses the previous 60/40 cluster split. Its CDF is refit using only the currently allowed feedback errors; validation error labels are not used. Validation biological truth and upstream calibration remain available and disclosed in the information ledger. Target learners with zero feedback are unfitted, not copies of Shared.

The code imports the frozen September helper definitions without modifying those files. `ASSET_HASH_REGISTRY.json` preserves the original frozen-file hashes. Each added phase records its execution-code hash. Numeric results are reproducible; PDF metadata timestamps can differ.

Verification completed: five unit tests covering budget-CDF isolation, mid-rank/ties, block-label permutation, the weighted-history identity, and U20 edge cases. Nested cluster-disjointness and prediction/truth alignment are also checked at runtime. The three numerical runs and all 54 bootstrap comparisons completed successfully.

Scope: the tested tasks have historical effects for the same perturbation in other conditions/contexts. These results do not establish zero-history novel-gene risk estimation. All October evaluation data were already seen, and new comparisons do not inherit pristine confirmation status.


## Completed October common-axis version and isolated reproduction

The original-contract results and original frozen helpers remain separate. The canonical cache reuses full native predictions on 2840 genes and reconstructs actual train/validation biology. Common-axis downstream fits, target-feedback adaptations, all-metric intervals and physical content controls are now completed. All are SEEN sensitivity/contract analyses.

An independent rerun uses fresh result and risk-cache roots, so completed result files are not overwritten:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/miniconda/bin/python -u tools/scripts/run_safeconf_common_axis_closure.py --phase all --result-root /home/yyf/runtime_artifacts/safeconf_reproduction_new/common_axis_reports --risk-cache-root /home/yyf/runtime_artifacts/safeconf_reproduction_new/common_axis_risk_cache
```

The registered biology, raw prediction and competence inputs must already exist. The new output root copies the same competence registration; it cannot choose another gate. `run_safeconf_official_pertema_feedback.py` accepts the reproduced risk-cache root and its `results` subdirectory through `--runtime-root` and `--result-root`. It requires the pinned official checkout and xgboost environment. Native-control runs require the separately audited TRAIN-NTC feature cache.

The fixed common-axis historical-content controls use `run_safeconf_public_mechanisms.py --common-axis --phase content`. Saved predictions feed `run_safeconf_feedback_metric_uncertainty.py --results <result-directory>`, `run_safeconf_common_attribution_statistics.py`, and the figure script. Guide and cell-sampling sensitivity is explicitly diagnostic; `run_safeconf_common_truth_reproducibility.py --cell-sampling` does not change the primary evaluator.

`run_safeconf_validation_reuse_baseline.py` adds the strongest existing-validation control: 230 held-out-gradient validation records in 123 clusters, excluding all test genes. Full validation was already used for upstream selection and Public biology; these extra risk/CDF error uses are separately counted, and no new upstream calls are made.

Seven meaningful invariants now pass, including float-safe task pairing and rejection of different truths/duplicate predictions. Five specifically registered original frozen assets pass byte-hash checks. The original common-axis pretruth CSV was filled with truth after scoring, so its pretruth bytes were not independently retained; a retrospective score-only snapshot is explicitly not prospective proof. Fresh runs now retain a separate immutable score-only file. Original September freeze artifacts remain unchanged.

Cost rows record measured tabular fit/predict durations. Their sum is workload, not elapsed wall-clock time or GPU time. No new large upstream has been trained in this research closure. Per-task predictions and dense biological vectors stay in runtime storage; reviewable result/contract/statistical summaries are indexed by SHA-256 in `RESULT_INDEX.csv`.
