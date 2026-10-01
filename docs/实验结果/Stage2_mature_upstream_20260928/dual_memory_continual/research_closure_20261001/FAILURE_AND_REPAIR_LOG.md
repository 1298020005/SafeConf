# Implementation and scientific observations

## Nested isolation repair

The previous public-OOF feature cohort could carry outer-query biology into risk-training reference learning. Reconstructed the five outer cohorts with four inner public folds each. September frozen code and confirmation were preserved. The new results are explicitly DEV/SEEN.

## Method budget accounting repair

A first run-scope ledger described total source fitting regardless of individual method. Replaced it with per-method accounting; magnitude and direct-history methods now record zero source-error labels. Prediction/learner outputs were unaffected.

## Nonmonotonic scaling and external algorithm boundary

Source HGB did not exceed strong simple public distance on the external dataset. Completed block-label nulls, source coverage, equal-record diversity and source/target feature-range diagnostics. No test-driven model redesign was performed. Results and confidence intervals are preserved.

## Small-feedback HGB capacity

At 10% budget, the feedback cohort is too small for two leaves of at least 20 records. Constant predictions produce lexical tie rankings and undefined Spearman. The fixed learner is kept for the fair budget comparison; this point is not presented as successful target learning.


## 2026-10-01 native-control preprocessing repairs

- The small/medium official CSVs contain all raw cell IDs but intentionally leave unsampled roles NA. The initial all-role guard confused this with missing IDs. The repaired guard checks ID coverage separately and excludes every non-TRAIN or NA role.
- Object-string `np.isin` on 878229 IDs was quadratic. The confirmed owned process was intentionally stopped and replaced with a pandas hash membership check, preserving the selected control mask.
- Native-control extraction completed with 69502 training NTCs and 95.95% query-gene mapping. PCR plate variance is a technical proxy, not donor reproducibility.
- Gene audit initially assumed the checkpoint was 512-output. Inspection of full frozen outputs showed 15009 native genes, so the audit was corrected and common-gene reaggregation was started.
- The single OOF-temperature candidate failed its fixed gate and was stopped without variants.


## 2026-10-01: common-axis precision and sampling audit

- Official feedback comparisons initially used floating RMSE as part of the task join key. CSV/Parquet last-bit differences left only 10 of 212 tasks in some new comparisons. The repair joins biological identifiers, asserts equivalent truth, and rejects missing paired coverage. Seven invariant tests pass. All saved scores and original invalid statistics remain unchanged; `PAIRED_BOOTSTRAP_REPAIRED.csv` is authoritative for these common-axis adaptations. No learner was refitted. Earlier native-axis statistics already used all 212 tasks.
- Source content controls exposed float32 CSV formatting differences up to 4.92e-9. All full context error rank orders are identical. A dedicated alignment audit maps these derived control labels to the existing primary matrix for statistics; no primary label or CDF/risk score was changed.
- Common-axis source HGB underperforms weighted history distance externally: delta -0.034419, paired 95% CI [-0.132144,-0.008644]. The support-only rule also exceeds history distance by 0.097457 (inverse comparison CI [-0.162306,-0.031880]). Preserve these failures.
- Test cell counts correlate with observed McFaline RMSE at rho approximately -0.96 to -0.98. Guide-block ranks are unstable; within-guide cell halves retain the count-driven rankings. Five fixed 20-cell truth diagnoses remove the near-positive U20 for history/support/HGB. These are retrospective sensitivity checks, not a new primary metric, noiseless ground truth, or method selection. The observed external risk signal cannot yet be described as learning biological/model-specific failure independently of measurement precision.
- The same count association is not universal in TxPert: approximately -0.02 in K562/hepg2 and -0.38 to -0.50 in Jurkat/RPE1. Do not attribute all existing within-family results to the McFaline mechanism.
- Literature audit found prior general cross-model failure prediction and PertEMA noise/co-failure analyses. A broad first-of-kind or generic noise-boundary claim is disallowed. The experimental goal remains active.


## Existing-validation strong control and freeze retention

- Completed the preplanned practical alternative of fitting C directly on existing validation predictions. Excluding all test gene clusters leaves 230 validation records / 123 clusters. PublicValidation HGB U20=0.755390 vs Shared=0.695253, difference CI [-0.017049,0.163774]; no established superiority. Adding Shared gives delta=-0.016076, CI [-0.030772,0.041165]. No additional model calls were required. Upstream validation selection bias is disclosed, so these labels are not called pristine OOF.
- The new common-axis full prediction CSV was filled with test truth after its freeze hash was registered. The pretruth bytes were not retained. This cannot be repaired retroactively into prospective proof; the run is already SEEN. Future isolated reproductions preserve a separate score-only snapshot and reject accidental overwriting of completed result versions. Original September freeze bytes verified unchanged.
