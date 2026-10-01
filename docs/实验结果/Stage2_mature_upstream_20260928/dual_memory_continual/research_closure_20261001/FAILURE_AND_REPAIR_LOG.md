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

## Fixed control-scale diagnostic: rejected

- Hypothesis: source/target absolute amplitude scales explain cross-study risk failure.
- One registered rule scales amplitude features by training-control RMS; it does not refit CDFs, choose a scale on query truth, or change learners/primary cohorts.
- External Learned HGB falls from 0.695253 to 0.491130; delta -0.204123, 5000 paired-cluster CI [-0.284540,-0.070207]. Internal deltas are small and uncertain.
- Stop this repair. It rejects this particular normalization rule, not every possible scale/domain mechanism. No new confirmed method is selected.

## Guide-equal vs cell-equal public-reference estimand

- Independent review located a concrete estimand difference: within-history guide-equal averages vs cell-equal query truth. Historical record support weighting does not resolve within-record guide imbalance.
- A zero-fit diagnostic reconstructs only allowed train/validation treated cell means, preserving original controls, eligibility, inter-record weights, predictions, primary errors and all 543 queries.
- Fixed Manual weighted-history distance improves from 0.711295 to 0.834314; delta 0.123019, CI [0.056150,0.198610]. Against the strong support-only rule the delta is 0.007185, CI [-0.039087,0.055330].
- Thus the mismatch explains some observed loss, but biological effect content has not been shown superior to support. The canonical frozen memory and earlier confirmation are not overwritten. The alternate reference is diagnostic, not automatically a final candidate.
- One necessary follow-up uses the five previously defined support-matched effect permutations on the cell-weighted reference, with zero fitting. No temperature/learner search is added.

- The completed five fixed content permutations score 0.602882–0.678152; real cell reference exceeds every null with positive paired 95% CIs. This supports the role of actual content/correspondence for this rule, not independent predictive gain beyond support.
- A precise replay of the existing Source Manual HGB reproduces all original scores before any query change. After updating only the query prior estimand, U20 is 0.575302, vs original 0.648814 and cell historical distance 0.834314. Delta vs cell distance is -0.259012, CI [-0.371281,-0.127813]. Thus this one correction does not rescue source transfer. Stop this diagnostic's method variants.

## Independent Source truth and sampling audit

- Single-pass released vectors and metadata cover 354208 cells, including 187628 primary cells for all 1808 tasks. Full released NPY file SHA256 and cell-stream SHA256 match their registered truth seals; all primary centroids and common-axis effects reproduce exactly using the original float32 conversion rule. The large original H5AD was independently spot-checked on 128 cells, not fully rehashed.
- Conditional iid treated-cell sampling MSE is about 12%–30% of observed mean MSE depending on context/model; the observed batch-cluster calculation is similar. Controls are fixed and their uncertainty is excluded; guide identities and independent biological replicate guarantees are absent.
- Do not call the remainder a proved latent biological error component. Do not generalize the McFaline sampling pattern to every Source domain. All fixed-count diagnostics remain retrospective sensitivities, not replacement primary metrics.


## Orion authorized reader execution timeout, 2026-10-02

- Actualv1jobPID162145 exited1 after1497.943seconds. FullpipelineSTATUS and traceback confirm per-file600s time-bound at loader line385, not a stale polling timeout. HCT116_Batch1 finished; HCT116_Batch10 token spool is607,513,264bytes. No test numeric materialization or model fitting occurred.
- Hypothesis: the per-value Python decoder dominates large lists. Independent fixed synthetic SNAPPY workload1000cells×5000values×2columns showed v1=15.407seconds and vectorized byte-selectionv2=0.727seconds (21.2×). This is a synthetic measured speedup, not yet wholepipeline timing.
- One technical repair: vectorize structural row routing and copy ONLYauthorized8-byte payload/dictionary blocks before INT64/DOUBLE typing. The version1code, failedstage and Source/upper/math/evaluation/scientificcontract hashes remain preserved. No algorithm, labels, split, normalization or test boundary is tuned.
- Newbackend/wrapper will require independentprivacy/normalization tests and a separate hash-bound permit/outputversion before restart. Model gates remainpending.

### Registered source-diversity seed completion

The earlier diversity summaries used only seed20260930. The fixed common2840 comparison was replayed with all three registered seeds in a separate output, without touching frozen source models or Orion.24 fits took5.873s; first-seed scores match the released scores within9.72e-17; maximum score range across seeds is0. This completes the seed reporting requirement and supplies no new independent observations or positive method claim.

### Before-test evaluation contract correction

Independent review found that bootstrap duplicates could yield finite intervals for an original context with fewerthan20 tasks. Fixed original cohort validity now propagates through every draw and macro. The evaluator also rejects generic completion receipts, checks the exact guarded TEST operation/reader/backend and source/model hashes, and the TEST reader checks the frozen evaluator code before raw access. Sixteen generated reader cases, hand metrics and a joint prepare→guard→evaluate fixture passed. No actual TEST labels were read. The coordinator additionally reserves the registered test scope exclusively across output directories after all actual hash gates and before raw reading.

### Source rows versus new source information

Mean-normalized cluster weights change total weighted loss when record count doubles. A fixed, information-free copied-record and matched-total-weight diagnostic was executed without changing any original model or Orion parameter. Twelve fits, six original reproductions,5000 paired gene draws and an independent review completed. No uniform gain from copies was found; the best copied Exphormer contrast remains uncertain. No parameter sweep or method promotion follows this diagnostic.

### Missing uncertainty and lifecycle evidence

A completion audit found point-only Source scaling/diversity results. Existing saved predictions were reused for5000jointgenecluster draws across3contexts, all budgets/orders and7metrics;19.44s, no refits/newlabels/upstream calls. Original150scaling/30diversity points reproduced within1.67e-16. Static Public growth and error-budget fits did not execute a real publish/rollback cycle; SYS-A readiness is corrected to partial and a separate realDEV operational replay is underway. Frozen Orion methods remain unchanged.
