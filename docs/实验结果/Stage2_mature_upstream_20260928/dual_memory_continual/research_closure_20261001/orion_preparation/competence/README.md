# Frozen Orion validation competence

`tools/scripts/evaluate_safeconf_orion_competence_agent.py` evaluates the one published linear family after both context fits and pretruth predictions are saved. This preparation has executed only synthetic tests; it has not evaluated actual Orion validation effects or run SafeConf scoring.

Inputs are the finalized R fit root, authorized biology root, blind bridge root, scientific method contract, full pipeline v2 permit and pre-frozen VALIDATION_ELIGIBILITY.tsv. Each must refer to the same biology/permit/method lineage. Artifact hashes, fitted model hashes, locked PCA10/ridge0.1/seed1, endpoint order, own-context TRAIN NTC reference and biological query identities are checked. Exactly HCT116 and HEK293T are required. These are two context models in one family attempt; they neither add three independent models nor meet the older three-background requirement. Measurement Quality remains proxy/biological_replicate_unknown.

Only eligible VALIDATION rows of ENDPOINT3285_EFFECTS.npy are indexed numerically. The frozen PREDICTIONS_DELTA.tsv.gz files are projected to validation query columns; TEST prediction columns are not converted to floats. Raw files, TEST truth, TRAIN effect rows, Source error vectors and SafeConf scores are unnecessary. Hashing a complete registered artifact reads opaque bytes and is distinct from indexing its numeric rows.

EXTRACT_FROZEN_BASELINE.R reads the persisted model and SIMPLE_BASELINES.rds. It verifies the original model/input receipt, model defaults, equality with the baseline saved in model provenance, endpoint IDs, and BASELINE_SELECTION.tsv. The selected baseline must equal the exact minimum of the stored TRAIN grouped OOF errors. It extracts only that candidate as little-endian float64. No baseline is chosen using VALIDATION errors or inside bootstrap draws.

## Registered calculation

- Task error is float64 RMSE over the 3,285 signed CP4000 effect coordinates. Predictions are parsed with round_trip precision from the frozen R TSV representation; derived errors are saved with17 digits. No scaling, clipping, fitted CDF or prediction adjustment occurs.
- Compute each context's mean task RMSE, then average the two context means equally. The reported relative gap is the ratio of the model and selected-baseline context macros minus1. It is neither a pooled-row mean nor an average of context relative gaps.
- Model context macro must be at most1.02 times the baseline macro, and at least60% of the two context strata must be noninferior. With two contexts, this requires2/2.
- Use5,000 paired gene-cluster draws with numpy default_rng seed20260929 and sorted Ensembl cluster IDs. Each draw samples the union gene set with replacement. The same multiplicities weight both methods and both contexts; each context divides by its available sampled task count. No missing task is inserted as zero.
- Two-sided95% percentile bounds use .025/.975 and numpy linear quantiles. A lower bootstrap ratio above1.02 is stable disadvantage. Ratios are compared before subtracting1 to handle the exact2% floating-point boundary without adding an epsilon.
- Missing/nonfinite eligible predictions are explicit failures; eligible tasks cannot silently disappear. A nonpositive baseline error or an empty context in a bootstrap draw is invalid/inconclusive. Denominators are not floored and draws are not repeated.

A pass is the registered operational competence screen. A lower bound below2% does not establish statistical noninferiority when the upper bound is above2%. Bootstrap uncertainty is conditional on the fitted models and the fixed truth/control estimates; it does not invent biological replicates or additional backgrounds.

## CLI and outputs

Required flags: `--fit-root`, `--biology-root`, `--bridge-root`, `--permit`, `--contract`, `--validation-eligibility`, `--output`. An actual run also requires `--expected-evaluator-sha256`, `--expected-baseline-helper-sha256` and `--expected-validation-eligibility-sha256`, fixed before truth. Output must be a new version directory. The R executable may be provided through `--rscript`.

Outputs are COMPETENCE_RESULT.json; VALIDATION_TASK_ERRORS.csv with gene/context/query IDs, exact vector hashes and selected-baseline errors; CONTEXT_COMPETENCE.csv; all5,000 PAIRED_GENE_BOOTSTRAP.csv draws with gene-draw hashes; INPUT_IDENTITIES.json; C_VALIDATION_ACCESS_LEDGER.json; extracted selected baseline files; and ARTIFACT_HASHES.json.

The C_VALIDATION ledger permits only fixed upstream competence. It records no TEST truth reads, no TEST prediction conversion, no Source/risk parameter changes and no new training attempt. These errors cannot train risk/CDF/core parameters, select SafeConf models, reselect the baseline, or fit prediction scaling/calibration.

## Synthetic verification

VERIFY_COMPETENCE_STATISTICS.py independently verifies16 scenarios, including every draw of seven valid scenarios against an expanded-task oracle, exact RNG hashes, equal-context weighting versus pooled rows, shared-gene cancellation, the2/2 gate, exact margin boundaries, sparse-context failures and deterministic row-order invariant draws.

VERIFY_COMPETENCE_PIPELINE.py creates runtime-only synthetic biology, bridge, fits and validation effects. Four task RMSEs match injected errors; forbidden TRAIN/NTC effect-row sentinels remain unread. Two guards reject changing the frozen TRAIN baseline choice or rebinding a fitted model to another receipt. Additional actual-execution guards reject missing code hashes and changed eligibility hashes before any numeric validation row is read. The evaluator fits no parameters; the fixture's two synthetic context fits belong to its test setup.

Large artifacts are under `/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_competence_pipeline_synthetic_20261002_v1/`. Independent statistics evidence is under `orion_competence_statistics_synthetic_20261002_v2/` in the same runtime parent. COMPETENCE_PREPARATION_STATUS.json records frozen code hashes and the queued-call contract. The R mathematical core, blind bridge and LM CLI remain unchanged.
