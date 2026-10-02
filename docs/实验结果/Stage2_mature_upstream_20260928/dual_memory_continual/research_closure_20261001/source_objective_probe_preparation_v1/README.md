# Source DEV label-objective probe preparation

Status: **PREPARED, NOT RUN**. Only the synthetic self-check has run. No real Source parquet, main result CSV, target/external truth, download, or actual fit was used during preparation. The frozen method, core helpers, and existing results were not edited.

The hypothesis is a possible objective mismatch, not an established root cause. Squared loss on ECDF-rank labels estimates a conditional mean of ranks; selecting a fixed top 20% for U20 rewards a conditional mean of raw error. A nonlinear rank transform can reverse these preferences under heteroskedasticity. The toy self-check has group A errors `[0,0,0,0,0,0,0,0,0,100]` and group B errors `[2,2,2,2,2,2,2,2,2,2]`: raw means are `10` versus `2`, while the combined mid-rank means are `0.3` versus `0.7`. This establishes mathematical possibility only.

The isolated script is `tools/scripts/probe_safeconf_source_label_objective.py`, SHA256 **`f77968fb5537c850b9812e58cd8d69eda904255babcec1a210eeddd3185be496`**. It reuses `research.rank_labels`, `fit_risk`, `cluster_weights`, `metrics`, `summarize`, and `SEEDS[0]`.

## Fixed scope and cost

- Source DEV nested GAT→Exphormer and Exphormer→GAT; five existing gene folds; seed `20260930`.
- Two label arms: current training-only ranks, and positive-affine raw training errors whose weighted mean/std match the rank labels separately within existing `CDF_KEYS`.
- Exactly the same 13 `P + PUBLIC` features, training rows, preprocessing, gene-cluster weights, and HGB recipe: 200 iterations, learning rate 0.05, depth 3, leaf minimum 20, L2 10.
- At most **20 fits**: 2 arms × 2 directions × 5 folds; at most 4,000 HGB boosting iterations; one CPU thread; no upstream training, bootstrap, tuning, downloads, or model selection.
- Three fixed references add zero fits: Magnitude, Learned DirectRMSE, and Learned WeightedHistoryDistance `sqrt(prediction_prior_rmse² + prior_uncertainty²)`.
- A **1,200-second alarm** bounds the real read/fit/evaluation phase. Input hashing and final SHA/status recording add overhead outside that phase; a successful runtime estimate is unavailable because no real fit has run.
- Both arms use `predict(..., clip=False)`. The original `[0,1]`-clipped rank reference is recorded separately for replay. Affine labels can fall outside `[0,1]`; these unbounded predictions are diagnostic values and do not become a new formal risk score.

The transforms use only outer-training errors. Cached Source query truth is already DEV and is allowed for metrics only after explicit approval. No query/evaluation-group errors enter label transforms, preprocessing, weights, or fitting. Matching moments separately by context changes effective weighting of raw squared loss across pooled CDF groups; this probe is not a proof that pooled HGB estimates one globally comparable raw-error conditional mean.

## Input and replay gates

The exact parquet whitelist is the Cartesian product below, all under `/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache`:

`nested_{0,1,2,3,4}_{TxPert_GAT,TxPert_Exphormer}_{Learned,Manual}.parquet`

That is **20 unique Source files**. Ten Learned files provide fit features; ten corresponding Manual files are read only for exact task ID/order, gene/fold, CDF-key, and cached-error identity verification. Each file is loaded once. Symlink/path substitutions are refused.

The only CSV reference is the fixed `common_gene_axis/results/MATRIX_TASK_PREDICTIONS.csv.gz` in the existing closure docs. It contains external rows. The streaming reader discards every row unless string metadata matches a Source direction, seed `20260930`, and method `Learned_hgb`, **before converting any risk, error, or fold value to a number**. Only these Source rows enter replay. Whole-file SHA hashing does not interpret numeric content. The rank arm must reproduce each existing clipped Source fold score within `1e-14` and match task/error identity before that fold's raw arm runs. A failed replay stops the diagnostic; there is no fallback to another method.

SHA256 is recorded before and after for all 20 Source inputs, the fixed CSV reference, the script, and three unchanged core helper files. Any mutation fails completion. Outputs go only to a fresh absolute child of `/home/yyf/runtime_artifacts/safeconf_research_20261001/source_objective_probe_v1`; an existing directory, including a completed run, is refused.

## Explicit approval requirement

Changing the frozen label rule requires the user's explicit exception before any real Source read or fit. The preparation does not grant that exception or create an approval token.

The supplied approval JSON must record the user's explicit authorization reference and contain all of the following exact fields. The script checks these before hashing/reading Source inputs or fitting:

- `status`: `USER_APPROVED_SOURCE_DEV_OBJECTIVE_PROBE`.
- `script_sha256`: the script SHA above; any code edit invalidates the approval.
- `scope`: exactly the script's `SCOPE` constant, including 20 Source files, two arms, both directions, five folds, same features/seed/weights, unbounded predictions, three zero-fit rules, model saving, score freezing, and no bootstrap/search.
- `max_fits`: integer `20`; `core_unchanged`: boolean `true`; `no_external`: boolean `true`.
- `output_dir`: the exact fresh runtime child selected for this run.
- `source_input_paths`: the exact ordered `SOURCE_PATHS` array; `source_reference_csv`: the exact `ARCHIVE` path.
- `user_approval_reference`: a nonempty reference to the user's explicit label-rule exception. A generated configuration file is not authorization.

After that authorization exists, the bounded invocation is:

```bash
python -B tools/scripts/probe_safeconf_source_label_objective.py \
  --approval /absolute/path/to/user-authorized-approval.json \
  --output /home/yyf/runtime_artifacts/safeconf_research_20261001/source_objective_probe_v1/approved_run_id
```

Each fitted preprocessor/model is saved with joblib in the isolated runtime output. After all fold scores are saved, `SCORE_FREEZE.json` binds the prediction file and every model SHA before `summarize` or `metrics` runs. The first-stage outputs are per-fold/context and per-context/line point metrics (U20, Spearman, AURC, retention errors), training transform/CDF audits, prediction tails and clipping/tie counts, exact clipped-current-score replay, fit costs, and input SHA/status manifests. They are descriptive DEV diagnostics. No root-cause, superiority, uncertainty, or new confirmation claim is authorized. A later inferential comparison would need a separately fixed saved-count CI plan after exact replay and all fold scores exist.

## Synthetic self-check evidence

Executed from the repository root:

```bash
python -B tools/scripts/probe_safeconf_source_label_objective.py --self-check
```

Result: `SELF_CHECK_PASS`; raw means `[10.0, 2.0]`; rank means `[0.3, 0.7]`; positive affine slope `0.011838732170259688`; weighted moments matched. Missing approval was rejected while Source readers, archive reader, fit helper, and SHA reader were patched to fail if reached: **0 Source read attempts, 0 actual fits, 0 files written**. This path does not fabricate the approved status token.
