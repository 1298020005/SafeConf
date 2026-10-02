# Source scale augmentation: feasibility and scientific value

**Verdict: one fixed Source DEV authorization proposal is defensible as a limited test of synthetic scale robustness. It is not ready to run under the earlier 20-fit approval, and it cannot establish a new upstream family, external transfer, or the cause of a prior failure.** The useful distinction is that the earlier scale stress evaluated fitted Source rows, whereas this proposal would hold out genes, transfer between GAT and Exphormer, and hold out the evaluation scale from fitting. This tests whether adding scale coverage helps beyond copying the same training rows. It is sufficiently distinct from the failed raw-affine label intervention to justify one bounded proposal, provided the choices below are fixed in advance. A negative or inconsistent primary result should close this augmentation direction without trying more scale grids.

This review read repository code, Source manifests, NPY headers, parquet footer schemas, and the existing Source objective decision. It did not read vector values or parquet rows, perform fits, access external numeric data, modify a core, or implement a runner.

## What the existing assets support

The three Source files below exist under `/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis`. Header-only inspection confirms **1808 × 2840, float32, Fortran order** for each, with numeric payload starting after the 128-byte header:

- `SOURCE_TRUE_EFFECTS.npy`
- `SOURCE_TxPert_GAT_PREDICTED_EFFECTS.npy`
- `SOURCE_TxPert_Exphormer_PREDICTED_EFFECTS.npy`

`SOURCE_TASKS.csv` also exists. The two inspected fold-0 Learned parquets each declare 1808 rows, string task/gene/CDF identities and integer folds; cached error and prediction magnitude are float32, while prior magnitude, distance, cosine and uncertainty are float64. No vector values or task rows were inspected, so exact row/gene-axis alignment and input SHA binding remain prerequisites for an authorized run. The preparation code explicitly projects these existing Source vectors and saves their task order ([vector preparation](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/prepare_safeconf_common_gene_predictions.py:115)).

The common2840 `risk_cache` has nested scalar features but **no persisted nested prior-vector files**. Its builder creates priors in memory and saves feature frames ([nested cache construction](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_safeconf_research_closure.py:141)). The separate `orion_source_core_20261002_v1` has 1808 × 3285 float64 Source vectors and all five nested Learned/Manual prior vectors, explicitly saved by [Source preparation](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/prepare_safeconf_orion_source_core_agent.py:167). Those are a different frozen axis/reference contract. They cannot supply missing priors for the common2840 cache.

For a continuation of the completed objective probe, use the common2840 Learned cache and disclose the scalar reconstruction limitation below. If the intended question instead follows the earlier 3285 Manual scale stress, register that complete Source contract explicitly. **Choose one axis and one prior reference before approval; do not mix or try both.** The old formal 3285 core remains unchanged either way.

## Features can be updated without Public fits

Let `p` be the existing predicted DELTA, `h` the fixed fold-specific prior, `n` the gene count, `a = RMS(p)`, `b = RMS(h)`, `d = RMSE(p,h)`, and `g = mean(p*h) = (a²+b²-d²)/2`. Only `p` changes to `λp`.

| Existing feature | At nonnegative scale λ |
| --- | --- |
| `predicted_magnitude` | `λa` |
| `prediction_abs_mean` | `λ mean(abs(p))` |
| `prediction_signed_mean` | `λ mean(p)` |
| `prediction_std` | `λ std(p)`, `ddof=0` |
| `prediction_abs_q95` | `λ quantile(abs(p),0.95)` |
| `prediction_sparsity` | `mean(abs(λp) <= 1e-8)`; equals 1 at λ=0 |
| `prior_magnitude` | `b`, unchanged |
| `prediction_prior_rmse` | `sqrt(λ²a²+b²−2λg)` |
| `prediction_prior_cosine` | Original cosine only while `λ norm(p) norm(h) > 1e-12`; otherwise 0 |
| `prior_uncertainty` | unchanged |
| `log_history_support` | unchanged |
| `effective_sources` | unchanged |
| `history_conflict` | unchanged |

The definitions are in [prediction features](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/seal_mcfaline_dual_memory_risk.py:139), [cosine cutoff](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_dual_memory_txpert_public_biology.py:99), and [prior feature construction](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_safeconf_research_closure.py:109). At λ=0 the first five features are zero, sparsity is one, prior RMSE is `b`, and cosine is zero.

The distance identity is also `dλ² = λd² + (λ²−λ)a² + (1−λ)b²`; therefore full prior vectors are unnecessary for an analytic reconstruction from scalar summaries. **This identity is exact in real arithmetic, not a guarantee of NumPy bit equality from rounded cache values.** In particular, common2840 prediction RMS was reduced in float32 and the cached prior quantities are float64. Exact sparsity at a new positive scale cannot be inferred from original sparsity/q95: the threshold becomes `1e-8/λ`, so existing prediction vectors are necessary. The cosine cutoff must be retained; treating every positive λ as invariant is incorrect near the cutoff. Use the untouched λ=1 cached features for exact baseline replay. New-scale reconstruction precision and any roundoff clipping must be fixed and audited, rather than silently called the original vector calculation. If original vector-level feature equality is mandatory, the complete 3285 prior assets offer the clean path on their own contract.

Public retrieval need not change. Its nine pair features use historical effects, controls and support, without the upstream prediction vector ([pair feature construction](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_dual_memory_txpert_public_biology.py:174)). Holding Source truth, history, eligibility, controls, folds and existing retrieval scores fixed keeps Learned weights and priors fixed ([prior weighting](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_dual_memory_txpert_public_biology.py:227)). Scaling `p` alone calls for **zero new Public fits**. Rebuilding Public would introduce 25 nested fits and exceed the proposed scope unnecessarily.

Synthetic errors must be recomputed as `RMSE(λp,y)` using existing Source truth `y`. Their square is `λ² mean(p²)+mean(y²)−2λ mean(p*y)`. Original cached error and prediction magnitude alone do not determine this curve; the Source truth vectors are sufficient. Held-out query errors may be used for evaluation after score freezing, never to set CDFs, training weights, preprocessing, or scale choices.

## The one comparison worth proposing

Freeze train scales `{0, 0.25, 1}`, query scales `{0.1, 1}`, seed `20260930`, both predictor directions, the existing five gene folds, one chosen prior contract, the same 13 numerical features, and the fixed HGB recipe. All contexts, predictors and synthetic variants of a gene stay in its original fold. Scale and model identity are metadata for label grouping/audits; neither becomes a new numerical input.

The three arms would be original λ=1 rows; three exact copies of λ=1 rows; and the three synthetic scale variants. The duplicate arm copies the original training labels. The augmentation arm fits the existing mid-rank CDF helper separately for each Source/synthetic-λ variant and context, using only outer-training errors. This requires a fixed synthetic variant identifier in CDF metadata and a separate exception for the training/label flow, despite leaving the CDF algorithm unchanged. No CDF uses held-out genes or λ=0.1 evaluation errors.

**Per-gene total loss weight must match across arms.** `research.cluster_weights` normalizes mean weight to one ([implementation](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/safeconf_continual/research.py:89)). Applied directly to three copies/variants, it triples each gene's total weight and changes the relative effect of L2. The expanded arms need their weights divided by three, equivalently scaled to the original fold's total weight. The existing [scaled fitting helper](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_safeconf_source_row_weight_diagnostic_agent.py:57) already expresses this correction without changing the HGB recipe or core.

Matched loss weight does not remove all differences: triplication changes unweighted row counts used by the fixed leaf minimum; synthetic feature distributions change histogram bins, available splits, and training preprocessing; per-variant CDFs change the pooled label relationship. The duplicate arm checks some row-count effects, not every bin/preprocessing/label confound. Any observed benefit would belong to this complete augmentation procedure, not prove a unique causal explanation.

Primary reporting should compare augmentation with **both original and duplicate arms** for U20 at λ=0.1, in both directions and every context/fold, alongside the unchanged λ=1 view and fixed Magnitude/DirectRMSE/WeightedHistoryDistance rules. Report all planned results. An improvement over original that vanishes against duplication does not support the intended augmentation explanation. Improvement only under artificial scaling does not support native performance improvement. The query scale 0.1 was already examined in the earlier in-sample stress, so it is held out from this fit grid but is not a fresh confirmation condition.

## Cost, approval and stopping boundary

The ceiling is **30 HGB fits**: 2 directions × 5 folds × 3 arms, each predicting two query views; at most 6,000 boosting iterations. Expanded fits have three times the original fold rows. With 1808 tasks per predictor, the complete plan processes 101,248 training-row instances across fits (`2 × 4 × 1808 × (1+3+3)`), without adding biological tasks or Source families. The three vector payloads total about 62 MB on common2840; no downloads, upstream training, Public refits, external evaluation, tuning or GPU are needed. The recent original-feature diagnostic completed its 19 new small fits in a few seconds, but augmentation runtime is unmeasured. A fixed single-CPU 20-minute run-phase ceiling and bounded memory allocation would be conservative authorization limits, not a measured forecast.

A concrete approval should bind one script/axis/prior contract, the exact Source input paths and SHAs, the fixed scale grid/arms/folds/seed, 30-fit ceiling, original per-gene loss budget, zero Public/external operations, fresh isolated outputs, unchanged frozen core, original λ=1 replay, and all models/scores frozen before metrics. No such runner or approval is created by this review. The previous approval covered the completed 20-fit objective probe only.

This diagnostic could reject or provisionally support a specific scale-coverage intervention. It remains Source DEV evidence drawn from two variants of the existing family and repeatedly inspected biological tasks. A positive result would still require an independent confirmation design before any generalization or formal-method claim; it is not grounds for immediately rescoring a known external population. That limit does not erase the value of one falsifiable Source experiment, but it makes repeated grids without new information unproductive.
