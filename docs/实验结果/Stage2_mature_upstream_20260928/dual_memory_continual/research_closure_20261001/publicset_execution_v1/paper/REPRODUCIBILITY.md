# Reproducing the manuscript package

These commands are run from `/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921`. They use existing result artifacts and local runtime paths. Model training commands are documented separately from the manuscript rebuild.

## Evidence refresh and PDF

```bash
python docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/paper/refresh_paper_evidence.py
python docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/paper/build_paper.py
```

Python needs pandas, numpy and matplotlib. The refresh copies completed table sources, then records their original path, row count and SHA-256 in `evidence/RESULT_MANIFEST.json`. The build renders `PAPER.md` into `main.tex`, creates three figures and checks citation keys. Neither command fits a model or reads new test truth. The build uses pdflatex plus BibTeX when installed, then Tectonic from PATH or `build/toolchain/tectonic`. Tectonic is the official 0.17.0 Linux release; its source URL, byte count and archive hash are in `build/toolchain/TOOLCHAIN.json`. Its bundle cache stays below `build/toolchain/cache`.

`build/BUILD_STATUS.json` is the binding record of compilation success or failure. A successful build writes `build/main.pdf`; compilation logs are `build/compile_*.log` and engine logs are retained. A prior PDF does not substitute for the current status record.

## Input contracts and upstream assets

The common-axis runtime root is `/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis`; Source public records are in `/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201`. The PublicSet run manifests bind the task tables, aligned effects, gene IDs and public-memory manifests. Source/McFaline use 2,840 ordered genes. Orion keeps its separately frozen 3,285-gene contract, and Frangieh retains native512. Raw errors and arrays across those contracts are not pooled or zero-filled.

The public records contain biological effects and provenance. Source error records are scoped to predictor/checkpoint. The Source public builder uses outer gene splits shared across contexts; a Source risk reader must additionally enforce the current inference-context teacher-exposure scope. McFaline public fitting excludes all outer evaluation experiment IDs globally, including their occurrence as another query's history. PCA fits only unique permitted fitting-history rows. Repeat training seeds are fitting sensitivity, not additional biological samples.

## PublicSet execution entry point

The implemented runner exposes the following completed-run entry point. `full_v2_resume` is the checked continuation of interrupted `full_v1`; its COMPLETE manifests preserve the reuse history. Completed checkpoints are reused with verified preprocessing/input bindings.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/yyf/.venvs/txpert-08d82eea/bin/python tools/scripts/run_safeconf_publicset_v1.py --phase full --run-id full_v2_resume --domains McFaline Source --folds 0 1 2 3 4 --seeds 20260930 20261001 20261002 --gpu 0 --gpu-hours-cap 12 --cpu-python /home/miniconda/bin/python --reuse-run full_v1 --reuse-run prototype_v2_cpu_environment_repair
```

This command is an execution command, not part of the paper-only rebuild. It needs PyTorch and the registered local arrays. Its declared role is DEV/SEEN_NEW_PUBLICSET. It performs no new upstream training and does not read McFaline test arrays. `TASK_PREDICTIONS.csv.gz`, `HISTORY_SET_AUDIT.csv`, `FIT_LEDGER.csv` and the run manifests expose predictions, history membership, fitted parameters and cost. The completed statistics consume this run and preserve the shared biological query folds.

The runner has a lightweight semantic entry point:

```bash
/home/yyf/.venvs/txpert-08d82eea/bin/python tools/scripts/run_safeconf_publicset_v1.py --phase semantic-tests
```

## Frangieh results and the score repair

The primary retrospective run is recorded under `frangieh_cross_family_v1`; it saves 120 risk fits and preserves original upstream fold boundaries. Reports can be regenerated from completed outputs:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/scripts/run_safeconf_frangieh_cross_family_v1.py --report-only
```

The one fixed zero-refit repair is implemented by:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/scripts/run_safeconf_frangieh_cross_family_v1.py --score-repair
```

The repair command requires the saved primary fits and refuses to overwrite completed repair predictions. For this completed branch, inspect `RidgeScoreRepair/README.md`, `MACRO_RESULTS.csv`, `PAIRED_BOOTSTRAP.csv`, `SCORE_REPAIR_LEDGER.csv` and `PROTECTED_ORIGINAL_HASHES.json`. The repaired version applies `0.5 + arctan(z - 0.5) / pi` without model fitting. It does not change the six failed upstream competence gates or the primary HGB tables.

## Evidence role and remaining author inputs

The historical Source/McFaline matrices are development/seen evidence. The original Orion primary comparison remains frozen external evidence with negative or uncertain outcomes; later Orion diagnostics are seen. Frangieh is seen native-axis stress evidence and fails the capability qualification. Current PublicSet statistics and nested-reader outputs should be imported only from completed, registered versions. Permanent evaluation truth does not select a release.

The paper imports completed registered results only. Author identities, declarations and the public release identifier are listed in `AUTHOR_PLACEHOLDERS.md`; the TCBB cover letter requires author completion.

## Fixed matching references and exact paired statistics

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python tools/scripts/run_safeconf_publicset_simple_references_v1.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python tools/scripts/analyze_safeconf_publicset_v1.py --run-dir docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/full_v2_resume --output-dir docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/statistics_v1 --replicates 5000 --threads 4 --max-seconds 600
```

The three fixed matching references are declared before metric computation. Statistics calculate each neural seed's metrics first, average within context-by-fold, and then macro-average strata. Source gene draws synchronize all upstreams and seeds. The semantic receipt tests integer gene multiplicities against sixty literal-repeat examples.

## Registered joint-reader version

The exact prediction-only contract is `registered_universal_v1`: RMS, mean absolute effect, signed mean, standard deviation, raw absolute 95th percentile and `mean(abs(effect) <= 1e-8)`. The quantile has no artificial floor, and sparsity is not rescaled by the task's magnitude. `TECHNICAL_REPAIR_RECEIPT.json` distinguishes this version from the earlier relative-sparsity implementation. Four inner gene folds create out-of-fold public training features under each shared outer fold. Genuine source labels come only from the opposite architecture in the inference context and outer training genes; the CDF uses exactly those source errors.

The runner phases are `nested-gpu`, `outer-gpu`, `cpu-risk`, `aggregate` and `statistics`. For the registered version the final commands have this form:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/yyf/.venvs/txpert-08d82eea/bin/python tools/scripts/run_safeconf_publicset_risk_followup_v1.py --phase nested-gpu --risk-version registered_universal_v1 --builders B0_SupportMean B1_HGB B2_Pointwise B3_DeepSets --folds 0 1 2 3 4 --seeds 20260930 20261001 20261002 --gpu 1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python tools/scripts/run_safeconf_publicset_risk_followup_v1.py --phase cpu-risk --risk-version registered_universal_v1 --builders B0_SupportMean B1_HGB B2_Pointwise B3_DeepSets --folds 0 1 2 3 4 --seeds 20260930 20261001 20261002
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python tools/scripts/run_safeconf_publicset_risk_followup_v1.py --phase aggregate --risk-version registered_universal_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python tools/scripts/run_safeconf_publicset_risk_followup_v1.py --phase statistics --risk-version registered_universal_v1 --bootstrap 5000
```

Outer biological models from the completed four-arm run are reused with binding checks. The three-way null keeps biological builder/readout, inputs, folds and error-label count fixed while changing genuine source errors to a cluster permutation or the registered biological-difficulty proxy. The source snapshots included under `repro/source/` bind the exact implementation. Large arrays are external prerequisites described in `DATA_ACCESS.md`; the submission ZIP is a paper/evidence/source package, not a self-contained training dataset.
