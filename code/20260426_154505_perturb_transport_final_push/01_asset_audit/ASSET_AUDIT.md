# Asset audit

Direction is fixed to cross-context causal transport for single-cell perturbation effects.

Found local perturbation atlas: `/home/yyf/datasets/singlecell_perturbation_atlas`.

Key findings:

- Genetic/RNA perturbation datasets are present in scPerturBench/scPerturb atlas; `DATASET_INVENTORY.csv` records 83 scanned H5AD entries.
- At least 8 genetic RNA perturbation datasets are available, including Adamson, Norman, Replogle, Schmidt, TianActivation, TianInhibition, Wessels, Haber, Parekh, kangCrossCell/kangCrossPatient.
- Split registry: existing atlas metadata includes `iid_test` and split columns for several datasets; this run also implements explicit context/perturbation split construction in `03_code/build_context_splits.py`.
- Unified evaluator: implemented in `03_code/evaluators.py`.
- GEARS historical code/results: not found as a dedicated local GEARS repository; V0 uses a strong same-perturbation/context residual baseline rather than a weak null baseline.
- control_mean_heuristic: implemented as part of V0 baseline behavior.
- scGPT historical code/assets: found under `/home/yyf/codex_scgpt_attnres_workspace` and `/home/yyf/data/scGPT_immune`.
- AttnRes historical outputs: archived under `/home/yyf/codex_archive/20260426_154505_before_perturb_transport/home__yyf__codex_scgpt_attnres_outputs`; only historical/probe use is allowed.
- Perturbation atlas reports: found under `/home/yyf/datasets/singlecell_perturbation_atlas/reports`.
- Preferred conda env: `/home/yyf/.conda/envs/scgpt_env`.
- GPU status: two idle 24GB Quadro RTX 6000 cards at audit time.
