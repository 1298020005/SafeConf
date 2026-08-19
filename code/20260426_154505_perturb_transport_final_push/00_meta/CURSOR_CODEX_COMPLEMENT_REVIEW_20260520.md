# Cursor / Codex complement review

Time: 2026-05-20 CST

## What Cursor added

- `00_meta/Q1_TOP_Q2_MASTER_STANDARD.md`
  - Useful. It turns the publication goal into explicit pass/fail criteria.
- `03_code/evaluate_q1_readiness.py`
  - Useful. It gives a repeatable readiness label instead of judging by feeling.
  - I fixed two points:
    - added optional `--primary-model` for ablation/probe scoring;
    - made Q1 require an actual `ContextSimBaseline`, not silently pass when it is missing.
- `transport_models.ContextSimilarityBaseline`
  - Useful reviewer-facing comparator: answers whether simple context similarity is enough.
- `03_code/run_q1_cpu_master_20260520.sh`
  - Useful wide CPU evidence run. It is currently running under `46_q1_cpu_push_20260520/`.
- `00_meta/CODEX_CURSOR_HANDOFF.md`
  - Useful coordination note. Main line is fixed as `PolicySafeTransPT`; GPU/probe methods should not steal the main story until they pass the evaluator.

## Current running work

- `gpu_graft_main_20260520`
  - New GPU/probe run with `EffectBlendV2` and `TopRankGraftV2`.
- `gpu_graft_tian_20260520`
  - External `TianKampmann2019` run with the same graft/probe models.
- CPU process:
  - `46_q1_cpu_push_20260520`
  - Adds `ContextSimBaseline` and broad PolicySafeTrans evidence.

## How to combine the work

- Cursor's work should be the evaluation and publication gate.
- Codex's new `EffectBlendV2` / `TopRankGraftV2` should remain an ablation/probe for now.
- If graft/probe models keep improving effect metrics without damaging RMSE too much, fold the idea back into `PolicySafeTransPT` as a routing/expert option, not as a new paper direction.
- If `ContextSimBaseline` is strong, the paper must argue safety/routing/abstention, not just context similarity.

## Immediate next checks

1. Wait for `46_q1_cpu_push_20260520/results/Q1_READINESS_REPORT.json`.
2. Score `43_gpu_effect_objective_main_20260520` and `48_gpu_graft_tian_20260520` with probe overrides:
   - `--primary-model TopRankGraftV2`
   - `--primary-model EffectBlendV2`
3. Build `GEARS_HEAD_TO_HEAD.md` after current runs settle.
4. Add a real pathway prior only if it can be wired cleanly; do not fake Reactome/GO evidence.
