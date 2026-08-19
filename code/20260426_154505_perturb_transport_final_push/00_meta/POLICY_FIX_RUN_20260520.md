# PolicySafeTransPT Fix Run - 2026-05-20

## Why this rerun exists

Opus7/Codex strict review found two core failures:

1. `PolicySafeTransPT` did not reliably beat the strong `V0` baseline on held-out perturbation settings.
2. The previous `transportability_score` behaved too much like router confidence, not like calibrated prediction risk.

## Code changes

- `transport_models.RidgeRegressor` now fits an intercept by centering the target before solving ridge regression and adding the target mean back at prediction time.
- `PolicySafeTransPT` now routes among internal experts: `V0`, `V1`, `V2`, `Safe`, `Network`, and `ContextSim`.
- `PolicySafeTransPT` now learns out-of-fold expert utility and RMSE predictors.
- `transportability_score` is now a calibrated risk score combining predicted RMSE, task support, context similarity, perturbation consistency, expert agreement, and retrieval confidence.
- `unsafe_flag` now uses a calibrated threshold plus per-split low-confidence quantile, so the abstention branch is testable rather than dormant.

## Background sessions

- tmux session: `safetrans_policy_fix_20260520`
- CPU window: `cpu_policy`
- GPU0 window: `gpu_external`
- GPU1 window: `gpu_main`

## Log paths

- CPU policy log: `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/51_policy_calibrated_q1_20260520/logs/run_safety.log`
- GPU main log: `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/52_gpu_policy_fix_main_20260520/logs/run_gpu_main.log`
- GPU external log: `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/53_gpu_policy_fix_external_20260520/logs/run_gpu_external.log`

## How to check

```bash
tmux attach -t safetrans_policy_fix_20260520
tail -f /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/51_policy_calibrated_q1_20260520/logs/run_safety.log
tail -f /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/52_gpu_policy_fix_main_20260520/logs/run_gpu_main.log
tail -f /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/53_gpu_policy_fix_external_20260520/logs/run_gpu_external.log
```

## Expected timing

- CPU policy rerun: several hours.
- GPU main/external reruns: overnight scale, depending on dataset size.
- First interpretable checkpoint: when each `results/*SUMMARY*.csv` appears.
