# Plan and ETA

Generated: 2026-04-26 15:45 CST.

1. GPU situation: two Quadro RTX 6000 GPUs, 24GB each, idle at audit time. CUDA is visible through `scgpt_env` (`torch.cuda.is_available() == True`, device count 2).
2. Existing data/code assets: found local scPerturBench/scPerturb H5AD atlas with verified genetic RNA perturbation datasets and context-generalization files; found scGPT/AttnRes historical assets; did not find a dedicated GEARS repository, so the gate uses a strong same-perturbation/context residual baseline plus control heuristic as V0.
3. Fast gate expected runtime: 6-12 hours from tmux launch. This is shorter than the generic 24-48h estimate because the needed H5AD files already exist locally and the first gate is effect/program-level rather than full cell-level neural training. Conservative latest gate decision: 2026-04-27 04:00 CST.
4. Full run expected runtime: only if gate passes, 3-5 days on this server for V0-V3 over 4-6 settings and extra seeds. Conservative full completion window: 2026-04-29 to 2026-05-01 CST.
5. If the user disconnects: the experiment runs inside tmux session `perturb_transport_gate`; server-side process continues without Codex or SSH staying attached.
6. tmux/session name: `perturb_transport_gate`.
7. Log paths: `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/tmux_pipeline.log`, `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/gate.log`, `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/full.log`, `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/finalize.log`.
8. Expected final gate conclusion: by 2026-04-27 04:00 CST unless an earlier explicit gate failure occurs. If gate fails, final zip is produced immediately with `NOT_Q2_READY_STOP`; if gate passes, full run starts automatically.
