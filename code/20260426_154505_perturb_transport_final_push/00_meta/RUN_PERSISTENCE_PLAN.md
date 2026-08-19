# Run persistence plan

- tmux session: `perturb_transport_gate`
- Launch command: `tmux new-session -d -s perturb_transport_gate /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/03_code/run_background.sh`
- Main log: `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/tmux_pipeline.log`
- Gate log: `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/gate.log`
- Full log: `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/full.log`
- Finalize log: `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/finalize.log`
- Check logs: `tail -f /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs/tmux_pipeline.log`
- Reattach: `tmux attach -t perturb_transport_gate`
- Detach: Ctrl-b then d
- Resume behavior: existing result files are preserved; finalization can be rerun without deleting prior results. Gate/full runners append logs and overwrite result tables deterministically for the configured seeds.
## 2026-05-19 运行记录

- tmux session: `policy_router_20260519`
- 启动脚本: `03_code/run_policy_router_focus.sh`
- 输出目录: `29_policy_router_refresh_20260519/`
- 日志文件: `29_policy_router_refresh_20260519/logs/run_safety_abstention_evidence.log`
- 当前任务: `PolicySafeTransPT` 聚焦实验，先看路由 + 拒判是否比 `V2` / `NetworkSafeTransPT` 更稳

## 2026-05-19 扩展批次

- tmux session: `policy_router_wide_20260519`
- 启动脚本: `03_code/run_policy_router_wide.sh`
- 输出目录: `30_policy_router_wide_20260519/`
- 日志文件: `30_policy_router_wide_20260519/logs/run_safety_abstention_evidence.log`
- 当前任务: 更大范围的 main + external 队列批次，给明天晚上的汇报准备更宽的证据面

## 2026-05-19 总表批次

- tmux session: `policy_full_20260519`
- 启动脚本: `03_code/run_policy_full_wide.sh`
- 输出目录: `31_policy_full_20260519/`
- 日志文件: `31_policy_full_20260519/logs/run_full.log`
- 当前任务: 生成主结果、外部验证、summary 和 full_status，方便直接汇报

## 2026-05-19 路由对照

- tmux session: `policy_router_soft_20260519`
- 启动脚本: `03_code/run_policy_router_soft.sh`
- 输出目录: `32_policy_router_soft_20260519/`
- 日志文件: `32_policy_router_soft_20260519/logs/run_safety_abstention_evidence.log`
- 当前任务: soft routing 对照，和当前 hard routing 做直接比较

## 2026-05-19 路由混合对照

- tmux session: `policy_router_hybrid_20260519`
- 启动脚本: `03_code/run_policy_router_hybrid.sh`
- 输出目录: `33_policy_router_hybrid_20260519/`
- 日志文件: `33_policy_router_hybrid_20260519/logs/run_safety_abstention_evidence.log`
- 当前任务: hybrid routing 对照，补齐 hard/soft/hybrid 三组比较

## 2026-05-19 外部验证补强

- tmux session: `policy_tian_ext_20260519`
- 启动脚本: `03_code/run_policy_tian_ext.sh`
- 输出目录: `34_policy_tian_ext_20260519/`
- 日志文件: `34_policy_tian_ext_20260519/logs/run_safety_abstention_evidence.log`
- 当前任务: 专门跑 `TianKampmann2019` 外部验证，给明天汇报补一块更有分量的外部证据

## 2026-05-19 GPU 深度对照

- tmux session: `gpu_deep_tian_20260519`
- 启动脚本: `03_code/run_gpu_deep_gpu0.sh`
- 输出目录: `35_gpu_deep_gpu0_20260519/`
- 日志文件: `35_gpu_deep_gpu0_20260519/logs/run_gpu_deep.log`
- 当前任务: 用 `scgpt_env` 在 GPU0 上跑深度 residual transport，主打 `TianKampmann2019`

- tmux session: `gpu_deep_main_20260519`
- 启动脚本: `03_code/run_gpu_deep_gpu1.sh`
- 输出目录: `36_gpu_deep_gpu1_20260519/`
- 日志文件: `36_gpu_deep_gpu1_20260519/logs/run_gpu_deep.log`
- 当前任务: 用 `scgpt_env` 在 GPU1 上跑 main 队列的深度 transport 对照
