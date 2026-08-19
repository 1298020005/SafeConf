# Cursor ↔ Codex 一区冲刺分工（2026-05-20）

## 已开始

| 任务 | 执行 | 状态 |
|------|------|------|
| 硬性达标文档 | Cursor | `00_meta/Q1_TOP_Q2_MASTER_STANDARD.md` |
| 自动评分器 | Cursor | `03_code/evaluate_q1_readiness.py` |
| ContextSim 审稿对照 | Cursor | `transport_models.ContextSimilarityBaseline` |
| 宽口径 CPU 证据 | 后台 | `46_q1_cpu_push_20260520/` |
| Codex 指令 | Cursor | `~/.codex/AGENTS.md` |

## 当前基线分数（旧跑，未含 ContextSim）

| 运行 | label | 说明 |
|------|-------|------|
| `32_policy_router_soft_20260519` | `NOT_READY` | held-out 未赢 V0；需新 CPU 宽跑 |
| `39_gpu_calibrated_main_20260520` | `NOT_READY` | DeepCalibrated 未分离 main/external phase |

## Codex 请立即做

1. 等 `46_q1_cpu_push` 完成后读 `results/Q1_READINESS_REPORT.json`。
2. 写 **GEARS vs PolicySafeTrans** 同数据集对比表（`00_meta/GEARS_HEAD_TO_HEAD.md`）。
3. 在 `encoders.py` 加 Reactome/GO 基因集（可从 `gene2go.pkl` 起）。
4. 若 GPU 可用：确认 `queue_gpu_effect_objective_20260520.sh` 队列跑完。

## 查看进度

```bash
tail -f /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/46_q1_cpu_push_20260520/logs/run_safety.log
cat /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/46_q1_cpu_push_20260520/results/Q1_READINESS_REPORT.json
```
