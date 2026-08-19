# confidence_task 是早期流水线，不是当前主入口

这个目录保留，是因为它能复现 Phase 2.1 的早期大脚本。

当前主线代码在 `safetrans_confidence/`：

- `safetrans_confidence/scoring/protocol_v0_2.py`：冻结后的可信度打分公式。
- `safetrans_confidence/eval/metrics.py`：Spearman（相关性）和 risk-coverage（风险覆盖）评估。
- `safetrans_confidence/cli/run_benchmark.py`：正式 benchmark（标准测试）入口。

请不要把本目录里的旧脚本当作新的主方法继续扩展。
