# E274：18:00 前 SafeConf 实验队列

本目录集中保存 2026-09-27 下午追加的可审计结果。目标是验证方法可行性、历史来源是否完整，以及误差记忆在不同可用率下是否稳定；不把开发集结果写成最终论文结论。

## 队列

1. `tahoe_raw_audit/`：只读取 E263 已锁定的 train/validation treatment 和 available control，核对 panel key、每任务细胞上限、稀疏表达有限性；禁止读取、聚合或评估 sealed test treatment。
2. `error_memory_curve_cpu/`：在 fold 0 的 train/validation、扰动冷分桶上模拟误差记忆可用率 0/5/10/25/50/100%，比较 P+Q+H+E 与零记忆基线。
3. `gpu_stability_gpu0/`、`gpu_stability_gpu1/`：用两个 GPU、多个随机种子重复 E273 MLP 可行性验证，检查动态风险学习是否依赖单一 seed。

## 解释边界

这些是方法审查和实验设计阶段的结果。只有在 raw history provenance、嵌套 cross-fitting、sealed test blind evaluation 都通过后，才能形成确认性论文结论。

方法判断与升级/停止门见：
[`docs/方法设计/20260927_SafeConf_Codex_主方法审查与实验计划.md`](../../方法设计/20260927_SafeConf_Codex_主方法审查与实验计划.md)。
