# E204 风险指导训练：当前入口

更新时间：2026-09-05

E204 回答周老师提出的第二个用途：既然 source 数据能提前指出某些扰动更难预测，训练时给这些条件更多权重，能否降低难任务误差？

## 当前状态

| 阶段 | 状态 | 证明了什么 |
| --- | --- | --- |
| source-only 难度公式冻结 | 完成 | 权重只依赖 source 支持数、覆盖背景数和跨背景效应离散度 |
| 权重清单和防泄漏审计 | 完成 | 不使用 target 扰动后表达生成权重 |
| K562/RPE1/HepG2/Jurkat 一轮 profile | 全部 PASS | 数据加载、权重覆盖、反向传播、验证和审计状态能完整运行 |
| 正式 80 轮加权训练 | 待运行/待完成 | 尚不能判断难任务是否改善 |
| 新数据前瞻确认 | 待后续 | 同数据开发的训练结果只能作为次级证据 |

四个 profile 的训练样本数分别为 K562 294,951、RPE1 273,003、HepG2 314,391、Jurkat 282,132，真实训练权重回退均为 0，target 扰动表达读取均为 0。

## 公式

```text
difficulty = mean[
  z(-log(1 + n_source_cells)),
  z(3 - n_source_contexts),
  z(source_delta_dispersion)
]

weight = clip(1 + 0.5 × difficulty, 0.5, 2.0)
```

对照细胞固定权重 1。正式比较包含：

- `uniform`：普通训练；
- `risk_weighted`：三个成分等权生成的难度权重；
- `dispersion_only`：只使用 E201 中最强的 source dispersion 成分。

## 先读文件

1. [ANALYSIS_FREEZE.md](./ANALYSIS_FREEZE.md)：正式比较和停止规则；
2. [IMPLEMENTATION_AMENDMENT_20260905.md](./IMPLEMENTATION_AMENDMENT_20260905.md)：验证集误计为训练回退的原因和修复；
3. [PROFILE_ACCEPTANCE_20260905.md](./PROFILE_ACCEPTANCE_20260905.md)：四个 target 的工程验收；
4. `profiles/<target>/profile_result.json`：每个 target 的机器可读结果。

## 正式运行的边界

E201 的 16 个 uniform 模型直接复用。需要新增 16 个 `risk_weighted` 和 16 个 `dispersion_only` 模型，每个 80 轮。主要比较高难 20% 任务 RMSE，同时检查全体 RMSE；若高难任务没有改善，或整体 RMSE 恶化超过 5%，当前加权方案停止扩展。

profile PASS 只说明程序能正确训练，不能写成模型性能已经提高。
