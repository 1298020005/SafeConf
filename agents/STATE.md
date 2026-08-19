# SafeConf 当前权威状态

更新时间：2026-07-14

当前事实基线：`f3af843 experiments: confirm six-dataset SafeConf evidence`

当前工作分支：`exp/task-risk-audit-20260611`

本文件是多 Agent 的共享状态摘要。完整事实以 `docs/实验结果/GATE_STATUS_20260714.md` 为最高入口。

## 1. 当前定位

> SafeConf 是单细胞扰动预测后的任务风险路由方法，用来判断哪些已有预测更可能失败、最值得优先复核。

SafeConf 不替代 scGPT、GEARS 等扰动预测器，也不能无条件称为 predictor-agnostic（预测器无关）置信度。

## 2. 当前正式闭环

```text
正式 scGPT–GEARS 数据集：6
外层 folds：30
测试任务：2,953
strict PredictionRecord：5,906
严格合同问题：0
```

六数据集 calibrated SafeConf 相对：

- predicted magnitude：Δρ=0.111，dataset-population 95% CI `[0.0007, 0.215]`；
- model disagreement：Δρ=0.155，95% CI `[0.059, 0.251]`；
- frozen SafeConf：Δρ=0.059，95% CI `[-0.0067, 0.150]`。

主证据：

```text
docs/实验结果/E131_formal_six_dataset_meta_20260714/reports/E131_REPORT.md
docs/实验结果/E132_six_dataset_triage_utility_20260714/reports/E132_REPORT.md
```

## 3. 当前边界

- Santinha 是弱复制；Shifrut 未超过 magnitude；Tian 含负 fold。
- Tian 的 context 是技术批次，不是新的生物细胞类型。
- E111 显示风险信号对 GEARS 明显强于 scGPT。
- E132 支持 normalized AURC 相对 disagreement 改善；固定 top-20% 捕获增益未稳定。
- E114 的 90% split-conformal 上界经验覆盖 98%，但约为真实平均误差 1.86 倍。
- E117 紧化后覆盖 0.726，不能使用。
- E118 正式 CPA chemical 合同中 magnitude 强于 disagreement，chemical 属于失败边界。
- E126 和 E130 的学习型路由器未通过预设门槛，不继续在已解封真值上调参。
- 高风险基因通路没有通过预设显著性阈值。

## 4. 三个版本

| 名称 | 当前含义 |
|---|---|
| calibrated SafeConf | 使用 fold 内 source validation pair 校准的当前主结果 |
| frozen SafeConf v0.2 | 预先固定、可解释的协议基线 |
| learned router / reliability layer | 补充或负结果，不能修改 frozen 的成功率 |

## 5. 当前论文状态

研究证据已经形成投稿闭环，但现有 `PHASE5A1_METHODS_DRAFT.md` 和 `PHASE5A2_RESULTS_DRAFT.md` 仍是 2026-06-16 的旧七数据集/V0-ContextSim 主线。下一步是以 E131/E132 为主线重写统一 manuscript，而不是继续在当前六数据上堆事后路由器。

论文接力先读：

```text
docs/学习导航/04_论文创作接力说明.md
```

## 6. 权威顺序

```text
当前 gate
  > E131/E132 主证据
  > E111/E114/E116/E117/E118/E130 边界与解释
  > 阶段性实验
  > agents/workspace/旧草稿/archive
```
