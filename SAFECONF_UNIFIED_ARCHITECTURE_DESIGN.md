# SAFECONF_UNIFIED_ARCHITECTURE_DESIGN

> 2026-09-28 更新：当前开题主候选见 [方法推导](docs/方法设计/20260928_证据驱动开题/方法推导.md)。从幅度与质量出发，公共历史按条件增量决定保留，错误记忆研究残差收缩；尚未训练或验证。本文的共享主干/gate 保留为后续备选。

**版本**：v0.2（2026-09-28）
**状态**：外部研究后的主候选，不是最终验证结论。

## 1. 一句话定义

SafeConf 是一个接在冻结单细胞扰动预测器之后的 **evidence-aware post-hoc risk judge**：它根据预测侧证据、公共实验历史、模型自身合法反馈估计预测误差，并在证据不足时给出复核或 abstain 决策。

本定义经过 Phase -1 外部研究审查。它不是“把所有相关模块都塞进一个网络”：PertEMA 已经覆盖冻结预测器 + prediction-time features + OOF error 的近邻合同，HopCPT 已经覆盖相似历史 error regime + 校准，PRESCRIBE/GPerturb 已经覆盖上游内生 uncertainty。SafeConf 的待验证差异只能来自严格的黑盒接口、公共实验历史与模型反馈的分层、gene/chemical 的统一风险协议，以及在冷启动和泄漏约束下仍成立的增量。

## 2. 推荐结构

```text
Frozen predictor
      |
Universal Prediction Adapter
      |
Canonical PredictionRecord
      |
      +--> Prediction Evidence P
      |        magnitude / shape / disagreement / native uncertainty
      |
      +--> Public History Memory H
      |        retrieve -> quality judge -> fixed weighted summary
      |
      +--> Model Error Memory E
               OOF-only append-only records

P -> BaseRisk(P)
P,Q,H -> residual_H
P,E   -> residual_E
support/masks -> optional gates

raw risk = base + gH * residual_H + gE * residual_E
raw risk -> held-out calibration -> expected error / interval / REVIEW / ABSTAIN
```

### 2.1 第一版实际模型

先采用简单风险器：Ridge、HistGradientBoosting 或小 MLP。记录：

- raw magnitude 与 reference percentile；
- prediction effect 的均值、RMS、稀疏度、shape summary；
- ensemble/native uncertainty（有则用，无则 mask）；
- history source/cell/replicate/noise/provenance；
- history mean/dispersion/prediction gap；
- error memory count/similarity/variance/availability；
- `has_H`、`has_E`、`support`。

只有 V0 证明 H/E 有 residual signal，才将简单拼接改成 residual correction；只有 residual 明显优于 concat，才学习 gate。

第一版明确不采用大 Transformer、LLM judge、未经验证的 learned retrieval 或双套大型 gene/chemical encoder。TabM 类轻量表格 ensemble 可以作为效率基线；TabR、Deep Sets、Set Transformer 只在固定汇总的增量通过后进入 V1。2026 年测量可靠性预印本带来的 `Q` 质量变量必须单独记录，不能直接当作上游模型错误标签。

### 2.2 跨扰动类型

共享：

- task/context/control 语义；
- canonical predicted effect；
- output axis and error object；
- P/Q/H/E 的风险学习语义；
- leakage/split/calibration protocol。

类型特有：

- 基因：机制、guide、KO/CRISPRi/CRISPRa/OE、组合成员；
- 化学：compound、dose、time、vehicle、MoA、plate/batch。

实现：共享小主干 + `perturbation_type` embedding + typed masks + 小型 gene/chemical calibration head。结果按类型单独报告。

## 3. 与相邻路线的明确区别

| 路线 | SafeConf 的边界 |
|---|---|
| GEARS/CPA/scGPT/scFoundation | 它们产生预测；SafeConf 不重训、不替代上游。 |
| PRESCRIBE/GPerturb | 它们在上游内建 uncertainty；SafeConf 可以包住黑盒输出并使用外部历史。 |
| PertEMA | 是必须的近邻基线；SafeConf 只有在 public history、cold-start、类型统一或严格审计上有额外证据才有独立主张。 |
| HopCPT | 证明相似历史 error regime 可与 conformal 结合；SafeConf 不能把 retrieval + error memory + calibration 单独写成新颖贡献。 |
| TabR | 提供 retrieval 结构启发；SafeConf 不默认使用可学习 TabR retrieval。 |
| Deep Sets/Set Transformer | 提供历史集合的表示工具；只有历史内容增量成立才启用。 |
| Conformal Risk Control | 放在 raw risk 后做校准和风险控制；不作为 SafeConf 的 evidence encoder。 |
| Judge/verifier | 只借鉴 predictor/judge 分工，不引入 LLM judge。 |

## 4. 论文主张的安全版本

在实验完成前只能写：

> We study an evidence-aware post-hoc reliability layer for frozen perturbation predictors. The design separates prediction evidence, public-history quality/content, and model-specific OOF error memory, with explicit degradation under missing evidence.

只有当对应增量通过后，才分别写：

- magnitude-anchored residual risk；
- public history 的 residual value；
- OOF error memory 的 feedback adaptation；
- gene/chemical shared backbone 的泛化价值。

不得写“统一风险模型已经跨所有扰动类型成立”，除非两条线的外部确认都通过。

## 5. 研究审查后的停止条件

| 结果 | 架构动作 |
|---|---|
| `P` 不稳定或上游 competence 不通过 | 停止下游解释，先修复输入/上游合同。 |
| 控制 `Q` 后 `Delta_H≈0` | 删除公共历史内容主分支，保留质量与 support 诊断。 |
| 合法 OOF 的 `Delta_E` 不稳定 | E 仅保留为反馈协议，不写成当前方法贡献。 |
| concat、residual、gate 差异不稳定 | 采用最小 A，移除复杂结构。 |
| gene/chemical 方向相反 | 统一接口和评估协议，性能结论分线收缩。 |
| 只有随机切分提升，冷切分消失 | 判定为 shortcut 或 leakage，拒绝升级架构。 |
