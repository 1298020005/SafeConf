# E184｜SafeConf 直接竞品与理论来源审计

## 2026-07-29 更新

后续检索新增三类必须正面处理的近邻：

1. CPA 自带基于测试组合到训练扰动/协变量嵌入距离的 uncertainty score，不能只把
   CPA 当作普通预测器；
2. 2026 年预印本已经在 Replogle K562 上研究 intervention 下的 selective
   conformal inference，因此不能再声称“首次把 conformal 用于单细胞扰动”；
3. CAP、Selective Conformal Risk Control 和 SCoRE 说明选择后覆盖或接受样本风险
   需要额外统计条件，SafeConf 当前 `ABSTAIN` gate 不能写成已经证明的 selective
   risk control。

同时，Systema、PerturBench、scPertEval 和 27 方法 × 29 数据集基准已经把多指标、
简单强基线和明确评价空间变成审稿的基本要求。完整更新见
[`期刊与文献定位_20260729`](../../../投稿准备/期刊与文献定位_20260729/README.md)。
本更新不改变 E184 的原裁决：未检索到与注册冻结家族下界证书完全相同的对象，但
SafeConf 可主张的范围比“通用不确定性方法”窄得多。

E194 随后证明，证书对象严格依赖成员与权重：复制输出、调整架构比例或加入合成
成员都会改变 target family；只有按 prediction hash、lineage 和 architecture
冻结治理合同后，主 family 才能稳定解释。

## 裁决

SafeConf 与现有工作的关系可以分成三层：

```text
单模型内部不确定性
GEARS-unc / PRESCRIBE / GPerturb
        │ 输出模型自身的方差、置信度或后验
        │
通用统计校准
split conformal / CQR / hierarchical conformal
        │ 用独立校准真值提供覆盖事件
        │
SafeConf 当前主线
冻结预测家族 + 经典平方误差分解
+ 靶点簇 conformal 上界 + 质心搬移 + fail-closed 审计
```

没有查到与 SafeConf 当前对象完全相同的单细胞扰动方法：输入满足同一输出合同、
事先注册成员与权重的多模型预测向量，在不读取目标真值时同时输出该声明 family
RMS 误差和最坏成员误差的确定性下界，并把独立校准得到的参考质心上界搬移到注册
family。改变成员或权重即改变误差对象，不能跨 family 挪用下界。

这不等于每个数学零件都是新的。平方误差分解是经典 ensemble ambiguity decomposition（集成歧义分解）；`Δ/2` 下界来自三角不等式；split conformal 的有限样本边际覆盖已有成熟理论。稿件若把这些基础成分称为全新定理，容易被直接质疑。SafeConf 应定位为新的问题形式化、证书组合和单细胞实证系统。

## 逐项比较

| 方法 | 它真正输出什么 | 需要改造或重训预测器 | 分布无关有限样本覆盖 | 能否给冻结家族确定性下界 | 与 SafeConf 的关系 |
|---|---|---:|---:|---:|---|
| GEARS uncertainty | GEARS 内部额外方差头学习的逐基因误差代理 | 是 | 否 | 否 | 单一架构内部不确定性 |
| PRESCRIBE | 结合潜空间密度和预测分布的 epistemic / aleatoric 置信度 | 是 | 未提供 | 否 | 强直接竞品，但目标是自身预测置信度和筛选 |
| GPerturb | Gaussian process 对基因扰动效应的模型后验不确定性 | 是 | 不是分布无关覆盖 | 否 | 概率预测器，不是外接审计层 |
| CPA uncertainty | 测试组合与训练扰动/协变量嵌入的距离型 uncertainty score | 是；依赖 CPA 表征 | 否 | 否 | 直接经验风险基线，必须同协议比较 |
| deep ensemble / disagreement | 多模型预测离散程度 | 通常需要多次训练 | 单独使用时否 | 在平方损失下可由经典分解得到家族平均误差下界 | SafeConf 下界的直接数学前身 |
| split conformal | 黑盒预测误差的边际预测集或上界 | 否 | 是，需交换性和正确校准单位 | 否 | SafeConf 上界的统计基础 |
| conformalized quantile regression | 随输入变化的 conformal 区间 | 需要拟合分位数模型 | 是 | 否 | 自适应上界竞品；不替代家族几何下界 |
| hierarchical conformal | 对群组或重复观测依赖进行校准 | 视基模型而定 | 在对应层级假设下是 | 否 | 支持“guide 嵌套于靶点”必须按簇处理 |
| selective conformal under interventions | 被选择任务的覆盖或风险控制 | 依方法而定 | 在论文给定条件下是 | 否 | 阻断“conformal 单细胞首创”，但不替代 family lower bound |
| SafeConf registered-family certificate | 声明 family 的 RMS / 最坏成员下界，加参考质心上界的 family 搬移 | 否；要求预测向量、成员、lineage 和权重满足冻结合同 | 下界为确定性；上界只在注册校准单位、交换性假设和质心搬移条件下继承边际事件 | 是 | 外接式 family-conditional 误差证书 |

完整字段见 `tables/E184_METHOD_DEFINITION_MATRIX.csv`。

## 现有单细胞不确定性方法没有解决的对象

### GEARS uncertainty

GEARS 通过额外的逐基因 log-variance 头学习误差代理。它与 GEARS 隐状态和训练损失绑定，不能在不重训的情况下直接附着到 scGPT、CPA 或任意外部预测器，也不输出任务级 RMSE 的分布无关覆盖。

### PRESCRIBE

PRESCRIBE 是目前最直接的单细胞扰动不确定性竞品。它用 multivariate evidential regression（多变量证据回归）联合描述数据不确定性与模型不确定性，论文用置信度—准确率相关、分位分层、ECE 和过滤后准确率评价。它没有声称任务 RMSE 的 conformal 覆盖，也不为任意冻结预测家族提供真值无关误差下界。

仓库 E91–E96 已按两个事先冻结、互不重叠的 Norman 面板运行其原生流程。E96 的结论只能写成“当前双面板没有形成稳定正相关”，不能写成 SafeConf 全面击败 PRESCRIBE，因为二者审计的是不同预测器。

### GPerturb

GPerturb 是一个 Gaussian process（高斯过程）扰动效应预测器。它的模型后验可以表达单基因效应的存在与强度不确定性，适合解释生物效应。它不是给现成黑盒预测家族增加误差证书的方法，后验区间也不自动等价于交换性条件下的分布无关覆盖。

## SafeConf 的数学新颖性必须怎样写

### 可以写

1. 把冻结单细胞扰动预测家族形式化为同时包含 family RMS 与最坏成员误差的部署
   证书对象；
2. 将参考质心的 conformal 上界通过显式质心搬移代价 `s` 传到注册模型家族；
3. 把 guide 任务嵌套在靶点内，先取靶点最坏 guide，再在靶点层校准和评价；
4. 用分阶段解封、输入哈希和零目标真值读取记录实现 fail-closed 审计；
5. 在四项研究、2,433 个任务和 737 个靶点簇上报告确定性下界、经验上界及失败边界。

### 不应写

1. “首次发现模型分歧可以反映误差”；
2. “提出全新的平方误差分解”；
3. “提出 conformal prediction”；
4. “输出每个模型正确的概率”；
5. “四项合并 90.37% 构成新的 90% conformal 保证”；
6. “SafeConf 全面优于 PRESCRIBE、GEARS uncertainty 或 GPerturb”。

### 最稳的贡献句

> We formulate post-hoc reliability auditing for single-cell perturbation prediction
> as an error-certificate problem for a preregistered weighted family whose members
> satisfy a common output contract. Building on the classical squared-error ambiguity
> decomposition and split conformal calibration, SafeConf provides truth-free
> deterministic lower bounds for the declared family-level errors and transfers a
> calibrated reference-centroid upper bound under explicit calibration and
> centroid-shift conditions.

中文：

> 我们把单细胞扰动预测后的可靠性审计形式化为预注册加权预测家族的误差证书问题，
> 其成员须满足同一冻结输出合同。SafeConf 以经典平方误差集成分解和
> split-conformal 校准为基础，在不读取目标真值时给出声明 family 误差的确定性
> 下界，并在明确的校准和质心搬移条件下传递参考质心上界。

## 对“稳定二区”的实际影响

这次审计关闭了一个高风险问题：过度声称理论原创。当前稿件适合按 computational biology / bioinformatics methods（计算生物学或生物信息方法）定位，卖点是可靠性问题的形式化、严谨协议和跨研究证据，不宜伪装成纯机器学习理论论文。

E185 已完成最小复现入口，E186 已完成完整性审计；这两项不再列为未完成。当前投稿
阻断项为：

1. 在相同 dataset/split/task/gene/metric/budget 下做双轨直接竞品比较：同一 A0
   error target 上比较 SafeConf/disagreement/magnitude；分别评价
   PRESCRIBE、GEARS-UQ、CPA 的 predictor–uncertainty 对；
2. 接入 Systema exact 与 scPertEval 的 representation/transform/metric/reporting
   合同，补 no-change、perturbed/matching mean、PCA/linear、rank、DE 和分布指标；
3. 增加至少两类不同机制的真实预测输出。E194 已完成重复、失衡和合成攻击治理，
   但没有扩大真实模型机制；
4. 再做一次全流程事前冻结的新外部解盲；
5. 写作时将 E 编号证据改写为 Methods、Results、Discussion，而非实验流水账。

## 本轮证据边界

- 原 E184 检索截至 2026-07-24；本更新检索截至 2026-07-29，并纳入 7 月 27 日
  发布的 scPertEval 预印本。优先使用论文官网、期刊页和正式会议论文；
- “没有查到完全相同方法”是检索结论，不是不可推翻的全球唯一性证明；
- E184 没有读取新实验真值，没有修改 E182 的 FAIL，也没有改变 E183 的描述性地位；
- 任何数值胜负仍以对应冻结实验为准，不能从定义矩阵推导性能优劣。
