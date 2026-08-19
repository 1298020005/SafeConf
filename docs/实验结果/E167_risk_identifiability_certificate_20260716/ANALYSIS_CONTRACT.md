# E167 分析合同｜风险可识别性与适用性证书（RIAG v1）

冻结日期：2026-07-16。E167 是已解封历史资产上的方法开发，不是新的外部确认。它只读取已经发布的任务分数、预测向量和任务级评价表，不读取任何原始 `.h5ad`、单细胞表达矩阵或新候选数据真值。

## 1. 目的

SafeConf、PRESCRIBE 或其他置信度方法进入测试评价前，先回答三个更基础的问题：候选分数是否有足够任务间分辨率；上游预测是否真正随任务变化；候选排序是否只是 predicted magnitude 的同序复制。任何硬门失败时，后续测试相关性不得把状态改回通过。

证书区分两条路线：

- `MODEL_UQ`：分数来自预测模型输出或其后验，必须通过预测任务依赖和重复稳定性检查。
- `STRUCTURAL_RISK`：分数还可使用 context、support、perturbation novelty 等任务元数据。预测塌缩不等于结构风险必然无效，但此时不得把结构分数称作模型内生置信度。

## 2. 开发集合

- E153 八研究正式 GEARS–scGPT 任务与向量：非塌缩对照。
- Norman PRESCRIBE P1/P2：非塌缩、但与 magnitude 高度冗余的对照。
- Norman PRESCRIBE P3/P4：官方分数饱和、预测向量塌缩的真实负例；raw log-probability 作为“分数变化但预测不变”的负例。
- Wessels PRESCRIBE 三 seed：raw 分数变化，但三套预测各自在 48 个任务上塌缩；用于重复稳定性检查。
- E87/E89 化学跨数据集预测：预测和分数均可估计，但 E87 上游预测器输给 no-change；用于证明证书是必要条件而非准确率保证。

全部输入路径和 SHA-256 固定在 `SOURCE_LOCK.csv`。正式运行必须验证 30 个输入哈希以及合同、source lock、runner 均与当前 Git HEAD 一致。

## 3. 真值前硬门

### G0｜访问与哈希

只允许读取 `SOURCE_LOCK.csv` 中的已发布文件；raw expression access 固定为 0。正式 runner 与合同先提交，再生成 `release/`。

### G1｜端点登记

每个单元固定 `endpoint_id`、`predictor_family`、`perturbation_family` 和 `lane`。E167 不在看到结果后切换端点。

### G2｜分数可估计性

候选分数必须全部有限，同时满足：

- `n_unique >= max(12, ceil(n_tasks / 2))`；
- `population_std > 1e-6`。

该阈值沿用此前 PRESCRIBE 真值解封前使用的非退化门槛。常数或 machine-epsilon jitter 均输出 `ABSTAIN_SCORE_SATURATION`。

### G3｜预测任务依赖

预测向量按 `1e-6` 精度量化。每个冻结预测器分别满足：

- 全部坐标有限；
- `quantized_unique_vectors >= max(12, ceil(n_tasks / 2))`；
- 至少一个坐标的跨任务 population std 大于 `1e-6`。

双预测器单元取两者中的最差值。失败输出 `ABSTAIN_PREDICTOR_COLLAPSE`。raw label score、相关性或换端点不得覆盖。

### G4｜重复稳定性

存在三个或更多 seed/checkpoint 时，计算任务秩的 Kendall W 和 seed 间两两 Spearman 中位数。预设通过线为两两中位数至少 0.5，按任务重采样的 2,000 次 bootstrap 95% CI 下界大于 0。没有 K≥3 重复的历史单元只写 `NOT_EVALUATED`，不能获得完整 `MODEL_UQ` 授权。

### G5｜magnitude 同序

若候选分数与 predicted magnitude 具有完全相同的 weak order，则所有 coverage 下接受集合相同，风险—覆盖曲线和 AURC 也相同，增量效用严格为 0。该单元标记 `BASELINE_EQUIVALENT`。Spearman 绝对值至少 0.98 但 weak order 不完全相同只标记 `NEAR_REDUNDANT`，不作为数学等价。

## 4. 预设输出状态

按以下顺序判定：

1. G2 失败：`ABSTAIN_SCORE_SATURATION`；
2. G3 失败：`ABSTAIN_PREDICTOR_COLLAPSE`；
3. G4 已评价且失败：`ABSTAIN_UNSTABLE`；
4. G5 完全同序：`ASSOCIATION_ONLY_BASELINE_EQUIVALENT`；
5. 其余历史单元：`ELIGIBLE_G2_G3_ONLY`。

最后一种状态仍不是完整部署授权，因为多数历史资产没有三重复和新的未解封确认。

## 5. 回顾性评价

硬门结果先计算并固定；随后才读取已发布表中的任务损失，描述候选风险与固定端点的 Spearman、相对 magnitude 的差值及 2,000 次 cluster/task bootstrap 区间。该部分只能验证“硬门是否正确拦截不可估计案例”，不能把历史数据升格为独立确认。

预设开发通过条件：

1. 100% 拦截 Norman P3/P4 official/raw 与 Wessels 三 seed 的真实预测塌缩单元；
2. 100% 拦截 exact-constant、`1e-12` jitter 和 synthetic prediction-collapse 对照；
3. 100% 识别 magnitude-clone 的 weak-order 等价；
4. E153 八研究、PRESCRIBE P1/P2、E87、E89 共 12 个非塌缩参考单元全部通过 G2/G3；
5. Wessels 三 seed 的重复稳定性不得因测试相关性被覆盖。

## 6. 理论边界

- 若 `Var(rank(score))=0`，Spearman 未定义，top-k 与 AURC 依赖任意 tie-break，非平凡选择性排序不可识别。
- 若所有任务的预测向量相同且分数只由预测向量决定，则分数也相同。任务标签或元数据分数不直接受该命题约束，因此需单独称为结构风险。
- 若候选分数与 magnitude 同 weak order，则二者在任意 coverage 下选出相同集合，增量 AURC 为 0。
- 通过 RIAG 只说明“可以评价”，不保证预测准确、风险相关为正、跨域有效或超过 magnitude。E87 是预先保留的反例。

## 7. 后续确认

E167 通过后，证书规则保持不变。E168 首选尚未读取表达真值的 TianKampmann2019 iPSC/day7-neuron 双状态数据；E169 预留给独立的 DatlingerBock2021。新实验必须在 test truth 前落盘证书，失败后不得 jitter、换 endpoint 或换 seed 重新解释。
