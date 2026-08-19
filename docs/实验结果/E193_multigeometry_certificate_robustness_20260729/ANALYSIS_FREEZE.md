# E193 多几何注册家族证书稳健性分析冻结

冻结日期：2026-07-29

## 证据性质

E193 使用 E190 和 E192 已经解封的目标真值，属于 **post-truth
metric-robustness analysis（开真值后的指标稳健性分析）**，不是新的前瞻预注册
实验。本文档在 E193 计算代码首次运行前提交，用于冻结分析定义，防止看结果后更换
几何、过滤阈值、基线或裁决条件。

E193 可以确认确定性恒等式和下界是否按实现成立；风险排序、相关性和复核收益只作
探索性证据，不能替代新的锁定外部确认。

本文定义的是 **effect-vector directional geometry（效应向量方向几何）**。
Systema 还包含目标 control、post-perturbation state 和训练扰动质心等参考空间处理；
E193 不宣称复现 Systema exact metric。

## 为什么需要这项分析

现有注册家族证书主要使用表达效应向量的 RMSE。2025–2026 年的单细胞扰动评测
工作强调：

1. 单一 RMSE 不能完整表示扰动方向和基因间相对变化；
2. 简单基线可在多项常规指标上达到或超过复杂模型；
3. 可靠性方法需要说明结论是否依赖某一种误差定义。

因此，E193 不再训练模型，也不调整 E190/E192 的六成员预测家族，而是在三个事先
冻结的欧氏嵌入空间中复核同一注册家族证书。

## 冻结输入

| 数据块 | 目标环境 | 任务数 | 基因簇 | 预测家族 |
|---|---|---:|---:|---|
| E190 | Adamson K562 → Replogle K562 | 692 | 47 | 3 scGPT + 3 GEARS |
| E192 | Adamson K562 → Replogle RPE1 | 175 | 21 | 3 scGPT + 3 GEARS |

输入只允许来自下列已冻结文件：

- `../E190_adamson_to_replogle_direct_transfer_20260729/pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz`
- `../E190_adamson_to_replogle_direct_transfer_20260729/evaluation_truth/arrays/TARGET_TRUE_EFFECTS.npz`
- `../E192_adamson_to_replogle_rpe1_locked_transfer_20260729/pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz`
- `../E192_adamson_to_replogle_rpe1_locked_transfer_20260729/evaluation_truth/arrays/TARGET_TRUE_EFFECTS.npz`
- 两项实验各自的 `QUERY_ORDER.csv`、`TARGET_TRUTH_INDEX.csv` 和
  `SOURCE_GENE_EFFECTS.npz`。

程序必须复核 E190/E192 release lock 与 truth lock；任何哈希不匹配均 fail
closed。

## 三种冻结几何

设任务的六个预测向量为 \(p_1,\ldots,p_6\)，真实效应为 \(y\)，基因维数为
\(G=512\)。每种几何先用固定映射 \(\phi_g\) 变换向量，再在变换后的欧氏空间计算
家族证书。

### 1. absolute_rmse

\[
\phi_{\mathrm{rmse}}(x)=x/\sqrt{G}.
\]

变换后欧氏距离等于原始效应向量 RMSE，是 E190/E192 的复算阳性对照。

### 2. cosine

\[
\phi_{\cos}(x)=x/\lVert x\rVert_2,\qquad
d_{\cos}(a,b)=\lVert \phi_{\cos}(a)-\phi_{\cos}(b)\rVert_2/\sqrt{2}.
\]

其平方为 \(1-\cos(a,b)\)，重点检查预测方向，不让整体幅度主导距离。

### 3. pearson

\[
x_c=x-\bar{x},\qquad
\phi_{\rho}(x)=x_c/\lVert x_c\rVert_2,\qquad
d_{\rho}(a,b)=\lVert \phi_{\rho}(a)-\phi_{\rho}(b)\rVert_2/\sqrt{2}.
\]

其平方为 \(1-\mathrm{Pearson}(a,b)\)，去掉向量的基因均值后检查相对表达模式。

映射仅作逐向量确定性变换。不能在看到目标真值后选择基因、重加权基因或改变映射。

## 有效任务规则

- 所有计算使用 `float64`；
- cosine 要求真值、六个预测和 source-effect 的原始 L2 范数均大于
  \(10^{-12}\)；
- pearson 要求上述向量中心化后的 L2 范数均大于 \(10^{-12}\)；
- 不对零范数或近零范数向量添加 epsilon 后强行归一化；
- 因 source-effect 无效而其他向量有效时，证书任务保留，只有 source 基线记为
  缺失；
- 每个数据块、每种几何必须报告输入任务数、证书有效数和各原因排除数。

## 冻结证书量

令 \(z_i=\phi_g(p_i)\)，\(\bar z=6^{-1}\sum_i z_i\)，
\(z_y=\phi_g(y)\)。距离中的固定缩放因子记为 \(s_g\)：RMSE 为 1，cosine 和
pearson 的变换向量再除以 \(\sqrt{2}\)。

逐任务计算：

\[
E_{\mathrm{family}}=
\sqrt{\frac{1}{6}\sum_i d_g(p_i,y)^2},
\]

\[
D_{\mathrm{family}}=
\sqrt{\frac{1}{6}\sum_i
\left\lVert (z_i-\bar z)/s_g\right\rVert_2^2},
\]

\[
E_{\mathrm{centroid}}=
\left\lVert(\bar z-z_y)/s_g\right\rVert_2.
\]

复核恒等式

\[
E_{\mathrm{family}}^2
=E_{\mathrm{centroid}}^2+D_{\mathrm{family}}^2.
\]

同时计算：

- `family_worst_error`：六个成员误差最大值；
- `diameter_half_lower_bound`：六成员最大两两距离的一半；
- `diversity_lower_bound`：上式 \(D_{\mathrm{family}}\)；
- `family_mean_standard_loss`：\(E_{\mathrm{family}}^2\)，方向几何下分别等于
  六成员平均的 \(1-\cos\) 或 \(1-\mathrm{Pearson}\)；
- `diversity_standard_loss_lower_bound`：\(D_{\mathrm{family}}^2\)；
- `worst_standard_loss` 和 `diameter_standard_loss_lower_bound`：最坏 root
  error 与直径下界的平方；
- `source_to_family_centroid_distance`：source-effect 与变换后家族质心的距离；
- `source_directional_error`：source-effect 在同一几何下相对真值的误差；
- `normalized_raw_centroid_error`：先在原始空间求六成员均值、再按同一几何映射后
  相对真值的常规方向误差；它不参与家族恒等式；
- `raw_predicted_magnitude`：原始空间六成员质心相对零向量的 RMSE；
- `source_effect_magnitude`：原始 source-effect 相对零向量的 RMSE；
- `raw_diversity_lower_bound`：原始 RMSE 几何的家族离散度。

## 确认性实现门槛

E193 的唯一确认性 gate 是数学实现一致性：

1. 三种几何、两个数据块的 `family_rms_lower_violation` 总数为 0；
2. `family_worst_lower_violation` 总数为 0；
3. 所有有效任务的平方恒等式最大绝对残差不超过 \(10^{-10}\)；
4. E190/E192 的 `absolute_rmse` 复算结果与原任务表最大绝对差不超过
   \(10^{-7}\)；
5. 所有输入锁校验通过，输出不存在无穷值。

任何一项不满足，E193 状态为 `FAIL`，不得通过放宽容差、删除任务或更换距离来修复。

## 探索性风险排序

每个数据块和每种几何分别报告以下 Spearman 相关：

- 同几何 `diversity_lower_bound` 对 `family_rms_error`；
- 同几何 `diameter_half_lower_bound` 对 `family_worst_error`；
- `raw_diversity_lower_bound` 对方向型 `family_rms_error`；
- `raw_predicted_magnitude` 对 `family_rms_error`；
- `source_effect_magnitude` 对 `family_rms_error`；
- `source_to_family_centroid_distance` 对 `family_rms_error`。

置信区间按目标基因整簇 bootstrap，5,000 次；同一基因在不同 batch 的任务不得拆开。
E190 和 E192 分开报告，不用任务数加权合并制造显著性。

同一次基因簇 bootstrap 还要报告 `diversity_lower_bound` 相对
`raw_predicted_magnitude`、`raw_diversity_lower_bound` 和
`source_effect_magnitude`、`source_to_family_centroid_distance` 的 Spearman
相关系数差。配对差的区间若跨 0，不得声称证书排序优于该基线。

对 `diversity_lower_bound`、`raw_predicted_magnitude`、
`source_effect_magnitude` 和 `source_to_family_centroid_distance`，按
10%、20%、30% 固定复核预算报告：

- high-error capture；
- error lift；
- oracle-normalized utility；
- 基因整簇 bootstrap 95% 区间（3,000 次）。

这些探索性结果没有 PASS/FAIL 门槛。若某指标为负或区间跨 0，原样保留。

## 冻结输出

- `E193_STATUS.json`
- `tables/E193_TASK_METRICS.csv`
- `tables/E193_VALIDITY_AUDIT.csv`
- `tables/E193_CERTIFICATE_AUDIT.csv`
- `tables/E193_RISK_ASSOCIATIONS.csv`
- `tables/E193_PAIRED_RHO_DELTAS.csv`
- `tables/E193_BUDGET_UTILITY.csv`
- `tables/E193_INVALID_VECTOR_LEDGER.csv`
- `tables/E193_INPUT_HASHES.csv`
- `figures/E193_multigeometry_summary.png`
- `figures/E193_multigeometry_summary.pdf`
- `reports/E193_REPORT.md`
- `reports/E193_INTERPRETATION.md`

图统一白底，颜色只区分三种几何；图中不得把探索性相关标成确认性验证。
