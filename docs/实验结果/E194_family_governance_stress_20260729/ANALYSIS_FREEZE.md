# E194 注册家族构成与治理压力测试

冻结日期：2026-07-29

实现前修订：2026-07-29。在代码尚未提交、结果尚未运行时完成独立数学审阅，
补充固定 A0 误差对象、E193 复现、方差分解和成员数组哈希检查；同时把 C4 限定
为 absolute RMSE。修订不接触目标结果。

## 证据性质

E194 使用 E190/E192 已打开的真值，是
`POSTTRUTH_FAMILY_GOVERNANCE_STRESS`，不是独立确认。分析目的有三个：

1. 检查确定性证书在不同合法 family 定义下是否仍按各自对象成立；
2. 量化成员选择、成员重复和架构权重对 tightness 与经验排序的影响；
3. 把“什么可以进入主 family”写成可执行的治理合同，阻止靠复制模型或加入异常
   成员制造表面上的大分歧。

## 冻结输入

- E190 Adamson K562 → Replogle K562：692 个任务、47 个目标基因簇；
- E192 Adamson K562 → Replogle RPE1：175 个任务、21 个目标基因簇；
- 每项均使用预真值冻结的 3 个 scGPT 和 3 个 GEARS 成员；
- 三种几何沿用 E193：`absolute_rmse`、`cosine`、`pearson`；
- 程序必须复核 E190/E192 的 prediction release、truth release 和 source-effect
  锁。

E194 不读取新数据，不重训模型，不修改 E190–E193 的任何状态。

## Family-conditional 语义

对任意冻结的有限 family \(F=\{p_i\}\) 及非负权重
\(\sum_i w_i=1\)，在指定 Hilbert 嵌入 \(\phi\) 中定义

\[
\mu_F=\sum_iw_i\phi(p_i),
\]

\[
R_F^2=\sum_iw_i\lVert\phi(p_i)-\phi(y)\rVert^2,
\qquad
D_F^2=\sum_iw_i\lVert\phi(p_i)-\mu_F\rVert^2.
\]

必须复核

\[
R_F^2=\lVert\mu_F-\phi(y)\rVert^2+D_F^2.
\]

该恒等式只描述当前声明的 \(F\)。加入 zero、source 或合成成员后，\(D_F\) 只能
下界新 family 的 \(R_F\)，不能用于下界原六成员 family 的误差。所有场景必须按
自己的成员和权重重算 target error。

## 主 family 治理合同

唯一主 family 为 `A0_primary_balanced_3x3`：

- scGPT、GEARS 各占总权重 \(1/2\)；
- 每个架构内 3 个唯一 seed lineage 等权；
- 成员必须是目标真值解封前已提交哈希的真实预测输出；
- 相同 prediction hash 或相同 lineage 的副本不能增加权重；
- 不允许按目标真值、相关系数或复核收益选择成员；
- zero、source-effect、架构质心和合成向量都不能进入主 family。

## 冻结场景

### A. 合理 family 与诊断 family

- `A0_primary_balanced_3x3`：3 scGPT + 3 GEARS，各成员 \(1/6\)；
- `A1_scgpt_seed_only`：3 个 scGPT seed；
- `A2_gears_seed_only`：3 个 GEARS seed；
- `A3_matched_pair_seed{3407,3408,3409}`：同 seed 的 scGPT + GEARS；
- `A4_architecture_centroids`：scGPT 三 seed 质心与 GEARS 三 seed 质心组成两个
  诊断成员；
- `A5_pair_1x1_*`：枚举全部 9 个 1 scGPT + 1 GEARS 组合；
- `A5_balanced_2x2_*`：枚举全部 9 个 2 scGPT + 2 GEARS 组合。

A4 及 A5 只解释构成敏感性，不替代 A0。

### B. 重复、遗漏与架构失衡

- `B1_duplicate_all_flat`：六成员各复制一次，12 个条目普通等权；
- `B1_duplicate_all_governed`：相同 12 个条目，但每个 lineage 的原权重由副本
  均分；
- `B2_duplicate_one_flat_*`：逐个复制一个成员，7 个条目普通等权；
- `B2_duplicate_one_governed_*`：副本拆分原 lineage 权重；
- `B2_leave_one_out_*`：逐一删除一个成员；
- `B3_overweight_scgpt_flat`：scGPT 三成员各复制一次，9 个条目普通等权；
- `B3_overweight_scgpt_governed`：架构仍各占 \(1/2\)，副本只拆分 lineage 权重；
- `B3_overweight_gears_flat/governed`：对 GEARS 做同样处理。

### C. Portfolio 与 gaming 负对照

- `C1_add_zero_portfolio`：原六成员加 zero-effect，仅在 absolute RMSE 定义；
- `C2_add_source_portfolio`：原六成员加已冻结 source-effect；
- `C3_add_zero_source_portfolio`：加 zero 与 source，仅在 absolute RMSE 定义；
- `C4_symmetric_attack_lambda{1,2,4}`：不读取真值，先计算主 family 质心
  \(\mu\) 和架构质心差 \(v=\mu_{\mathrm{scGPT}}-\mu_{\mathrm{GEARS}}\)，再加入
  \(\mu+\lambda v\) 与 \(\mu-\lambda v\) 两个对称合成成员；仅在 absolute
  RMSE 中运行。cosine/Pearson 的 Hilbert 嵌入位于固定半径球面，这两个合成点
  通常不对应合法的原始预测方向，故不创建该场景。

C4 的新 family 质心必须与 A0 完全相同，但 diversity 可随 \(\lambda\) 增大。它只
用于展示“更换 family target 可以制造更大证书”，不能作为方法有效证据。

## 固定输出量

每个 `dataset×geometry×scenario` 输出：

- 条目数、唯一 lineage 数、entry effective N、合并重复 lineage 后的 effective
  N、scGPT/GEARS 总权重和最大 lineage 权重；
- family RMS、centroid error、worst-member error；
- diversity lower bound、diameter/2 lower bound；
- family 与 worst 下界违例数；
- 平方恒等式残差；
- `D/R`、`D²/R²`、`(diameter/2)/worst`；
- diversity 与自身 family RMS 的 Spearman 相关；
- diversity 的 20% high-error capture、error lift、oracle-normalized utility；
- 相对 A0 diversity 排序的 20% top-set Jaccard；
- 相对 A0 的 family target error 和 diversity 变化。

全部枚举组合逐项保留，并另按 scenario group 报告 median、min、max；不选择最好组合。

`D_F → own family RMS` 的相关和 utility 是证书排序诊断，因为
\(R_F^2=C_F^2+D_F^2\) 存在结构耦合，不能单独解释为“预测未知误差”。为避免在
不同 family 间悄悄更换结果变量，每个场景还必须报告：

- \(D_F\) 对固定 `A0 family RMS` 的 Spearman 与 20% utility；
- \(D_F\) 对固定 `A0 centroid error` 的 Spearman 与 20% utility；
- \(D_F\) 对自身 centroid error 的 Spearman；
- 上述相关的 gene-macro 敏感性：先在每个目标基因内取均值，再跨基因计算。

A1/A2/A4 与 A0 的经验差异只作描述。未提供配对差异区间时，不得声称某种 family
显著优于另一种。

## 确认性实现 gate

1. 所有有效场景的 family RMS / worst lower violation 为 0；
2. 最大平方恒等式残差不超过 \(10^{-10}\)；
3. `B1_duplicate_all_flat` 与 A0 的逐任务全部核心量最大差不超过
   \(10^{-10}\)；
4. 三项 governed 架构重复场景与 A0 的逐任务全部核心量最大差不超过
   \(10^{-10}\)；
5. `B2_duplicate_one_governed_*` 与 A0 最大差不超过 \(10^{-10}\)；
6. C4 各 \(\lambda\) 的 family centroid error 与 A0 最大差不超过
   \(10^{-10}\)，且 mean diversity 随 \(\lambda=1,2,4\) 严格递增；
7. cosine/Pearson 不创建 zero 向量场景，也不使用 epsilon 强行归一化；
8. 每个输出明确记录 `target_family_id`，不得跨 family 使用下界。
9. E194 的 A0 必须逐任务复现 E193 的 family RMS、centroid、worst、diversity
   和 diameter/2，最大差不超过 \(10^{-10}\)；
10. 每个任务必须满足
    \(R_{A0}^2=\frac12R_{A1}^2+\frac12R_{A2}^2\) 和
    \(D_{A0}^2=\frac12D_{A1}^2+\frac12D_{A2}^2+D_{A4}^2\)，残差不超过
    \(10^{-10}\)；
11. A4 与 A0 的嵌入空间质心及 centroid error 必须一致；
12. 六个真实预测数组逐一计算 SHA-256；主 family 中 prediction hash 与
    lineage ID 均必须唯一；
13. 场景数固定为每个数据集 absolute RMSE 55 个、cosine 50 个、Pearson
    50 个；每个 `dataset×geometry×scenario×task_id` 恰好一行。

这些 gate 检查实现与治理不变量，不检验经验排序好坏。

## 探索性统计

- E190、E192 分开报告；
- 三种几何均报告点估计；
- 只对 absolute RMSE 的 A0、A1、A2、A4 做基因整簇 bootstrap：
  Spearman 5,000 次，20% utility 3,000 次；
- 同一基因的全部 batch 任务一起重采样；
- 不把 867 个任务当独立样本，不把两个 target 合并制造显著性；
- 不把 C 组压力负控的高 utility 当支持证据。

## 冻结输出

- `E194_STATUS.json`
- `tables/E194_SCENARIO_TASK_METRICS.csv.gz`（逐任务长表使用 gzip 压缩，避免重复
  向远程仓库写入大体积纯文本；解压后仍为标准 CSV）
- `tables/E194_SCENARIO_SUMMARY.csv`
- `tables/E194_GROUP_RANGE_SUMMARY.csv`
- `tables/E194_BOOTSTRAP_SUMMARY.csv`
- `tables/E194_INVARIANT_AUDIT.csv`
- `tables/E194_FAMILY_MEMBER_AUDIT.csv`
- `tables/E194_INPUT_HASHES.csv`
- `figures/E194_family_governance_stress.png/.pdf`
- `reports/E194_REPORT.md`
- `reports/E194_INTERPRETATION.md`
