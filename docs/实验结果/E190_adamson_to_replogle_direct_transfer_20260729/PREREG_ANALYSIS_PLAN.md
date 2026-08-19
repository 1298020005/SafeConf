# E190｜Adamson→Replogle 基因扰动直接跨研究预测

冻结日期：2026-07-29  
性质：公开旧数据上的回顾性外部迁移；不冒充前瞻实验。

## 研究问题

在 Adamson 2016 K562 CRISPRi 中训练的扰动预测器，不用 Replogle 2022
扰动表达做微调，能否直接预测 Replogle K562 essential 数据中相同基因扰动的
表达效应？这对应周老师提出的“在一个数据集上学习、到另一个数据集预测”。

## 为什么选这两个数据集

两者细胞系与干预类型相同，研究、实验批次和测量流程不同。元数据审计得到 59 个
共同单基因扰动。目标研究按 `(batch, gene)` 至少 5 个扰动细胞、同 batch 至少
20 个 non-targeting control 后，冻结 47 个共同基因、692 个目标任务。

Primary CD4 与 Sunshine 不作为本实验主配对：两套已冻结靶点没有交集，既有 512
基因面板只交叠 73 个基因。直接拼接需要大规模缺失填充，无法区分迁移失败与输入
构造失败。

## 冻结输入与基因轴

- 源：`AdamsonWeissman2016_GSM2406681_10X010.h5ad`；
- 目标：`ReplogleWeissman2022_K562_essential.h5ad`；
- 共同输出轴 512 基因：先纳入 47 个冻结扰动基因，其余位置只按 Adamson control
  平均表达排序，并要求同时存在于两个数据集和 scGPT 词表；
- 使用公开文件中的处理后 `X`，不在两个研究之间做使用目标扰动真值的缩放。

## 源研究监督任务

Adamson 中同一基因可能有多个 guide。每个 `(gene, guide)` 的细胞只按 cell ID 的
SHA-256 顺序分成 5 个伪重复：4 个进入训练、1 个进入验证。效应为伪重复平均表达减
Adamson control 平均表达。该拆分只服务模型拟合与早停，不当作独立生物重复。

## 目标研究查询

每个查询是一个 Replogle `(batch, gene)`：

- 输入：该 batch 的 non-targeting control 平均表达、扰动基因身份；
- 禁止输入：该任务的扰动细胞表达、真实效应、误差；
- 真值：同 batch 扰动平均表达减同 batch control 平均表达，仅在 prediction
  release 提交到 GitHub/Gitee 后由独立程序构建。

## 模型与比较对象

- scGPT：3 个随机种子 3407–3409；
- GEARS：3 个相同随机种子；
- zero-effect；
- Adamson 该基因的直接平均效应（source-effect transfer）；
- 六成员 centroid、family RMS error、worst-member error；
- 六成员多样性下界和直径一半下界。

## 主终点

1. 每个模型和 source-effect baseline 相对 zero-effect 的逐任务胜率；
2. 按目标基因整簇 bootstrap 的平均 RMSE 差及 95% 置信区间；
3. family RMS/worst 的确定性下界违例数；
4. 分歧、predicted magnitude、source-effect magnitude 与真实 family error 的
   Spearman 相关及整簇置信区间；
5. 按目标 batch 和基因检查失败是否集中。

预测器不优于 zero-effect 不是实现失败，必须原样保留。确定性下界出现超过
`1e-10` 的违例、query 图含 `y`、训练程序读取目标扰动表达，才判合同失败。

## 停止规则

- 不根据结果删基因、删 batch、换输出基因或改归一化；
- 不将同一细胞系写成同一数据分布；
- 不把 guide 伪重复写成 biological replicate；
- 若直接迁移失败，结论是跨研究预测器失准，不能把分歧小解释为安全；
- 本实验不使用目标真值校准 conformal 上界。
