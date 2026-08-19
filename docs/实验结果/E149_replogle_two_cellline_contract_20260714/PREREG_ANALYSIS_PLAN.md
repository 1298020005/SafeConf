# E149｜Replogle K562/RPE1 外部复制预注册合同

## 数据和信息边界

- 使用 Replogle 等人的 K562 essential 与 RPE1 CRISPRi Perturb-seq 数据（Cell 2022，DOI: 10.1016/j.cell.2022.05.013；PMCID: PMC9380471）。本地输入是 scPerturb 统一整理后的 h5ad，源文件哈希已冻结。
- 两个细胞系来自同一研究，均为公开回顾性数据；它们不是新做的湿实验，也不检验跨研究批次迁移。
- RPE1 原始筛选库包含 common-essential genes 以及依据 K562 现象挑选的部分基因。因此 E149 的结论范围限定为共享的高覆盖 CRISPRi 靶标，不能外推成全基因组随机靶标的细胞系泛化。
- 本次冻结只读取 AnnData 的 `obs`、`var_names`、矩阵形状和文件字节哈希。脚本没有索引或解码 `X`，没有形成表达效应、模型预测或误差。
- K562 control=10691 个；RPE1 control=11485 个。

## 固定选择规则

1. 扰动标签同时存在于 K562 与 RPE1，排除 control/non-targeting 等标签；
2. 靶基因同时位于两个表达基因身份轴，并位于固定 scGPT whole-human 词表；
3. 每个 cell line × perturbation 至少 100 个细胞，且覆盖至少 10 个 batch；
4. 对全部 394 个合格扰动按预先固定的 SHA-256 种子排序，取前 128 个。细胞数只作预设门槛，不参与门槛后的优先排序。

## 外层划分

- 两个外层 folds 分别留出整个 K562 或 RPE1 的扰动效应。留出细胞系的 control 均值可作为推理时基础状态；留出细胞系的 perturbed expression 不进入训练、验证、校准或风险打分。该任务应称为 control-observed cross-cell-line prediction，而不是完全看不见目标细胞系的 zero-shot。
- 每折哈希留出 26 个 perturbations；source cell line 内固定 16 个 validation pairs、16 个 random seen test pairs，其余 seen pairs 用于训练。
- 主分析只使用 heldout cell line 的 256 个 fold-specific test rows（每折 128 个）；每个 context × perturbation 只出现一次。source cell line 的 random-seen 和 perturbation-unseen rows 仅作次要诊断。

## 固定模型流程

- 完整运行后按 E112/E138 流程构建 control-only 512 基因面板，分别训练 scGPT 与 GEARS；训练 epoch、early stopping、优化器、验证校准、预测记录 schema 均不得根据 Replogle 测试结果改动。
- 使用 E135 已冻结的四特征方向风险模型。Replogle 测试真值不得参与风险分数、标准化、阈值或模型系数。

## 主要终点和通过规则

- 两个主要终点为两预测器平均 centered Pearson error 与 centered cosine error；每折分别算 Spearman，再对两个 cell-line folds 等权平均。
- Pearson/cosine error 在 fold 内转换为 percentile rank 后取平均，形成复合方向误差 rank。
- 按 perturbation 整簇重采样 3000 次；同一扰动在两个细胞系的记录一起进入或离开，避免把配对细胞系记录当独立样本。
- 外部复制 gate：两个主要终点的 fold-macro Spearman 均大于 0，并且复合方向误差 rank 的 perturbation-cluster bootstrap 95% CI 下界大于 0。
- 同时完整报告 predicted magnitude、model disagreement、原 pair-risk、冻结方向风险及训练扰动质心简单基线；不得只展示有利分数。

## 次要分析

- 分别报告 context_unseen 与 context_and_perturbation_unseen；source 内 perturbation_unseen 只作机制诊断。
- 报告 absolute RMSE、各上游模型误差、两模型平均/最大误差，以及风险分数相对 predicted magnitude 的 bootstrap 差值。
- 若 gate 未通过，不允许在 Replogle 上重调 E135 系数后仍称独立复制；失败结果照常保留。
