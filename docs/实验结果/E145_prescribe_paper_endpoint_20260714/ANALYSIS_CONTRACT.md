# E145 分析合同｜PRESCRIBE 论文原始终点口径纠正

冻结时间：2026-07-14（脚本运行前）

## 1. 分析性质

E145 是对已经完成并已经查看真实测试结果的 E95/E96 进行评价口径纠正。它不重新训练模型，不改变测试任务，不学习新分数，也不是独立外部确认。所有结果必须标记为 **post-unblinding metric correction / 已解封数据的口径纠正**。

## 2. 为什么需要纠正

PRESCRIBE 原论文把预测准确度定义为预测与真实扰动效应（log-fold-change）的 Pearson 相关及方向准确度；置信度校准主要检验置信度与 Pearson 准确度之间的 Pearson/Spearman 相关。论文的统一置信度为

\[
\widetilde E = 2\widetilde\nu_{\mathrm{post}}-\widetilde H[P(y\mid\omega)] .
\]

E95 上游实现保存的 `epistemic_conf` 是归一化证据，`aleatoric_conf` 是负预测熵；官方代码按任务取均值，并使用 `2 * epistemic + aleatoric` 得到组合置信度。E96 主要以 RMSE 为终点，因此不能单独代表论文所声称的方向准确度校准。

核对来源：

- PRESCRIBE NeurIPS 2025 论文，式（5）与第 4.2 节：<https://papers.nips.cc/paper_files/paper/2025/file/d6383e7643415842b48a5077a1b09c98-Paper-Conference.pdf>
- 固定上游实现：`/home/yyf/archive/external/PRESCRIBE/Step3_test.py`
- 置信度实现：`/home/yyf/archive/external/PRESCRIBE/src/model/lightening_module.py`

## 3. 冻结输入

只读取两个 E95 formal run，各含 24 个互不重叠的 Norman 单基因测试任务：

| 面板 | 原始逐细胞预测文件 SHA256 | 任务表 SHA256 |
|---|---|---|
| Norman_P1 | `98c0c57e755dc18f5a325bad657e1a850f8868e13fea956de5b76849acbb0831` | `c17af2fee00f7694e435de8ecce9aa81cb0f5a5d44edbcaee72dff0e49b06d05` |
| Norman_P2 | `5ab3bdf59f82bd637d16f401f805a3a5cb981b50aa40f87b6cfea3738c59b4f3` | `a11fcd538647f5ed7e7a0afdebc0ee21daeab201aaa548f93efd7dc41b6c0fcf` |

共同对照均值从与训练完全一致的 `perturb_processed.h5ad` 中读取，不重新归一化。

## 4. 冻结任务级量

对每个任务先在细胞维求预测与真实平均表达，再减相同的 control 平均表达，得到 2,037 基因上的预测效应与真实效应。

- `pearson_effect_accuracy`：两个效应向量的 Pearson 相关；越高越好，是论文口径主终点。
- `cosine_effect_accuracy`：两个效应向量的 cosine similarity；越高越好，是方向补充终点。
- `rmse_effect_error`：两个效应向量的 RMSE；越低越好，是误差补充终点。
- `predicted_magnitude_rms`：预测效应的 RMS；测试真值不可用于计算，是冻结基线。
- `epistemic_confidence`：任务内细胞的 `epistemic_conf` 均值。
- `aleatoric_confidence`：任务内细胞的 `aleatoric_conf` 均值。
- `combined_confidence`：`2 * epistemic_confidence + aleatoric_confidence`。

不因结果选择基因、任务、面板、覆盖率或分数方向。若效应向量方差为零，相关记为缺失并在状态文件报告；不得用零替代。

## 5. 冻结统计分析

### 5.1 面板内关联

在 Norman_P1 和 Norman_P2 内分别计算四个分数与三个终点的 Spearman 相关：

- epistemic confidence；
- aleatoric confidence；
- combined confidence；
- predicted magnitude RMS。

Pearson/cosine 的预期方向为正，RMSE 的预期方向为负。每个相关使用 10,000 次任务 bootstrap 百分位区间。双面板结果取两个面板 Spearman 的等权宏平均，并在每次 bootstrap 中分别对两个面板重采样后取均值。

### 5.2 相对 magnitude 的增量

对每种 PRESCRIBE 置信度，计算其 Spearman 与 magnitude Spearman 的配对差值；同一 bootstrap 样本同时计算两种分数。主比较是：

`combined confidence vs predicted magnitude` 对 `pearson_effect_accuracy` 的双面板宏平均 Δρ。

### 5.3 分数冗余

每个面板计算三种置信度与 predicted magnitude 之间的 Spearman，并给出 10,000 次任务 bootstrap 区间。该分析用于判断置信度是否只是重现幅度排序，不把相关低或高解释为因果关系。

### 5.4 Coverage / filtering

按照论文做法，置信度从高到低保留任务；magnitude 基线同样从高到低保留。冻结 coverage 为 50%–100%，每 5% 一档；保留数使用 `floor(n_tasks * coverage)`，100% 保留全部任务。重点报告论文使用的 95% 和 90% coverage（过滤最低置信度 5% 和 10%）。

对每个面板、分数、coverage 报告保留任务的平均 Pearson、cosine 和 RMSE，以及相对全部任务的变化。95%/90% coverage 下，使用 10,000 次配对任务 bootstrap 比较每种置信度与 magnitude 的保留集平均表现。排序和终点评价在同一已解封测试任务上完成，因此只属于回顾性选择性表现描述。

## 6. 解释规则

- 只有两个面板方向一致且双面板 bootstrap 区间不跨 0，才称为“当前双面板稳定关联”。
- combined confidence 的主 Δρ 区间若跨 0，不宣称其优于 magnitude。
- combined confidence 与 magnitude 高度相关时，明确报告排序冗余。
- E145 不改变 E96 的 RMSE 结果；它只纠正“PRESCRIBE 是否具有论文定义的方向准确度信号”的表述。
- 不将本分析写成对 SafeConf 的直接胜负试验，因为 E145 未在同一预测器、同一评分构造下比较 SafeConf。

## 7. 冻结随机性与输出

- 任务 bootstrap：10,000 次。
- 每个统计量的种子由固定字符串的 SHA256 前 8 位生成。
- 输出任务表、关联表、增量表、冗余表、coverage 曲线、coverage 配对 bootstrap、白底 SVG 图、报告和运行状态。
- 运行失败也不得修改本合同，只能修正实现错误并在状态中记录。
