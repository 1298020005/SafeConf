# SafeConf 第二阶段论文结果包（2026-09-28）

## 当前已经得到的主结果

本阶段不再把三个均值/相似性预测器当成熟上游。实际主线使用已冻结的 TxPert STRING-GAT 和 TxPert Exphormer，分别在相同的 1,808 个主任务、K562/RPE1/HepG2/Jurkat 四个背景上做五折风险学习；另用 CPA sciPlex3 八个固定划分做化学压力线。

### 基因扰动：跨结构结果

| 上游 | 原始幅度 U20 | Ridge P+Q U20 | Ridge P+Q+H U20 | 最优排序 Spearman |
|---|---:|---:|---:|---:|
| TxPert STRING-GAT | 0.7682 | 0.7692 | **0.7842** | 0.7709（P+Q+H） |
| TxPert Exphormer | 0.7587 | **0.7732** | 0.7684 | 0.7643（P+Q+H） |

这两种结构共同支持三点：预测幅度是强基线；当前预测形状/分歧信息有额外排序信息；P+Q 是跨结构都保留的输入。H 的收益依赖结构和背景：GAT 上增加，Exphormer 上固定预算收益下降但连续排序提高。因此第一版统一 SafeConf 定义为 **Ridge(P+Q)**，H 作为可选历史内容扩展，不再宣称双历史固定结构必然有效。

随后进行的跨上游迁移实验把两个上游记录合并，并按基因完全留出，再测试每个上游。Ridge(P+Q) 相对幅度在 GAT 上为 `+0.0068`（4/4 背景正向），在 Exphormer 上为 `+0.0177`（4/4 背景正向）。这比单独拟合更直接地支持共享风险接口；它仍共享同一批生物任务，不能替代新数据确认。

### 化学扰动：CPA 压力结果

CPA 八个 sciPlex3 划分的原始幅度宏 U20 为 0.5816，Ridge P+Q 为 0.5396，Ridge P+Q+H 为 0.6069。P+Q+H 相对幅度点增量 +0.0253，但八划分 bootstrap 区间为 [-0.0952,+0.1590]；上游 CPA 自身此前没有稳定胜过简单来源均值。它证明了化学数据可以接入统一风险接口，也显示剂量/来源覆盖不足会造成明显异质性，暂不能作为主方法独立确认。

## 第一版统一方法

输入任务表示保持统一：`cell context + perturbation type/target + condition + upstream prediction + control/reference`。风险侧只看预测和预测前可获得的公共历史。

- 当前预测 P：预测 RMS 幅度、模型分歧/半径、预测效应形状统计；
- 历史支持 Q：来源细胞数、来源背景数、来源 batch 数、最少来源支持；
- 历史内容 H：来源真实效应幅度、来源间离散度、模型—来源距离和方向一致性；
- 模型：Ridge `alpha=10`，输入 P+Q 共 11 个数值字段和 11 个缺失标记，23 个参数；H 扩展为 15 个数值字段及标记，31 个参数；
- 训练：在同一冻结上游状态内按扰动对象分组交叉拟合，标签是该上游预测的 task-level RMSE；推理输出预期错误风险；
- 反馈：本阶段不使用 Error Memory。后续只在同一模型状态的真实 held-out 反馈闭合后加入，先做记录检索和周期性重训，不做在线改参数。

## 论文的结果表雏形

1. **上游能力表**：TxPert GAT/Exphormer 的预测误差、简单基线和幅度—误差关联；CPA 作为化学压力线并保留负能力结果。
2. **主方法表**：Raw magnitude、Ridge M、Ridge P、Ridge P+Q、Ridge P+Q+H；四背景分项和宏平均。
3. **跨结构表**：同一风险接口在 GAT 与 Exphormer 的增量方向，确定 P+Q 为主结构、H 为可选扩展。
4. **跨上游迁移表**：按基因完全留出的联合风险学习，检验风险模型是否依赖某一上游结构。
4. **化学表**：八个 CPA 划分的逐划分结果、剂量/来源支持分层和压力边界。
5. **消融表**：P→P+Q、P+Q→P+Q+H、Ridge/HGB/MLP；不再上 Transformer 或复杂 gate。

## 当前论文定位

最适合的论文主线不是“又一个扰动预测器”，而是：

> **SafeConf：面向基因和化学扰动预测的预测后风险排序。它用当前预测的不确定性形状和公共实验支持度，在不改变上游模型的情况下，估计哪项预测最值得复核；历史生物内容作为可选扩展，并由跨结构和跨模态实验决定是否启用。**

这条主线已经有可展示的新结果：TxPert 两种结构、1,808 个任务、四背景复现，以及化学压力线。它足以支撑 9 月 30 日进展/开题汇报，并形成二区/CCF-B 方向的完整方法雏形。正式投稿前仍需要把化学线换成能力合格的成熟药物上游或明确将 CPA 定义为压力测试，并补一套真正独立的新背景/新数据确认。

## 结果位置

- GAT 主结果：[txpert_risk_batch/REPORT.md](txpert_risk_batch/REPORT.md)
- Exphormer 复现：[txpert_exphormer_risk_batch/REPORT.md](txpert_exphormer_risk_batch/REPORT.md)
- CPA 化学压力：[cpa_risk_batch_v2/REPORT.md](cpa_risk_batch_v2/REPORT.md)
- 统一运行脚本：`tools/scripts/run_stage2_txpert_risk_batch.py`、`tools/scripts/run_stage2_cpa_risk_batch.py`
