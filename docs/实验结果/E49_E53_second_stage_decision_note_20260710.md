# E49-E53 第二阶段实验决策记录

这一步从“第一批 smoke”继续往正式实验推进。重点不是多跑几个表，而是把老师要求的多维度数据拆成能支撑论文判断的几条线。

## 1. 已经完成的第二阶段实验

| 编号 | 内容 | 输出目录 | 当前判断 |
| --- | --- | --- | --- |
| E49 | OpenProblems DGE：官方 train→test、内部 cell-type holdout、内部 compound holdout | `E49_E52_formalization_batch_20260710` | 可进入正式补充，尤其 official split 和 pooled cell-type holdout |
| E50 | sciplex3：1000 基因 cell-line holdout | `E49_E52_formalization_batch_20260710` | 当前最稳的一条 chemical/cell-line 外推证据 |
| E51 | Norman：单基因到组合扰动，mean/sum 单基因组合预测 | `E49_E52_formalization_batch_20260710` | 可支撑 combination perturbation 方向 |
| E52 | TCDD：dose-aware 设计，最近剂量 + log-dose 线性趋势 | `E49_E52_formalization_batch_20260710` | dose-aware 后信号明显增强，值得保留 |
| E53 | Tahoe raw：128 个完整 shard、361 万行字段审计 | `E53_tahoe_raw_expanded_audit_20260710` | 证明 Tahoe raw 足够支撑 drug/cell-line/MoA/plate 分层 |
| E54 | sciplex3：1000/3000/5000 基因数敏感性 | `E54_sciplex3_gene_sensitivity_20260710` | 结果稳定，sciplex3 可作为正式 chemical/cell-line 主线 |

## 2. 关键结果

- OpenProblems official train→test：`risk_predicted_magnitude` 对 additive error 的 Spearman 约 0.64，top 20% enrichment 约 1.25；`risk_op_formal` 对 cell-mean error 的 Spearman 约 0.63。
- OpenProblems pooled cell-type holdout：602 个任务，`risk_predicted_magnitude` Spearman 约 0.62，top 20% enrichment 约 1.57；`risk_op_formal` Spearman 约 0.52。
- sciplex3 cell-line holdout：104 个任务，1000 基因；`risk_safeconf_smoke` Spearman 约 0.90，top 20% enrichment 约 1.65；`risk_predicted_magnitude` 更强，Spearman 约 0.94。
- Norman single-to-combo：125 个组合扰动；`risk_norman_formal` 对 sum-single error 的 Spearman 约 0.62，top 20% enrichment 约 1.29。
- TCDD dose-aware：48 个剂量任务；`risk_tcdd_dose_aware` 对 mean error 的 Spearman 约 0.71，top 20% enrichment 约 1.46。第一批普通 dose smoke 很弱，改成 dose-aware 后明显变好。
- Tahoe raw expanded audit：当前完整 shard 中抽 128 个，合计 3612767 行；drug 291 类，cell line 50 类，MoA 27 类，plate 8 类。`drug_x_cell_line` 有 14467 个组合，其中 4428 个组合至少 300 个细胞；`moa_x_cell_line` 有 1349 个组合，其中 1009 个至少 300 个细胞。
- sciplex3 gene sensitivity：1000/3000/5000 基因下，`risk_safeconf_smoke` 对 mean error 的 Spearman 分别约 0.90、0.92、0.91，top 20% enrichment 约 1.65；说明 sciplex3 结果不是单一基因数设置造成的偶然现象。

## 3. 现在的判断

最值得进入正式结果的顺序：

1. sciplex3 cell-line holdout：信号强，字段干净，且 1000/3000/5000 基因敏感性稳定，能说明 chemical/cell-line 外推。
2. TCDD dose-aware：修正后信号明显，能说明“剂量不能粗暴当普通类别”，这条有方法设计意义。
3. OpenProblems official + cell-type pooled：独立公开 benchmark，适合增强外部可信度。
4. Norman single-to-combo：支撑组合基因扰动，和 chemical 线形成互补。
5. Tahoe raw 分层：当前先作为可行性和后续数据底座；等 raw 下载更多后，再接 cell-line / drug / MoA / plate 正式分层。

## 4. 不能乱写的地方

- OpenProblems 的 compound holdout 单个 split 只有 4–6 个任务，不要拿单药 Spearman 当主证据。已经改成 pooled 汇总。
- sciplex3 中 magnitude 很强，SafeConf 不能假装独占贡献。正式写法要承认 magnitude 是强基线，再看 SafeConf 是否提供可解释分层或组合优势。
- TCDD 不能再使用普通 perturbation 分类口径。dose-aware 是必须的。
- Tahoe raw 现在是 raw 字段和组合覆盖审计，还不是预测误差结果。不能写成 Tahoe raw 已经完成模型评估。
- Gasperini 标签极稀疏，适合写成 regulatory 边界和挑战，不适合短期作为主结果。

## 5. 下一步继续跑的方向

1. OpenProblems 加 public/private 分开汇总；等 `metadata/moa_annotations.csv` 下载完成后，跑 MoA holdout。
2. TCDD 增加 dose trend 图和每个 cell type 的 dose-error 曲线。
3. Norman 增加 gene-overlap 分类：两个单基因都见过、只见过一个、都没见过；目前 Norman 这份数据主要是“都见过”，需要更细字段确认。
4. Tahoe raw 下载继续跑；超过 500/1000 个完整 shard 后，再做一次 E53 扩展审计，准备真正的 raw 分层任务。
