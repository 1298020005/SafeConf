# SafeConf 当前证据与投稿关卡（2026-07-14，E144 后）

## 一句话判断

项目已经达到**较强二区稿件的投稿水准**，但不能称为“稳定录用二区”。当前最可靠的论文形态是一个**误差定义感知的双风险头系统**：原 SafeConf 排 absolute RMSE 风险；Directional-SafeConf 排 Systema 定义下的 Pearson/cosine 方向误差。新增机制审计只提供一条较弱的通路层支持，跨蛋白、靶基因自身和 STRING 知识距离均未形成确认性证据，不能把项目写成已经证明因果机制。

## 当前正式规模

- 7 套正式 scGPT–GEARS 数据：Frangieh、Lara ex vivo、Santinha、Shifrut、Liang、Tian CRISPRi、Nadig HepG2/Jurkat。
- 32 个外层 folds、3,209 个测试任务、6,418 条 strict PredictionRecord；strict issues=0。
- 覆盖肿瘤刺激、造血分化、神经亚型、供体/TCR、类器官培养、技术批次和跨细胞系变化。
- Nadig 第七数据合同先于表达预处理、模型预测和误差冻结；E135 方向模型哈希先写入 E136，再解封 E139。

## 结果一：absolute RMSE 风险

| 数据集 | SafeConf ρ | magnitude ρ | disagreement ρ | 结论 |
|---|---:|---:|---:|---|
| Frangieh | 0.253 | 0.148 | 0.137 | 正结果 |
| Lara ex vivo | 0.387 | 0.148 | 0.176 | 最强外部复制 |
| Santinha | 0.065 | -0.089 | -0.127 | 弱复制 |
| Shifrut | 0.173 | 0.209 | 0.051 | magnitude 更强 |
| Liang | 0.212 | 0.074 | 0.075 | 正复制 |
| Tian CRISPRi | 0.134 | 0.067 | -0.018 | 技术批次正结果 |
| Nadig HepG2/Jurkat | 0.231 | 0.403 | 0.230 | magnitude 明显更强 |

七数据集等权宏平均：

- SafeConf 相对 disagreement：Δρ=0.133；固定七数据集 95% CI `[0.052, 0.211]`，dataset-population CI `[0.045, 0.223]`。该增量稳定。
- SafeConf 相对 magnitude：Δρ=0.071；固定七数据集 CI `[-0.013, 0.149]`，dataset-population CI `[-0.056, 0.186]`。点估计为正，区间跨 0。
- SafeConf 相对 frozen 公式：Δρ=0.049；区间跨 0。

因此不能继续写“absolute RMSE 上稳定超过 magnitude”。可以写“总体超过单纯模型分歧；相对预测幅度有平均正趋势，但跨数据集不稳定”。完整结果见 [E140](./E140_formal_seven_dataset_meta_20260714/reports/E140_REPORT.md)。

## 结果二：方向误差风险

2025 年 Systema 方法学提醒，RMSE 对共同参考平移不敏感，而 Pearson/cosine 会受到参考状态影响。E133–E134 因此分别检查训练平均效应和训练受扰动表达质心。

### 原 SafeConf 的边界

- 六数据集 exact expression-space audit：centered Pearson ρ=-0.039，centered cosine ρ=-0.032。
- 说明原 SafeConf 的固定结构项不能直接套到方向误差，负结果保留在 [E134](./E134_systema_exact_expression_space_audit_20260714/E134_REPORT.md)。

### Directional-SafeConf

- E135 使用四个部署可见特征做透明的五候选探索。Ridge(alpha=10) 留一数据集结果：Pearson ρ=0.346，cosine ρ=0.336；六个留出数据集均为正。
- 由于候选选择看过六数据集，E135 只算探索性证据；系数随后冻结。
- E139 在未参与设计的 Nadig 双细胞系第七数据确认：Pearson ρ=0.748，cosine ρ=0.757，复合方向 rank 的 perturbation-cluster bootstrap 95% CI `[0.693, 0.801]`；预注册 gate 通过。
- E139 相对 magnitude 的方向排序增量稳定为正；模型未在 Nadig 重拟合，分数文件与 E136 冻结哈希一致。

可以写“方向风险头获得第七数据确认”，不能写“原 SafeConf 对所有误差定义天然稳健”。完整证据见 [E135](./E135_directional_risk_lodo_20260714/E135_REPORT.md) 和 [E139](./E139_nadig_directional_confirmation_20260714/reports/E139_REPORT.md)。

## 简单预测器审计

按 Systema 的训练受扰动表达质心定义，scGPT/GEARS ensemble 在原六数据集中的 5 套优于简单质心；Tian 是例外。Nadig 中 ensemble RMSE=0.1315，简单质心 RMSE=0.4007，79.7% 任务胜出。总体上游模型并非只复现系统性平均变化，但该结论不等于每套数据都领先。

## 生物机制与正交读出审计

| 审计 | 结果 | 允许的表述 |
|---|---|---|
| E141 PROGENy 通路误差 | 原 SafeConf→通路 activity RMSE 的七数据等权 ρ=0.117，分层 bootstrap 95% CI `[0.029, 0.206]`；相对 magnitude 的 Δρ 区间跨 0 | SafeConf 风险与通路层误差存在较弱稳定联系；未证明优于幅度，更不是因果通路验证 |
| E142 Frangieh RNA→蛋白 | SafeConf 与 protein RMSE/cosine error 为正，但 train-only RNA→蛋白 decoder 比训练蛋白均值基线更差，预注册 gate 失败 | 保留为失败的正交验证；不能宣称跨模态确认 |
| E144 STRING/靶基因自身 | SafeConf→target-gene absolute error ρ=0.015，95% CI `[-0.065, 0.099]`；STRING 距离→全向量 RMSE ρ=-0.095 | SafeConf 目前是任务级整体风险，不识别靶基因自身下调，也没有证据表明风险来自到训练基因的 STRING 距离 |

E141 使“高风险对应部分生物过程预测不忠实”有了初步统计依据；E142 和 E144 则排除了两个过强故事。三项均为既有数据上的冻结后二级审计，不增加独立数据集数量。完整结果见 [E141](./E141_progeny_pathway_fidelity_20260714/reports/E141_REPORT.md)、[E142](./E142_frangieh_cite_orthogonal_20260714/reports/E142_REPORT.md) 和 [E144](./E144_string_target_failure_20260714/reports/E144_REPORT.md)。

## 前瞻湿实验状态

E143 已完成服务器可执行部分：48 个正式候选槽位、24 基因 Nadig 技术预实验面板、相关性功效表、双背景三批次/6 文库布局、盲法、QC、排除规则和预注册主 gate。ρ=0.40、80% 功效的 Fisher 近似需要 47 个独立基因，因此正式规模定为 48；细胞数不能替代独立扰动数。

物理实验尚未执行，原因是仍需实验室确认新细胞背景、dCas9-KRAB 稳定株、慢病毒/CRISPRi 条件、平台、预算和负责人。Nadig 面板只能调流程；正式确认至少包含一个未进入七数据开发/评价的新背景。交接材料见 [E143](./E143_prospective_wetlab_validation_20260714/reports/E143_DECISION_AND_HANDOFF.md)。

## 周老师的问题现在怎样回答

| 老师追问 | 当前回答 | 证据 |
|---|---|---|
| 实际预测错误来自谁 | 每折正式训练的 scGPT 与 GEARS；不是代理误差 | E108、E112、E120、E123、E129、E138 |
| 未见组合输入什么 | 训练子矩阵、目标背景 control、扰动标签；目标受扰动表达只用于事后评价 | E97、E99、E119、E122、E128、E136 |
| 随机缺失是否太容易 | 随机 pair、整背景、整扰动、背景与扰动双未见分开报告 | 同上六份合同 |
| 数据是否足够多样 | 7 套数据、32 folds、3,209 个测试任务；生物背景和技术偏移均覆盖 | E140 |
| 风险到底识别什么 | absolute RMSE 与方向误差使用两个风险头；不能混成万能置信度 | E134、E135、E139、E140 |
| 是否比较简单基线 | predicted magnitude、disagreement、training perturbed centroid 均已比较 | E133、E134、E139、E140 |
| 能否节约复核预算 | 六数据 AURC 相对 disagreement 稳定；固定 top-20% 捕获增益仍不稳定 | E132 |
| 是否有误差上界 | E114 覆盖充分但偏宽；E117 紧化后覆盖失败 | E114、E117 |
| 是否跨 chemical | 未通过，magnitude 更强 | E118 |

## 已解决的主要审稿风险

1. **数据太少**：扩展至 7 套正式数据与 32 folds。
2. **只看容易随机缺失**：四类 Cartesian 缺失合同均保留。
3. **复杂模型只复现平均效应**：加入 training perturbed-centroid 简单预测器。
4. **RMSE 指标过单一**：增加 Systema 表达空间 Pearson/cosine，并得到独立第七数据确认。
5. **结果导向改公式**：E135 明示探索性候选比较，E136 冻结系数哈希，E139 再评价。
6. **失败结果被删**：Santinha、Shifrut、Tian simple baseline、Nadig absolute、chemical、紧 conformal、学习型高错误路由器均保留。

## 仍然存在的边界

- absolute RMSE 风险相对 magnitude 的七数据总体 CI 跨 0；Nadig 是清楚的反例。
- 固定 top-20% 复核预算收益尚未稳定。
- conformal 上界覆盖充分但偏宽；紧化版本失败。
- chemical 模态未形成独立增量。
- 高风险通路富集未显著，没有湿实验因果机制。
- Directional-SafeConf 只有一个真正冻结后的第七数据确认；还不是跨很多独立研究的终局证据。

## 当前投稿判断

| 维度 | 状态 |
|---|---|
| 老师要求的困难 setting | 完成 |
| 正式双模型、严格向量合同 | 完成，7 数据/6,418 records/0 issues |
| absolute vs disagreement | 稳定正增量 |
| absolute vs magnitude | 平均为正，跨数据 CI 未闭环 |
| direction vs magnitude | 第七数据预注册确认通过 |
| 简单 perturbed-centroid 基线 | 完成，保留 Tian 例外 |
| 固定预算效用 | 部分完成 |
| 风险上界 | 完成但保守 |
| chemical 跨模态 | 失败边界 |
| 通路层机制 | E141 弱正结果；相对 magnitude 未闭环 |
| RNA→蛋白正交验证 | E142 gate 失败 |
| STRING/靶基因自身机制 | E144 gate 失败 |
| 前瞻湿实验 | E143 方案、功效、盲法与模板完成；物理实验未开始 |

严肃结论：**可以按较强二区稿件组织和投稿，也具备冲击一区的实验基础；不能说“稳定录用二区”，也不能说生物机制已经闭环。** 冲一区最有价值的新增证据是 E143 的前瞻新背景湿实验；继续在同一批公开 RNA 数据上寻找更多事后机制相关，收益已经明显低于其多重检验和叙事风险。

## 论文主张应当改成

1. SafeConf 是单细胞扰动预测后的任务级风险层，不改变上游预测器。
2. 风险排序依赖误差定义：absolute RMSE 与 perturbation-oriented direction error 使用不同风险头。
3. 原 SafeConf 在七数据总体上稳定优于 disagreement，但未稳定优于 magnitude。
4. Directional-SafeConf 在六数据 LODO 后冻结，并通过第七数据确认，显著优于 magnitude。
5. 失败模态、失败数据集、简单基线和统计边界共同定义适用范围。
6. PROGENy 结果只支持较弱的通路误差联系；蛋白、靶基因自身和 STRING 距离审计均不能作为正向主张。

## 当前阅读顺序

1. [E140 七数据 absolute-RMSE 元分析](./E140_formal_seven_dataset_meta_20260714/reports/E140_REPORT.md)
2. [E139 Nadig 第七数据方向确认](./E139_nadig_directional_confirmation_20260714/reports/E139_REPORT.md)
3. [E143 前瞻湿实验交接](./E143_prospective_wetlab_validation_20260714/reports/E143_DECISION_AND_HANDOFF.md)
4. [E141 PROGENy 通路误差](./E141_progeny_pathway_fidelity_20260714/reports/E141_REPORT.md)
5. [E142 Frangieh 蛋白正交失败](./E142_frangieh_cite_orthogonal_20260714/reports/E142_REPORT.md)
6. [E144 STRING 与靶基因自身失败](./E144_string_target_failure_20260714/reports/E144_REPORT.md)
7. [E135 方向风险 LODO 与冻结模型](./E135_directional_risk_lodo_20260714/E135_REPORT.md)
8. [E134 Systema 精确定义审计](./E134_systema_exact_expression_space_audit_20260714/E134_REPORT.md)
9. [E132 六数据分诊效用](./E132_six_dataset_triage_utility_20260714/reports/E132_REPORT.md)
10. [E114/E117 误差界正负对照](./E114_split_conformal_error_bounds_20260713/reports/E114_REPORT.md)
11. [E118 chemical 边界](./E118_chemical_contract_meta_20260713/reports/E118_REPORT.md)

## 复现入口

```bash
python tools/scripts/run_e133_systema_aware_baseline_audit.py
python tools/scripts/run_e134_systema_exact_expression_space_audit.py
python tools/scripts/run_e135_directional_risk_lodo.py
python tools/scripts/run_e136_nadig_two_cellline_contract.py
python tools/scripts/run_e137_build_nadig_combined_asset.py
/home/yyf/.conda/envs/scgpt_env/bin/python tools/scripts/run_e138_nadig_formal_dual_models.py --device cuda:0
/home/yyf/.conda/envs/scgpt_env/bin/python tools/scripts/run_e139_nadig_directional_confirmation.py
python tools/scripts/run_e140_formal_seven_dataset_meta.py
python tools/scripts/run_e141_progeny_pathway_fidelity_audit.py
python tools/scripts/run_e142_frangieh_cite_orthogonal_audit.py
python tools/scripts/build_e143_prospective_wetlab_package.py
python tools/scripts/run_e144_string_target_failure_audit.py
```
