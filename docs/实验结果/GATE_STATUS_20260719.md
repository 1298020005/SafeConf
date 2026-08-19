# SafeConf 当前证据与投稿关卡（2026-07-22，E178 综合审计完成）

## 当前判断

SafeConf 已形成一套可复核的单细胞扰动预测可靠性审计流程，但目前不能客观称为“稳定二区”或“一区水准已完成”。七数据集的历史结果、Directional-SafeConf 的冻结确认和 E176 的证书/覆盖结果都可以进入稿件；固定 absolute-RMSE SafeConf 稳定超过 predicted magnitude 的说法已经被两次全新靶点实验否定，必须删除。

目前最稳妥的论文结构是：

1. 对预测模型输出做严格的数据与真值访问审计；
2. 用模型对距离给出不依赖目标真值的确定性误差下界；
3. 用供体专属 split-conformal 校准给出经验误差上界；
4. 对经验排序器设置增量门，未超过 magnitude 时自动弃权；
5. 将 absolute RMSE 与方向误差分开处理，不包装成统一万能置信度。

## 7 月 14 日之后发生了什么

| 实验 | 结果 | 对论文结论的影响 |
|---|---|---|
| E168 | 200 个全新靶点，SafeConf 相对 magnitude 的 ΔAURC=0.00106，CI 跨 0 | 第一次未确认旧增量主张 |
| E172 | 800 个互斥新靶点，ΔAURC=-0.000285，CI 跨 0 | 第二次未复制；旧主张正式停止 |
| E173 | 1,000 靶点、3,000 任务核验 pair-distance 下界，mean/max 违例均为 0 | 保留可证明证书，排序增量仍不成立 |
| E174 | 四供体轮换的三 seed pretruth 稳定性 8/24 单元失败 | 在读取 held-out truth 前中止，未把失败面板挑掉 |
| E175 | 扩为五 seed 后，24/24 稳定性单元通过 | 只作为无真值方法开发，不回收 E174 |
| E176 | 4 供体、800 新靶点；640 隐藏评价靶点的证书零违例，90% 上界经验覆盖 90.31% | 证书与覆盖闭环；排序仍未超过 magnitude |
| E177 | 独立公开处理数据最终评价完成；50 个隐藏评价靶点、400 任务；pair-distance 下界 mean/max 违例均为 0；target-cluster 上界覆盖 44/50=88.0% | 外部数据支持下界证书，但上界点估计略低于 90% 目标，排序相关很弱 |
| E178 | 冻结 E176/E177 后做跨研究综合；690 靶点簇、2,320 任务；scGPT–GEARS 误差相关为 0.975/0.992 | 任务共享难度占主导；模型分歧可作模型对下界，不能当作单模型置信度或强排序器 |

## 当前有效证据

### 1. 七数据集 absolute RMSE 风险

E140 的结论不变：SafeConf 相对 disagreement 的七数据等权增量稳定为正；相对 magnitude 的点估计为正，但 dataset-population 95% CI 跨 0。E168 和 E172 又在同一 Primary CD4 研究的全新靶点上两次未确认增量，因此不能再使用“稳定超过 magnitude”的概括。

### 2. 方向误差风险

E135 的 Directional-SafeConf 在六数据探索后冻结，E139 在未参与设计的 Nadig 第七数据上通过确认。该结果只适用于 Systema 定义下的 Pearson/cosine 方向误差，不能替代 absolute RMSE 结果，也不能与 E176 的证书合并成一个分数。

### 3. 模型对确定性下界

对任意两个预测 `p1`、`p2` 和真值 `y`，`d(p1,p2)/2` 下界两模型平均 RMSE 和最大 RMSE。E173 与 E176 的数值审计均为零违例。它能够证明“大分歧意味着至少存在一定误差”，但不能判断哪一个模型错，也不能证明“小分歧就是安全”。

### 4. 供体专属 conformal 上界

E176 用每位供体 40 个校准靶点、每靶点三个状态组成一个 cluster，冻结第 37 个次序统计量。640 个从未参与开发或校准的评价靶点中，578 个靶点的三个状态同时被覆盖，经验覆盖率 90.31%，精确二项 95% CI 为 87.75%–92.49%。该结果属于同一研究内的多供体确认，不等于独立研究复现。

### 5. 独立公开处理数据 E177

E177 使用独立公开处理数据，按 144 个单基因靶点、8 个技术组冻结任务。pretruth gate 通过，校准使用 30 个靶点，最终评价使用 50 个隐藏靶点。最终结果：pair-distance 下界对 pair mean/max RMSE 均为 0 违例；split-conformal target-cluster 覆盖为 44/50=88.0%，95% 精确二项 CI 为 75.69%–95.47%；task-level 覆盖为 98.25%。ranking 诊断很弱，SafeConf risk 对 ensemble/pair-mean RMSE 的 Spearman 约 0.057/0.059。该结果应写成“证书可转移，上界接近但点估计未达 90%，排序能力不足”，不能写成外部强阳性。

### 6. 跨研究模型特异性与共享难度 E178

E178 不调模型、不改权重、不重算校准分位数，只综合已完成的 E176/E177。scGPT 与 GEARS 的任务级误差相关分别为 0.975 和 0.992，高误差 Top-20% 集合 Jaccard 为 0.892 和 0.951，说明当前两模型主要面对共同的任务难度。模型分歧与单模型误差在 E176 为负相关、E177 接近零；它仍由三角不等式保证为 pair mean/max RMSE 下界，但不能识别哪一个模型错，也不能作为强排序器。两研究各自校准后的覆盖率描述性合计为 622/690=90.14%，该合计不构成新的 conformal 覆盖保证。

## 周老师的问题对应到哪里

| 问题 | 当前回答 | 主要证据 |
|---|---|---|
| 预测错误来自谁 | 正式训练的 scGPT 与 GEARS，不用代理误差替代 | E108、E112、E120、E123、E129、E138、E176 |
| 未见任务有没有偷看真值 | 合同先冻结；pretruth、校准和最终评价分阶段解封并保留访问记录 | E136、E168、E172、E174、E176、E177 |
| 随机划分是否过于容易 | random pair、整背景、整扰动、双未见和完整供体轮换均分别评价 | E97–E140、E176 |
| 模型随机种子是否稳定 | 三 seed 在 E174 未通过；五 seed 在 E175/E176 的 24/24 单元通过 | E174–E176 |
| 风险有没有可证明部分 | 模型对距离除以 2 是 pair mean/max RMSE 下界，数值零违例 | E173、E176、E177 |
| 风险上界是否可信 | E176 pooled 经验覆盖达到 90% 目标；E177 外部点估计为 88%，CI 覆盖 90%，属于边界结果 | E176、E177 |
| 能不能优于简单强基线 | absolute 排序未稳定超过 magnitude；方向风险在第七数据上通过 | E139、E140、E168、E172、E176 |
| 分数针对单模型还是任务总体 | 两模型误差高度同步，主要是共享任务难度；分歧只能给模型对证书，不能给单模型置信度 | E178 |
| 能否直接指导实验 | 目前上界偏宽、下界较松，尚无部署授权 | E176、E143 |

## 仍未解决的投稿缺口

1. **独立研究确认**：E176 是同一 Primary CD4 研究的多供体验证。E177 已完成独立公开处理数据最终评价，但上界覆盖点估计为 88%，只支持“接近目标且区间覆盖 90%”，不能写成稳健外部达标。
2. **方法增量**：固定 SafeConf 不超过 magnitude；E178 进一步表明两个预测模型高度共享任务难度。证书的数学核心较基础，需要把真正的新意落在完整协议、可识别性、双边风险和 fail-closed 决策上，并与现有 uncertainty/conformal 方法系统比较。
3. **边界紧度**：E176 下界紧度中位数为 15.18%，上界平均宽度仍大。覆盖正确不等于决策效率足够。
4. **生物学验证**：E141 只有较弱通路联系，E142/E144 为负；E143 物理湿实验尚未执行。
5. **稿件完成度**：主张、方法、统计估计量、失败结果和图表尚需统一成一篇干净的 manuscript。

因此，当前可以进入正式写作和投稿准备，也可以按较强期刊标准继续补证据；不能把“具备投稿基础”写成“稳定录用”。录用还受方法新颖性、审稿人判断、期刊范围和同期竞争影响，任何实验数量都不能把这些外部随机性降为零。

## 下一步优先级

1. 围绕 E176–E178 重新收缩论文主张：主线写“可证明下界 + 校准上界 + fail-closed 访问协议”，排序分诊只作为弱诊断，不再作为核心贡献。
2. 在开发数据上比较多模型几何证书、现有 uncertainty 与 conformal baselines，先设增量门，再找新数据确认；未过门就保留 E176/E177 的简单证书版本。
3. 推进 E143 新背景湿实验，由实验室确认细胞系、CRISPRi 条件、预算和执行人后再启动物理实验。
4. 按“证书—校准—弃权—适用边界”重写论文主线，旧 fixed SafeConf 优于 magnitude 的句子全部清理。

## 当前阅读顺序

1. [E178 跨研究双边证书与老师问题逐项回答](./E178_crossstudy_bilateral_certificate_audit_20260722/reports/E178_REPORT.md)
2. [E177 元数据冻结](./E177_sunshine_external_certificate_20260719/README_先看这个.md)
3. [E177 F2 资产构建报告](./E177_sunshine_external_certificate_20260719/pretruth_assets/E177_PRETRUTH_ASSET_REPORT.md)
4. [E177 pretruth gate 报告](./E177_sunshine_external_certificate_20260719/pretruth_release/reports/E177_PRETRUTH_GATE_REPORT.md)
5. [E177 calibration 报告](./E177_sunshine_external_certificate_20260719/calibration_release/reports/E177_CALIBRATION_REPORT.md)
6. [E177 final evaluation 报告](./E177_sunshine_external_certificate_20260719/final_evaluation/reports/E177_FINAL_EVALUATION_REPORT.md)
7. [E176 最终说明](./E176_four_donor_fresh_confirmation_20260719/E176_FINAL_SUMMARY.md)
8. [E176 正式报告](./E176_four_donor_fresh_confirmation_20260719/final_evaluation/reports/E176_FINAL_REPORT.md)
9. [E173 可证伪方法收缩](./E173_falsification_aware_pair_certificate_20260719/reports/E173_REPORT.md)
10. [E172 第二次全新靶点未复制](./E172_primary_cd4_fresh_targets_20260718/postgate_release/reports/E172_JOINT_POSTGATE_REPORT.md)
11. [E168 第一次全新靶点未确认](./E168_primary_human_cd4_fresh_confirmation_20260716/postgate_release/reports/E168_POSTGATE_REPORT.md)
12. [E140 七数据 absolute-RMSE 元分析](./E140_formal_seven_dataset_meta_20260714/reports/E140_REPORT.md)
13. [E139 第七数据方向确认](./E139_nadig_directional_confirmation_20260714/reports/E139_REPORT.md)
14. [E143 前瞻湿实验交接](./E143_prospective_wetlab_validation_20260714/reports/E143_DECISION_AND_HANDOFF.md)
