# SafeConf docs 入口

`docs/` 只放稳定资料：正式实验结果、当前协议/系统设计和小白学习资料。

旧方案、旧提示词和旧审计报告已迁往 `/home/yyf/archive/safeconf/`。

## 当前应该看哪里

给 Qoder、Gemini、Claude 或其他 Agent 学习项目时，先打开：

[SafeConf 学习导航](./学习导航/README.md)

最新证据与投稿关卡先看：

[SafeConf 当前实验总账（2026-07-29，E194 完成）](./实验结果/GATE_STATUS_20260729.md)

周老师的小矩阵、整行整列双未见、基因跨研究预测、有限复核预算、RPE1 锁定确认、
多几何证书和 family 治理：

[E189–E194 最新实验结论](./实验结果/GATE_STATUS_20260729.md)

当前证书的一条命令、标准库最小复现：

[E185 当前证书最小复现验证](./实验结果/E185_minimal_release_validation_20260724/README_先看这个.md)

投稿前证据链、口径、复现和双远端完整性审计：

[E186 投稿前完整性对抗审计](./实验结果/E186_presubmission_integrity_audit_20260724/README_先看这个.md)

投稿前的直接竞品、理论来源和可写贡献边界：

[E184 直接竞品与理论来源定位](./实验结果/E184_direct_competitor_positioning_20260724/README_先看这个.md)

最新四项研究合并审计、有限样本解释和三张白底图：

[E183 四项研究家族证书合并审计](./实验结果/E183_all_study_family_synthesis_20260724/README_先看这个.md)

最新未使用公开研究的一次性注册验证：

[E182 GSE225807 最终评价](./实验结果/E182_gse225807_registered_family_20260724/final_evaluation/README_先看这个.md)

注册模型家族双侧证书、完整表格和五张白底图：

[E181 注册模型家族 Hilbert 误差证书](./实验结果/E181_registered_family_hilbert_certificate_20260724/README_先看这个.md)

前一项一次性独立数据确认：

[E180 XuCao 最终评价](./实验结果/E180_xucao_fresh_guide_certificate_20260723/final_evaluation/reports/E180_FINAL_REPORT.md)

学习型上界与简单基线比较：

[E179 nested UQ 基线审计](./实验结果/E179_nested_uq_baseline_benchmark_20260723/reports/E179_REPORT.md)

老师问题、模型特异性和跨研究双边证书：

[E178 跨研究双边证书与模型特异性审计](./实验结果/E178_crossstudy_bilateral_certificate_audit_20260722/reports/E178_REPORT.md)

最新确认实验及四张白底汇报图：

[E176 四供体全新靶点最终说明](./实验结果/E176_four_donor_fresh_confirmation_20260719/E176_FINAL_SUMMARY.md)

最新外部独立数据 E177 全流程：

[E177 独立公开处理数据元数据冻结](./实验结果/E177_sunshine_external_certificate_20260719/README_先看这个.md)

[E177 F2 资产构建报告](./实验结果/E177_sunshine_external_certificate_20260719/pretruth_assets/E177_PRETRUTH_ASSET_REPORT.md)

[E177 pretruth gate 报告](./实验结果/E177_sunshine_external_certificate_20260719/pretruth_release/reports/E177_PRETRUTH_GATE_REPORT.md)

[E177 calibration 报告](./实验结果/E177_sunshine_external_certificate_20260719/calibration_release/reports/E177_CALIBRATION_REPORT.md)

[E177 final evaluation 报告](./实验结果/E177_sunshine_external_certificate_20260719/final_evaluation/reports/E177_FINAL_EVALUATION_REPORT.md)

需要理解“老师的问题是否答完、为什么仍不能承诺录用、当前一区/二区定位”时打开：

[SafeConf 录用判断与项目总账（白底可视化）](./投稿准备/录用判断与项目总账_20260713/index.html)

| 目录 | 内容 |
|---|---|
| `学习导航/` | 目录权威、当前证据链、Agent 任务和论文创作接力 |
| `实验结果/` | 正式实验结果、figure-ready 表、证据矩阵 |
| `投稿准备/` | 当前证据如何转化为投稿判断、期刊门槛和待补问题 |
| `投稿升级/` | 一区/CCF-A 投稿升级工作台、强基线审计和下一轮实验路线 |
| `代码设计/` | 冻结协议、系统设计和服务器结构 |
| `SafeConf_完整项目讲解/` | 从零到完整项目的十二章可视化教程 |
| `小白科普/` | 早期专题材料与真实数据样例 |

## 小白学习入口

完整学习先打开：

[SafeConf_完整项目讲解/index.html](./SafeConf_完整项目讲解/index.html)

十二章覆盖单细胞基础、数据流、数据集、系统、目录与代码、实验、结果边界、下一阶段、术语词典、协议设计、完整实验证据谱系和真实文件实习。

需要看真实数据实物时再打开：

1. [专题科普入口](./小白科普/index.html)
2. [h5ad 到底长什么样](./小白科普/h5ad真实数据格式/README_h5ad到底长什么样.md)
3. [CSV 浏览器样例](./小白科普/h5ad可打开CSV样例/CSV样例_浏览器查看.html)

## 正式证据入口

主要看：

```text
docs/实验结果/
```

这里保留原路径，避免论文证据引用被打断。

### 2026-07-24 最新覆盖口径

E168 与 E172 均未确认 fixed SafeConf 相对 predicted magnitude 的排序增量，该主张已经停止。E176 在 4 位供体、640 个隐藏评价靶点上得到确定性下界零违反和 90.47% 的家族上界靶点簇同时覆盖；E177 在独立公开研究的 50 个评价靶点上得到下界零违反和 88.0% 覆盖；E180 在 XuCao 独立 CRISPRi guide 任务上得到 73 个任务下界零违反和 27/27 靶点覆盖，并事前否定 ExtraTrees 自适应上界的效率增益。E181 将 scGPT 五个种子和 GEARS 五个种子定义为注册家族。E182 又在此前未使用的 GSE225807 上执行完整事前冻结：下界仍为零违反，但上界覆盖 16/20，未达到注册的 17/20 门槛。E183 保留该 FAIL，并将四项研究统一合并为 2,433 个任务、737 个靶点簇：两类下界零违反，家族上界靶点同时覆盖 666/737=90.37%。当前主线是“注册家族的确定性下界 + conformal 上界 + fail-closed 访问协议”；排序只保留为诊断。

## 投稿升级入口

如果目标按一区/CCF-A 标准推进，先看：

1. [Q1 / CCF-A 投稿升级工作台](./投稿升级/Q1_CCFA_upgrade_20260707/Q1_PUBLICATION_WORKBENCH.html)
2. [Q1 readiness 报告](./投稿升级/Q1_CCFA_upgrade_20260707/Q1_READINESS_REPORT.md)
3. [E9 强基线统一审计](./实验结果/E9_strong_baseline_audit_20260707/reports/E9_STRONG_BASELINE_AUDIT.html)
4. [E11 selective prediction 审计](./实验结果/E11_selective_prediction_audit_20260707/reports/E11_SELECTIVE_PREDICTION_AUDIT.html)
5. [E10 外部数据资产审计](./实验结果/E10_external_task_validation_assets_20260707/reports/E10_EXTERNAL_ASSET_AUDIT.html)
6. [E10 外部任务级验证探针](./实验结果/E10_external_task_validation_probe_20260707/reports/E10_EXTERNAL_PROBE.html)
7. [E12 外部面板扩展探针](./实验结果/E12_external_panel_probe_20260707/reports/E12_EXTERNAL_PANEL_PROBE.html)
8. [E13 官方 sciplex3 三细胞系 focused panel](./实验结果/E13_sciplex3_official_3cell_panel_20260707/reports/E13_SCIPLEX3_OFFICIAL_3CELL.html)
9. [E14 官方 sciplex3 full-743 gene1000](./实验结果/E14_sciplex3_full743_gene1000_20260707/reports/E14_SCIPLEX3_FULL743_GENE1000.html)
10. [E15 官方 sciplex3 full-743 gene2000 sensitivity](./实验结果/E15_sciplex3_full743_gene2000_20260707/reports/E15_SCIPLEX3_FULL743_GENE2000.html)
11. [E16 官方 sciplex3 full-743 gene3000 sensitivity](./实验结果/E16_sciplex3_full743_gene3000_20260707/reports/E16_SCIPLEX3_FULL743_GENE3000.html)
12. [E17 官方 sciplex3 full-743 gene5000 formal](./实验结果/E17_sciplex3_full743_gene5000_20260707/reports/E17_SCIPLEX3_FULL743_GENE5000.html)
13. [E18 真实模型预测向量资产审计](./实验结果/E18_model_vector_asset_audit_20260707/reports/E18_MODEL_VECTOR_ASSET_AUDIT.html)
14. [E19 GEARS-only supplement](./实验结果/E19_gears_only_supplement_20260707/reports/E19_GEARS_ONLY_SUPPLEMENT.html)
15. [E20 adapter contract validator](./实验结果/E20_adapter_contract_validator_20260707/reports/E20_ADAPTER_CONTRACT_VALIDATOR.html)
16. [E21 strict contract remediation smoke](./实验结果/E21_strict_contract_remediation_20260707/reports/E21_STRICT_CONTRACT_REMEDIATION.html)
17. [E22 generator strict smoke](./实验结果/E22_generator_strict_smoke_20260707/reports/E22_GENERATOR_STRICT_SMOKE.html)
18. [E23 shared benchmark adapter workbench](./实验结果/E23_shared_benchmark_adapter_workbench_20260708/reports/E23_SHARED_BENCHMARK_ADAPTER_WORKBENCH.html)
19. [E24 model-family compatibility audit](./实验结果/E24_model_family_compatibility_audit_20260708/reports/E24_MODEL_FAMILY_COMPATIBILITY_AUDIT.html)
20. [E25 GEARS strict PredictionRecord remediation](./实验结果/E25_gears_strict_prediction_records_20260708/reports/E25_GEARS_STRICT_REMEDIATION.html)
21. [E26 GEARS single-model risk audit](./实验结果/E26_gears_single_model_risk_audit_20260708/reports/E26_GEARS_SINGLE_MODEL_RISK_AUDIT.html)
22. [E27 scGPT forward PredictionRecord smoke](./实验结果/E27_scgpt_forward_prediction_record_smoke_20260708/reports/E27_SCGPT_FORWARD_PREDICTION_RECORD_SMOKE.html)
23. [E28 GEARS–scGPT shared Adamson smoke](./实验结果/E28_gears_scgpt_shared_adamson_smoke_20260708/reports/E28_GEARS_SCGPT_SHARED_ADAMSON_SMOKE.html)
24. [E29 GEARS–scGPT shared Adamson risk audit](./实验结果/E29_gears_scgpt_shared_adamson_risk_audit_20260708/reports/E29_GEARS_SCGPT_SHARED_ADAMSON_RISK_AUDIT.html)
25. [E30 GEARS seed-overlap feasibility audit](./实验结果/E30_gears_seed_overlap_feasibility_audit_20260708/reports/E30_GEARS_SEED_OVERLAP_FEASIBILITY_AUDIT.html)
26. [E31 GEARS fixed-test split smoke](./实验结果/E31_gears_fixed_test_split_smoke_20260708/reports/E31_GEARS_FIXED_TEST_SPLIT_SMOKE.html)
27. [E32 GEARS fixed-test 3-seed smoke](./实验结果/E32_gears_fixed_test_3seed_smoke_20260708/reports/E32_GEARS_FIXED_TEST_3SEED_SMOKE.html)
28. [E97 Frangieh 三背景遗传扰动矩阵冻结合同](./实验结果/E97_frangieh_gene_cartesian_contract_20260713/reports/E97_CONTRACT_REPORT.md)
29. [E98 Frangieh 四类难设置预测与风险审计](./实验结果/E98_frangieh_gene_cartesian_predictions_20260713/reports/E98_REPORT.md)
30. [E99 三套多背景外部矩阵冻结合同](./实验结果/E99_multicontext_external_contract_20260713/reports/E99_CONTRACT_REPORT.md)
31. [E100 Lara/Santinha 遗传扰动外部复制](./实验结果/E100_gene_external_cartesian_predictions_20260713/reports/E100_REPORT.md)
32. [E101 三套遗传扰动矩阵元分析](./实验结果/E101_gene_cartesian_meta_audit_20260713/reports/E101_REPORT.md)
33. [E102 Cui 细胞因子直接映射子集合同](./实验结果/E102_cui_direct_mapping_contract_20260713/reports/E102_CONTRACT_REPORT.md)
34. [E103 Cui 六背景刺激预测与风险审计](./实验结果/E103_cui_cartesian_predictions_20260713/reports/E103_REPORT.md)
35. [E104 周老师要求逐项闭环](./实验结果/E104_advisor_requirement_closure_20260713/reports/E104_ADVISOR_CLOSURE.md)
36. [E108 正式 scGPT–GEARS 风险审计](./实验结果/E108_formal_dual_model_risk_audit_20260713/reports/E108_REPORT.md)
37. [E110 困难设置内层校准负结果](./实验结果/E110_nested_hard_calibration_audit_20260713/reports/E110_REPORT.md)
38. [E111 预测器依赖机制审计](./实验结果/E111_target_specific_mechanism_audit_20260713/reports/E111_REPORT.md)
39. [E112 外部五背景正式双模型复制](./实验结果/E112_external_formal_dual_models_20260713/E112_REPORT.md)
40. [E131 六 formal 数据集元分析](./实验结果/E131_formal_six_dataset_meta_20260714/reports/E131_REPORT.md)
41. [E114 split-conformal 任务误差上界](./实验结果/E114_split_conformal_error_bounds_20260713/reports/E114_REPORT.md)
42. [E132 六数据集分诊效用](./实验结果/E132_six_dataset_triage_utility_20260714/reports/E132_REPORT.md)
43. [E133 平均效应空间简单基线与方向误差审计](./实验结果/E133_systema_aware_baseline_audit_20260714/E133_REPORT.md)
44. [E134 Systema 表达空间精确定义审计](./实验结果/E134_systema_exact_expression_space_audit_20260714/E134_REPORT.md)
45. [E135 Directional-SafeConf 六数据 LODO 与冻结模型](./实验结果/E135_directional_risk_lodo_20260714/E135_REPORT.md)
46. [E136 Nadig 双细胞系冻结合同](./实验结果/E136_nadig_two_cellline_contract_20260714/reports/E136_CONTRACT_REPORT.md)
47. [E139 Nadig 第七数据方向确认](./实验结果/E139_nadig_directional_confirmation_20260714/reports/E139_REPORT.md)
48. [E140 七数据 absolute-RMSE 元分析](./实验结果/E140_formal_seven_dataset_meta_20260714/reports/E140_REPORT.md)

E10 当前口径：服务器已有 83 个 h5ad、约 107.3 GiB，官方 scPerturBench/scPerturb 元数据覆盖 66/66；KaggleCrossCell/Haber/Parekh 小型探针已跑通，但 learned risk 未超过 model disagreement，组合分数在 KaggleCrossCell 上较弱。它用于下一轮外部验证设计，尚不能作为最终投稿结论。

E12 当前口径：6 个外部候选中 `KaggleCrossPatient`、`crossPatient`、`sciplex3` 可按 held-out pair 评价；`kangCrossCell`、`kangCrossPatient`、`TCDD` 只有 1 个 perturbation，需改用 leave-context-out 或 dose/context split。可评估部分 overall 最强为 model_disagreement_risk，aligned Spearman = 0.734。

E13 当前口径：官方 `sciplex3_A549/K562/MCF7` 已合成为三细胞系 drug-dose focused panel。top-80 shared drug-dose 共 240 tasks、480 test records；model_disagreement_risk aligned Spearman = 0.576，learned_risk_score = 0.572。全量 743 shared drug-dose 需先优化高维 feature 计算。

E14 当前口径：官方 sciplex3 三细胞系 full-743 shared drug-dose 已在 1,000-gene 面板跑通。2,229 tasks、4,458 test records；learned_risk_score aligned Spearman = 0.862，model_disagreement_risk = 0.418。它是 full perturbation / low-gene 快速验证；5,000-gene full-743 仍需复跑。

E15 当前口径：同一 full-743 面板提升到 2,000 genes 后，learned_risk_score aligned Spearman = 0.899，model_disagreement_risk = 0.739，80% coverage RMSE 改善分别为 17.56% 与 17.06%。这说明 E14 信号不是 1,000-gene 单点偶然；5,000-gene 仍待正式复跑。

E16 当前口径：同一 full-743 面板提升到 3,000 genes 后，learned_risk_score aligned Spearman = 0.903，model_disagreement_risk = 0.702，80% coverage RMSE 改善分别为 19.14% 与 18.48%。1,000→2,000→3,000 genes 的连续性支持 chemical 外部验证信号稳定。

E17 当前口径：同一 full-743 面板提升到 5,000 genes 后，learned_risk_score aligned Spearman = 0.891，model_disagreement_risk = 0.692，simple_combined_confidence = 0.674；80% coverage RMSE 改善均约 18.7%。这可作为官方 sciplex3 full-743 5,000-gene 正式外部 chemical 验证证据。

E18 当前口径：GEARS 有部分可用 PredictionRecord + predicted/true NPZ，覆盖 Norman、Adamson、Dixit 的 54 条 single-gene 记录；scGPT 和 CPA 当前没有可直接进入 SafeConf 协议的逐任务预测向量。E18 用于界定真实多模型扩展的入口条件，不能写成 GEARS/CPA/scGPT 已完成统一验证。

E19 当前口径：已有 GEARS-only 结果可作为补充证据整理。Norman/Adamson/Dixit formal 54 条记录 overall aligned Spearman = 0.624；Frangieh run03 supplement 62 条记录 magnitude risk aligned Spearman = 0.941，但该来源带 smoke lineage，只能作为 supplement probe，不能写成主线 formal 多模型验证。

E20 当前口径：10 个现有预测输出 bundle 全部可做 non-strict 审计，但 0 个 strict pass。E17 的数组 key coverage 正常，严格失败主因是同一任务下不同预测器仍使用 record-scoped true_effect_key；旧 GEARS formal 输出严格失败主因是缺少 gene_panel_id、gene_order_hash、normalization_id 等来源字段。GEARS 导出脚本已补写这些字段，下一步应做 task-scoped true effect 和 scGPT/CPA adapter，而不是继续把旧结果说成统一多模型验证。

E21 当前口径：从 E17 抽样 60 条记录、30 个任务组，将 true_effect_key 改为 task-scoped 后 strict validator 通过，issue_count = 0，同一任务内 true effect 最大差异 = 0。E21 不是替换 E17 正式结果，而是证明下一轮 full rerun / shared benchmark adapter 应采用 task-scoped true effect。

E22 当前口径：修改后的 `confidence_task/run_confidence_mvp_v2_1.py` 新跑 Haber 200-gene smoke，生成 240 条 PredictionRecord、120 个任务组，strict validator 通过，issue_count = 0。E22 不是生物学性能结论，只说明未来 E17 类复跑会直接生成 task-scoped true effect。

E23 当前口径：已将 E22 strict-pass 输出固化为 shared benchmark adapter workbench，生成 120 个 task groups 的 `SHARED_BENCHMARK_TASK_MANIFEST.csv`，manifest checks 5/5 pass。E23 是 adapter 合同地基，不是三模型统一验证结果，也不是 GEARS biological benchmark。

E24 当前口径：E23 的 perturbation 是 Hpoly/Salmonella stimulus/timecourse，不适合直接给 GEARS 使用。`scgpt_env` 可以 import GEARS，本地有 Norman/Adamson/Dixit/Frangieh processed GEARS assets，并有 54 条 gene-perturbation-like legacy candidate rows。后续已由 E25 接上 GEARS strict remediation；不能跑 GEARS-on-E23。scGPT 仍需解决源码安装/权重，CPA/chemCPA 仍缺可执行资产。

E25 当前口径：已有真实 GEARS formal 输出已补齐 strict PredictionRecord provenance。Norman/Adamson/Dixit 共 9 个 formal runs、54 条 PredictionRecord、54 个 predicted effect arrays 和 54 个 true effect arrays，`validate_prediction_record_artifacts(strict=True)` issue_count = 0。E25 解决的是 GEARS 旧输出的严格合同与 gene order provenance 问题；它仍不是 GEARS/scGPT/CPA 统一多模型验证。

E26 当前口径：在 E25 strict GEARS 包上完成 GEARS-only 单模型风险审计。预测效应 abs-mean 风险 overall aligned Spearman = 0.679，80% coverage RMSE 改善约 34.2%；预测效应 L2 风险 aligned Spearman = 0.624，80% coverage 改善约 33.4%。GEARS native uncertainty 在 E25 formal records 中不可用。E26 是 GEARS-only 补充分析，不能写成多模型不确定性验证。

E27 当前口径：已用归档 scGPT 源码和 whole-human checkpoint 跑通 Replogle K562 essential forward-only smoke，生成 3 条 strict-pass PredictionRecord 和 predicted/true effect arrays，strict issue_count = 0。E27 证明 scGPT 可以进入 SafeConf adapter 合同；但它只是 forward smoke，不是正式 scGPT 性能结果，也还没有与 GEARS 放到同一任务/gene panel 对齐。

E28 当前口径：已在 Adamson 上完成 GEARS–scGPT 同任务、同 512-gene panel、同 task-scoped true effect 的 strict smoke。3 个扰动、2 个 predictor、6 条 PredictionRecord，strict issue_count = 0。E28 证明双模型合同对齐可行；它仍是 smoke，不是正式多模型 benchmark。

E29 当前口径：把 E25 中 Adamson fold-1 的 7 个可用单基因任务全部扩展为 GEARS–scGPT shared strict 合同。输出 14 条 PredictionRecord、512-gene shared panel、strict issue_count = 0。E29 新增任务级风险排序：GEARS–scGPT disagreement 与平均误差为弱正相关（Spearman = 0.357），true-effect magnitude 作为非部署诊断信号很强（Spearman = 0.964）。它说明双模型风险审计流程可跑通，但 n=7，不能当正式 benchmark。

E30 当前口径：审计 E25 的 3 个 GEARS formal runs 是否能直接支持 seed/ensemble uncertainty。结果显示 54 records 对应 47 个 unique task groups，其中 42 个 singleton，只有 5 个任务重复 ≥2 次、2 个任务重复 3 次；重复任务内 true effect 最大差异为 0。seed disagreement 在 5 个重复任务上与误差相关很高，但样本太小，不能写成正式 seed-uncertainty 证据。E30 的价值是给出固定任务、三 seed 重跑的必要性。

E31 当前口径：已给 GEARS PredictionRecord exporter 增加 `--test-perturbations-file` 和 `--run-type` 参数，并用 Adamson E29 的 7 个任务跑通 1-epoch fixed-test smoke。输出 7 条 `run_type=smoke` 的 strict PredictionRecord，固定清单全部命中，strict issue_count = 0。E31 不是性能 benchmark，只证明后续可以在同一批任务上做三 seed 正式重跑。

E32 当前口径：在 E31 的固定 test 清单上跑通 Adamson 7 tasks × GEARS 3 seeds × 1 epoch smoke，合并 21 条 strict PredictionRecord，7/7 任务三 seed 齐全，strict issue_count = 0。seed_disagreement_rmse 与 seed 平均误差 Spearman = 0.679，true-effect magnitude 诊断 Spearman = 0.964。E32 是 fixed-task seed-uncertainty 工作流 smoke，不是正式 GEARS 性能 benchmark。

E97 当前口径：回查 Frangieh 原始 h5ad 后确认 3 个真实细胞背景与 189 个共同单基因扰动，可形成完整 3×189 矩阵。三折合同只用标签、细胞数和哈希顺序冻结；每折包含训练、验证、随机缺失 pair、整行新背景、整列新扰动和双未见任务，并有 25%/50%/75%/100% 嵌套训练子矩阵。E97 只定义合同，不包含预测结果。

E98 当前口径：在 E97 上运行 SourceEffect-scGPTKNN 与 scGPTEmbedding-ContextRidge，形成 3,708 个任务行、7,416 条 strict PredictionRecord，issue_count=0。100% 训练量四 setting pooled 的校准 pair-risk ρ=.693，分歧=.596，magnitude=.643；但 outer-fold+perturbation cluster bootstrap 的 SafeConf−magnitude 区间为 [-.098,.255]，仍不能声称稳定超过强基线。validation q80 在双未见任务上也未降低误差。E98 完成矩阵 setting 与输入防泄漏验证，不冒充 GEARS+端到端 scGPT 重训。

E99 当前口径：只按 context、perturbation 标签和每 pair 细胞数冻结三套外部矩阵，未读取表达矩阵、效应、预测或误差。Lara ex vivo 为 5×31、Santinha 为 5×23，均属遗传扰动；Cui 为 6×86 细胞因子刺激。缺失标签 `nan`、Noise、NT1 已硬排除。共 16 个整行留出 fold、4,446 条 manifest 记录。

E100 当前口径：在 Lara ex vivo 与 Santinha 两套独立遗传扰动矩阵运行与 E98 同类的 embedding/transfer 双预测器，共 2,760 个任务行、5,520 条 strict PredictionRecord，issue_count=0。Lara pooled 校准 pair-risk ρ=.255，显著超过 magnitude；Santinha 校准后 ρ=.176，低于 frozen risk=.357 和 magnitude=.385。该结果证明 validation-only 校准具有数据集依赖，不能统一替换 frozen 分数。

E101 当前口径：不重新拟合权重，以 test truth 解封前已计算的 frozen pair risk 做三数据集元分析。Frangieh/Lara/Santinha 宏平均 ρ=.425，magnitude=.357，disagreement=.341。相对 disagreement 的 dataset-population cluster CI 为 [.017,.151]，稳定为正；相对 magnitude 的 CI 为 [-.057,.213]，仍跨 0。删除 Lara 后对 magnitude 的增量仅 .008，说明稳定强基线增量尚未闭环。

E102 当前口径：Cui 原矩形有 6 个免疫细胞背景、86 个细胞因子刺激。只保留经过大写、去连字符或去非字母数字字符后直接命中 scGPT 词表的 41 个标签；45 个商品名、复合亚基或别名全部排除，不做手工猜测。按 41 个刺激重新冻结 6 个整行 fold 和四档训练子矩阵。

E103 当前口径：Cui 41-stimulus 子集生成 2,832 个任务行、5,664 条 strict PredictionRecord，issue_count=0。100% 训练量 pooled 校准 pair-risk ρ=.413，frozen=.303，disagreement=.288，magnitude=.391；校准 risk 相对 magnitude 的 cluster CI 为 [-.104,.193]，只支持正趋势。该实验是 cytokine stimulus 独立线，不与 gene knockout 主表混合。

E104 历史口径：它在 E105–E114 完成前，将周老师追问逐项映射到早期证据。当时记录的“相对 magnitude、正式 GEARS/scGPT 和风险保证未闭环”已经由 E106–E194 更新。当前不得继续引用 E104 的旧 gate 判断，应以 `GATE_STATUS_20260729.md` 和 E194–E176 的正式结果为准。

E108–E140 当前口径：七套数据已完成 context-aware scGPT–GEARS 正式验证，共 3,209 个测试任务、32 个外层 folds。E140 中 SafeConf 相对 disagreement 的七数据 Δρ=0.133，dataset-population 95% CI `[0.045, 0.223]`；相对 magnitude Δρ=0.071，CI `[-0.056, 0.186]`。E135 方向风险头在六数据探索后冻结，E139 在 Nadig 第七数据确认通过：Pearson ρ=0.748、cosine ρ=0.757。absolute 和 direction 必须分开写；Nadig absolute 上 magnitude 更强。完整边界见 [录用判断与项目总账](./投稿准备/录用判断与项目总账_20260713/index.html)。

## 历史查证

需要找旧内容时用：

```bash
rg "关键词" /home/yyf/archive/safeconf
```
