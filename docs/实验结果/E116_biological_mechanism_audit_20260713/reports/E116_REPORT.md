# E116｜高风险任务与预测器失效机制

该分析不改变 SafeConf 分数。它解释哪些部署前特征与 GEARS 相对 scGPT 的额外错误同向，并把高风险扰动基因转成可复核通路线索。

## 三数据集等权关联

| feature | ρ with GEARS−scGPT error | ρ with GEARS error | ρ with scGPT error |
|---|---:|---:|---:|
| context_novelty_scaled | 0.195 | 0.051 | 0.014 |
| safeconf_calibrated_pair_risk | 0.156 | 0.262 | 0.169 |
| risk_model_disagreement | 0.084 | 0.073 | 0.024 |
| baseline_predicted_magnitude | 0.043 | 0.073 | 0.036 |
| perturbation_novelty | 0.030 | 0.060 | -0.001 |
| training_support_count | -0.030 | -0.060 | 0.001 |

## 高风险基因重叠

这里的重叠指同时进入 SafeConf 风险前 20% 与 GEARS excess error 前 20%。

- Frangieh: 16 个；IFNGR1, JAK2, SLC19A1, TGFB1, SLC25A13, DAG1, FMN1, UCN2, RAB27A, SLC5A3, C19orf48, NGFR, ATP1B1, IRF3, SLC26A2, GPNMB。
- Lara_exvivo: 0 个；无。
- Santinha: 0 个；无。

## 细胞背景失效差异

| dataset | context | biological meaning | tasks | GEARS error | scGPT error | GEARS−scGPT |
|---|---|---|---:|---:|---:|---:|
| Frangieh | Co-culture | 肿瘤细胞与免疫细胞共培养状态 | 282 | 0.0537 | 0.0493 | 0.0044 |
| Frangieh | IFNγ | 干扰素-γ刺激状态 | 285 | 0.0576 | 0.0533 | 0.0043 |
| Frangieh | Control | 未额外刺激的基线状态 | 270 | 0.0586 | 0.0552 | 0.0033 |
| Lara_exvivo | GMP (late) | 较晚期粒细胞-单核细胞祖细胞 | 68 | 0.1361 | 0.1378 | -0.0016 |
| Lara_exvivo | GMP | 粒细胞-单核细胞祖细胞 | 68 | 0.0947 | 0.0987 | -0.0041 |
| Lara_exvivo | HSC | 造血干细胞 | 69 | 0.0842 | 0.0919 | -0.0078 |
| Lara_exvivo | MkP | 巨核细胞祖细胞 | 68 | 0.1003 | 0.1083 | -0.0081 |
| Lara_exvivo | EBMP | 红系/嗜碱/巨核祖细胞群 | 72 | 0.0884 | 0.1004 | -0.0120 |
| Santinha | Neurons_L_5 | 皮层第 5 层神经元 | 52 | 0.0423 | 0.0417 | 0.0006 |
| Santinha | Neurons_L_2_3 | 皮层第 2/3 层神经元 | 49 | 0.0422 | 0.0418 | 0.0004 |
| Santinha | Neurons_L_6 | 皮层第 6 层神经元 | 52 | 0.0543 | 0.0561 | -0.0018 |
| Santinha | Interneurons_Sst_Pvalb | Sst/Pvalb 类中间神经元 | 51 | 0.0461 | 0.0484 | -0.0023 |
| Santinha | Interneurons_Vip_Adarb2 | Vip/Adarb2 类中间神经元 | 51 | 0.0750 | 0.0788 | -0.0037 |

背景新颖度与 GEARS−scGPT 额外误差的三数据集宏平均相关为正，说明跨细胞状态的基础表达变化是当前最清楚的失效来源。该结果来自任务级误差关联，不能进一步写成某条通路导致模型失败。

## 通路富集

没有满足预设阈值的通路，或远程注释查询失败。审计状态见 `tables/E116_ENRICHMENT_AUDIT.json`。

## 解释边界

这些结果是机制假设，不是因果证明。高风险基因富集使用各数据集全部被测扰动基因作背景；不同物种不混合查询。后续湿实验或外部功能证据应围绕重叠基因与显著通路，而不是只挑单个好看的案例。
