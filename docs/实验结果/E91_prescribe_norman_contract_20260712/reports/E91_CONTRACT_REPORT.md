# E91｜PRESCRIBE × Norman 双面板冻结合同

PRESCRIBE 将在 Norman 的两套既有不重叠面板上分别训练。面板来自 E67 与 E76b，早于本轮 PRESCRIBE 接入；本轮没有根据 PRESCRIBE 的预测、误差或不确定性挑任务。

| panel | train | val | test | test 最少细胞 | 与另一面板重叠 |
|---|---:|---:|---:|---:|---:|
| Norman_P1 | 183 | 20 | 24 | 241 | 0 |
| Norman_P2 | 183 | 20 | 24 | 230 | 0 |

主要比较固定为：

1. PRESCRIBE epistemic、aleatoric 与论文组合分数，对 PRESCRIBE 自身 RMSE 的排序；
2. PRESCRIBE predicted magnitude 对同一 RMSE 的排序；
3. 拒绝最高风险 10%、20%、30% 后的 remaining error 与 AURC；
4. 在相同 48 个任务上并列展示既有 GEARS–scGPT disagreement，但明确二者对应不同 predictor error，不能混成一条相关系数。

PRESCRIBE 是 integrated predictor；SafeConf 是异构 predictor 的 post-hoc pair-risk。公平比较单位是“各自的风险分数能否排序各自预测器的错误”，并同时给 magnitude 基线。不能让 PRESCRIBE uncertainty 去解释 GEARS 或 scGPT 的误差。
