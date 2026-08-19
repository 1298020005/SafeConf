# E167｜风险可识别性与适用性证书（RIAG v1）

预设开发 gate：`FAIL`。本结果只证明证书能识别不可估计与基线等价情形，不证明通过后的分数一定可靠。

## 真实塌缩案例

| unit_id | score_unique | min_quantized_unique_vectors | authorization_status | candidate_vs_loss_spearman |
|---|---|---|---|---|
| E159::Norman_P3::official | 1 | 1 | ABSTAIN_SCORE_SATURATION | NA |
| E159::Norman_P3::raw_log_prob | 24 | 1 | ABSTAIN_PREDICTOR_COLLAPSE | 0.1774 |
| E159::Norman_P4::official | 1 | 1 | ABSTAIN_SCORE_SATURATION | NA |
| E159::Norman_P4::raw_log_prob | 24 | 1 | ABSTAIN_PREDICTOR_COLLAPSE | 0.2904 |
| E165::Wessels::PRESCRIBE_seed3407 | 48 | 1 | ABSTAIN_PREDICTOR_COLLAPSE | -0.2097 |
| E165::Wessels::PRESCRIBE_seed3408 | 48 | 1 | ABSTAIN_PREDICTOR_COLLAPSE | 0.0217 |
| E165::Wessels::PRESCRIBE_seed3409 | 48 | 1 | ABSTAIN_PREDICTOR_COLLAPSE | -0.1616 |

Wessels raw score 三 seed 的两两 Spearman 中位数为 `0.0225`，bootstrap 95% CI `[-0.2053, 0.3280]`，Kendall W=`0.4401`；G4=`FAIL`。

## 非塌缩参考

| unit_id | score_unique_fraction | prediction_unique_fraction_min | score_vs_magnitude_spearman | authorization_status |
|---|---|---|---|---|
| E153::Frangieh | 0.6714 | 1.0000 | 0.4372 | ELIGIBLE_G2_G3_ONLY |
| E153::Lara_exvivo | 1.0000 | 0.9855 | 0.5771 | ELIGIBLE_G2_G3_ONLY |
| E153::Liang | 0.9020 | 0.9150 | 0.4108 | ELIGIBLE_G2_G3_ONLY |
| E153::Nadig_two_cellline | 0.9492 | 1.0000 | 0.3045 | ELIGIBLE_G2_G3_ONLY |
| E153::Replogle_two_cellline | 0.4883 | 1.0000 | 0.1877 | ABSTAIN_SCORE_SATURATION |
| E153::Santinha | 0.2863 | 0.9961 | 0.6321 | ABSTAIN_SCORE_SATURATION |
| E153::Shifrut | 1.0000 | 1.0000 | 0.9281 | ELIGIBLE_G2_G3_ONLY |
| E153::Tian_CRISPRi | 1.0000 | 0.9604 | 0.1409 | ELIGIBLE_G2_G3_ONLY |
| E96::Norman_P1 | 1.0000 | 1.0000 | 0.9965 | ELIGIBLE_G2_G3_ONLY |
| E96::Norman_P2 | 1.0000 | 1.0000 | 0.9939 | ELIGIBLE_G2_G3_ONLY |
| E87::sciPlex3_to_OpenProblems | 1.0000 | 1.0000 | 0.9994 | ELIGIBLE_G2_G3_ONLY |
| E89::sciPlex3_to_sciPlex4 | 1.0000 | 1.0000 | 0.8708 | ELIGIBLE_G2_G3_ONLY |

## 必要条件不等于准确率保证

E87 的 score 和预测向量通过 G2/G3，但任一预测器优于 no-change 的任务比例为 `0.0000`。因此 RIAG 通过只授权评价流程继续，不能替代上游预测器有效性检查。

## 解释边界

Norman、Wessels 和其余历史资产真值均已解封，E167 属于方法开发。下一项确认必须在新数据 test truth 前生成证书；G2/G3/G4 失败后，任何测试相关性都不得覆盖拒绝状态。结构风险可以在 predictor collapse 后单独研究，但不能再称为模型内生 uncertainty。
