# E222：直接拟合误差的跨研究验证

这次实际训练了轻量风险评分器，未重训上游扰动预测器。每次外层留一研究；内层六研究训练、一个研究验证。

| 输入与比较 | utility@20% 差值 | 描述性 95% 区间 | 正方向研究 |
|---|---:|---|---:|
| 完整特征相对 magnitude | +0.00167 | [-0.03734, +0.04070] | 4/8 |
| 完整特征相对 learned_magnitude | +0.00167 | [-0.03734, +0.04070] | 4/8 |

预定开发门：**NOT_SUPPORTED**。所有研究保留，未根据结果改变主指标或删除任务。

## 该结果的范围

- 历史评分标签来自双预测器平均 RMSE；不等同于 TxPert 平均预测误差，也不等同于五特征 E201 分数。
- 幅度单项学习组得到相同的标签和调参预算。完整组胜过原幅度而未胜过它时，不能把提高归给 SafeConf 特征。
- 区间在固定外层预测上按研究重采样；八个研究的训练集重叠，未包含完整重训练不确定性。
- 本轮是已公开数据上的开发检验。即使通过，也需要新的外部确认；E208 的主公式未修改。
- 正文图为白底 SVG/PDF/PNG；每个研究都画出，负方向不隐藏。

## 复算

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/scripts/run_e222_error_regression_transfer.py --input <E153_ABSOLUTE_TASK_INPUT.csv> --output <new_output_dir> --workers 4
```
