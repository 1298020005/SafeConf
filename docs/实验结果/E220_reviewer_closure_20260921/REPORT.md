# E220 同口径基线与实际预测误差复核

本分析只使用已公开 E205 表；所有结论属于事后稳健性分析。原始任务、误差和 E208 协议未改。

## 固定组合相对同口径幅度

基线在每个背景内先转百分位，与 SafeConf-M 使用相同的排名批次。宏平均表示每个背景各复核 20%，合并表示所有背景共用一份预算。

| 误差对象 | 汇总方式 | 指标 | 增量 | 配对簇自助 95% 区间 |
|---|---|---|---:|---|
| family_rms_error | pooled | spearman | +0.01685 | [+0.00911, +0.02522] |
| family_rms_error | pooled | utility_20 | +0.01391 | [-0.00790, +0.03833] |
| family_rms_error | macro | spearman | +0.02179 | [+0.01344, +0.03073] |
| family_rms_error | macro | utility_20 | +0.01506 | [-0.00905, +0.04602] |
| family_centroid_rmse | pooled | spearman | +0.01590 | [+0.00810, +0.02422] |
| family_centroid_rmse | pooled | utility_20 | +0.01301 | [-0.00887, +0.03784] |
| family_centroid_rmse | macro | spearman | +0.02082 | [+0.01240, +0.02986] |
| family_centroid_rmse | macro | utility_20 | +0.01410 | [-0.01002, +0.04547] |

## 图件与完整记录

- `scale_matched_comparison.svg`：原始幅度、同背景幅度百分位和固定组合，区分合并与背景等权。
- `context_and_centroid_controls.svg`：四个背景、两种误差对象和配对区间。
- `ablation_and_weight_sensitivity.svg`：单项对照、删除分量和全部候选权重。
- 每张图提供 PDF、PNG；图内无 Figure 编号。
- `SUMMARY.csv`：全部方法、全部背景、主任务与全任务敏感性分析。
- `PAIRED_INTERVALS.csv`：每个方法相对同口径幅度的探索性区间。

## 解释边界

这些配对区间固定四个观察背景；同一扰动跨背景一起抽样。不能将其解释为新背景总体保证。
消融和权重结果用于理解机制，不按最好的结果重定义主方法。当前确认性外部协议仍为 E208。
E205 沿用 E201 已公开的真实标签，因此属于新增预测结构检验，不是全新独立标签数据的盲测。
真实预测向量的误差没有被风险排序降低；效用指固定预算内优先找到错误预测。
