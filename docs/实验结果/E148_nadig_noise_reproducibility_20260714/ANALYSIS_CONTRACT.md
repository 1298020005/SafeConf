# E148 分析合同｜Nadig 实验噪声与模型风险

在不打开表达矩阵数值的阶段冻结 96 个基因、两个细胞背景、guide/batch 元数据和全部风险分数。

- 每个 context×perturbation 的细胞在 batch 内随机二分；control 同样独立二分。
- 在 E138 固定 512 基因轴上重复 50 次，计算 split-half RMSE、centered Pearson error、centered cosine error。
- 主诊断：控制 split-half 方向噪声和 log(n_cells) 后，Directional-SafeConf 与模型方向误差的 partial Spearman。
- 补充：高复现任务子集、风险与实验噪声直接关联，以及满足每组≥10细胞的跨 transcript/guide 一致性。
- 同一扰动基因跨背景、fold 和重复整体聚类 bootstrap；细胞不是统计独立单位。
- 本数据已用于 E139，故属于新技术终点审计。结果不能重新包装为独立确认。
