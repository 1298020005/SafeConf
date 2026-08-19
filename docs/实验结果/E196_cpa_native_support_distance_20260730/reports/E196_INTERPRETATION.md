# E196｜解释边界

## 可以写

- 这是 CPA 0.8.8 自带语义下的训练支持距离审计。
- distance 与 magnitude 都对同一 CPA predictor 的 RMSE 评价，属于可比的
  same-outcome audit。
- frozen manifest 先各自计算，再等权宏平均；task-key cluster 重采样保持相同
  估计量，只作共享生物任务依赖的描述性敏感性。

## 结果如何读

- `native_cosine_distance` 的宏平均 Δρ=-0.201，数值低于 magnitude；manifest 描述性区间 [-0.319, -0.062]，task-cluster 描述性区间 [-0.358, -0.047]；两种描述性区间均在 0 以下，8 个 manifest 均可估。
- `native_euclidean_distance` 的宏平均 Δρ=-0.245，数值低于 magnitude；manifest 描述性区间 [-0.406, -0.077]，task-cluster 描述性区间 [-0.399, -0.092]；两种描述性区间均在 0 以下，8 个 manifest 均可估。

## 不能写

- 不能称作 predictive variance、校准置信度、误差概率或误差下界。
- 不能把 E94 或八个 manifest 写成九个独立外部数据集。
- 不能按观察到的 RMSE 改 reference set、距离、阈值或任务。
- 不能把 CPA–ridge disagreement 的 pair-mean target 与 CPA distance 的 own-RMSE
  target 合成同一个优劣结论。
- 不能把重采样描述性区间当作八个独立外部样本形成的常规置信区间。
