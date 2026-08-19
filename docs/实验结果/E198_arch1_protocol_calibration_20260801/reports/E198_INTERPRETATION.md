# E198 结果解释

E198 回答的是“这个指标在 arch1 上能否把技术重复与无信息参考分开”，不是“哪个
预测模型最好”。BDS 检查方向，DRF 检查恢复了多少有效动态范围；两个条件都稳定
通过才进入 E199 的主要端点。

固定优先级实际选择如下：

- absolute: `mse`
- direction: `pearson_pert`
- retrieval: `rank`
- population: `energy_distance_pca_k=50`
- de: `de_auprc`

即便多个协议通过，也不能把它们当作相互独立的生物证据；它们使用同一批 150 个
扰动。`arch1` 只有 H1 hESC 一个背景，下一轮只能做未见扰动，不能把结果改写成整行
或跨细胞背景泛化。群体和 DE 协议使用真实细胞，不是由 centroid 复制出来的伪细胞。
