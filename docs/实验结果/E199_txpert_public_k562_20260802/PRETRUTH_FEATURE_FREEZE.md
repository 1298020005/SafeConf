# E199 风险特征生成冻结

冻结时间：2026-08-02

## 输入边界

特征程序只能打开：

1. GAT、Exphormer 和 Exphormer-MG 的逐细胞预测；
2. GAT 运行保存的 batch-matched control；
3. TxPert 官方 general baseline 预测；
4. 官方 train/test split、STRING/GO 图和扰动基因集。

目标表达文件不得出现在脚本常量、读取路径或特征函数中。输入哈希、脚本
和本冻结文件必须已提交，且本地、GitHub、Gitee 的分支顶端完全一致。

## 特征定义

- `family_diversity`：三个模型的扰动级 centroid 相对等权家族 centroid
  的平均每基因平方离差；
- `diversity_lower_bound`：`sqrt(family_diversity)`；
- `predicted_magnitude`：家族 centroid 与同一扰动对应的 batch-matched
  control centroid 之间的 RMSE；
- `model_baseline_gap`：家族 centroid 与官方 general baseline centroid 的
  RMSE；
- `string_train_neighbor_count` / `go_train_neighbor_count`：先按 TxPert
  配置将图约化到扰动基因集，再为每个 target 保留权重最高的 20 条入边；
  将处理后图视为直接相连关系，计数与训练扰动基因相连的不重复邻居；
- `graph_isolated`：两张图的训练邻居数均为 0；
- `historical_support`：未见扰动必须全部为 0；
- `context_similarity`：K562 单背景固定写为
  `NOT_APPLICABLE_SINGLE_CONTEXT`。

图边计数是支持度特征，不是生物机制因果解释。如果它与误差无稳定关联，仍然保留
结果并返回无经验支持的结论。
