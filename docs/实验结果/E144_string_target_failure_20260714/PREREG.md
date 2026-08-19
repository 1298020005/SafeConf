# E144 预注册｜STRING 知识距离与靶基因自身恢复失败

本分析由 TxPert 2026 报告的两个失败因素触发：知识图谱位置和未见扰动靶基因自身下调。STRING v12.0、combined score≥700、训练目标集合、网络特征和 SafeConf 分数先冻结，再打开保存的目标基因真值。

- 主终点：原 SafeConf 与两个预测器对被扰动基因自身的平均绝对误差。
- 支持终点：只在训练未见目标中，STRING 到训练目标的最短距离与全向量平均 RMSE。
- 统计单位为 perturbation；先 fold 等权，再 dataset 等权；3,000 次 dataset→perturbation 整簇 bootstrap。
- 主 gate：七数据主相关为正，且 95% CI 下界大于 0。若失败，保留失败；不改阈值或分数。
- degree 和 distance 的基因标签置换为机制负对照；这不是 TxPert 的重训练或因果网络证明。
- 该假设是在已有数据完成整体误差分析后由新文献触发，属于冻结后的二级机制审计，不冒充全新独立验证。

依据：[TxPert, Nature Biotechnology (2026)](https://www.nature.com/articles/s41587-026-03113-4)；[STRING v12 download](https://string-db.org/cgi/download.pl)。
