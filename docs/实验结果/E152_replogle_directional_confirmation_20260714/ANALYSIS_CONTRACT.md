# E152 分析合同｜Replogle 方向风险确认

E149 已固定 128 个扰动、两个细胞系留出折、256 个唯一主任务、E135 模型哈希和 gate。E152 第一阶段只读取四个部署特征并冻结方向风险；第二阶段才读取预测与真实向量。

主终点为两模型平均的 Systema-centered Pearson error 与 cosine error。每折转换为 percentile rank 后取平均形成复合方向误差。按 128 个 perturbation 整簇 bootstrap 3,000 次；同一基因的 K562/RPE1任务同步重采样。gate 要求两个终点宏平均均大于0，且复合终点95%区间下界大于0。

该数据没有参与 E135 方向模型开发，但两个细胞系属于同一研究，目标 control 可见；结果只支持 control-observed 跨细胞系复制，不能写成跨研究或完全 zero-shot。
