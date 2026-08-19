# E105｜同背景对照图构建合同

E105 修复的是数据入口。Frangieh 每个任务都写成“细胞背景::扰动”唯一标识；基础表达只从该背景的 `ctrl` 细胞中确定性抽取，目标表达来自同一背景的扰动细胞。567 个冻结任务逐一通过检查，训练、验证和测试任务没有交集。

旧版 `PertData.create_cell_graph_dataset` 从全局 `ctrl_adata` 抽样，可能把 Control、IFNγ 和 Co-culture 混用。E105 不调用该路径。

## 当前检查

- 同背景对照：`True`
- 任务拆分互斥：`True`
- 测试目标参与优化：`False`
- 图数量：`567`；基因数：`512`
- 拆分：`{'test': 279, 'train': 258, 'val': 30}`

## 模型接口

`smoke` 模式会执行 scGPT whole-human 预训练权重加载、一个真实反向传播步骤和测试只读推理；GEARS 也执行一次前向/反向与测试只读推理。GEARS 在本实验只用 self-loop 图验证架构接口，不能作为正式 GEARS 性能。正式实验必须按每个外层 fold 的训练背景重新构造 GO/共表达图。

详细的逐任务控制池大小、目标池大小、行索引和扰动位点见 `tables/E105_GRAPH_PROVENANCE.csv`。
