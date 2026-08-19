# E174 pretruth gate 结果

E174 在测试真值打开前终止。R02 通过全部预注册门；R01、R03、R04 的 G2 分数可变性、G3 预测非塌缩与合成回归测试均通过，但 seen-160 的 G4 leave-one-seed-out family-mean 排序稳定性不足。四个面板共 24 个 G4 单元，其中 8 个失败，全部来自 seen-160。

R01 的失败最明显，三个状态的 median pairwise Spearman 为 0.324–0.351；R03 为 0.477–0.481；R04 的 Stim8hr/Stim48hr 为 0.498 左右，仍低于预注册阈值 0.5。R02 的最小值为 0.808。该差异说明三 seed family mean 在轮换训练供体后仍有明显目标排序方差，不能把一组固定 seed 当成稳定风险量。

本阶段 held-out donor targeting X、calibration truth 和 evaluation truth 读取数均为 0，test query graph 含 y 数为 0。根据四面板必须全部通过的规则，不允许只解封 R02，也不允许在看到 G4 后把 0.5 改为更低阈值；F3A 与 F4 均未启动。

这次失败可以作为无标签模型开发证据：下一协议若继续，必须在新目标上预先固定更大的 seed ensemble、明确计算预算和新的稳定性估计器，再从零运行 pretruth gate。E174 的 640 个最终评价目标继续保持未读，不能被写成一次阴性性能实验。
