# E176 donor-specific calibration

联合 pretruth gate 提交并通过后，只开放每位测试供体预分配的 40 个校准靶点，共 160 个靶点、480 个任务；640 个评价靶点 targeting X 读取数仍为 0。

每位供体单独计算 target-cluster residual 的第 37 顺序统计量。基础模型仍是 E174 在任何 E176 真值开放前冻结的 magnitude 规格；本阶段没有重选特征、阈值或供体。pair mean/max 下界违例均为 0，平方误差分解最大残差为 8.39e-11。
