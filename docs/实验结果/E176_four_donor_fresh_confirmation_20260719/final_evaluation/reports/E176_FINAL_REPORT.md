# E176 final four-donor evaluation

正式状态：**CERTIFICATE_AND_EMPIRICAL_COVERAGE_AUDIT_PASS**。主要评价只包含 640 个从未参与开发或校准的靶点、1,920 个任务；四位供体各贡献 160 个靶点。

模型对下界在 pair mean 与 pair max 上的违例均为 0；平方误差分解最大残差为 9.81e-11。冻结 magnitude 基础模型加供体专属 conformal 校准后，三个状态同时覆盖的总体 target-level 经验覆盖率为：ensemble RMSE 0.903，pair-mean RMSE 0.903，目标值 0.90。逐供体 Clopper–Pearson 区间和上界宽度均保留在表中，没有根据评价真值调整分位数。

fixed SafeConf、magnitude 与 disagreement 的 Spearman/AURC 仅作诊断。E176 属于同一 Primary CD4 研究的多供体内部确认，不能替代独立研究、湿实验或临床验证。
