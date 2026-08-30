# E204 先看这个

E204 是在 E201 盲预测之后新增的训练方向：把训练数据中“支持少、背景覆盖窄、不同 source 背景反应差异大”的扰动条件标为难任务，提高它们的训练权重，再和原始 TxPert-GAT 比较。

当前只完成协议冻结和代码准备，尚未有 E204 结果。E201 的 target 真值仍然封存，任何“训练后变好”都不能提前写成结论。

执行顺序：

1. 完成 E201 四个细胞系 × 四个 seed 的盲预测；
2. 封存预测分歧、预测幅度、source support 等预测前特征；
3. 生成每个 target 的 source-only 任务权重清单并做列级审计；
4. 先跑 `uniform` 与 `risk_weighted` 的 source smoke/profile，再跑正式四 seed；
5. 三种训练条件全部封存后，才释放 E201 target 真值并评价。

权重生成脚本只读取 E201 的 source 字段，不读取 `n_target_cells`、target expression 或任何 E199/E200 结果。
