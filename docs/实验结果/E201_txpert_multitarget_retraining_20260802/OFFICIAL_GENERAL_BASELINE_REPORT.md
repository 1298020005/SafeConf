# E201 TxPert general baseline 预测前封存

- 任务：2008（主分析 1808）；
- source-context support：5,238；
- target 扰动表达访问：0 行；
- target truth：未打开；
- 计算：按 source 扰动细胞数加权的公开 MeanBaseline 单扰动分支；
- E200 公开类输出等价性最大绝对残差：2.7865171e-06；
- 等价性 RMS 残差：4.9673143e-08；
- 固定容差：5.0e-06，超过容差任务 0。

该结果是预测前强基线封存，不包含 E201 target error。正式评价会同时比较四种子 family centroid、该 general baseline、batch-matched control 和source-transfer baseline。
