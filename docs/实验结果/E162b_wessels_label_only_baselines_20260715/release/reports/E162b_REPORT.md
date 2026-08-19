# E162b Wessels 解封前简单基线

- 48 个 test inputs 仅为 E160 canonical condition 字符串；
- 训练统计只来自 E161 的 11,779 个 train cells；
- validation/test expression 访问均为 0；
- 四个冻结预测器：control、cell-weighted perturbed mean、matching single mean、single additive；
- matching unique profiles：48；
- additive unique profiles：48；
- additive effect 与 2×matching effect 最大差：0；
- additive effect RMS 与 2×matching effect RMS 最大差：0；
- 风险列已统一为 higher expected accuracy；magnitude 与 SE 的原始正值和固定负号同时保留；
- PRESCRIBE predicted magnitude 不在本 runner 计算，后续从 E162 原值合并并固定用负 RMS 作为 confidence。

本阶段没有 test truth、effect、error 或评价指标，不能据此报告模型优劣。
