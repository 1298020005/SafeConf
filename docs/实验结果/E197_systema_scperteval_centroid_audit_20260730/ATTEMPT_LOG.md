# E197 执行记录

## Attempt 1｜2026-08-01

- 冻结提交：`91c739c9b7cfbf6cae1dd165f69428161db4fb87`；
- GitHub 与 Gitee 在运行前均已核对到同一提交；
- 状态：`FAILED_POSTTRUTH_EXPLORATORY`；
- 停止门槛：`E190_K562 / scperteval_pearson_pert_matches_independent_formula`；
- 正式表、图、报告与输出哈希均未发布。

诊断显示，官方 scPertEval 在 `Dataset.all_perturbed_mean_except()` 中先对 float32
perturbation-centroid 矩阵求和，再构造 leave-one-out reference；Attempt 1 的独立
公式先转换为 float64 后求和。两条路径的 NA mask 完全相同，E190 最大绝对差为
`2.390424965956206e-08`，E192 为 `1.4611718834878218e-08`。

修正方案不是放宽门槛。独立公式保留同一数学定义，并显式复现输入矩阵的数据类型
与求和顺序。诊断重算后，E190/E192 的最大差分别为
`1.1102230246251565e-16` 和 `2.220446049250313e-16`。修正 runner 后重新执行
synthetic smoke、prepare、提交、双远程推送与正式分析。
