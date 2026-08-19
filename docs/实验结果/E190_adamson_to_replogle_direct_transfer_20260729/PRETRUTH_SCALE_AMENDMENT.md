# E190 pretruth 输入尺度修正

时间：2026-07-29  
发生阶段：模型资产构建；Replogle 扰动表达读取数仍为 0。

第一次尺度审计显示，公开 H5AD 的 `X` 不是可直接用于模型的 log 表达：

- Adamson control：范围 0–1446，median 8；
- Adamson 扰动训练细胞：范围 0–3449，median 8；
- Replogle control：范围 0–1475，median 5。

原计划中“直接使用处理后 X”的假设不成立，第一次资产作废。冻结修正规则为：每个
细胞使用其全基因行和计算 library size，缩放到 10,000 后执行 `log1p`；Adamson
control、Adamson 扰动训练细胞、Replogle control 以及之后封存的 Replogle 扰动
细胞使用同一公式。规则不读取目标扰动真值，不做跨研究均值对齐、ComBat 或任何
依赖目标扰动表达的校正。
