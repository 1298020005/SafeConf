# E190 pretruth 资产构建修复记录

首次构建在读取任何 `X` 之前停止。47 个冻结靶点中的 `DARS`、`HARS`、`MARS`、
`QARS`、`TARS` 使用旧基因符号，scGPT 词表使用对应现行符号 `DARS1`、`HARS1`、
`MARS1`、`QARS1`、`TARS1`。

修复保留全部 47 个任务和两个数据集中的原始表达列，只在 scGPT/GEARS 模型 token
层应用上述五个显式别名，并拒绝 panel 中出现 token 重复。任务、细胞、阈值和评价
方案均未修改。
