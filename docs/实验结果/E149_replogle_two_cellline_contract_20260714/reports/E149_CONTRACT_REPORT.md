# E149｜元数据审计与冻结结果

- 两个源文件共有 2055 个非对照扰动标签；满足全部门槛 394 个。
- SHA-256 固定选择 128 个；跨细胞系最少细胞数 100，最少 batch 数 40。
- 合同 manifest 共 512 行，test=340，主分析 heldout-cell-line test=256。
- 主分析的 context × perturbation 唯一；后续按 perturbation 聚类 bootstrap。
- 目前只完成元数据冻结。尚未读取表达矩阵、构建资产、训练 scGPT/GEARS 或计算任何终点。
