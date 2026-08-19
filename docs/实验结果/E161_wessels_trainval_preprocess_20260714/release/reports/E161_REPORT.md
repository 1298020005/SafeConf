# E161 Wessels train/validation 预处理报告

- train: 72 conditions / 11,779 cells;
- validation: 24 conditions / 5,102 cells;
- test: 48 conditions / 9,902 cells，expression 未索引、未物化、未转换；
- 容器共 21,052 features；8 个实验构造和 413 个 guide/barcode 列均未读取；
- 归一化分母：每个细胞前 20,631 个内源基因的 raw-count library；
- 原始 `obs[ncounts]` 是上游 `nCount_RNA` 元数据，只作差值审计，不作为分母硬门控；
- feature: train-only seurat_v3 top-2,000 与 27 个 train-single genes 并集；
- PCA/control prior/E-distance: 全部 train cells 拟合；
- E-distance: unequal-n moment formula，不做 15-cell 平衡；
- development graphs: 96 conditions / 16,881 graphs，test graph = 0；
- 本阶段未训练模型，未产生 test prediction 或 endpoint。
