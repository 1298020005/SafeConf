# E156｜PRESCRIBE P3/P4 train-only 预处理报告

## 完成状态

E156 只完成预处理和官方 PertData 兼容资产生成。没有建立或训练模型，没有生成置信度、预测、任务误差、Pearson、方向准确度或 RMSE。

| panel | development cells | test cells（metadata only） | model genes | train conditions | val | test |
|---|---:|---:|---:|---:|---:|---:|
| Norman_P3 | 44132 | 13543 | 2044 | 64 | 20 | 24 |
| Norman_P4 | 44132 | 16489 | 2044 | 64 | 20 | 24 |

## 泄漏控制

HVG、基因检测阈值、PCA 均只在两个面板共享的 64 个训练条件上拟合。模型轴由 train-only top-2000 HVG 与 E155 已冻结的扰动基因身份取并集，共 2044 个基因；没有用 val/test 表达决定基因是否入轴。

每个开发细胞使用固定 `target_sum=10000` 后 `log1p`，没有从 val/test 估计全局归一化常数。PCA10 的均值和 components 只来自 train。训练 E-distance 在 train 内按每条件 54 个细胞平衡后计算；val 的 `y_n/y_d/y_s` 是 NaN sentinel。48 个 held-out 测试基因在 train/val 的命中数均为 0，逐基因证明见 `tables/E156_GENE_LEAKAGE_AUDIT.csv`。

E156 为整文件 SHA256 校验读取了原始 H5AD 字节，并以 backed 模式读取 `obs` 元数据；测试 X 行从未被索引、载入内存或执行变换。表中的测试细胞数只来自 E155 已冻结的 `obs`。正式运行时 `perturb_processed.h5ad` 和 `cell_graphs.pkl` 均只有 train+val。E157 必须先锁定 checkpoint 和每任务置信度；随后由 E158 才能读取测试 X，执行固定归一化与 train-PCA transform 并评价。开发资产中的上游 callback 兼容字段使用固定 train-HVG 顺序，不能把它当作 condition-specific top-DE。

两个源码边界已经在训练前登记。第一，native `test_step` 的 `truth` 是 `y_pca` 经 PCA10 逆变换后的重构值，因此 P3/P4 的预注册 Pearson 主终点必须继续用 PCA10-reconstructed truth；raw log-normalized truth只能作敏感性分析。第二，native `ListMLELoss` 对按 `y_n` 排序后的整列做 `-sum(log_softmax)`，该和对排列不变，当前源码中的 E-distance 排序标签实际上不会改变 loss。E156 仍生成 train-only `y_n` 以保持原生接口，但不能宣称它提供了有效的排序监督。逐文件证据见 `tables/E156_NATIVE_CODE_AUDIT.csv`。

## PertData 兼容性

大文件位于 `/home/yyf/data/safeconf_e156_prescribe/`。PRESCRIBE 的 `data/norman_p3` 和 `data/norman_p4` 是指向该目录的符号链接。每个 panel 的运行时 `perturb_processed.h5ad` 和 cell graphs 只含 train+val；E156 不生成测试表达资产。其余资产包括 train-only `perturb_e_distance.h5ad`、冻结 split、PCA prior mean/cov 和训练 E-distance 表。

- Norman_P3: 84 conditions / 44132 cell graphs；SHA256 `ab6a2357afbbf446543b070dc68c56d8cb21d02d2e0f646a0bc38eb1bec27550`
- Norman_P4: 84 conditions / 44132 cell graphs；SHA256 `70163d5da327fe33d2a4a79de7176e2d67f8ec5500c9db755a4119b366eea188`

全部 25 个计算与数据资产的路径、字节数与 SHA256 在 `tables/E156_ARTIFACT_HASHES.csv`。该表不把脚本、状态、报告和自身计入计算资产。运行脚本 SHA256 与 Git HEAD blob 一致性另存于 `RUN_STATUS.json`。完整 33,694 基因审计、split 输入审计、X/obs provenance 和 train E-distance 均已单独保存。
