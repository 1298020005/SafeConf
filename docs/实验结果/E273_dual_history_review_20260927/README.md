# E273 SafeConf 双历史方法审查：最终查看目录

这是 2026-09-27 本轮实验的唯一汇总目录。桌面上的审查报告是阅读入口：

`/home/yyf/Desktop/SafeConf_方法审查与实验结果_20260927.md`

## 实验合同

- 只使用原始 `fold_id=0` 的 `train/val` 行。
- 再按扰动名称做 5 个整体冷桶，同一扰动不会同时进入 fit 和 test。
- 没有读取 final test 行，也没有读取 final test treated truth。
- CPU 比较 `M`、`P`、`P+Q`、`P+Q+H`、`P+Q+E`、`P+Q+H+E`。
- 两张 RTX 6000 分别用不同随机种子运行轻量风险 MLP，检查风险学习层稳定性。

## 文件

- `CPU_SUMMARY.csv`：CPU 每个数据集、预测器和冷折的结果汇总。
- `CPU_MACRO_SUMMARY.csv`：CPU 宏平均结果。
- `CPU_CELL_BOOTSTRAP.csv`：21 个数据集-预测器单元的近似自助区间。
- `GPU_SEED273_SUMMARY.csv`、`GPU_SEED1273_SUMMARY.csv`：两张 GPU 的 MLP 汇总。
- `*_FOLD_RESULTS.csv`：逐冷折结果。
- `*_STATUS.json`：数据范围、设备和限制。

原始运行目录仍保留在同一 `实验结果` 目录下作为审计底稿；下午只需查看本目录和桌面报告。
