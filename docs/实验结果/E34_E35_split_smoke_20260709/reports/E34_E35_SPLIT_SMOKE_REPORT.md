# E34/E35 split smoke

生成时间：2026-07-09 20:17

## 已触发的内容

- 已检查/构建数据集：8
- 成功构建任务矩阵：7
- E34 submatrix split rows：36
- E35 row/column split rows：85

## 这一步的意义

这一步把周老师说的“小矩阵、整行、整列”从口头计划变成了真实 split manifest。下一步可以在这些 split 上重算 SafeConf 分数和 predictor error。

## 输出表

- `tables/E34_E35_DATASET_TASK_SUMMARY.csv`
- `tables/E34_SUBMATRIX_SPLIT_MANIFEST.csv`
- `tables/E34_SUBMATRIX_SPLIT_SUMMARY.csv`
- `tables/E35_ROW_COLUMN_SPLIT_MANIFEST.csv`
- `tables/E35_ROW_COLUMN_SPLIT_SUMMARY.csv`

## 明天汇报口径

我已经触发了新 setting 的数据准备：现有 P1/P2/P3 数据都在本地，不需要重新下载；先把小矩阵和整行整列的 split manifest 生成出来。下一步就是在这些 split 上计算 score 与真实 predictor error。
