# E174 runtime log

- 2026-07-19：F1 identity freeze PASS；800 个目标 expression X 读取数为 0。
- 2026-07-19：F1B prior-data method gate PASS；复合候选未达到 0.5% 效率增量和 75% 重复胜率，正式回退到 magnitude。E174 calibration/evaluation truth 均未使用。
- 2026-07-19：四个 F2 asset 均 PASS；每面板只读取 11,018 control、1,920 train 与 960 validation 行，held-out donor targeting X 读取数为 0。
- 2026-07-19：R02 pretruth gate PASS；R01、R03、R04 因 seen-160 的 G4 leave-one-seed-out 排序稳定性未达到 median Spearman 0.5 而 FAIL。G2、G3 与合成回归测试无失败，test query graph 含 y 数为 0。
- 2026-07-19：依照一次性四面板规则登记 `ABORTED_PRETRUTH_GATE`。F3A calibration 与 F4 evaluation 均不启动；E174 held-out donor targeting truth 读取数保持 0。
- 后续运行只追加事实记录；启动器、环境、失败和修正不得覆盖历史条目。
