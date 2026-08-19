# E95｜PRESCRIBE Norman 原生训练

正式结果位于 `norman_p1_formal_seed3407/` 与 `norman_p2_formal_seed3407/`。两套冻结面板各 24 个测试任务，P1 共 12,721 个测试细胞，P2 共 11,256 个测试细胞。

P1 在主训练 epoch 7 早停，最佳检查点来自 epoch 4；P2 在 epoch 22 早停，最佳检查点来自 epoch 19。两者均使用 5 epoch flow 预热、作者原生模型与损失、batch size 4096、seed 3407。

`norman_p1_formal_seed3407_aborted_go_filter/` 保留一次中止记录：官方 GEARS 数据层按 GO 表静默删除了冻结任务 `IER5L+ctrl`，运行在测试前停止。记录补丁后重建缓存，P1/P2 测试 DataLoader 均核验为 24/24。

模型检查点、TensorBoard 日志和逐细胞原始 NPZ 体积约 3 GB，只保留在本地并由 `.gitignore` 排除；Git 保存状态、哈希、任务级记录与完整复现脚本。双面板统计结论见 `../E96_prescribe_native_comparison_20260713/reports/E96_REPORT.md`。
