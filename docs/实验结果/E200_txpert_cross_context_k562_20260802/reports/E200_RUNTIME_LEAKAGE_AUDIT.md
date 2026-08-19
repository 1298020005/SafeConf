# E200 运行时目标泄漏审计

- 执行时间：2026-08-02 04:20:05–04:21:13（Asia/Shanghai）
- 预测器：TxPert 官方 `K562_unseen_cell_gat.ckpt`
- 设备：CPU
- 样本矩阵：8 个细胞 × 3,352 个基因

同一批次先保留 K562 扰动真值做前向，再将真值张量全部清零后重复前向。两次预测逐元素完全相同，最大绝对差为 0；对照输入在前后两次运行中也未变化。

该检验证明这个 checkpoint 的预测前向不依赖批次中保存的目标扰动真值。它不替代拆分审计；拆分、静态代码和运行时三层证据已分别通过。

原始运行产物位于 `DATA/txpert_official_20260802/e200/leakage_smoke_gat_cpu/`，不进入 Git。原始 manifest SHA-256 为 `ed7b8a2880c9a01cbd5daa2ed6eb2968ffe8a2dd85b44e58ccaa43cd48e88466`。
