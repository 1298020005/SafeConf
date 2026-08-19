# E88｜sciPlex3 → sciPlex4 同族外部合同

这个合同使用同一研究系列中的两个独立筛选文件。源域为 sciPlex3，目标域为 sciPlex4；只保留两边都出现的 Abexinostat、Pracinostat，以及 A549、MCF7 共同细胞系。sciPlex3 额外提供 K562 源训练任务。

- 源训练任务：24（3 个 context × 2 个药物 × 4 个剂量）
- 目标测试任务：28（2 个 context × 2 个药物 × 7 个剂量）
- 目标精确剂量任务：16
- 目标插值剂量任务：12
- 共同基因：58347；冻结 panel：1000
- panel 选择：两个数据集的 control-only 检出率与方差；未读取 sciPlex4 扰动表达
- gene hash：`sha256:b176450e8f3d363060f0beb82bc1c1c40e98d3633e0a802fd6f1e9f91b945172`

E87 检验的是无共享药物、无共享细胞体系的强外推；E88 检验独立实验批次间的可迁移性。E88 任务数只有 28，定位为同族外部复核，统计结果必须给 bootstrap 区间，不能替代大规模跨域证据。
