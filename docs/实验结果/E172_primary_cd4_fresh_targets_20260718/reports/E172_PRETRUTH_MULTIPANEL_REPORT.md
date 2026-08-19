# E172｜四面板 pretruth gate

正式状态：**POSTGATE_AUTHORIZED**。

Q01–Q04 均按预注册顺序完成，未读取 test targeting X，也未读取 column-unseen targeting X。四个面板的 G2 为 24/24、G3 为 96/96、G4 为 24/24，synthetic regression tests 为 40/40；所有 test query graph 均不含 `y`。

G4 使用三组 leave-one-seed-out two-seed family means，与最终三 seed family-mean SafeConf 的估计对象对齐。四面板全部注册单元中，最低 median pairwise Spearman 为 0.597419，最低 target-cluster bootstrap 95% CI 下界为 0.488382，均高于预先固定门槛。

本状态只授权一次性建立四个 F3 truth bundles 和联合 E172 检验，不代表 SafeConf 已优于 magnitude，更不构成部署授权。正式性能结论必须由全 800 个新目标、12 个 panel×state 的冻结联合终点给出。
