# E172｜修正 seed gate 后的未读目标确认

## 背景

E168 的 200-target test 得到小幅正点估计但未确认；E170 的四个新面板在读取 test truth 前因 single-seed-pair G4 不稳定而正式终止。E171 只用 E170 pretruth 预测和允许的 validation effects 发现：最终部署分数使用三 seed family mean，而原 G4 比较单 seed 配对，二者估计器不一致。三组 leave-one-seed-out two-seed family means 在 E170 的 24/24 单元通过同一稳定性阈值。validation 上 SafeConf 相对 magnitude 仍不确定，因此 E172 只修正 gate 对齐，不修改 SafeConf 分数，也不预设性能会改善。

## 新目标与隔离

- 从 5,510 个 label-free eligible targets 中排除 E168 的 200 和 E170 的 800，剩余 4,510 个。
- 用新的身份 SHA-256 一次性固定 Q01–Q04 共 800 个目标；每个面板 160 seen + 40 column-unseen。
- 选择不使用 X、counts、effect、error、DE、guide efficacy 或 E171 validation 表现。
- E168、E170 目标不进入 E172；E170 的 800 个 test outcomes 继续保持未读。

## 模型、分数和修正后的 G4

scGPT、GEARS、seeds 3407/3408/3409、512-gene panel、训练轮数、SafeConf 公式、magnitude comparator 和全部 G1/G2/G3/G5 均与 E170 相同。最终 risk 仍由三 seed scGPT mean 与三 seed GEARS mean 的 disagreement 计算。

G4 改为三组 leave-one-seed-out estimators：每次在两个模型族中同时去掉同一个 seed，用保留的两个 seeds 分别求 family mean，再计算 disagreement 和 SafeConf risk。三组 risk 的 pairwise Spearman 中位数必须 ≥0.5，target-cluster bootstrap 2,000 次的 95% CI 下界必须 >0；all-200 与 seen-160、三个 states 都要通过。该修正只使稳定性检查对齐部署估计器，不改变最终分数。

## 主要终点

仅 E172 新 800 targets 进入推断。12 个 panel×state 的 tie-aware `AURC_magnitude − AURC_SafeConf` 等权平均；target-cluster stratified bootstrap 10,000 次、paired permutation 100,000 次。确认要求全部四面板 gate PASS、全 800 的 delta>0/CI 下界>0/p<0.05/至少 8/12 单元为正，并且 seen 640 的 CI 下界>0、p<0.05。E168/E170 不并入显著性计算，不允许 optional stopping。

## 边界

E172 仍是同一 test donor/study 内的 fresh-target replication，不是新 donor 或独立研究。若不通过，必须放弃“稳定优于 magnitude”的普遍主张；若通过，也只支持该 donor/study 和指定 scGPT–GEARS 上游模型。
