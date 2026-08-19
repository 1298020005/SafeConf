# E172 pretruth 执行冻结

## 固定对象

- Q01–Q04 共 800 个新目标；E168 的 200 个和 E170 的 800 个全部排除。
- 每个面板使用相同的 scGPT、GEARS、seeds 3407/3408/3409、512 基因轴和训练设置。
- SafeConf 最终分数仍由三 seed 的 scGPT family mean 与三 seed 的 GEARS family mean 计算，没有改公式或权重。
- G4 使用三组 leave-one-seed-out two-seed family means；每组同时从两个模型族去掉同一个 seed。
- G4 阈值保持 E170 的中位 pairwise Spearman ≥0.5 且 target-cluster bootstrap 95% CI 下界 >0。

## 执行顺序

1. 本文件、asset builder、pretruth runner、joint evaluator 和全部依赖先提交并推送 GitHub/Gitee。
2. 每个面板只建立 F2：允许 control、train-seen、validation-seen X；test targeting X 与 column-unseen targeting X 的读取数必须为 0。
3. 每个面板训练六个正式模型，生成 test query prediction；query graph 不得含 `y`。
4. 检查 G1/G2/G3/G4、synthetic tests。四个面板的 PASS/FAIL 产物全部保留，不删除失败面板。
5. 将四个 pretruth release 原字节提交并推送两个远程。
6. 只有四个面板全部 PASS，才允许用该 gate commit 分别建立四个 F3；否则 E172 正式终止且不读任何 test targeting X。
7. F3 完成后只运行一次 joint evaluator。主要推断仅包含 E172 800 个目标，不并入 E168/E170，也不丢弃任何面板或 state。

## 主要判定

全 800 目标需要 Δ(AURC_magnitude−AURC_SafeConf)>0、bootstrap 95% CI 下界>0、单侧 paired permutation p<0.05，并且 12 个 panel×state 至少 8 个为正；seen 640 还需 CI 下界>0 且 p<0.05。全部结果均保留，失败不能换盐、换目标或追加面板来“补显著”。

## 解释边界

E172 仍是同一研究、同一 test donor 的新目标复现。即使通过，也不能写成独立 donor/study replication；若失败，则放弃“SafeConf 稳定优于 magnitude”的普遍主张。
