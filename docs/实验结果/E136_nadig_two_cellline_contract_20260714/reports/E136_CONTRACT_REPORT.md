# E136｜Nadig 双细胞系第七数据确认合同

HepG2 与 Jurkat 来自同一研究、不同生物细胞系。合同只读取 obs 标签、每个 cell line × perturbation 的细胞数、表达基因身份轴和 scGPT 词表；未读取表达矩阵数值。

- 候选共同扰动：785；哈希固定抽取：96。
- 每个 pair 至少 50 个细胞。
- 两个外层 cell-line holdout folds；测试任务共 256。
- 每折包含 source 内随机 seen pair、整 cell-line 未见、整 perturbation 未见和二者同时未见。
- E135 方向风险模型的文件哈希已经写入状态文件；后续不得根据 Nadig 结果改系数后仍称确认。
