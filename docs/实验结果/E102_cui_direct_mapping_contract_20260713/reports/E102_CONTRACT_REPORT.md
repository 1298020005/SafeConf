# E102｜Cui 细胞因子直接映射子集合同

Cui 原矩形有 6 个免疫细胞背景、86 个刺激。E102 只保留经 `uppercase`、去连字符或去非字母数字字符后直接命中 scGPT 词表的 41 个标签；没有使用手工别名、表达效应、预测或误差。其余 45 个商品名、复合亚基或别名全部进入排除表，不猜测映射。

新合同按 41 个可执行刺激重新哈希：每折整行新背景 32 tasks、整列新刺激 45 tasks、双未见 9 tasks、随机缺失 16 tasks，另有 16 validation pairs 和 128 train pairs；训练集有 25%/50%/75%/100% 嵌套子矩阵。

- `tables/E102_CYTOKINE_MAPPING_AUDIT.csv`
- `tables/E102_EXCLUDED_UNMAPPED.csv`
- `manifests/E102_TASK_MANIFEST.csv`
