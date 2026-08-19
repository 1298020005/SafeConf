# E86｜sciPlex3 → OpenProblems 跨数据集合同

来源域只使用 sciPlex3 的 perturbed cells；目标域 OpenProblems 的 perturbed cells 全部封存到评价阶段。目标域 vehicle control 可以用于描述新 context。1000 基因面板只按两域 control variance 的平均秩选择，任务只按标签和细胞数筛选。

| item | value |
|---|---|
| common_genes | 2421 |
| frozen_gene_panel | 1000 |
| source_tasks | 108 |
| target_tasks | 553 |
| source_contexts | 3 |
| target_contexts | 4 |
| source_drugs | 9 |
| target_drugs | 141 |
| shared_drugs_by_exact_name | 0 |

这是 chemical→chemical 的强迁移：来源是 3 个癌细胞系与 9 种药，目标是 4 类 PBMC 与 141 种药，精确同名药物重叠为 0。模型需要同时处理新实验、新 context 和几乎全部新药；负结果也具有解释价值。

- `tables/E86_GENE_PANEL.csv`
- `tables/E86_CROSS_DATASET_MANIFEST.csv`
