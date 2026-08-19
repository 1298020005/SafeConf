# Tahoe Eligibility Audit

## 结论

Tahoe 当前更适合作为 `external mega-scale validation candidate`。

它不是现在 7 主表的一部分；它的价值是后续做 external validation（外部验证）时证明 SafeConf 能不能扩展到超大药物扰动图谱。

## 已下载状态

- local root: `/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M`
- pseudobulk path: `/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M/metadata/pseudobulk_differential_expression`
- downloaded size: 84.9 GB
- obs metadata rows: 100,648,790
- obs row groups: 96
- pseudobulk shards downloaded: 1,026
- pseudobulk sample shards scanned: 25
- genes in metadata: 62,710
- drugs in drug metadata: 379
- cell lines in cell-line metadata: 1,000

## obs metadata 支持的任务结构

- unique drugs in obs: 380
- unique cell lines in obs: 50
- observed drug × cell_line pairs: 19,000
- pairs with at least 6 cells: 19,000
- pairs with at least 20 cells: 18,999
- control-like drug labels found in obs: DMSO_TF
- pass_filter counts: {"full": 95624334, "minimal": 5024456}

## pseudobulk 支持的 effect 信息

Sampled pseudobulk shards contain `log2FoldChange`, `n_cells_trt`, `n_cells_ctrl`, `drug`, `concentration`, and cell-line identifiers.
This means Tahoe may not need raw single-cell matrices for first-pass effect evaluation; a pseudobulk adapter could directly use gene-level log2FoldChange as the true effect vector.

## 下一步

1. Stop broad download at the current 85G scale unless a missing shard blocks the pseudobulk adapter.
2. Write a Tahoe pseudobulk adapter that treats `(Cell_Name_Vevo or Cell_ID_DepMap, drug, concentration)` as the task key.
3. Before putting Tahoe in the paper, run leakage checks and verify enough train support per drug and cell line.
