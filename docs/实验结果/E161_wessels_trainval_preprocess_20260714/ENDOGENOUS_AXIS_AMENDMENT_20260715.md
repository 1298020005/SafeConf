# E161 第二次勘误：内源基因轴与 `ncounts` 的实际含义

时间：2026-07-15。性质：test expression 解封前的方法与数据语义修订。

## 触发经过

E161 首次正式运行只物化 11,779 个训练细胞的前 20,639 列，在归一化的第一道门停止：逐细胞列和与 `obs[ncounts]` 最大相差 4。validation、test、excluded 表达当时均未读取。旧事务随后按 transaction sentinel 严格回滚，失败记录和旧 journal 已提交留存。

E161a 在修改主分析前冻结候选边界 20,631–20,639，并再次只读训练表达。结果显示没有候选前缀与 `obs[ncounts]` 完全一致：20,631 列有 10,215 个训练细胞不一致；加入 `Cas13d` 后的 20,636 列仍有 591 个细胞不一致。完整结果见相邻 E161a 目录。

## 上游定义核查

官方 scPerturb 仓库在 Wessels notebook 中执行了以下顺序：

1. 从 `*.GEXGDO*.matrix.mtx` 读取包含 gene expression 与 guide-derived oligo 的矩阵；
2. 从论文作者提供的 metadata 读入 `nCount_RNA`；
3. 将 `nCount_RNA` 改名为 `ncounts`；
4. 调用 `annotate_qc`，而该函数只在 `ncounts` 不存在时才用 `adata.X` 求和。

因此当前 H5AD 的 `obs[ncounts]` 是上游 Seurat QC 字段，不是最终合并后 `X` 的逐行和，不能用于推断最终表达轴边界，也不能作为归一化等式门。

锁定来源：

- scPerturb commit：`b69f72a070a92bcbaf41e7f9897b11598109ab48`
- `dataset_processing/notebooks/WesselsSatija2023.ipynb` SHA256：`b7ce4d66890831210d20b0bcc865b8eb27f84326a7176d5b179b19c00480e3d1`
- `utils.py` SHA256：`5647f6ddeaad80a8bd596928e767f60406bd7fb959a9966c192247ae19015975`

## 最终冻结轴

| block | count | first → last | SHA256 | policy |
|---|---:|---|---|---|
| endogenous genes | 20,631 | `OR4F5` → `AC213203.1` | `dbed3dad178ea500b01625abf5121c9ee17bdd501b87d2fcdede0b6bade654e7` | train/validation 可读 |
| engineered constructs | 8 | `eGFP` → `KRAB` | `103c2df8585646aa6dccde85866353889a699420b5536157b8babbd9b9aec554` | 不读取 |
| guide/barcode | 413 | `ATXN7L3_g1:NT_g2` → `CD71-Mpknot` | `9088328f4ac6b2a1b109c254f0068504d25618478383fcbb3f43be8e59dd06d2` | 不读取 |
| all excluded | 421 | `eGFP` → `CD71-Mpknot` | `e6e54ba5c0f63d62b599754ab3866da7cdf8194be4dfefd46dabc7d6a73e8116` | 不读取 |

正式归一化分母直接等于每个允许细胞在 20,631 个内源基因上的原始计数和。`obs[ncounts]` 仅保存差值统计，不再决定通过或失败。HVG、PCA、control prior、E-distance 和 graph 全部从该轴重新构建，旧的 20,639 运行不得复用。

## 对第一次勘误的关系

`PREFLIGHT_FAILURE_AND_AMENDMENT_20260715.md` 是真实历史，不修改。它正确识别并排除了后 413 个标签列，但“前 20,639 列均作为 RNA 轴且必须等于 ncounts”的判断已被 E161a 和上游源码核查推翻。本文件取代该部分规则；E160 的条件切分、测试封存和主评价终点不变。
