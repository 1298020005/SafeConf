# E201 四目标盲训练视图审计

审计日期：2026-08-02

原始跨细胞系 H5AD 位于数据盘，共 632,488 个细胞、3,352 个基因，
SHA-256 为
`1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8`。
每个目标背景都建立了单独的物理 H5AD，而不是运行时才用布尔 mask 遮盖真值。

| 目标 | 总行数 | 目标对照 | source 扰动 | 目标扰动 | H5AD SHA-256 |
|---|---:|---:|---:|---:|---|
| K562 | 375,291 | 10,691 | 336,126 | 0 | `8a19d3d4048800c06827e2f28e983bfa6f67b1945d2081692ce3d69a45d471db` |
| RPE1 | 349,953 | 11,485 | 310,788 | 0 | `9aeec2bdc56461713e9039348428d63aa1b98f6000835ed3253c35d6d85a387d` |
| HepG2 | 403,508 | 4,976 | 364,343 | 0 | `1f0dc20806bd40cd151ebfebb59a9fdac5ad14c4e223a655ae0ed6de890ed891` |
| Jurkat | 360,814 | 12,013 | 321,649 | 0 | `5a944ec0f114e2398f2058072121d130deee5af1f97def926619f5ee30c231fb` |

四份视图均保留 39,165 个公开细胞系对照，只保留另外三个细胞系中
属于官方 train/validation condition 的扰动细胞。`K562_adamson` 为 0 行，
H5AD `uns` 为空，没有目标差异表达排名等结果元数据。

## 独立复核

复核程序没有调用构建函数，而是使用 backed AnnData 重新读取每个物理文件。
实际行数、基因数、对照数、目标扰动数、允许 condition、`uns`、
`K562_adamson` 和清单中 20 个文件的字节数/SHA-256 全部通过。

源 H5AD 的 observation name 在跨实验拼接后不唯一，四份视图中被标记为重复的
行数分别为 269,684、219,457、283,955 和 233,510。训练 dataloader 通过位置
取数，预测对齐使用 `condition_name × cell_line × experimental_batch`，
不使用 observation name 作生物任务主键。本实验不调用 `obs_names_make_unique()`，
以免在盲视图中人为改写官方索引。

数据盘 manifest 的 SHA-256：

- K562：`8114be7febf1f166da729cf23cb9dc2356d7e54b6f255581b665997d7651265d`；
- RPE1：`e2f1e6aaa91a5e4143bbacf578453eaeaa135dc6e0340921d048de5a20ce9202`；
- HepG2：`cf3b0ce0a93d669c7963b2198ce27dcf053249f1af0e0928777a4de4d84e2873`；
- Jurkat：`fa9bc3cf9d22f8df2635d3932402db9cc455826f5226aceb366b64a19a4ddb21`。
