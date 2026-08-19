# 请复核 SafeConf supplement v0.2

角色：请你当客观审稿人，不要默认同意 Codex，也不要顺着用户想冲一区的愿望。

## 背景

当前主线是 SafeConf：给 single-cell perturbation prediction output（单细胞扰动预测结果）做 confidence scoring（可信度打分）。

已经完成 7 主表 formal audit。现在 Codex 又用服务器已有数据做了一轮 supplement v0.2，不改 frozen protocol，不下载新数据，不训练深度模型。

## 本轮 supplement 结果

请先看：

- `tables/SUPPLEMENT_RUN_STATUS.csv`
- `tables/SUPPLEMENT_MAIN_PER_DATASET_SUMMARY.csv`
- `README_先看这个.md`

核心结果：

| 数据集 | 状态 | simple_combined aligned rho | RC@80% |
| --- | --- | ---: | ---: |
| XieHon2017 | 跑通 | 0.424 | 16.85% |
| sciplex3_small | 跑通 | 0.515 | 12.48% |
| SrivatsanTrapnell2020_sciplex4 | 跑通 | 0.734 | 5.08% |
| ShifrutMarson2018 | 跑通 | 0.122 | 1.35% |
| LaraAstiasoHuntly2023_leukemia | 未跑通 | NA | NA |

Lara leukemia 未跑通原因：当前使用 `celltype`（细胞类型）作为 context（背景），`guide_id`（向导 RNA/基因扰动标识）作为 perturbation（扰动）时，构造出 0 个 task（任务）。

## 请你回答

1. 这 4 个跑通 supplement 数据集，哪些适合放论文 supplement，哪些只适合放审计记录？
2. `SrivatsanTrapnell2020_sciplex4` rho 很高但 test 只有 28 条，应该怎么写才不夸大？
3. `ShifrutMarson2018` rho 只有 0.122，是否应该保留为弱信号/失败边界，还是直接放弃？
4. Lara leukemia 是否值得换 context/perturbation 列再试，还是暂时排除？
5. 这些 supplement 结果是否改变你对 SafeConf 的判断？
6. 下一步更值得做 Tahoe pseudobulk adapter，还是继续扩 supplement 数据？

## 注意

- 不要把 supplement 和 7 主表 pooled 到一起当主结论。
- 不要为了好看删掉弱结果。
- 请明确区分：已证明、初步支持、还没证明。
