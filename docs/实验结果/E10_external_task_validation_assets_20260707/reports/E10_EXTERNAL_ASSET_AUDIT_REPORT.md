# E10 外部任务级验证资产审计

生成时间：2026-07-07 06:27

## 1. 当前数据现实

- 实际数据根目录：`/home/yyf/data/singlecell_perturbation_atlas`
- 服务器已有 h5ad：83 个
- 已有 h5ad 总大小：107.3 GiB
- 官方 Zenodo 文件覆盖：66/66
- scPerturBench 官方文件为 `.h5ad.gz`：12 个；服务器保存为解压后 `.h5ad`，大小不能直接与 gzip 包比较。
- scPerturb 官方非压缩 h5ad 大小匹配：54 个。

结论：E10 不需要盲目重新下载全量数据。当前服务器已经具备外部任务级验证的数据基础。下一步应从候选数据中选择 1–3 个冻结任务级外部验证，而不是重复下载 TB 级数据。

路径注意：当前真实数据根目录是 `/home/yyf/data/singlecell_perturbation_atlas`。为了兼容历史脚本，服务器上已建立本地软链接 `/home/yyf/datasets -> /home/yyf/data`；该软链接不属于 Git 仓库，新机器需要重新建立或直接传入真实数据根目录。

## 2. 推荐 E10 第一批候选



| 文件 | 分组 | study | perturbation | modality | cells | genes | score | reason |
|---|---|---|---|---|---:|---:|---:|---|
| `sciplex3_MCF7.h5ad` | official_generalization | sciplex3 | chemical_combinatorial | RNA | 223630 | 5839 | 12 | scPerturBench generalization asset; perturbation-generalization suitable; combination task possible; contains context/donor/cell-type metadata; chemical external stress-test; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `sciplex3_K562.h5ad` | official_generalization | sciplex3 | chemical_combinatorial | RNA | 150013 | 5839 | 12 | scPerturBench generalization asset; perturbation-generalization suitable; combination task possible; contains context/donor/cell-type metadata; chemical external stress-test; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `sciplex3_comb.h5ad` | official_generalization | sciplex3 | chemical_combinatorial | RNA | 63378 | 5000 | 12 | scPerturBench generalization asset; perturbation-generalization suitable; combination task possible; contains context/donor/cell-type metadata; chemical external stress-test; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `sciplex3_A549.h5ad` | official_generalization | sciplex3 | chemical_combinatorial | RNA | 82975 | 5839 | 12 | scPerturBench generalization asset; perturbation-generalization suitable; combination task possible; contains context/donor/cell-type metadata; chemical external stress-test; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `TCDD.h5ad` | extra_official | TCDD | chemical_single | RNA | 103745 | 5000 | 11 | scPerturBench generalization asset; perturbation-generalization suitable; contains context/donor/cell-type metadata; chemical external stress-test; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `sciplex3.h5ad` | extra_official | sciplex3 | chemical_single | RNA | 26046 | 5000 | 11 | scPerturBench generalization asset; perturbation-generalization suitable; contains context/donor/cell-type metadata; chemical external stress-test; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `crossPatient.h5ad` | extra_official | crossPatient | genetic_single | RNA | 117363 | 5000 | 10 | scPerturBench generalization asset; perturbation-generalization suitable; contains context/donor/cell-type metadata; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `KaggleCrossPatient.h5ad` | extra_official | KaggleCrossPatient | genetic_single | RNA | 25583 | 5000 | 10 | scPerturBench generalization asset; perturbation-generalization suitable; contains context/donor/cell-type metadata; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `KaggleCrossCell.h5ad` | extra_official | KaggleCrossCell | genetic_single | RNA | 23653 | 5000 | 10 | scPerturBench generalization asset; perturbation-generalization suitable; contains context/donor/cell-type metadata; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `kangCrossCell.h5ad` | extra_official | kangCrossCell | genetic_single | RNA | 13576 | 5000 | 10 | scPerturBench generalization asset; perturbation-generalization suitable; contains context/donor/cell-type metadata; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `kangCrossPatient.h5ad` | extra_official | kangCrossPatient | genetic_single | RNA | 13093 | 5000 | 10 | scPerturBench generalization asset; perturbation-generalization suitable; contains context/donor/cell-type metadata; RNA expression available; enough cells for task-level aggregation; high-value external validation target |
| `crossSpecies.h5ad` | extra_official | crossSpecies | genetic_single | RNA | 112903 | 5000 | 9 | scPerturBench generalization asset; perturbation-generalization suitable; RNA expression available; enough cells for task-level aggregation; high-value external validation target |


## 3. 当前缺失文件

官方 metadata 中未在 `/home/yyf/data/singlecell_perturbation_atlas` 找到的文件数：0。

如果后续确实需要补齐，使用：

```bash
bash docs/实验结果/E10_external_task_validation_assets_20260707/download_manifests/download_missing_official_files.sh
```

## 4. 下一步实验建议

1. 首选 `kangCrossCell` / `kangCrossPatient`：数据量适中、外部泛化场景清楚，适合做 E10 task-level validation。
2. 次选 `TCDD` / `sciplex3`：chemical stress-test，可和 Tahoe chemical 边界互相印证。
3. 保留 `Frangieh`：已有 E8b 聚合证据，可用于连接外部 benchmark method-error association 与任务级审计。
4. 不建议先跑 Replogle 大文件：资源消耗高，且不一定立刻提升论文主线。

## 5. 输出文件

- `tables/E10_ACTUAL_H5AD_FILES.csv`
- `tables/E10_OFFICIAL_FILE_COVERAGE.csv`
- `tables/E10_CANDIDATE_RANKING.csv`
- `tables/E10_MISSING_OFFICIAL_FILES.csv`
- `download_manifests/download_missing_official_files.sh`
