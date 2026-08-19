# 数据集审计

生成时间：2026-05-21 16:43:37

## 1. 搜索范围

- `/home/yyf/datasets`
- `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push`

本次没有下载数据，没有完整加载大矩阵。主要依据：

- `/home/yyf/datasets/singlecell_perturbation_atlas/metadata/h5ad_scan.tsv`
- 文件名、大小、修改时间
- 已有 selected dataset / result 文件列名

## 2. 总体文件数量

共检索到 552 个候选数据/结果文件。完整清单已写入：`docs/data_and_result_file_inventory.csv`。

## 3. h5ad 扫描表

扫描表路径：`/home/yyf/datasets/singlecell_perturbation_atlas/metadata/h5ad_scan.tsv`

扫描表维度：(83, 29)

可用于判断的列包括：`study_family`、`local_path`、`n_obs`、`n_vars`、`has_control_like`、`has_cell_type`、`has_donor`、`has_batch`、`has_condition`、`perturbation_type`、`suitable_perturbation_generalization`。

## 4. 重点数据集判断

完整数据集表见：`docs/dataset_inventory_for_confidence.csv`

| dataset_name | file_path | file_type | size | modified_time | possible_content | usable_for_confidence_scoring | has_context_field | has_perturbation_field | has_control_treated | can_compute_true_effect | has_multi_context_proxy | suitable_heldout_context_proxy | suitable_pair_split_proxy | has_existing_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Haber | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/Haber.h5ad | .h5ad | 565.5MB | 2026-04-17 15:43 | genetic_single / RNA / n_obs=9842 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| KaggleCrossCell | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/KaggleCrossCell.h5ad | .h5ad | 1.1GB | 2026-04-17 15:43 | genetic_single / RNA / n_obs=23653 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| KaggleCrossPatient | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/KaggleCrossPatient.h5ad | .h5ad | 1.2GB | 2026-04-17 15:43 | genetic_single / RNA / n_obs=25583 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| McFarland | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/McFarland.h5ad | .h5ad | 259.8MB | 2026-04-17 15:43 | genetic_single / RNA / n_obs=8464 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | False | False | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Parekh | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/Parekh.h5ad | .h5ad | 249.6MB | 2026-04-17 15:43 | genetic_single / RNA / n_obs=4346 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| TCDD | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/TCDD.h5ad | .h5ad | 2.4GB | 2026-04-17 15:43 | chemical_single / RNA / n_obs=103745 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| crossPatient | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/crossPatient.h5ad | .h5ad | 4.7GB | 2026-04-17 15:44 | genetic_single / RNA / n_obs=117363 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| kangCrossCell | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/kangCrossCell.h5ad | .h5ad | 779.1MB | 2026-04-17 15:44 | genetic_single / RNA / n_obs=13576 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| kangCrossPatient | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/kangCrossPatient.h5ad | .h5ad | 751.6MB | 2026-04-17 15:44 | genetic_single / RNA / n_obs=13093 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| sciplex3 | /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/sciplex3.h5ad | .h5ad | 561.8MB | 2026-04-17 15:44 | chemical_single / RNA / n_obs=26046 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Adamson | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Adamson.h5ad | .h5ad | 2.1GB | 2026-04-17 15:44 | genetic_single / RNA / n_obs=56998 / n_vars=5043 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | False | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Frangieh | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Frangieh.h5ad | .h5ad | 3.9GB | 2026-04-17 15:45 | genetic_single / multimodal / n_obs=110188 / n_vars=5124 | 是-可直接构造effect task | True | True | True | True | False | False | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Norman | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Norman.h5ad | .h5ad | 3.2GB | 2026-04-17 15:46 | genetic_combinatorial / RNA / n_obs=96994 / n_vars=5025 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | False | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Papalexi | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Papalexi.h5ad | .h5ad | 682.1MB | 2026-04-17 15:46 | genetic_single / multimodal / n_obs=20343 / n_vars=5013 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | False | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Replogle | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Replogle_K562essential.h5ad | .h5ad | 17.2GB | 2026-04-17 15:51 | genetic_combinatorial / RNA / n_obs=263245 / n_vars=5822 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | False | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Replogle | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Replogle_RPE1essential.h5ad | .h5ad | 11.1GB | 2026-04-17 15:55 | genetic_combinatorial / RNA / n_obs=173912 / n_vars=5687 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | False | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Replogle | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Replogle_exp6.h5ad | .h5ad | 856.7MB | 2026-04-17 15:55 | genetic_combinatorial / RNA / n_obs=27104 / n_vars=5019 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | False | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Replogle | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Replogle_exp7.h5ad | .h5ad | 2.5GB | 2026-04-17 15:56 | genetic_single / RNA / n_obs=102148 / n_vars=5087 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | False | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Replogle | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Replogle_exp8.h5ad | .h5ad | 1.5GB | 2026-04-17 15:56 | genetic_single / RNA / n_obs=54956 / n_vars=5031 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | False | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| TianActivation | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/TianActivation.h5ad | .h5ad | 583.7MB | 2026-04-17 15:57 | genetic_single / RNA / n_obs=20169 / n_vars=5045 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | True | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| TianInhibition | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/TianInhibition.h5ad | .h5ad | 992.3MB | 2026-04-17 15:57 | genetic_single / RNA / n_obs=31815 / n_vars=5152 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | True | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| Wessels | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Wessels.h5ad | .h5ad | 645.5MB | 2026-04-17 15:57 | genetic_combinatorial / RNA / n_obs=19775 / n_vars=5020 | 是-可直接构造effect task | True | True | True | True | False | False | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| sciplex3 | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/sciplex3_A549.h5ad | .h5ad | 394.5MB | 2026-04-17 15:58 | chemical_combinatorial / RNA / n_obs=82975 / n_vars=5839 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | True | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| sciplex3 | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/sciplex3_K562.h5ad | .h5ad | 787.6MB | 2026-04-17 15:58 | chemical_combinatorial / RNA / n_obs=150013 / n_vars=5839 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | True | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| sciplex3 | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/sciplex3_MCF7.h5ad | .h5ad | 1.1GB | 2026-04-17 15:58 | chemical_combinatorial / RNA / n_obs=223630 / n_vars=5839 | 可能不够-缺context/pert/control之一或scan异常 | True | True | False | False | True | False | False | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| sciplex3 | /home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/sciplex3_comb.h5ad | .h5ad | 460.5MB | 2026-04-17 15:58 | chemical_combinatorial / RNA / n_obs=63378 / n_vars=5000 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| SrivatsanTrapnell2020 | /home/yyf/datasets/singlecell_perturbation_atlas/official_scperturb/SrivatsanTrapnell2020_sciplex2.h5ad | .h5ad | 138.5MB | 2026-04-17 15:16 | chemical_single / RNA / n_obs=24262 / n_vars=58347 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| SrivatsanTrapnell2020 | /home/yyf/datasets/singlecell_perturbation_atlas/official_scperturb/SrivatsanTrapnell2020_sciplex3.h5ad | .h5ad | 2.4GB | 2026-04-17 15:30 | chemical_combinatorial / RNA / n_obs=799317 / n_vars=110983 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| SrivatsanTrapnell2020 | /home/yyf/datasets/singlecell_perturbation_atlas/official_scperturb/SrivatsanTrapnell2020_sciplex4.h5ad | .h5ad | 241.6MB | 2026-04-17 15:18 | chemical_single / RNA / n_obs=98437 / n_vars=58347 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| TianKampmann2019 | /home/yyf/datasets/singlecell_perturbation_atlas/official_scperturb/TianKampmann2019_day7neuron.h5ad | .h5ad | 256.5MB | 2026-04-17 15:22 | genetic_single / RNA / n_obs=182790 / n_vars=33752 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |
| TianKampmann2019 | /home/yyf/datasets/singlecell_perturbation_atlas/official_scperturb/TianKampmann2019_iPSC.h5ad | .h5ad | 334.6MB | 2026-04-17 15:24 | genetic_single / RNA / n_obs=275708 / n_vars=33752 | 是-可直接构造effect task | True | True | True | True | True | True | True | 代码可生成 leave_context / heldout_perturbation；未见持久 split registry 覆盖全部数据 |

## 5. 对 confidence scoring 的判断

能比较自然支持第一版 confidence scoring 的数据集：

- `KaggleCrossCell`：有 control、context 代理字段、perturbation 字段，规模中等，适合 MVP。
- `Haber`：有 control、cell_type/batch/condition，规模较小，适合快速复核。
- `Parekh`：有 control、cell_type/condition，规模小，适合 smoke test。
- `KaggleCrossPatient`：有 control、patient/donor/batch/cell_type，适合 external 或跨 patient/context 检查。
- `McFarland`：有 control 和 condition，但 `has_cell_type=False`，context 可能只能用 condition，需谨慎。
- `Frangieh` / `Wessels`：有潜力，但规模或组合扰动更复杂，适合第二阶段。

暂时不建议第一版就用的大数据：

- `Replogle`、`TianKampmann2019`、`TCDD` 等规模较大，适合后续扩大验证，不适合一开始排查 confidence task 逻辑。

## 6. 字段可用性结论

- 是否有 context 字段：不少数据有 `cell_type` / `donor` / `batch` / `condition` 之一，可以作为 context 代理；具体使用哪个字段由 `build_context_splits.py:13` 的候选顺序决定。
- 是否有 perturbation 字段：扫描表中 `has_perturbation_label=True` 或 `perturbation_label_columns` 非空的数据可用。
- 是否有 control / treated：扫描表 `has_control_like=True` 的数据才适合当前代码直接计算 effect。
- 是否可计算 `true_effect = perturbed_mean - control_mean`：需要同时有 context、perturbation、control。当前最稳的是 `KaggleCrossCell`、`Haber`、`Parekh`、`KaggleCrossPatient`。
- 是否有同一个 perturbation 覆盖多个 context：扫描表只能代理判断，需用 task 生成后统计；当前已有代码的 feasible split 会要求 shared perturbation/context。
- 是否适合 held-out context：有多 context 且同 perturbation 跨 context 出现时适合。
- 是否适合 held-out context-perturbation pair：从字段上可做，但当前代码没有找到正式 pair split。
- 是否已有 train/val/test split：没有找到统一全局 split registry；现有 runner 是运行时由 `feasible_splits()` 生成 split。
- 是否已有 prediction result：已有 safety task metrics，但不是完整 PredictionRecord。
