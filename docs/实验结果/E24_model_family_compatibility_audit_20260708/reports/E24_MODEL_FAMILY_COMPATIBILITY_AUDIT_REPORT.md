# E24 model-family compatibility audit

生成时间：2026-07-08 02:35

## 1. 目的

E23 已经建立 strict adapter contract smoke，但它来自 Haber stimulus/timecourse 数据。E24 检查这个 manifest 是否适合直接给 GEARS、scGPT、CPA/chemCPA 使用。

## 2. E23 compatibility summary

| manifest | n_task_groups | dataset_names | perturbation_examples | gene_order_hashes | effect_definitions | gears_compatible | gears_reason | scgpt_compatible | scgpt_reason | cpa_compatible | cpa_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E23_HABER_200GENE_STRICT_SMOKE | 120 | Haber | Hpoly_Day10,Hpoly_Day3,Salmonella | 1 | mean_diff | False | Perturbations are Hpoly/Salmonella stimuli, not gene knockout/overexpression conditions. | adapter_possible_but_not_ready | Generic single-cell model source exists, but perturbation prediction adapter/checkpoint is absent. | conceptually_possible_for_stimulus_or_drug_but_not_ready | CPA/chemCPA assets are not locally executable; E23 stimuli are not a prepared CPA dataset. |

## 3. E23 perturbation class

| perturbation_class | n_task_groups |
| --- | --- |
| stimulus_or_timecourse | 120 |

## 4. GEARS candidate legacy records

| dataset_name | n_records | n_task_like_rows | n_seeds | n_perturbations | perturbation_class | mean_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| adamson | 21 | 18 | 3 | 18 | gene_perturbation_like | 0.0420858 |
| dixit | 3 | 2 | 3 | 2 | gene_perturbation_like | 0.424841 |
| norman | 30 | 27 | 3 | 27 | gene_perturbation_like | 0.0558785 |

## 5. GEARS processed assets

| dataset | processed_h5ad | exists | size_mb | has_split_files | n_split_files |
| --- | --- | --- | --- | --- | --- |
| adamson | $YYF_DATA/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad | True | 1629.4 | True | 3 |
| dixit | $YYF_DATA/gears_formal_baselines_v2/dixit_local_atlas/perturb_processed.h5ad | True | 1008.95 | True | 3 |
| frangieh | $YYF_DATA/gears_formal_baselines_v2/frangieh_local_atlas/perturb_processed.h5ad | True | 1786.68 | True | 3 |
| norman | $YYF_DATA/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad | True | 2158.86 | True | 3 |

## 6. Runtime environments

| environment | path | gears_importable | scgpt_importable |
| --- | --- | --- | --- |
| default_python | /home/miniconda/bin/python3 | False | False |
| scgpt_env | $YYF_CONDA/envs/scgpt_env/bin/python | True | False |

## 7. Decisions

| decision | reason | action | priority |
| --- | --- | --- | --- |
| Do not run GEARS on E23 Haber manifest | E23 perturbations are stimulus/timecourse labels, not gene perturbations. | Keep E23 as adapter contract smoke only. | 1 |
| Use GEARS-compatible gene perturbation data for the first true model adapter | Norman/Adamson/Dixit/Frangieh local processed GEARS assets exist and scgpt_env imports gears. | Run a small GEARS strict smoke in scgpt_env, then package as E25 if runtime succeeds. | 2 |
| Do not claim scGPT validation yet | scGPT source zip exists but import is false and no perturbation checkpoint/output was found. | Unpack/install source or point PYTHONPATH; then locate checkpoint/tutorial assets. | 3 |
| Do not claim CPA/chemCPA validation yet | Only a CPA PDF trace was found locally. | Acquire executable assets or choose an accessible perturbation model. | 4 |

## 8. 结论

- 不应把 E23 Haber stimulus manifest 直接作为 GEARS biological benchmark。
- GEARS 的可执行路径存在：`scgpt_env` 可以 import GEARS，且本地有 Norman/Adamson/Dixit/Frangieh processed assets。
- 下一步应做 GEARS gene-perturbation strict smoke，而不是 GEARS-on-E23。
- scGPT/CPA 仍需先解决安装、checkpoint 或可执行输出资产。
