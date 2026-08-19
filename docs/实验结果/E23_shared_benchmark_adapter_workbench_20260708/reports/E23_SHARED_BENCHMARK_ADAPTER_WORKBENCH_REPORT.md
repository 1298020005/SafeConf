# E23 shared benchmark adapter workbench

生成时间：2026-07-08 02:28

## 1. 目的

E23 固化一个小型 strict-pass shared benchmark。以后 GEARS、scGPT、CPA/chemCPA 不再各跑各的，而是必须对 `SHARED_BENCHMARK_TASK_MANIFEST.csv` 逐任务输出预测向量。

## 2. Manifest checks

| check | status | value | why_it_matters |
| --- | --- | --- | --- |
| one_true_key_per_task_group | pass | 120/120 | 不同模型必须共享同一个 true effect，才能比较模型特异错误。 |
| records_are_two_predictor_reference | pass | ContextSimBaseline,V0StrongBaseline | adapter 开发时保留参考预测器，便于 sanity check。 |
| single_gene_space | pass | 1 | 跨模型向量必须处在同一 gene order 上。 |
| single_effect_definition | pass | mean_diff | mean_diff、logFC 等 effect 定义不能混用。 |
| contains_test_split | pass | test,train,val | adapter smoke 至少要覆盖 test 行。 |

## 3. Local asset inventory

| asset | kind | path | exists | usable_now | limitation |
| --- | --- | --- | --- | --- | --- |
| GEARS legacy formal vectors | prediction_records_and_npz | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal | True | True | 旧 GEARS 输出是 legacy/non-strict；需要按 shared manifest 重跑或转换。 |
| Patched GEARS exporter | code | code/20260426_154505_perturb_transport_final_push/safetrans_confidence/cli/run_gears_prediction_records.py | True | True | 能写 strict provenance；尚未对 E23 shared manifest 运行。 |
| scGPT conda env | environment | $YYF_CONDA/envs/scgpt_env | True | False | scgpt import result: False; env 有 torch/scanpy 但未安装 scgpt 包。 |
| scGPT source archive | source_zip | $YYF_ARCHIVE/code/20260519_0958_home_cleanup/moved_top_level/scGPT-main.zip | True | True | zip files=109; has perturbation tutorial=True; no local checkpoint found. |
| CPA / chemCPA local executable output | missing_adapter_asset | $SAFECONF_RUNTIME/outputs/current_project_explanation_20260530/合并版_给我看的/papers/CPA_2023_MSB.pdf | True | False | 本地只有论文 PDF 痕迹，没有可运行代码、checkpoint 或 SafeConf PredictionRecord。 |

## 4. Adapter backlog

| model | current_state | next_adapter_step | claim_allowed_now | claim_not_allowed | priority |
| --- | --- | --- | --- | --- | --- |
| SafeConf reference baselines | E22 strict-pass generator output available | Use E23 manifest as the shared reference input. | Can be used as adapter smoke and sanity baseline. | Not a deep model comparison. | 1 |
| GEARS | Legacy vectors exist; exporter patched for provenance | Add a GEARS runner that reads SHARED_BENCHMARK_TASK_MANIFEST.csv and writes strict PredictionRecord on the same gene order. | GEARS-only supplementary legacy evidence exists. | Unified GEARS/scGPT/CPA validation. | 2 |
| scGPT | Source zip exists; env exists; package import currently false | Unpack/install source or point PYTHONPATH to source; locate perturbation checkpoint/tutorial assets; export predicted_effects on E23 manifest. | Local source availability only. | scGPT prediction-vector validation. | 3 |
| CPA / chemCPA | Only literature PDF trace found locally | Acquire executable implementation/checkpoint or replace with another accessible perturbation model for shared benchmark. | Future adapter target. | CPA validation or comparison. | 4 |

## 5. Required output schema

| field | type | requirement |
| --- | --- | --- |
| record_id | string | adapter-specific unique prediction row id |
| task_id | int/string | reuse task id from manifest when possible |
| task_key | string | must match manifest |
| dataset_name | string | must match manifest |
| fold_id | int | must match manifest |
| split | train/val/test | must match manifest |
| context | string | must match manifest |
| perturbation | string | must match manifest |
| predictor_name | string | GEARS/scGPT/CPA/etc. |
| gene_panel_id | string | must match manifest |
| gene_order_hash | sha256 | must match manifest exactly |
| effect_definition | string | must match manifest |
| normalization_id | string | must match manifest |
| predicted_effect_key | string | key in predicted_effects.npz |
| true_effect_key | string | reuse task-scoped key from manifest |
| true_error_rmse | float | computed after predicted vector is exported |

## 6. 结论

- E23 已给出 120 个 task groups 的 shared benchmark manifest。
- GEARS 有 legacy 结果和 patched exporter，是最先值得接到 manifest 的模型。
- scGPT 有源码压缩包和 conda 环境，但当前 `import scgpt` 为 false，且没有本地 checkpoint / PredictionRecord。
- CPA/chemCPA 当前只有论文 PDF 痕迹，没有本地可执行向量输出。
- 因此下一项实质工作应是 GEARS-on-E23 或 scGPT source install + adapter smoke，不能写成三模型统一验证已经完成。
