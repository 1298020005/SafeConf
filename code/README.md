# SafeConf 代码入口

当前唯一代码根：

```text
/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/
```

这个日期名保留是为了避免破坏实验证据、测试和运行产物中已冻结的路径。不再另建第二份代码目录。

| 目录 | 定位 |
|---|---|
| `safetrans_confidence/` | SafeConf 当前 Python 包、CLI 和测试 |
| `scripts/` | 当前复跑与审计入口 |
| `confidence_task/` | 被回归测试和早期协议复现依赖的兼容实现 |
| `configs/` | 数据集分组等冻结配置 |
| `03_code/` | SafeTrans-PT 方法演化基线；仅作兼容与追溯，不是新实验入口 |
| `00_meta/`、`01_asset_audit/`、`02_data/`、`04_logs/`、`docs/` | 历史证据与设计背景，不继续演化 |
| `outputs` | 指向 `/home/yyf/safeconf_runtime/outputs` 的软链接 |

新的 SafeConf 代码改动优先进 `safetrans_confidence/`；新运行产物只进 `outputs`指向的 runtime 目录。

## 当前外部探针复跑入口

`confidence_task/run_confidence_mvp_v2_1.py` 已保留原三数据集默认行为，并新增 `--datasets` 参数，可指定 `metadata/h5ad_scan.tsv` 中的 `study_family`。

`--datasets` 支持三种写法：

- `sciplex3`：按 `study_family` 选择，保持旧行为；
- `sciplex3_A549.h5ad` 或 `sciplex3_A549`：按具体文件名选择；
- `sciplex3@sciplex3_A549.h5ad`：同时限定家族和文件名。

E12 外部面板命令：

```bash
python3 code/20260426_154505_perturb_transport_final_push/confidence_task/run_confidence_mvp_v2_1.py \
  --atlas-root /home/yyf/data/singlecell_perturbation_atlas \
  --out-dir runtime/e12_external_panel_probe_20260707 \
  --datasets kangCrossCell,kangCrossPatient,KaggleCrossPatient,crossPatient,TCDD,sciplex3 \
  --n-genes 5000 \
  --min-cells 6 \
  --max-cells-per-group 1600 \
  --seed 5202
```

轻量汇报包生成：

```bash
python3 tools/scripts/package_e12_external_panel_probe.py
```

E13 官方 sciplex3 三细胞系 focused panel：

```bash
python3 tools/scripts/run_e13_sciplex3_official_3cell_panel.py \
  --atlas-root /home/yyf/data/singlecell_perturbation_atlas \
  --out-dir runtime/e13_sciplex3_official_3cell_panel_focused_20260707 \
  --n-genes 5000 \
  --min-cells 6 \
  --max-cells-per-group 800 \
  --max-perturbations 80 \
  --seed 5204

python3 tools/scripts/package_e13_sciplex3_official_3cell_panel.py
```

E14 官方 sciplex3 三细胞系 full-743 快速验证：

```bash
python3 tools/scripts/run_e13_sciplex3_official_3cell_panel.py \
  --atlas-root /home/yyf/data/singlecell_perturbation_atlas \
  --out-dir runtime/e14_sciplex3_official_3cell_panel_full743_gene1000_20260707 \
  --n-genes 1000 \
  --min-cells 6 \
  --max-cells-per-group 800 \
  --max-perturbations 743 \
  --seed 5206

python3 tools/scripts/package_e14_sciplex3_full743_gene1000.py
```

E14 是 full-743 drug-dose 的 1,000 基因快速版本，用来确认全量任务口径和风险信号；5,000 基因 full-743 仍需更长运行时间单独复跑。

E15 官方 sciplex3 三细胞系 full-743 2,000-gene sensitivity：

```bash
python3 tools/scripts/run_e13_sciplex3_official_3cell_panel.py \
  --atlas-root /home/yyf/data/singlecell_perturbation_atlas \
  --out-dir runtime/e15_sciplex3_full743_gene2000_20260707 \
  --n-genes 2000 \
  --min-cells 6 \
  --max-cells-per-group 800 \
  --max-perturbations 743 \
  --seed 5216

python3 tools/scripts/package_sciplex3_full743_sensitivity.py \
  --runtime-dir runtime/e15_sciplex3_full743_gene2000_20260707 \
  --out-dir docs/实验结果/E15_sciplex3_full743_gene2000_20260707 \
  --run-id E15 \
  --genes 2000 \
  --report-stem E15_SCIPLEX3_FULL743_GENE2000 \
  --previous 'E14 gene1000，learned aligned Spearman = 0.862，80% coverage 改善 = 13.63%' \
  --next-step '继续运行 3,000/5,000-gene sensitivity；同时开始 GEARS/CPA/scGPT 逐模型预测向量对齐审计。'
```

E16 官方 sciplex3 三细胞系 full-743 3,000-gene sensitivity：

```bash
python3 tools/scripts/run_e13_sciplex3_official_3cell_panel.py \
  --atlas-root /home/yyf/data/singlecell_perturbation_atlas \
  --out-dir runtime/e16_sciplex3_full743_gene3000_20260707 \
  --n-genes 3000 \
  --min-cells 6 \
  --max-cells-per-group 800 \
  --max-perturbations 743 \
  --seed 5217

python3 tools/scripts/package_sciplex3_full743_sensitivity.py \
  --runtime-dir runtime/e16_sciplex3_full743_gene3000_20260707 \
  --out-dir docs/实验结果/E16_sciplex3_full743_gene3000_20260707 \
  --run-id E16 \
  --genes 3000 \
  --report-stem E16_SCIPLEX3_FULL743_GENE3000 \
  --previous 'E15 gene2000，learned aligned Spearman = 0.899，80% coverage 改善 = 17.56%' \
  --next-step '若资源允许，继续运行 5,000-gene full-743 正式版；同时开始 GEARS/CPA/scGPT 逐模型预测向量对齐审计。'
```

E17 官方 sciplex3 三细胞系 full-743 5,000-gene formal：

```bash
python3 tools/scripts/run_e13_sciplex3_official_3cell_panel.py \
  --atlas-root /home/yyf/data/singlecell_perturbation_atlas \
  --out-dir runtime/e17_sciplex3_full743_gene5000_20260707 \
  --n-genes 5000 \
  --min-cells 6 \
  --max-cells-per-group 800 \
  --max-perturbations 743 \
  --seed 5218

python3 tools/scripts/package_sciplex3_full743_sensitivity.py \
  --runtime-dir runtime/e17_sciplex3_full743_gene5000_20260707 \
  --out-dir docs/实验结果/E17_sciplex3_full743_gene5000_20260707 \
  --run-id E17 \
  --genes 5000 \
  --report-stem E17_SCIPLEX3_FULL743_GENE5000 \
  --previous 'E16 gene3000，learned aligned Spearman = 0.903，80% coverage 改善 = 19.14%' \
  --next-step '将 E14–E17 写成 full-743 chemical 外部验证连续证据；下一项转向 GEARS/CPA/scGPT 逐模型预测向量对齐审计。'
```

E18 真实模型预测向量资产审计：

```bash
python3 tools/scripts/run_e18_model_vector_asset_audit.py
```

E18 只审计本地是否已有 GEARS、scGPT、CPA 的逐任务 predicted/true vector；它不训练新模型，也不把 GEARS-only 结果写成多模型统一验证。

E19 GEARS-only supplement：

```bash
python3 tools/scripts/run_e19_gears_only_supplement.py
```

E19 只整理已有 GEARS-only formal/probe 结果，用作补充证据；它仍不代表 CPA/scGPT 已完成统一验证。

E20 adapter contract validator：

```bash
python3 tools/scripts/run_e20_adapter_contract_validator.py
```

E20 检查已有预测输出是否满足 SafeConf 统一适配器合同。当前 10 个 bundle 均可 non-strict 审计，但 strict contract 尚未通过；GEARS 新导出脚本已补写 gene panel、gene order hash 和 normalization 字段。

E21 strict contract remediation smoke：

```bash
python3 tools/scripts/run_e21_strict_contract_remediation.py
```

E21 从 E17 抽样构造 task-scoped true effect 小包，并用 strict PredictionRecord 合同验证通过。`confidence_task/run_confidence_mvp_v2_1.py` 已同步改为 future outputs 使用 task-scoped `true_effect_key` / `target_control_key`。

E22 generator strict smoke：

```bash
python3 code/20260426_154505_perturb_transport_final_push/confidence_task/run_confidence_mvp_v2_1.py \
  --atlas-root /home/yyf/data/singlecell_perturbation_atlas \
  --out-dir runtime/e22_generator_strict_smoke_20260707 \
  --datasets Haber \
  --n-genes 200 \
  --min-cells 6 \
  --max-cells-per-group 300 \
  --seed 5221

python3 tools/scripts/package_e22_generator_strict_smoke.py
```

E22 验证修改后的生成器新产物可直接 strict pass。它不是性能实验，只是合同修复 smoke。

E23 shared benchmark adapter workbench：

```bash
python3 tools/scripts/run_e23_shared_benchmark_adapter_workbench.py
```

E23 从 E22 strict-pass 产物生成 `SHARED_BENCHMARK_TASK_MANIFEST.csv`，固定 task、true effect key、gene order、normalization 和输出 schema，避免下一轮 adapter 各写各的。注意：E23 是合同 smoke，不是 GEARS biological benchmark。

E24 model-family compatibility audit：

```bash
python3 tools/scripts/run_e24_model_family_compatibility_audit.py
```

E24 检查 E23 manifest 与模型家族是否匹配。结论是：E23 的 Haber stimulus/timecourse 不适合直接给 GEARS；`scgpt_env` 可以 import GEARS；后续已由 E25 转入真实 GEARS formal 输出的 strict remediation。

E25 GEARS strict PredictionRecord remediation：

```bash
python3 tools/scripts/run_e25_gears_strict_remediation.py
```

E25 读取 `/home/yyf/safeconf_runtime/outputs/gears_prediction_records_formal/` 中已有的真实 GEARS formal 输出，从 processed GEARS h5ad 恢复 gene order，补齐 `gene_panel_id`、`gene_order_hash`、`normalization_id` 等 strict contract 字段，并生成 `docs/实验结果/E25_gears_strict_prediction_records_20260708/`。合并包覆盖 Adamson、Dixit、Norman 的 9 个 formal runs、54 条 PredictionRecord，strict validator issue_count = 0。注意：E25 是 GEARS 真实输出的严格合同修复，不是 GEARS/scGPT/CPA 统一多模型验证。

E26 GEARS single-model risk audit：

```bash
python3 tools/scripts/run_e26_gears_single_model_risk_audit.py
```

E26 基于 E25 strict GEARS 包计算单模型风险分数：predicted-effect L2、predicted-effect abs mean、cell support、low support，以及 true-effect magnitude 诊断项。结果显示 predicted-effect abs mean / L2 与 GEARS 误差明显相关；GEARS native uncertainty 在 E25 formal records 中不可用。E26 只能作为 GEARS-only 风险补充分析，不能替代 scGPT/CPA adapter。

E27 scGPT forward PredictionRecord smoke：

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python tools/scripts/run_e27_scgpt_forward_prediction_record_smoke.py
```

E27 使用归档 scGPT 源码和 whole-human checkpoint，在 Replogle K562 essential 上选取 control、RPL3、NCBP2、KIF11 与 128 个 checkpoint-vocab 覆盖基因，执行 forward-only smoke，并导出 strict SafeConf PredictionRecord 与 predicted/true effect arrays。该步骤证明 scGPT adapter 合同链路可跑；它不是正式 scGPT 性能评估。

E28 GEARS–scGPT shared Adamson smoke：

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python tools/scripts/run_e28_gears_scgpt_shared_adamson_smoke.py
```

E28 从 E25 读取 Adamson seed-1 GEARS 预测，使用 E27 的 scGPT forward adapter，在 Adamson 上构造 CCND3、DAD1、DERL2 三个共同任务和 512-gene shared panel。输出 6 条 PredictionRecord，两个 predictor 共用 task-scoped true effect key，strict validator issue_count = 0。该步骤证明 GEARS–scGPT 双模型合同对齐可行，但仍是 smoke。

E29 GEARS–scGPT shared Adamson risk audit：

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python tools/scripts/run_e29_gears_scgpt_shared_adamson_risk_audit.py
```

E29 将 E25 中 Adamson fold-1 的全部 7 个可用单基因任务扩展成 GEARS/scGPT 双预测器 shared strict 合同。输出 14 条 PredictionRecord、512-gene shared panel、strict validator issue_count = 0，并生成 `E29_TASK_RISK_SCORES.csv` 与 `E29_RISK_AUDIT_SUMMARY.csv`。当前结果显示 GEARS–scGPT disagreement 对平均误差只有弱正相关（Spearman = 0.357）；true-effect magnitude 诊断很强但不可部署。该步骤用于风险审计流程推进，不作为正式 benchmark。

E30 GEARS seed-overlap feasibility audit：

```bash
python3 tools/scripts/run_e30_gears_seed_overlap_feasibility_audit.py
```

E30 审计 E25 的 GEARS formal records 是否能直接支持 seed/ensemble uncertainty。54 条 records 对应 47 个 unique task groups，其中 42 个 singleton，只有 5 个任务重复 ≥2 次、2 个任务重复 3 次；重复任务内 true effect 最大差异为 0。该步骤说明当前不能把 E25 写成正式 seed-ensemble uncertainty，只能作为固定任务三 seed 重跑的依据。

E31 GEARS fixed-test split smoke：

```bash
python3 tools/scripts/run_e31_gears_fixed_test_split_smoke.py
```

E31 给 `safetrans_confidence.cli.run_gears_prediction_records` 增加了两个可复用参数：`--test-perturbations-file` 用于固定 GEARS test perturbations，`--run-type smoke/formal` 用于区分工程 smoke 和正式运行。E31 在 Adamson 上用 E29 的 7 个任务跑 1 epoch smoke，输出 7 条 strict PredictionRecord，固定清单全部命中，strict issue_count = 0。该步骤不是性能 benchmark，而是后续固定任务三 seed 正式重跑的工程入口。

E32 GEARS fixed-test 3-seed smoke：

```bash
python3 tools/scripts/run_e32_gears_fixed_test_3seed_smoke.py
```

E32 使用 E31 的固定 test 清单，在 Adamson 7 个任务上跑 GEARS 3 seeds × 1 epoch，并启用 `--fixed-test-deterministic-val` 固定验证条件选择。输出 21 条 strict PredictionRecord，7/7 tasks 均有三 seed，strict issue_count = 0。seed disagreement 与平均误差在 n=7 上为正相关（Spearman = 0.679），但该步骤仍是 smoke，不是正式性能 benchmark。
