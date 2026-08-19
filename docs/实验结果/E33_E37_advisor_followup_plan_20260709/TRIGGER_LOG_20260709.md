# 2026-07-09 后续实验触发记录

这份记录说明：周老师要求的“其他 setting / 其他数据类型”已经开始执行，不停留在文字计划。

## 1. 数据状态

已运行：

```bash
bash tools/scripts/download_datasets.sh
```

结果：

- P1 gene 扩展数据已在本地：`kangCrossCell.h5ad`、`kangCrossPatient.h5ad`
- P2 chemical 数据已在本地：`TCDD.h5ad`、`sciplex3.h5ad`
- P3 可选数据已在本地：`ShifrutMarson2018.h5ad`、`AissaBenevolenskaya2021.h5ad`、`LaraAstiasoHuntly2023_exvivo.h5ad`、`LaraAstiasoHuntly2023_invivo.h5ad`

结论：本轮不需要重新下载，直接进入实验。

## 2. 已触发实验

### E33 输入来源与评价对象审计

命令：

```bash
python3 tools/scripts/run_e33_feature_provenance_error_source_audit.py
```

输出目录：

`docs/实验结果/E33_feature_provenance_error_source_audit_20260709/`

关键结果：

- 可前置使用的 score/feature：7
- 只能事后使用的项目：2
- leakage checklist：FAIL = 0，WARN = 1

给老师的口径：

> 后续所有 error 都绑定 predictor_name。true effect magnitude 只作事后控制；predicted magnitude 才能作为前置 baseline。disagreement 是 task-level difficulty 线索，不是 per-model reliability。

### E34/E35 split smoke

命令：

```bash
python3 tools/scripts/run_e34_e35_split_smoke.py \
  --n-genes 96 \
  --min-cells 10 \
  --max-cells-per-group 80 \
  --datasets Haber,Parekh,kangCrossCell,kangCrossPatient,TCDD,sciplex3,ShifrutMarson2018,AissaBenevolenskaya2021
```

输出目录：

`docs/实验结果/E34_E35_split_smoke_20260709/`

关键结果：

- 检查/构建数据集：8
- 成功构建任务矩阵：7
- `sciplex3` 已修正为 extra_official 多 context 文件：3 contexts × 36 perturbations × 105 tasks
- `AissaBenevolenskaya2021` 暂时未进入任务矩阵：自动识别不到 context/perturbation
- E34 submatrix summary：36 条
- E35 row/column holdout summary：85 条

### E34/E35 scoring smoke

命令：

```bash
python3 tools/scripts/run_e34_e35_scoring_smoke.py \
  --n-genes 96 \
  --min-cells 10 \
  --max-cells-per-group 80 \
  --datasets Haber,Parekh,kangCrossCell,kangCrossPatient,TCDD,sciplex3,ShifrutMarson2018,AissaBenevolenskaya2021
```

输出目录：

`docs/实验结果/E34_E35_scoring_smoke_20260709/`

关键结果：

- 数据任务构建成功：7/8
- split scoring 成功：98/121
- score table：978 行
- 已在小矩阵、整行、整列 setting 上计算：
  - `risk_safeconf_smoke`
  - `risk_disagreement`
  - `risk_predicted_magnitude`
  - `risk_inverse_support`
  - `risk_inverse_context_similarity`
  - `error_v0_rmse`
  - `error_contextsim_rmse`
  - `error_mean_rmse`

初步强信号示例：

- `Parekh` row holdout：`risk_safeconf_smoke` vs `error_mean_rmse` Spearman ≈ 0.787
- `sciplex3` row holdout：`risk_safeconf_smoke` vs `error_mean_rmse` Spearman ≈ 0.761
- `sciplex3` submatrix 0.75：`risk_safeconf_smoke` vs `error_mean_rmse` Spearman ≈ 0.718

注意：这是 smoke，不是 formal。它证明实验链已经跑通，不能直接写成最终论文结论。

## 3. 明天给老师怎么说

可以直接说：

> 我按您上次说的几个 setting 已经开始触发了。数据不用重新下，P1/P2/P3 都在本地。  
>   
> 我先做了输入来源审计，确认 true effect magnitude 不进入前置打分，error 后面都会绑定 predictor_name。  
>   
> 然后我把小矩阵、整行、整列 holdout 的 split manifest 生成出来，并跑了一版轻量 score-vs-error smoke。现在 7 个数据集成功构建任务矩阵，98 个 split 跑通了初步评分。  
>   
> 下一步如果您认可，我会把 smoke 升级成 formal：固定数据集、扩大 gene panel、补更强 predictor，再把 gene 和 chemical 分开汇总。

## 4. 下一步

1. 把 E34/E35 从 96-gene smoke 升级到 formal gene panel；
2. 固定可用数据集：Haber、Parekh、kangCrossCell、kangCrossPatient、TCDD、sciplex3、Shifrut；
3. 对 `AissaBenevolenskaya2021` 单独做 obs column 适配，不让它卡主线；
4. 引入更强 predictor 或复用现有 formal PredictionRecord；
5. 形成 E37 setting × domain 总表；
6. 再启动 E36 gene→gene / chemical→chemical transfer。
