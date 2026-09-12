# E207 图件说明与复算入口

这些图只读取本目录已经提交的 CSV，不删除数据集、不重选权重，也不生成新统计结果。图内采用英文，便于后续直接用于英文稿；中文讲解见 `../../组会汇报_20260913.md`。

## E207_ci_summary

- 圆点：`E207_POINT_DELTAS.csv` 中的点估计；
- 横线：`E207_BOOTSTRAP_SUMMARY.csv` 中的 95% 簇自举区间；
- 横轴：SafeConf-M 指标减去只看预测幅度的指标；
- 虚线 0：两种方法没有差异；
- 横线跨 0：按冻结判据记为未确认。

## E207_eight_study_deltas

- 数据：`E207_E153_BATCH_METRICS.csv`；
- 每个研究先对内部留出批次取平均；
- 左图：SafeConf-M 与误差的 Spearman 减去幅度与误差的 Spearman；
- 右图：在固定复核 20% 任务时，两种分数的标准化效用差；
- 绿色表示差值大于等于 0，红色表示差值小于 0。

## E207_scenario_heatmap

- 数据：`E207_E187_SCENARIO_SUMMARY.csv`；
- 每个格子对 25%、50%、75%、100% 四种来源数据量取平均；
- 行为数据集，列为随机缺 pair、未见扰动、未见背景和两者都未见；
- 数字为 SafeConf-M 相对幅度的 Spearman 差值；
- Cui 是细胞因子扰动，与上方三套遗传扰动用白线分开。

## E207_modality_boundary

该图只显示冻结统计判定：95% 区间下限大于 0 才写 `confirmed`。药物实验没有与 E206 同定义的完整 SafeConf-M 输入，图中明确写 `do not merge`，没有把缺失输入补造成结果。

## 重画

```bash
python tools/scripts/plot_e207_formula_stress.py
```

脚本：`tools/scripts/plot_e207_formula_stress.py`。每张图同时输出 320 dpi PNG 和保留可编辑文字的 PDF。
