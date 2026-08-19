# SafeConf 当前证据与投稿关卡（2026-07-13）

这份文件取代 `GATE_STATUS_20260712.md` 作为当前入口。旧文件保留实验演进记录。

需要把周老师的逐项要求、当前证据、审稿风险和“为什么不能保证录用”放在一起阅读时，打开 [SafeConf 录用判断与项目总账](../投稿准备/录用判断与项目总账_20260713/index.html)。

## 当前结论

SafeConf 已从参考预测器实验推进到正式的 context-aware scGPT–GEARS 双模型验证。Frangieh、Lara ex vivo、Santinha 共 3 套独立遗传扰动数据、13 个外层 context-holdout fold、1,437 个测试任务。每个测试任务只使用训练子矩阵、目标背景未扰动 control、扰动标记和模型预测；受扰动后的真实表达在风险冻结后才用于评价。

三数据集等权宏平均结果：

| 数据集 | SafeConf calibrated ρ | frozen ρ | disagreement ρ | magnitude ρ |
|---|---:|---:|---:|---:|
| Frangieh | 0.253 | 0.242 | 0.137 | 0.148 |
| Lara ex vivo | 0.387 | 0.355 | 0.176 | 0.148 |
| Santinha | 0.065 | -0.095 | -0.127 | -0.089 |

正式三数据集元分析中，SafeConf 相对 magnitude 的 Δρ=0.166：固定三数据集聚类 bootstrap 95% CI `[0.030, 0.286]`，dataset-population CI `[0.022, 0.298]`。相对 disagreement 的 Δρ=0.173，dataset-population CI `[0.051, 0.305]`。Santinha 的弱结果保留在主表，不作删除或回调权重。

## 周老师的问题是否逐项回答

| 问题 | 当前实现 | 证据 |
|---|---|---|
| 未见组合输入什么 | 同背景 control expression + perturbation flag；test perturbed expression 不进入模型或风险 | E105–E108、E112 strict records |
| 随机缺失 pair | 每个外层 fold 独立冻结 | E97–E100、E108、E112 |
| 训练矩阵很小 | 25/50/75/100% 参考预测器线已完成 | E97–E103 |
| 整行新背景 | Frangieh 3 folds、Lara 5 folds、Santinha 5 folds | E108、E112、E113 |
| 整列新扰动 | 每折约 20% perturbations 整列留出 | E97、E99、E108、E112 |
| 背景与扰动双未见 | row × column 交叉任务独立报告 | E108、E112 |
| 更多数据集 | 三套 formal gene 数据；另有 cytokine 与 chemical 独立线 | E103、E108、E112 |
| 不同扰动类型 | gene、cytokine、chemical 均有结果 | E84/E87/E89、E103、E113 |
| 模型真的正式训练了吗 | scGPT 加载 whole-human 预训练参数并按 fold 微调；GEARS 共表达图只读训练背景 control | E106、E107、E112 |

## 新完成的方法闭环

1. `E105` 修复 GEARS 全局 control 池混背景问题，567 个任务通过同背景对照与拆分互斥检查。
2. `E106/E107` 在 Frangieh 三折正式微调 scGPT 和 GEARS，得到 837 个测试任务、1,674 条 strict PredictionRecord，issue count 为 0。
3. `E108` 完成正式双模型风险审计；pooled ρ=0.253，但单数据集相对强基线的聚类区间仍跨 0。
4. `E109/E110` 用内层 row/column/double 重训做 setting-matched calibration。该方案 pooled ρ=0.176，低于 E108 的 0.253，作为负结果封存，不替换主方法。
5. `E111` 证明信号具有预测器依赖：对 GEARS 误差，SafeConf ρ=0.397；相对 magnitude 的 Δρ 95% CI `[0.007, 0.275]`。对 scGPT 误差没有同等证据，因此主张应写成“预测器/任务级风险路由”。
6. `E112/E113` 完成 Lara、Santinha 正式复制与三数据集元分析，形成当前最强主证据。
7. `E114` 增加 90% split-conformal 误差上界。经验覆盖率 0.980，但平均上界约为真实平均误差的 1.86 倍；可作保守兜底，尚不紧致。

## 当前投稿判断

| 项目 | 状态 | 判断 |
|---|---|---|
| 老师要求的矩阵 setting | 完成 | 随机 pair、row、column、double、训练量均有冻结合同 |
| 正式模型与输入一致性 | 完成 | 三数据集 formal scGPT–GEARS；同背景 control；训练专属共表达图 |
| 严格可追溯性 | 完成 | E108/E112 strict issue count=0；失败运行与负结果保留 |
| 超过 disagreement | 完成 | 三数据集 dataset-population CI 不跨 0 |
| 超过 magnitude | 完成于三 formal gene 数据主目标 | dataset-population CI `[0.022, 0.298]` |
| 有限样本误差上界 | 完成但保守 | 90% 名义覆盖，经验 98%；上界偏宽 |
| 跨数据集完全稳定 | 未完成 | Santinha 较弱，不能写普适有效 |
| 不同模态正式端到端复制 | 未完成 | cytokine/chemical 已有独立线，但不是同一 formal scGPT–GEARS 合同 |
| 生物机制与高风险类别解释 | 部分完成 | 已确认预测器依赖，仍需通路/细胞类型层面的机制分析 |

当前实验包已经具备一篇可靠性方法论文的完整主干，稳定二区是合理投稿目标，一区可以冲，但任何分区或录用都不能由代码和实验“百分百保证”。一区最值得继续补的内容是：独立模态的正式预测器合同、更加紧致的分布偏移误差界、以及高风险基因/细胞背景的生物机制解释。继续堆同类数据集的边际收益已经低于这三项。

这里的“稳定二区”指证据完整度已经达到认真选择二区目标期刊的水平，不是对第三方编辑决定的概率承诺。期刊分区不是自动录用阈值；选刊范围、新颖性判断、独立同行评审、同期竞争和期刊政策仍会影响结果。详细解释和项目特有风险见上方“录用判断与项目总账”。

## 当前阅读顺序

1. `E113_formal_three_dataset_meta_audit_20260713/reports/E113_REPORT.md`
2. `E111_target_specific_mechanism_audit_20260713/reports/E111_REPORT.md`
3. `E114_split_conformal_error_bounds_20260713/reports/E114_REPORT.md`
4. `E108_formal_dual_model_risk_audit_20260713/reports/E108_REPORT.md`
5. `E112_external_formal_dual_models_20260713/E112_REPORT.md`
6. `E110_nested_hard_calibration_audit_20260713/reports/E110_REPORT.md`

## 复现入口

```bash
python tools/scripts/run_e105_context_graph_adapter.py --mode smoke
python tools/scripts/run_e106_frangieh_context_scgpt.py --fold all
python tools/scripts/run_e107_frangieh_context_gears.py --fold all
python tools/scripts/run_e108_formal_dual_model_risk_audit.py
python tools/scripts/run_e109_inner_hard_setting_predictions.py --outer all
python tools/scripts/run_e110_nested_hard_calibration_audit.py
python tools/scripts/run_e111_target_specific_mechanism_audit.py
python tools/scripts/run_e112_external_formal_dual_models.py --dataset all
python tools/scripts/run_e113_formal_three_dataset_meta_audit.py
python tools/scripts/run_e114_split_conformal_error_bounds.py
```
