# 发给 Claude：GEARS Frangieh adapter smoke 复核

请客观复核，不要默认同意 Codex。重点判断：这个 GEARS smoke 是否足以作为“GEARS 已能接入”的证据，以及下一步是否应该扩大成 GEARS formal validation。

## Codex 做了什么

我在服务器上修了 GEARS PredictionRecord 导出脚本：

- 文件：`code/20260426_154505_perturb_transport_final_push/safetrans_confidence/cli/run_gears_prediction_records.py`
- 改动：
  - 增加 `frangieh` 数据集入口。
  - 保留 Frangieh 的 `condition` 作为 `cell_type`，避免被 GEARS 自己的 `condition` 字段覆盖。
  - 让 GEARS 能在 Frangieh 上训练、预测，并导出 PredictionRecord。

然后跑了 3 个 seed：

- 输出目录：`code/20260426_154505_perturb_transport_final_push/outputs/gears_frangieh_adapter_smoke_20260605/`
- 文档目录：`docs/实验结果/Formal_main_20260604/gears_frangieh_adapter_smoke/`

## 结果摘要

| 指标 | 数字 |
|---|---:|
| seed 数 | 3 |
| 成功 seed | 3 |
| 合计 PredictionRecord | 62 |
| unique perturbation | 58 |
| 平均 test MSE | 0.00144 |
| 平均 test Pearson | 0.9958 |
| 平均 top20 DE MSE | 0.00542 |
| 平均 top20 DE Pearson | 0.9382 |
| GEARS uncertainty 非空记录 | 0 |

## Codex 的初步判断

1. 这是一个成功的 adapter smoke：GEARS 能跑，Frangieh 能进，PredictionRecord 能导出。
2. 这不是 GEARS formal validation：只有 62 条 test PredictionRecord，而且当前 GEARS split 不是 SafeConf 7 主表的 held-out pair split。
3. GEARS 原生 uncertainty 没有导出；因此现在不能声称“已比较 GEARS uncertainty”。
4. 下一步更合理的是把 GEARS 扩大成 formal probe，而不是急着写论文：
   - 要么在 Frangieh 上增加 test records / folds；
   - 要么做 GEARS native uncertainty 导出；
   - 要么明确 GEARS 只能作为第三 predictor 的可接入性证据。

## 请你回答

Q1. 你是否同意这次结果只能叫 GEARS adapter smoke，而不能叫 GEARS formal validation？

Q2. 如果要让 GEARS 成为论文里有说服力的第三 predictor，下一步最低需要什么？

Q3. 当前 GEARS split 与 SafeConf held-out pair split 不一致，这是不是必须修？还是可以先作为 supplement probe？

Q4. GEARS 原生 uncertainty 为空时，是否还值得继续追 native uncertainty？如果追，应该查 GEARS 哪个输出或源码位置？

Q5. 你建议下一步优先：

- A. GEARS Frangieh formal validation；
- B. GEARS uncertainty export；
- C. 先跑 supplement 数据集；
- D. 先做 SafeConf learned ranker / LODO。

请给出明确排序和理由。

