# E27 scGPT forward PredictionRecord smoke

先看结论：E27 已经让 scGPT 从“资产存在但默认 import 失败”推进到“可用归档源码 + whole-human checkpoint 生成 strict PredictionRecord smoke”。

- PredictionRecords：3
- strict issue：0
- selected perturbations：control, RPL3, NCBP2, KIF11
- selected genes：128

边界：这是 forward-only smoke，不是正式 scGPT 性能实验。下一步要把 scGPT 与 GEARS 放到同一任务/同一 gene panel 上做 adapter 对齐。
