# E28 GEARS–scGPT shared Adamson smoke

生成时间：2026-07-08T12:46:11

## 结论

E28 在 Adamson 上构造了一个最小的 GEARS–scGPT 同任务、同 gene panel、同 true effect 的 strict smoke。

- predictors：GEARS_formal_seed1_subset512；scGPT_whole_human_forward_subset512
- perturbations：CCND3, DAD1, DERL2
- genes：512
- PredictionRecords：6
- strict issue_count：0

边界：这是合同和对齐 smoke，不是正式性能 benchmark。scGPT 使用 forward-only adapter；true effect 来自同一 Adamson 小子集，用于统一合同检查。
