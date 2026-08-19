# E27 scGPT forward PredictionRecord smoke

生成时间：2026-07-08T03:12:03

## 结论

E27 修复了“scGPT 资产存在但默认 import 失败”的定位问题：默认 conda env 的 `scgpt.pth` 指向旧路径；使用归档源码路径后，scGPT 可以 import，并且 whole-human checkpoint 可以完成 forward-only smoke。

- PredictionRecords: 3
- strict issue_count: 0
- selected perturbations: control, RPL3, NCBP2, KIF11
- selected genes: 128
- checkpoint matched keys: 129 / 176

这只是 scGPT 第二模型 adapter 的 smoke。它不代表正式 scGPT 性能，也不替代和 GEARS 的统一 benchmark 对齐。
