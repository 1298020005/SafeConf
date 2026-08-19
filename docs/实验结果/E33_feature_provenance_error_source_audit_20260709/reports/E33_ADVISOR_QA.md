# E33 输入来源与评价对象审计

生成时间：2026-07-09 20:16

## 核心结果

- 可前置使用的 score/feature：7
- 只能事后使用的项目：2
- leakage checklist：FAIL = 0，WARN = 1

## 给周老师的回答

1. 后续所有结果都要绑定 `predictor_name`。不能只写“预测错误”，要写清这个 RMSE 来自哪个 predictor。
2. `model_disagreement_risk` 确实使用模型输出。它的定位是 task-level difficulty，不是 per-model reliability。
3. `true_effect_magnitude` 只能作为事后混杂控制；`predicted_magnitude` 才能作为前置 baseline。
4. 如果一个 prospective score 使用了 held-out true effect，就不能进入主方法。

## 下一步

E33 已经把输入来源口径锁住。下一步可以触发 E34 小矩阵和 E35 整行/整列 holdout。

## 输出表

- `tables/E33_FEATURE_PROVENANCE.csv`
- `tables/E33_ERROR_SOURCE_MAP.csv`
- `tables/E33_MAGNITUDE_SPLIT_TABLE.csv`
- `tables/E33_LEAKAGE_CHECKLIST.csv`
