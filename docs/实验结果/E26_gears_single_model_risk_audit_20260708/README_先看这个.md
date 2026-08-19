# E26 GEARS 单模型风险审计

先看结论：E26 在 E25 strict GEARS 包上做单模型风险分析。它能说明 GEARS 自身哪些线索和误差相关，不能说明 GEARS/scGPT/CPA 三模型统一验证已经完成。

## 关键数字

- PredictionRecords：54
- 数据集：3
- 可部署分数：4
- E25 strict issue：0
- GEARS native uncertainty：absent_in_e25_formal_records

## 文件

- `tables/GEARS_SINGLE_MODEL_ENRICHED_RECORDS.csv`
- `tables/GEARS_SINGLE_MODEL_SCORES.csv`
- `tables/GEARS_SINGLE_MODEL_EVAL_SUMMARY.csv`
- `tables/GEARS_SINGLE_MODEL_PARTIAL_SPEARMAN.csv`
- `reports/E26_GEARS_SINGLE_MODEL_RISK_AUDIT.html`

## 下一步

如果要冲更高投稿等级，E26 后面应该接 scGPT 或 CPA adapter，而不是继续只在 GEARS-only 上打磨。
