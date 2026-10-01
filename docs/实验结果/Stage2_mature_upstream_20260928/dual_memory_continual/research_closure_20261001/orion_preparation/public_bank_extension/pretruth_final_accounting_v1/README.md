# Final pretruth accounting append

实际 partitioned pretruth seal 与固定13候选 comparison 已完成，独立43项检查全部通过；232条已见元数据限定查询均有真实同gene历史且13候选有限，覆盖144个生物gene clusters（HCT116 107、HEK293T 125）。446条旧历史查询的scores/features/priors/pairs/weights全部bytes保持，原七个Source模型、R模型和competence结果复用，新增拟合为0。该PASS仅指执行、契约和预评分覆盖，不是SafeConf效果或优越性结论；TEST真值仍未用于本包，metadata为SEEN。后续需独立Root的once-only TEST操作授权。

| 新增独立阶段 | Wall seconds |
|---|---:|
| 成功decimal TRAIN/NTC bridge | 90.2187113580294 |
| Legacy batch诊断 | 41.740193638019264 |
| 最终partitioned seal | 37.18312448798679 |
| 固定comparison | 未记录，CSV留空 |

三个已测阶段合计169.14202948403545秒；此合计不等于全pipeline成本。CPU seconds和comparison wall没有精确测量，不推断。现有334成本行中的Source银行、聚合、失败bridge、R fit、COMP、original seal和失败expanded seal均不重复追加。四个新阶段均无新fit/GPU/上游调用；既有R两context模型与Source七模型复用。

`COST_MASTER_APPEND_ROWS.csv`保留主表列格式；`RESULT_INDEX_APPEND_ROWS.csv`只加入当前index不存在的小manifest/review；`EXECUTION_STATUS_APPEND.json`是建议merge对象。所有共享主表保持原样，runtime数据未复制。
