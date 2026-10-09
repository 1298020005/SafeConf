# v2.1 implementation repairs

1. Native control NaNs were initially confused with missing task identities. Repaired identity coverage check; official XGBoost missing-value handling retained. First attempt stopped before any fit or scoring.

2. Added missing math import; second startup had stopped before its first fit.

3. XGBoost sklearn wrapper cannot serialize estimator_type in this environment. Use the Booster serialization API, matching prior successful scripts. One startup fit was consumed; model parameters unchanged.

4. Initial remote extraction measured 526 s/250 rows. Switched to prefetch of permitted-row HDF5 chunk offsets using four HTTP workers; same role registry, cohort and data construction. Opaque cache retained; no confirmation expression materialized.

5. Added persistent HTTP sessions to Range reader to avoid a fresh TLS handshake per storage block. Existing opaque blocks/roles preserved; scientific parameters unchanged.

6. Coalesced adjacent permitted HDF5 storage blocks into at most 8 MiB HTTP requests. Same immutable cache keys and logical role guard; no extra genomic blocks requested. Restarted owned extraction, protected E208 untouched.

7. Adamson upstream train list already includes ctrl. Removed duplicate condition indexing before scoring; prepared Public vectors reused, no second source scan. Calibration proxies retain explicit control entry.

8. DEV native features were not in the TEST-only task lookup. Generated them from the same frozen train-NTC control reference using gene/state metadata; no new expression or evaluation-label fit.

9. After reviewing saved-stage resume, require the final role-scoped read receipt (not an early array file) before handoff. Predictor freeze now follows successful reload/input-isolation tests.
# 2026-10-09 接续：已执行的故障修复与外部能力决定

- 外部训练完成后，MLP的float32批量推理与小批重载出现约2.5e-7的差异，触发原容差断言。查明是批大小带来的数值归约差异；输入添加伪造答案列并不改变输出。
- 已用原权重统一为float64计算、一次保存float32结果；参数哈希未改变，原预测另存。180个跨状态/角色查询重载差为0，删除/添加答案列输出严格一致，未放宽容差。此修复新增拟合数0。
- 原始能力门实际结果：Ridge相对条件均值误差+5.11%，95%区间[4.48%,5.81%]；MLP+211.18%，区间[196.89%,225.35%]。最终确认未读取。
- 诊断MLP训练误差仍高于均值基线，控制特征尺度跨度较大，只有30次梯度更新。另立一次训练修复：相同网络/特征，训练区标准化、中心化响应、零初始化残差输出头，上游训练区内部10%基因留出选择停止点，再全训练区拟合。
- 修复墙时14.95秒，GPU0、下载0。Ridge仍+4.13%；MLP内部留出最优为第0步，回到均值输出，被原能力门的非退化要求拒绝。保留全部模型、学习曲线和失败门，不继续扫描同一模型。
- Gladstone改为DEV压力测试：公共规则相对幅度macro U20增量，修复Ridge+0.2330、95%区间[0.0491,0.3861]；退化MLP+0.2201、区间[0.0196,0.4478]。这些结果不登记为合格上游的独立确认。
- 已接续CM4AI/KOLF作者备用资产。浏览器元数据含10,167扰动/38,606基因，但响应为int8量化z-score，不用作原RMSE高精度真值。作者原始basic-QC计数文件的Range访问已重新打通，启动仅身份/计数结构/压缩块成本的有界预检。
- 仍以基因簇为生物统计单位；保留目标开发准备成本、原累计资源账本与E208进程。正文/PDF保持暂停。
