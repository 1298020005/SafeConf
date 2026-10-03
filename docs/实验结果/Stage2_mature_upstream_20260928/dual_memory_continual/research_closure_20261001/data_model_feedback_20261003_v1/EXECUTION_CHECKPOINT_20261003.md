# SafeConf 执行检查点（2026-10-03）

## 作用

这是实验阶段的运行记录，不是论文正文、PDF或最终投稿结论。当前继续保留论文工作暂停状态；本文件只绑定已经完成的结果和仍在运行的实验。

负责人按既定合同推进三条线：公共历史、目标反馈学习、跨模型上游验证。已完成的线先固定证据，未完成的线继续运行；负结果不删除，也不通过增加网络掩盖。

## 已完成并固定

### 公共历史：GWPS 覆盖扩展

- Replogle K562 GWPS 已完成流式建库与评分，无新增 GPU、无读取 Orion 永久真值。
- 原历史覆盖 232 个任务；GWPS-only 覆盖 1,378 个任务；旧历史与 GWPS 的 union 为 1,379 个任务；相对旧全目标历史新增 1,147 个任务。
- union 与 GWPS-only 的差异由 XRN1 这个旧非 K562 历史目标造成；旧 K562 记录没有重复追加。
- 扩展覆盖任务的 Public direct-distance U20 = 0.307930，95% bootstrap CI [0.200574, 0.394654]；相对幅度基线点增量 0.173347，bootstrap 均值 0.162896，CI [0.034534, 0.291658]。
- 扩展覆盖任务的固定 HGB U20 = 0.215107，CI [0.101782, 0.323474]；相对幅度的点增量 0.080525，CI [−0.068055, 0.214849]。因此目前能确认的是公共直接参照的增量，不能把扩展 HGB 写成稳定胜出。
- 5 个固定哈希顺序 × 5 个覆盖比例已生成，仅报告覆盖变化；没有重新拟合风险模型，不改变既有 score seal。

证据：`gwps/ALL_METRICS.csv`、`gwps/ALL_PAIRED_COMPARISONS.csv`、`gwps/ACTUAL_COVERAGE_AMENDMENT.csv`、`gwps/PUBLIC_GROWTH_5_FIXED_ORDERS.csv`、`gwps/PUBLIC_GROWTH_COMPLETE.json`。

### 目标反馈：公平选型与固定评价

- DEV 使用 542 个任务、377 个基因、3 个折；重新执行了全局 query-ID 禁止和训练样本自实验排除；270 次开发拟合完成。
- F1（预测特征 + 对齐同背景 Public 特征）与小样本 HGB H1 被选为开发候选，DEV macro U20 = 0.782311，AURC = 0.024752。
- 固定评价使用 331 个反馈任务中的 212 个评价任务、228/152 个基因，反馈预算 10/25/50/75/100%，预测先冻结，再做 5,000 次配对 bootstrap。
- H1/F1 相对原 H0/F1 的配对 U20 增量仅在 10% 预算通过预设采用门；25–100% 的区间跨零或无增量。H1/F1 的固定点 U20 依次为 0.824656、0.796760、0.839911、0.755890、0.756103。
- 同评价池无反馈的强 Public 历史规则 U20 = 0.837668。因此目标反馈学习器目前不能替代强简单 Public 规则；H1 只作为小反馈预算下的候选保留，不能宣称一般性反馈收益。
- 反馈线实际新增 fit 数 315，低于 360 的硬上限；其余结果均复用已冻结输入或旧模型。

证据：`feedback/DEV_METRICS.csv`、`feedback/DEV_SELECTION_TABLE.csv`、`feedback/SELECTION_LOCKED_BEFORE_HOLDOUT.json`、`feedback/METRIC_INTERVALS.csv`、`feedback/PAIRED_COMPARISONS.csv`、`feedback/ADOPTION_GATES.csv`、`feedback/REUSE_CACHE_AUDIT.json`。

## 当前仍在运行

### SAMS-VAE 真实生成上游

- 训练服务：`safeconf-sams-v1-continuation-20261003.service`，PID 由运行状态文件记录，使用 GPU0；不触碰 E208 的长期进程。
- 继续使用已登记 McFaline generative-counterfactual 配置、原始数据 split 和真值隔离；训练期间只读 train/validation 表达，test 表达没有打开。
- 合成 truth-free 输入、读入隔离和向量化等价性检查已通过。当前 `TRAINING_STATUS.json` 的 `upstream_competence_status` 仍为 `PENDING_TRUE_GENERATION`，不能用 reconstruction val loss 代替预测能力。
- 训练结束后自动执行：冻结 checkpoint → 真实 metadata-only 生成 → 能力门 → 双向 source-error 迁移。若预算停止或能力门不通过，记录为模型线边界，公共历史和反馈结果继续有效。

证据：`sams/TRAINING_STATUS.json`、`sams/PREDICTION_INPUT_TESTS.json`、`sams/READER_ISOLATION_TESTS.json`、`sams/INDEXING_SPEEDUP_RESULT.json`、`sams/ENGINEERING_REPAIR_*.json`。

## 当前负责人决定

1. 不再扩 DeepSets、Transformer、GNN 或门控结构；已有 Public 内容利用和 GWPS 覆盖证据足以支撑下一步判断。
2. 公共默认优先采用同背景、支持加权的历史距离；GWPS 作为版本化扩展，覆盖条件与无历史状态单独报告。
3. 反馈学习器不按反馈量自动启用。H1/F1 只在训练侧明确通过门时可作为低预算候选；强 Public 规则仍是当前 incumbent。
4. 旧模型错误监督继续作为条件性组件验证；不能因为跨家族压力测试未稳定超过简单 Public 规则就删除其历史结果，也不能把它升级为普遍主贡献。
5. SAMS 只承担“不同预测机制是否提供可迁移错误经验”的一次独立压力测试，不阻塞已完成的公共和反馈线。
6. 正文、PDF、投稿打包保持暂停。下一次状态更新只在 SAMS 训练状态变化、真实生成能力门闭合或出现新的可复核错误时推进。

## 下一动作

- 继续监控 SAMS 训练，不在 `PENDING_TRUE_GENERATION` 阶段提前解释。
- 训练结束后运行并核验 metadata-only 预测、能力门、双向迁移和对应 5,000 次配对统计。
- 将 SAMS 结果与 GWPS/反馈结果放入统一结果矩阵，随后冻结方法采用范围；不根据永久评价集重新挑选算法。
- 保持现有分支和未提交论文文件不变，下一次提交只包含新实验审计产物。
