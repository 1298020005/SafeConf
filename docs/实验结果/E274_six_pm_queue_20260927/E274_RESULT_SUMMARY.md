# E274 下午队列结果摘要

时间：2026-09-27 17:35–17:59 CST。结果均为开发/合同审计阶段，未读取 Tahoe sealed test treated truth。

## 1. Tahoe 原始历史审计（完成）

- E263 Parquet 共 265,984 条选中记录：230,400 条 `train_validation_treatment`，35,584 条 `available_control`。
- 1,800/1,800 个 train/validation treatment key 与 E260 锁定面板匹配。
- 每个 key 最多 128 个细胞；稀疏表达每细胞 378–6,591 个基因条目；有限值比例 1.0。
- `test_treated` 输出记录为 0。
- 初版脚本曾把 control key 误和 treatment panel 比较，出现假性 mismatch；已修正为按 role 只核对 treatment，并保留初版目录作为工程审计痕迹。最终以 `tahoe_raw_audit_v2/STATUS.json` 为准。

这一步只证明抽取和面板合同可用，尚未证明上游 predictor competence，也没有把表达聚合成风险特征。

## 2. Error Memory 可用率曲线（完成）

在 fold 0 的 train/validation、5 个整体扰动冷桶上，固定 `P+Q+H`，将误差记忆池按固定哈希模拟为 0/5/10/25/50/100%：

| E 可用率 | Utility@20 宏平均 | Spearman 宏平均 | 相对 0% 的 Utility 增量 | 正向单元 |
|---:|---:|---:|---:|---:|
| 0% | 0.6795 | 0.7030 | 0 | 102/102 |
| 5% | 0.7052 | 0.7159 | +0.0257 | 42/102 |
| 10% | 0.7032 | 0.7169 | +0.0238 | 54/102 |
| 25% | 0.7138 | 0.7355 | +0.0344 | 56/102 |
| 50% | 0.7402 | 0.7467 | +0.0607 | 64/102 |
| 100% | 0.7493 | 0.7501 | +0.0699 | 66/102 |

这是对现有开发误差表的敏感性分析；不是新的 OOF 证明。它支持“反馈量可能影响 E 的价值”这一待验证假设，不支持直接把 E 写成已验证主贡献。

## 3. GPU 多种子稳定性（完成）

两张 RTX 6000 各用 8 个随机种子、80 epoch、相同 7 数据集 × 3 predictor × 5 冷桶合同。完整 `P+Q+H+E` 相对 `P+Q` 的逐单元增量：

| 设备 | Utility 增量均值 | Utility 正向单元 | Spearman 增量均值 | Spearman 正向单元 |
|---|---:|---:|---:|---:|
| GPU0 | +0.0388 | 56/102 | +0.0562 | 64/102 |
| GPU1 | +0.0220 | 55/102 | +0.0524 | 62/102 |

方向与幅度有一定稳定性，但并非所有单元正向；GPU 结果仍继承开发 feature matrix 和误差标签的 provenance 限制。它能支持风险 learner 的工程可运行性，不能替代盲测。

## 4. 本轮判断

1. Tahoe 的 raw history 管线可以继续，但下一步必须先做固定基因轴、同板 control pseudobulk 和上游 competence gate。
2. E 的曲线有梯度，但当前不能证明是合法 OOF error memory；在 OOF 字段补齐前，论文主线暂时采用 `P+Q+H`，E 作为预注册 optional 分支。
3. 不启动 Deep Sets/门控 V1。先完成同标签预算的简单 concat、残差和 no-history 对照；若 H/E 的分层区间不稳定，方法主动收缩。
4. 17:35–18:00 队列已全部形成可审计输出；GPU/CPU 已在问题完成后释放，不以空转设备作为科研指标。

## 5. 复现入口

- 方法判断：[20260927_SafeConf_Codex_主方法审查与实验计划.md](../../方法设计/20260927_SafeConf_Codex_主方法审查与实验计划.md)
- Tahoe：`tahoe_raw_audit_v2/STATUS.json`、`TASK_COUNTS.csv`、`EXPRESSION_ROLE_SUMMARY.csv`
- E 曲线：`error_memory_curve_cpu/SUMMARY.csv`、`DELTA_VS_ZERO.csv`
- GPU：`gpu_stability_gpu0/SUMMARY.csv`、`gpu_stability_gpu1/SUMMARY.csv`
