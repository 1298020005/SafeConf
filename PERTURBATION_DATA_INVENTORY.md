# PERTURBATION_DATA_INVENTORY

**盘点日期**：2026-09-27
**状态**：metadata-only inventory complete
**范围**：当前 manifest 和服务器已有本地文件；不重新下载、不载入表达矩阵、不启动模型。

## 结论摘要

- manifest 共 83 条本地数据记录，83/83 `READ_BACKED`，0 个缺失，0 个读取错误。
- 盘点时只读取 h5ad 的 backed `.obs`/维度/字段，不读取表达矩阵 `X`。
- manifest 粗分类为：`genetic_single` 59、`genetic_combinatorial` 4、`chemical_single` 13、`chemical_combinatorial` 1、`enhancer_regulatory` 6。
- 粗分类不能直接替代实验机制。例如 `genetic_single` 不自动等于 knockout；具体机制必须按原始协议核对。
- 逐文件字段、背景数、扰动数、任务数、剂量/时间/药物字段见 [`PERTURBATION_DATA_INVENTORY.csv`](docs/数据审计/20260927_perturbation_inventory/PERTURBATION_DATA_INVENTORY.csv)，机器状态见 [`STATUS.json`](docs/数据审计/20260927_perturbation_inventory/STATUS.json)。

## 当前最适合做接口验证的线路

| 线路 | 数据集/实验 | 适合用途 | 当前边界 |
|---|---|---|---|
| 基因开发 | E131/LOPO 的 Frangieh、Lara、Cui、Tian 等 | 任务键、基因对象、LOPO split、已有 V0/GEARS/scGPT 适配 | 机制/guide 在部分文件中缺失；需保留 unknown，不得强行标注 KO。 |
| 基因外部确认 | E208 Jiang24、E258 Feng | context/condition 映射、封存真值和 donor/context-held-out 泄漏审计 | E208 test treated truth、E258 最终测试真值不能提前打开。 |
| 化学开发 | E84/E87/E89/E118 CPA 与 sciPlex3/OpenProblems/sciPlex4 | compound/dose/time/control、CPA effect 向量适配、跨数据集契约 | E118 既有报告的化学独立增量门未通过，不能在接口验证中改写该结论。 |
| 化学外部确认 | Tahoe E260/E263 | cell line、drug、dose、plate、matched control、sealed split | 本轮只用任务元数据；不追加 Tahoe 模型或读取 test treated truth。 |
| 组合/扩展 | Norman、Wessels、TianActivation/Inhibition、调控/刺激数据 | 组合对象排序、机制枚举和可选字段压力测试 | 没有合法逐任务预测向量的记录不进入风险训练。 |

## 实验接口要求

每条记录都要区分：

1. **任务层**：`task_id`、背景、扰动类型、对象、条件、对照、输出空间；
2. **预测层**：上游模型、版本、适配器、预测向量、输出类型、split 和 truth-access provenance。

具体 schema、T0–T7 验证实验和接受标准见 [`UNIFIED_PERTURBATION_TASK_SPEC.md`](UNIFIED_PERTURBATION_TASK_SPEC.md)。本盘点本身不是模型效果报告。
