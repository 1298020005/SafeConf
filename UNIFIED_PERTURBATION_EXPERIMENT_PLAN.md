# SafeConf 统一扰动任务表示：接口验证实验计划

**版本**：v0.1（2026-09-27）
**性质**：实验设计与验收协议，不是新的模型结果。

## 1. 要回答的实验问题

本轮不问“哪个风险网络分数更高”，而问三个可以被数据证伪的问题：

1. 同一套任务身份能否无歧义地表示基因单扰动、基因组合、化学单药和化学组合？
2. GEARS/scGPT/V0/ContextSim/CPA 等已有预测结果能否转换为同一 `predicted_effect` 输出契约？
3. 在基因和化学两条线上，公共字段、可选字段和历史/错误来源能否使用同一套泄漏规则审计？

如果任一问题失败，先修接口，不训练 SafeConf。

## 2. 输入和禁止事项

**允许使用**：

- 83 条 manifest 和 h5ad backed 元数据；
- E131/LOPO、E118 CPA、E208 Jiang24、E258 Feng、E260/E263 Tahoe 的既有报告、任务表、预测向量和 split manifest；
- 已有 adapter、NPZ、任务级 CSV。

**禁止**：

- 重新下载或重新预处理数据；
- 载入所有表达矩阵做大规模计算；
- 打开 E208/Tahoe 的 sealed test treated truth；
- 训练或扩展 E274 风险模型；
- 把只有 RMSE/相关系数的汇总表当成预测向量。

## 3. 实验矩阵

| 阶段 | 实验 | 主要输入 | 产出 | 资源 | 通过门槛 |
|---|---|---|---|---|---|
| T0 | 元数据冻结 | manifest、83 个本地 h5ad | `inventory.csv`、`STATUS.json`、输入 hash | CPU/磁盘；backed 读取 | 83/83 可读，0 下载，0 表达矩阵载入 |
| T1 | 字段覆盖审计 | T0、E131/E118/E208/E260 | 每条记录的硬字段状态、缺失 mask、subtype confidence | CPU | 无静默 null；基因和化学各至少一套 valid |
| T2 | task_id round-trip | 基因、化学、Tahoe 任务元数据 | canonical JSON、task-key 冲突表 | CPU | 0 collision；换 predictor 不换 key；dose/time/context 改变必换 key |
| T3 | 预测适配器 round-trip | 既有 GEARS/scGPT/V0/ContextSim/CPA 向量 | canonical prediction table、adapter report | CPU；仅读取向量 | 0 metric-only 被接受；axis/shape/finite/provenance 全通过 |
| T4 | 跨类型公共契约 | 一套基因线、一套化学线、Tahoe 任务层 | 公共字段对照表、失败码对照表 | CPU | validator 和失败码相同；类型字段只走对应 mask |
| T5 | split/泄漏审计 | LOPO、E118、E208、Tahoe split | provenance ledger、truth-access report | CPU/只读 | sealed truth 未读；无 OOF 证据的错误不进 history |
| T6 | 缺失字段压力测试 | T1/T3 的副本 | masking matrix、fail-closed report | CPU | 公共分支可运行；专用分支显式关闭；硬字段缺失硬失败 |
| T7 | 后续模型比较 | 仅在 T0–T6 全通过后 | 基因/化学分线风险结果 | 后续再登记 GPU/CPU 预算 | 不合并 headline；先过各线 coverage/risk 门槛 |

当前应执行并归档 T0；T1–T6 是下一步可直接执行的验证队列。T7 明确保持 pending。

## 4. T1–T6 的具体实验定义

### T1：字段覆盖审计

把每个数据文件中实际存在的字段映射到 `UnifiedTaskRecord`：`context_id`、`perturbation_type`、`target`、`condition`、`control_reference`、`output_contract`。每个字段必须输出四态值：`known`、`unknown`、`not_applicable`、`invalid`。把“无法从 manifest 证明机制”记录为 unknown，不根据文件名猜 KO/CRISPRi。

统计以下量：硬字段覆盖率、类型细化率、背景数、任务数、剂量/时间/guide/SMILES 缺失率、split 可追溯率、控制状态可追溯率。验收要求是没有静默缺失；unknown 可以存在，但必须阻止依赖该字段的专用分支。

### T2：身份 round-trip

用规范化 JSON 构造 `task_id`。至少做以下成对测试：

- 同一任务换 `GEARS`、`scGPT`、`CPA`，task_id 必须不变；
- 同一化合物 dose 从 0.1 改为 1.0、处理时间改变、context 改变，task_id 必须改变；
- 组合基因/组合药物成员换顺序，task_id 必须不变；
- `unknown` 与 `not_applicable` 不能被编码成同一个隐式空字符串；
- 单背景数据仍然有显式 context_id。

输出 collision、duplicate、normalization mismatch 三类失败表。任何 collision 都阻断后续历史构造。

### T3：预测适配器

对已有预测产物执行三种路径：

1. 原生 effect：直接登记 `output_kind=predicted_effect`；
2. post-expression：只有匹配 control 与同一 gene axis 同时存在时，执行差分；
3. 指标表：拒绝并写入 `metric_only_rejected`。

每条输出都验证任务键、预测器版本、adapter 版本、向量长度、有限值、axis hash、split、训练来源和 `target_truth_accessed`。不得为缺失的预测向量填零。

### T4：跨类型契约

取一组 E131/E208 基因任务和一组 E83/E118 化学任务，使用完全相同的公共 validator。比较失败码、字段类型和 mask 规则；单独检查基因机制字段、药物 dose/time/vehicle 字段。Tahoe 只用 E260 的任务元数据测试化学任务层，不进入预测层。

成功标准不是两类的生物学误差相同，而是两类都能通过同一层公共契约，并且特有信息缺失时行为可预期。

### T5：split 和历史泄漏

建立 `provenance_ledger`：每一条错误记录写入来源任务、预测器版本、fold、是否 OOF、训练数据 hash、是否访问目标真值。E208/Tahoe 的 sealed test 在 freeze 前只能出现 `task metadata`，不能出现 treated truth。没有证据证明来自未参与训练任务的错误记录统一标记 `provenance_unverified`，不进入公共历史或错误记忆。

### T6：缺失字段压力

复制合法记录，依次遮蔽 dose、time、guide、mechanism、SMILES、batch 和 replicate。检查：

- 公共 SafeConf 特征仍可在硬字段完整时运行；
- dose-response、guide-level、MoA 等专用特征显式关闭；
- unknown 和 not_applicable 保持可区分；
- context、target、control、output axis、预测向量缺失时 fail closed。

## 5. 后续模型实验的登记方式（不在本轮启动）

接口通过后再登记三条对照：

- **Shared-only**：只用公共任务和预测向量特征；
- **Shared + typed**：共享主干、扰动类型 embedding、字段 mask 和小型类型校准器；
- **Separate calibrators**：共享数据契约但基因/化学分开校准。

基因线使用 E131 开发、E208 或 E258 外部确认；化学线使用 E118/sciPlex 开发、Tahoe 或独立化学任务确认。每条线单独报告 coverage、风险和外推 split；只有两条线都通过预注册门槛，才讨论统一模型是否成立。不要把“接口统一”写成“性能已统一”。

## 6. 产物和复核顺序

1. `UNIFIED_PERTURBATION_TASK_SPEC.md`：字段、准入和映射规范；
2. `PERTURBATION_DATA_INVENTORY.md` 与 CSV：现实数据覆盖；
3. T0–T6 的 validator 输出和状态 JSON；
4. 通过后再提交模型实验预注册和资源计划。

当前提交包含 1、2 和 T0 的结果；没有新模型结果，也没有改变 E274 状态。
