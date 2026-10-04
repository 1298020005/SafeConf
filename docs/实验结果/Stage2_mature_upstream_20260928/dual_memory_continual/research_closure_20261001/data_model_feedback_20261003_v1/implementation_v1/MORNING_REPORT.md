# SafeConf implementation receipt

## 实际完成

- Preflight: `PASS`。
- 当前 truth contract PertEMA: `COMPLETE`，P6 与 Native61 分开保存。
- 公共可靠性：E258 jackknife 验证 `PASS`；McFaline 仅生成特征和开发比较。
- Source gate：完成 always-public、三种切换比例和五个置乱种子；真实标签 gate 相对 always-public 在两个方向均有正的 5000 次基因簇 bootstrap 区间。当前 source cache 没有实验级历史向量，因此 gate 使用既有 `prior_uncertainty`；E258 的 J 只作验证结果，未被冒充接入。
- Target TabPFN：DEV gate 后完成当前 212 holdout 固定分数，状态 `COMPLETE`。
- 完整当前系统：`COMPLETE`。

## 采用决定

- 当前默认保持 `PublicRule`。
- Target XGB 的点估计高于 PublicRule，但配对区间跨零，未替换默认。
- PertEMA P6/Native61 当前适配低于 PublicRule，保留为公平适配结果，不宣称官方模型复现。
- TabPFN Target 50% 的 DEV 点增益未形成当前 holdout 的稳定收益，保留为固定学习器对照。
- Jackknife 能预测 E258 留出参照偏差，但与历史分散度高度相关，暂不升级为默认风险规则。

- 独立资产核验：Replogle GWPS 有数据但没有合格冻结预测；Nadig 有预测但属于 SEEN 同研究；E192 已用 10000-scale 公共效应完成 173 个任务的跨背景审计，但仍是同研究 SEEN 证据；E208 Jiang24 的两个上游家族均未通过 no-change competence gate，已保留为负结果，不绕过门生成测试预测。

## 失败与修复

- Source gate 首次运行在汇总阶段出现索引错误；已修复指标聚合并在 `source_gate_v3` 完整重跑真实与置乱标签流程。
- Native61 的部分字段为 NaN；采用 XGBoost 原生缺失值路径，完整任务覆盖与有限字段覆盖分别登记。
- PublicMeanJackknife 有 210/212 个完整任务，系统排序未将缺失值伪造成零，回退并单列覆盖。

## 当前系统指标

- `Amplitude`：U20=-0.210252，AURC=0.029116，有效 context=3/3。
- `PublicRule`：U20=0.837668，AURC=0.024944，有效 context=3/3。
- `TargetH1F1_50`：U20=0.839911，AURC=0.025083，有效 context=3/3。
- `TargetX0F2_50`：U20=0.859384，AURC=0.025108，有效 context=3/3。
- `TargetTabPFN50`：U20=0.800915，AURC=0.024932，有效 context=3/3。
- `PertEMA_Native61_50`：U20=0.136321，AURC=0.028264，有效 context=3/3。
- `PertEMA_P6_50`：U20=-0.045179，AURC=0.028150，有效 context=3/3。

## 混合历史队列（SEEN）

- `Amplitude`：U20=0.124787，复核20%发现 154/599 个真实高误差任务，剩余平均误差=0.029771。
- `PublicRule_mixed`：U20=0.231355，复核20%发现 187/599 个真实高误差任务，剩余平均误差=0.029555。
- `SourceHGB_mixed`：U20=0.076827，复核20%发现 155/599 个真实高误差任务，剩余平均误差=0.029868。
- PublicRule_mixed 相对 Amplitude 多发现 33 个高误差任务，剩余误差减少 0.000216；这是冻结分数的 SEEN 回顾，不是独立确认。

## E192 跨背景审计（SEEN）

- `GEARS_seed3407`：PublicRule U20=0.562103，Amplitude U20=0.702569，点差=-0.140466。
- `GEARS_seed3408`：PublicRule U20=0.638821，Amplitude U20=0.674467，点差=-0.035646。
- `GEARS_seed3409`：PublicRule U20=0.542807，Amplitude U20=0.695891，点差=-0.153083。
- `scGPT_seed3407`：PublicRule U20=0.170629，Amplitude U20=0.122285，点差=0.048344。
- `scGPT_seed3408`：PublicRule U20=0.219523，Amplitude U20=0.223864，点差=-0.004341。
- `scGPT_seed3409`：PublicRule U20=0.215838，Amplitude U20=0.118049，点差=0.097788。
- 20 个基因簇的 5000 次区间均跨零；该结果用于跨背景适用范围，不升级默认方法。

## 下一动作

1. 继续保留 PublicRule 作为当前系统默认。
2. 以 E258/J 与 Source gate 结果形成适用范围和机制证据，不继续扩网络。
3. 独立确认资产仍按资格表处理；当前没有打开永久测试真值。
4. 正文、PDF和投稿材料仍暂停。
