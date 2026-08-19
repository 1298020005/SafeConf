# E201 任务集合与 source 证据冻结

冻结日期：2026-08-02

## 任务集合

每个正式任务是一个 `target cell line × perturbation condition`。目标背景的同一
扰动在三个 source 背景中属于公开训练 condition，在 target 背景中属于官方
`unseen_cell`。目标至少有 10 个扰动细胞；≥30 个进入主分析，10–29 个只进入
低细胞数敏感性分析。

experimental batch 不拆成独立任务。它只用于在 source 中计算 batch-matched
control delta，以及在 target 中确认模型使用的 average control。任务 centroid
按实际细胞数加权，不把小 batch 复制成等权生物重复。

blind prediction view 的 obs 已事前给出固定任务数：K562 580、RPE1 467、
HepG2 480、Jurkat 481，共 2,008；主分析 1,808，低细胞数敏感性 200。

## 允许读取的 source 证据

每个 target 只读取对应 `E201_blind_<target>` 物理 H5AD：

- target control 可以读取；
- 另外三个公开细胞系的 train/validation 扰动可以读取；
- target 扰动行为 0，不能读取；
- 正式任务只使用 split 中的 train condition，validation condition 不进入
  source 特征。

对每个 source context 和扰动，先求扰动 centroid，再减去按该扰动各 batch
细胞数加权的 matched-control centroid，得到 source delta。三个 source delta
等权平均得到简单 source-transfer effect；可用背景之间相对等权中心的每基因
RMS 为 `source_delta_dispersion`。

只有一个 source context 时 dispersion 缺失，使用同一 target 内至少两个
source context 任务的事前中位数填补，同时保留 `dispersion_imputed=true` 和
`support_context_deficit=3-n_contexts`。不跨 target 借用填补值。

## 在模型预测前即可固定的列

- `n_target_cells`、`n_target_batches`；
- `n_source_cells`、`n_source_contexts`；
- `negative_log_source_cells = -log(1+n_source_cells)`；
- `support_context_deficit = 3-n_source_contexts`；
- `source_delta_dispersion` 及填补标记；
- `source_mean_delta` 向量及其 SHA-256；
- 主分析/敏感性分层。

这些列和向量在四种子预测完成前即可生成。输出必须记录每次 source X 访问的
细胞系、行类型和行数，并要求对应 target 扰动访问数为 0。

## 预测完成后追加的风险量

四个 seed 的任务 prediction centroid 记为 `p_s`，其均值为 `p_bar`：

- `family_disagreement = RMS_s,g(p_s-p_bar)`；
- `family_radius = max_s RMS_g(p_s-p_bar)`；
- `predicted_magnitude = RMS_g(p_bar-control)`；
- `model_source_gap = RMS_g(p_bar-(control+source_mean_delta))`。

每个原始风险分量在同一 target 的主分析任务（≥30 cells）上估计均值和总体
标准差，再把同一参数应用于该 target 的主分析与敏感性任务。单个分量方差为 0
时流程失败，不会静默删除该分量。E201 主风险分数固定为：

`mean[z(family_disagreement), z(model_source_gap),
z(source_delta_dispersion), z(negative_log_source_cells),
z(support_context_deficit)]`。

predicted magnitude 不进入主风险分数，作为必须击败的简单对照。目标误差打开后
同时报告原始关联、控制 magnitude 的 partial Spearman，以及固定 20% 人工复核
预算下的效用。权重、方向、标准化层级和任务筛选不会按结果调整。
