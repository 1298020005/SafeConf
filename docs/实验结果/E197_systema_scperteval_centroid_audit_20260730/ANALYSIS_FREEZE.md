# E197｜E190/E192 的 Systema 与 scPertEval 均值层评价冻结

冻结日期：2026-08-01
性质：`POSTTRUTH_EXPLORATORY`。E190 与 E192 的预测曾在真值读取前锁定，但两项
实验现在已经开封；E197 的指标协议不是当时预注册的，因此只作为旧结果的事后
敏感性分析。E197 不改变 E190/E192 的 PASS、ABSTAIN 或失败结论。

## 1. 固定输入与软件

设置固定为：

- `E190_K562`：Adamson K562 → Replogle K562，692 个 batch×gene 任务、47 个基因；
- `E192_RPE1`：Adamson K562 → Replogle RPE1，175 个 batch×gene 任务、21 个基因。

每个设置只读取已锁定的预测、目标真值、query order、truth index、最终 task
metrics，以及 model assets 中各自独立的 512 基因轴、source control、train
effects、source-gene effects 与 target-control profiles。E190/E192 的 512 基因轴
不能拼接或直接比较；任务顺序、模型键、数组形状或输入哈希不一致即停止。

官方软件固定为：

- scPertEval commit `8709eb07a0e7d4ecf1c60c977f2018690a749975`；
- Systema commit `aaf5b5353993b48b78543f2f93b3e18ca65df515`。

E197 保存两个仓库的实际 HEAD、clean-worktree gate 和所调用核心文件哈希。

## 2. 预测器与简单基线

每个设置固定 12 个 predictor：

1. `scGPT_seed3407/3408/3409`；
2. `GEARS_seed3407/3408/3409`；
3. 六成员均值 `family_centroid`；
4. `matching_source_train_mean`：target control 加同基因 Adamson train-only
   guide-balanced mean effect；
5. `matching_source_all_folds_mean`：target control 加既有 Adamson all-fold
   source-gene effect，只作为更强的敏感性基线；
6. `target_control_plus_source_mean_effect`：target control 加 Adamson train 的全局
   guide-balanced mean effect；
7. `source_absolute_noncontrol_mean`：Adamson source control 加同一个全局 mean
   effect，保留 source 的绝对表达基线；
8. `zero_effect`：target control。

Adamson 每个 guide 的四个 train 伪重复先按实际细胞数恢复 guide effect，再对 guide
等权。validation effects 不进入 train-only matching 或全局 mean effect。all-fold
matching baseline 明确单列，不能与 train-only baseline、target-adapted mean 或
source-absolute mean 混用。正式输出保存逐 predictor 输入审计表。

## 3. Systema 参考空间

Systema 官方思想是把 control reference 换成 perturbed reference，以削弱所有扰动
共有的系统变化。跨数据集时，source 与 target 的绝对表达基线不同，因此固定报告
一个 Systema-inspired 适配和一个训练集原始参考：

- 主分析（`systema_inspired_transported_source_reference`）：对任务 i 使用
  `r_i = target_control_i + source_mean_effect`。所以比较的是
  `predicted_effect_i - source_mean_effect` 与
  `true_effect_i - source_mean_effect`；
- 敏感性分析（`systema_source_train_reference`）：使用
  `r_source = source_control + source_mean_effect`。它忠实保留训练集绝对 perturbed
  centroid，但可能混入 Adamson→Replogle 的平台或批次偏移。前一个参考不是
  Systema 官方原式，任何表和报告都必须保留 `inspired` 名称。

固定输出：

- 主参考的全 512 基因 Pearson-Δ；
- 按目标真实 effect 绝对值稳定排序的 top20 Pearson-Δ。它是事后
  `absolute-effect top20 proxy`；现有文件没有逐细胞分布和 DE 统计量，不能称为
  官方 Systema Pearson-Δ20；
- 只按 Adamson train-only matching effect 选出的 source-top20 Pearson-Δ；
- source-absolute 参考的全基因 Pearson-Δ；
- control-reference effect Pearson、cosine、RMSE 和 post-state RMSE。

Pearson/cosine 遇到常数、零范数或非有限向量必须为 NA，并保存原因，不能填 0。

## 4. Systema centroid accuracy

固定保留两个层次：

1. `systema_gene_centroid_accuracy`：对每个目标基因，把各 batch 的真实和预测
   centroid 按该 batch 的目标细胞数合并，再在 47 或 21 个 selected genes 之间
   使用官方欧氏距离公式。固定同时报告 `matched_control_effect`（先减去每个 batch
   的 target control，作为主解释）和 `post_state`（保留 batch 构成，作为敏感性）
   两个空间。正确 centroid 比其他 centroid 更近的比例为 accuracy；
2. `systema_context_stratified_centroid_accuracy`：保持 batch×gene 任务单位，只与
   同一 target batch 的其他 gene truth centroids 比较。该项用于检查 context 内
   路由，不冒充官方未分层结果；同 batch 没有竞争任务时为 NA。

两种 accuracy 都采用严格 `<`；tie 计失败，同时保存 nearest-centroid hit 和竞争
centroid 数量。

## 5. scPertEval 的合法边界

E190/E192 只保存每个任务的真实和预测均值，没有预测单细胞群。scPertEval 主协议
先按 `n_target_cells` 把各 batch 的真实和预测 effect 合并成每个 target gene 一个
centroid，所以 E190 为 47 个、E192 为 21 个 perturbation labels；再增加一个零
effect control row，调用官方 `prepare`/`score` 接口，`min_cells=1`。

允许协议固定为：

- `pearson`
- `pearson_ctrl`
- `pearson_pert`
- `mse`
- `rank`
- `transpose_rank`

rank 在当前设置的全部唯一 target genes 上运行。同一 gene 的其他 batch 不会作为
竞争 perturbation 重复进入 rank 或 `pearson_pert`。这里只能称为 gene-level
effect-centroid pseudobulk score。禁止 MMD、Energy、Sinkhorn、DE-AUPRC、DE-AUROC、
DE-overlap、WMSE、DE/top-k feature space；禁止复制均值向量伪造预测细胞；禁止
运行 calibration，因为单个 pseudobulk row 不能估计 split-half reproducibility。

## 6. 风险端点与统计量

风险量只取 E190/E192 已保存的：`diversity_lower_bound`、
`diameter_half_lower_bound`、`predicted_magnitude`、
`source_effect_magnitude`。任务端点固定包括六成员均值误差、family-centroid 误差、
Systema-inspired Pearson error、直接 effect MSE，以及 context-stratified centroid
error。scPertEval MSE/rank 的评价单位是跨 batch 合并后的 gene centroid，只用于
预测器多指标比较，不回填到 batch×gene 风险关联，避免混淆统计单位。

每个 setting×risk×endpoint 固定报告两个估计对象：

- `task_weighted_gene_cluster_bootstrap`：batch×gene task-level Spearman；同一 gene
  的全部 batch tasks 整簇抽样；
- `gene_equal`：先在每个 gene 内等权平均 batch tasks，再让 47/21 个 gene 等权。

两者各 bootstrap 5,000 次。seed 固定为
`SHA256("E197\0" + estimand + "\0" + setting + "\0" + risk + "\0" + endpoint)`
前八位十六进制。
有效任务少于 20 或有效 gene 少于 5 时只报告 point/NA interval。区间是描述性
cluster-resampling interval，不写成 iid 置信区间或跨数据集总体推断。

## 7. 执行与结论边界

- `prepare` 和 synthetic smoke 不得加载正式 prediction/truth 数组内容；
- 正式运行必须显式传入 `--allow-posttruth-evaluation`；
- 正式运行前，runner、freeze、prepare gates 和 input hashes 必须提交并推送；
- 正式运行还必须核对当前 HEAD 与 GitHub、Gitee 两个远程分支 tip 完全一致；
- 状态使用 `RUNNING`、`FAILED`、`COMPLETE` 原子写入；COMPLETE 后保存正式输出
  SHA-256；
- 官方 scPertEval 分数与独立公式必须具有相同 NA mask，有限值最大差不得超过
  `1e-8`；独立 `pearson_pert` 复算保持官方输入的 float32 centroid 求和顺序，再以
  float64 计算相关性，该容差只覆盖浮点舍入；
- 已存在正式 payload 或 COMPLETE 状态时拒绝覆盖，失败产物必须先归档或移除；
- 图固定白底，同时输出 300 dpi PNG 和矢量 PDF；
- 无论正负都保留。E197 不能写成新的盲法确认、完整 population benchmark、官方
  Systema 数据复现、通用 SafeConf 优越性或期刊录用保证。
