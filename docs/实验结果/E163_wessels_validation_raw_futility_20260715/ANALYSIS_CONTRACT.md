# E163｜Wessels validation-only raw-log-probability futility diagnostic

冻结日期：2026-07-15。

E163 是一次**训练后、test 解封前的 validation-only 去留诊断**。它回答的唯一问题是：E162 的 native PCA10 预测虽已塌缩为常数，条件特异的 `raw_log_prob` 是否至少能在 24 个 validation 组合上排序该模型自己的预测准确度。E163 不访问 Wessels test label、test expression 或 raw Wessels H5AD，不把 validation 结果写成独立验证，也不追溯改写 E162 的失败结论。

## 1. 已知事实与分析时点

- E162 attempt 002 的三个训练种子均已结束；3407 是固定主种子，3408、3409 是训练敏感性种子。
- 三个种子的 validation `raw_log_prob` 均为 24/24 个不同值；PCA10 MAP prediction、official combined confidence 与 predicted magnitude 在 24 个 validation 任务间均为常数。
- 因 prediction non-degeneracy gate 失败，E162 的 `TEST_LABEL_QUERY_EVENT.json` 从未创建；test label、test X 和 test truth 均未访问。
- E163 的终点是在看见 E162 prediction collapse 和 raw score 非退化之后才冻结，属于**validation-informed futility diagnostic**。它不能提供外部确认性证据。
- 早先 E159 的 Norman post-unseal 取证显示：raw score 对 PCA10 truth 的 Pearson accuracy 呈正趋势，对 raw selected-gene truth 的 Pearson accuracy呈负趋势。E163 必须并行输出 raw selected-gene truth 敏感性，不能只保留有利口径。

## 2. 允许输入与禁止输入

formal runner 只允许读取：

1. E161 development H5AD `perturb_processed.h5ad`，其中只有 train/validation 细胞；
2. E161 train-only PCA10、train-control prior、selected-gene axis 与 E161→E162 interface；
3. E161 release status/asset manifest；
4. E162 attempt 002 status；
5. 三个种子的已锁定 validation label-only score CSV 与 non-degeneracy gate JSON；
6. 已提交的 E163 runner 与本合同。

明确禁止：

- `/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/WesselsSatija2023.h5ad`；
- 任何 E160 test condition label 列表或 E162b test-label artifact；
- Wessels test cell count、test expression、test truth、test error 或 test endpoint；
- 重新训练、重算 validation score、改变种子、挑选 validation 子集；
- 用 validation 结果重新定义分数方向、主要 truth 或 gate。

formal 运行前，runner 和本合同必须与当前 Git `HEAD` blob 逐字节一致。所有允许输入必须匹配 runner 中固定的 SHA256；任一不符立即终止，不创建 release。

## 3. 固定任务与分数

- validation 任务数必须恰好为 24；每个 condition 恰好出现一次。
- 三个 score CSV 的 condition 集合及顺序必须完全一致，并与 E161 development H5AD 中 `e161_split == "val"` 的 24 个 condition 完全一致。
- 主种子固定为 3407；训练敏感性种子固定为 3408、3409。
- 唯一研究分数为 native `raw_log_prob`；值越大表示 confidence 越高。不得取反、标准化后换方向或改用 official combined/magnitude。
- 每个 score CSV 必须保留 `query_has_test_expression=False`、`query_has_y=False`、`query_has_y_pca=False`，且 selected-gene order hash 必须一致。

## 4. 固定 own-model prediction 与 validation truth

对每个种子 \(s\) 和 validation 任务 \(t\)，E162 已锁定 PCA10 MAP prediction \(\widehat z_{s,t}\)。E163 不重新前向推理。

### 4.1 主要 PCA10 inverse-transform truth

E161 的 PCA mean 为 (m\)，components 为 (W\)，train-control selected-gene mean 为 \(\mu_{ctrl}\)。validation task 的细胞级 PCA10 坐标均值为：

\[
z_t=\frac{1}{n_t}\sum_{i\in t}z_i.
\]

固定重构效应：

\[
e_t^{PCA10}=z_tW+m-\mu_{ctrl}, \qquad
\widehat e_{s,t}=\widehat z_{s,t}W+m-\mu_{ctrl}.
\]

主要 own-model accuracy：

\[
A_{s,t}^{PCA10}=\operatorname{Pearson}(\widehat e_{s,t},e_t^{PCA10}).
\]

PCA10 RMSE 是次要终点：

\[
R_{s,t}^{PCA10}=\sqrt{\operatorname{mean}_g(\widehat e_{s,t,g}-e_{t,g}^{PCA10})^2}.
\]

task mean 必须先对 E161 已规范化的 validation cells 求均值。PCA、gene axis 和 control prior 全部来自 train-only E161 资产。

### 4.2 强制 raw selected-gene truth 敏感性

令 \(\mu_t^{raw}\) 为 E161 development H5AD 中该 validation condition 的 selected-gene normalized/log1p 表达均值：

\[
e_t^{raw}=\mu_t^{raw}-\mu_{ctrl}.
\]

每个种子必须输出：

- `raw_pearson_effect_accuracy_sensitivity = Pearson(predicted effect, raw effect)`；
- `raw_rmse_effect_error_sensitivity = RMSE(predicted effect, raw effect)`。

这组结果无论符号是否与 PCA10 主结果一致都必须进入 task table、association table 和报告。它不替换主要 truth，也不参与 authorization gate。

## 5. 可估计规则

每个种子、每个 association 都检查：

- 24 个 score 与 endpoint 全部 finite；
- score 与 endpoint 各至少 2 个 exact unique values；
- 两者 sample SD (`ddof=1`) 均严格大于 `1e-12`。

不满足时 Spearman rho、bootstrap CI 记 `NA`，status 写入具体 failure code。不得把常数或非有限关联记为 0。

## 6. 固定主要统计与 authorization gate

每个种子的主要统计为：

\[
\rho_s=\operatorname{Spearman}(raw\_log\_prob_s,A_s^{PCA10}).
\]

预期方向为正。**test 后续授权门**只使用以下三个条件：

1. 3407、3408、3409 的主要关联全部可估计；
2. 主种子 \(\rho_{3407}>0\)；
3. 三个种子中至少 2 个 \(\rho_s>0\)。

三个条件同时满足才写 `authorize_future_test_label_lock=true`。task-bootstrap CI、component-gene cluster bootstrap、LOGO 和 raw-truth 敏感性全部强制报告，但不改变 gate。gate 通过只允许另行冻结新的 test-label-only 合同；不等于模型有效、SafeConf 有效或 Wessels 外部确认成功。gate 失败则该 E162 checkpoint family 的 raw-score Wessels test 路径停止，不得靠换 seed、取反或换 endpoint 复活。

## 7. task bootstrap

- RNG：`numpy.random.default_rng(3407)`；
- 10,000 replicates；
- 每次从 24 个 task index 有放回抽 24 个；
- 三个种子及所有四个 endpoint 使用同一 replicate index；
- CI：`numpy.quantile(finite_rhos, [0.025, 0.975], method="linear")`；
- finite 且可估计 replicate 少于 9,500 时 CI 记 `NA`。

四个 endpoint 均报告 bootstrap summary：PCA10 Pearson、PCA10 RMSE、raw Pearson sensitivity、raw RMSE sensitivity。分数与 accuracy 预期正相关，分数与 RMSE 预期负相关。bootstrap 不参与 authorization gate。

## 8. component-gene cluster bootstrap

只对主要 PCA10 Pearson association 执行。取得 24 个 pair 中所有不同 component genes，共 (K\) 个。每个 replicate 用新的 `numpy.random.default_rng(3407)` 从这 (K\) 个 genes 有放回抽 (K\) 次；每抽到一个 gene，把所有包含该 gene 的 validation tasks 各加入一次重采样多重集。同一 pair 可以重复进入。三个种子使用相同的 gene draws。执行 10,000 次，CI 与 9,500-valid 规则同 task bootstrap。

这是共享组分依赖的敏感性分析，不改变 authorization gate。

## 9. leave-one-gene-out（LOGO）

对每个 component gene，删除所有含该 gene 的 validation pairs，在其余任务上重算三个种子的主要 rho。每行输出 removed gene、removed task count、remaining task count、rho、estimability 和 failure code；每个种子汇总有效 rho 的 min/median/max、正号比例与有效 gene 数。不得删除不利 gene，也不得据此修改 24 个任务。

## 10. 发布与结论边界

runner 在同一文件系统的 staging 目录完成所有表、gate、报告、status 和 SHA256 manifest，拒绝 symlink 与未知路径，再原子 rename 为 `release/`。`release/` 已存在时拒绝覆盖。

`release/E163_E164_INTERFACE.json` 使用 schema `safeconf_e163_to_e164_v1`，同时写入 `validation_gate_passed`、`authorize_future_test_label_lock`、decision、test-access flags，以及 gate、status、task metrics、associations 和三个输入/访问 manifest 的 release 相对路径与 SHA256。后续实验只能按该 interface 的布尔门分支，不能根据报告措辞自行解释授权。

允许结论只有三类：

- `stop_raw_score_path`：authorization gate 失败；
- `allow_new_test_label_only_preregistration`：gate 通过，可另行冻结后续合同；
- `runtime_or_integrity_failure`：输入、hash、schema 或运行失败，不能作科学判断。

E163 永远不能写成 test validation、external validation、confirmatory success 或 publication-level evidence。
