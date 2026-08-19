# E164｜Wessels test-label-only 锁定与 E165 一次性评价合同

冻结日期：2026-07-15。E164 是 test truth 解封前的最后一道锁，采用两条彼此独立的路线。E163 的 validation-only futility diagnostic 只决定 PRESCRIBE 路线是否获准；E162b 已在更早阶段独立预锁的 Wessels 基线与 E164 train-only Systema reference，在 E163 任一完整终态发布后都必须冻结，不能因 PRESCRIBE 失败而作废。只有 `validation_gate_passed=true` 时，E164 才向 E162 `attempt_002` 的三个不可变 checkpoint 提交 E160 冻结的48个 canonical condition 字符串和 E161 train-control mean。E164 不读取 Wessels raw H5AD、test cell count、test `X`、test truth/effect/error/DE，也不训练或修改模型。

E162 的正式结论保持为 `failed_main_validation_nondegeneracy_gate_no_test_label_query`：三个 seed 的 validation `raw_log_prob` 非退化，但 PCA10 prediction 均只有一个向量。E164 不能把这个失败改写为通过。E164 的 test-label query 是在 validation 结果已知、test truth 仍封存时新冻结的 pretruth 评价阶段；PRESCRIBE 结果必须附带这一时序说明。

## 1. 硬门与固定输入

正式运行前必须逐项通过：

1. 本 runner 与合同已提交，工作文件逐字节等于当前 Git HEAD blob；
2. E163 固定入口：
   - `docs/实验结果/E163_wessels_validation_raw_futility_20260715/release/RUN_STATUS.json` 的 phase 恰为 `complete_validation_only_futility_diagnostic_no_test_label_or_X_access`；
   - `E163_E164_INTERFACE.json` 的 schema 为 `safeconf_e163_to_e164_v1`，并提供冻结布尔值 `validation_gate_passed`；
   - E163 明确 `test_label_queries_started=false`、`test_X_accessed=false`、`test_truth_accessed=false`；
   - E163 status、interface、`RESULTS_SHA256.csv` 及其全部 manifest payload 哈希通过；
3. E162 `attempt_002/RUN_STATUS.json` 的失败 phase、三个 validation gate、`test_label_queries_started=false`、`test_X/test_truth=false` 保持不变；三个 seed 的 locked checkpoint SHA256 与 seed status 完全一致；
4. E162b release phase 恰为 `complete_pretest_label_only_baselines_no_val_or_test_X`，`RESULTS_SHA256.csv`、status、interface及全部 artifact hash 通过；
5. E160 split、E161 v2 interface/data manifest、PRESCRIBE commit/source/weights 与 E162 正式门逐字节通过；
6. 正式解释器固定为 `/home/yyf/.conda/envs/prescribe_env/bin/python`，GPU 使用 `CUDA_VISIBLE_DEVICES=<physical index>` 后的内部 `cuda:0`。

`preflight` 不打开 development H5AD/graph/checkpoint，不创建输出。`formal` 总是只读取 E161 H5AD 的11,779个 train `X` rows计算 Systema reference，validation `X`为0。只有 E163 通过时，才允许读取 development metadata/graph/checkpoint以重建 native module；test graph 数始终为0。

## 2. 双路线授权与不可逆 query event

E164 interface 固定写 `baseline_arm_authorized=true`，并原样保留 E163 的 `e163_validation_gate_passed` 与 `e163_authorize_future_test_label_lock`。这两个布尔值决定是否允许query；最终 `prescribe_arm_authorized` 还必须要求seed3407 `main_raw_gate_passed=true`：

- E163失败：完整冻结 E162b+Systema baseline arm，不创建 query event，不加载checkpoint，不forward；E165仍可按冻结合同进行 baseline-only truth unseal；
- E163通过：冻结同一 baseline arm，并额外执行以下不可逆 PRESCRIBE label-only query。

获准路线在第一次48-label forward 之前，必须先原子写入并 fsync：

```text
docs/实验结果/E164_wessels_pretruth_lock_20260715/TEST_LABEL_QUERY_EVENT.json
```

event 必须锁定 E163全部门、E162失败状态、三个 checkpoint、E162b manifest、E160 test-label顺序/hash、E161 gene/PCA/control轴、runner/合同Git blob、PRESCRIBE source hash和 GPU。event 写入后禁止删除、覆盖或重试另一套输入；进程中断只能人工审计，禁止无记录重放 forward。E163失败路线不得创建占位event。

label-only graph allowlist 仅为 `x/pert/batch/ptr`：`x` 是 E161 train-control 2,023-gene mean，`pert` 是48个字符串。禁止 `y/y_pca/y_n/y_d/y_s/de_idx/cell_count/test_truth/test_expression/error`。

## 3. PRESCRIBE获准时的三 seed 输出与通过门

对 seed 3407、3408、3409 固定保存：

- `raw_log_prob`；
- epistemic、aleatoric 与 `official_combined=2*epistemic+aleatoric`；
- predicted PCA10、inverse-PCA post profile、相对 train control 的 predicted-effect RMS；
- label-only/no-truth字段审计。

主 seed 3407 的 E164 PRESCRIBE通过门只约束 `raw_log_prob`：48行、condition顺序完全一致、全部 finite、精确 unique≥24、sample std (`ddof=1`) >1e-6。prediction 非退化不再是 E164 通过条件，但必须如实保存 exact unique vector、逐坐标std和constant标记。3407 raw门失败时，E164 interface仍可发布 baseline arm，但必须写 `prescribe_arm_authorized=false`，不向E165暴露PRESCRIBE路径；已发生的event与部分PRESCRIBE审计输出永久保留。

三个 seed 的 official/magnitude/prediction 分别执行 estimability：finite、unique≥2且std>1e-12才可进入下游统计；否则写 `constant_or_nonfinite_baseline`，下游统计固定为NA，禁止jitter、去重误差或符号翻转。seed 3408/3409 raw门是敏感性，不阻断主seed通过。

## 4. train-only Systema reference

E164 必须从 E161 train `X` 独立计算 condition-balanced perturbed centroid：

\[
O_{pert}^{train}=\frac1{71}\sum_{c\in train,c\ne ctrl}\mu_c.
\]

每个非control condition先按细胞等权求均值，再对71个condition等权。它既是 Systema reference，也作为新增的 `condition_balanced_perturbed_mean` 描述性预测器。必须同时读取并审计 E162b 的 `cell_weighted_perturbed_mean`；两者名称、公式和向量不得混用。Systema将 train perturbation centroids 的均值作为参考，以弱化 systematic variation（DOI: [10.1038/s41587-025-02777-8](https://doi.org/10.1038/s41587-025-02777-8)）。

## 5. 冻结给 E165 的 truth 构造

E165 只能在 E164 release 完整且至少一个路线获准后写永久 unseal event，再首次打开 Wessels raw。baseline-only unseal要求 `baseline_arm_authorized=true`；PRESCRIBE端点还必须要求 `prescribe_arm_authorized=true` 与 `main_raw_gate_passed=true`。固定规则：

- raw身份、E160 row/condition、20,631 endogenous axis和E161 selected axis全部复核；
- 只读取9,902个test rows的前20,631列；8个construct和413个guide/barcode列访问为0；excluded/validation rows访问为0；
- 每细胞按20,631内源基因library执行 `log1p(10000*x/library)`，再取2,023 selected genes；
- 每个test condition的全部细胞等权求post mean；control固定为E161 train-control mean；
- PCA10只使用E161 mean/components；raw truth和PCA10 projected truth必须同时发布。

### split-half实验复现参考

每个test task按以下固定散列将cell IDs分半：

```text
SHA256("E165|Wessels|split-half|3407\t" + condition + "\t" + cell_id)
```

按digest、cell ID稳定排序，交替分配A/B；奇数时A多1个。`n<4`时该任务split-half记NA，不改规则、不借其他任务细胞。A/B分别独立归一化后求均值并计算与正式端点相同的PCA10/raw Pearson、RMSE，以及PCA10-truth-top20和raw-truth-top20指标。它只叫 **split-half experimental reproducibility benchmark/reference**，绝不能称理论上界或upper bound；半样本本身会降低复现性。依据 TxPert 的澄清与 seen-single/unseen-double additive baseline 语境（DOI: [10.1038/s41587-026-03113-4](https://doi.org/10.1038/s41587-026-03113-4)）。同时记录2026 SBB评价工作（DOI: [10.64898/2026.04.20.719650](https://doi.org/10.64898/2026.04.20.719650)）作为该参考的设计来源之一。

## 6. E165 predictor层级与端点

固定 predictor 顺序：

1. `control_no_change`；
2. E162b `cell_weighted_perturbed_mean`；
3. E164 `condition_balanced_perturbed_mean`；
4. E162b `matching_single_mean`；
5. E162b `single_additive`；
6. PRESCRIBE seed3407；3408/3409仅seed敏感性（仅在PRESCRIBE路线获准时存在）。

### 6.1 PRESCRIBE-own family（α=0.025）

仅在PRESCRIBE路线获准时沿用E160主端点：

\[
\rho_P=Spearman(raw\_log\_prob_{3407},\ PCA10\ Pearson\ effect\ accuracy_{PRESCRIBE}).
\]

预期正向，确认要求 task bootstrap 与 component-gene cluster bootstrap 的95% CI下界都>0。必须同时计算同一score对 **raw full-selected-gene Pearson effect accuracy** 的敏感性；无论方向是否一致都完整报告。PCA10 prediction若恒定也照常计算每任务accuracy，但必须在标题、表和讨论中声明模型坍缩以及E162 gate失败，不能称E162原确认性成功。

### 6.2 E162b baseline family（α=0.025，固定顺序）

H1：

\[
\Delta_{RMSE10}=mean(RMSE10_{cellweighted}-RMSE10_{matching})>0.
\]

H2 仅在H1通过后：

\[
\rho_{SE}=Spearman(matching\_se\_pca10\_confidence,-RMSE10_{matching})>0.
\]

正值固定表示matching优于cell-weighted mean。两者都要求 task 与 gene-cluster 95% CI 下界同时大于0。H1失败时H2只作描述。PRESCRIBE family与baseline family各分配α=0.025；baseline内部固定顺序，整体FWER≤0.05。

### 6.3 强制次要/描述端点

每task、每predictor必须报告：PCA10 projected与raw 2,023-gene RMSE；control-reference Pearson-Δ/cosine-Δ；train-Systema-reference Pearson-Δ/cosine-Δ；Systema Euclidean centroid accuracy；direction fraction。Pearson/cosine在零方差或弱信号不可估计时为NA，不得填0。RMSE不随reference改变，是区分matching与2×matching additive幅度的关键指标。

top20固定为两套truth-only集合：`PCA10_truth_top20`按|PCA10 projected truth effect|取前20，`raw_truth_top20`按|raw truth effect|取前20；两者精确tie均按E161 gene index。分别报告RMSE20、Pearson-Δ20、direction20，并明确它们在解封test truth后选择、test-dependent、signal-sensitive，只作次要描述，不进入确认门。禁止另加prediction-union集合。scPerturBench兼容部分限于MSE/RMSE、PCC-delta等centroid指标；当前预测器没有单细胞分布，E-distance、Wasserstein、KL、Common-DEGs必须NA（DOI: [10.1038/s41592-025-02980-0](https://doi.org/10.1038/s41592-025-02980-0)）。

## 7. risk、coverage、bootstrap与多重性

所有risk表统一 `higher=expected accuracy`。固定评估 E162b PCA10-SE（H2）、gene-SE、min single-cell count、min pair degree、negative matching magnitude、hash-random、constant、exact-pair support；以及PRESCRIBE三seed raw score、seed3407 official与negative predicted RMS。常数列ρ/CI/coverage比较为NA。

每个可估计score报告 Spearman、50%到100%（步长5%，保留数=`ceil(coverage*48)`）selective risk、AURC、最低置信四分位错误富集和相对magnitude的paired Δρ；排序为score降序、condition升序tie-break。

固定两套10,000次paired bootstrap，NumPy `default_rng(3407)`、quantile linear、最低有效9,500：

1. task bootstrap：48 tasks有放回抽48；
2. component-gene cluster bootstrap：从K个component genes有放回抽K次；每抽中一个gene就按canonical condition顺序追加所有含该gene的test tasks一次。

所有方法/score共用同一批indices。另做完整leave-one-component-gene-out，报告removed/remaining n、效应/rho及min/median/max/正向比例。主family按第6节控制；其余prediction contrasts与risk contrasts分别为两个secondary family，各自Holm校正。未校正项目只能称描述性。

## 8. 固定输出与访问账本

E164成功release总是包含：E162b预锁baseline profiles/risk的不可变副本、Systema centroid、E165完整spec、source/runtime/access、report、status、interface、manifest。PRESCRIBE获准时再包含三seed label-only表与inverse-PCA profiles、raw gate、estimability/degeneracy；未获准时这些路径必须不存在。interface schema固定为`safeconf_e164_to_e165_v1`，至少提供：`baseline_arm_authorized=true`、`prescribe_arm_authorized=<bool>`、五项`baseline_order`、`test_label_order`、`paths.baseline_post_profiles`、`paths.risk_wide`、可选`paths.prescribe_scores`三seed字典，以及按release相对路径索引的`artifact_sha256`。repo输出先写 `.release.staging`，allowlist/hash/fsync通过后一次rename为`release/`；已有event/release/failed attempt时拒绝重放。

最终访问事实必须是：raw opened=false；test label strings=48（只作为冻结基线任务轴；PRESCRIBE失败路线forward数为0）；test/validation/excluded X rows=0；test truth/effect/error/DE=false；train X rows=11,779（仅Systema reference）；test graph=0。E164不得产生任何E165 endpoint数值。
