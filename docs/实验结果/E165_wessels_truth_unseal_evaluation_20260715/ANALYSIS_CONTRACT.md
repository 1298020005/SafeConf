# E165｜Wessels 一次性 test truth 解封与冻结评价合同

初稿日期为2026-07-15，最终冻结日期为2026-07-16，均早于 test truth 解封。E165 是 Wessels `seen-single / unseen-double` 路径唯一允许读取 test expression 的阶段。只有 E164 的解封前 release 已成功、E160–E164 全部接口与 Git 字节门通过，且 E164 `baseline_arm_authorized=true` 时才能执行。PRESCRIBE arm 与 baseline arm 独立：E163/PRESCRIBE raw-score arm 失败不能浪费已经预注册的 E162b/E164 baseline predictions；只有 E164 同时给出 `prescribe_arm_authorized=true` 和完整 locked score paths时才评价 native family。E165 不训练、不调参、不重新生成预测，不根据 test 结果改变分数方向、任务集合、基线、指标或结论门。

相关方法学背景固定为：Systema（DOI `10.1038/s41587-025-02777-8`）、SBB principles（DOI `10.64898/2026.04.20.719650`）和 TxPert（DOI `10.1038/s41587-026-03113-4`）。引用只解释评价选择，不构成结果先验。TxPert 的 split-half experimental reproducibility 在减少样本数后**不是性能上界**；本合同统一称为“split-half reproducibility benchmark/reference”。

## 1. 运行前硬门

`preflight` 只读取 Git、JSON、CSV、文本、NPZ、opaque 文件哈希与 raw 文件 `stat`，不得用 h5py/anndata 打开 Wessels raw H5AD。它必须逐字节验证以下已提交资产：

1. E160 freeze：固定 72 train、24 validation、48 test conditions，48 test 均为两个 component singleton 已见、组合未见；raw identity 与 9,902 test cells；
2. E161 release：20,631 endogenous、8 engineered construct、413 guide/barcode；2,023 selected genes；train-only PCA10/control prior；test X 未访问；
3. E162 attempt 002 失败记录：三个种子 prediction collapse，test label query/test X/test truth 均未开始；
4. E162b release：四个预注册预测器、风险分数、train-only 统计量，无 validation/test X；
5. E163 release：validation-only raw-score futility诊断、`safeconf_e163_authorization_gate_v1` 与原样保留的通过或失败决定；E163 failure只关闭PRESCRIBE raw arm；
6. E164 release：`safeconf_e164_to_e165_v1`，五个冻结 profile predictors、全部相对路径和 SHA256、`baseline_arm_authorized=true`、raw/test truth 未访问；如 `prescribe_arm_authorized=true`，还必须有三种子 PRESCRIBE label-only 表，否则这些 paths 必须为空并给出关闭原因；
7. E165 runner 与本合同和当前 Git `HEAD` blob 完全一致。

任一输入未提交、hash 不符、schema/phase/访问边界不符时停止。E165 不允许用“最接近的文件”或目录扫描猜输入。

## 2. 不可逆解封事件

formal 在任何 raw H5AD 语义访问、opaque bytes hash 或 h5py/anndata open 之前，必须先在同目录以 `xb + flush + fsync` 写临时文件，再用 `link(2)` no-replace 原子创建目标、删除临时名并 fsync parent directory。该方式不允许并发进程覆盖已存在的事件：

`E165_wessels_truth_unseal_evaluation_20260715/TEST_TRUTH_UNSEAL_EVENT.json`

事件锁定 Git HEAD、runner/contract SHA256、E160–E164 gate fingerprint、raw stat identity、48 test condition order、预期 9,902 rows、20,631/421 feature boundary、2,023 selected axis、PCA/control hashes和 transaction id。文件一旦存在不得删除、覆盖或改写；恢复只允许 event 内容与当前 gate 完全相同。事件写成后，即使后续运行失败，也必须永久记录“test truth 已解封”。

## 3. 唯一允许的 raw expression 访问

raw 固定为 `/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/WesselsSatija2023.h5ad`。event 后先核对完整 SHA256，再以 read-only backed 模式读取 metadata，依据 E160 `raw_condition → canonical_condition` 映射得到 split。

- test rows 必须恰好 9,902，排序唯一，condition 集合和顺序与 E160/E164 48 条完全一致；
- `X` 只允许一次索引 `raw.X[test_rows, :20631]`；
- 禁止索引 train、validation、excluded rows；
- 禁止索引 8 engineered construct 与 413 guide/barcode，共 421 excluded columns；
- counts 必须为非负整数 CSR；每个 cell 的 20,631-endogenous library 必须有限且大于 0；
- 每 cell 固定变换：`log1p(10000 * count / endogenous_library)`，随后只取 E161 锁定的 2,023 selected indices；
- 不读取 raw `layers`、`obsm`、DE、上游 `ncounts` 或任何分布预测对象。

访问账本保存 raw hash、row/column index hashes、test/intersection 数量和转换行数。E165 禁止 energy distance、MMD、Wasserstein、单细胞分布拟合等 distribution metrics。

## 4. 冻结 truth 与预测器

对每个 test condition，truth post profile 是其 normalized selected-gene cells 的等权均值；raw effect 为 post truth 减 E161 train-control mean。PCA10 truth 由 E161 frozen mean/components 直接 transform 后按 task 求均值；PCA10 reconstructed effect 是 inverse transform 后减 train-control mean。发布包同时保存 raw post、raw effect、PCA10 reconstructed post、PCA10 reconstructed effect和10维坐标，避免只保存指标而无法复核真值变换。

E161 的逐细胞 PCA 坐标以 float32 保存后再求 control mean，从 float64 gene mean 重新投影不会逐位相同；解封前观测的最大绝对差为 `0.00021519727355714946`，固定一致性门为 `≤5e-4`。该门只核对数值存储误差，不使用 test expression。

E164 baseline arm 五个 profile predictors固定顺序：

1. `control_no_change`
2. `cell_weighted_perturbed_mean`
3. `condition_balanced_perturbed_mean`
4. `matching_single_mean`
5. `single_additive`

不得改名、重算或删除。若 `prescribe_arm_authorized=true`，三份 `PRESCRIBE_TEST_LABEL_ONLY_SEED{3407,3408,3409}.csv` 只按 interface 相对路径读取；3407 是主模型，3408/3409 是训练敏感性。PRESCRIBE `predicted_pca_0..9` 用 frozen PCA inverse transform 得到 selected-gene post/effect；`raw_log_prob` 越大固定为越可信。若 `prescribe_arm_authorized=false`，runner不得读取或猜测任何 PRESCRIBE test table，native family表保留结构化 `not_run_prescribe_arm_not_authorized` 状态。official combined confidence、constant magnitude或其他事后分数不替代 raw score。

## 5. task-level 预测指标

每个 predictor × 48 tasks 必须保存：

- PCA10 reconstructed truth：Pearson effect、cosine effect、direction accuracy、RMSE；
- raw selected-gene truth sensitivity：同四项；
- truth-only top20：分别按 `abs(PCA10 reconstructed truth effect)` 与 `abs(raw truth effect)` 取恰好 20 genes，输出同四项；
- Systema perturbed-reference：以 E164 `condition_balanced_perturbed_mean` profile 作为 condition-balanced perturbed centroid reference，输出 Pearson、cosine、direction；RMSE 保持 reference-insensitive，和 post-profile RMSE相同；
- Systema centroid accuracy：按官方实现，计算预测 post profile 到正确 test truth centroid 的 Euclidean distance严格小于其余47个truth centroid距离的比例，tie计失败。另保存更严格的nearest-centroid hit（正确中心严格近于全部47个竞争中心）；两列不得混名。语义锁定于Systema官方代码commit `aaf5b5353993b48b78543f2f93b3e18ca65df515`。

Pearson/cosine 在任一向量常数、零范数或非有限时为 `NA` 并记录 reason，不能填 0。direction accuracy 固定为逐基因严格同向 `(pred_effect * truth_effect) > 0` 的均值；任一侧为零都计 miss，与 E158/E160 口径一致。

## 6. baseline hierarchy、H1 与 H2

固定 hierarchy 是 control → cell-weighted perturbed mean → condition-balanced perturbed mean → matching → additive。全部均值、任务分布与相邻差异如实报告。

H1 为预注册的 paired improvement：

`cell_weighted_perturbed_mean PCA10 RMSE - matching_single_mean PCA10 RMSE`

正值表示 matching 更好。H1 只有 observed mean > 0，且 task bootstrap 与 component-gene cluster bootstrap 的 95% CI 下界都 > 0 才通过。

H2 为 `matching_se_pca10_confidence` 与 `-matching PCA10 RMSE` 的 Spearman 关联。H2 数值始终计算并保存；只有 H1 通过时才有 confirmatory 解释。H2 通过还需 rho > 0 且 task/gene-cluster 两个 95% CI 下界 > 0。H1 失败时 H2 标记 `descriptive_not_confirmatory_due_to_H1`，不删除结果。

## 7. PRESCRIBE raw-score 主分析

本节仅在 `prescribe_arm_authorized=true` 时执行。主终点固定为 seed3407 `Spearman(raw_log_prob, PCA10 Pearson effect accuracy)`，预期为正。并行强制报告 raw-score 对 PCA10 cosine、direction、`-RMSE`，以及 raw selected-gene truth 的四个对应端点；3408/3409 全部作为训练敏感性。arm关闭时不影响H1/H2、五基线、Systema、SBB和split-half结果发布。

- task bootstrap：`numpy.random.default_rng(3407)`，10,000 次，从 48 tasks 有放回抽48；
- component-gene cluster bootstrap：新的同 seed RNG，从 test component genes 有放回抽 K 个，每次把含该 gene 的任务加入多重集；同一 pair 可重复；10,000 次；
- CI：finite replicates 至少 9,500，`quantile([.025,.975], method="linear")`；否则 CI 为 NA；
- LOGO：逐 component gene 删除所有含该 gene 的 test pairs后重算，全部行保存。

不允许取反 raw score、挑 seed、删除不利 task/gene 或用 raw-truth结果改主要 truth。

## 8. 选择性预测

覆盖率固定 `0.50,0.55,...,1.00`，保留数 `ceil(coverage×48)`；confidence 从高到低，condition 原序只作 deterministic tie-break。每条可估计 score 输出：

- retained mean RMSE/accuracy；
- AURC（coverage 0.50–1.00 trapezoid，除以区间宽度）；
- high-error capture：全体最高 20% error tasks 中被低置信拒绝的比例；
- rejected-set enrichment：拒绝集 high-error rate / 全体 high-error rate。

常数或非有限 score 的全部排序统计为 NA，reason=`constant_or_nonfinite_score`；不得 jitter。native raw score只评价同 seed own prediction；E164 matching风险分数评价 matching predictor。

所有冻结matching风险分数均报告上述关联与区间；PRESCRIBE三seed raw score及主seed official/negative-magnitude同样保留。相对冻结magnitude score的paired delta-rho使用同一批bootstrap indices。次要prediction contrasts与risk associations分属两个family，各自Holm校正；这些扩展均不替代H1、H2或PRESCRIBE主终点。

## 9. split-half reproducibility reference 与 SBB诊断

对每个 condition，以 `SHA256("E165|Wessels|split-half|3407\t" + condition + "\t" + obs_name)` 排序，偶数 rank进 half A、奇数 rank进 half B。两半必须非空且并集/交集正确。分别计算 PCA10/raw task effects，并输出 Pearson、cosine、direction、RMSE和 top20 指标。

该表是确定性的 split-half reproducibility **benchmark/reference**，不是 upper bound：每半样本少于完整 test truth，不能据此断言模型理论上不可超过。SBB部分只做已冻结诊断：top20检验指标对信号的敏感性；split-half提供经验参照；五级baseline hierarchy提供性能地板。除H1/H2和PRESCRIBE主分析外均为 contextual/descriptive。

## 10. 原子发布与失败保留

所有输出先写同一文件系统 `.release.staging`，包含 truth profiles/PCA、task metrics、baseline hierarchy、H1/H2、native associations、全部 bootstrap replicates、LOGO、coverage/AURC、split-half、centroid accuracy、访问账本、报告和白底 SVG。gzip固定 `mtime=0`。allowlist、无symlink、shape、finite/NA reason、hash、fsync通过后单次 rename 为 `release/`；已有 release拒绝覆盖。

失败时保留不可逆 event、staging和 `failures/E165_FAILURE_*.json`。无论结果有利与否都进入报告；允许结论不超过：native prediction performance、raw-score routing evidence/失败、H1/H2结果、baseline/Systema/SBB描述和本 Wessels split 的边界。不能据此保证期刊录用，也不能把 validation-informed前史隐藏为纯确认性研究。

## 11. 固定命令

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e165_wessels_truth_unseal_evaluation.py --mode preflight

/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e165_wessels_truth_unseal_evaluation.py --mode formal
```
