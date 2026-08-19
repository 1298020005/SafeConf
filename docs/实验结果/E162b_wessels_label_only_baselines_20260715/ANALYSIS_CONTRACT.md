# E162b｜Wessels 解封前 label-only 简单预测器与风险基线合同

冻结日期：2026-07-15。E162b 必须在 E163 首次读取 Wessels test expression 之前完成并提交。它只使用 E161 development H5AD 中的 **train rows** 和 E160 冻结的 48 个 test condition 字符串，生成可直接与 PRESCRIBE 比较的简单预测器及风险基线。validation/test expression、test cell count、test truth、test effect、test error、DE 结果均不可读取或推导。

E162b 是独立的预注册基线，不依赖 E162 模型输出，也不根据 E162 validation gate 或模型表现改变公式。PRESCRIBE 的 predicted-effect RMS 由后续接口合并；本 runner 不打开 E162 checkpoint 或预测文件。

## 1. 固定入口、Git 门与运行方式

正式输入固定为：

- E160 `freeze/manifests/E160_set2conditions.json`：只取 48 个 test canonical condition 字符串；
- E161 `safeconf_e161_to_e162_v2` 接口与 data asset manifest；
- E161 `perturb_processed.h5ad`：HDF5 read-only 打开，只允许索引连续 CSR 前缀中的 11,779 个 train `X` rows；
- E161 `SELECTED_GENE_AXIS.txt`、`TRAIN_ONLY_PCA_MODEL.npz` 和 `TRAIN_ONLY_CONTROL_PRIOR.npz`。

禁止打开 Wessels raw H5AD，禁止调用 E161 graph cache，禁止读取 development H5AD 的 validation `X/layers/obsm`。正式 runner 只用 h5py 读取 obs/var metadata，并在确认 train rows 恰为 0–11,778 的连续前缀后，仅读取该 CSR prefix 的 `indptr/data/indices`。PCA 坐标必须由 train `X` 和冻结 PCA mean/components 重新计算，不能读取包含 validation 坐标的 `obsm[X_pca]`。

正式解释器固定为 `/home/yyf/.conda/envs/prescribe_env/bin/python`（Python 3.9.25；NumPy 1.26.4、Pandas 2.3.3、anndata 0.10.8、SciPy 1.13.1）。`preflight` 与 `formal` 都必须验证本合同和 runner 已提交且工作文件逐字节等于当前 Git HEAD blob；同时核对 E160/E161 状态、接口、manifest 和实际所需资产哈希。`preflight` 只能读取 Git、JSON、CSV、文本、NPZ header/arrays和 opaque 文件字节流哈希；不得通过 h5py/anndata 解析 development H5AD。

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e162b_wessels_label_only_baselines.py --mode preflight
/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e162b_wessels_label_only_baselines.py --mode formal
```

## 2. 固定 train 统计量

所有向量均位于 E161 的 2,023-selected-gene normalized expression space。令：

- \(\mu_0\)：424 个 train control cells 的逐基因均值；
- \(\mu_P\)：所有非-control train cells 的逐基因均值，按细胞等权，因此自然按每个 condition 的细胞数加权；
- \(\mu_g\)：train singleton `g+ctrl` 的逐基因均值；
- \(d_g=\mu_g-\mu_0\)；
- \(n_0,n_g\)：对应 cell 数；\(s_0^2,s_g^2\)：逐坐标、`ddof=1` 的样本方差。

control、全部非-control 和 27 个 singleton 的 gene-space 与 PCA10 mean/variance 必须发布。E161 control prior 中的 `control_gene_mean` 必须与重算 \(\mu_0\) 在浮点容差内相同。所有 48 个 test pairs 的两个 component genes 都必须有 train singleton；缺一即整体失败，不能回退到全局均值。

## 3. 四个冻结预测器

对 test pair \(g+h\)，固定输出 post profile 与相对 \(\mu_0\) 的 effect：

| baseline | post profile | effect |
|---|---|---|
| `control` | \(\mu_0\) | \(0\) |
| `cell_weighted_perturbed_mean` | \(\mu_P\) | \(\mu_P-\mu_0\) |
| `matching_single_mean` | \((\mu_g+\mu_h)/2\) | \((d_g+d_h)/2\) |
| `single_additive` | \(\mu_0+d_g+d_h\) | \(d_g+d_h\) |

不裁剪负值，不按 validation/test 调参。必须逐元素审计：control effect 精确为 0；`single_additive` effect 精确为 `2 × matching_single_mean effect`；matching/additive 在 48 个任务中各至少有 24 个不同预测向量。

每个 post profile 的 PCA10 坐标固定为

\[
z=(\text{post}-m_{E161})W_{E161}^{T},
\]

其中 \(m_{E161}\) 和 \(W_{E161}\) 来自冻结 `TRAIN_ONLY_PCA_MODEL.npz`。effect PCA10 坐标定义为 post PCA10 减 control PCA10，只作透明审计。

## 4. 解封前风险/置信基线

所有用于选择性预测的 `analysis_score` 统一为 **越大表示预期越可靠**；原始量及原始方向同时保存，禁止 E163 解封后翻转符号。

对 component gene \(g\)：\(k_g\) 是 44 个 train pair conditions 中包含 \(g\) 的不同 pair 数。test exact-pair support 固定为 0，因为 E160 train/test pair 不重叠。

| score | 冻结公式（higher = expected accuracy） |
|---|---|
| `min_single_cell_count_confidence` | \(\log(1+\min(n_g,n_h))\) |
| `min_train_pair_degree_confidence` | \(\log(1+\min(k_g,k_h))\) |
| `matching_se_pca10_confidence`（主 SE） | \(-\sqrt{\operatorname{mean}_j[s_{gj}^2/(4n_g)+s_{hj}^2/(4n_h)+s_{0j}^2/n_0]}\)；PCA10 coordinates |
| `matching_se_gene_confidence`（敏感性） | 同式；2,023 selected genes |
| `matching_magnitude_confidence` | \(-\operatorname{RMS}[(d_g+d_h)/2]\) |
| `hash_random_confidence` | `uint64BE(SHA256(salt + condition)[:8]) / 2^64` |
| `constant_confidence` | 0 |
| `exact_pair_support_confidence` | 0 |

random salt 精确为 `E162b|Wessels|random-confidence|3407\t<condition>`。同时保存正值 `matching_effect_rms`，其 `raw_orientation=higher_effect_size`；分析列取负值并固定为 `analysis_orientation=higher_expected_accuracy`。additive effect RMS 只用于验证它等于 matching effect RMS 的 2 倍，不作为新的主风险基线。常数列保留并明确标记 `constant_non_estimable`，不得加 jitter。

## 5. 固定输出与原子发布

正式运行先写同一文件系统的 `.release.staging`，完成形状、有限性、公式、哈希、allowlist、无 symlink 和 fsync 审计后，单次 directory rename 发布为 `release/`。已有 `release/` 时拒绝覆盖；失败 staging 保留供审计。

发布 allowlist 固定为：

```text
.E162b_TRANSACTION.json
RUN_STATUS.json
README_先看这个.md
RESULTS_SHA256.csv
E162b_E163_INTERFACE.json
reports/E162b_REPORT.md
profiles/E162b_TEST_POST_PROFILES.csv.gz
profiles/E162b_TEST_EFFECT_PROFILES.csv.gz
profiles/E162b_TEST_PCA10_COORDINATES.csv
tables/E162b_TRAIN_REFERENCE_GENE_STATS.csv.gz
tables/E162b_TRAIN_REFERENCE_PCA10_STATS.csv
tables/E162b_TEST_TASKS.csv
tables/E162b_RISK_BASELINES_WIDE.csv
tables/E162b_RISK_BASELINES_LONG.csv
tables/E162b_BASELINE_AUDIT.csv
tables/E162b_X_ACCESS_LEDGER.csv
tables/E162b_SOURCE_HASHES.csv
tables/E162b_RUNTIME_ENVIRONMENT.csv
```

gzip 文件必须使用 `mtime=0` 生成。profile 行顺序固定为四个 baseline 的上述顺序，再按 E160 test condition 原顺序；gene columns 固定为 E161 selected axis。发布 4×48×2,023 post profiles、同形状 effects、4×48×10 post/effect PCA coordinates、train statistics、任务表、wide/long 风险表、访问账本和完整结果哈希。`RESULTS_SHA256.csv` 哈希所有 scientific/operational payload（包含接口和 transaction sentinel），仅排除其自身与为避免循环依赖而后写的 `RUN_STATUS.json`；status 反向锁定 manifest hash 及 manifest 中全部条目。

## 6. E163 接口与访问账本

`safeconf_e162b_to_e163_v1` 必须锁定 runner/合同 Git blob、E160 split hash、E161 interface/asset hashes、test label order、gene order、四类预测器、风险公式、所有发布文件哈希及以下事实：

- Wessels raw file opened = false；
- raw/test/validation/excluded `X` rows indexed/materialized/transformed = 0；
- train `X` rows indexed/materialized/transformed = 11,779；
- test inputs = 48 canonical condition strings only；
- test cell count/truth/effect/error/DE used = false；
- PRESCRIBE magnitude not computed here；后续只允许合并 E162 的 raw predicted-effect RMS，并以负 RMS 作为同方向 confidence。

E163 解封后只能评价上述冻结 profile/risk columns，不能重算 train statistics、改变 matching/additive 公式、翻转方向、替换 random salt 或删除失败/常数基线。

## 7. 失败规则

任一 Git、source hash、E161 no-test invariant、train membership、singleton coverage、PCA/gene axis、有限性、公式等价性、unique-vector、访问账本或原子发布断言失败，E162b 必须终止并保留 failure record/staging。禁止读取 validation/test expression 诊断失败，禁止用 E162/E163 结果修补基线，禁止静默跳过 test task。
