# E161｜Wessels train/validation 严格预处理与 development graph 合同

冻结日期：2026-07-14；两次解封前勘误：2026-07-15。E161 只允许读取 E160 冻结的 train/validation expression，完成 train-only feature fitting、PCA10、control prior、E-distance 和 development-only PRESCRIBE graph。E161 不训练模型，不生成 test prediction，不读取、索引、转换或物化任何 test/excluded `X` row。

首次 metadata-only preflight 发现 21,052 列不是纯基因轴。随后 E161 首次正式运行只读取训练表达，在 `obs[ncounts]` 等式门停止；E161a 冻结诊断证明 20,631–20,639 的任何连续前缀都不能逐细胞复现该字段。上游 scPerturb 代码表明，`ncounts` 是从原始 Seurat metadata 的 `nCount_RNA` 改名保留，不是根据最终 H5AD `X` 重算。最终规则锁定为：前 20,631 列内源基因可用于 train/validation；随后 8 个实验构造与 413 个 guide/barcode 列全部排除。`obs[ncounts]` 只作非绑定差值审计。上述决定发生在 validation 表达读取及任何 test/excluded 表达读取之前，不改变 E160 条件切分。

## 1. 固定上游和 Git 门

正式执行前，下列文件必须已提交且与当前 Git `HEAD` blob 逐字节一致：

- `tools/scripts/run_e161_wessels_trainval_preprocess.py`；
- 本合同；
- `PREFLIGHT_FAILURE_AND_AMENDMENT_20260715.md`；
- `ENDOGENOUS_AXIS_AMENDMENT_20260715.md`；
- E161a 合同、访问账本、候选边界结果、状态与结果 manifest；
- E160 合同、scGPT 纯文本 vocabulary 与 `freeze/` 中全部文件。

E160 必须为 `requirements_frozen_test_expression_unopened`，所有 artifact hash 必须与 `freeze/RUN_STATUS.json` 相符，任务数必须为 train 72、val 24、test 48。PRESCRIBE commit、实际导入的 Python 源码、`gene2go_all.pkl` 和 scGPT `embedding.pkl` 必须哈希锁定。用于追溯 `ncounts` 的 scPerturb commit、Wessels notebook 和 QC utility 也必须哈希锁定。

固定运行环境为 `/home/yyf/.conda/envs/prescribe_env/bin/python`（Python 3.9.25），runner 必须在任何元数据或 expression 读取前逐项核对 anndata 0.10.8、Scanpy 1.10.3、NumPy 1.26.4、Pandas 2.3.3、SciPy 1.13.1、scikit-learn 1.6.1、scikit-misc 0.3.1、h5py 3.14.0、PyTorch 2.1.2+cu118 和 PyG 2.6.1。正式命令为：

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e161_wessels_trainval_preprocess.py --mode preflight
/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e161_wessels_trainval_preprocess.py --mode formal
```

PRESCRIBE 的 `gears/__init__.py`、`gears/pertdata.py`、`gears/utils.py`、`gears/data_utils.py`、数据 adapter/dataloader 及两个非代码资产全部单独哈希锁定；发布前必须再次核对 Git HEAD、本 runner/合同、E160、上述依赖和 raw 身份。

`preflight` 只可读取 Git、E160 文件、raw 路径身份和 AnnData `obs/var_names/shape`；不得访问 `X`。只有 `formal` 分支可进入语义 expression 读取。

## 2. raw 身份与固定切分

raw 必须是普通文件，不得是 symlink。正式 expression 读取前同时计算 MD5/SHA256，并与 E160 锁定的以下身份相符：

- 30,707 cells × 21,052 raw features，其中 20,631 个 endogenous genes、8 个 engineered constructs、413 个 guide/barcode features；
- 219,393,529 bytes；
- MD5 `6897bfdcda928a678208fecf4eeb282e`；
- SHA256 由 E160 `raw_integrity.sha256` 提供；
- device/inode/size/mtime 在哈希前后及 train/val 读取后保持一致。

canonical 映射只从 E160 `E160_CONDITION_AUDIT.csv` 读取。固定规则为 `control→ctrl`、single `GENE→GENE+ctrl`、pair 基因大写后字典序用 `+` 连接。固定资源量：

| split | conditions | cells | E161 expression role |
|---|---:|---:|---|
| train | 72 | 11,779 | fit |
| validation | 24 | 5,102 | fixed transform only |
| test | 48 | 9,902 | sealed, zero access |
| excluded | — | 3,924 | zero access |

所有语义 `X` 读取必须经过单一 helper。它只接受 `train` 或 `val`，要求请求行与该 role 的冻结行逐项完全一致，并在索引前确认与其他 role 无交集。

## 3. 内源基因 library-size 归一化

Wessels `X` 是 CSR integer raw counts。对每个允许访问的细胞，先在锁定的 20,631-endogenous-gene raw axis 上直接计算 library size，然后才取 selected genes：

\[
L(x_{ig})=\log\left(1+10{,}000\frac{x_{ig}}{\sum_{h=1}^{20631}x_{ih}}\right).
\]

禁止先切 selected genes 再调用 `normalize_total/normalize_per_cell`；8 个实验构造和 413 个标签编码变量不得纳入分母。归一化审计必须保存 endogenous library-size 范围、selected/endogenous count fraction、公式重算最大误差，以及 `endogenous_sum - obs[ncounts]` 的最小值、最大值、总和和不一致细胞数。`obs[ncounts]` 不是通过门，差值不得用 validation/test 调参。

## 4. train-only gene axis

- `seurat_v3` 在全部 11,779 个 train cells 的 20,631-endogenous-gene raw counts 上选 top 2,000 HVGs；
- raw `var[ncounts/ncells]` 不可用于筛选，因为它们可能含全数据汇总信息；
- 从 27 个 train single conditions 得到强制 perturbation genes，不从 test truth 得到；
- selected axis 是 top-2,000 HVGs 与 27 个 perturbation genes 的并集；
- 顺序保持 raw Wessels var axis，正式数量应为 2,000–2,027；
- raw/selected gene-order 文本、index 和 SHA256 必须锁定。

## 5. PCA10、control prior 和 E-distance

PCA10 只对全部 train normalized selected-gene matrix 拟合：`randomized PCA`、seed 3407、不 whiten。validation 只做固定 transform。

必须同时保存：

- 424 个 train controls 在 selected-gene space 的 `control_gene_mean`，作为所有 graph `x` 和 effect reference；
- train-control PCA10 mean/covariance，作为 PRESCRIBE Normal-Wishart prior；
- PCA mean/components/explained variance 及 gene order。

E-distance 使用每个 train condition 的全部细胞，不按最小组15 cells平衡。对 PCA coordinate set \(Z\)：

\[
\sigma_Z=\frac{2}{n_Z-1}\left(\sum_i\lVert z_i\rVert^2-n_Z\lVert\bar z\rVert^2\right),
\]

\[
\delta_{C,Z}=\overline{\lVert c\rVert^2}+\overline{\lVert z\rVert^2}-2\bar c^T\bar z,
\qquad E_Z=2\delta_{C,Z}-\sigma_C-\sigma_Z.
\]

`y_d=δ`、`y_s=σ`、`y_n=E`。train 值全部 finite；validation 是 NaN sentinel。control 固定 `y_d=y_s=σ_control,y_n=0`。当前 native `ListMLELoss` 对排列不敏感，因此 E-distance 在 E161/E162 native arm 中是接口字段，不得声称已有效提供排序监督。

## 6. development AnnData 和 native graph

`perturb_processed.h5ad` 只含 train+validation。最小 schema：

- `obs`: `raw_perturbation`、canonical `perturbation/condition`、`condition_name`、`e161_split`、`cell_line`、`cell_type`、`nperts`、`Guide.Class`、raw row index、full-20631-endogenous library size；
- `var`: `gene_name`、raw index、train detection/count/HVG/forced-gene 字段；
- `layers[counts]`: selected-gene raw counts；
- `X`: 按全 20,631-endogenous library size 归一化的 selected matrix；
- `obsm[X_pca]`: train-fitted PCA10；
- `uns`: PCA、HVG、normalization、`y_d/y_s/y_n`、compatibility callback 和 provenance。

Wessels 必须使用直接 `PertData.load(data_path=...) + prepare_split(split="custom")` adapter。禁止使用 `LoadData`，因为其非 Norman 分支会改成 simulation split。在加载前对 27 genes 执行内存中 `gene2go.setdefault(gene,set())`，防止 GO membership 静默丢任务。

graph 必须满足：

- keys 恰为 72 train + 24 validation conditions；
- 11,779 train graphs + 5,102 validation graphs = 16,881；
- test graph 数为 0；
- `x` 是 train-control gene mean，`y` 是开发细胞 selected expression，`y_pca` 是 10 维；
- train `y_d/y_s/y_n` finite，validation 为 NaN；
- `pert/pert_idx/de_idx`、shape、finite 和与 H5AD 的逐 graph 等价性穷举通过。

compatibility `rank_genes_groups_cov_all` 只是 train-HVG 顺序 placeholder，不是 condition-specific DE，不得用于生物学 top-DE endpoint。

## 7. 原子发布、资产和恢复

大资产先写入同文件系统 staging：

```text
/home/yyf/data/safeconf_e161_prescribe/.wessels_e160.staging
```

穷举审计、资产哈希、allowlist 和 fsync 通过后，directory rename 发布为：

```text
/home/yyf/data/safeconf_e161_prescribe/wessels_e160
```

仓库审计输出先写入 `E161.../.release.staging`，然后发布为 `release/`。两个目录共享持久化 `.E161_TRANSACTION.json`；状态顺序固定为 `building → ready_to_publish → asset_published → link_published → complete`。transaction sentinel 随 staging 目录一起 rename，作为永久运行身份保留，不得在 rename 前删除。任一发布步骤中断后，`--recover-staging` 只能在 transaction id、精确 allowlist、manifest/status 哈希和非 symlink 全部通过时 roll-forward；`building` 阶段只能严格删除带同一 sentinel 且文件为允许子集的 staging。已完整发布的 final/release 禁止删除或覆盖。

数据资产至少包含：

```text
perturb_processed.h5ad
set2conditions_3407.pkl
frozen_pert_gene_set_3407.pkl
data_pyg/cell_graphs.pkl
data_pyg/mean.npy
data_pyg/cov.npy
TRAIN_ONLY_PCA_MODEL.npz
TRAIN_ONLY_CONTROL_PRIOR.npz
FULL_RAW_FEATURE_AXIS.txt
ENDOGENOUS_GENE_AXIS.txt
ENGINEERED_CONSTRUCT_FEATURE_AXIS.txt
GUIDE_BARCODE_FEATURE_AXIS.txt
EXCLUDED_FEATURE_AXIS.txt
SELECTED_GENE_AXIS.txt
train_only_edistance_labels.csv
E161_E162_INTERFACE.json
ASSET_MANIFEST.csv
```

## 8. 访问账本和泄漏门

`E161_X_ACCESS_LEDGER.csv` 必须区分 opaque file hashing 和 semantic expression access，并固定：

- train semantic rows indexed/materialized/transformed = 11,779；
- validation = 5,102；
- test = 0；
- excluded = 0；
- engineered-construct columns indexed/materialized = 0；
- guide/barcode columns indexed/materialized = 0；
- test prediction/truth/effect/error = 0；
- model training = false。

最终递归确认 H5AD、graph keys、`uns[y_*]`、callback keys 均不含 test condition，dev obs name 与test obs name 交集为空，且数据目录不存在 test H5AD、sealed transform 或 test graph。

## 9. E162 接口

E161 输出 `safeconf_e161_to_e162_v2` 接口，分别锁定 20,631 / 8 / 413 / 421 的数量和轴哈希，以及 data root、全部资产哈希、split、selected gene order、PCA、control prior、graph counts 和无 test-expression 事实。E162 必须拒绝 v1 接口，不再打开 Wessels raw，只通过 custom PertData adapter 加载 development assets，并自建 train/validation-only DataModule；`test_dataloader()` 必须直接报错。

test label-only query 只允许：

- `x = frozen train-control selected-gene mean`；
- E160 canonical test perturbation string；
- locked scGPT embedding/checkpoint/PCA/gene order。

不得包含 `y`、`y_pca`、test cell count、test expression、DE 或 error。E163 完成一次性、不可逆解封之后，test truth 的固定变换只能先用前 20,631 个 endogenous counts 计算 library size，再取 E161 selected axis；8 个实验构造和 413 个 guide/barcode 列仍不进入预测真值或归一化。E163 解封前，E161/E162 对 test expression 的访问仍为零。

## 10. 失败规则

任一身份、split、normalization、PCA、prior、graph、哈希、原子发布或泄漏断言失败，E161 都必须终止并保留 failure record/staging。不得用 validation/test 重选基因、PCA、条件、细胞或归一化方式；不得静默丢任务；不得覆盖已发布资产。
