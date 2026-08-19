# E162｜Wessels PRESCRIBE 训练、validation 非退化门与 label-only 测试锁定合同

冻结日期：2026-07-14；接口与执行勘误：2026-07-15。E162 只使用 E161 已发布的 train/validation development 资产训练 PRESCRIBE，并用不含真值的扰动字符串查询锁定测试分数。本阶段不打开 Wessels raw h5ad，不读取 test `X`/`y`/`y_pca`/细胞数/DE/误差，不计算任何测试端点。

## 1. 固定输入与 Git 门

正式运行前，E162 runner、本合同、E160 freeze 与 E161 release 必须已提交，且工作树字节与 `HEAD` blob 一致。固定上游：

- E160：72 train、24 validation、48 test 条件字符串；
- E161 `safeconf_e161_to_e162_v2`：20,631 endogenous genes、8 excluded engineered constructs、413 excluded guide/barcode features；11,779 train graphs、5,102 validation graphs、0 test graph；
- E161 selected-gene 顺序、PCA10、train-control gene mean/PCA prior、split pickle、graph cache及发布清单中的逐文件哈希；
- PRESCRIBE upstream commit `6f7264a205aaff654a9594863c5c10b656f88ebe`、实际导入源码、scGPT perturbation embedding 与 `gene2go_all.pkl`。

E162 的显式 data root 固定为 `/home/yyf/data/safeconf_e161_prescribe/wessels_e160`。native prior 和 Lightning module 会按上游实现读取相对路径 `PRESCRIBE/data/wessels_e160`；正式运行前必须证明该路径是 symlink 且精确解析到上述 data root。代码不含 raw Wessels 路径，也不接受相关 CLI 参数，不允许通过 `LoadData` 重新分区。

`preflight` 核对 Git `HEAD`、JSON/CSV、E160 JSON 与 E161 split pickle 的逐项顺序一致性，并校验小型资产；它不打开或计算 development H5AD、graph pickle、checkpoint 的内容哈希，也不打开任何 raw H5AD。`formal` 在加载 development 资产前按 E161 manifest 完整计算 H5AD/graph 在内的全部资产哈希。Git `HEAD` 也进入 attempt fingerprint。

## 2. development-only adapter 与 DataModule

adapter 顺序固定为：

1. `PertData(PRESCRIBE/data, gene_set_path=E161/frozen_pert_gene_set_3407.pkl, default_pert_graph=False)`；
2. 对 27 个已冻结扰动基因执行 `gene2go.setdefault(gene,set())`；
3. `load(data_path=E161_DATA_ROOT)`；
4. `prepare_split(split="custom", seed=3407, split_dict_path=...)`；
5. `Get_Graph(overlap=True)`。

禁止调用 PRESCRIBE `LoadData`。`Get_Graph` 返回的 edge tensors 必须为 `None`，`nodes_num` 必须等于 E161 selected-gene 数，`num_pert=27`，扰动名及顺序必须等于冻结的 27 个基因，reindex keys 必须精确为 `0..26` 加 control sentinel `-1`，graph keys 只能是 train∪validation。接口给出的所有相对路径必须留在固定 data root 内，拒绝绝对路径、`..`、越界解析与叶节点 symlink。

DataModule 仅含 train/validation：

- train：11,779 graphs，batch size 512，shuffle，`drop_last=True`，每 epoch 23 batches；
- validation：5,102 graphs，batch size 512，`shuffle=False`、`drop_last=False`，每 epoch 10 batches；
- gradient accumulation = 1；
- `test_dataloader()` 必须立即抛错。

训练前穷举核对 graph key、condition、graph 数、`x/y/y_pca/y_n/y_d/y_s/pert/pert_idx/de_idx`、`x` 的 train-control mean、train 的 finite `y_n/y_d/y_s`、validation 的 NaN sentinel 与 test graph=0。

## 3. 三种子固定训练

种子用途不可互换：

| seed | 角色 |
|---:|---|
| 3407 | 唯一主分析模型 |
| 3408 | 训练随机性敏感性 |
| 3409 | 训练随机性敏感性 |

三者沿用 E157 的模型、损失、先验和标量优化参数：64 latent dimensions、PCA10 output、flow layers 10、flow size 0.774、flow hidden 2、MAF layers 10、budget `exp`、bound 30、warmup lr `1e-3`、main lr `1e-4`、`lam1=1e-7`、`lam2=0.1`、`lam3=1e-5`、plateau scheduler、change step 2、reduce rate 0.99。E162 显式覆盖 `batch_size=512`、`accumulate_grad_batches=1`；warmup 5 epochs，main 最多 50 epochs。main 按 `val/loss` 最小选 checkpoint，early stopping 使用 native module 自带 callback：`mode=min`、`min_delta=1e-3`、`patience=3`。不得再叠加第二个 EarlyStopping。若 best path/分数缺失或非有限，该种子失败，禁止用 last checkpoint 替代。

先完成三个种子的 checkpoint 锁定，再进入任何 label-only forward。主种子失败时，禁止改用 3408/3409、挑选最优种子或求平均救结果。

## 4. checkpoint 边界与锁定

Lightning hyperparameters 删除 `adata` 和 `model` 对象。每个种子锁定：

- best Lightning checkpoint 哈希与最低 finite `val/loss`；
- state-dict-only immutable checkpoint；
- runner/contract/PRESCRIBE source/E160/E161 interface 与资产哈希；
- gene order、PCA model、control prior 哈希；
- 递归 checkpoint 对象审计，不得含 AnnData、raw/test 路径或真值对象。

中断只允许在 runner、contract、input/source manifest 完全一致时从已登记的 warmup/main `last.ckpt` 恢复。

## 5. validation label-only 非退化门

三种子全部锁定后，先对固定的同一组 8 个 validation 条件执行 native graph/label-only graph forward 等价性审计。排序键固定为 `SHA256("E162|Wessels|forward-equivalence|20260714|v1\t" + condition)`，取最小 8 项。所有 prediction/`raw_log_prob`/epistemic/aleatoric 差值必须 finite 且最大绝对差 `<=1e-5`。

随后每个种子只用 24 个 validation 扰动字符串与 frozen train-control mean 生成 label-only query，不附带 validation truth。固定门：

- `raw_log_prob`：24 finite，精确 unique 数 `>=12`，sample std(ddof=1) `>1e-6`；
- prediction：`24×10` finite，精确 unique vectors `>=12`，至少一个 coordinate sample std(ddof=1) `>1e-6`。

3407 未通过时立即固定失败记录，不得查询任何 test label。3408/3409 的 validation 门只用于敏感性可解释性，不改变 3407 决策。

## 6. 48 个 test label-only 分数

test labels 的存在性和顺序可在输入核对阶段读取；任何 test label 的模型 forward query 只能发生在 3407 validation 门通过之后。首次 query 前先原子写入 `TEST_LABEL_QUERY_EVENT.json`，锁定 checkpoint、输入、来源和 validation gate 哈希。每个 query 只含：

- `x = E161 TRAIN_ONLY_CONTROL_PRIOR.npz::control_gene_mean`；
- canonical perturbation string；
- frozen scGPT embedding 与 locked checkpoint。

query 中不得存在 `y`、`y_pca`、test cell count、expression、DE 或 error。先锁定 3407 的 48 行分数，并执行 E160 主门：

- `raw_log_prob`：48 finite，精确 unique `>=24`，sample std(ddof=1) `>1e-6`；
- PCA prediction：`48×10` finite，精确 unique vectors `>=24`，至少一个 coordinate std `>1e-6`。

主门失败时保留 3407 label-only 失败表，E163 禁止解封，也不查询敏感性 test 分数。主门通过后，3408/3409 只有各自 validation 门通过才可查询各自的 48 个 test labels；某个敏感种子 validation 失败时，其 test 表/hash 留空并记录 `not_queried_due_to_validation_gate`，不改变 3407 的主决策。

每行保存 PCA10 prediction、`raw_log_prob`、epistemic、aleatoric、`2×epistemic+aleatoric` 与 PCA reconstructed predicted magnitude RMS。official/magnitude 如为常数或非有限，只在 estimability 表中标记 `NA/constant_or_nonfinite_baseline`，不使主流程报错，不用 0 填充统计量。

## 7. 原子发布与 E163 接口

三个 checkpoint 和 label-only 表各自先写 temporary file，哈希/fsync 后原子 rename。可提交的 release 先写 `.release.staging`，递归 allowlist、哈希和 fsync 通过后一次 directory rename 为 `release/`。

E162→E163 interface 必须明确：3407 是唯一主模型，3408/3409 仅敏感性；3407 checkpoint/分数 SHA256 必填，敏感种子 hash 可因 validation 门失败留空但必须给原因；gene/PCA/control/source/input 哈希；validation 与 test 非退化门；`raw_h5ad_opened=false`、`test_X_accessed=false`、`test_truth_accessed=false`、`test_endpoint_computed=false`、`test_graphs=0`。E163 不能只信布尔字段，必须重新核对表、hash 和主门统计。

## 8. 失败与禁止项

每次运行使用 append-only `attempt_NNN`。任一 input/source/hash/graph/checkpoint/validation/main test 门失败都写入该 attempt 的 `RUN_STATUS.json`，旧 attempt 永不覆盖。只有 runner/contract/input/source manifest 完全一致的同一 attempt 才可从 `last.ckpt` 恢复；代码或 HEAD 改变必须开始新 attempt。禁止：

- 打开 Wessels raw h5ad 或建立 test graph/DataLoader；
- 用 validation/test 重选 gene、PCA、split、seed、epoch、batch 或 score 公式；
- 为了非退化而 jitter、改 clamp/bound、换 checkpoint、换种子或挑选结果；
- 在 E162 计算 test truth、accuracy、相关、coverage 或其他端点。

当前 native `ListMLELoss` 已确认对排列不敏感，E-distance 在这一 arm 中只是接口字段，不能声称为模型提供了有效的排序监督。

## 9. 固定运行接口

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e162_wessels_prescribe_native.py --mode preflight

CUDA_VISIBLE_DEVICES=0 /home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e162_wessels_prescribe_native.py --mode formal --gpu-index 0
```

`formal` 依次锁定 3407/3408/3409，然后执行 validation 门与有条件的 test label-only 锁定。正式运行前 runner 与本合同必须已提交 Git。
