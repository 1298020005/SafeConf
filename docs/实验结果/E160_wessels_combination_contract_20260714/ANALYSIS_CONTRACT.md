# E160｜Wessels 已见组分、未见组合前瞻 requirements 合同

冻结日期：2026-07-14。E160 只冻结任务、切分、访问边界、预解封非退化要求和评价算法。它不声称模型已训练、requirements 已通过或 Wessels test expression 已解封。

## 1. 事前侦察和设计时间线

2026-07-14，在 E160 合同定稿前，已经对 `official_scperturb/WesselsSatija2023.h5ad` 做过一次元数据侦察：读取 AnnData `obs`、`var_names`、shape，并查看 HDF5 中 `X` 的存储类型/dtype 和 layers 存在性。该侦察确认：

- shape 为 30,707 cells × 21,052 genes；
- 187 个 raw conditions；
- `X` 为 CSR `int64` raw counts，无 layers；
- 除 VPRBP 外有 27 个可对齐 single；
- 排除含 VPRBP 的组合后有 142 个可对齐 pair，其中 116 个至少 75 cells。

这些 hard-coded counts 来自上述事前元数据/存储结构侦察。侦察没有索引、切片、转换或物化任何 `X` values，没有计算 Wessels effect、prediction、score 或 error。E160 正式 runner 不重复检查 `X` 存储结构；它只读 `obs`、`var_names`和 shape。

## 2. 研究问题

control 和所有可编码 single 进入 train，可编码 pair 按固定 SHA256 分为 train/val/test。每个 test pair 的两个组成基因都已作为 single 出现在 train。因此它回答 `seen-component / unseen-combination`，不是 unseen-gene 泛化。

## 3. 正式 runner 的访问边界

runner 只允许：

- 以 AnnData backed read-only 方式读取 `obs`、`var_names`和 `(n_obs, n_vars)`；
- 读取已提交的纯文本 `SCGPT_PERTURBATION_VOCABULARY.txt`；
- 对原 scGPT embedding 文件做 SHA256 字节哈希，但不反序列化；
- 在同一次流式读取中同时计算 raw MD5 和 SHA256。这是不解析语义的全文件字节哈希，不是读取表达值。

runner 禁止访问 `adata.X`、`layers`、`raw.X`、test expression 行、预测结果或 Wessels 任务误差。raw 必须是普通文件而非 symlink。哈希前后和 AnnData metadata open 前后的 device/inode/size/mtime 必须一致，用于拒绝路径替换/TOCTOU。

## 4. Git blob 与固定输入门

在 raw 哈希或 AnnData open 前，runner 必须验证下列工作树文件与当前 Git `HEAD` blob 逐字节一致：

- `tools/scripts/run_e160_wessels_combination_contract.py`；
- 本 `ANALYSIS_CONTRACT.md`；
- `SCGPT_PERTURBATION_VOCABULARY.txt`；
- E158 attempt-1 `RUN_STATUS.json`；
- E158 attempt-1 `UNSEAL_EVENT.json`；
- E159 `RUN_STATUS.json`；
- E159 `tables/E159_POSTHOC_SPEARMAN.csv`。

任一文件未提交或与 `HEAD` 不同时立即终止。原 scGPT embedding 必须匹配固定 SHA256 `9a5be69676bc09fbf996ae7be1d4faa09c9f32abbf733f33fc130153829ad8ce`，但正式 runner 不对它执行 `pickle.load`。纯文本 vocabulary 必须大写、字典序、无重复且正好 60,697 行。

## 5. E158/E159 事后来源和不可追溯性

E160 必须如实锁定下列已发生事实：

- E158 在 P3/P4 test X 已解封并物化后失败；
- E159 是 `post_unseal_forensic_not_preregistered`；
- E159 确认 official combined、magnitude 和十个 PCA prediction coordinates 在面板内退化为常数；
- E159 对 raw log probability 的分析只是 hypothesis-generating。

E159 的 post-hoc Spearman 给出了相反证据：

| truth/accuracy | P3 raw-log-probability ρ | P4 raw-log-probability ρ |
|---|---:|---:|
| train-only PCA10 inverse-transform accuracy | 0.1773913043 | 0.2904347826 |
| full selected-gene raw log-normalized truth sensitivity | -0.2730434783 | -0.5000000000 |

因此，Wessels 上“raw log probability 能排序 PCA10 accuracy”是受 E159 解封后正负结果共同启发的新假设。它在 Wessels test truth 封存状态下可作为新前瞻验证，但不得追溯声称为 E155/E158 事前假设，也不得隐去 raw-truth 负结果。

## 6. canonical condition 和数据资格

canonical 基因名强制大写：

- raw `control` → `ctrl`；
- raw single `GENE` → `GENE+ctrl`；
- raw pair 的两个基因先大写、再按字典序排列并用 `+` 连接。

`Guide.Class` 必须与 `nperts` 严格一致：`0↔NT`、`1↔Single`、`2↔Dual`。可编码表示扰动基因同时存在于固定 scGPT vocabulary 和 Wessels `var_names`。VPRBP 必须同时被确认为不在 vocabulary 和不在 expression var axis。

- `ctrl` 强制进入 train；
- 27 个 compatible singles 全部进入 train；
- compatible pairs 必须为 142；
- 其中 `n_cells >= 75` 的 116 pairs 进入切分；
- 116 pairs 的每个 component 都必须出现在 27 train singles 中。

对每个入选 pair 计算：

```text
SHA256("E160|Wessels|seen-component-unseen-pair|20260714|v1" + TAB + canonical_condition)
```

按 `(sha256, canonical_condition)` 升序：前 48 test，随后 24 val，余下 44 train。最终 train 为 `ctrl + 27 singles + 44 pairs = 72 conditions`，val 24 pairs，test 48 pairs。无 reserve，不替换任务。

## 7. test seal 与 train-only 预处理

开发资产只能含 train/val conditions。从 raw counts 对每个细胞单独计算：

\[
L(x_{ig})=\log\left(1+10{,}000\frac{x_{ig}}{\sum_{h\in G_{raw}}x_{ih}}\right).
\]

HVG/selected-gene set \(G\)、PCA mean \(m_G\)、PCA10 components \(W\in\mathbb{R}^{10\times |G|}\)、train-control mean、标准化和排序标签只能在 train 拟合。val/test 只做固定变换。test expression 不得出现在开发 h5ad、graph、checkpoint、DE index、score table 或 label-only query 中。

## 8. 主真值、敏感性真值和 task mean

对任务 \(t\) 的 \(n_t\) 个细胞，task mean 固定为“先按细胞 normalize/log1p，再求均值”：

\[
\mu_t=\frac{1}{n_t}\sum_{i\in t}L(x_i)_G,
\qquad
\mu_{ctrl}=\frac{1}{n_{ctrl}^{train}}\sum_{i\in ctrl,train}L(x_i)_G.
\]

不得先将 counts 做 pseudobulk 求和再 normalize。

### 8.1 强制主真值：train-only PCA10 inverse transform

\[
z_t=(\mu_t-m_G)W^\top,
\qquad
\widetilde\mu_t^{PCA10}=z_tW+m_G,
\qquad
e_t^{PCA10}=\widetilde\mu_t^{PCA10}-\mu_{ctrl}.
\]

若模型的 label-only PCA10 prediction 为 \(\widehat z_t\)，则：

\[
\widehat\mu_t=\widehat z_tW+m_G,
\qquad
\widehat e_t=\widehat\mu_t-\mu_{ctrl},
\qquad
A_t^{PCA10}=\operatorname{Pearson}(\widehat e_t,e_t^{PCA10}).
\]

`pearson_effect_accuracy = A_t^{PCA10}` 是主 accuracy。PCA 必须完全由 train 拟合。

### 8.2 强制 raw full-selected-gene 敏感性

\[
e_t^{raw}=\mu_t-\mu_{ctrl},
\qquad
A_t^{raw}=\operatorname{Pearson}(\widehat e_t,e_t^{raw}).
\]

48 个 test pairs 必须同时报告 `raw_pearson_effect_accuracy_sensitivity = A_t^{raw}`。它不替换 PCA10 主真值，但无论方向是否一致都不得删除。raw direction 和 raw RMSE 也必须作对应敏感性分析。

## 9. required 预解封非退化门

下列都是未来运行必须满足的 `required_*` requirements，不是 E160 已经通过的结果。

### 9.1 `required_raw_log_prob_gate`

- 锁分表正好 48 test rows，一任务一行；
- 48 个 `raw_log_prob` 全部 finite；
- exact unique 值数至少 24；
- sample SD (`ddof=1`) 严格大于 `1e-6`。

### 9.2 `required_prediction_non_degeneracy_gate`

- label-only PCA prediction 形状正好为 `48×10`；
- 480 个数值全部 finite；
- 按十个坐标的 exact tuple 计算，unique prediction vectors 至少 24；
- 十个 PCA coordinates 中至少一个的 task-level sample SD (`ddof=1`) 严格大于 `1e-6`。

任一主门失败时 test X 不得解封，只能用 train/val 排查或按未解封时已写入的 fallback 规则重训。

### 9.3 `required_baseline_estimability`

official combined confidence 和 predicted magnitude 都必须预先锁定。某基线只有在 48 个值全部 finite、exact unique 至少 2、sample SD (`ddof=1`) 严格大于 `1e-12` 时才可估计。未通过则该基线的 rho、CI 和 delta-rho 记 `NA` 并写明 `constant_or_nonfinite_baseline`，不得 zero-fill。

## 10. 固定主统计与常数规则

- 主分数：PRESCRIBE native `raw_log_prob`，越大表示 latent density 越高。
- 主统计：\(\rho_{raw}=\operatorname{Spearman}(raw\_log\_prob,A^{PCA10})\)，预期为正。
- 主确认门：\(\rho_{raw}>0\) 且 task-bootstrap 95% percentile CI 下界大于 0。
- 若主分数或主 accuracy 非 finite、exact unique <2 或 sample SD 不大于 `1e-12`，主 rho 和 CI 记 `NA`，主门直接失败；不得记 0，不得换 endpoint。

task bootstrap 完全固定为：

- `numpy.random.default_rng(3407)`；
- 10,000 replicates；
- 每次从 48 task indices 有放回抽 48 个；
- raw log probability 和所有基线使用同一组 paired indices；
- CI 为 `numpy.quantile([0.025,0.975], method="linear")`；
- finite 且可估计的 replicate 少于 9,500 时 CI 记 `NA`。

对每个可估计基线 \(b\) 同时报告 paired 10,000-bootstrap：

\[
\Delta\rho_b=\rho_{raw}-\rho_b.
\]

基线为 `official_combined_confidence = 2*epistemic + aleatoric` 和 `predicted_magnitude_rms`。这些 delta 是增量比较，不改变主 raw-log-probability 确认门。

## 11. 共享基因的固定敏感性算法

### 11.1 gene-cluster bootstrap

从 48 test pairs 取得所有不同 component genes，设数量为 \(K\)。每个 replicate 用新的 `numpy.random.default_rng(3407)` 从 \(K\) genes 有放回抽 \(K\) 次。对每个抽到的 gene，将所有包含该 gene 的 test tasks 各加入一次重采样多重集；同一 pair 可因两个 component 或某 gene 被重复抽中而重复出现。在该多重集上计算 Spearman，执行 10,000 次，CI 同样使用 linear percentile 和 9,500-valid 门。这是依赖性敏感性，不替换 task-bootstrap 主 CI。

### 11.2 leave-one-gene-out

对每个不同 component gene \(g\)，删除所有含 \(g\) 的 test pairs，在剩余任务上重算主 Spearman。每行固定输出 removed gene、removed/remaining task counts、rho 和 estimability status；汇总有效 rho 的 min/median/max 以及正号比例。不删除不利 gene，不用该结果改 test set。

## 12. 次要终点和 coverage

- PCA10 truth 下 `frac_correct_direction_all`、true-absolute-effect top-20 genes direction 和 `rmse_effect_error`；
- raw full-selected-gene truth 下 Pearson、direction 和 RMSE 强制敏感性；
- raw log probability 对 direction 预期正相关，对 RMSE 预期负相关；
- coverage grid 固定为 0.50, 0.55, …, 1.00；
- 每档 retained task count 严格为 `ceil(coverage * 48)`；
- 按 raw log probability 从高到低保留，tie 用 canonical condition 升序打破；
- 每档报告 retained PCA10 Pearson accuracy mean/median 和 PCA10 RMSE mean/median，并平行报告 raw-truth 敏感性。

## 13. E10/E40 资产历史

WesselsSatija2023 曾出现在 E10/E40 账本。runner 必须哈希并静态检查：E10 仅处理路径、大小、官方 metadata 和 inventory CSV；E40 仅读 `h5ad_scan.tsv`、路径/大小汇总和下载状态。这证明的是“仓库内已记录用途为资产/元数据”，不是对过去一切人工操作的绝对证明。

## 14. 原子发布、allowlist 和恢复

E160 正式输出统一在 `freeze/`。runner 先在同一文件系统的 `.freeze.staging/` 中写入显式 allowlist，逐文件 fsync、重构比较 payload、拒绝 symlink/unknown path，然后用一次 directory rename 原子发布为 `freeze/`。

- `freeze/` 已存在时 formal 拒绝覆盖，只允许 `--verify`；
- staging 残留时默认终止；
- 只有显式 `--recover-staging`、sentinel 内容正确、全部路径在 allowlist 内且无 symlink 时，才删除 staging 并从头重构；
- `--verify` 先从已发布 status 取得正式冻结时的 Git commit，要求工作树字节同时等于该正式 commit 与当前 `HEAD` 的 blob；随后重算 raw 双哈希、metadata、split 和所有 expected payload，对 manifest 逐字节比较，并保留原 formal Git gate 与 `executed_at` 重建完整 status 做全字段逐字节比较。这样后续提交冻结产物导致 `HEAD` 前进时不会产生伪失败，任一固定输入发生变化仍会失败。

## 15. 解封顺序

1. 提交 runner、本合同和纯文本 vocabulary。
2. 运行 E160，原子发布 requirements manifest/status。
3. 仅用 train/val 训练并锁 checkpoint。
4. 对 48 test labels 生成且哈希锁定 `48×10` predictions、raw log probability 和基线。
5. 验证两个 required 主非退化门及所有 source/checkpoint/gene-order hashes。
6. 写一次性 `UNSEAL_EVENT.json`。
7. 最后才读 48 test pair X rows，同时计算 PCA10 主真值和 raw 强制敏感性。

解封后任何修复必须保留原 attempt，不删除、覆盖或重写时间线。
