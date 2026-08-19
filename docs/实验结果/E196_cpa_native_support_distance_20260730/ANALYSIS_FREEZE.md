# E196｜CPA 原生潜空间支持距离审计

冻结日期：2026-07-30  
冻结协议起始基线：`ee6e1d91515d330b2c4468fc496013908a450de2`

正式运行实现与输入锁：`E196_CODE_LOCK.json`（运行前提交并由 preflight 对照 HEAD）

## 证据性质

E196 不重训模型。它从 E94 的 pinned CPA 权重和 E84 八个 frozen-manifest 权重中
导出 CPA 定义的 uncertainty：目标 `context + dose-weighted perturbation`
潜向量到训练条件潜向量的最近距离。

所有这些任务的真值已经在既往实验中解封，因此证据标签固定为
`POSTTRUTH_DIRECT_COMPETITOR_AUDIT`。本实验可以补齐真实 CPA 原生分数和运行成本，
不能充当新的盲测。

## 方法语义

CPA 0.8.8 的原生 uncertainty 不是 decoder predictive variance。它是训练支持度
或 OOD 距离：

1. 计算目标条件的 context embedding 与 dose-weighted perturbation embedding；
2. 对训练中出现的条件计算相同组合 embedding；
3. 返回到最近训练条件的 cosine distance、Euclidean distance及最近邻标签。

全文统一称为“CPA native latent support distance”，不得写成概率校准、不确定性
方差或误差下界。

## 冻结输入

### 开发适配器检查

- E94 pinned reproduction，manifest `E81_r1_p75`；
- 59 个测试任务；
- CPA state SHA-256：
  `7e457786638cb85a183a82e7ce7b6442ed1c63dc76fccacfa5d28c3866b9d51d`；
- E94 task score SHA-256：
  `3a71241a9c0effe9eba4776e660054d84c87fe2ea6ff8b0f7b29e68f31f1b1a7`。

E94 只用于确认模型重建、权重加载、任务连接和距离实现，不进入 E84 正式
manifest 宏平均。

### 八个既有 formal manifest

| manifest | state SHA-256 |
|---|---|
| E81_r1_p25 | `cf8c449a08ba6bcdc9aecdf89d0429410aacdea078013e614196ea7059329540` |
| E81_r1_p50 | `1c79f7976d9a3230a1c88e5192167855b861129c06bf9493f7a2fec99fc727de` |
| E81_r2_p25 | `830a317f6cff5fc7496d894f2c6dbfc37c2041f6d3eea1e3cb015ef974b5a2bb` |
| E81_r2_p50 | `5b1c5e33ae174e234696069c7e1bd50e93d78361b52bf30dde42a54d6040ce62` |
| E81_r2_p75 | `d33e6b3c8c9258c8c355cce6f8a9ef979b3317db86f5ade23d5673052aaafaea` |
| E81_r3_p25 | `fd6aa4b87fc5fc84535dacc70ad7e362f598947691cc76e759ec094931be0f18` |
| E81_r3_p50 | `0710eee8153d534f72208e83f0db23a58b7ebe12a570f7e10fc008643f98b2e0` |
| E81_r3_p75 | `f45131b07e593da1dfd19fd36fc87c1e92f9c834bb265ac70525438241208f7b` |

共同来源：

- E81 split manifest：
  `00aaed01fece99b55595c982faa75f6b134b4de277a2415a982036eb7ce427e2`；
- E81 1000-gene panel：
  `71cb9dd8d16897f2fa8ebbcdf6cab0981a080847602ba7c3f006be2c3308280e`；
- E83 frozen construction/training script：
  `b748f8a455698873c89af9635d6ff5ed0824747f0eb21770869667233853ad72`；
- CPA source commit：
  `fbd7c0250edc23eff003a10c99655579c53afd63`。

## 冻结模型重建

每个 manifest 必须使用原运行的：

- CPA 0.8.8；
- RDKit Morgan perturbation embedding；
- `n_latent=32`；
- Gaussian reconstruction；
- linear doser；
- log10-dose 输入；
- cell-line categorical covariate；
- 原 seed、细胞抽样、train/valid/test 标记和网络参数。

程序调用 E83 frozen input builder 重建 AnnData 和模型结构，加载已保存
`model.module.state_dict()`；不得再训练或更新任何参数。

## 冻结参考集合

必须显式从 `split_cpa == "train"` 构造参考条件，validation 和 test 条件不得进入。
控制条件与 perturbed train 条件都保留，并记录类别。

主参考集合：

- `all_explicit_train_conditions`；
- 每个唯一 `cell_line × drug_cpa × dose_cpa` 恰有一个组合向量；
- 不使用细胞数阈值。

敏感性参考集合：

- `official_gt30_train_conditions`；
- 只保留训练细胞数严格大于 30 的条件；
- 用于复核 CPA API 中硬编码的 `thrh=30` 行为。

另报告 `perturbed_train_only`，用于检查 control embedding 是否经常成为最近邻。
三种参考集合全部保留，主结论以无阈值的显式 train 集为准。

## 冻结分数与结果变量

每个 `manifest × task × reference_set` 输出：

- `native_cosine_distance`；
- `native_euclidean_distance`；
- cosine / Euclidean 各自最近的训练条件；
- 最近条件的 context、drug、dose、是否 control；
- 训练参考条件数；
- CPA 自身 predicted-effect magnitude；
- CPA 自身 RMSE。

两个距离均为数值越大风险越高。主结果变量固定为同一 CPA predictor 的
`error_cpa_rmse`。`cpa_ridge_disagreement → pair_mean_rmse` 仅作为不同 family
目标的并列补充，不能与 CPA 距离混成相同结果变量。

## 统计

- E94 单独报告，不进入 formal 汇总；
- 八个 E84 manifest 各自报告 Spearman、10%/20%/30% routing、
  normalized risk–coverage、normalized AURC 和 20% oracle-normalized utility；
- formal 主汇总先在每个 manifest 内计算指标，再对 8 个 manifest 等权宏平均；
- manifest bootstrap 10,000 次；
- 因同一 biological task 可出现在多个 manifest，另做 task-key cluster bootstrap
  敏感性；每次抽样后仍在各 manifest 内计算 Spearman，再对 manifest 等权平均，
  不得跨不同 CPA 潜空间直接对 629 行 raw distance 排名；
- manifest 重采样区间与 task-key cluster 区间均称为描述性区间。八个 manifest
  共享任务且不是 iid 外部样本，10,000 次只控制 Monte Carlo 误差；
- cosine distance、Euclidean distance、predicted magnitude 全量报告；
- 同一 CPA outcome 上报告 distance 相对 magnitude 的 paired delta；
- 分层报告四个 Cartesian quadrant，不选择最有利象限代替总体。

## 实现 gate

1. E94 与 8 个 E84 state 文件 SHA-256 全部匹配；
2. 九个模型结构均严格加载 state，missing/unexpected key 都为 0；
3. 模型处于 evaluation mode，推理期间参数哈希不变；
4. 每个 manifest 的测试 task 与既有 prediction manifest 一对一连接；
5. 冻结 CPA builder 会将来源 h5ad 载入进程以重建训练输入、pseudo-control
   预测与 control mean；distance 函数只接收 train reference、模型参数和目标
   标签/剂量/context，target perturbed expression 与 error 列不进入距离数值；
6. validation/test 条件进入 reference set 的数量必须为 0；
7. 主参考集合中所有显式 train condition 均保留，重复组合恰好去重；
8. 每种 reference set 的距离动态范围、唯一值数和 NaN 数显式输出；
9. constant score 标记 `NON_ESTIMABLE`，不得写成相关系数 0；
10. E94 重建模型对既有 pseudo-test 的预测 effect 最大差不超过 \(10^{-5}\)；
11. 全部运行命令、环境、GPU/CPU、耗时、input/output SHA-256 留档；
12. 不得按真值改变 reference set、距离、阈值、manifest 或任务。
13. 所有九个模型先完成 pre-outcome distance 阶段并写入带哈希的阶段文件，再读取
    task-score outcome；split、prediction manifest 与 score metadata 必须逐列一致；
14. routing 对每个 `manifest × reference_set × scope` 单独检查动态范围；常数分数
    不得通过任意 tie-break 生成 top-k 效用。非恒定分数使用跨 score 共享的 task-key
    tie 顺序；
15. AURC 不带 review-budget 维度，只报告一份；主 headline 与 task-cluster
    sensitivity 必须在全部八个 formal manifest 中可估，否则 fail closed。

gate 检查工程一致性与泄漏，不要求距离相关为正。

## 冻结输出

- `E196_STATUS.json`
- `E196_CODE_LOCK.json`
- `tables/E196_PREOUTCOME_TASK_DISTANCES.csv`
- `tables/E196_PREOUTCOME_REFERENCE_CONDITIONS.csv`
- `tables/E196_PREOUTCOME_PROVENANCE.json`
- `tables/E196_TASK_DISTANCES.csv`
- `tables/E196_REFERENCE_CONDITIONS.csv`
- `tables/E196_DYNAMIC_RANGE_AUDIT.csv`
- `tables/E196_ASSOCIATION.csv`
- `tables/E196_PAIRED_DELTAS.csv`
- `tables/E196_ROUTING_METRICS.csv`
- `tables/E196_RISK_COVERAGE.csv`
- `tables/E196_QUADRANT_SUMMARY.csv`
- `tables/E196_INVARIANT_AUDIT.csv`
- `tables/E196_INPUT_HASHES.csv`
- `tables/E196_RUNTIME_ENVIRONMENT.csv`
- `figures/E196_cpa_native_support_distance.png/.pdf`
- `reports/E196_REPORT.md`
- `reports/E196_INTERPRETATION.md`
- `reports/RUN_RECORD.md`

E196 不生成新模型权重，不复制 E84/E94 的已有大文件。

## 正式输出生成前的审计修订（2026-07-30）

初版 runner 经两次只读审计发现：constant-score routing 会由 score 名称相关的
hash 制造任意排序；task-key bootstrap 将不同 CPA 模型的 raw latent distance
错误合并；AURC 在三个 budget 上重复；runner、自身依赖和既有 outcome 未全部
预锁定。上述问题均在正式 full 运行前修正，修订后的 runner、协议、CPA 运行源码、
原始 h5ad、SMILES、九组 state/prediction/outcome/status，以及每组
seed/train-task/validation-task/reference-condition 集合统一写入
`E196_CODE_LOCK.json`。修订项来自预先列出的工程和估计量缺陷，不依据结果方向
选择更有利的实现。

审计过程中为验证九个模型的 state 加载、预测复现和数据结构，执行过只读内存
复算，审计者可能接触主统计数值；没有落盘为正式 E196 结果，也没有依据方向选择
阈值、reference set、距离或任务。因此 E196 始终保持
`POSTTRUTH_DIRECT_COMPETITOR_AUDIT` 的描述性证据边界，不能表述为盲测。
