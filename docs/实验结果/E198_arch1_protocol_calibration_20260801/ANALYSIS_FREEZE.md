# E198｜arch1 外部评价协议校准冻结

冻结时间：2026-08-02T00:02:40+08:00

分析性质：`EXTERNAL_PROTOCOL_CALIBRATION`。本实验不训练模型、不读取预测文件、
不验证 SafeConf，也不把 `arch1` 写成盲法 leaderboard 测试。目标是在后续 E199
打开模型结果之前，先确定哪些评价协议能区分技术重复与无信息参考。

## 数据合同

- 数据集：scPertEval 官方 `arch1` processed training split；
- 公开元数据：H1 hESC、150 个 CRISPRi 扰动；
- 文件大小：`4377281643` bytes；
- 文件 SHA-256：`f85862ecc9d9c34c1765b395105c74af08480471ade27ce8466d742d73c1449e`；
- `prepare` 只做字节哈希，没有调用 AnnData/HDF5 解析；
- E198 formal 首次按数据对象打开文件；
- 该数据只有一个 context，不能作为 unseen-row 或 cross-context 证据。

## 实现与资源合同

- scPertEval commit：`8709eb07a0e7d4ecf1c60c977f2018690a749975`；
- runner SHA-256：`fb25c4415caa708b8d9122893e18aed5c5d8b7d6aec212ce0988808c6907fdc9`；
- seed：`20260801`；min cells：`30`；
- all-perturbed/control subsample：`2048`；workers：`8`；
- DE：`t-test`；PCA：50 components；bootstrap：`5000`；
- 不运行 Sinkhorn，不安装或临时补入可选依赖；
- formal 前 runner、冻结文件、输入哈希和 prepare gates 必须已提交，且 GitHub、
  Gitee 与本地 HEAD 三者完全一致。

## 固定协议

- `pearson`
- `pearson_ctrl`
- `pearson_pert`
- `mse`
- `wmse_exp2`
- `rank`
- `transpose_rank`
- `energy_distance_pca_k=50`
- `unbiased_mmd_median_pca_k=50`
- `de_auprc`
- `de_auroc`
- `de_overlap_k=50`

## 评价门槛

每个协议保存逐扰动 `raw_positive`、`raw_negative` 和官方 DRF，并独立复算 DRF 与
BDS。完整性失败会中止实验；指标表现差只形成科学负结果，不中止或删表。

- `REJECT`：BDS < 0.5、DRF 中位数 ≤ 0，或有限 DRF 比例 < 90%；
- `SECONDARY_ONLY`：达到 BDS ≥ 0.5 且 DRF 中位数 > 0，但任一主要不确定性门槛
  未过；
- `PRIMARY_ELIGIBLE`：BDS 的双侧 95% Wilson 下界 > 0.5，且 5,000 次按扰动
  重抽样的 DRF 中位数 95% 下界 > 0，有限 DRF 比例 ≥ 90%。

官方文档只明确规定 BDS < 0.5 的协议不可信；Wilson 与 bootstrap 是本实验事前
增加的主要端点门槛，不冒充 scPertEval 官方阈值。

## 后续端点选择优先级

每个生物学轴最多选择一个 `PRIMARY_ELIGIBLE` 协议，严格按以下顺序取第一个；
没有通过者就留空，不能按 E199 模型表现改选：

- `absolute`: `mse` → `wmse_exp2`
- `direction`: `pearson_pert` → `pearson_ctrl` → `pearson`
- `retrieval`: `rank` → `transpose_rank`
- `population`: `energy_distance_pca_k=50` → `unbiased_mmd_median_pca_k=50`
- `de`: `de_auprc` → `de_overlap_k=50` → `de_auroc`

## 失败与停止规则

1. 数据大小/SHA、scPertEval commit/source hash 或冻结 Git 状态不符，拒绝 formal；
2. 任一协议不是 150 个扰动、raw controls 出现正负无穷、官方与独立 NA mask
   不同，或 DRF 公式差值超过 `1e-12`，拒绝发布完整状态；官方定义允许的
   `NaN` 保留，并通过有限 DRF 比例进入科学裁决；
3. 不因某个协议 BDS/DRF 低而换参数、换 PCA、换 DE 或删除该协议；
4. 不把 E198 写成预测模型性能、SafeConf 外部确认或跨 context 结果；
5. 正式输出存在或已标记 COMPLETE 时拒绝覆盖。
