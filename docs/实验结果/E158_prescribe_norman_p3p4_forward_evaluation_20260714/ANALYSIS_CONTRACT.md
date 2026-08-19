# E158｜PRESCRIBE Norman P3/P4 严格前瞻解封与评价合同

冻结日期：2026-07-14。本合同和 E158 runner 必须在 E157 formal 训练完成、测试表达解封之前提交 Git。E158 只做一次固定规则的前瞻评价；看到 P3/P4 真值后不得更换任务、checkpoint、分数、基因轴、终点、bootstrap 规则或 coverage。

## 1. 解封前必须同时满足的条件

E158 在访问 Norman raw H5AD 的任何字节之前，必须对 P3、P4 两个 E157 formal run 完成以下校验：

1. `STATUS.json` 的 phase 均为 `complete_checkpoint_and_label_only_scores_locked_no_test_truth_access`，`mode=formal`，`seed=3407`；
2. `test_expression_accessed=false` 且 `test_endpoint_computed=false`；
3. best Lightning checkpoint 和 state-dict-only locked checkpoint 均存在，实际 SHA256 与 E157 STATUS 记录一致；
4. `SOURCE_MANIFEST.csv`、`INPUT_MANIFEST.csv`、开发集 forward-equivalence 审计和 label-only task-score CSV 的 SHA256 均与 STATUS 一致；
5. 两个 task-score CSV 各含 24 个不重复任务，任务集与 E155 P3/P4 合同逐项一致，数值全部有限；
6. E157 记录的 runner 和 contract 必须与当前文件字节一致，E156 的 dev H5AD、cell graph 和 artifact manifest 哈希未变；
7. E158 runner 和本合同均已提交，工作树文件与 Git HEAD blob 一致。

状态文件中记录的 checkpoint、locked score、dev H5AD 和 graph 路径在任何 open/hash 之前必须先通过固定 allowlist；不得用 STATUS 里的任意路径进行“校验”。E155/E156 STATUS、E155 split/condition audit、E156 artifact manifest，以及 E157 STATUS/source/input/graph/forward/score 审计资产均必须与 Git HEAD blob 逐字节一致。E155 condition audit 须在解封前证明 48 个任务唯一覆盖且细胞数为正整数。E157 的 gene-order hash 和 48 条 predicted magnitude 须仅用锁定预测与 E156 transform 在解封前全部复现。

任一条失败时，runner 必须在触碰 raw H5AD 之前终止。两个面板不允许分别解封。

## 2. 一次性解封记录

全部锁定资产验证通过后，runner 先原子写入 `UNSEAL_EVENT.json` 和未完成 STATUS，记录时间、Git HEAD、E158 源码/合同哈希、两个 E157 STATUS/checkpoint/task-score 哈希；然后才允许读取 raw H5AD 字节、`obs` 和固定测试行的 X。从写入该事件起，E158 已解封；即使后续因 I/O 或实现错误中断，也不得删除记录或宣称真值未解封。

raw Norman 数据的冻结 SHA256 为 `efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc`。解封后先校验整文件哈希，再索引测试 X；哈希不一致时不计算结果。

## 3. 固定测试变换

- 原始 condition 按 E155/E156 规则归一化：`control`→`ctrl`，单基因补 `+ctrl`，组分按字典序组合。
- 只索引 E155 冻结的 P3/P4 各 24 个 test condition，不使用 reserve，不丢弃任何难任务。
- 基因顺序严格使用 E156 dev H5AD 的 2,044 基因轴。每个测试细胞独立做 `target_sum=10000`、再 `log1p`；该变换不拟合跨细胞参数。
- 使用 E156 只在 shared train 拟合的 `pca_mean` 和 `pca_components`做 PCA10 transform 与 inverse transform。不重新选 HVG，不重新拟合 PCA。
- 预测从 E157 已锁定的 `predicted_pca_0…9` 重构，不重跑 checkpoint，不重算置信度。
- 预测效应和真实效应都减去同一个 E156 shared-train control 平均向量。

## 4. 冻结终点

### 4.1 主终点

对每个任务，将测试细胞 PCA10 坐标取均值并 inverse transform，得到 `PCA10-reconstructed task mean`。主要准确度 `pearson_effect_accuracy` 是 2,044 维预测效应与 PCA10-reconstructed 真实效应的 Pearson 相关，越高越好。

主分数是 E157 预锁的官方组合置信度：

`combined_confidence = 2 × epistemic_confidence + aleatoric_confidence`。

固定比较项为 E157 预锁的 `predicted_magnitude_rms`。不拟合新分数或新权重。

主统计量为：P3、P4 内分别计算 score 与 Pearson accuracy 的 Spearman 相关，再对两个 panel rho 做等权宏平均。P3 与 P4 各自按 24 个任务有放回重采样 10,000 次，每次取两 panel rho 的等权平均。combined 和 magnitude 必须使用同一组重采样索引，以计算配对 `Δrho = rho_combined − rho_magnitude`。区间为 bootstrap percentile 95% CI；10,000 次中有效相关估计少于 9,500 时判实现失败。外层 Pearson 相关只作敏感性分析。

P3/P4 使用完全相同的 shared development 数据、种子和训练协议，只是 48 个不相交 test task 的固定 SHA 分区，因此不得将两面板写成两个独立研究复现。冻结的等权宏平均仍是主分析；另对全部 48 任务做 pooled task bootstrap 敏感性，用来显示人为分区对结果的影响，不参与预注册 gate。

### 4.2 次要与敏感性终点

- `frac_correct_direction_all`：2,044 个基因中预测效应与真实效应严格同号（乘积 `>0`）的比例，越高越好；恰为 0 的基因计为未命中。
- `frac_correct_direction_top20_de`：按本任务 PCA10-reconstructed 真实效应绝对值选 top 20 后计算同号率，只作补充；该基因集不参与分数或任务选择。
- `rmse_effect_error`：预测效应与 PCA10-reconstructed 真实效应的 RMSE，越低越好。
- raw-truth sensitivity：用同一预测效应对每任务未经 PCA 投影的固定归一化平均表达重算 Pearson、direction 和 RMSE。它不替换主终点。
- 组合置信度与 direction 的预期相关方向为正，与 RMSE 的预期方向为负。

## 5. 固定 coverage 分析

coverage grid 为 50%、55%、60%、65%、70%、75%、80%、85%、90%、95%、100%。每个 panel 按分数从高到低保留 `floor(24×coverage)` 个任务（最少 2 个），计算保留集平均 Pearson accuracy；双 panel 结果为两个保留集平均值的等权宏平均。combined 与 magnitude 使用同一 bootstrap 任务索引，输出每个 coverage 的配对差值及 10,000 次 percentile 95% CI。90% 和 95% 是预先指定的重点 coverage，其余档位完整报告。

同分时使用 E157 task-score CSV 的原始行顺序作 stable tie-break；不使用真值破平局。

## 6. 预注册判定

1. **论文口径信号确认**：P3 和 P4 的 combined→Pearson Spearman 都 `>0`，且双 panel 等权宏平均 bootstrap 95% CI 下界 `>0`。
2. **增量优势确认**：只有第 1 条通过，且 combined 相对 magnitude 的宏平均配对 `Δrho` 95% CI 下界 `>0`时通过。
3. **方向生物学确认**：P3 和 P4 的 combined→`frac_correct_direction_all` Spearman 都 `>0`，且宏平均 CI 下界 `>0`。

第 1 条通过而第 2 条不通过时，只能称“P3/P4 有置信度排序信号”，不能称“超越 magnitude”或“提供独立增益”。第 1 条不通过时，P3/P4 主复现失败；不得用 raw sensitivity、RMSE、top20 DE、单 panel、coverage 某一档或 reserve 改写该判定。

## 7. 失败、重跑与完整留存

下列任一情况先判实现失败：锁定哈希不符；任务缺失/重复；基因顺序不符；测试细胞计数与 E155 冻结元数据不符；非有限预测/终点；重算 magnitude 与 E157 锁定值不符；或 10,000 次 bootstrap 有效样本不足。每次运行固定写入 `attempt_001`、`attempt_002`…的首个合法追加目录；完成的 attempt 禁止再跑，未结束的 attempt 禁止覆盖。可在不改动预注册对象的前提下修复纯实现错误，但必须保留原 `UNSEAL_EVENT.json`、失败 STATUS 和 traceback，然后建立下一个 attempt，不得覆盖。

正常完成时必须保存：48 任务指标，panel/宏平均相关及 CI，combined-vs-magnitude 配对差，10,000 次主 bootstrap draws，全 coverage 曲线及配对 CI，raw-truth sensitivity，输入/输出哈希清单，解封时间和最终判定。

## 8. 固定运行接口

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e158_prescribe_norman_p3p4_forward_evaluation.py
```

本 runner 不提供改任务、换终点、换种子、降低 bootstrap 次数或改 coverage 的 CLI 参数。
