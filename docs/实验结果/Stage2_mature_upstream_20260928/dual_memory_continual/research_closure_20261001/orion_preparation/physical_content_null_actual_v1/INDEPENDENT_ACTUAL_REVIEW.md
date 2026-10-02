# 固定 Orion physical-content null：独立实际审阅

**结论：实际执行与配对统计核验 PASS；支持跨 MC/Orion 的窄描述性一致方向，不支持稳定增益、Core/Gate 修复或测量因果隔离。**

## 实际证据核验

- 实际回执 `COMPLETE_FIXED_SEEN_PHYSICAL_CONTENT_NULL`：235.777741 秒，232 tasks／144 genes，3 原规则×5 固定种子，378 metric rows／315 paired rows，0 fit／0 model inference／0 raw-expression read／0 new bootstrap draw。原三规则 bitwise 复算门已在正式运行通过。
- 本次独立只读核对回执所列 3 个 CSV 的 SHA/size、score/donor seal 的 SHA/size；315／378 行均无重复键。从已有 `5000×3×18×7` draw 数组逐项复现全部 **315 个配对 CI、378 个单方法 CI，精确一致**；315 配对项均有 5000 valid draws。每个配对点差也精确等于 actual−null。未重采样或重新计算 truth metrics。
- 实际 donor map 每种子均为 5365 行的完整一对一排列，5200 行移动；冻结 matching-group 字段及 recipient ID 全部保持，移动后的 target 均不同。query identities 和 score 行序仍为完整 232／144；3 个不可移动 query 未排除，它们全部种子／规则的分数与原值逐位相同。
- 实际 query 仅使用 338 个 unique recipient history records。独立复算其 donor/recipient cell-count ratio：96.4497–98.2249% 位于 `[0.8,1.25]`，99.7041–100% 位于 `[0.5,2]`，与保存的 support audit 精确一致。这是近似支持匹配，不是逐值 cell-count 或测量精度匹配。

本审阅只读取已有 summaries、sealed scores、donor metadata 和保存的 metric draws；未读取 cached error 数值、raw expression 或模型参数，也未运行正式器／新 RNG。执行代码仍为 `9dcfe3ee508da55058a2b64556d36311a2ea16f7b6c091c5e69f722204a04150`。

## 完整 macro U20 结果

表中均为 **真实对应−physical null** 点差和名义 95% paired gene-bootstrap CI。三个原 macro 点分别为 Uniform Direct `0.262120`、Manual Direct `0.244031`、Manual Weighted `0.186661`。

| 固定种子 | Uniform Direct | Manual Direct | Manual WeightedHistoryDistance |
|---|---|---|---|
| 20260930 | +0.211145 [-0.075537, +0.466576] | +0.251763 [-0.064881, +0.549066] | +0.092861 [-0.261342, +0.406105] |
| 20261001 | +0.348908 [+0.001912, +0.619209] | +0.336914 [+0.004091, +0.609942] | +0.225739 [-0.125647, +0.566380] |
| 20261002 | +0.142859 [-0.163499, +0.408378] | +0.337874 [-0.071527, +0.584705] | +0.007061 [-0.320109, +0.315629] |
| 20261003 | +0.357451 [+0.044121, +0.631498] | +0.420330 [+0.060701, +0.649853] | +0.213118 [-0.135002, +0.572421] |
| 20261004 | +0.131240 [-0.150691, +0.437003] | +0.347968 [-0.058381, +0.587828] | +0.188504 [-0.196703, +0.524728] |

15 个 macro 点差均正；仅 4 个名义 CI 下界大于零，分别为两种 Direct 规则的 `20261001`／`20261003`。Manual Weighted 全 5 个 CI 跨零。context 分开看，HEK293T 的全部 15 个 U20 CI 均跨零，且 Weighted 的两个点差为负。不能将 macro 的一致正点差升级为两个 context 均稳定获益。

五种子及三规则共享历史、query、gene draws，不能算 15 个独立复制；4 个名义正区间未做多重比较校正，不能挑选作为确认结论。宽 CI 既不证明无效，也不证明等价。当前区间条件于固定队列及各个 donor realization，不是对所有可能支持匹配方案的普遍保证。

## 跨 MC／Orion 的可支持边界

既有 MC `evidence_review_agent/cellweighted_content_followup/PAIRED_BOOTSTRAP.csv`：543 tasks／380 gene clusters，真实 cell-weighted 参照相对 5 个 content null 的 ΔU20 为 `+0.156162` 至 `+0.231432`，5 个名义区间均正；但其相对 NegativeHistorySupport 的增量仅 `+0.007185 [-0.039087, +0.055330]`。本次 Orion 在另一目标研究、不同上游家族上的固定参照负对照提供了方向相同、精度较弱的证据。两边的合同、匹配和参照不同，不合并 CI、不把种子或同家族模型算成独立研究。

**可用结论句：**“在已见 MC 与 Orion 固定队列上，保留 recipient 支持与权重、打乱匹配组内历史 gene-effect 对应关系，会降低人工／均匀参照的风险排序点估计。MC 证据较强，Orion 的不确定性仍大，尤其带历史不确定性项的加权距离尚无稳定增量证据。”

这为公共参照的效应对应关系提供了诊断证据。公共生物内容相对数量规则的稳定增量，以及支持／测量之外的跨研究独立作用，仍未建立。

## 不支持的升级

1. **Core／跨预测器错误监督：**本项只检验 3 个无拟合参照规则，未检验学习型 Public Biology、6 个 Source risk models、跨预测器错误监督或目标反馈的独立贡献。不能以其替代已失败的学习型风险主比较或宣称 Source transfer 成立。
2. **Gate／系统成熟度：**原 13 主比较、两个 context 安全性和全部冻结参数未改；正 null 点差不修复原 HCT 安全失败，不产生新确认资格或新 winner，也不证明持续更新单调获益。
3. **测量因果：**quartile／n_batches／control-contract 匹配保留了固定支持结构，但没有精确匹配 donor 估计方差、guide／生物重复、control uncertainty 或潜在生物真值。交换向量还会改变历史分散度及 weighted uncertainty；高比例 cell-count ratio 接近 1 不能消除这些差异。因此不能把排序下降完全归因于潜在生物信息损失。
4. **后续选择：**不挑正种子、不删 3 个不可移动 query、不重扫 caliper／grouping，不用本 SEEN 诊断更换方法或阈值。

## 精确审阅对象

- `RESULT_MANIFEST.json`：`ce6ae55d0cf69d6aaba236e5e51b0ec375b36f7b1f58994fb588826d48837cd0`。
- `ALL_ACTUAL_MINUS_PHYSICAL_NULL_PAIRED_INTERVALS.csv`：`24d5b682a90431037ba192ba01a6a1059c1da7384211628c80a1e04c9ff97dd5`。
- runtime `ALL_FIXED_ACTUAL_NULL_METRIC_DRAWS.npy`：`7d3b528a1d83427f297a67d28ba841340554bc5d34dde84b03bc086cf40aa399`。
- runtime `PHYSICAL_DONOR_ASSIGNMENTS.csv.gz`：`7eb135f38db24952baf6fa112e05f6b5b8348841187a885d416dcf51451cd336`。

封存时序措辞仍须准确：正式器封存前为完整性 SHA 流式读取 cached truth 文件 bytes；全部 scores seal 后才数值解析／使用 errors。不能表述为“封存前完全未打开 truth 文件”。本审阅未再次读取该 truth 文件。
