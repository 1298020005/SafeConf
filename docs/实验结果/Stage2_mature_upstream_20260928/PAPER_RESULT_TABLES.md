# SafeConf v4 论文结果包

生成日期：2026-09-29。证据角色：TxPert/E190 为开发或已见数据；E170 四面板在方法冻结、远端授权和字节级核验后一次性打开；E208/E247/E258 测试真值继续封存；E216 为此前已经揭盲的负结果。

## 主结果

| 上游 | 任务 | Magnitude U20 | V1 U20 | V2 U20 | V2−Magnitude | V2 Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| TxPert GAT | 1808 | 0.7682 | 0.7864 | 0.7963 | 0.0281 | 0.7640 |
| TxPert Exphormer | 1808 | 0.7587 | 0.7823 | 0.7918 | 0.0331 | 0.7577 |

开发阶段的 Final Candidate 已按机器门冻结为 `V2_nested_evidence_shrinkage`。合并 8 个评价 strata 的 `V2−V1` 为 **0.0097**，87.5% strata 非负；所有安全退化门均通过。

## 一次性确认：E170 四面板

| 方法 | Utility@20 | Spearman | AURC | risk@20 | 高风险漏检率 |
| --- | --- | --- | --- | --- | --- |
| Magnitude_raw | 0.1655 | 0.1734 | 0.1161 | 0.1121 | 0.7125 |
| Ridge_USR | 0.2010 | 0.1897 | 0.1137 | 0.1087 | 0.6979 |
| V2_nested | 0.2159 | 0.1890 | 0.1140 | 0.1089 | 0.6937 |

- 冻结 V2 相对 Magnitude：ΔU20=**+0.0504**，12 个预注册 panel×state strata 中 **83.3%** 非负，Gate A **PASS**。
- 成对 target-cluster bootstrap：ΔU20 95% CI **[-0.0266, +0.1136]**；区间跨 0，说明估计仍不精确，不能写成传统显著性确认。
- safety：risk@10/20 无恶化，risk@50 相对恶化 0.60%，AURC 和高风险漏检率均未恶化，全部通过预注册阈值。
- 证据等级：同研究内的新 perturbation + held-out donor confirmation；不是 external-study confirmation。
- 有合法两供体历史的 1,920 tasks：Magnitude/V1/V2 U20 为 **0.2383/0.2718/0.2857**。无历史的 480 column-unseen tasks 三者均接近 0；这是必须保留的适用边界，不把它写成成功的冷启动 fallback。

## PertEMA 公平比较

| 上游 | PertEMA* U20 | SafeConf V2 U20 | 差值 | 95% CI |
| --- | --- | --- | --- | --- |
| TxPert GAT | 0.7237 | 0.7963 | 0.0656 | [0.0256, 0.1074] |
| TxPert Exphormer | 0.7181 | 0.7918 | 0.0688 | [0.0251, 0.1126] |

`PertEMA*` 是官方算法与原超参数在同一 TxPert 任务、同一 gene-disjoint outer split、同一错误标签预算上的适配，不是把官方 CD4 冻结权重直接搬到 TxPert。SafeConf 多使用合法历史证据，因此这里检验的是“历史证据在同标签预算下能否提供增量”，不声称覆盖一切 PertEMA 特征工程。

## 独立 GEARS 家族压力结果

E190 在运行 SafeConf 前通过既定能力门：三 seed GEARS centroid 相对最强 Adamson source-effect baseline 的误差差为 **+0.115%**，48/48 batch strata 在 +2% margin 内，gene-cluster bootstrap 95% CI 为 **[-0.607%, +0.893%]**。该资产此前已开封，因此属于 SEEN cross-family development evidence。

| 方法 | Utility@20 | Spearman | AURC |
| --- | --- | --- | --- |
| Magnitude_raw | 0.2348 | 0.2388 | 0.2321 |
| Ridge_U | -0.0491 | 0.0902 | 0.2361 |
| Ridge_USR | 0.3035 | 0.2489 | 0.2265 |
| Ridge_USRC | 0.3894 | 0.3752 | 0.2228 |
| V2_nested | -0.0114 | 0.2030 | 0.2273 |

- V1−Magnitude：ΔU20=+0.0687，95% CI [-0.2586, +0.2853]。
- Relevance given Support：ΔU20=+0.3948；ΔSpearman=+0.1969。
- V2−V1：ΔU20=-0.3149。V2 没有跨家族复现 V1 的点增量，作为正式失败边界保留。
- Conflict proxy 带来 ΔSpearman=+0.1263，95% CI [+0.0294, +0.2343]；该信号来自单一 source study 的 fold dispersion，不能升级成完整 Quality 结论。

## Error Memory 部署增强

| 上游 | 反馈预算 | 平均反馈基因 | Utility@20 | Spearman |
| --- | --- | --- | --- | --- |
| TxPert_Exphormer | 0% | 0.0000 | 0.7514 | 0.7286 |
| TxPert_Exphormer | 10% | 36.7500 | 0.7241 | 0.6828 |
| TxPert_Exphormer | 25% | 90.8500 | 0.7619 | 0.7251 |
| TxPert_Exphormer | 50% | 181.1000 | 0.7688 | 0.7366 |
| TxPert_Exphormer | 75% | 271.3500 | 0.7681 | 0.7484 |
| TxPert_Exphormer | 100% | 361.6000 | 0.7643 | 0.7480 |
| TxPert_GAT | 0% | 0.0000 | 0.7615 | 0.7438 |
| TxPert_GAT | 10% | 36.7500 | 0.7432 | 0.7354 |
| TxPert_GAT | 25% | 90.8500 | 0.7600 | 0.7567 |
| TxPert_GAT | 50% | 181.1000 | 0.7569 | 0.7606 |
| TxPert_GAT | 75% | 271.3500 | 0.7699 | 0.7618 |
| TxPert_GAT | 100% | 361.6000 | 0.7759 | 0.7620 |

反馈按 gene cluster 从 outer-train 逐步提供，评价基因保持留出。10% 反馈在两个上游都下降；GAT 到 75–100% 后 U20 超过零反馈，Exphormer 在 50–75% 达到平台。该实验支持“按批次周期重训”的部署增强，不冒充真实时间顺序的在线学习。

## 化学扰动压力线

CPA 八个固定 sciPlex3 划分中，Magnitude / P+Q / P+Q+H 的 U20 为 **0.5816 / 0.5396 / 0.6069**。P+Q+H 相对 Magnitude 的 ΔU20 为 +0.0253，95% CI [-0.0952, +0.1590]。CPA 本身未通过上游能力门，因此该结果只证明统一接口可以运行并暴露剂量/来源异质性，定位为 chemical stress test。

## 能支持的结论

1. 在两个通过能力门的 TxPert 架构上，Magnitude 是强基线，V1/V2 的开发点估计均进一步提高 Utility@20 和 Spearman。
2. Support 后加入 Relevance，在两个 TxPert 架构上的 Spearman 增量均为正且簇自举区间下限大于 0；历史字段打乱后，V1 的 Spearman 明显下降。E190 GEARS 上同方向但区间宽。
3. 固定 Ridge 比 HGB 和小 MLP 稳定；增加复杂度没有收益。
4. V2 的 40 个 outer fits 中，32 个（80%）产生随任务变化的 evidence weight；K562 多数折退化成常数混合，必须保留为边界。
5. E170 的一次性确认满足预注册 Gate A，证明冻结候选不是只在 TxPert 开发集上成立；其统计区间和同研究证据等级必须完整报告。
6. E190 提供一个能力合格的独立 GEARS family 和真正跨 study 的 Adamson history；V1 点估计超过 magnitude，但 V2 失败且 cluster CI 宽，因此它加强“跨家族可运行/有信号”，不构成确认。
7. Error Memory 在足量同模型反馈后提供增量，但 10% 反馈不稳定，因此只作为周期重训扩展。
8. 化学接口已在 CPA 压力线上运行；CPA 未通过能力门，不能承担化学主确认。
9. Quality 的合法 eligible coverage 为 **0.0%**。当前结果只能叫 Support/Relevance-aware，不能写成已经证明 Quality-aware。

## 尚未获得的证据

- Gate B 无法评价：冻结时没有合法 Quality 字段，不能事后补入确认。
- external-study confirmation 尚未成立：E170 属于同研究 held-out donor；E190 GEARS 是 692-task 已开封开发压力线；E208/E247/E258 均在打开测试真值前被能力门阻断。
- 外部历史在 E190 中合法存在且有开发点增量，但只有一个 source/target cell context，区间不足以支持广泛 external-public-history 主张。
- V2 相对 V1 的单架构 Utility@20 自举区间均跨 0；不能把开发门通过写成统计确认。

## 当前论文决策

本轮维持 **V2 为最终主方法**：它在 E170 一次性确认中通过预注册 Gate A；没有在看到结果后切换到 V1。当前最有价值的论文骨架是：

> 一个带上游能力门和输出合同的黑盒扰动预测风险审计框架；在合格 TxPert 上开发 Support/Relevance 历史证据与嵌套收缩，在此前封存的 E170 新扰动/留出供体上一次性通过 Gate A，并对无历史与跨家族失效条件实行完整披露。

据此项目进入 **Route A：SafeConf 主方法论文**。投稿增强项是 external-study/cross-family confirmation，而不是重新选择方法；不能用已揭盲 E190/E216、弱上游或重新切分看过的数据伪装该证据。
