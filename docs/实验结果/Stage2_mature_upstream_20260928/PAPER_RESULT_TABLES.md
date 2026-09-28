# SafeConf v4 论文结果包

生成日期：2026-09-29。证据角色：TxPert 为开发数据；E208/E247/E258 测试真值保持封存；E216 为此前已经揭盲的负结果。

## 主结果

| 上游 | 任务 | Magnitude U20 | V1 U20 | V2 U20 | V2−Magnitude | V2 Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| TxPert GAT | 1808 | 0.7682 | 0.7864 | 0.7963 | 0.0281 | 0.7640 |
| TxPert Exphormer | 1808 | 0.7587 | 0.7823 | 0.7918 | 0.0331 | 0.7577 |

开发阶段的 Final Candidate 已按机器门冻结为 `V2_nested_evidence_shrinkage`。合并 8 个评价 strata 的 `V2−V1` 为 **0.0097**，87.5% strata 非负；所有安全退化门均通过。

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

## 能支持的结论

1. 在两个通过能力门的 TxPert 架构上，Magnitude 是强基线，V1/V2 的开发点估计均进一步提高 Utility@20 和 Spearman。
2. Support 后加入 Relevance，在两个 TxPert 架构上的 Spearman 增量均为正且簇自举区间下限大于 0；历史字段打乱后，V1 的 Spearman 明显下降。E190 GEARS 上同方向但区间宽。
3. 固定 Ridge 比 HGB 和小 MLP 稳定；增加复杂度没有收益。
4. V2 的 40 个 outer fits 中，32 个（80%）产生随任务变化的 evidence weight；K562 多数折退化成常数混合，必须保留为边界。
5. E190 提供一个能力合格的独立 GEARS family 和真正跨 study 的 Adamson history；V1 点估计超过 magnitude，但 V2 失败且 cluster CI 宽，因此它加强“跨家族可运行/有信号”，不构成确认。
6. Quality 的合法 eligible coverage 为 **0.0%**。当前结果只能叫 Support/Relevance-aware，不能写成已经证明 Quality-aware。

## 尚未获得的证据

- Gate A/B 尚未通过：没有一个同时满足“未见、上游能力合格、V2 字段可用”的 sealed confirmation。
- 跨模型家族 confirmation 尚未成立：E190 GEARS 是 692-task 已开封开发压力线；E208/E247/E258 均在打开测试真值前被能力门阻断。
- 外部历史在 E190 中合法存在且有开发点增量，但只有一个 source/target cell context，区间不足以支持广泛 external-public-history 主张。
- V2 相对 V1 的单架构 Utility@20 自举区间均跨 0；不能把开发门通过写成统计确认。

## 当前论文决策

本轮维持 **V2 作为冻结候选，不因 E190 压力结果事后切换方法，也不宣布最终方法已获外部确认**。现阶段最有价值的论文骨架是：

> 一个带上游能力门和输出合同的黑盒扰动预测风险审计框架；在合格 TxPert 上，Support/Relevance 历史证据和嵌套收缩提高开发期风险排序；在不合格上游上，系统拒绝生成看似漂亮但无部署意义的风险结论。

E190 已补上能力合格的独立上游 family 开发线；投稿前的最高优先级收敛为新的 sealed family/context confirmation。不能用已揭盲 E190/E216、弱上游或重新切分看过的数据补这个空位。
