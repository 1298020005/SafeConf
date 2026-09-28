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

## 能支持的结论

1. 在两个通过能力门的 TxPert 架构上，Magnitude 是强基线，V1/V2 的开发点估计均进一步提高 Utility@20 和 Spearman。
2. Support 后加入 Relevance，在两个架构上的 Spearman 增量均为正且簇自举区间下限大于 0；历史字段打乱后，V1 的 Spearman 明显下降。
3. 固定 Ridge 比 HGB 和小 MLP 稳定；增加复杂度没有收益。
4. V2 的 40 个 outer fits 中，32 个（80%）产生随任务变化的 evidence weight；K562 多数折退化成常数混合，必须保留为边界。
5. Quality 的合法 eligible coverage 为 **0.0%**。当前结果只能叫 Support/Relevance-aware，不能写成已经证明 Quality-aware。

## 尚未获得的证据

- Gate A/B 尚未通过：没有一个同时满足“未见、上游能力合格、V2 字段可用”的 sealed confirmation。
- 跨模型家族强主张尚未成立：54-task GEARS 只能作小样本压力线；E208/E247/E258 均在打开测试真值前被能力门阻断。
- 外部公共历史尚未验证；TxPert 的有效历史是同研究、目标背景排除后的 internal historical evidence。
- V2 相对 V1 的单架构 Utility@20 自举区间均跨 0；不能把开发门通过写成统计确认。

## 当前论文决策

本轮选择 **V2 作为冻结候选，不宣布最终方法已获外部确认**。现阶段最有价值的论文骨架是：

> 一个带上游能力门和输出合同的黑盒扰动预测风险审计框架；在合格 TxPert 上，Support/Relevance 历史证据和嵌套收缩提高开发期风险排序；在不合格上游上，系统拒绝生成看似漂亮但无部署意义的风险结论。

这已经形成真实的方法与结果包，但投稿前的最高优先级仍是一个能力合格的独立上游 family 或新的 sealed context。不能用已揭盲 E216、弱上游或重新切分看过的数据补这个空位。
