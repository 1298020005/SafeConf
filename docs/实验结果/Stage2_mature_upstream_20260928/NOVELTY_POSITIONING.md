# SafeConf v4 创新定位

## 与 PertEMA 的关系

| 能力 | SafeConf v4 | PertEMA |
| --- | --- | --- |
| 黑盒预测器后置层 | 是 | 是 |
| 预测侧特征 | Universal P | prediction-time features |
| 历史实验 Support/Relevance | 是 | 官方核心不要求 |
| 历史 Conflict/Missingness 收缩 | V2 候选 | 无同一机制 |
| 无同任务 truth 推理 | 是 | 是 |
| 训练风险层需要 OOF error | 是 | 是 |
| isotonic/conformal interval | 本版不作为核心输出 | 是 |
| selective review/risk-coverage | 是 | 是 |
| 模型错误反馈 | 扩展线 | OOF error 是核心监督来源 |

SafeConf 不能把“黑盒后置可靠性”或“GBDT/Ridge 预测误差”写成首创。当前可辨识的方法差异是：**把合法历史证据拆成 Support、Relevance、Conflict 与 Missingness，并以嵌套、单调、低容量收缩决定历史修正的使用强度；上游能力门失败时拒绝路由。**

## 仍需防守的审稿问题

1. 当前 Quality 没有实证，名称必须收缩。
2. GAT/Exphormer 同属 TxPert，不能据此声称广泛 model-agnostic。
3. PertEMA 是官方算法适配，不是其 CD4 冻结模型复现；比较合同要在正文写清。
4. E170 的 V2−Magnitude bootstrap CI 跨 0，且 column-unseen 子集接近无信息；应完整报告，不把 Gate A PASS 写成统计显著或无历史普适性。
5. E170 是同研究确认；external-study、跨独立 family 的强 model-agnostic 主张仍需额外证据。
6. 旧 SafeConf 证书线与本次历史风险线是不同研究对象，稿件不能混成一个模糊贡献。
