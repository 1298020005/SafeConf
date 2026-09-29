# SafeConf v4 Claim–Evidence Matrix

| 论文主张 | 证据 | 当前裁决 | 论文措辞 |
| --- | --- | --- | --- |
| 输出接口统一 | Prediction/Error Contract；direct-effect 与 treated-state 两路径 | 支持 | common black-box adapter contract |
| 幅度是强风险基线 | TxPert 两架构 U20 0.759–0.768，Spearman 0.728–0.742 | 支持 | strong mandatory baseline |
| Support/Relevance 有额外信息 | TxPert 开发增量；E170 history-available 组 U20 0.2383→0.2718/0.2857 | 同研究确认支持 | confirmed on new perturbations/held-out donor; external confirmation pending |
| V2 可安全收缩历史 | 开发门与 E170 Gate A 通过；但 learned gate 未稳定胜过固定混合，E190 跨家族失败 | 结果有效，机制贡献未独立成立 | same-study confirmation; learned gating remains a conditional extension pending external validation |
| Quality-aware | TxPert legal Quality coverage 0 | 不支持 | 不使用该主张 |
| External cross-study history | E190 使用 target-truth 前冻结的 Adamson history；V1 点增量为正但 CI 宽 | 初步支持 | cross-study historical evidence; not confirmed broadly |
| 跨结构 | TxPert GAT 与 Exphormer 同方向 | 支持 | cross-architecture within one family |
| 跨模型家族/model-agnostic | E190 GEARS 692 tasks 通过能力门；V1 点增量为正，V2 失败，且资产已开封 | 开发压力支持，未确认 | black-box interface with cross-family development evidence |
| 胜过 PertEMA | 同任务适配中 V2 高 0.066–0.069，CI 下限 >0 | 支持限定比较 | beats official-algorithm adaptation under registered contract |
| 独立 confirmation | E170 全四面板在方法冻结后同开；2,400 tasks；Gate A PASS | 同研究确认完成 | one-shot held-out-donor/new-perturbation confirmation, not external-study |
| Error Memory 部署增强 | gene-disjoint 反馈预算曲线；GAT 75–100%、Exphormer 50–75% 后改善 | 开发支持 | periodic batch adaptation; not online temporal learning |
| 化学通用性 | CPA 多数资产弱于简单基线 | 压力测试 | chemical stress test only |
