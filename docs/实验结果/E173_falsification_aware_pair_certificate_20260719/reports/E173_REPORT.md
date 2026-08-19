# E173｜E172 失败后的可证伪方法收缩

## 正式结论先写清楚

E168 的 200 个新目标未确认 SafeConf 相对 magnitude 的 AURC 增量；E172 又在完全不重叠的 800 个新目标上得到 `NO_TARGET_REPLICATION`。因此，固定 absolute-RMSE SafeConf 不能再写成稳定超过 predicted magnitude 的主方法。E171 validation 也没有提前支持 performance rescue，这条负证据不是统计偶然或事后挑面板可以解决的问题。

## 失败来自什么

同一 state、同一 target stratum 内，context similarity 与 support 均为常数。seen strata 的 15/15 个 panel×state 单元中，SafeConf 与 disagreement 的任务排序完全一致。固定公式在这些单元没有提供独立于模型分歧的新排序信息；巨大 context z 值只改变单元间位置，不改善单元内 AURC。

## 仍然成立的模型对证书

对任意两个预测向量 `p1,p2` 和未知真值 `y`，`d(p1,p2)/2` 同时下界 pair mean RMSE 与 pair max RMSE。E168+E172 共 1,000 个互斥目标、3,000 个任务的数值核验中，mean/max 下界违例均为 0；平方误差分解最大绝对残差为 7e-10。

分歧对 pair mean RMSE 的 15 单元等权 Spearman 为 0.117，target-cluster bootstrap 95% CI [0.061, 0.175]；对 pair max RMSE 为 0.141，CI [0.085, 0.198]。相对 magnitude 的 pair-mean Δrho 仅 0.005，CI [-0.025, 0.035]，不能宣称排序增量。

证书的价值不依赖相关性：当 `d/2 > tau` 时，可在没有目标真值的情况下证明两模型平均误差和至少一个模型误差超过 `tau`。它不能指出哪一个模型错，也不能把小分歧解释为安全。

## 修正后的系统定义

`SafeConf-Cert` 分成两个输出。第一层始终报告模型对下界并运行 RIAG 的预测/分数可识别性检查。第二层经验路由必须在 validation 上相对 magnitude 通过预先冻结的增量 gate；未通过就输出 `ABSTAIN_INCREMENTAL_ROUTING`，保留 certificate 和 magnitude 参考，不进入测试性能宣传。按这条规则，E171 会在 test truth 前拒绝 E172 的增量路由，而不会事后把失败解释成成功。

E173 是已解封数据上的方法收缩和二级审计，不是新的独立确认。下一项正式实验需要换 test donor，并使用全新目标；其主要对象应为模型对证书与 fail-closed 决策，而不是再次检验已被两次否定的固定公式优势。
