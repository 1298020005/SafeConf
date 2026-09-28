# SafeConf v4 方法决策

## 决策

- Final Candidate：`V2_nested_evidence_shrinkage`。
- 冻结依据：合并开发 `ΔUtility@20(V2−V1)=0.009711`，非负 strata 比例 87.5%，有效 strata 100%。
- 证据等级：**开发候选**。Gate A/B 均等待合法 confirmation。
- V1 保留为预指定 secondary benchmark，confirmation 失败后不得用同一结果把 V1 事后改封为主方法。

## 方法结构

1. `rP`：Universal Prediction Evidence 的 Ridge 风险。
2. `rPQ`：Prediction + Support + Relevance + Conflict proxy + Content 的 Ridge 风险。
3. 两个分数只在 inner OOF 上校准到相同误差尺度。
4. 单调低容量 gate 仅使用预测前 evidence，得到 `rP + w(rPQ-rP)`。
5. outer-test 只执行 outer-train 冻结的标准化、校准和 gate。

## 独立判断

当前真正成立的是 Support/Relevance-aware shrinkage；Quality 未进入正式实证。V2 的价值来自在历史模块整体不稳时学习收缩，而不是因为更大的网络。小 MLP 明显落后，因此不扩 Transformer 或 set encoder。

正式投稿主张暂定为 `architecture-agnostic within TxPert`。取得独立 family 正结果后才升级为跨模型家族；取得 external history 增量后才使用 `public experimental history`。
