# SafeConf v4 方法决策

## 决策

- Final Candidate：`V2_nested_evidence_shrinkage`。
- 冻结依据：合并开发 `ΔUtility@20(V2−V1)=0.009711`，非负 strata 比例 87.5%，有效 strata 100%。
- 证据等级：**同研究 held-out-donor/new-perturbation confirmation**。E170 一次性 Gate A 通过；Gate B 因无合法 Quality 字段不可评价。
- V1 保留为预指定 secondary benchmark，confirmation 失败后不得用同一结果把 V1 事后改封为主方法。

## 方法结构

1. `rP`：Universal Prediction Evidence 的 Ridge 风险。
2. `rPQ`：Prediction + Support + Relevance + Conflict proxy + Content 的 Ridge 风险。
3. 两个分数只在 inner OOF 上校准到相同误差尺度。
4. 单调低容量 gate 仅使用预测前 evidence，得到 `rP + w(rPQ-rP)`。
5. outer-test 只执行 outer-train 冻结的标准化、校准和 gate。

## 独立判断

当前真正成立的是 Support/Relevance-aware shrinkage；Quality 未进入正式实证。V2 的价值来自在历史模块整体不稳时学习收缩，而不是因为更大的网络。小 MLP 明显落后，因此不扩 Transformer 或 set encoder。

E170 在方法冻结后一次性打开全部四面板，V2 相对 Magnitude 的 ΔU20 为 +0.0504，10/12 strata 非负并通过所有安全护栏，因此 V2 升格为最终主方法。其 target-cluster bootstrap CI 跨 0，证据属于同研究新扰动/留出供体确认，不扩写为 external-study confirmation。E190 表明 V1 的 Support/Relevance 点增量可延伸到独立 GEARS family，但冻结 V2 没有复现，因此跨 family 仍按开发压力证据表述。

## Gate 机制复核后的负责人判断

在完全复用 frozen `rP/rPQ` 的条件下，learned gate 相对开发集最佳固定混合的 ΔU20 为：GAT `+0.0015`、Exphormer `-0.0035`、GEARS `-0.0390`，三者的 gene-cluster bootstrap 区间均跨 0；逐 fold 选择更优分支的正确率仅为 55%–60%。因此，**现有证据没有证明 learned gate 比简单固定混合更有价值**。

这项机制复核不修改已经冻结并完成 E170 确认的 v4，也不使用它调整外部数据或上游。投稿定位暂时调整为：

1. `Universal Prediction Evidence + legal Historical Experimental Evidence` 是主方法信号；
2. learned gate 是需要 McFaline 外部实验继续检验的条件扩展；
3. 只有外部 v4 同时超过 Magnitude 且不明显劣于 V1，才把 evidence-aware gate 保留为主方法贡献；
4. 若外部 V1 有效而 V2 低于 V1，则按 Route B 收口，不再为 gate 追逐正结果。

## Public History 与 Error Memory 正交复核

在相同 upstream、outer split 和 gene-cluster 错误标签预算下，`P / P+Public History / P+Error Memory / P+两者` 已完成比较。Public History 在 50%–100% 预算下对两种 TxPert 架构的 Spearman 增量稳定为正；Error Memory 的 Utility@20 增量没有一个预算获得正置信区间，加入 Public History 后也没有体现 U20 互补。

因此最终方法优先级固定为：

1. 主体：Universal Prediction Evidence + legal Historical Experimental Evidence；
2. 外部待检：冻结 v4 evidence-aware shrinkage；
3. 补充分析：Error Memory / PertEMA 反馈条件，不宣称 Error Memory 必然改善主指标。
