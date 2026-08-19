# E193 结果解释

- 输入：E190 692 个任务与 E192 175 个任务，共 867 个独特目标任务；
- 三种几何的 family RMS / worst lower violation：0 / 0；
- 最大平方恒等式残差：6.661e-16；
- 原始 RMSE 结果最大复算差：1.413e-08；
- 确定性实现结论：**PASS**。

- E190_K562 / absolute_rmse：diversity–family error ρ=0.424，基因整簇 95% CI [0.141, 0.631]；
- E190_K562 / cosine：diversity–family error ρ=0.568，基因整簇 95% CI [0.278, 0.783]；
- E190_K562 / pearson：diversity–family error ρ=0.245，基因整簇 95% CI [-0.277, 0.605]；
- E192_RPE1 / absolute_rmse：diversity–family error ρ=0.300，基因整簇 95% CI [-0.057, 0.584]；
- E192_RPE1 / cosine：diversity–family error ρ=0.048，基因整簇 95% CI [-0.160, 0.290]；
- E192_RPE1 / pearson：diversity–family error ρ=-0.210，基因整簇 95% CI [-0.507, 0.039]；

## 这次真正得到什么

确定性部分不依赖原始 RMSE。把每个成员先映射到 effect-vector cosine 或 Pearson
几何后，家族平方误差恒等式、diversity lower bound 和 diameter/2 worst-member
lower bound 仍然全部成立，2,601 个几何任务实例没有出现违例。这一结论只涉及注册
家族在指定 Hilbert 几何中的误差，不是模型正确概率。

经验排序没有跨 target 和 metric 运输：

- E190 K562 的 cosine diversity 有明确正相关，20% 复核
  utility=0.782，95% CI [0.479, 0.922]；
- E190 Pearson 的相关区间跨 0，diversity 的 20% utility=-0.026；
  source-to-family-centroid distance 在该 setting 的 utility=0.634，
  95% CI [0.344, 0.820]；
- E192 RPE1 的 cosine 相关接近 0，Pearson 相关为负；两种方向几何下 diversity
  的 20% utility 区间都跨 0。

所以 E193 加强的是 **metric-aware registered-family certificate**，没有产生一个
跨细胞系通用的方向型风险分数。方向型排序仍应按 setting 和 metric 单独冻结
`ACTIVATE/ABSTAIN`。E192 的原始 `ABSTAIN` 不变。

## 不能从 E193 推出的结论

- 不能称为 Systema exact replication；E193 没有构造其 control、post-state 与训练
  扰动质心参考空间；
- 不能声称 diversity 普遍优于 magnitude 或 source-shift；
- 不能把开真值后的相关结果称为独立确认；
- 不能用零违例替代与 PRESCRIBE、GEARS-UQ、CPA uncertainty 和简单线性基线的
  同协议实证比较。

这项结果回答“证书是否只在 RMSE 定义下成立”。它不回答“方向型排序是否已在未见真值上确认”；后者若要形成确认性主张，需要另一个在预测和分析冻结后才解封的外部数据块。
