# E198 事后饱和与适用性诊断

性质：`POSTRUN_DIAGNOSTIC`。本页只读取已封存的逐扰动表，不重开 arch1，不改变
E198 的门槛、12 个正式协议、主要端点选择或 11 个正式输出哈希。

## 1. 12 个协议全部通过意味着什么

它只说明在 arch1 的 150 个扰动上，这些协议能够稳定地区分技术重复与无信息参考。
它不说明 12 个指标同样有用，不说明某个模型会在这些指标上表现好，也不验证
SafeConf。

按冻结优先级，E199 的候选主要端点仍是：

- absolute：`mse`；
- direction：`pearson_pert`；
- retrieval：`rank`；
- population：`energy_distance_pca_k=50`；
- DE：`de_auprc`。

## 2. 群体距离和 rank 存在明显饱和

`energy_distance_pca_k=50` 有 92/150 个技术重复分数低于理论 perfect 0，
`unbiased_mmd_median_pca_k=50` 为 96/150。两者采用无偏或偏差修正估计，有限样本下
出现小幅负值是估计器性质；DRF 公式会把超过 perfect 的恢复量截到 1。因此 Energy
有 92 个、MMD 有 95 个扰动的 DRF 精确等于 1。这不能解释为技术重复“真正达到
完美距离”，只能解释为该校准量在上界饱和。

`rank` 的技术重复中位数为 0，负参考中位数约 0.520；146/150 个 DRF 大于等于
0.99。它很适合检查扰动身份能否检索回来，但同样不能代替绝对表达误差。

`wmse_exp2` 也有 120/150 个 DRF 大于等于 0.99。后续模型评价不能因为该校准分数
高就只报告 wMSE，仍保留事前优先的普通 MSE。

## 3. 相关指标的细节

未中心化 Pearson 的技术重复和负参考中位数都接近 1（0.99971 对 0.99895），说明
全局表达结构本身会造成很高相关。control-centred Pearson 的中位间隔更大；最终按
冻结优先级选择 `pearson_pert`，其 BDS 为 0.94，9 个失败扰动为 ANTXR1、CAMSAP2、
CREG1、DZIP3、LAD1、NCK2、RAB3B、ZNF286A 和 ZNF562。它比未中心化 Pearson 更
接近“扰动特异形状”，但不是每个基因都能稳定分开。

DE-AUPRC 的 DRF 中位数约 0.456，明显低于饱和的群体距离和 rank；这不代表它差，
而是 DEG 排名比区分整体群体更严格。DZIP3 和 PMS1 的正参考没有胜过负参考。

## 4. E199 的合法输出约束

评价协议通过校准，不等于任意预测器都能合法使用：

- `mse`、`pearson_pert` 和 `rank` 接受每个扰动一个合法 centroid；
- Energy 需要模型生成的真实预测细胞群；
- DE-AUPRC 也需要足够的预测细胞来计算差异表达；
- 如果 E199 的模型只输出均值向量，population 和 DE 两个轴必须记为
  `NOT_APPLICABLE_OUTPUT_CONTRACT`，不能复制均值伪造预测细胞。

因此 E198 冻结的是“数据允许哪些指标”，E199 还必须另行冻结“预测器实际提供什么
输出”。两份合同都满足，某个端点才可进入模型评价。

## 5. 独立复核

- 正式输出哈希 11/11 一致；
- CSV 重新读取后，DRF 独立复算最大差 `6.317724121629453e-13`；
- BDS 独立复算最大差 `1.1102230246251565e-16`；
- 复核没有重新打开 arch1。

逐协议饱和统计见
[E198_POSTRUN_SATURATION_AUDIT.csv](../tables/E198_POSTRUN_SATURATION_AUDIT.csv)。
