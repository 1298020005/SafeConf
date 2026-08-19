# E147 分析合同｜扰动身份检索

## 定位与冻结顺序

这是七个已经用于 SafeConf 开发和评价的数据集上的**新终点审计**，不是新增外部验证，也不是前瞻确认。冻结顺序为：本合同与部署分数快照 → 快照哈希 → 打开已保存的预测/真值向量 → 计算身份检索终点。不得根据检索结果重拟合 SafeConf、Directional-SafeConf 或改变候选池规则。

## 查询、候选池与正确答案

- 每个查询是一个测试任务的 scGPT 或 GEARS 预测效应向量。
- 候选库只含同一 `dataset × fold_id × context` 中的测试任务真值向量，不跨背景、数据集或折检索。
- 候选按 `perturbation` 唯一；同一候选池中若一个扰动对应多个不同任务，则该池作结构异常排除并记录，不事后挑选。
- 仅分析至少含 10 个唯一扰动的候选池。
- 正确答案是与查询任务具有相同 `task_id` 和 `perturbation` 的真值向量。

## 相似度与归一化秩

scGPT、GEARS 分别用两种相似度检索：

1. Pearson：预测和候选真值分别中心化后的余弦相似度；
2. cosine：未中心化向量的余弦相似度。

每个相似度下先排除范数不大于 `1e-12` 或含非有限值的候选真值；查询预测无效或正确候选被排除时，该查询记为无效。有效候选仍须不少于 10。相似度由低到高取平均秩（并列取平均），定义

`normalized_correct_rank = (rank_correct - 1) / (n_candidates - 1)`，

因此正确候选唯一最高时为 1，随机检索期望约为 0.5。主误差为 `retrieval_error = 1 - normalized_correct_rank`。同时报告 top-1、top-5、候选池大小和无效/排除数量。

## 风险关联与聚合

四个冻结的部署侧指标为：

- `directional_risk_frozen`；
- `safeconf_calibrated_pair_risk`；
- `baseline_predicted_magnitude`；
- `risk_model_disagreement`。

对每个 `predictor × similarity` 终点，先在 `dataset × fold_id` 内计算各指标与 `retrieval_error` 的 Spearman 相关，再对 fold 取均值得到 dataset 结果，最后对七个 dataset 等权平均。若某折有效查询少于 10 或任一变量无变异，该折相关记为缺失且完整报告。

## 不确定性

进行 3,000 次层级 cluster bootstrap。生物学任务簇键固定为 `dataset × context × perturbation`；一个簇中的 fold、预测器和相似度记录整体重复抽样。每次在各 dataset 内按其唯一任务簇数有放回抽样，按原 fold 规则重算相关，再对 dataset 等权平均。七个固定 dataset 不作有放回抽样，以免少数据集研究中改变目标总体；缺失 dataset 的抽样结果记为缺失，不通过静默降权掩盖。

## 判读规则

本审计不设用来宣称“独立验证”的通过门。预先指定的核心读数是 Directional-SafeConf 与四个 `predictor × similarity` 检索误差的七数据等权相关及其 95% cluster-bootstrap 区间；原 SafeConf、magnitude、disagreement 全部并列报告。正相关表示高风险任务更难从同背景测试候选中找回正确扰动身份。无论结果方向或显著性如何均保留，并明确检索终点与逐基因预测误差不是同一概念。
