# E200 正式结果评价冻结

冻结日期：2026-08-02

## 评价对象

- 目标：K562 整个细胞背景留出后的单基因 CRISPRi 扰动预测；
- 预测器：TxPert 官方 cross-cell GAT；
- 对照：官方 cross-cell general baseline 和 batch-matched K562 control；
- 主分析：566 个不少于 30 个目标细胞的严格 context-only 任务；
- 敏感性：14 个 10–29 细胞任务，单列报告，不参与主裁决。

## 误差与性能

主误差为每任务 prediction centroid 与 target centroid 的每基因 RMSE。GAT 分别与 general baseline 和 batch-matched control 做任务配对差；5,000 次任务 bootstrap 给出均值差 95% CI。差值定义为 `GAT error - baseline error`，小于 0 表示 GAT 更好。

不只看 MSE。使用冻结 scPertEval commit 计算：

1. `mse`；
2. `pearson_pert`；
3. `rank`；
4. `energy_distance_pca_k=50`；
5. `de_auprc`。

MSE、rank 和 energy distance 直接作为 error；Pearson 和 DE-AUPRC 转为 `1-score`。每个端点独立报告，不合成一个方便的总分。

## 风险路由

主裁决只使用结果打开前封存的 `transfer_risk`，简单对照为 `predicted_magnitude`。

- 关联：Spearman 相关及 5,000 次任务 bootstrap 95% CI；
- 复核预算：固定 20%，主分析选取 `ceil(566×20%)=114` 个风险最高任务；
- 路由效用：`(选中均值误差-总体均值误差)/(神谕 top-20% 均值误差-总体均值误差)`；
- 经验路由通过：`transfer_risk` 的 Spearman 和路由效用 95% CI 下限均大于 0；
- 相对 magnitude 的新增价值通过：配对 `ΔSpearman` 或 `Δ路由效用` 至少一个 95% CI 下限大于 0。

所有冻结原始分量也会报告与 GAT 误差的关联，用于解释失败或成功来源，不反向改动组合权重。五个 scPertEval 端点上的风险关联用于检查指标依赖，不替换主 RMSE 裁决。

## 模型归属

分别计算 `transfer_risk` 对 GAT 和 general baseline 误差的关联，并报告两个预测器的任务难度排名一致性。这些结果只能区分“当前 GAT 专属”和“两预测器共享”的难度信号，不会被包装成多模型家族不确定性。

## 完整性与边界

- 正式程序、本冻结文件和 pretruth release 必须已提交，且本地、GitHub、Gitee 顶端一致；
- 正式程序重新校验原始预测、真值、特征和外部评价源码哈希；
- 预真值特征重算残差必须小于 `1e-12`；否则整个评价 FAIL；
- 输出图统一白底，PNG 320 dpi 与 PDF 同时保存；
- E200 只回答 K562 作为目标背景的公开单 checkpoint 审计，不回答其他目标细胞系、多架构、跨独立数据集或湿实验因果验证。
