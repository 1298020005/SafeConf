# SafeConf E1-E4 预注册实验契约

日期：2026-06-14

## 固定输入

- 输入矩阵：`LOPO_FEATURE_MATRIX_PertMeanPredictor.csv`
- 总行数：68,775
- 数据集：7 个正式主表数据集
- 外层切分：每个数据集 5 folds
- 训练预测器：`V0StrongBaseline`、`ContextSimBaseline`
- 留出预测器：`PertMeanPredictor`
- frozen protocol v0.2：不修改

所有 learned-model 实验必须按 outer fold 独立拟合：仅使用当前 fold 的
V0/ContextSim train+val 行，预测同一 fold 的 PertMean test 行。禁止将其他 fold
中同一 `task_key` 的误差标签并入训练。执行时必须断言 source 与 target 的
`(dataset_name, task_key)` 重叠为 0。

大型逐行分数和置换零分布只保存在
`/home/yyf/safeconf_runtime/outputs/`。本目录只保存可审计的小表、状态和报告。

## E1：六组 learned-risk 配对消融

六组特征固定为：

1. context：2 个
2. support：1 个
3. historical：3 个
4. disagreement：2 个
5. OOD：2 个
6. prediction output：4 个

总计 14 个特征。分别在 LOPO 与 LODO×LOPO 下运行 full 和
drop-one-group。按 `task_key` 做 1,000 次配对 cluster bootstrap。

Gate：至少两个特征组，各自在不少于 3/7 数据集上的
`full - drop_group` partial rho 95% CI 下界大于 0。该 gate 是跨数据集一致性
审计，未进行组间多重比较校正。

## E2：幅度校准后的剩余信号

仅做 dataset-local PertMean LOPO：

- 每个外层 fold 只使用本数据集 V0/ContextSim 的 train+val 行。
- 幅度标准化只使用上述训练行的 median/IQR。
- 内层 `GroupKFold(n_splits=3)` 按 `task_key` 分组。
- 主校准器为 isotonic；natural cubic spline df=4 仅作敏感性分析。
- HistGBT 预测 OOF magnitude residual，单位保持为 RMSE。
- PertMean test error 不参与训练或校准。

按 `task_key` 做 1,000 次 bootstrap。

Gate：

- residual partial rho 95% CI 下界大于 0：至少 4/7；
- `AURC(magnitude) - AURC(combined)` 95% CI 下界大于 0：至少 4/7。

## E3：伪信号和缺失模式负对照

运行 200 次置换：

- shuffled target：只置换训练任务，V0/ContextSim 同任务误差对成对移动；
- shuffled feature：在 `(dataset, fold, predictor)` 内整行联合置换 14 个特征；
- test label 和 test feature 均不置换。

缺失模式三路比较：

- current full values；
- missingness-only；
- full values + missingness indicators。

经验 p 值使用上尾检验，并在每类比较的 7 个数据集间做 BH FDR。

### 执行前统计方向更正

原始转述中的“真实模型相对置换零分布应有 `p > 0.10`”方向错误。若真实模型
确实优于随机置换，经验 p 应较小。正式 gate 固定为：

- observed vs shuffled-target：BH q < 0.05 至少 5/7；
- observed vs shuffled-feature：BH q < 0.05 至少 5/7；
- missingness-only：不得有数据集 BH q < 0.05；
- full+missingness 相比 current full 的 paired partial-rho CI 显著为正不得达到
  3/7，否则进入缺失模式诊断。

若 missingness-only 分数为常数，相关和经验 p 记作 `NaN/not testable`，不得
误记为显著。

## E4：模型稳定性

- HistGBT：10 个 seeds；
- 模型配置：5 个 HistGBT 配置 + 1 个 ElasticNet；
- 指标：PertMean LOPO partial rho。

Gate：

- 至少 5/7 数据集在 10 个 seeds 中有不少于 9 个为正；
- 至少 5/7 数据集在 6 个配置中有不少于 3 个为正。

HistGBT 接近确定性，因此配置敏感性比 seed 差异更有解释价值。

## E8：scPerturBench 只读资源审计

只审计官方仓库、网站和下载清单，确认是否公开：

- 逐任务预测向量；
- 与真实效应可对齐的 task/context/perturbation 标识；
- 可复用 split 和 predictor 名；
- 资源体积与许可证。

此阶段不下载大文件、不训练外部模型。只有资源结构通过后，才另行设计跨架构实验。
