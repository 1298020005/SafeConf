# E204 冻结协议：用预测前风险信息指导训练

冻结日期：2026-08-30
状态：设计已冻结，等待 E201 盲预测链完成后执行

## 1. 为什么增加这条线

周老师在最近组会上明确指出，当前结果有两种潜在价值：

1. 预测完成后，提前找出更可能出错的任务，把有限的人工或湿实验复核优先放在这些任务上；
2. 训练模型时，给难预测任务更高的学习权重，观察模型是否因此减少这些任务上的误差。

E199/E200 已经说明第一种价值依赖测试场景：未见基因时分歧信号有额外信息，整行细胞背景留出时预测幅度更强，固定地把两者相加会变差。因此第二种价值不能把一个固定风险公式直接推广到所有场景。本实验把“风险指导训练”单独登记，在不读取 target 真值的前提下，与原始 TxPert-GAT 做成对比较。

## 2. 训练单位和信息边界

训练单位是 `cell line × perturbation condition`（细胞系 × 扰动条件）；一个条件下的多条单细胞记录只提供该任务的重复观测。权重在训练开始前按条件生成，再复制到该条件的每个训练细胞。

权重只允许使用 source（训练侧已经看到的细胞背景）信息：

- `n_source_cells`：该扰动在 source 中已有多少细胞观测；
- `n_source_contexts`：该扰动在多少个 source 细胞背景中出现；
- `source_delta_dispersion`：source 背景之间的扰动效应离散程度；
- 训练数据自身的细胞/批次标签。

以下内容在 target 真值释放前禁止进入权重：target 扰动表达、target 误差、target 预测结果、E199/E200 的结果表、调参后挑选出来的任务或种子。

## 3. 预先固定的权重

对每个 target 单独计算 source-only difficulty：

`difficulty = mean[z(-log(1+n_source_cells)), z(3-n_source_contexts), z(source_delta_dispersion)]`

其中 `z` 只在当前 target 的训练条件上计算；缺失的 source dispersion 用冻结的 source 中位数填补，并记录 `dispersion_imputed=true`。权重固定为：

`w = clip(1 + λ × difficulty, 0.5, 2.0)`，主 λ=`0.5`。

不根据 E201 结果选择 λ。λ=`0` 是原始训练基线，λ=`0.5` 是主风险指导训练，λ=`1.0` 只作预先登记的强度敏感性分析。每个 target 使用相同的四个随机种子；训练 epoch、优化器、模型结构、批量大小和数据切分与 E201 完全一致。

另设一个控制条件：只用 source effect dispersion 产生权重，不加入支持度和背景覆盖度。它用来判断改进来自“风险组合”还是仅来自一个容易获得的难度指标。

## 4. 比较和评价

每个 target 运行三种训练条件 × 四个种子：

| 条件 | 权重来源 | 作用 |
|---|---|---|
| `uniform` | 全部为 1 | E201 同协议基线 |
| `risk_weighted` | 支持度、背景覆盖、source dispersion | 主实验 |
| `dispersion_only` | 仅 source dispersion | 成分对照 |

先报告 source validation，再在 E201 真值释放后报告四个 target 的盲测结果。主要指标固定为：任务级 centroid RMSE、Pearson delta、rank、四种子分歧、最高风险 20% 的错误富集和固定复核预算收益。模型整体平均误差与高难任务误差同时报告，避免只展示被加权的子集。

主比较是每个 target 内、同一 seed 的配对差值：

- `risk_weighted - uniform` 的平均任务 RMSE；
- source-only difficulty 最高 20% 任务的 RMSE 差值；
- 全部任务的 Pearson delta 和 rank 差值。

跨 target 汇总按 perturbation condition 整簇 bootstrap，保留每个 target 的单独结果。若高难任务改善但全体任务明显恶化，结论记为“有针对性的取舍”，不写成总体提升。

## 5. 成功、失败和停止规则

- `SUPPORTED`：主 target 集合中，高难任务 RMSE 的预注册方向为改善，且全体任务平均 RMSE 不超过基线容差（5% 相对恶化）；四个 target 的方向与区间完整报告。
- `PARTIAL`：只在部分 target 或只在高难子集改善，整体指标不稳定；保留为条件性结果。
- `NOT_SUPPORTED`：高难任务没有改善，或整体误差超过容差；停止扩展权重公式。

任何条件都不能因为 E201 结果不理想而删 target、删 seed、改权重截断或改评价指标。E201 的 16 个模型必须先完成盲预测、预测前特征封存和哈希存档；E204 不能提前打开 target 真值。

## 6. 这项实验能回答什么

它能回答“SafeConf 的预测前线索是否能改变模型训练资源的分配，并在难任务上带来可重复的收益”。它不能单凭一次公开数据集证明跨实验室泛化，也不能替代新的生物机制或湿实验验证。结果无论正负都保留，因为周老师关心的正是哪些考法下应当使用、哪些考法下应当停止。
