# E201 任务底表与 source 证据结果

生成时间：2026-08-02

任务底表已在不读取对应 target 扰动表达、不打开模型预测的条件下完成。

| target | 主任务 | 敏感性任务 | target 细胞 | 1/2/3 个 source 背景 |
|---|---:|---:|---:|---:|
| K562 | 566 | 14 | 80,153 | 75 / 162 / 343 |
| RPE1 | 416 | 51 | 38,543 | 28 / 96 / 343 |
| HepG2 | 405 | 75 | 30,139 | 17 / 120 / 343 |
| Jurkat | 421 | 60 | 43,604 | 30 / 108 / 343 |
| 合计 | 1,808 | 200 | 192,439 | 150 / 486 / 1,372 |

四个 target 都有 343 个扰动在三个 source 背景中同时出现。单 source 任务分别
为 75、28、17 和 30 个；这些任务的跨背景 dispersion 使用各 target 的事前
中位数填补，数值依次为 0.046304、0.037080、0.046592 和 0.044643，并保留
缺失标记与背景缺口，不把填补值解释成真实观测。

source support 共 5,238 条 context–perturbation 记录。表达访问审计共 24 条：
四个 target × 三个 source context × control/perturbation 两类。对应 target 的
扰动表达访问数为 0。模型预测和目标误差均未打开。

2,008 条 source 等权平均 delta 已按任务表中的 `source_mean_delta_row` 固定顺序
写入数据盘，形状为 `2,008 × 3,352`、float32、全部有限，SHA-256 为
`b068834e49d73aa74cae54154d047c1897611526cb186c40d0dc9ec274a9141a`。
该向量以后只用于构造简单 source-transfer baseline 和计算模型–source gap。

任务表、source support 和访问审计分别位于：

- `tables/E201_PRETRUTH_TASK_BASE.csv`；
- `tables/E201_SOURCE_CONTEXT_SUPPORT.csv`；
- `tables/E201_SOURCE_EXPRESSION_ACCESS_AUDIT.csv`。
