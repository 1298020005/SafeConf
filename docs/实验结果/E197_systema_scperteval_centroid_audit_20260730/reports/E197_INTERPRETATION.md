# E197 结果解释

性质：`POSTRUN_INTERPRETATION`。本页只解释已经封存的 E197 表，不改变协议、数值、
E190/E192 原判定或 19 个正式输出哈希。

## 1. 周老师问的“风险到底和什么误差相关”

答案是：与误差定义有关，不能把一个相关系数当成普遍结论。

在 E190 K562 中，四个既有风险量与六成员平均 effect MSE 的 gene-equal Spearman
均为正。例如 predicted magnitude 为 `0.440`，基因重抽样区间
`[0.160, 0.672]`；diversity lower bound 为 `0.429`，区间
`[0.138, 0.675]`。高风险任务的绝对表达误差更大，这与原来的风险解释一致。

同一批任务换成 Systema-inspired transported Pearson error 后，方向变成负相关：
diversity、diameter/2、predicted magnitude 和 source magnitude 分别约为
`-0.687`、`-0.676`、`-0.685` 和 `-0.651`，对应区间均未跨 0。Pearson 关注形状，
MSE 同时受尺度影响；减去 source mean effect 后，较强扰动可能拥有较大的绝对误差，
同时保持较好的相关形状。因此两类端点回答的是不同问题，不能互换。

在 E192 RPE1 中，MSE 相关为 `0.112–0.201`，transported Pearson error 相关为
`-0.341–-0.321`，所有 gene-bootstrap 区间均跨 0。21 个基因不足以支持稳定的
跨细胞类型结论，应报告为未确认，而不是把点估计写成成功。

## 2. 模型差异与共同任务难度

E190 中，family centroid 的 scPertEval MSE 为 `0.0250`，低于 train-only matching
baseline 的 `0.0264` 和 zero-effect 的 `0.0315`；但 effect-space gene centroid
accuracy 为 `0.606`，低于 train-only matching 的 `0.728`。同一预测器在不同指标上
排序不同，说明“模型好坏”和“任务难度”不能由单一平均指标代替。

成员结果也不相同。E190 的三次 GEARS effect Pearson 约为 `0.326–0.333`，三次
scGPT 接近 0；E192 的总体 Pearson 很弱，scGPT 的 retrieval rank 还出现明显 seed
波动。当前证据支持“共享任务因素与模型特异失效同时存在”，不支持把分歧全部解释
成共同难度，也不支持把它全部归因于某一个模型。

## 3. 跨数据集结果的边界

E192 中 family centroid MSE 为 `0.0876`，zero-effect 为 `0.0894`，绝对改善很小；
retrieval rank 为 `0.167`，优于 zero-effect 的 `0.495`。这说明模型还能保留一部分
基因间相对检索信息，但表达幅度预测仍弱。K562→RPE1 不能只凭 rank 写成成功迁移。

train-only matching、all-fold matching、target-control 加 source mean、source absolute
和 zero-effect 均单列。预测器输入审计确认目标扰动表达没有进入预测或风险分数；
all-fold matching 只作为敏感性基线。现有结果只有 centroid，不能运行 population
distance、DE recovery 或复现性校准。上述缺口留给新的盲法 E198，不在 E197 中补写。

## 4. 可以向老师直接说的结论

“这次把误差定义拆开后，K562 上的风险分数能够排序绝对误差，但换成去共同响应后的
相关形状误差，方向会反过来；RPE1 的区间又跨过 0。风险信号不是对所有评价指标都
成立。下一步需要在新的盲法数据上预先固定主要误差、分层方式和失败门槛，再判断它
能否跨细胞类型复现。”
