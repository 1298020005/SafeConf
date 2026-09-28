# E190 GEARS 跨家族开发实验合同

状态：在读取本实验的 SafeConf-v4 分数前登记。E190 的旧上游误差与旧风险摘要已经开封，因此本实验只能作为 **SEEN / exploratory cross-family development evidence**，不能作为 SEALED confirmation，也不能改变已经冻结的 Final Candidate。

## 1. 纳入依据

- 上游：E190 pretruth release 中三个 GEARS seed 的等权 centroid；
- 选择 centroid 的依据：它在目标 truth 建立前已经由三 seed release 唯一定义，不根据 SafeConf 涨幅选择；
- 目标：Adamson 训练、Replogle K562 迁移；
- 任务：692 个 batch×gene task，47 个 gene cluster，48 个 target batch；
- strongest simple baseline：冻结的 Adamson source effect；
- competence rule：相对 macro RMSE 不劣于 baseline 2%，至少 60% batch stratum non-inferior，gene-cluster bootstrap 不支持稳定劣势。

只有 competence gate 先通过，才允许运行下面的 SafeConf 实验。

## 2. Prediction/Error Contract

- output contract：`E190_GEARS_direct_effect_512gene_v1`；
- prediction effect：三个 GEARS pretruth direct-effect vector 的等权平均；
- true effect：Replogle observed treated pseudobulk − matched batch control；
- gene space：E190 冻结的 512-gene ordered panel；
- task error：aligned effect vector 上的 RMSE；
- 所有 target truth 仅用于构造 risk label 和评价，不进入 Prediction Evidence 或 History Evidence。

## 3. 历史证据合同

历史只来自独立 Adamson source study：

- Support：source cell 数、source context 数、guide 数、最小 fold cell 数；
- Relevance：GEARS prediction 与冻结 Adamson gene effect 的 cosine，以及负 RMSE gap；
- Conflict proxy：Adamson pseudobulk fold effect 的 RMS dispersion；
- Content：Adamson source gene effect magnitude；
- Quality：不进入正式字段，因为只有一个 source study/context，不满足 v4 的 Quality 资格门。

不得使用 Replogle target task、target replicate、target batch statistic 或 target truth 派生量作为历史。

## 4. 划分与模型

- outer split：5-fold GroupKFold，group=`gene`；同一 gene 的所有 target batch 必须位于同一 fold；
- inner split：V2 使用 4-fold gene-group cross-fitting；
- V1：冻结的 `Ridge(Universal P + Support + Relevance), alpha=10`；
- V2：冻结的 nested evidence shrinkage 实现，不改变字段方向、校准或 gate 形式；
- 比较：Magnitude、Universal P、P+Support、V1、Conflict/Content 增量、HGB、small MLP、Nested V2；
- Primary：Utility@20；同时报告 Spearman、AURC、risk@10/20/50 和 high-risk miss rate；
- bootstrap：5000 次，以 gene 为 paired cluster。

## 5. 解释边界

- 正结果可支持“SafeConf 信号跨 TxPert→GEARS 家族复现”；
- 因 E190 已经开封，正结果不能支持独立 confirmation；
- 因 source 与 target 都是 K562，不能支持跨 cell-context 泛化；
- 若只有 prediction-only 有效而 history 无增量，应完整保留并将跨家族主张收缩到 Universal Prediction Evidence；
- 本实验结果不允许反向修改 Frozen V1、Frozen V2 或 SEALED confirmation 规则。
