# E216｜Jiang24 限时外部确认合同

冻结时间：2026-09-20（Jiang24 测试扰动表达仍未读取）

证据身份：独立的资源受限外部确认。它不替代 E208 的全量 PerturBench
合同，也不改变 E208 的主次结局。设立本实验的原因是全量 Jiang24
在当前硬件上每批读取和训练时间过长，无法在 2026-09-24 前生成外部
成绩单。资源设计只根据未读取测试真值的工程冒烟时间，不根据结果
选数据或参数。

## 1. 数据和任务

- 仍使用 Jiang24 官方 train/val/test 划分；
- 官方 222,759 个 test 细胞的集合和标签原样保留；
- 主评价仍为 E208 冻结的 12 个状态、224 个单基因任务，不丢弃任何
  细胞系、处理状态或扰动基因；
- 测试扰动表达只能在预测、风险表和文件哈希完全封存后打开。

## 2. 训练层资源限制

抽样只作用于 train 和 val，且只读取细胞条码和任务元数据：

- 每个 `split × cell_type × treatment × gene` 最多保留 16 个扰动细胞；
- 每个 `split × cell_type × treatment` 最多保留 128 个 control 细胞；
- 优先级是 `BLAKE2b("E216_JIANG24_RESOURCE_V1|" + cell_barcode)`，不使用表达值、
  不使用模型误差，也不人工挑细胞；
- 基因轴使用发布文件 `var/highly_variable` 中已标记的 4,000 个基因，
  不使用 E216 测试误差重选基因。

## 3. 上游预测器

主家族保持两种结构：

- LatentAdditive：种子 1、2、3、4，沿用 PerturBench Jiang24 已发布的层数、
  宽度、潜变量、dropout、学习率和权重衰减；
- LinearAdditive：种子 1，沿用已发布超参数；
- batch size = 2,000，`deterministic=true`；
- 最多 10 epochs，最少 5 epochs，验证损失 early stopping patience = 3；
- 每个完成的成员全部进入家族，不按测试结果删除成员或只留最好种子。

该家族用于检验 SafeConf 的预测后质检，不宣称重现 PerturBench 全量预测
排名。全量 E208 保留为后续完整实验。

## 4. 冻结评价

测试真值打开前一次性封存：五个成员预测、架构平衡家族质心、预测
幅度、同结构分歧、异构家族下界、原 SafeConf、固定 4:1 SafeConf-M 和随机
排序。全部指标在 4,000-HVG 轴上计算。

预测阶段对每个 `cell_type × treatment` 状态使用最多 128 个未扰动
control 细胞。control 按
`BLAKE2b("E216_JIANG24_PREDICTION_CONTROL_V1|" + cell_barcode)` 的固定优先级
抽取；同一状态、不同模型和不同扰动使用同一组 control。每个任务的细胞级预测
取逐基因算术平均，形成唯一的 4,000 维任务质心。该上限、细胞集合和聚合规则
在测试扰动表达打开前冻结，不按结果改变。

主结局不变：固定 4:1 SafeConf-M 相对预测幅度的任务级 Spearman 差，
按扰动基因做簇自举；95% 区间下限大于 0 才记外部确认。证书分析沿用
E208 的架构平衡权重、恒等式数值门和非退化门。

同时必须报告上游预测是否超过 no-change；若 LatentAdditive 和 LinearAdditive
均不超过 no-change，风险排序只记为资源受限负结果，不宣称有实际部署
价值。

## 5. 与全量 E208 的关系

E216 只回答“固定方法在一个新数据源的资源受限设定下能否重现”。
正文若使用 E216，必须明示细胞上限、4,000-HVG 轴和 10-epoch 上限。不得将它
写成全量 Jiang24 或 PerturBench 官方成绩。E208 的 15,473 基因、全 train/val 和
已冻结的官方超参队列继续保留。
