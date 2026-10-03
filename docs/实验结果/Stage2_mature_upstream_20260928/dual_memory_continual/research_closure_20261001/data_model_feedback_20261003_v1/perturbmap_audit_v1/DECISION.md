# PerturbMap 资格审计决定（2026-10-03）

## 决定

**REVISE BEFORE EXPERIMENT。**本轮只完成元数据资格审计，没有拟合映射、没有读取永久 TEST 真值、没有改变 SafeConf 主方法。

## 真实结果

- E201 TxPert 公共库：2008 条历史、4 个背景、3352 基因轴。
- E201 Source 任务：1808 个任务；1808 个任务都有同背景历史；自然 cross-only 查询数为 0；无历史查询数为 0。
- 因此，PerturbMap 在 E201 上不能作为“自然缺失同背景历史”的主实验。若人为隐藏同背景历史，只能作为压力测试，不能当成自然缺失证据。
- Orion TRAIN/VALIDATION 元数据包含 HCT116 和 HEK293T，合计 28945 个聚合单元。E201 历史到 Orion 目标背景的 8 条有向路线均有原始 gene-string 元数据配对身份；这些数量尚未通过 endpoint gene identity 合同，不能直接作为合法配对数：
  - K562→HCT116：445；K562→HEK293T：437；
  - RPE1→HCT116：360；RPE1→HEK293T：351；
  - hepg2→HCT116：375；hepg2→HEK293T：369；
  - jurkat→HCT116：374；jurkat→HEK293T：371。
- E201 3352 基因轴与 Orion endpoint 3285 轴的元数据交集为 3285；但 E201 公共库当前实际扰动目标与 Orion endpoint gene-name 字段的交集只有 228。必须先完成 gene identity 对齐，才能把原始配对数缩减为合法配对数；不能把“轴交集”或 raw gene-string overlap 写成“可用配对身份”。
- Orion 的 TRAIN/VALIDATION target-background 数值资产已存在，但本次审计没有用它们拟合映射或打开永久 TEST 真值。

## 采用范围

PerturbMap 若继续，只能作为：

> **利用 Orion TRAIN/VALIDATION 目标背景锚点，把其他背景真实响应转换为 HCT116/HEK293T 参照的回顾性 DEV 方法适配。**

它不应直接替换当前同背景支持加权历史距离。主比较必须保留：原始历史、目标背景均值、低秩投影/标量校正、真实配对岭映射、配对置乱，以及原公共距离和 Prediction-only。

风险采用必须看 Utility@20/AURC，而不能只看参照重建 MSE。映射拟合、基底、均值、正则选择和风险训练必须按扰动身份嵌套隔离；目标背景锚点属于公共生物监督，若能合法派生目标上游错误，PertEMA 对照必须在同一信息账本中获得它们。

## 暂不执行的原因

当前 SAMS-VAE 仍在唯一授权的新上游训练，PerturbMap 不应与其竞争 GPU 或同时扩展风险模型。等 SAMS 后处理和统一方法矩阵闭合后，若 Orion TRAIN/VALIDATION 资产仍是最有价值的缺口，再启动独立版本的 CPU 映射开发；否则保留本资格审计和不适用边界，不强行加入主论文。

证据：`PAIR_COVERAGE.csv`、`E201_QUERY_HISTORY_COVERAGE.csv`、`INFORMATION_BUDGET.csv`、`UPSTREAM_QUALIFICATION.csv`、`QUALIFICATION_RECEIPT.json`。
