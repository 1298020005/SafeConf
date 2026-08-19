# E161a Wessels 表达轴边界核查合同

冻结日期：2026-07-15

## 为什么要做这一项核查

E161 首次正式运行只读取训练集后，在归一化的第一道硬门控中停止：前 20,639 列计数和与原始 `obs[ncounts]` 的最大差为 4。元数据表明第 20,632–20,639 列依次为 `eGFP`、`Blast`、`Cas9`、`Puro`、`Cas13d`、`AsCas12a`、`MeCP2`、`KRAB`，属于实验构造而非内源基因。E161a 在修改主分析前，专门判断 `ncounts` 对应的表达轴边界。

## 冻结问题

对训练行计算前 20,631、20,632、……、20,639 列的逐细胞计数和，哪一个前缀与原始 `obs[ncounts]` 完全一致？八个实验构造列各自贡献多少计数？

## 数据访问边界

- 使用 E160 冻结的行级切分和原始文件身份；
- 只允许读取 11,779 个训练细胞；
- 只允许物化前 20,639 列；
- validation、test 和 excluded 行的表达访问均为 0；
- 后 413 个 guide/feature-barcode 列的表达访问为 0；
- `obs`、`var_names` 与文件身份属于元数据检查，不视为表达解封。

## 候选边界与判断规则

候选边界固定为 20,631–20,639。对每个候选边界记录：

- 与 `obs[ncounts]` 不一致的训练细胞数；
- 最大绝对差、最小差、最大差和差值总和；
- 该边界末列名称。

“精确匹配”要求所有训练细胞差值均为 0。若多个前缀同时精确匹配，报告全部候选，不凭名称事后挑选；若没有候选精确匹配，E161 保持阻断并继续查找上游定义。E161a 本身不选择 HVG、不拟合 PCA、不训练模型，也不接触 validation/test truth。

## 固定轴身份

- 完整 21,052-feature SHA256：`dea725a87c973ca15590b08b309df3a926dc0233391cb2df76518c847229e780`
- 前 20,631 endogenous-feature SHA256：`dbed3dad178ea500b01625abf5121c9ee17bdd501b87d2fcdede0b6bade654e7`
- 8 个 engineered-construct SHA256：`103c2df8585646aa6dccde85866353889a699420b5536157b8babbd9b9aec554`
- 后 413 guide/barcode SHA256：`9088328f4ac6b2a1b109c254f0068504d25618478383fcbb3f43be8e59dd06d2`
- 后 421 全部候选排除列 SHA256：`e6e54ba5c0f63d62b599754ab3866da7cdf8194be4dfefd46dabc7d6a73e8116`

## 预定输出

```text
RUN_STATUS.json
CANDIDATE_BOUNDARY_AUDIT.csv
ENGINEERED_CONSTRUCT_COUNTS.csv
ACCESS_LEDGER.json
REPORT.md
RESULTS_SHA256.csv
```

所有结果保留原始正负结论。E161 只能依据这份冻结核查结果进行一次有记录的边界修订。
