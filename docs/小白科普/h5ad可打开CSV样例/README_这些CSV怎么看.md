# 这些 CSV 怎么看

这个文件夹是把真实 McFarland h5ad 和后续 SafeConf 表格拆成小样例。

如果你的 CSV 插件显示 `column 1 / column 2`，先不要用那个插件看。请直接打开：

[CSV样例_浏览器查看.html](./CSV样例_浏览器查看.html)

这个 HTML 会先解释列名，再显示表格，不依赖 CSV 插件。

所有列名解释也单独整理成了：

[字段说明_所有CSV列名.csv](./字段说明_所有CSV列名.csv)

如果你想看中文表头版本，可以打开：

[02_obs细胞表_22RV1样例_中文表头.tsv](./02_obs细胞表_22RV1样例_中文表头.tsv)

注意：没有把完整 h5ad 全量转成 CSV。完整表达矩阵大约是：

```text
154,710 个细胞 × 32,738 个基因 ≈ 50.6 亿个格子
```

全量转 CSV 会非常大，也不适合学习。这里保留的是“看得懂结构”的样例。

## 推荐打开顺序

### 1. 先看 h5ad 里面有什么

- [00_h5ad内部结构说明.csv](./00_h5ad内部结构说明.csv)

看这个你会明白：

```text
obs = 细胞表
X = 表达矩阵
var = 基因表
```

### 2. 看原始细胞表 obs

- [01_obs细胞表_前30行.csv](./01_obs细胞表_前30行.csv)
- [02_obs细胞表_22RV1_control和Trametinib样例.csv](./02_obs细胞表_22RV1_control和Trametinib样例.csv)

重点看这些列：

| 列 | 意思 |
| --- | --- |
| `cell_barcode` | 单个细胞的编号 |
| `cell_line` | 细胞系，也就是实验背景 |
| `perturbation` | 加了什么药，或者是不是 control |
| `dose_value` | 剂量 |
| `time` | 处理时间 |
| `ncounts` | 这个细胞总表达量 |
| `ngenes` | 这个细胞检测到多少基因 |

### 3. 看基因表 var

- [03_var基因表_前30行.csv](./03_var基因表_前30行.csv)

重点看：

| 列 | 意思 |
| --- | --- |
| `gene_name` | 基因名 |
| `ensembl_id` | 基因数据库 ID |
| `ncounts` | 这个基因总表达量 |
| `ncells` | 这个基因出现在哪些细胞里 |

### 4. 看表达矩阵 X 的小块

- [04_X表达矩阵小块_20个细胞x8个基因.csv](./04_X表达矩阵小块_20个细胞x8个基因.csv)

这个最重要。

它长这样：

```text
一行 = 一个细胞
一列 = 一个基因
格子里的数字 = 这个细胞里这个基因的表达量
```

你会看到：

```text
22RV1 control 细胞
22RV1 Trametinib 细胞
RPLP1 / RPL41 / RPS2 / ...
```

### 5. 看很多细胞怎么变成 effect

- [05_从表达矩阵到effect_8个基因演示.csv](./05_从表达矩阵到effect_8个基因演示.csv)

重点看：

```text
effect = trametinib_mean - control_mean
```

比如：

```text
RPLP1:
105.857 - 141.354 = -35.497
```

这就是“药物让这个基因表达下降了多少”。

### 6. 看那句 22RV1 control / Trametinib 到底什么意思

- [06_22RV1_Trametinib这句话逐词解释.csv](./06_22RV1_Trametinib这句话逐词解释.csv)

一句话：

```text
22RV1 control: 158 cells
= 22RV1 这个细胞系里，没加药的对照组有 158 个单细胞

22RV1 Trametinib: 84 cells
= 22RV1 这个细胞系里，加 Trametinib 这个药的实验组有 84 个单细胞
```

### 7. 看预测任务怎么进入 SafeConf

- [07_切分表_这道题为什么是test.csv](./07_切分表_这道题为什么是test.csv)
- [08_PredictionRecord_单条预测记录.csv](./08_PredictionRecord_单条预测记录.csv)

这两张表告诉你：

```text
22RV1 × Trametinib 是 test 题
V0StrongBaseline 对它做了一次预测
真实误差 RMSE = 0.5776
```

### 8. 看向量和 RMSE

- [11_true_effect_vs_predicted_effect_前20个位置.csv](./11_true_effect_vs_predicted_effect_前20个位置.csv)
- [12_true_effect_vs_predicted_effect_误差最大30个位置.csv](./12_true_effect_vs_predicted_effect_误差最大30个位置.csv)

重点看：

| 列 | 意思 |
| --- | --- |
| `gene_index` | 第几个基因位置 |
| `true_effect` | 真实答案 |
| `predicted_effect` | 预测答案 |
| `difference_pred_minus_true` | 预测 - 真实 |
| `squared_error` | 差值平方 |

RMSE 就是所有 5000 个位置的 squared_error 求平均再开方。

### 9. 看 SafeConf 质检线索和分数

- [09_ConfidenceFeatures_单条质检线索.csv](./09_ConfidenceFeatures_单条质检线索.csv)
- [10_ConfidenceScores_单条记录的多个分数.csv](./10_ConfidenceScores_单条记录的多个分数.csv)

这里才进入 SafeConf。

`ConfidenceFeatures` 是“能不能信”的线索。

`ConfidenceScores` 是根据线索算出来的分数。

### 10. 看最后论文主表

- [13_FORMAL_MAIN_TABLE_7个数据集总成绩.csv](./13_FORMAL_MAIN_TABLE_7个数据集总成绩.csv)

这是最后论文主表级别结果，不是单个细胞、不是单条预测。
