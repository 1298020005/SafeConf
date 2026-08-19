# h5ad 到底长什么样

先记住一句话：

```text
h5ad = 单细胞实验数据包
```

它不是一张普通表，而是一个“装了好几张表和一个大矩阵”的文件。

## 1. h5ad 里面最重要的三块

```text
h5ad
├── obs：细胞表
│   一行 = 一个细胞
│   记录：这个细胞来自哪个 context、做了什么 perturbation、是不是 control
│
├── var：基因表
│   一行 = 一个基因
│   记录：基因名、基因 ID、这个基因出现在哪些细胞里
│
└── X：表达矩阵
    行 = 细胞
    列 = 基因
    格子里的数字 = 这个细胞里这个基因的表达量
```

对应真实样例：

| 你要看什么 | 打开 |
|---|---|
| h5ad 内部结构 | [00_h5ad内部结构说明.csv](../h5ad可打开CSV样例/00_h5ad内部结构说明.csv) |
| obs 细胞表 | [01_obs细胞表_前30行.csv](../h5ad可打开CSV样例/01_obs细胞表_前30行.csv) |
| var 基因表 | [03_var基因表_前30行.csv](../h5ad可打开CSV样例/03_var基因表_前30行.csv) |
| X 表达矩阵小块 | [04_X表达矩阵小块_20个细胞x8个基因.csv](../h5ad可打开CSV样例/04_X表达矩阵小块_20个细胞x8个基因.csv) |

## 2. X 矩阵长什么样

可以想成这样：

```text
              gene_A   gene_B   gene_C   gene_D
cell_001        10       8        3        6
cell_002        11       7        4        5
cell_003        13       9        2        6
...
```

含义：

```text
cell_001 这个细胞里，gene_A 的表达量是 10
cell_001 这个细胞里，gene_B 的表达量是 8
```

真实项目里不是 4 个基因，而是成千上万个基因。

## 3. true_effect 怎么从 h5ad 算出来

SafeConf 关心的是扰动造成的变化。

```text
control 细胞：没加药 / 没敲基因
perturbed 细胞：加了药 / 敲了基因
```

计算流程：

```text
同一个 context 里

control 细胞们
  ↓ 求每个基因平均表达
control_mean

perturbed 细胞们
  ↓ 求每个基因平均表达
perturbed_mean

true_effect = perturbed_mean - control_mean
```

小例子：

```text
control_mean   = [10,  8,  3,  6]
perturbed_mean = [13,  7,  4, 10]

true_effect    = [ 3, -1,  1,  4]
```

意思：

```text
第 1 个基因升高 3
第 2 个基因降低 1
第 3 个基因升高 1
第 4 个基因升高 4
```

真实样例：

[05_从表达矩阵到effect_8个基因演示.csv](../h5ad可打开CSV样例/05_从表达矩阵到effect_8个基因演示.csv)

## 4. 预测器预测的是什么

预测器不是预测“这个药有效/没效”一句话。

它预测的是一个向量：

```text
predicted_effect = [基因1变化, 基因2变化, 基因3变化, ...]
```

真实答案也是一个向量：

```text
true_effect = [基因1真实变化, 基因2真实变化, 基因3真实变化, ...]
```

误差就是两条向量差多远：

```text
true_error_rmse = RMSE(predicted_effect, true_effect)
```

看具体向量：

| 你要看什么 | 打开 |
|---|---|
| 前 20 个基因位置 | [11_true_effect_vs_predicted_effect_前20个位置.csv](../h5ad可打开CSV样例/11_true_effect_vs_predicted_effect_前20个位置.csv) |
| 误差最大的 30 个位置 | [12_true_effect_vs_predicted_effect_误差最大30个位置.csv](../h5ad可打开CSV样例/12_true_effect_vs_predicted_effect_误差最大30个位置.csv) |

## 5. 最短流程图

```text
h5ad
  ↓
obs 告诉我们：哪个细胞是 control，哪个细胞被扰动
X 告诉我们：每个细胞每个基因表达多少
  ↓
control_mean
perturbed_mean
  ↓
true_effect
  ↓
预测器给 predicted_effect
  ↓
true_error_rmse = 预测向量和真实向量的距离
  ↓
SafeConf 判断：这条预测风险高不高
```

