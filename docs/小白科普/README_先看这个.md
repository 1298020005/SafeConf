# SafeConf 小白科普入口

这个目录只放稳定学习资料。它不是科研进展，也不是实验结果目录。

如果你把整个 `小白科普` 文件夹下载到本地，先打开：

[index.html](./index.html)

所有链接都是相对路径，只要不要拆散这个文件夹，本地也能打开。

## 最推荐顺序

| 顺序 | 打开 | 你会看懂什么 |
|---|---|---|
| 1 | [README_h5ad到底长什么样.md](./h5ad真实数据格式/README_h5ad到底长什么样.md) | h5ad、obs、var、X、effect 到底是什么 |
| 2 | [数据一页看懂 HTML](./h5ad真实数据格式/SafeConf_数据到底长什么样_一页看懂_20260608.html) | 用图看真实数据怎么变 |
| 3 | [CSV 浏览器查看](./h5ad可打开CSV样例/CSV样例_浏览器查看.html) | 不用 CSV 插件，直接看表格 |
| 4 | [README_这些CSV怎么看.md](./h5ad可打开CSV样例/README_这些CSV怎么看.md) | 每张 CSV 表该怎么看 |
| 5 | [真实数据实物拆解](./从h5ad到effect_真实数据实物拆解_20260608.md) | h5ad 怎么一步步变成 effect |
| 6 | [整体架构流程图](./SafeConf整体架构流程图_20260608.html) | SafeConf 整体怎么跑 |
| 7 | [全流程拆解](./全流程拆解_创造者到使用者/SafeConf_全流程拆解_创造者到使用者.html) | 创造者怎么造，使用者怎么用 |
| 8 | [组会逐页讲解](./组会逐页讲解_20260610/README_先看这个.md) | 准备讲给别人听时看 |

## 一句话主线

```text
h5ad 原始单细胞数据
  -> obs 细胞表 + var 基因表 + X 表达矩阵
  -> control 组平均表达、perturbation 组平均表达
  -> true_effect = perturbation_mean - control_mean
  -> predictor 给 predicted_effect
  -> true_error_rmse 衡量预测错多少
  -> SafeConf 用历史线索给这条预测打风险分
```

## 最关键的三个词

| 词 | 人话 |
|---|---|
| `true_effect` | 实验真正测出来的扰动影响，一串基因表达变化数字 |
| `predicted_effect` | 预测器猜出来的扰动影响 |
| `true_error_rmse` | 预测和真实答案差多远，越大越不靠谱 |

## 目录结构

```text
小白科普/
├── index.html
├── README_先看这个.md
├── SafeConf_终版学习手册.html
├── SafeConf_终版学习手册.md
├── h5ad真实数据格式/
├── h5ad可打开CSV样例/
├── 从h5ad到effect_真实数据实物拆解_20260608.md
├── SafeConf整体架构流程图_20260608.html
├── 全流程拆解_创造者到使用者/
├── 组会逐页讲解_20260610/
└── assets/
```
