# SafeConf 终版学习手册

这个文件是 `docs/小白科普/` 的总入口。它不是论文证据，也不是实验报告，而是给自己从零理解 SafeConf 用的学习路线。

如果你把整个 `小白科普` 文件夹下载到本地，只要文件夹内部结构不拆散，下面所有链接都能直接打开。

## 先看哪一个

先打开浏览器完整学习书：

[SafeConf_终版学习手册.html](./SafeConf_终版学习手册.html)

它是现在唯一维护的完整阅读版。以前放在桌面的 HTML 已经收回到本目录，后续统一在 `safe-conf` 里迭代。

如果想看小白科普目录页，再打开：

[index.html](./index.html)

如果你在 Cursor 里看 Markdown，就从这个文件继续往下走。

## 一句话总图

SafeConf 的数据链路可以先记成这一条：

```text
h5ad 原始单细胞数据
  -> obs 细胞表 + var 基因表 + X 表达矩阵
  -> 每个 context × perturbation 的 effect 向量
  -> predictor 预测的 predicted_effect
  -> true_effect 和 predicted_effect 比误差
  -> PredictionRecord 记录每条预测
  -> ConfidenceFeatures 提取质检线索
  -> ConfidenceScores 给可信度/风险分数
  -> 论文表格和图
```

## 推荐学习顺序

0. [SafeConf_终版学习手册.html](./SafeConf_终版学习手册.html)

   先看这个完整浏览器版。它已经合并最新问答：为什么分数可以是负数、IQR/鲁棒性是什么、5000 基因是不是固定规矩。

1. [SafeConf_数据到底长什么样_一页看懂_20260608.html](./h5ad真实数据格式/SafeConf_数据到底长什么样_一页看懂_20260608.html)

   先看这个。它回答最基础的问题：h5ad 到底长什么样，矩阵、细胞表、基因表、向量分别是什么。

2. [CSV样例_浏览器查看.html](./h5ad可打开CSV样例/CSV样例_浏览器查看.html)

   如果你被 CSV 插件显示的 `column 1 / column 2` 搞晕，先看浏览器版。它把列名和表格一起展示出来。

3. [README_这些CSV怎么看.md](./h5ad可打开CSV样例/README_这些CSV怎么看.md)

   这里解释每个 CSV 文件应该怎么看。重点看 `obs`、`var`、`X`、`PredictionRecord`、`ConfidenceFeatures`、`ConfidenceScores`。

4. [SafeConf_真实数据实物拆解_20260608.md](./从h5ad到effect_真实数据实物拆解_20260608.md)

   这是慢慢拆数据的文字版。适合边看边打开 CSV，对照字段理解。

5. [SafeConf_小白架构流程图_20260608.html](./SafeConf整体架构流程图_20260608.html)

   等你知道数据是什么之后，再看整体架构，否则容易只记住框图、不知道每个框里装的是什么。

6. [SafeConf_全流程拆解_创造者到使用者.html](./全流程拆解_创造者到使用者/SafeConf_全流程拆解_创造者到使用者.html)

   这是 Qoder 昨天补的全流程版。它按“创造者怎么造 SafeConf”和“使用者怎么用 SafeConf”两条线讲，适合解决“我到底是在做系统、实验，还是论文工具”的混乱感。

7. [01_SafeConf项目全景_先建立脑图.md](./组会逐页讲解_20260610/01_SafeConf项目全景_先建立脑图.md)

   用来建立项目全景。适合你要向别人讲 SafeConf 前先过一遍。

8. [02_SafeConf科研与单细胞名词零基础词典.md](./组会逐页讲解_20260610/02_SafeConf科研与单细胞名词零基础词典.md)

   遇到不懂的词就查这里，比如 `perturbation`、`context`、`fold`、`RMSE`、`Spearman rho`。

9. [03_SafeConf组会PPT逐页逐图讲解.md](./组会逐页讲解_20260610/03_SafeConf组会PPT逐页逐图讲解.md)

   适合做组会或论文讲解前复习。

10. [04_汇报前十分钟速查.md](./组会逐页讲解_20260610/04_汇报前十分钟速查.md)

   最后十分钟速查用，不适合从零学习。

## 最关键的几个文件

| 你想看什么 | 打开哪个文件 |
|---|---|
| h5ad、矩阵、向量到底是什么 | [数据一页看懂 HTML](./h5ad真实数据格式/SafeConf_数据到底长什么样_一页看懂_20260608.html) |
| 真实 CSV 表格长什么样 | [CSV 浏览器版](./h5ad可打开CSV样例/CSV样例_浏览器查看.html) |
| 每个 CSV 列名什么意思 | [字段说明 CSV](./h5ad可打开CSV样例/字段说明_所有CSV列名.csv) |
| 单条预测记录是什么 | [PredictionRecord 样例](./h5ad可打开CSV样例/08_PredictionRecord_单条预测记录.csv) |
| 质检线索是什么 | [ConfidenceFeatures 样例](./h5ad可打开CSV样例/09_ConfidenceFeatures_单条质检线索.csv) |
| 分数是什么 | [ConfidenceScores 样例](./h5ad可打开CSV样例/10_ConfidenceScores_单条记录的多个分数.csv) |
| 7 个数据集总成绩是什么 | [FORMAL_MAIN_TABLE 样例](./h5ad可打开CSV样例/13_FORMAL_MAIN_TABLE_7个数据集总成绩.csv) |

## 怎么理解几个核心词

`cell`：一个细胞。单细胞数据里，每一行通常是一个细胞。

`gene`：一个基因。表达矩阵里，每一列通常是一个基因。

`X`：表达矩阵。可以先想成一个大表：行是细胞，列是基因，格子里是这个细胞里这个基因的表达量。

`obs`：细胞表。每一行解释一个细胞来自哪个细胞系、做了什么扰动、是不是 control。

`var`：基因表。每一行解释一个基因叫什么。

`context`：背景。SafeConf 里通常指细胞系、条件、病人、实验环境这类“在哪个背景下做预测”。

`perturbation`：扰动。比如敲掉一个基因，或者加一个药。

`effect`：扰动效果向量。简单说，就是“扰动组平均表达”减去“control 组平均表达”后得到的一串数字。

`predicted_effect`：预测器猜的 effect。

`true_effect`：真实实验算出来的 effect。

`true_error_rmse`：预测错得有多远。越大说明预测越不靠谱。

`PredictionRecord`：一条预测的档案。记录谁预测了什么、真实答案是什么、误差是多少。

`ConfidenceFeatures`：SafeConf 用来判断“这条预测靠不靠谱”的线索，比如历史支持次数、背景相似度、两个预测器分歧。

`ConfidenceScores`：最终分数。分数不是预测表达值，而是对预测可靠性的打分。

## 本次问答补丁：三个最容易卡住的问题

### 1. 为什么 SafeConf 分数会有负数？不能直接做成 0-1 吗？

可以做成 0-1 的“展示层”，但原始分数不建议直接写成 0-1。

原因很简单：SafeConf 当前做的是排序（ranking），不是概率判断（probability judgment）。它要回答的是：

```text
在这一批预测里，哪些更值得优先复核？
```

不是回答：

```text
这条预测有 82% 概率正确。
```

SafeConf 公式里用的是 z-score（标准化分数），中等水平附近自然是 0；比中等更可靠就是正数，比中等更危险就是负数。负数不等于“预测一定错”，只表示“它在这一批里相对更危险”。

更稳的理解是：

```text
原始 z-score：给公式和论文用，诚实表达相对位置。
百分位 / low-medium-high：给人阅读用，可以后处理展示。
```

### 2. IQR 和“鲁棒性”是什么意思？

IQR（interquartile range，四分位距）就是中间 50% 数据的范围：

```text
IQR = 第 75 百分位 Q3 - 第 25 百分位 Q1
```

鲁棒性（robustness）可以先理解成“抗极端值能力”。

比如 10 个数里有 9 个都正常，只有 1 个特别离谱。平均值（mean）会被这个离谱值带歪；中位数（median）和 IQR 受影响小得多。所以 SafeConf 用 median/IQR 做标准化，而不是 mean/std。

### 3. “5000 基因”是固定规矩吗？所有扰动都这样吗？

不是。

扰动（perturbation）说的是“你故意改了什么条件”。可能是：

- 敲掉一个基因；
- 加一个药；
- 同时改两个基因；
- 加细胞因子；
- 改温度、缺氧、辐射等环境条件。

5000 基因说的是“最后拿多少个基因来表示细胞反应”。单细胞数据里可能有一两万个基因，但很多基因噪声大、信息少，所以工程上常选一组高变异基因（highly variable genes, HVGs）或模型需要的基因面板。

所以：

```text
单基因敲除 = 只改了一个基因。
5000 维向量 = 观察 5000 个基因的表达变化。
```

这两个不是一回事。

## 最容易混的地方

SafeConf 不是直接预测基因表达。

SafeConf 做的是：已有预测器给出一个预测后，SafeConf 判断这个预测大概靠不靠谱。

可以这样区分：

```text
V0 / ContextSim / GEARS 这类模型：负责做预测
SafeConf：负责给预测做质检
```

所以 SafeConf 更像“预测结果质检协议”，不是新的表达预测模型。

## 下载到本地怎么看

推荐直接下载整个目录：

```text
docs/小白科普/
```

不要只下载单个 HTML。因为 HTML、图片、CSV 之间用的是相对路径，拆散后链接会失效。

下载后优先打开：

```text
index.html
```

然后再按这个手册的顺序看。
