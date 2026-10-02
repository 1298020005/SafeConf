# PublicSet 独立输入作用域审计

日期：2026-10-02。仅读取 CSV/Parquet 元数据和构造代码；没有读取表达、预测或任何 McFaline TEST 数值，没有拟合模型。

## 决定

Source 用 **outer gene fold × evaluation context** 的20个推理视图；每个外折实际拟合仍可缓存为一份模型。McFaline沿用五个gene外折，每折统一排除全部该折validation实验，之后重算历史集合及其冲突。

这解决了两个互相竞争的问题：评价答案不能进入其他训练查询的历史，而合法的同gene其他背景实验必须继续作为公共知识。Source的含义是：在每个部署视图中，该背景的查询实验未知，其他背景实验已公开；不是四个背景的所有实验同时未知。

## 身份匹配

- Source：`E201::{target_context}::{condition}`。1808/1808查询与公共库唯一记录匹配；target/context、gene/perturbation_target、condition、treated-cell count和batch count均一致。现有builder明确把多个blind cache中的相同context/condition实验合并，不能把这些cache副本当独立实验。原始真值构造按同condition细胞求均值。
- McFaline：`McFaline23::{perturbation}::{context}::{treatment}`。542/542 DEV查询对应库中的542条`::val`记录；其余6631条为`::train`。每条的guide、plate、split-half质量是该实验的派生统计，禁止父实验时全部一起禁止；不能只删除效应向量却保留其质量摘要。
- 两域现有库均是一条canonical experiment一行。未来新增guide/plate派生行时沿用父实验ID的排除，不按派生文件名重新判断独立性。

身份判断来自元数据与已登记构造代码；没有在本次读取细胞表达重新证明逐数值相等。

## Source 具体范围

若每个gene外折四context评价实验全部从任何推理历史中删除，4737条原边仅剩417条，只有376/1808查询仍有历史。这改变了研究场景，不能悄悄执行后再把结果当原计划。

按context推理视图，保留全部4737条合法边和1808查询：

| 外折 | 训练查询 | 训练边 | 四context评价查询 | 四context评价边 |
|---|---:|---:|---:|---:|
| 0 | 1446 | 3795 | 362 | 942 |
| 1 | 1446 | 3781 | 362 | 956 |
| 2 | 1446 | 3793 | 362 | 944 |
| 3 | 1447 | 3793 | 361 | 944 |
| 4 | 1447 | 3786 | 361 | 951 |

每一fold的4个context视图拥有完全相同的训练query和训练history ID集合，`fit_signature`已核验唯一。因此PCA只拟合实际训练边中唯一history记录时，同参数可安全复用，**不需要把训练预算乘4**。允许用于推理的库不等于允许PCA拟合的库。

每视图完整计数见`SOURCE_SCOPE_COUNTS.csv`；禁用与允许ID、训练唯一history ID和全部边在runtime审计目录中保存，并由`INPUT_SCOPE_AUDIT.json`绑定SHA。

## McFaline 具体范围

| 外折 | 训练查询 | 训练边 | 评价查询 | 过滤前评价边 | 过滤后评价边 | 合法库记录 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 427 | 5610 | 115 | 1547 | 1451 | 7058 |
| 1 | 432 | 5722 | 110 | 1435 | 1367 | 7063 |
| 2 | 445 | 5875 | 97 | 1282 | 1216 | 7076 |
| 3 | 434 | 5737 | 108 | 1420 | 1344 | 7065 |
| 4 | 430 | 5684 | 112 | 1473 | 1397 | 7061 |

过滤后542个查询仍全部有合法历史，共6775条边。原7157条边中382条引用了同折其他validation实验，必须在该外折推理时删除。训练边由于同gene分折不受这个过滤影响，但PCA、汇总和派生特征仍须只按实际允许输入拟合/构造。

## 内部停止轮数选择与嵌套复用

内部early-stop或Public OOF分组先从outer-train查询中划出gene；任何拟合都排除内部validation gene的监督行。Source内部validation同样用context推理视图；McFaline再禁止内部validation全部父实验ID，并保留该外层的禁用表。随后重算query候选、support比例与conflict，不能过滤已生成的特征行后沿用原整组摘要。

缓存签名至少包含拟合query、实际预处理history ID、输入效应版本、外层禁用范围、模型参数/种子和特征定义。不同推理视图若所有参数拟合输入一致，可共享模型，但每个输出必须保留其推理历史scope。

## 执行记录

第一次审计脚本因pandas合并`condition`后缀导致导出列错误退出，未读取表达或产生拟合。将列冲突显式命名并增加condition一致性断言后，全部断言通过。没有改动root训练脚本或原资产。
