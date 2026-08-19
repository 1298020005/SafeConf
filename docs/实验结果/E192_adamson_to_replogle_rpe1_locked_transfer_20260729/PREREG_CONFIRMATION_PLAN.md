# E192 Adamson→Replogle RPE1 锁定迁移确认计划

冻结日期：2026-07-29

## 研究问题

E190 已在 Adamson K562 训练、Replogle K562 评价。E192 不再改模型公式，
直接把同一训练来源、随机种子和模型家族迁移到此前未读取扰动表达的 Replogle
RPE1。这里同时发生研究来源变化和细胞系变化，属于比随机缺一格更难的外部任务。

## 真值隔离

1. 基因候选固定为 E190 在 2026-07-29 已冻结的 47 个基因；
2. E192 只用 RPE1 的 `gene`、`gem_group`、`transcript` 和细胞行号建立任务；
3. 目标对照细胞表达可在预测前读取，用于构造每批次基础状态；
4. 目标扰动细胞表达在六个模型的预测数组提交到 GitHub、Gitee 前不得读取；
5. query graph 不含 `y`，模型资产目录不得出现目标真值。

## 固定任务

- 源：Adamson 2016，K562 CRISPRi；
- 目标：Replogle 2022，RPE1 essential CRISPRi；
- 目标任务最低细胞数：5；
- 每个目标批次最低对照细胞数：20；
- 预测家族：scGPT 3 个种子 + GEARS 3 个种子；
- 种子：3407、3408、3409；
- 表达处理：逐细胞全基因库归一化到 10,000 后 `log1p`；
- 评价轴：源与目标共同、且进入 scGPT 词表的 512 个基因；
- 主要误差：六成员 family RMS；
- 主要风险量：family diversity lower certificate；
- 强基线：predicted magnitude、source-effect magnitude、zero-effect。

元数据冻结后的预期规模为 21 个基因、175 个任务、53 个目标批次和 1,086 个
目标扰动细胞。任何数量变化都停止运行。

## 两个独立 gate

### A. 确定性证书 gate

- family RMS lower violation 必须为 0；
- worst-member lower violation 必须为 0；
- Hilbert identity 最大残差不高于 `1e-7`。

这部分检查数学实现和对齐，不等价于预测模型性能好。

### B. 经验排序激活 gate

RPE1 在 E191 中没有参与开发，默认状态是 `ABSTAIN`。只有同时满足以下条件，才可
把该 setting 从“排序未验证”改为“允许用 diversity 做复核排序”：

1. diversity 与 family RMS 的基因簇 bootstrap 95% CI 下限大于 0；
2. 20% 复核预算的 oracle-normalized utility 基因簇 bootstrap 95% CI 下限大于 0；
3. 10%、20%、30% 三个预算的 point utility 均大于 0。

如果任一项失败，状态保持 `ABSTAIN`。不得因 predicted magnitude 表现较弱而放宽
阈值，也不得在开真值后更换风险公式。

## 描述性结果

以下结果必须报告，但不控制 gate：

- scGPT、GEARS、六模型 family、source-effect 和 zero-effect 的 RMSE；
- 各方法相对 zero-effect 的按基因整簇置信区间；
- diversity、diameter/2、predicted magnitude、source-effect magnitude 的相关；
- 10%、20%、30% 复核预算下的 high-error capture、error lift 和 oracle utility；
- diversity 与 magnitude 的并列对比。

## 停止规则

- 元数据数量变化；
- 冻结文件未同时存在于 GitHub、Gitee；
- 目标扰动表达在 prediction release 前被读取；
- query graph 含 `y`；
- 输入哈希改变、预测塌缩、出现非有限值；
- 真值与 prediction release 的任务顺序不一致。

触发任一项即停止，不补任务、不改阈值、不用目标真值重训。
