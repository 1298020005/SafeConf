# E205 执行方案：TxPert-Exphormer 整细胞留出验证

更新时间：2026-09-10

状态：**训练前；先做工程 profile，不读取目标扰动真实表达**

## 1. 本实验回答什么

E201 使用四个 TxPert-STRING-GAT 随机种子。它能测量同一种网络结构对随机初始化是否敏感，但不能证明不同结构之间的预测分歧也能提示错误。

E205 固定 E201 的四个目标背景、训练数据、基因顺序、80 轮训练、四个随机种子和评价端点，只把扰动传播模块从 STRING-GAT 换为 Exphormer。实验同时回答两件事：

1. 跨结构分歧是否在预测幅度之外保留风险信息；
2. E206 冻结的 SafeConf-M 是否能在未参与权重开发的第二种模型结构上重复获得增量。

第二问属于“新模型家族确认”，不属于“新数据集外部验证”。E201 的真实结果此前已经打开，因此不能把 E205 写成完全未知真值的前瞻盲测。

## 2. 唯一允许的架构改动

保留 `config-x-cell-gat` 的以下部分：

- 整个目标细胞背景从训练集中移除；
- source-only 训练与验证数据；
- STRING 图及 top-20 邻接规则；
- 无 basal model 的跨背景设置；
- 解码器、优化器、学习率、批大小、80 轮上限和 checkpoint 规则。

只替换 `model.pert_model`：

```text
TxPert-STRING-GAT  →  TxPert-Exphormer
```

不使用现成 `config-exphormer.yaml` 的数据设置。该配置原本对应 K562 内部的未见基因任务，直接使用会改变考法。E205 只读取其中的 Exphormer 架构字段，再合入整细胞留出配置。

## 3. 分阶段运行

### 阶段 A：一轮合同 profile

固定 `RPE1 × seed 1 × batch size 64 × 1 epoch`。这一阶段只检查：

- 训练视图中目标背景的扰动细胞数为 0；
- Exphormer 确实被实例化；
- 一轮训练和 source-only 验证可以完成；
- `last.ckpt` 正常写出；
- 峰值显存能在 24 GB Quadro RTX 6000 上容纳；
- 没有构造目标测试数据，也没有读取目标扰动表达。

通过后才允许正式扩展。profile 的损失和验证指标不作为科学结果，也不用于换模型、换 target 或挑 seed。

### 阶段 B：正式训练

```text
4 个目标背景 × 4 个随机种子 × 1 个 Exphormer 家族 = 16 个模型
```

每个模型固定 80 轮。失败项最多重新运行一次；失败目录整体归档，不覆盖原日志。若同一原因连续失败，停止扩展并登记工程失败。

### 阶段 C：预测前封存

16 个模型全部通过后，先封存以下内容：

- 每个 checkpoint 的字节数和 SHA-256；
- 训练脚本、TxPert 提交号、解析后的模型配置；
- 目标、seed、训练轮数、训练数据访问审计；
- 16 个 checkpoint 的完整清单。

封存完成后才使用与 E201 相同的无真值预测视图生成预测。预测文件、任务顺序和控制表达再次封存后，才运行评价。

## 4. 预先固定的分数

### 4.1 Exphormer 原始 SafeConf

在每个目标背景内重新计算五个分量：

1. Exphormer 四种子的预测分歧；
2. Exphormer 家族均值与 source 历史预测的距离；
3. source 背景间历史效应离散度；
4. source 历史细胞量稀缺度；
5. source 背景覆盖缺失度。

五项在目标背景内分别标准化后等权平均。第 1、2 项随新模型预测更新；第 3–5 项保持 source-only 定义，不从目标真实结果重算。

### 4.2 SafeConf-M 冻结公式

```text
SafeConf-M = 0.80 × rank(Exphormer predicted magnitude)
           + 0.20 × rank(Exphormer SafeConf)
```

`rank` 是同一目标背景内的百分位。权重固定为 0.80/0.20，不在 E205 上重新扫描权重。

## 5. 评价端点与裁决

### 主要端点

20% 复核预算下，SafeConf-M 相对“只看预测幅度”的效用配对差。按扰动条件做簇自助法，5,000 次，主通过门为 95% 区间下限大于 0。

### 次要端点

- 四目标宏平均 Spearman 及相对幅度的差值；
- Top-20% 高错误任务抓取率；
- 原始 SafeConf 控制预测幅度后的偏 Spearman；
- 同家族跨种子分歧与跨家族质心分歧；
- 每个目标背景单独结果，不能只报告表现最好的一格；
- GAT 与 Exphormer 各自的 RMSE、Pearson delta 和 seed 稳定性。

### 结果解释

- 主要端点通过：可以写“固定融合规则在第二模型结构上获得复核效用增量”；
- 只有 Spearman 差值为正：写“改善排序一致性，但决策效用增量未确认”；
- 原始 SafeConf 偏相关为正、融合不增益：保留互补关联，不宣称路由优于幅度；
- 跨结构分歧无效：将 E201 的“模型分歧”收紧为“同架构初始化敏感性”；
- 任何目标出现反向或明显不稳定：逐项报告，不删除、不换权重。

## 6. 与其他实验的优先级

1. E205 Exphormer profile：当前最高优先级，先确认第二结构能否满足同一合同；
2. E205 正式 16 模型：profile 通过后扩展；
3. E204 风险加权训练：属于另一条训练期方法线，保留队列与记录，但不抢占 E205 的主验证资源；
4. 新外部数据确认：E205 完成后再选定数据，另行冻结，不能用已经看过的 E201 结果代替。

## 7. 当前可核验文件

- 训练适配器：`tools/scripts/txpert_blind_training_exphormer_adapter.py`
- profile 等待器：`tools/scripts/run_e205_exphormer_profile_queue.py`
- 原始冻结协议：`docs/实验结果/E205_cross_family_disagreement_20260830/ANALYSIS_FREEZE.md`
- E206 候选公式：`docs/实验结果/E206_safeconf_magnitude_fusion_20260909/E206_REPORT.md`

