# TxPert Exphormer 跨结构复现：执行前合同

日期：2026-09-28。E201 STRING-GAT 的同目标五折结果完成后，用户要求连续迭代。本批固定复用相同的 1,808 个主任务、四个目标、五折基因切分、P/Q/H 定义、Ridge/HGB/小 MLP 配置、U20、Spearman 和 2,000 次基因聚类重采样，只把当前预测换成已在目标真值释放前封存的 **TxPert Exphormer 四种子预测家族**。

本批用于检验第一版 SafeConf 在另一种模型结构上的可重复性。Exphormer 与 GAT 共享任务、来源公共历史和已发布目标标签，因此它是跨结构复现，不能计作新数据独立确认。上游能力先按 Exphormer 质心与 E201 官方简单基线、来源迁移和 control 的同任务 RMSE 比较；不合格则只保留压力测试。

所有禁止字段、特征阶梯、模型参数、缺失处理、运行成功门和报告要求与 `TXPERT_RISK_BATCH_CONTRACT.md` 相同。运行中不按 GAT 或 Exphormer 结果改变字段、模型参数和汇总口径；Error Memory 不进入。完成后停止本批并保存完整折外分数、分背景表、增量及区间。
