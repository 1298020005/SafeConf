# E189 post-truth 评价器修复记录

时间：2026-07-29

冻结评价器第一次运行时，`QUERY_TASKS.csv` 已含 `panel_id`，程序仍执行
`DataFrame.insert("panel_id")`，在 H01/support=1 的任务指标表写出之前抛出
`ValueError: cannot insert panel_id, already exists`。

修复仅做列结构检查：已有 `panel_id` 时核对其全部等于当前面板；缺少时才插入。
误差、下界、相关性、bootstrap、阈值、任务集合、模型预测和测试真值均未修改。

第二次运行完成全部指标、置信区间、CSV、状态与图片后，报告末尾调用
`DataFrame.to_markdown()`，因运行环境未安装可选依赖 `tabulate` 而停止。第二次修复
只用内置字符串格式化替代该报告表格函数；不修改任何数值计算或已生成结果。
