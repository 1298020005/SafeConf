# E33–E37 advisor follow-up plan

生成日期：2026-07-09

这是下一阶段实验队列登记和触发记录。来源是周老师聊天记录后半段的要求。

最重要文件：

- `EXPERIMENT_QUEUE.csv`
- `TRIGGER_LOG_20260709.md`
- `RUN_STATUS.json`

## 实验顺序

| 顺序 | 实验 | 目的 |
|---|---|---|
| 1 | E33 输入来源与评价对象审计 | 先回答老师卡住的输入/错误来源问题 |
| 2 | E34 小矩阵 / 低覆盖度 | 回答“只给矩阵里的一个小矩阵” |
| 3 | E35 整行 / 整列 holdout | 回答“整行整列 holdout” |
| 4 | E37 gene / chemical 分层总表 | 回答“不同类型都看看” |
| 5 | E36 跨数据集 transfer | 回答“一个数据集到另一个数据集” |
| 6 | E38 模型级可靠性 | 后续再处理 GEARS/scGPT/CPA 谁更可靠 |

## 当前判断

E33 已完成；E34/E35 已触发 split smoke 和 scoring smoke。

下一步是把 E34/E35 从 smoke 升级成 formal。E36 是升级项，等小矩阵和整行整列稳定后再跑。

## 已触发输出

- `docs/实验结果/E33_feature_provenance_error_source_audit_20260709/`
- `docs/实验结果/E34_E35_split_smoke_20260709/`
- `docs/实验结果/E34_E35_scoring_smoke_20260709/`

## 给组会材料的对应文件

`workspace/group_meeting_20260709_MAINLINE_WHITE/后续实验执行安排_按周老师要求_20260709.md`

这份是给人看的讲稿式安排；本目录是给项目留档和后续执行看的登记表。
