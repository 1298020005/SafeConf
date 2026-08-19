# SafeConf 当前研究状态

更新时间：2026-08-14 23:25（Asia/Shanghai）

组会已经过去。当前最高事实以本页和
[`NEXT_PHASE_PLAN_20260814.md`](NEXT_PHASE_PLAN_20260814.md) 为准。
`CURRENT_RESEARCH_STATUS_20260813.md` 只保留昨天的汇报口径。

## 一句话

E201 仍在盲训练：10/16 完成，HepG2 seed 3 接近收尾。现在不能写 E201 结论，
也不能开始按结果改论文。下一部分是把 16 个模型跑完，再按冻结顺序封存、预测、
开真值、评价。

## 研究问题（不再改）

单细胞扰动预测给出“敲掉某个基因或加药后表达怎么变”。SafeConf 在预测之后问：
**真值未知时，哪些任务更值得优先复核？哪些 setting 应该停止使用经验排序？**

## 已有、正在补、还不能说

| 层 | 内容 | 状态 |
|---|---|---|
| 老师三问 | 误差对象、magnitude 是否泄漏、未见任务怎么打分 | 已回答边界 |
| 三档更难 setting | 小矩阵、行列/双未见、跨数据集 | 能算；不都更好 |
| 公开图模型局部证据 | E198 协议；E199 K562 未见扰动；E200 K562 整背景留出 | 已有真值，须保留 magnitude 更强的边界 |
| 多背景整列留出 | E201，4 背景 × 4 种子 | 训练中，0 行真值访问 |
| 可投稿正文 | 题目、摘要、主结果 | 等 E201 正式评价和老师拍板 |

## 下一步入口

1. 训练账：[`E201_txpert_multitarget_retraining_20260802/EXECUTION_CHECKPOINT_20260814.md`](E201_txpert_multitarget_retraining_20260802/EXECUTION_CHECKPOINT_20260814.md)
2. 完整安排与历史审核：[`NEXT_PHASE_PLAN_20260814.md`](NEXT_PHASE_PLAN_20260814.md)
3. 昨天组会稿：[`../../workspace/group_meeting_20260813/README_先看这个.md`](../../workspace/group_meeting_20260813/README_先看这个.md)
