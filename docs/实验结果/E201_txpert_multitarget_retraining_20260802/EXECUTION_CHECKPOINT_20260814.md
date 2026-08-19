# E201 执行检查点：10 个模型完成，HepG2 seed 3 运行中

记录时间：2026-08-14 23:25（Asia/Shanghai）

实验：`E201_txpert_multitarget_retraining`

## 当前状态

| target | seed | 状态 | epochs / steps | 完成时间 | best source validation | target 扰动访问 |
|---|---:|---|---:|---|---:|---:|
| K562 | 1 | COMPLETE | 80 / 368,640 | 08-02 20:00 | 0.453448 | 0 |
| RPE1 | 1 | COMPLETE | 80 / 341,200 | 08-10 07:47 | 0.407204 | 0 |
| HepG2 | 1 | COMPLETE | 80 / 392,960 | 08-10 21:01 | 0.482933 | 0 |
| Jurkat | 1 | COMPLETE | 80 / 352,640 | 08-11 09:04 | 0.476302 | 0 |
| K562 | 2 | COMPLETE | 80 / 368,640 | 08-11 21:30 | 0.452209 | 0 |
| RPE1 | 2 | COMPLETE | 80 / 341,200 | 08-12 09:09 | 0.415321 | 0 |
| HepG2 | 2 | COMPLETE | 80 / 392,960 | 08-12 22:29 | 0.481172 | 0 |
| Jurkat | 2 | COMPLETE | 80 / 352,640 | 08-13 10:29 | 0.475845 | 0 |
| K562 | 3 | COMPLETE | 80 / 368,640 | 08-13 22:58 | 0.458906 | 0 |
| RPE1 | 3 | COMPLETE | 80 / 341,200 | 08-14 10:25 | 0.408336 | 0 |
| HepG2 | 3 | RUNNING | epoch 78 / 80 | — | source val ~0.476 | 目标盲视图为 0 行 |
| Jurkat | 3 | QUEUED | — | — | — | — |
| K562–Jurkat | 4 | QUEUED | — | — | — | — |

队列位于第 11/16 项。用户级 systemd 服务
`safeconf-e201-txpert-20260809.service` 自 8 月 9 日持续运行。GPU1 只跑当前
一个 batch-64 正式训练；GPU0 仍留给系统 VLLM。

已完成作业的 train/validation 行数和 batch 数与
`FORMAL_TRAINING_FREEZE.md` 一致；adapter SHA-256 仍是
`274344472a2e19d91cf8e971e4e46cf08e39868a36d7892310281d1d847d5e7a`。

按已完成作业的墙钟时间，HepG2 seed 3 约在 8 月 15 日 00:00 前后结束；随后
Jurkat seed 3 约 12 小时，seed 4 四个背景合计约 49 小时。16/16 训练大约在
**8 月 17 日下午**结束。这只是工程预估，不是评价时间表。

## 仍然不能说什么

- 没有读取 target perturbation expression；
- 没有构造 target test dataset；
- 没有 E201 正式预测、风险表或 target error；
- source validation 只证明训练正常，不是跨背景测试结果。

16 个训练完成后的唯一顺序不变：checkpoint family seal → 零真值预测 →
风险量和 general baseline 封存 → 双远程提交 → target truth release →
formal core evaluation。
