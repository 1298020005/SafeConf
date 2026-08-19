# E201 执行检查点：7 个模型完成，Jurkat seed 2 运行中

记录时间：2026-08-13 10:32（Asia/Shanghai）

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
| Jurkat | 2 | RUNNING | epoch 79 / 80 at metrics.csv | — | — | 目标盲视图为 0 行 |

队列位于第 8/16 项；Jurkat seed 2 完成后，自动执行四个背景的 seeds 3–4。用户级
systemd 服务 `safeconf-e201-txpert-20260809.service` 正常运行。GPU1 只运行当前一个
batch-64 正式训练；GPU0 为系统 VLLM 服务保留。

## 仍然不能说什么

截至本检查点：

- 没有读取 target perturbation expression；
- 没有构造 target test dataset；
- 没有生成 E201 正式预测、风险表或 target error；
- 因而没有任何 E201 的成功、失败或相对 predicted magnitude 的科学结论。

完成 16 个训练后仍须执行既定顺序：checkpoint family seal → 零真值预测 → 风险量和
general baseline 封存 → 双远程提交 → target truth release → formal core evaluation。
不得因为 source validation、某个 target 的结果或汇报需要改变种子、任务、模型或
风险规则。
