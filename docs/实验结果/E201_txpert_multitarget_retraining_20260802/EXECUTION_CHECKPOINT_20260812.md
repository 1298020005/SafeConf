# E201 执行检查点：6 个模型完成，HepG2 seed 2 运行中

记录时间：2026-08-12 19:20（Asia/Shanghai）

实验：`E201_txpert_multitarget_retraining`
阶段：16 个正式模型的盲训练；尚未预测、封存风险量或释放 target 真值。

## 完成状态

| target | seed | 状态 | epochs / steps | 结束时间 | best source validation | target 扰动访问 | 状态文件 SHA-256 |
|---|---:|---|---:|---|---:|---:|---|
| K562 | 1 | COMPLETE | 80 / 368,640 | 08-02 20:00 | 0.453448 | 0 | `9f1f93b0ba992440a55db087ca07a6bb945011d067ce944cb67bff9cdd2509c2` |
| RPE1 | 1 | COMPLETE | 80 / 341,200 | 08-10 07:47 | 0.407204 | 0 | `07125f73cece2afd78e36fedf069d2ff7f769c9504f78be8d04bcf3ebd3cec1d` |
| HepG2 | 1 | COMPLETE | 80 / 392,960 | 08-10 21:01 | 0.482933 | 0 | `7cbcaa47a231d9d6f2b5204ba863ea4e90fd17fe5b19ad47ca318b4035c86861` |
| Jurkat | 1 | COMPLETE | 80 / 352,640 | 08-11 09:04 | 0.476302 | 0 | `8867d79cc9f41a7f2cc12ff22ae5b24d7517392c916d357b85a4ef3137df6b1c` |
| K562 | 2 | COMPLETE | 80 / 368,640 | 08-11 21:30 | 0.452209 | 0 | `28214113272333142b7e2c98029e80c44b2109e32d1b605c104358d862975470` |
| RPE1 | 2 | COMPLETE | 80 / 341,200 | 08-12 09:09 | 0.415321 | 0 | `bc70cdcdb01101d7b3e7d56bb97f9cb2847ef31f28bd441819e24775102d3cf3` |
| HepG2 | 2 | RUNNING | epoch 60 / 80 | — | — | 目标盲视图中 0 行 | `4151f4516bc177dca5ee3307b503025f924cd458ad6be4c469f76c542e6b245f` |

`best source validation` 只来自 source 背景 validation，不是 target test 指标。六个
完成模型均记录 `target_test_dataset_constructed=false` 与
`target_perturbed_cells_accessed=0`。

## 队列与资源

- systemd 服务：`safeconf-e201-txpert-20260809.service`；监督 PID `1965937`；
- 队列位置：第 7/16 项；完成 HepG2 seed 2 后自动运行 Jurkat seed 2，再按固定顺序
  完成 seeds 3–4；
- GPU1：当前 HepG2 seed 2；GPU0：系统 VLLM 服务，未抢占；
- 数据盘仍有约 2.8 TiB 空间；
- TxPert commit、训练 adapter、种子、batch size、轮数、评价合同均未改变。

## 真值边界

本检查点没有读取 target perturbation expression，没有构造 target test dataset，
没有执行正式预测、风险排序、误差计算或任何 E201 结论。完成 16 个模型后必须遵循
`TARGET_RELEASE_AND_EVALUATION_FREEZE.md` 的封存→双远程提交→释放真值→正式评价顺序。
