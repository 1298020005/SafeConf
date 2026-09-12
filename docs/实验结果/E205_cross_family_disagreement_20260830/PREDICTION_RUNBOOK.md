# E205 封存预测运行说明

更新时间：2026-09-12

## 当前状态

Exphormer 正按四个留出细胞背景、四个随机种子训练，共 16 项。预测入口已经准备完成，但在 16 项全部训练完、检查点家族封存并提交到 GitHub/Gitee 之前不会启动。目标扰动表达仍未授权读取。

## 为什么不能边训练边挑结果

E205 要检验跨模型结构分歧。若看到部分目标误差后再决定用最后一轮还是源验证最优检查点，就会把目标答案带回模型选择。因此全体模型统一使用预先登记的 `last` 检查点；`best_source_validation` 只作预先标明的灵敏度分析，两者不能按目标分别挑选。

## 训练结束后的固定顺序

1. `seal_e205_exphormer_checkpoint_family.py` 核对 16 项训练状态、80 轮计数、架构、数据视图、检查点哈希与目标真值零访问。
2. 生成 `E205_EXPHORMER_FAMILY_SEAL.json` 和对应 CSV。
3. 只提交上述封存文件；同步 GitHub、Gitee，使本地 HEAD 与两个远程完全一致。
4. 使用 `run_e205_prediction_resume.sh` 生成 Exphormer 预测。该脚本调用统一封存预测入口，并显式指定 `--architecture exphormer`。
5. 四个种子的预测、对照和任务顺序全部哈希封存后，才计算同结构分歧、GAT–Exphormer 质心分歧和 SafeConf-M 风险表。
6. 风险表再次同步双远程后，才允许一次性释放目标真实表达进行正式评价。

## 两卡并行方式

训练全部结束后可将四个目标分成两组；每张卡一次只运行一个预测进程：

```bash
CUDA_VISIBLE_DEVICES=0 TARGETS_CSV=K562,hepg2 \
  bash tools/scripts/run_e205_prediction_resume.sh

CUDA_VISIBLE_DEVICES=1 TARGETS_CSV=RPE1,jurkat \
  bash tools/scripts/run_e205_prediction_resume.sh
```

两个进程写入不同目标目录，不共享可变文件。每个目标内部必须先完成 seed 1，再运行 seed 2–4，因为 seed 1 负责封存共同的对照矩阵和任务顺序。

## 允许回答的问题

- Exphormer 的四种子分歧能否排列本家族预测误差；
- GAT 与 Exphormer 的跨结构分歧在控制预测幅度后是否仍有增量；
- SafeConf-M 在两种结构上方向是否一致；
- 任一模型家族是否出现相反方向或停止条件。

E205 不能单独证明适用于所有模型、所有数据集或所有扰动类型，也不能在复核效用区间跨 0 时宣称节省实验成本。
