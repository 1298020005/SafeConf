# E157｜PRESCRIBE Norman P3/P4 训练与无真值锁分合同

冻结日期：2026-07-14。E157 只完成开发集训练、checkpoint 固定和测试任务的 label-only 预测预锁。正式执行脚本和本合同必须先提交 Git；运行中恢复时源码与输入清单必须逐字节一致。E156 为文件哈希和任务元数据读取过原始容器与 `obs`，但没有索引、载入或变换任何测试 X 行，也没有生成测试表达文件。E157 若发现旧版 `sealed_test_transform.h5ad` 必须拒绝运行。本阶段不得打开原始 Norman 数据，不得读取测试表达、测试 DE、测试误差或任何测试真值统计。

## 1. 固定输入

每个面板只能使用：

- E155 冻结的 64 个 train、20 个 val 和 24 个 test 任务名称；
- E156 的 dev-only AnnData、train/val graph cache、train-only feature/PCA、control prior 和资产哈希；
- 冻结的 scGPT perturbation embedding；
- PRESCRIBE upstream commit `6f7264a205aaff654a9594863c5c10b656f88ebe` 及 E155/E156 已记录的 Norman loader 补丁。

E157 不接受原始 Norman h5ad，也不接收 test h5ad 或 test graph。运行前必须确认 E156 的测试 X 行未被索引、载入或变换，且数据目录不存在旧版 sealed test 文件。E155/E156 状态中可见预先冻结的 24 个扰动字符串和 `obs` 元数据计数；后者只用于审计，不进入 DataModule、模型、checkpoint 或 label-only query。E157 对测试任务的实际查询输入只有扰动字符串、train-control mean 和冻结模型资产。

## 2. Development-only DataModule

DataModule 只建立 train 和 val loader，`test_dataloader()` 必须直接报错。运行前必须断言：

1. train 恰为 E155 的 64 个条件且包含 `ctrl`，val 恰为 20 个条件；
2. dev AnnData 与 graph cache 不含任何 test condition 或 test cell；
3. graph key 恰等于 `train ∪ val`，没有 GO/embedding/stale cache 导致的静默缺失；
4. E155 split、E156 ordered genes、PCA、dev H5AD 和 graph cache 的哈希全部匹配；
5. 穷举审计每个 graph：`x` 等于 train-control mean，`y_pca` 为有限 10 维开发标签；train `y_n` 有限、val `y_n` 为 NaN sentinel；
6. train/val loader 非空，test graph 数为 0。

E156 split pickle 可以保留 test 字符串供 checkpoint 后查询，但这些字符串不能被 DataModule 解释成 test loader。

## 3. 固定训练协议

采用 E95 相同的 PRESCRIBE native architecture 与 native loss：

- seed `3407`；
- warmup `5 epochs`；
- main training 最多 `50 epochs`，保留 E95 固定的 native early stopping（patience `3`）；
- batch size `4096`，gradient accumulation `4`；
- 按最低有限 `val/loss` 选择 checkpoint；若 best path 缺失、文件不存在或 best score 非有限，正式运行失败，禁止用 last checkpoint 代替最终模型；
- P3 使用物理 GPU 0，P4 使用物理 GPU 1，每进程只暴露一块 GPU；
- latent dimension、flow、prior、学习率和正则参数与 E95 formal 保持一致。

启用 deterministic training，固定 Python、NumPy、PyTorch 和 CUDA 随机性。不得根据中间结果调整 epoch、seed、batch size、split、feature、PCA、score 公式或任务名单。

## 4. Checkpoint 数据边界

`adata` 不参与 native forward 或 loss。E157 构造模块时使用 `adata=None`，并从 Lightning hyperparameters 中删除 `adata` 和 `model` 对象。训练结束保存：

- best Lightning checkpoint；
- state-dict-only locked checkpoint；
- checkpoint SHA256、源码 SHA256、E156 输入 SHA256；
- 最低 `val/loss`；
- checkpoint 内容递归审计。

任一 checkpoint 含 AnnData、`sealed_test_transform` 路径或测试数据引用，均判实现失败。必须先把 checkpoint 哈希、模型源码哈希、E156 输入哈希和锁定时间写入 STATUS，才允许生成测试任务 label-only 分数。

best/last/warmup Lightning checkpoint 与 TensorBoard 日志是本地运行资产，不进入 Git。可复核的 slim state-dict 固定保存到 `/home/yyf/data/safeconf_e157_locked_models/<panel>/E157_LOCKED_NATIVE_STATE.pt`；远程仓库保存其路径、字节数、SHA256、源码、输入和重训练合同。根目录与 E157 目录的 `.gitignore` 排除 `.pt/.ckpt` 与运行日志，提交时只能纳入 STATUS、manifest、audit、合同和 locked task-score CSV，不能绕过 GitHub 单文件限制强制加入模型文件。

## 5. Label-only forward 等价性

正式查询前，预先按 SHA 固定 8 个 val 任务，对比：

1. E156 native development graph；
2. 只含同一 train-control mean 与扰动字符串的 label-only graph。

在 `model.eval()` 与 `torch.inference_mode()` 下，native/query 的 PCA prediction、log probability、epistemic、aleatoric 及其差值必须全部有限，最大绝对差必须不超过 `1e-5`。若失败，不得查询 test task，也不得事后放宽容差。

## 6. 无真值测试任务预锁

等价性通过后，只从 E155 manifest 读取固定 24 个 test perturbation strings。每个 query 只能含：

- train-control mean；
- canonical perturbation string；
- frozen scGPT embedding 与 locked checkpoint。

query 不得含 `y`、`y_pca`、DE index、测试细胞数、测试表达或测试真值。固定保存：

- PCA10 prediction；
- epistemic confidence；
- aleatoric confidence；
- `combined_confidence = 2 × epistemic + aleatoric`；
- PCA10 reconstructed prediction 相对 train-control mean 的 `predicted_magnitude_rms`。

CSV 必须恰有 24 个不重复任务，任务集合与 E155 完全一致，所有值有限。完成 CSV SHA256 并写入 STATUS 后，本阶段结束。E157 不计算 Pearson、direction、RMSE、风险相关或任何通过状态。

## 7. Native ListMLE 限制

native `ListMLELoss` 按 `y_n` 排序后对整列计算 `-sum(log_softmax(scores))`，该值对排列不变。因此 E156 train-only E-distance 在当前源码中是接口字段，其排序实际上不改变 native loss。E157 保持原实现以避免暗改主模型，但论文不得声称 E-distance 有效提供了排序监督。修正 ListMLE 只能在主结果锁定后另立合同作为敏感性模型。

## 8. 失败恢复

warmup 与 main training 分别保留 last checkpoint。OOM、进程中断只允许在同一 STATUS、任务、参数、源码和输入哈希下恢复：run 目录有 checkpoint 却没有 STATUS 时拒绝；completed warmup state 必须与 STATUS 登记的 SHA256 一致；last checkpoint 必须来自前一轮明确的 warmup/main training 或已记录失败阶段，并在恢复时重新登记 SHA256。以下情况均判实现失败：

- 输入或源码哈希变化；
- train/val/test 任务不匹配；
- test loader 被创建、旧 sealed 文件出现，或任何测试表达被访问；
- checkpoint 含 AnnData 或测试引用；
- label-only 等价性失败；
- 24 个任务缺失、重复或输出非有限。

允许修复代码接线、缓存或运行环境后按原参数重跑，但必须保留失败 STATUS 和异常。中断恢复保持同一科学协议，不宣称与未中断运行逐位一致。不得更换 reserve、任务、seed、split、feature、PCA、batch size、epoch 或置信度公式。

## 9. 固定运行接口

```bash
CUDA_VISIBLE_DEVICES=0 /home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e157_prescribe_norman_p3p4_native.py --panel p3 --mode formal --seed 3407

CUDA_VISIBLE_DEVICES=1 /home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e157_prescribe_norman_p3p4_native.py --panel p4 --mode formal --seed 3407
```

正式执行前必须同时满足：E156 严格重跑通过、P3/P4 dry-run 通过、E157 脚本与本合同已提交 Git。此后不得再改源码、合同或输入；任何改动都要建立新实验编号。
