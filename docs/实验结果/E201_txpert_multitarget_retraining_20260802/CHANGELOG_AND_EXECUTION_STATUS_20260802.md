# E201 完整变更记录与执行状态

记录日期：2026-08-02

实验：`E201_txpert_multitarget_retraining`

分支：`exp/task-risk-audit-20260611`

本记录生成前基线提交：`5b048f1`

代码远程：Gitee `librety/safe-conf`、GitHub `1298020005/SafeConf`

## 1. 这批工作解决什么问题

E201 针对周老师提出的整列留出问题：同一扰动在若干 source 细胞背景中可见，
在一个新的 target 细胞背景中完全没有扰动表达，预测模型能否给出结果，SafeConf
又能否在目标真值不可见时提前识别高风险任务。

前序 E200 只使用官方公开的 K562 单 checkpoint。它在 566 个主任务上得到：

- `transfer_risk` 与 GAT centroid RMSE 的 Spearman 为 0.4240，95% CI
  0.3506–0.4953；
- predicted magnitude 的 Spearman 为 0.8797，明显强于 transfer risk；
- transfer risk 的 20% 复核效用为 0.3648，predicted magnitude 为 0.9133；
- transfer risk 相对 magnitude 的新增价值没有得到支持。

E202 随后专门检查“控制预测幅度后，训练背景分歧能否识别 GAT 相对 general
baseline 的额外失败”。主 partial Spearman 为 -0.06795，95% CI
-0.15221–0.01906，正式结论为 `NOT_SUPPORTED`。这两个负面边界均保留，不会
在 E201 中改名或删除。

E201 因此不再依赖单一 K562 checkpoint，也不把原有风险分数重复包装。它建立
四个目标细胞系、四个固定种子、物理真值隔离、预测前风险封存和独立真值释放的
完整审计链，检查多模型 family 的失败证书、经验路由能力和相对 magnitude 的
增量。

## 2. 固定的实验对象

### 2.1 模型与公开材料边界

模型使用 TxPert 公开仓库 commit
`08d82eea86746b044cf7531f4ec8c5f60e1cb73f` 的 STRING-GAT 配置：

- STRING top-20 扰动图；
- 4 层 GATv2、hidden 128、2 个注意力头；
- batch-matched average control；
- AdamW，初始学习率 `3e-4`，weight decay 0；
- 5 epochs 线性 warmup，随后 75 epochs cosine decay；
- 总训练 80 epochs；
- source validation 每 5 epochs 检查一次；
- 主分析 checkpoint 固定为 epoch 80 的 `last.ckpt`；
- source-validation best 只进入事先声明的敏感性分析。

准确实验名称是“TxPert 公开代码、公开数据和可恢复训练设置下的 STRING-GAT
重训练审计”。公开仓库没有作者内部的 `Trainer.fit()` 入口，最强 PxMap/TxMap
图也未公开，因此不声称逐字节复现作者内部流水线。

### 2.2 四个目标与四个种子

四个目标依次为 K562、RPE1、HepG2、Jurkat。每次训练留出一个 target 的全部
扰动细胞，保留四个细胞系的 control，并用另外三个 source 细胞系训练。每个
target 固定 seeds `{1, 2, 3, 4}`，共 16 个正式模型。

| target | 训练行 | 训练 batches/epoch | source validation 行 | validation batches/epoch |
|---|---:|---:|---:|---:|
| K562 | 294,951 | 4,608 | 80,340 | 1,256 |
| RPE1 | 273,003 | 4,265 | 76,950 | 1,203 |
| HepG2 | 314,391 | 4,912 | 89,117 | 1,393 |
| Jurkat | 282,132 | 4,408 | 78,682 | 1,230 |

种子和 target 顺序在任何 E201 目标误差打开前登记。不会因为 source validation
或后续 target 结果较弱而删种子、换 checkpoint 或补选有利作业。

## 3. 已完成的数据隔离

### 3.1 四份物理盲训练视图

原始跨细胞系 H5AD 有 632,488 个细胞和 3,352 个基因，SHA-256 为
`1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8`。
程序没有只在运行时做布尔 mask，而是为每个 target 写出独立物理文件，删除该
target 的全部扰动行、删除 `K562_adamson`、清空 H5AD `uns`。

| target | 盲训练视图行数 | target control | source 扰动 | target 扰动 | H5AD SHA-256 |
|---|---:|---:|---:|---:|---|
| K562 | 375,291 | 10,691 | 336,126 | 0 | `8a19d3d4048800c06827e2f28e983bfa6f67b1945d2081692ce3d69a45d471db` |
| RPE1 | 349,953 | 11,485 | 310,788 | 0 | `9aeec2bdc56461713e9039348428d63aa1b98f6000835ed3253c35d6d85a387d` |
| HepG2 | 403,508 | 4,976 | 364,343 | 0 | `1f0dc20806bd40cd151ebfebb59a9fdac5ad14c4e223a655ae0ed6de890ed891` |
| Jurkat | 360,814 | 12,013 | 321,649 | 0 | `5a944ec0f114e2398f2058072121d130deee5af1f97def926619f5ee30c231fb` |

独立审计以 backed AnnData 重新打开每个物理文件，复核行数、基因数、control、
允许 condition、`uns`、`K562_adamson`、字节数和 SHA-256；四份视图全部通过。
详细记录见 [BLIND_VIEW_AUDIT.md](BLIND_VIEW_AUDIT.md)。

### 3.2 零真值预测视图

checkpoint 封存后需要目标任务标签和 control 才能生成预测，但风险封存前不能
取得目标扰动表达。因此另外建立 `E201_prediction_blind`：

- 581,172 行、3,352 个基因；
- 39,165 个 control 行逐值保留；
- 542,007 个扰动行的 X 物理置零；
- `K562_adamson` 排除 51,316 行；
- `uns` 为空；
- H5AD 为 140,792,831 bytes；
- H5AD SHA-256：
  `85f93d1b29ded34d9dcece9ecdba1ef722a3f14aeedbfbe740eed9f045fbe486`；
- manifest SHA-256：
  `27448df0378aab32e1a9fd22bf20c18c90089816cee6c28b9710cd2d6f812e7d`。

独立审计逐值比较全部 39,165 个 control，mismatch 为 0、最大绝对差为 0；分块
扫描全部扰动行，非零值为 0；审计打开的 source 扰动表达行数为 0。CPU 推理预检
还把 dummy `batch.x` 从全 0 改成全 1，两次预测逐元素一致，最大差为 0。
详细记录见 [PREDICTION_VIEW_AUDIT.md](PREDICTION_VIEW_AUDIT.md)。

第一次构建已正确写出零真值 H5AD，但最后创建 runtime hardlink 时发现
`TxPert/cache` 本身就是数据盘 cache 的符号链接，两个路径解析为同一位置。失败
产物没有删除，保存在
`DATA/txpert_official_20260802/cache/E201_prediction_blind_attempt1_failed_runtime_alias/`。
构建程序修正路径同一性判断后，第二次构建完成正式视图。这是工程路径错误，不是
数据内容错误，也没有打开目标结果。

## 4. 训练入口修正与资源审计

### 4.1 smoke attempt 1

RPE1 seed 1 的 20-batch smoke 在 optimizer step 0 停止。上游 `SequentialLR`
不接受当前 smoke 的单 scheduler/milestone 组合。smoke 改为恒定学习率，只用于
检查单步通路；正式 80-epoch warmup+cosine 协议没有改变。

### 4.2 smoke attempt 2

第二次在 optimizer step 0 停止。上游 dataset 将追加到训练集的 control 条件写成
字符串 `["ctrl"]`，模型图索引实际要求公开整数 ID `[-1]`，导致
`IndexError`。局部 dataset 子类只修正这些新增 control 的编码，其他扰动 ID、
表达、batch 匹配和模型结构不变。

### 4.3 smoke attempt 3

20/20 个优化步完成，重建损失从约 0.254 降至 0.120146；39,165 个训练 control
全部编码为 `-1`；target test dataset 未构造；目标扰动访问数为 0。

### 4.4 完整 1-epoch 资源门

RPE1 seed 1 完成 4,265 个训练 batch 和 1,203 个 source validation batch：

- `global_step=4265`；
- `Trainer.fit` 为 535.97 秒；
- 峰值 allocated 显存约 13.65 GiB；
- 峰值 reserved 显存约 14.14 GiB；
- source `val_pearson_delta=0.33827`，只用于通路与资源检查；
- target test dataset 未构造，目标扰动访问数为 0。

线性资源估计约 11.9 GPU 小时/模型，16 个模型约 190 GPU 小时，另加 checkpoint
I/O。原始失败目录、状态和 checkpoint 均保留；详细记录见
[SMOKE_LOG.md](SMOKE_LOG.md)。

## 5. 正式训练与 checkpoint 封存程序

正式训练 adapter 的 SHA-256 固定为
`274344472a2e19d91cf8e971e4e46cf08e39868a36d7892310281d1d847d5e7a`。
正式 K562 seed 1 启动时记录的 SafeConf commit 为
`14dd8efc39bf0ced2cd210310b2a18de1c898126`，当时本地、Gitee 和 GitHub 跟踪
哈希一致。后续 E201 预测/评价代码继续提交，但不会改变已经运行的训练 adapter。

`run_e201_blind_training_queue.py` 已完成并启动，固定顺序为：先四个 target 的
seed 1，再依次完成 seeds 2–4。队列程序具有以下停止条件：

- 用 `fcntl` 文件锁防止重复监督器；
- 识别当前精确训练 PID，不把其他用户 GPU 任务当作本作业；
- 检查指定 GPU 是否被非当前 E201 进程占用；
- 不覆盖既有正式目录；
- 任一作业工程失败后停止，不静默跳过；
- 队列状态持续保持 `target_truth_release=NOT_AUTHORIZED`。

`seal_e201_txpert_checkpoint_family.py` 已完成但尚未运行。它只在 16/16 作业全部
完成时工作，并检查：80 epochs、行数、batch 数、step、目标访问计数、adapter
哈希、checkpoint 有限值、参数结构一致性、last/best 文件哈希。任何条件不符均
拒绝 family seal。

## 6. 预测、风险和真值三段式隔离

### 6.1 盲态预测程序：已完成代码，尚未运行正式预测

`run_e201_txpert_sealed_prediction.py` 只接受已双远程封存的 16-checkpoint family
seal 和零真值预测 H5AD。它逐 batch 要求 `batch.x` 非零数为 0，并在首 batch
执行 0→1 dummy X 不变性测试。

每个 target 的 seed 1 只写共享 `controls.npy`、`observations.csv` 和自己的
`predictions.npy`；seeds 2–4 复核共享文件哈希后只写各自预测。该阶段不会写
`truth.npy`。

### 6.2 预测前任务底表：已经实际生成并封存

主任务键最初考虑 `target × condition × experimental batch`。在任何目标预测和
真值打开前检查 blind obs 后发现，这会把 1–3 个细胞的小 batch 当成独立生物
重复，K562 和 HepG2 甚至没有严格 batch 任务达到 30 cells。任务定义因此在
结果前一次性修正为：

`target cell line × perturbation condition`

experimental batch 只用于 matched-control delta 和按细胞数加权聚合。最终固定：

| target | 全部任务 | target 扰动细胞 | 主分析 ≥30 cells | 敏感性 10–29 cells |
|---|---:|---:|---:|---:|
| K562 | 580 | 80,153 | 566 | 14 |
| RPE1 | 467 | 38,543 | 416 | 51 |
| HepG2 | 480 | 30,139 | 405 | 75 |
| Jurkat | 481 | 43,604 | 421 | 60 |
| 合计 | 2,008 | 192,439 | 1,808 | 200 |

`build_e201_pretruth_task_base.py` 已实际运行。它只打开对应 target 的物理盲训练
H5AD，从三个 source 背景计算 batch-matched source delta：

- 2,008 个任务；
- 5,238 条 source context support；
- 24 条表达访问审计记录；
- 对应 target 扰动表达打开 0 行；
- target predictions 未打开；
- target outcomes 未评价；
- source mean delta 为 `2,008 × 3,352` float32；
- 数组 SHA-256：
  `b068834e49d73aa74cae54154d047c1897611526cb186c40d0dc9ec274a9141a`。

四个 target 都有 343 个扰动在三个 source 背景中同时出现。只有一个 source 的
任务使用各 target 内至少两个 source 任务的事前 dispersion 中位数填补，同时
保留 `dispersion_imputed` 和 support deficit，不把填补值解释成观测。

详细结果见 [PRETRUTH_TASK_BASE_REPORT.md](PRETRUTH_TASK_BASE_REPORT.md)，固定规则
见 [PRETRUTH_TASK_BASE_FREEZE.md](PRETRUTH_TASK_BASE_FREEZE.md)。

### 6.3 风险特征程序：已完成代码并冻结，等待 16 份预测

`run_e201_pretruth_risk_features.py` 在运行前重新检查 family seal、16 份预测状态、
所有文件哈希、dummy X 审计以及预测目录中不存在 truth。每个任务计算：

- `family_disagreement`：四种子围绕 family centroid 的基因与种子 RMS；
- `family_radius`：离 family centroid 最远种子的 RMSE；
- `predicted_magnitude`：family centroid 相对 matched control 的 RMS；
- `model_source_gap`：family centroid 与 control+source mean delta 的 RMSE；
- `source_delta_dispersion`：三个 source 扰动效应的不一致程度；
- `negative_log_source_cells`：source 支持量越少，风险越高；
- `support_context_deficit`：缺少的 source 背景数。

每个风险分量只用同一 target 的主任务估计均值和总体标准差，再把参数应用到该
target 的主任务和敏感性任务。主风险固定为五个 z 分量等权平均：

`family_disagreement + model_source_gap + source_delta_dispersion +`
`negative_log_source_cells + support_context_deficit`

除以 5。predicted magnitude 不进入风险分数，保留为必须正面对比的简单强基线。

### 6.4 TxPert general baseline：已完成代码并冻结，等待风险 control 向量

补充实现 `build_e201_official_general_baseline.py`，用于兑现正式协议中的 official
general baseline 强基线。TxPert 公开 `MeanBaseline` 对单扰动的计算是：各 source
背景先做 batch-matched 扰动 delta，再按该扰动的 source 细胞数跨背景加权，最后
加到 target 的 batch-matched control。E201 直接计算完全相同的任务 centroid，
不需要把 target 扰动表达交给官方 baseline datamodule。

程序另设独立工程验证：只打开 E200 已经通过 TxPert 公开 `MeanBaseline` 类生成的
K562 prediction 与 control，不打开 E200 truth；在相同 580 个任务上比较
`prediction centroid - control centroid` 和 E201 的 cell-count-weighted source
delta。最大绝对残差容差在 E201 真值前固定为 `5e-6`。

2026-08-02 10:29 已在双远程一致的提交 `2b729e9` 上实际运行只读 preflight：

- 580/580 个 K562 任务完成；
- 3,352 个基因顺序一致；
- 最大绝对 delta 残差为 `2.7865171e-6`；
- RMS delta 残差为 `4.9673143e-8`；
- 超过 `5e-6` 固定容差的任务为 0；
- E201 target 扰动表达访问 0 行；
- E200 truth 和 E201 target truth 均未打开；
- preflight 状态 SHA-256 为
  `977010624679fa0192ebc19f7576830a8a0c6e2f2b4acf68f2bdd4b2f9109346`。

该程序正式运行后将写出 2,008 个 weighted delta 和 general-baseline centroid，
并记录 5,238 条 source support、24 条表达访问审计及 target 扰动访问 0 行。当前
风险 control centroid 尚未生成，因此程序和规则已冻结，正式数组尚未生成；不能
把“代码完成”写成“baseline 结果完成”。

### 6.5 独立真值释放：已完成代码并冻结，当前无权运行

`release_e201_target_truth.py` 只在风险特征表、向量哈希、代码和冻结文档已经提交
且本地/Gitee/GitHub 完全一致后运行。现在又增加 official-general-baseline 硬门：
2,008 个 baseline centroid、E200 等价性结果、支持/访问审计必须先提交到两端
远程。程序重新验证官方 H5AD、split、gene set、observations 顺序、四份风险向量
和 general-baseline centroid，随后才生成各 target 的 `truth.npy` 和 release
manifest。

截至本记录生成时：

- E201 target truth 尚未释放；
- `target_truth_release=NOT_AUTHORIZED`；
- 正式 target error 尚未计算；
- 尚无 E201 风险有效性结论。

## 7. 已冻结的正式评价门

正式主裁决只使用 1,808 个 ≥30-cell 任务。跨 target 区间按 perturbation
condition 整簇 bootstrap 5,000 次；同一扰动在不同 target 的记录同时重采样，
避免把共享扰动误作完全独立样本。四个 target 还要逐一报告。

### 7.1 预测误差和确定性证书

正式程序将计算：

- 四种子 family centroid RMSE；
- 每个 seed 的 centroid RMSE；
- family RMS error；
- worst-seed error；
- batch-matched control error；
- TxPert official-general-baseline error；
- source-transfer error；
- `family_RMS² = centroid_RMSE² + family_disagreement²` 的数值恒等式残差。

证书门要求恒等式最大残差不超过数值容差，且不能出现 family RMS 小于
disagreement 的任务。这个下界是代数性质；它不等于经验上已经证明 SafeConf
可以排序所有高误差任务。

### 7.2 经验路由与强基线

- `safeconf_e201_risk` 对 family RMS error 的 pooled Spearman 95% CI 下限 > 0；
- 最高风险 20% 的 review utility 95% CI 下限 > 0；
- 控制 predicted magnitude 的 partial Spearman 95% CI 下限 > 0，或相对
  magnitude 的配对 review-utility 增量 95% CI 下限 > 0；
- 每个 target 的点估计和区间完整保留；
- 门未通过时不修改权重、任务集合、bootstrap 单位或结果方向。

补充质量评价沿用 E198 在 E201 结果前固定的 `mse`、`pearson_pert`、`rank`、
`energy_distance_pca_k=50` 和 `de_auprc`。简单 control/source-transfer/general
baseline 与深模型同时报告。

## 8. 2026-08-02 09:02 实时执行快照

| 项目 | 状态 |
|---|---|
| 正式训练 | K562 / seed 1 正在运行 |
| 训练 PID | `3873663` |
| 队列监督 PID | `3879346` |
| 当前 epoch | 10 |
| 已记录 step | 46,779 |
| 最近重建损失 | 约 0.0476–0.0568，全部有限 |
| 当前 best | epoch 9，source `val_pearson_delta=0.42782` |
| GPU | Quadro RTX 6000，GPU1，约 11.56 GiB，66% utilization，67°C |
| 其他 GPU 任务 | GPU0 上存在其他用户任务，E201 未占用或终止它们 |
| 队列位置 | 1/16，后续 15 个作业已固定登记 |
| target truth | `NOT_AUTHORIZED` |

这里的 best 只反映 source validation，不是 target 测试结果，也不用于修改正式
种子、任务或风险规则。epoch 80 的 `last.ckpt` 才是主 checkpoint。

10:14 再次复核时，同一进程已推进到 epoch 17、step 82,139；最近重建损失约
0.0495–0.0524，全部有限。训练 PID、队列监督 PID 和 GPU1 归属未改变，GPU1
约占用 11.56 GiB；target truth 状态仍为 `NOT_AUTHORIZED`。

## 9. 周老师问题与当前证据对应

| 周老师提出的问题 | 已采取的处理 | 当前状态 |
|---|---|---|
| 实际误差由哪个预测模型产生，风险是否依赖模型 | 明确用 TxPert 公开 STRING-GAT；每个 seed、family centroid、family RMS、worst seed 分开；风险解释限定在该模型 family，不冒充模型无关 | 设计与代码已冻结，正式结果待训练 |
| 目标真值出现前能取得哪些输入 | 只允许 target control、source 扰动支持、source delta、四种子盲预测与分歧；target 扰动表达物理隔离并由独立 release 门控制 | 盲训练视图、零真值预测视图、source 底表均已审计通过 |
| 随机缺 pair/缺 cell、整行/整列、A 训 B 测 | E201 主问题是整列/跨 context：三个 source 训练，第四个 target 完整留出；任务为 context×perturbation，不把小 batch 当独立重复 | 四目标设计已落实；其他缺失机制保留在既有证据矩阵中，不混入 E201 主裁决 |
| 不同扰动模态和更多数据 | E201 增加四个细胞背景和大规模 CRISPRi 任务，直接补多背景；chemical、基因扰动和既有多数据集结果继续独立报告 | E201 不能单独代表 chemical 或所有模态，论文级结论必须结合已有独立数据证据 |

## 10. 文件级变更清单

### 10.1 新增或修改的程序

| 文件 | 作用 | 是否已正式运行 |
|---|---|---|
| `tools/scripts/build_e201_txpert_blind_training_view.py` | 建立四份 target-specific 物理盲训练 H5AD | 是 |
| `tools/scripts/txpert_blind_training_adapter.py` | 恢复公开训练设置、修正 control 整数编码、写运行 provenance | 是，正式 K562 seed 1 运行中 |
| `tools/scripts/run_e201_blind_training_queue.py` | 固定顺序监督 16 个正式作业并防止 GPU/目录冲突 | 是，运行中 |
| `tools/scripts/seal_e201_txpert_checkpoint_family.py` | 检查并封存 16 个 last/best checkpoint family | 否，等待训练完成 |
| `tools/scripts/build_e201_txpert_blind_prediction_view.py` | 建立扰动表达全零的预测物理视图 | 是 |
| `tools/scripts/audit_e201_txpert_blind_prediction_view.py` | 独立复核 obs/var/control/零表达和哈希 | 是 |
| `tools/scripts/run_e201_txpert_sealed_prediction.py` | 使用封存 checkpoint 在零真值视图上生成 16 份预测 | 否，等待 family seal |
| `tools/scripts/build_e201_pretruth_task_base.py` | 建立 2,008 任务和 source-only 证据底表 | 是 |
| `tools/scripts/run_e201_pretruth_risk_features.py` | 生成并封存预测前风险量 | 否，等待 16 份预测 |
| `tools/scripts/build_e201_official_general_baseline.py` | 生成公开 MeanBaseline 等价 centroid，并用 E200 实际类输出做独立等价性检查 | 否，等待风险 control 向量 |
| `tools/scripts/release_e201_target_truth.py` | 风险双远程封存后独立释放 target truth | 否，当前未授权 |
| `tools/scripts/run_e201_formal_core_evaluation.py` | 计算四种子误差证书、路由、magnitude 增量、三类预测基线与 5,000 次 condition-cluster bootstrap | 否，等待 truth release |

### 10.2 已跟踪的说明与表

- [FORMAL_TRAINING_FREEZE.md](FORMAL_TRAINING_FREEZE.md)：正式训练设置和边界；
- [ENGINEERING_AUDIT.md](ENGINEERING_AUDIT.md)：公开材料可恢复与不可恢复内容；
- [SMOKE_LOG.md](SMOKE_LOG.md)：失败尝试、修复和资源门；
- [BLIND_VIEW_AUDIT.md](BLIND_VIEW_AUDIT.md)：四份训练视图独立审计；
- [PREDICTION_VIEW_AUDIT.md](PREDICTION_VIEW_AUDIT.md)：零真值预测视图审计；
- [PRETRUTH_TASK_BASE_FREEZE.md](PRETRUTH_TASK_BASE_FREEZE.md)：任务单位和风险输入冻结；
- [PRETRUTH_TASK_BASE_REPORT.md](PRETRUTH_TASK_BASE_REPORT.md)：source-only 底表结果；
- [TARGET_RELEASE_AND_EVALUATION_FREEZE.md](TARGET_RELEASE_AND_EVALUATION_FREEZE.md)：预测、风险、真值释放与正式评价门；
- [OFFICIAL_GENERAL_BASELINE_FREEZE.md](OFFICIAL_GENERAL_BASELINE_FREEZE.md)：公开 MeanBaseline 等价实现和 E200 独立验证；
- [FORMAL_CORE_EVALUATION_FREEZE.md](FORMAL_CORE_EVALUATION_FREEZE.md)：四目标四种子核心误差、统计单位和科学裁决门；
- `tables/E201_PRETRUTH_TASK_BASE.csv`：2,008 个固定任务；
- `tables/E201_SOURCE_CONTEXT_SUPPORT.csv`：5,238 条 source 支持记录；
- `tables/E201_SOURCE_EXPRESSION_ACCESS_AUDIT.csv`：24 条 source 表达访问审计；
- `E201_PRETRUTH_TASK_BASE_STATUS.json`：任务底表状态、文件哈希和真值访问状态。

原始 H5AD、checkpoint、预测矩阵、source delta 和未来 truth 均位于 `DATA/`，不进
Git。Git 只保存代码、冻结协议、紧凑任务表、状态和哈希，避免将数百 MB/GB 产物
推入已经接近容量限制的仓库。

## 11. 提交时间线

| 提交 | 内容 |
|---|---|
| `336470c` | 新增 E201 盲重训练审计入口 |
| `0c2fb05` | 修正 smoke scheduler 组合 |
| `f7e3664` | 将追加 control 编码为公开整数 ID `-1` |
| `d0b130e` | 记录成功的 20-batch blind smoke |
| `21424d8` | 新增完整 1-epoch 资源门 |
| `0d8e12d` | 测量 fit 时间和峰值显存 |
| `2548861` | 记录资源审计结果 |
| `f157a9c` | 冻结四目标×四种子正式训练协议 |
| `14dd8ef` | 封存 provenance 预检，随后启动正式 K562 seed 1 |
| `8b41d1e` | 新增 16-checkpoint family seal |
| `712367c` | 冻结盲态释放流程与训练队列 |
| `6dc43fe` | 新增零真值预测视图构建程序 |
| `ab595fb` | 修正 runtime cache 已为共享符号链接的路径判断 |
| `1d90357` | 将目标预测与 target truth 物理分离 |
| `6ea03e0` | 在结果前冻结 context×perturbation 任务底表 |
| `b17511f` | 实际生成并封存 source-only 任务证据 |
| `5b048f1` | 新增风险特征封存门和独立 truth-release 程序 |
| `39f99e9` | 提交完整执行记录并冻结主裁决门 |
| `2b729e9` | 冻结 official-general-baseline、真值附加门和正式核心评价程序 |

## 12. 接下来自动执行的顺序

1. 当前队列继续完成 K562 seed 1；
2. 依固定顺序完成其余 15 个 target×seed，不根据 source 分数选择性停止；
3. 运行 checkpoint family seal，并提交 seal/审计状态到两端远程；
4. 在零真值预测视图上生成 16 份 target 预测；
5. 运行预测前风险程序，提交风险表、向量哈希和状态；
6. 生成 official-general-baseline centroid，完成 K562/E200 580 任务等价性检查，
   提交状态、审计表和数组哈希；
7. 本地、Gitee、GitHub 三方哈希一致后，独立释放 target truth；
8. 运行冻结的正式核心评价和补充 scPertEval 端点；
9. 输出每个 target、pooled cluster bootstrap、强基线、失败边界和周老师问题证据矩阵；
10. 保留所有未通过门的结果，不按结果回改权重、种子、任务阈值或评价单位。

## 13. 当前能说和不能说的结论

现在可以确认：E201 的四目标正式训练已经启动；数据隔离、source-only 任务底表、
预测前风险定义、official-general-baseline 封存程序、真值释放门和核心评价程序已经
建立，且 E201 target error 仍未打开。baseline 和核心评价的纯合成自测均通过，
自测明确记录 `target_truth_opened=false`。

现在不能声称：SafeConf 已在四个 target 上有效、已经超过 magnitude、已经完成
16 个模型、已经得到 E201 生物学结论，或已经满足某个期刊录用标准。这些判断必须
等待固定队列、family seal、盲预测、风险封存、truth release 和正式统计全部完成。
