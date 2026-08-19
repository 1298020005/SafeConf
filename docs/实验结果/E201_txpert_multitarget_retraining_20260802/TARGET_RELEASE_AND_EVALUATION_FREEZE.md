# E201 目标真值释放与评价冻结

冻结日期：2026-08-02

## 冻结目的

E201 要检验整列留出：同一批基因扰动在三个 source 细胞系可见，在第四个
target 细胞系的扰动表达全部隐藏。风险分数只能使用预测前可取得的信息。
如果先查看 target 误差再挑模型、种子、指标或任务，会把测试集信息带回方法
设计，因此正式流程分成模型封存、目标预测和事前评价三段。

“盲态”在本实验中的准确含义是：正式训练进程不接收 target 扰动表达矩阵，
不构造 target test dataset，SafeConf 也不以 target 误差拟合参数。构建物理盲
H5AD 时必须读取原始文件并删除相应行，所以这里不声称原始数据文件从未被
任何数据准备程序打开。

## 释放门

满足以下全部条件后才运行目标预测：

1. K562、RPE1、HepG2、Jurkat × seeds 1–4 共 16 个正式作业全部结束；
2. 每个作业完成 80 epochs，实际 train/validation 行数和 batch 数等于冻结表；
3. 训练记录显示 target perturbed rows accessed 为 0，target test dataset 未构造；
4. `last.ckpt` 与 source-validation best 的字节数、SHA-256、epoch、global step、
   参数结构和有限值检查全部通过；
5. `seal_e201_txpert_checkpoint_family.py` 产生唯一的 16-checkpoint family seal；
6. seal 与预测程序提交到当前分支，GitHub、Gitee 和本地 HEAD 完全一致。

family seal 产生后不再增删种子，也不因某个 target 的结果较弱而重训或换
checkpoint。80-epoch last 是主分析；source-validation best 只进入预先声明的
敏感性分析。

## 盲态目标预测文件

`run_e201_txpert_sealed_prediction.py` 在启动时重新验证：

- TxPert 固定 commit 与干净工作区；
- SafeConf 当前 commit 已同时推送到 GitHub、Gitee；
- family seal 含完整的四目标 × 四种子笛卡尔积，内部记录哈希未改变；
- 被选 checkpoint 的大小和 SHA-256 与 seal 一致；
- E201 blind-prediction H5AD 的 SHA-256 一致；该物理视图保留公开四个细胞系
  的任务标签和 control 表达，但 542,007 个扰动细胞的表达矩阵全部为 0，
  `uns` 为空。

每个 target 的 seed 1 只写共享 `controls.npy` 和 `observations.csv`，不会生成
`truth.npy`。seed 2–4 逐 batch 核对目标顺序和 batch-matched average control，
并在运行前复核两个共享文件的 SHA-256。每个 checkpoint 只新增自己的
`predictions.npy`。数组用磁盘 memmap 顺序写入，避免一次在内存中保留多份约
十亿字节的矩阵。

每个 batch 都要求 `batch.x` 的非零值数为 0。第一个 batch 额外把 dummy X 从
全 0 改成全 1 后重复前向，两次预测必须逐元素完全相同。这个检查证明当前推理
函数没有读取 `batch.x`，不等于对所有潜在软件泄漏给出数学证明，因此仍保留
文件隔离、访问计数和哈希链。

四目标的 16 份预测完成后，先计算并封存 family disagreement、predicted
magnitude、source-context support 和 source-context delta dispersion 等预测前
特征。随后按 TxPert 公开 `MeanBaseline` 的 source-cell-count 权重封存四目标
general-baseline centroid，并用 E200 的公开类实际输出检查任务级等价性。风险表、
general baseline、代码和哈希全部提交到 GitHub/Gitee 后，单独的 truth-release
程序才能从官方 H5AD 提取目标表达。正式误差计算不与预测进程共用入口。

真值释放的附加门固定为：

- 2,008 个任务的预测前风险状态为 PASS，目标表达访问仍为 0；
- official-general-baseline 的 2,008 个 centroid 已封存；
- K562 580 个任务与 E200 实际 `MeanBaseline` 输出的 delta 等价性最大绝对残差
  不超过 `5e-6`；
- general-baseline 状态、支持审计、访问审计和数组哈希已经双远程提交；
- `release_e201_target_truth.py` 重新计算以上文件哈希后才打开官方 target X。

## 事前评价单位

主任务键固定为：

`target cell line × perturbation condition`

experimental batch 是对照匹配与加权聚合层，不作为独立生物任务。先在每个
batch 内使用匹配 control，再按该 batch 的目标细胞数合成 context–perturbation
centroid。重复的 AnnData observation name 不作对齐主键。

这个定义在目标结果打开前依据 blind obs 做过一次修正。若把 batch 也放进任务
键，K562 和 HepG2 的严格任务中没有任何一项达到 30 个细胞，大多数 batch 小组
只有 1–3 个细胞，会把技术小组误当独立任务并放大样本量。修正后的严格
context–perturbation 任务如下：

| target | 严格任务 | 细胞 | 主分析（≥30） | 敏感性（10–29） |
|---|---:|---:|---:|---:|
| K562 | 580 | 80,153 | 566 | 14 |
| RPE1 | 467 | 38,543 | 416 | 51 |
| HepG2 | 480 | 30,139 | 405 | 75 |
| Jurkat | 481 | 43,604 | 421 | 60 |
| 合计 | 2,008 | 192,439 | 1,808 | 200 |

主误差和风险分析：

1. centroid RMSE；
2. 四种子预测 centroid 的 family RMS 与 worst-seed error；
3. 四种子分歧作为预测前不确定性证据；
4. SafeConf 风险分数与误差的 Spearman 相关、bootstrap 95% CI；
5. 控制 predicted magnitude 后的 partial Spearman；
6. 最高风险 20% 的错误富集、相对随机复核的 recall/precision 和固定预算收益；
7. 每个 target、每个 seed 单独报告，再做预先固定的跨 target 汇总。

强基线固定为 official general baseline、batch-matched control、predicted
magnitude。E202 已经否定的“训练分歧能够预测 GAT 相对 general baseline
regret”不改写为正结果。

补充预测质量指标沿用 E198 在看 E201 结果前校准出的五项：`mse`、
`pearson_pert`、`rank`、`energy_distance_pca_k=50`、`de_auprc`。不会因 E201
结果改变主指标集合。

## 主裁决门

主裁决使用 1,808 个 ≥30-cell 任务。跨 target 汇总的置信区间按 perturbation
condition 整簇 bootstrap 5,000 次；同一扰动在不同 target 的记录一起重采样，
避免把它们当成完全独立样本。四个 target 的结果另行逐一报告。

- 证书门：`family_RMS² = centroid_RMSE² + family_disagreement²` 的最大绝对
  残差不超过数值容差，且 family RMS 低于 disagreement 的任务数为 0；
- 经验路由门：`safeconf_e201_risk` 对 family RMS error 的 pooled Spearman 和
  20% review utility 的 95% CI 下限都大于 0；
- magnitude 增量门：控制 predicted magnitude 的 partial Spearman 95% CI
  下限大于 0，或相对 magnitude 的配对 review-utility 增量 95% CI 下限大于 0；
- target 稳定性：四个 target 的点估计和区间完整报告，不以“多数 target 为正”
  替代 pooled cluster interval，也不删除方向不一致的 target。

门未通过时保留负结果，不修改风险权重、任务集合或 bootstrap 单位。

## 文献核对后的约束

- [TxPert](https://www.nature.com/articles/s41587-026-03113-4) 的跨细胞系设置
  也是四个 target 分别整列留出，并报告四个训练种子；E201 据此固定四目标和
  四种子，但公开仓库缺少作者内部训练入口，所以实验名称保留“公开
  STRING-GAT 重训练审计”。
- [PerturBench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8aee537279a66ced96319dfca3c00002-Abstract-Datasets_and_Benchmarks_Track.html)
  发现简单方法通常有竞争力，rank 指标能补充 RMSE 并暴露 mode collapse；
  因此 E201 不只与深模型比较，也保留简单强基线和任务检索指标。
- [scPertEval](https://www.biorxiv.org/content/10.64898/2026.07.23.740433v1)
  区分“指标能否分开有效与无效预测”的校准问题和正式评分问题；E198 完成指标
  校准，E201 只使用通过事前筛选的协议。
- [Nature Methods 29-dataset benchmark](https://www.nature.com/articles/s41592-025-02980-0)
  显示细胞背景异质性直接影响跨背景泛化，且基础模型并非在所有场景稳定领先；
  因此 E201 必须分 target 报告，并把 context 难度和 predictor identity 分开。
- 2026-07-30 发布的 [PerturbMap](https://arxiv.org/abs/2607.28090) 与周老师的
  “A 中见过扰动、B 中缺失同一扰动”问题高度接近。它在 Frangieh 数据上用
  train-only route reliability 控制跨背景转移。该工作发布晚于 E201 冻结，
  先登记为后续独立对照/讨论对象，不回头改变 E201 模型、种子或指标。

## 正式执行顺序

1. 完成四个 target 的 seed 1；
2. 按相同顺序完成 seeds 2、3、4；
3. 运行 family seal，独立复核并双远程提交；
4. 对每个 target 先运行 seed 1，封存共享 control 与任务顺序；
5. 运行同一 target 的 seeds 2–4，并核对共享输入；
6. 生成预测前风险特征，封存并双远程提交；
7. 生成 TxPert official-general-baseline 等价 centroid，完成 E200 独立等价性
   检查并双远程提交；
8. 运行独立 truth release，再执行冻结评价；
9. 输出任务级明细、汇总表、负结果、失败边界和周老师四项问题的证据矩阵。

当前正式队列从 K562 seed 1 开始。训练期间只查看 source validation、损失、
资源占用和工程状态，不计算 target error。
