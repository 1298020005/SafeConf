# E195｜GEARS 原生不确定性双面板直接竞品复现

冻结日期：2026-07-30  
冻结代码基线：`ae936d59c77ea0a77816ff27244ff4fa98eb8422`

实现前修订：2026-07-30。在 E195 代码尚未提交、训练尚未启动时完成 PRESCRIBE
资产审计，确认 PRESCRIBE 论文默认可靠性终点是 effect Pearson accuracy，而非
E96 的 mean-profile RMSE。Track B 因而把 E145 论文终点设为 PRESCRIBE 主臂，
E96 RMSE 降为敏感性分析。修订时没有产生或读取任何 E195 结果。

第二次实现前修订：2026-07-30。代码审计确认旧 GEARS 包在模块导入时执行
`torch.manual_seed(0)`，且 `GEARS.train()` 会在训练结束后自动评价 test loader。
E195 尚未生成任何结果，因此在运行前固定以下工程修正：

- 每个 run 在 dataloader 与模型初始化前显式设置 Python、NumPy、PyTorch CPU
  和全部 CUDA RNG 为冻结 seed 11、22 或 33；不得只把 seed 写进目录名；
- 训练阶段从 GEARS 对象中暂时移除 test loader，只允许 train/validation
  参与拟合与模型选择；
- 测试阶段先执行不访问 `batch.y` 的前向推理，落盘 predicted effect、
  native logvar、magnitude、任务顺序与文件哈希，形成不可改写的 score lock；
  score lock 完成后才在独立 truth-unlock 步骤读取 test `batch.y`；
- P1/P2 使用隔离的 GEARS 运行目录，并记录 GO/coexpression 缓存哈希，防止面板
  运行顺序静默改变图资产。

这些修正恢复了冻结文本原本要求的真实多 seed 与 risk-before-truth 语义，没有
改变任务、超参数、分数方向或结果变量。

第三次运行前审计：2026-07-30。E195 正式六次训练仍未启动。独立代码复核发现，
仅保存 score CSV 的哈希不足以形成完整证据链，且旧测试烟测留下了一个无完整来源
的 seed 11 缓存。正式运行据此追加以下不改变科学问题的完整性措施：

- 改用全新的 `panel × seed` 隔离缓存根；旧烟测缓存不参与正式运行；
- 明确重建 test loader，并同时记录请求与实际 test batch size；
- score lock 同时绑定 score CSV、prediction NPZ、逐细胞任务顺序和 lock JSON，
  truth unlock 再绑定 records CSV、truth NPZ 与解锁时间；
- 保存每个成员的 `model.pt`、`config.pkl`，并把模型、split、coexpression 缓存及
  原始数组的 SHA-256 写入本地原始资产清单；
- 每个阶段重新核验冻结输入；结果复用必须同时满足任务、参数、seed、设备、代码、
  输出文件和哈希合同；
- 训练子进程失败时返回非零退出码，同一并行波的所有进程回收完成后统一报错；
- support leakage 由每次运行实际产生的 `set2conditions` 验证，不再用
  “全集减测试集”的循环定义代替。

这次审计只加强可复核性和失败关闭规则，没有查看 E195 结果，也没有改 seed、
epoch、面板、风险方向或结果变量。

结果后完整性修订：2026-07-30。六次训练与首次冻结分析完成后，独立复核确认原始
NPZ、任务级误差、family 恒等式、风险方向和全部哈希均正确，但发现首次输出仅给出
每个分数各自的 20% utility bootstrap 区间，没有输出冻结文本要求的“同一面板内
两分数使用同一 bootstrap 索引”的 utility 配对差。首次点估计没有错误；独立区间
也不能代替配对比较。正式归档因此固定以下纯统计修订：

- 对冻结的五组 score pair，使用与 paired Spearman 相同的任务重采样索引；
- 新增两分数的 20% utility 点估计、`A-B` 配对差及 95% bootstrap CI；
- 不改变任何单分数 utility、任务、预测、误差终点、方向或训练资产；
- 若配对差区间跨 0，只能写“未证明稳定差异”，不得依据两个单独区间下结论；
- 最终报告必须同时呈现 magnitude、seed 稳定性、预算依赖、double-perturbation
  历史暴露和 PRESCRIBE 终点依赖等负结果；
- 总图的跨系统长标签面板改为同预测内比较，避免把不同 outcome 排成胜负榜。

这是对既有冻结统计合同的补全，不是根据结果另选终点或比较对象。六次模型训练、
pretruth lock 和原始数组保持原样，只重新运行确定性分析阶段。

## 证据性质

E195 使用 Norman P1、P2 两组已经评估过的测试任务，因此证据标签固定为
`POSTTRUTH_DIRECT_COMPETITOR_REPLICATION`。本实验不能作为新的独立盲测，也不能
通过更换训练参数反复追逐有利结果。

实验要回答三个具体问题：

1. 在固定未见单基因任务上，GEARS 自己输出的 log-variance 能否排序 GEARS
   自己的预测误差；
2. 对完全相同的一组 GEARS-UQ 预测，原生 log-variance、三 seed 分歧和预测幅度
   哪个更有路由信息；
3. 在相同 P1/P2 任务标签和拒绝预算下，GEARS-UQ、PRESCRIBE 和
   GEARS–scGPT post-hoc pair 各自的风险分数对各自预测误差表现如何。

第三项是 predictor–uncertainty system 的并列评价。三种系统的预测值、基因空间
和误差目标不同，不允许把一张并列表解释为在同一个结果变量上的直接胜负。

## 冻结任务

### Norman P1

- 24 个未见单基因任务；
- 任务文件：
  `../E66_norman_gears_fixed_panel_formal_20260711/tables/E60_FIXED_TEST_PERTURBATIONS.csv`；
- SHA-256：
  `f1162e8378fa186153b393b9e3e2a7d5a99189f44e7e0afc6f079d76677e565a`；
- train/validation condition sampling seed：`20260766`。

### Norman P2

- 24 个与 P1 不重叠的未见单基因任务；
- 任务文件：
  `../E75b_norman_gears_panel2_20260711/tables/E60_FIXED_TEST_PERTURBATIONS.csv`；
- SHA-256：
  `36597e0cf025948598bc2195e34e4dd87517be38e7dc3c35bcc9fc05c42df8db`；
- train/validation condition sampling seed：`202607752`。

程序必须复核每个面板恰有 24 个任务、P1/P2 交集为 0、每个测试任务对应的精确
单基因条件不在 train/validation 中，并复核上述哈希。另行报告测试基因出现在
train/validation 双扰动条件中的次数；该次数不得被硬编码成 0，也不得临时改变
既有 P1/P2 split 去追求更有利结果。任务不得按已有误差、PRESCRIBE 分数或
SafeConf 分数筛选。

## 冻结 GEARS-UQ 训练

沿用 E66/E75b 的训练合同，只把 GEARS 官方 `uncertainty` 开关从关闭改为开启：

- predictor：GEARS；
- 数据：本地 Norman atlas；
- seeds：`11, 22, 33`；
- epochs：10；
- hidden size：48；
- decoder hidden size：16；
- GO/coexpression similar genes：10；
- batch size：32；
- test batch size：64；
- 每个 train/validation condition 最多 32 个细胞图；
- 测试图不抽样；
- fixed deterministic validation；
- learning rate、weight decay、direction lambda 使用现有 GEARS runner 默认值；
- `uncertainty=True`。

每个 seed 必须在 dataloader 构造和模型初始化前显式控制 Python、NumPy、
PyTorch CPU 与 CUDA RNG。运行记录同时保存请求 seed、实际 RNG seed、模型初始
参数哈希和训练后参数哈希，以证明三个成员不是同一初始化的重复标签。

GEARS 源码中的 uncertainty head 学习并输出逐基因 `logvar`，但其损失不是标准的
完整高斯异方差负对数似然。E195 统一称它为“GEARS 原生学习型误差代理”或
“native log-variance score”，不得写成经过校准的预测方差或概率保证。

一次冻结运行使用 GPU0/GPU1 并行。若某个 seed 因工程故障失败，只允许用完全相同
的参数恢复该 seed；不得因结果不佳重启或改 seed。

## 同预测内评价（Track A）

### 单 seed 层

每个 `panel × seed × task` 保留：

- GEARS 原生 `logvar_mean`，方向为数值越大风险越高；
- 该 seed 的 predicted-effect RMS magnitude；
- 同一 seed 对同一任务的 RMSE；
- 任务与基因顺序哈希。

主统计是每个 `panel × seed` 内 `logvar_mean → own RMSE` 的 Spearman。三个 seed
不作为 72 个独立任务混在一起；面板内先报告三项 seed 结果，再报告 seed 中位数。

### 三 seed family 层

对每个任务定义：

\[
\bar p = \frac{1}{3}\sum_{s=1}^3p_s,
\]

\[
R_{\mathrm{family}}
=\sqrt{\frac{1}{3}\sum_{s=1}^3\lVert p_s-y\rVert_{\mathrm{RMSE}}^2},
\]

\[
D_{\mathrm{seed}}
=\sqrt{\frac{1}{3}\sum_{s=1}^3
\lVert p_s-\bar p\rVert_{\mathrm{RMSE}}^2}.
\]

三种冻结风险分数：

1. `native_logvar_mean`：三个 seed 原生 log-variance 的算术均值；
2. `seed_disagreement`：\(D_{\mathrm{seed}}\)；
3. `predicted_magnitude`：三个成员 predicted-effect RMS magnitude 的均值。

主结果变量固定为 `family_rms_error`。程序必须复核

\[
R_{\mathrm{family}}^2
=\lVert\bar p-y\rVert_{\mathrm{RMSE}}^2+D_{\mathrm{seed}}^2
\]

的最大残差不超过 \(10^{-10}\)。另报告 centroid RMSE，但不得把 disagreement 对
family RMS 的确定性下界关系写成对 centroid RMSE 的保证。

## Predictor–uncertainty system 并列评价（Track B）

三种系统都限定为相同的 Norman P1/P2 任务标签：

| 系统 | 风险分数 | 自己的结果变量 | 冻结来源 |
|---|---|---|---|
| GEARS-UQ 三 seed family | native logvar、seed disagreement、magnitude | GEARS-UQ family RMS | E195 |
| GEARS–scGPT post-hoc pair | model disagreement、两种 magnitude | pair mean RMSE | E67/E76b |
| PRESCRIBE integrated predictor（论文主终点） | `-epistemic`、`-aleatoric`、`-combined`、`-magnitude` | `1 - effect Pearson accuracy` | E145 |
| PRESCRIBE integrated predictor（RMSE 敏感性） | epistemic risk、aleatoric risk、combined risk、magnitude | mean-profile RMSE | E96 |

冻结来源哈希：

- E67 P1 task table：
  `d11c88c53d799948b9ebf6d229fd24ea52bfb3ab51e0bda5ca1e3e4ed8b2f74b`；
- E76b P2 task table：
  `5d1d28b39c5f617eceaa5c30fc93ae9e98585466b17b3f5a2c9004607f1bce71`；
- E145 PRESCRIBE paper-endpoint task table：
  `dfc7bcb138aff82e0921158b34dc3ffe4b23b7b02d70ec0afc811fa9cd9a7eb6`；
- E96 PRESCRIBE task table：
  `423de938a4aaaf445187900f958310075322afdf42dd2649018f37100a2c4170`。

Track B 只允许比较无量纲排序和路由量：

- panel 内 Spearman；
- 10%、20%、30% 拒绝预算的 remaining-error reduction；
- high-error recall；
- coverage 0.50–1.00、步长 0.05 的 normalized risk–coverage；
- normalized AURC；
- 20% oracle-normalized utility。

不同系统的原始 RMSE 数值不直接比较。跨系统指标只作描述，不据此声称某种方法
全面优于另一种方法。

PRESCRIBE 论文终点的方向统一如下：

- confidence 与 accuracy 都是数值越大越好；
- 进入统一风险表时固定转换为 `risk = -confidence`、
  `error = 1 - Pearson accuracy`；
- predicted magnitude 在 E145 中按 confidence-like baseline 处理，因此统一表中
  使用 `risk = -predicted_magnitude`；
- E145 是主臂；E96 的 RMSE 结果只用于检查结论是否依赖终点。

已知 E145 的 combined confidence 与 magnitude 排序高度重合，E195 必须同时报告
两者的分数相关和 paired delta，不得只展示 combined confidence 的有利点估计。

## 统计与并列处理

- P1、P2 分开报告；
- 每个面板只有 24 个任务，所有区间均保留小样本不确定性；
- Spearman 与 20% routing utility 做 5,000 次任务 bootstrap；
- 同一面板内比较两种分数时用相同 bootstrap 索引；
- 并列分数按 `task_id` 稳定排序；
- constant score 的 Spearman 写为缺失，并标记
  `undefined_constant_score`，不得改写成 0；
- pooled 48-task 数值仅作为补充，主结论使用 P1/P2 等权 macro，不把两个面板
  混合制造显著性。

## 实现 gate

1. 2 个面板 × 3 个 seed 全部训练和导出成功；
2. 每个 `panel × seed` 恰有 24 条严格 PredictionRecord；
3. P1/P2 任务分别与冻结 manifest 完全一致，且交集为 0；
4. 所有部署风险分数在读取目标真值前写入原始记录；
5. 每个 `panel × seed` 的 native logvar 必须显式报告动态范围和不同值数；
6. 任何 constant/NaN native logvar 均触发 `NON_ESTIMABLE`，不能静默删除；
7. 三 seed 真值数组逐任务一致，最大差不超过 \(10^{-7}\)；
8. family 平方恒等式最大残差不超过 \(10^{-10}\)；
9. E67、E76b、E96 的任务连接必须一对一且零缺失；
10. 全部 input/output SHA-256、训练命令、环境版本、GPU 与耗时进入运行记录；
11. 代码不得依据任何结果改变 seed、epoch、基因面板、分数方向或主结果变量。

gate 检验复现完整性与可估计性，不要求经验相关必须为正。

## 冻结输出

- `E195_STATUS.json`
- `tables/E195_SINGLE_SEED_TASKS.csv`
- `tables/E195_FAMILY_TASKS.csv`
- `tables/E195_ASSOCIATION.csv`
- `tables/E195_PAIRED_SCORE_DELTAS.csv`
- `tables/E195_ROUTING_METRICS.csv`
- `tables/E195_RISK_COVERAGE.csv`
- `tables/E195_SYSTEM_COMPARISON.csv`
- `tables/E195_DYNAMIC_RANGE_AUDIT.csv`
- `tables/E195_INVARIANT_AUDIT.csv`
- `tables/E195_SCORE_LOCK_AUDIT.csv`
- `tables/E195_SUPPORT_EXPOSURE_AUDIT.csv`
- `tables/E195_RAW_ARTIFACT_HASHES.csv`
- `tables/E195_INPUT_HASHES.csv`
- `tables/E195_RUNTIME_ENVIRONMENT.csv`
- `figures/E195_native_uq_comparison.png/.pdf`
- `reports/E195_REPORT.md`
- `reports/E195_INTERPRETATION.md`
- `reports/RUN_RECORD.md`

本目录的 `raw_gears/`、模型权重和大数组只保存在本地并由 `.gitignore` 排除；远程
保存冻结合同、代码、任务级表、哈希和可复现命令。
