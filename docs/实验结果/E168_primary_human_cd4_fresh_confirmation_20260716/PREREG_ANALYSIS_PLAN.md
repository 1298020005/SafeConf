# E168｜Primary Human CD4+ T-cell fresh external confirmation

## 研究问题

在 SafeConf、RIAG v2、上游模型和所有判定线固定后，使用一位完整留出的真实供体，检验 SafeConf 能否比预测幅度更有效地优先发现 scGPT–GEARS 双模型的高误差扰动预测。

本实验是一个新的公开数据集上的 prospective-style sealed evaluation。它不是新做湿实验；四位供体中只有一位最终 test donor，因此 600 个任务不能当成 600 位独立受试者。

## 信息边界

- metadata freeze 只解码 `obs/10xrun_id`, `obs/donor_id`, `obs/culture_condition`, `obs/guide_id`, `obs/perturbed_gene_name`, `obs/perturbed_gene_id`, `obs/guide_type`, `var/gene_ids`, `var/gene_name`。
- `X`、`layers`、`n_cells`、`total_counts`、`log10_n_cells`、全部 `keep_*` 值均未读取。
- DE、guide knockdown、显著性和效应筛选文件不下载、不打开。
- 下载 H5AD 和计算整文件哈希只搬运字节；首次解析表达值发生在下一阶段的隔离 asset builder。
- test donor 的 non-targeting control 是预测时给定的基础状态，可在 pretruth 阶段读取；test donor 的 targeting rows 必须等 RIAG v2 snapshot 提交后才解封。

## 固定任务

- 状态：Rest、Stim8hr、Stim48hr。
- donor：2 train、1 validation、1 test；角色通过 run-aware 可行分配中的最小 SHA-256 决定。
- 靶基因：用官方 expression-independent sgRNA design/annotation、表达基因身份轴、冻结 scGPT 词表，以及每条 guide 在 12 个 donor×state 中是否存在身份行，定义 label-free assay-available universe；随后按 SHA-256 固定 200 个。H5AD 中 test targeting 的 `n_cells`、表达、效应和显著性均不参与选择。身份可用性可能产生轻微 availability selection，必须披露。
- 其中 40 个 `COLUMN_UNSEEN` 在 train/validation 阶段都不读取 targeting X；160 个 `DONOR_UNSEEN_ONLY` 用两个 train donor 监督训练、一个 validation donor 选 epoch，最终检验整供体迁移。
- test：200 targets × 3 states = 600 tasks；三种状态分别形成实际排序 batch。

## 固定表达定义

同一 guide×donor×state 的原始 UMI 行先求和，再计算 `log1p(1e4 * counts / library_sum)`；同 donor×state 的 non-targeting guide 原始计数先合并再归一化。单 guide effect 是 targeting profile 减 matched NTC，target effect 是预先合格的共同 guides 等权平均。test guide 一致性不用于删任务。

## 固定 512-gene panel

200 个 target genes 全部纳入；其余 312 个只根据两个 train donors 的六个 NTC 状态平均表达补足，并限制在固定 scGPT vocab。相同表达按 Ensembl ID 排序。该结果只称为 512-gene reduced-panel pseudobulk benchmark。

## 上游预测器

- scGPT：whole-human checkpoint，seeds (3407, 3408, 3409)，最多 10 epochs，Adam lr=1e-4，batch=16，patience=3。
- GEARS：E112 架构，seeds (3407, 3408, 3409)，最多 40 epochs，Adam lr=1e-3，batch=16，patience=6；GO 图来自冻结的外部 E107 GO 文件，coexpression 只用两个 train donors 的 NTC profiles 构建。
- test graph 不含 `y`；必须使用 query-only forward，禁止复用 E112 同时构造 test truth 的接口。
- 强基线：NoChange；160 seen targets 另报同 state 的 TrainDonorEffectMean。

## SafeConf 与 pretruth gate

固定 confidence 为 `z(context_similarity_max) + z(log1p(perturbation_support_count)) - z(model_disagreement_rmse)`，risk 为其相反数。它直接调用 `protocol_v0_2.zscore_by_ref`，reference 是 960 个监督训练 tasks 的部署前特征，不改成 validation reference。`perturbation_support_count` 是进入监督训练的 donor×state contexts 数，因此 seen=6、column-unseen=0，不用细胞数替换。z 的中心为 train median，scale 为 train IQR；IQR 无效或 ≤1e-9 时退到 train std，std 仍无效时 scale=1。40 个 unseen validation truth 保持不读。预测幅度是双模型 seed-mean ensemble 的 512-gene RMS。

`context_similarity_max` 明确定义为 query donor×state 的 512-gene NTC 向量，对六个 train donor×state NTC 向量的最大 cosine similarity；`model_disagreement_rmse` 是 scGPT 三 seed 均值效应与 GEARS 三 seed 均值效应的 512-gene RMSE。正式 risk 使用两个 family seed-mean；G4 的 seed-risk 则固定配对同编号 scGPT/GEARS seed。所有 z 在三种 state 合并的 train reference 上一次性计算，之后才按 state 分批排序。

RIAG v2 按每个 state 分别检查：源文件/访问隔离、risk 非退化、cutoff ties、每个 predictor 的任务依赖、三 seed 排名稳定性及与 magnitude 的同序性。risk/prediction/magnitude 的 operational tolerance 都固定为 1e-6，rounding 为 NumPy ties-to-even。low-risk coverage 登记 20%、25%……95%、100%，另登记最高风险 review 20%。cutoff 穿过并列组时记 `TIEBREAK_REQUIRED`，不加 jitter。

G2 要求每个 state 至少两个量化 risk levels 且 population std>1e-6。G3 对 3 个 scGPT seed、3 个 GEARS seed及两个 family seed-mean 分别要求向量有限、至少两个量化后不同的任务向量、至少一个坐标跨任务 std>1e-6。G4 将同 seed 的 scGPT/GEARS 组成 seed-risk，要求三组 pairwise Spearman 中位数≥0.5，并以 target gene 整簇 bootstrap 2,000 次后的 95% CI 下界>0。G2 与 G4 必须在 `all_200` 和 `seen_160` 两个 registered strata × 3 states 分别通过，防止 support 的 160/40 二分掩盖 seen targets 内的分歧塌缩；column-unseen 40 只作描述。G1/G2/G3/G4 任一失败，test targeting X 保持未读并记录 `PRETRUTH_ABORTED`；不换 seed、不抖动分数、不降低门槛。risk 与 magnitude 同序不阻止解封，但状态必须写成 `EVALUABLE_BASELINE_EQUIVALENT`，不得宣称增量价值。

## 主要终点

每个 task 的 loss 是 ensemble effect 与 truth effect 的 512-gene RMSE。coverage 固定为 0.20–1.00、步长 0.05，AURC 对 coverage 宽度 0.8 归一化；ties 使用 E167a 的 tie-average/best/worst legal order，primary 使用 tie-average。主要效应为三个 state 等权平均的 `AURC_magnitude - AURC_SafeConf`。

- 按 target gene 整簇 bootstrap 10,000 次，seed=2026071681；一个 target 的三个 states 同进同出。
- 按 target gene 整簇交换 candidate/magnitude，单侧 permutation 100,000 次，seed=2026071682。
- `CONFIRMATION_PASS_NONTRIVIAL` 要求：全部 pretruth gates 通过；三个 states 上 ensemble 胜 NoChange 的任务比例都 >0.5；全 200 targets 的 delta>0、CI 下界>0、p<0.05、至少 2/3 state delta>0；随后 160 seen targets 也需 CI 下界>0 且 p<0.05。
- 全 200 通过但 160 seen 不通过，记为 `PARTIAL_SUPPORT_STRATIFICATION_ONLY`；其余为 `NO_CONFIRMATION`。

所有 ties、失败状态和负结果原样保留。`deployment_authorized=false`；本实验不授权临床或自动湿实验决策。

## 解释边界

三种 state 分批排序后，context similarity 在单个 state 的 200 个 targets 内完全相同，因此不会改变该 batch 的名次；E168 实际检验 disagreement 与 support 两部分，不能单独证明完整 context component。必须同时报告 disagreement-only、support-only、context-only 与 magnitude 的 tie-aware AURC。TrainDonorEffectMean 若优于深度模型，只能把结论写成“对指定上游预测器的风险分诊”，不能写成预测模型达到 SOTA。
