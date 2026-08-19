# E170｜Primary CD4 未读目标多面板精度复现

## 为什么做

E168 在一位完整留出供体的 200 个目标上得到小幅正点估计，但置信区间跨 0，正式判定为 `NO_CONFIRMATION`。E169 进一步发现，同一 state 和同一 seen/unseen 层内，context similarity 与 support 都是常数，SafeConf 实际排序只剩 model disagreement。E170 不修改 SafeConf、不换阈值，也不把 E168 的 200 个已解封目标并入新显著性检验；它用 800 个仍未读取 targeting X 的目标检验该小效应能否在更高目标样本量下复现。

## 一次性冻结的四个面板

- 从 E168 已冻结的 5,510 个 label-free eligible targets 中排除 E168 primary 200，剩余 5,310 个。
- 只按 `E170_FRESH_TARGET_MULTIPANEL_V1` 的 SHA-256 身份哈希选择前 800 个，分为 P01–P04，每个 200 个且互不重叠。
- 每个面板再按面板专属 SHA-256 固定 40 个 `COLUMN_UNSEEN`；其余 160 个为 `DONOR_UNSEEN_ONLY`。
- 选择不使用 targeting X、细胞数、总 counts、扰动效应、DE、guide efficacy、模型误差或 E168 目标表现。
- 四个面板的任务、可读行、模型、seed、gate、统计终点一次性冻结。禁止根据 P01 结果决定是否继续 P02–P04。

## 模型与风险分数

每个面板独立构造 512-gene panel：200 个注册 target genes 加 312 个只由两个 train donors 的 NTC 表达选出的背景基因。scGPT 与 GEARS 各运行 seeds 3407/3408/3409；test query graph 不含 `y`。风险分数、z-score reference、RIAG G1–G5 和 E168 完全相同，不加入 E169 看过 truth 后提出的新特征。

## 主要终点

- 新的 confirmatory population 仅为 E170 的 800 个目标、2,400 个 test tasks。
- 每个 panel×state 单独计算 tie-aware AURC；主要效应是 12 个 panel×state 的等权平均 `AURC_magnitude − AURC_SafeConf`。
- cluster bootstrap 以 target gene 为簇，在每个 panel 内分层重采样 10,000 次，三个 states 同进同出；paired permutation 以 target 为单位交换 SafeConf/magnitude，100,000 次。
- `TARGET_REPLICATION_PASS_NONTRIVIAL` 要求四个 pretruth gate 全部 PASS；全部 800 targets 的 delta>0、95% CI 下界>0、单侧 p<0.05、至少 8/12 panel×state delta>0；随后 seen 640 targets 也必须 CI 下界>0 且 p<0.05。其余结果原样记为 no confirmation 或 partial support，不换面板。

## 解释边界

E170 是同一公开数据、同一 test donor 上的未读目标复现，可以提高目标层面的估计精度，不能提供新的 donor/study 生物学重复。即使通过，也只能支持“在该 donor/study 和指定 scGPT–GEARS 预测器上，相对 magnitude 的目标排序增量”；独立队列和真实湿实验验证仍是更强投稿所需的另一层证据。
