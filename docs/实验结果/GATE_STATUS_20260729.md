# SafeConf 当前实验总账

更新时间：2026-08-01

当前最高事实版本：E197 完成

当前分支：`exp/task-risk-audit-20260611`

## 当前结论

周老师提出的三类难度实验已经形成可审计任务，但“做完设置”和“方法在该设置下
有效”是两件事。下面同时保留通过、失败和尚未闭合的部分：

1. E189 用同一 scGPT–GEARS 六成员合同覆盖小训练子矩阵、随机缺一格、整行、
   整列和行列双未见；
2. E190 在 Adamson 训练预测器，直接预测 Replogle 的 47 个共同基因、692 个任务；
3. E84/E87/E89 已覆盖化学扰动的小矩阵、四象限和跨数据集；
4. E191 固定 10%/20%/30% 复核预算，检验证书是否真能节省人工复核资源。
5. E192 在不读取目标扰动表达的条件下，锁定 Adamson K562→Replogle RPE1
   跨研究、跨细胞系任务，并按事前双 gate 裁决。
6. E193 在 RMSE、effect cosine 和 effect Pearson 三种 Hilbert 几何中复核同一
   注册家族证书，并与 magnitude、source magnitude 和 source-shift 作配对比较。
7. E194 冻结 55/50/50 个 family 构成场景，检查重复模型、架构失衡、成员遗漏和
   合成攻击，明确证书对象随 family 成员及权重改变。
8. E195 在同一 GEARS-UQ 预测与自身误差上直接比较 native logvar、seed
   disagreement 和 predicted magnitude；
9. E196 在同一 CPA predictor 与自身 RMSE 上比较原生潜空间支持距离和 magnitude；
10. E197 按唯一 target gene 运行 Systema-inspired 与官方 scPertEval centroid
    协议，拆开 MSE、Pearson-Δ、centroid accuracy 和 retrieval rank。

两类确定性 family-error 下界继续保持零违例。经验排序不具有普适性：random pair、
unseen column、E190 和 E192 的复核预算中有用，unseen row 较弱，double unseen
中会产生负收益。E192 的预算收益为正，但相关区间跨 0，按冻结规则仍返回
`ABSTAIN`。

E193 的 867 个独特任务形成 2,601 个几何任务实例，family RMS 与 worst-member
下界继续保持 0 违例。方向型经验排序没有跨细胞系运输：E190 K562 cosine 有信号，
E192 RPE1 cosine 接近 0、Pearson 为负。E193 因而加强了多几何确定性证书，没有
得到通用方向型风险路由器，也不改变 E192 的 `ABSTAIN`。

E194 进一步说明：证书不能脱离 family 定义解释。867 个任务上共形成 310 个
`dataset×geometry×scenario`、134,385 条逐任务记录；family/worst 下界仍为 0
违例，492/492 个复现与治理不变量通过。但同架构 seed 分歧很弱，A0 的主要
diversity 来自 scGPT 与 GEARS 的架构间差异；absolute RMSE 下的对称合成成员可在
保持质心误差不变时使 mean diversity 增加约 286%–300%，即达到 A0 的
3.86–4.00 倍。故 prediction hash、lineage 和架构权重必须写入方法合同。

E195/E196 的直接竞品复现没有得到新的稳定优势。GEARS native logvar 在两个 Norman
面板均为正相关，但点估计低于 magnitude，配对区间跨 0；CPA cosine/Euclidean
support distance 的宏平均相关为 0.390/0.346，magnitude 为 0.591，两个配对差区间
均在 0 以下。E197 又显示误差定义会改变方向：K562 上风险量与 effect MSE 正相关，
与 Systema-inspired transported Pearson error 负相关；RPE1 的对应区间普遍跨 0。

因此当前方法定位是：

```text
预测前冻结 scGPT / GEARS 多种子家族
                 ↓
始终输出具有严格含义的 family-error lower certificate
                 ↓
只有经过外部验证的 setting 才启用经验复核排序
未验证或 double-unseen setting 明确返回“排序未验证”
```

不能再写成“一个通用 confidence score”“disagreement 普遍优于 magnitude”或
“对任意误差定义都能保持同一排序”。

## 周老师的问题是否答完

详细拆解、输入来源和下一项实验见
[周老师问题证据矩阵（2026-08-01）](./周老师问题_证据矩阵_20260801.md)。

| 问题 | 正式证据 | 当前状态 | 结果 |
|---|---|---|---|
| 风险和哪个真实误差比较 | E189/E190 RMSE；E193 多几何；E197 Systema/scPertEval | 已回答 | 已拆成绝对误差、方向误差、centroid 与 retrieval；结论依端点变化 |
| 实际误差是否依赖预测模型 | E194 family 构成；E195 GEARS-UQ；E196 CPA；E197 逐预测器 | 已回答边界 | family 证书不能指出哪一个模型错；共享任务难度与模型特异失效同时存在 |
| predicted magnitude 是否需要目标真值 | 16 个 E189 release 与 E190 release 均先预测、query 无 `y` | 已回答 | 不需要；它是冻结预测 effect 到零向量的 RMSE |
| holdout pair 打分是否偷看该 pair 的扰动表达 | E189/E190/E192 pretruth release 与 E197 输入审计 | 已回答 | 不读取；严格实验只在预测和分数落盘后读取目标 truth |
| 小训练子矩阵 | E189 每个已见扰动只给 1/2/3/5 个背景 | 已完成 | setting 已跑，结果随支持量和缺失方式变化 |
| 随机 pair 是否太简单 | E189 同表比较 random、row、column、double | 已完成 | 证明确实太乐观 |
| 整行、整列、双未见 | E189 四 donor 轮换，13,440 个任务实例 | 已完成 | setting 闭合；double-unseen 的经验排序可失败 |
| 一个数据集训练、另一个数据集预测 | E190 Adamson→Replogle K562；E192 Adamson K562→Replogle RPE1 | setting 已完成，效能有边界 | 两个目标环境均完成；预测器只轻微优于或未明确优于简单基线 |
| 未见扰动能否使用结构/知识相似性 | E84/E87/E89 的 RDKit Morgan/SMILES；E196 CPA latent distance | 部分完成 | 化学侧已实现；基因侧仍缺 TxPert 等真实知识图模型同协议接入 |
| 基因、化学和组合扰动 | E189/E190；E84/E87/E89/E196；E164/E165 | 覆盖但深度不同 | chemical 中 magnitude 更强；组合扰动不并入单基因结论 |
| 有限复核预算是否有收益 | E191 | 已完成 | 有收益，但依赖 setting |

逐项状态以本表为准；旧版可追溯到
[E192 后老师要求矩阵](./E192_adamson_to_replogle_rpe1_locked_transfer_20260729/ADVISOR_REQUIREMENTS_AFTER_E192.csv)。

## E189：同模型笛卡尔缺失实验

- 4 个 donor 轮换面板；
- 4 个训练支持量；
- 4 种缺失 setting；
- 3 个 scGPT + 3 个 GEARS；
- 13,440 个任务实例、3,360 个独特生物任务；
- family RMS 下界违例 0，worst-member 下界违例 0。

support=5 时，六成员优于 zero-effect 的成员比例：

| setting | 比例 |
|---|---:|
| random pair | 51.70% |
| unseen row | 48.81% |
| unseen column | 49.60% |
| double unseen | 47.81% |

全部任务平均 RMSE：scGPT 0.11404、GEARS 0.11848、六成员 family 0.11638、
zero-effect 0.11425。scGPT 只有很小改善，GEARS 拖累 family。

diversity 与 family error 的关系：

- random pair：ρ=0.368–0.412；
- unseen column：ρ=0.210–0.247；
- unseen row：接近 0；
- double unseen：ρ=-0.349 至 -0.241。

这些结果表明，分歧与 family error 的关联会随 missingness setting 改变，不能自动
解释为单模型正确概率。

## E190：基因侧直接跨研究预测

选择 Adamson 2016→Replogle 2022，因为两者都是 K562 CRISPRi，存在 59 个元数据
共同扰动；冻结条件后保留 47 个基因、692 个 `(batch, gene)` 任务。Primary
CD4→Sunshine 没有冻结靶点交集，不做缺失填充式硬拼。

prediction release 在提交 `75a71fa` 双远程后才读取 4,959 个目标扰动细胞。

| 预测器 | 平均 RMSE | 任务胜率 vs zero |
|---|---:|---:|
| scGPT | 0.25826 | 57.08% |
| GEARS | 0.23850 | 60.69% |
| 六模型 family | 0.24919 | 59.83% |
| source-effect | 0.23706 | 62.43% |
| zero-effect | 0.25828 | — |

按 47 个基因整簇后，family 相对 zero 的 RMSE 差为 -0.00254，
95% CI [-0.00675, 0.00110]；source-effect 为 -0.00734，
95% CI [-0.01631, 0.00028]。不能声称跨研究预测在基因层面明确胜过简单基线。

diversity 与 family error 的 ρ=0.424，95% CI [0.135, 0.632]；
predicted magnitude 为 0.420。分歧有信号，但没有超过幅度。

## E191：有限复核预算

E190 的 20% 复核预算：

| 风险量 | high-error capture | error lift | oracle utility |
|---|---:|---:|---:|
| diversity lower bound | 47.48% | 1.150 | 0.443 |
| predicted magnitude | 47.48% | 1.149 | 0.441 |
| source-effect magnitude | 48.20% | 1.160 | 0.473 |
| 随机期望 | 20.09% | 1.000 | 0 |

证书能把复核资源集中到高错误任务，但没有形成相对 magnitude 的独特优势。

E189 的 16 个 `support×setting` 层中，diversity utility 相对 magnitude 为 8 胜
8 负。random pair 和 unseen column 有正收益；double unseen 中两者均为负收益。

## E192：锁定的跨研究、跨细胞系确认

E192 的目标 RPE1 扰动表达此前没有参与模型训练或 E191 开发。任务基因先限定为
E190 已冻结的 47 个候选，再只用 RPE1 元数据得到 21 个基因、175 个任务、53 个
批次。六模型预测提交 `f0dfd46` 并推到双远程后，才读取 1,086 个目标扰动细胞。

- family RMS / worst lower violation：0 / 0；
- Hilbert identity 最大残差：`6.44e-10`；
- diversity–family RMS：ρ=0.300，95% CI [-0.040, 0.580]；
- 20% 复核预算：high-error capture 62.86%，utility=0.696，
  95% CI [0.113, 0.872]；
- predicted magnitude 的 20% utility=0.725，仍略高于 diversity；
- 经验排序激活 gate：**FAIL / ABSTAIN**。

失败原因不是预算收益没有信号，而是事前要求的相关系数区间下限大于 0 未满足。
开真值后不能删除该条件。E192 支持“证书始终输出、经验排序按 setting 保守关闭”
的系统设计，不支持“diversity 普遍超过 magnitude”。

## E193：多几何证书与方向型误差

E193 使用 E190/E192 已经打开的真值，是明确标注的 post-truth metric robustness，
不冒充新的前瞻确认。

- 867 个独特目标任务，2,601 个 `task×geometry` 实例；
- absolute RMSE、effect cosine、effect Pearson；
- family RMS / worst lower violation：0 / 0；
- 最大恒等式残差：`6.66e-16`；
- 旧 RMSE 表最大复算差：`1.41e-08`；
- 方向向量无效任务：0。

E190 K562：

- cosine diversity–error ρ=0.568，95% CI [0.278, 0.783]；
- cosine 20% utility=0.782，95% CI [0.479, 0.922]；
- Pearson diversity 相关区间跨 0，20% utility=-0.026；
- Pearson 中 source-to-family-centroid distance 的 20% utility=0.634。

E192 RPE1：

- cosine ρ=0.048，95% CI [-0.160, 0.290]；
- Pearson ρ=-0.210，95% CI [-0.507, 0.039]；
- 两种方向几何的 diversity 20% utility 区间均跨 0。

这使主张进一步收窄为 `metric-aware registered-family certificate`。effect-vector
cosine/Pearson 不是 Systema exact；完整 Systema/scPertEval、原生 GEARS-UQ、CPA
uncertainty 与 PRESCRIBE 同协议对照仍是投稿阻断项。

## E194：预测家族治理与防操纵压力测试

E194 是使用已解封 E190/E192 真值的 post-truth governance stress，不增加独立
外部确认。

- 两个 target、三种几何、310 个 family 场景、134,385 条逐任务记录；
- family RMS / worst lower violation：0 / 0；
- 最大平方恒等式残差：`6.66e-16`；
- E193 逐任务复现、架构内/架构间方差分解、重复成员治理和成员哈希：
  492/492 项通过；
- governed duplicate 将 A0 的质心、diversity 和 error certificate 恢复到数值
  误差范围内；
- K562 的单架构 seed-only diversity 对固定 A0 error 很弱：
  scGPT ρ=-0.081、GEARS ρ=0.156；A4 架构质心差为 ρ=0.429；
- RPE1 对应值为 0.328、0.185、0.294，21 个基因簇下区间仍较宽；
- C4 的自身 family-error 相关会随合成分歧被结构性抬高，但对固定 A0 error 的
  相关几乎不变，证明跨 family 比较必须固定结果对象。

因此，最稳主张改为 `registered weighted-family certificate`。只有满足同一冻结
输出合同、成员哈希唯一且权重事前确定的 family 才是一个可解释对象。新增成员必须
形成新版本 family，不能事后并入 A0。

## E195：GEARS 原生学习型误差代理

E195 在两个事先固定的 Norman 面板上重新训练三种子 GEARS-UQ；prediction、native
logvar 和 magnitude 先锁定，再读取测试真值。每个面板 24 个任务。

- native logvar mean：P1 ρ=0.412，95% CI [-0.046, 0.731]；P2 ρ=0.454，
  95% CI [0.029, 0.762]；
- predicted magnitude：P1 ρ=0.687；P2 ρ=0.680；
- native logvar 相对 magnitude 的配对 Δρ：P1 -0.275、P2 -0.226，两个区间均跨 0；
- seed disagreement 更弱，单 seed 的 native logvar 波动明显；
- PRESCRIBE combined confidence 与 magnitude 的 Spearman 为 0.997/0.994，增量很小，
  且结果依赖 effect-Pearson 或 RMSE 终点。

因此 GEARS logvar 是候选风险信号，不是已校准预测方差，也没有稳定超过 magnitude。
E195 是旧真值上的 post-truth 竞品复现，不是新的外部盲测。

## E196：CPA 原生训练支持距离

E196 不重训 CPA，严格加载 E84 八个 formal manifest 的冻结权重，在同一个 CPA
predictor、自身 RMSE 和相同任务上比较训练条件最近距离与 predicted magnitude。

- cosine distance：宏平均 ρ=0.390；相对 magnitude 的 Δρ=-0.201，task-cluster
  描述性区间 [-0.358, -0.047]；
- Euclidean distance：宏平均 ρ=0.346；Δρ=-0.245，task-cluster 描述性区间
  [-0.399, -0.092]；
- magnitude：宏平均 ρ=0.591；
- 153/153 不变量通过，参考条件、control 最近邻与负结果均完整保存。

CPA latent distance 确实包含训练支持信息，但在该同 outcome 审计中明确弱于
magnitude。它不是预测方差、误差概率或理论下界。

## E197：Systema 与 scPertEval 多指标审计

E197 使用已经开封的 E190/E192 旧结果，性质为 `POSTTRUTH_EXPLORATORY`。它先按
目标细胞数把 batch×gene 合并为 47/21 个唯一 target-gene centroid，再运行官方
scPertEval 的 pearson、pearson_ctrl、pearson_pert、MSE、rank 和 transpose-rank。
现有预测没有单细胞群，因此没有运行 MMD、Energy、Sinkhorn、DE、WMSE，也没有复制
均值伪造预测细胞。

- 12 个 predictor、10,404 条 Systema task 指标、4,896 条 scPertEval 分数；
- 47/47 formal gates、19/19 输出哈希通过；官方分数与独立公式最大差
  `2.22e-16`；
- E190 family centroid MSE=0.0250，低于 train-only matching=0.0264 和
  zero-effect=0.0315；但 effect-space centroid accuracy=0.606，低于 matching=0.728；
- E192 family centroid MSE=0.0876，zero-effect=0.0894，绝对改善很小；retrieval
  rank=0.167，优于 zero-effect=0.495；
- E190 predicted magnitude 与 family-member effect MSE 的 gene-equal ρ=0.440，
  区间 [0.160, 0.672]；同一风险量与 transported Pearson error 的 ρ=-0.685，
  区间 [-0.817, -0.478]；
- E192 两类关联区间均跨 0。

这直接回答“和什么误差相关”：MSE、相关形状、centroid discrimination 和 retrieval
不是同一个端点，模型与风险量的排序可以改变。E197 不改变 E190/E192 的原 gate。

## 与 E176–E186 主证书证据的关系

E189–E197 不覆盖旧裁决：

- E176/E177/E180/E182 四研究仍是双侧证书与上界覆盖主证据；
- E182 的 16/20 注册 FAIL 继续保留；
- E183 的 666/737=90.37% 仍只是描述性合并；
- E184 的理论与竞品归属不变；
- E185/E186 的最小复现和完整性审计不变。

新增实验解决的是老师追问的矩阵难度、真实预测器跨研究迁移、跨细胞系确认、决策
收益、family 构成边界、原生竞品不确定性和多指标评价，不重新调 SafeConf 公式。

## 当前技术判断

可控的计算证据已经形成较完整的方法骨架：

- 方法对象明确；
- 同模型难度矩阵齐全；
- 基因和化学均有跨数据集结果；
- predictor effectiveness、zero-effect、magnitude、source-effect 均同表比较；
- GEARS-UQ、CPA native distance、PRESCRIBE 复现与 SafeConf 的正负结果均已留存；
- Systema/scPertEval centroid 合法协议已接入，并明确阻止用均值伪造单细胞分布；
- 失败 setting、原始任务表、模型预测、真值读取顺序和 SHA-256 均保留；
- 有真实复核预算指标，而不只报告相关系数。
- 有一项未参与排序开发的 RPE1 锁定目标，且失败 gate 没有被事后改判。

2025–2026 新文献提高了投稿门槛，当前不能仅凭实验数量称为“稳定二区”。任何期刊
录用也不能由实验代码保证。现在最关键的缺口是：

1. E195/E196 已做 same-outcome 直接对照，但 PRESCRIBE、GEARS-UQ、CPA 与 SafeConf
   仍未在同一 predictor family、同一 split 和同一 outcome 上全部闭合；
2. E197 已接入合法 centroid/rank 协议；WMSE、DE 与 population distance 必须等到
   真实预测细胞或合法模型分布，不能从均值补造；
3. 将注册家族扩展到 scGPT/GEARS 以外的真实机制。优先接入公开 TxPert 及其
   batch-matched control、general baseline、retrieval 与 split-half reproducibility；
4. 在一个新的外部数据上，预测、主要误差、metric gate 和停止规则全部事前冻结。
   已下载的 `arch1` 只有 H1 hESC 一个 context，适合外部 protocol calibration 与
   unseen-perturbation，不足以单独充当 row/cross-context 证据；
5. 用冻结证书选择任务后获得真实后续实验或机制验证。

服务器可继续完成第 1–4 项中具有公开数据、代码和可兼容输出的计算部分。第 5 项
如果指新增湿实验或真实后续验证，则需要课题组的材料、人员和实验条件，不能由计算
实验替代。

## 当前阅读顺序

1. [E197 多指标结论](./E197_systema_scperteval_centroid_audit_20260730/reports/E197_INTERPRETATION.md)
2. [E196 CPA 原生距离](./E196_cpa_native_support_distance_20260730/reports/E196_INTERPRETATION.md)
3. [E195 GEARS-UQ](./E195_native_gears_uq_norman_p1p2_20260730/reports/E195_INTERPRETATION.md)
4. [E194 family 治理解释](./E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md)
5. [E193 多几何证书解释](./E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md)
6. [E192 RPE1 锁定确认](./E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md)
7. [E191 决策收益解释](./E191_certificate_decision_utility_20260729/reports/E191_INTERPRETATION.md)
8. [E190 跨研究解释](./E190_adamson_to_replogle_direct_transfer_20260729/final_evaluation/reports/E190_INTERPRETATION.md)
9. [E189 老师问题解释](./E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md)
10. [2026-07-29 期刊与文献定位](../投稿准备/期刊与文献定位_20260729/README.md)
11. [E186 投稿前完整性审计](./E186_presubmission_integrity_audit_20260724/reports/E186_REPORT.md)

旧 gate 只用于追溯，不再覆盖本文件。
