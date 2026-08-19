# SafeConf 证据关卡状态（2026-07-12）

颜色只表示证据是否闭环：绿=已达到当前合同；黄=有结果但合同或基线不完整；红=尚未形成可投稿证据。周老师的原话与投稿补强要求分开记录。

| 关卡 | 状态 | 已有证据 | 欠缺 | 下一动作 |
|---|---|---|---|---|
| 主张与边界锁定 | 绿 | `CLAIM_LOCK_20260712.md` | 后续总账需持续同步 | 新实验不得改主目标追结果 |
| 输入来源、错误归属 | 绿 | E33、E65/E67/E72 strict records、E74/E77/E90 | 旧实验不能自动继承 | 新 adapter 强制写 provenance 与 predictor_name |
| 整列未见基因 + 重复面板 | 绿 | E60/E65、E66/E67、E71/E72、E75/E76、E77、E90、E91–E96 | 原生 uncertainty 已完成但无正信号 | 不再继续第三面板 |
| 基因侧难设置总表 | 绿（可计算）/黄（强基线与模型） | E90 整列；E97–E101 三数据集、13 folds、四 setting、四训练量 | 对 disagreement 稳定；对 magnitude population CI 跨 0；非 GEARS+端到端 scGPT | 补独立域/Cui 映射子集；评估正式模型适配 |
| 小矩阵 / 低覆盖 | 绿（可计算）/黄（增量） | E84 chem；E97/E98 gene 25%–100% 嵌套子矩阵 | gene cluster CI 跨 0；双未见 q80 失效 | 做匹配 setting 的内层校准与外部复制 |
| 整行新 context | 绿（可计算）/黄（正式模型） | E84 chem；E97–E101 gene 13 个整行 fold | embedding/transfer predictors；Santinha 校准失败 | 同合同接正式模型并保留 frozen score |
| 整列新 perturbation | 绿 | 遗传 E77/E90；化学 E84 | chem 幅度更强 | 不在 E84 调分数 |
| context×perturbation 双未见 | 绿（可计算）/黄（增量） | E84 分歧平均ρ=.769 | 相对 magnitude Δρ=.019，区间跨 0 | 封存 |
| 不同扰动类型矩阵 | 绿（可计算）/黄（幅度增量） | E102/E103 Cui 6×41 direct-mapped cytokines | 只覆盖 41/86；vs magnitude cluster CI 跨 0 | 作为独立刺激线，不混 gene 主表 |
| 跨数据集 | 绿（可运行）/黄（增量） | E69 gene；E87/E89 chem | E87 失准；E89 Δ CI 跨 0；E69 仅部分方向赢 magnitude | 封存测试集；接竞品 |
| 化学扰动闭环 | 绿（合同）/黄（增量） | E84/E87/E89；McFarland 负 | 无稳定幅度增量 | 不调 E84/E87/E89 |
| 直接不确定性与选择性预测 | 绿（合同）/黄（有效性） | E85、E91–E96：PRESCRIBE 双面板 48 tasks、23,977 cells | 原生 uncertainty 对自身误差无稳定正相关；相对 magnitude 的 Δρ 均为负且 CI 跨 0 | 封存为直接竞品负结果，不在 P1/P2 调分数 |
| 机制解释与复现包 | 黄 | strict 合同、脚本较完整 | 端到端命令与环境锁定 | 主 setting 后补齐 |
| 投稿攻击清单 | 红 | E68 | 尚未逐项 PASS | 黄色关卡收口后再审 |

## 本轮更新（E90）

已实现并落盘：`docs/实验结果/E90_gene_hard_setting_matrix_20260712/`

- 脚本：`tools/scripts/run_e90_gene_hard_setting_matrix.py`
- 池化 144 任务 col_holdout：ρ_disagree≈0.489，ρ_mag≈0.369，Δ≈0.120，Δ 95% CI **[0.007, 0.238]**（不含 0）
- 诚实缺口：gene 行 holdout / 训练子矩阵 **未做**（单 context + 需重训）

## 本轮更新（E91–E96）

- E91 冻结 Norman 两套互不重叠的 24-task 面板；PRESCRIBE 输出未参与任务选择。
- E92 用本地 whole-human scGPT checkpoint 生成并哈希 60,697×512 基因嵌入；E93 完成两套官方 Step1 预处理。
- 官方 GEARS 层曾按 GO 表静默删除 `IER5L+ctrl`。该运行在测试前中止，适配补丁和中止记录均保留；重建后 P1/P2 DataLoader 都是 24/24。
- E95 使用 PRESCRIBE 原生损失、5 epoch flow 预热、最多 50 epoch 主训练并按作者 early stopping，P1/P2 共评估 48 tasks、23,977 cells。
- E96 主目标为任务平均表达谱 RMSE。epistemic、aleatoric、combined 的双面板平均 ρ 分别为 −.053、−.165、−.056，magnitude 为 .059；三种 uncertainty 相对 magnitude 的 Δρ 均为负，区间均跨 0。

## 本轮更新（E97–E98）

- 回查 Frangieh 原始 h5ad，确认 3 个真实背景；至少 50 cells/pair 时有 189 个共同单基因扰动，形成完整 3×189 矩阵。旧导出把 context 覆盖为统一数据集名，E90 的“Frangieh 单 context”据此更正。
- E97 在不读取表达矩阵、效应或误差的条件下冻结三折合同。每折含 258 train、30 val，以及整行 159、整列 60、双未见 30、随机缺失 30 个 test pair；训练集另有 25%/50%/75%/100% 嵌套子矩阵。
- E98 运行 SourceEffect-scGPTKNN 与 scGPTEmbedding-ContextRidge，共 3,708 个任务行、7,416 条 strict PredictionRecord，issue_count=0。100% 训练量 pooled ρ：校准 pair risk=.693，分歧=.596，magnitude=.643。
- 普通 task bootstrap 的 SafeConf−magnitude Δρ=.050，95% CI [.004,.096]；更保守的 outer-fold+perturbation cluster CI 为 [-.098,.255]。投稿口径只能写正趋势，不能写稳定超过。
- validation q80 在 pooled 上接受 57.7% 并降低平均 RMSE，但在双未见任务上接受集误差反而略高，尚未形成覆盖保证。

## 本轮更新（E99–E101）

- E99 仅按标签与细胞数冻结外部矩阵：Lara ex vivo 5×31、Santinha 5×23、Cui 6×86。`nan`、Noise、NT1 等缺失/无效标签已排除；三份源文件均记录 SHA256。
- E100 在两套外部 gene 矩阵生成 2,760 个任务行、5,520 条 strict PredictionRecord，issue_count=0。基因面板只由 control 细胞选择，原始计数统一 normalize-total 1e4 + log1p。
- Lara pooled 校准 pair-risk ρ=.255，magnitude=.043；Santinha 校准 pair-risk ρ=.176，低于 frozen=.357 和 magnitude=.385。校准分数不能作为跨数据集统一主分数。
- E101 固定 frozen pair risk，不在三份 test 上重拟合。三数据集宏平均 ρ=.425，magnitude=.357，disagreement=.341。相对 disagreement 的 dataset-population cluster CI [.017,.151]；相对 magnitude 为 [-.057,.213]。
- Leave-one-dataset-out 删除 Lara 后，frozen−magnitude 只剩 .008。当前可以写“稳定超过单纯模型分歧”，不能写“稳定超过预测幅度”。

## 本轮更新（E102–E103）

- Cui 86 个刺激中，41 个可经机械字符串规范化直接命中 scGPT 词表；45 个商品名、复合亚基或别名全部排除，没有手工猜映射。E102 按 6×41 重新冻结合同。
- E103 生成 2,832 个任务行、5,664 条 strict PredictionRecord，issue_count=0。pooled 校准 pair-risk ρ=.413，frozen=.303，disagreement=.288，magnitude=.391。
- 校准 risk 相对 magnitude 的 outer-fold+perturbation cluster CI 为 [-.104,.193]。该实验补齐不同扰动类型的可计算性，不提供稳定幅度增量。
- 另做跨 gene 数据集校准可行性核验：能改善 Santinha，但明显损害 Frangieh/Lara，未固化为主方法，也未写成正式正结果。

## 接下来的计算顺序

1. 用训练子矩阵内部模拟整行/整列/双未见，建立 setting-matched calibration；E98/E100 test 继续锁定。
2. 评估 GEARS/端到端 scGPT 在 context-aware 矩阵合同下的适配成本与输入一致性。
3. 对高风险任务做共同失准/单模型失准和生物类别分组；继续保留 Santinha、双未见 q80、chemical 跨域负结果。
3. 补高风险任务的生物分组和失败机制，检查分歧来自真实模型偏差还是共同失准。
4. 摘要只写 CLAIM_LOCK 允许句；E87/E89/McFarland/E85/E96 作边界，投稿攻击清单逐项复核。
