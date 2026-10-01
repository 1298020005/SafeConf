# SafeConf 独立证据复核，2026-10-01

本文件是审稿与诊断记录，不是论文正文。工作树基线为 `8fa4152ad33985fa0444823652f41d102d50dbb9`。我只读取已有代码、冻结向量与 CSV；新增文件仅在本目录。共同 2840 基因结果均为已经见过数据的复核，不能称为新的封存确认。

## 判断

当前资产支持“合法历史信息能预测有限样本 centroid 的观测误差”，以及“同一上游 family 内，真实 Source 错误监督比历史距离更有信息”。它们没有同时支持独立研究上的 Source 增量、历史效应内容超过数量信息、Shared 对便宜目标监督的额外收益。原生预测与本数据真值之间暂未发现明显归一化错配；但是 Public 的 guide 等权效应与目标 cell 等权效应不一致，仍有一个具体、低成本的估计目标对照没有排除。

完成更多下游拟合不等于二区/CCF-B 就绪。合理的转向是审查“观测误差排序何时代表预测器特异失败”，并测试内容信息在相同测量目标下是否仍有增量。这个方向只有在区别于既有共同失败/噪声上限研究的条件下，才可能形成方法或验证贡献；目前不能先宣布发现首创。

## 四个 P0 的逐项结论

数值来自本目录 `P0_PRIMARY_EVIDENCE_ROWS.csv`，该表保留原 CSV 路径。区间为原实现的 5000 次配对 gene 簇 bootstrap。这里只取第一固定种子；相同种子的重复拟合不是独立研究。

| P0 | 最强可支持结论 | 不能支持的结论 |
|---|---|---|
| 1. Public 超过 prediction-only | 外部 Learned_hgb − Prediction_hgb 的 ΔU20 为 +0.703786，CI [0.524452, 0.847324]；这是 Public 信息包相对该 P-only 学习器的增量 | 同 family 两方向 Δ 为 +0.024181、+0.031029，但两 CI 均跨零。外部 support-only 0.827129 超过历史距离 0.729672，因此不能把 Public 信息包的收益全归给效应内容 |
| 2. 同参照真实标签同时超过加权距离及 shuffled HGB | 同 family 两方向相对 Learned distance 增量 +0.189616、+0.191407，CI 均为正，且相对五个 Source 标签打乱均为正 | 外部相对同参照 distance 为 −0.034419，CI [−0.132144, −0.008644]；对 shuffled0 为 +0.088318，CI [−0.026193, 0.208865]。故外部双条件没有通过 |
| 3. 增加独立簇/预测器的收益是否来自监督内容而非行数 | 原 512 基因资产已登记独立 gene 数、行数、单来源、同任务等行数 pooled、簇加权 pooled、独立模型分数平均；这些是正确的控制设计 | 五顺序所有十条曲线均有负相邻步；两 source predictor 仍属同一 family。Manual full pooled 0.583540 低于单 GAT 0.632051 与单 Exph 0.629335，equal-record pooled 为 0.523990；Learned full pooled 0.551331 与单 Exph 0.549173 接近。没有独立 predictor 多样性必然增益。此处是 512 结果，不能直接充当新 2840 轴上的正确认 |
| 4. 同 C-error 预算下谁增加信息 | full budget PublicTarget_HGB 超过 TargetOnly_HGB +0.691494，CI [0.488824, 0.960460]；Public 在该目标学习器中有增量 | SharedTarget − PublicTarget 为 −0.000570，CI [−0.037496, 0.056238]；full PublicTarget 对无需反馈的 support rule 为 −0.020518，CI [−0.102483, 0.049494]。没有证明反馈超强历史规则，也没有证明 Shared 与 PublicTarget 互补 |

C validation 的 230 合法风险训练任务/123 簇可以零新增预测调用提供强对照。PublicValidation_HGB 0.755390 相对 Source +0.060137 的 CI 跨零；相对 support-only 反而 −0.071739，CI [−0.154736, −0.019900]。已有验证预测使“必须用 Source 才省调用”不是当前资产上的成立主张。checkpoint/alpha 曾经由完整 validation 选择，因此这些标签也不能称为纯净 upstream OOF。

## 测量与输出合同

### 未发现的具体错误

[共同轴代码](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_safeconf_common_axis_closure.py:95)使用固定 `alpha=.25`：

`p = .25*(generated_state_mean - test_control_mean) + .75*TRAIN_state_mean_effect`

[目标真值](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_safeconf_common_axis_closure.py:153)是同任务全部 TEST treated cells 的 X 均值减相同 TEST context×treatment control；RMSE 在相同注册基因轴上计算。冻结预测按 condition 的生成行数平均。[聚合实现](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/prepare_safeconf_common_gene_predictions.py:53)核验了 gene 顺序与 task 顺序。我的向量复算与注册逐任务 RMSE 的最大差为 2.10e−10，且 543 任务全部覆盖。

原生 McFaline 数据 X 有 log1p 元数据，独立 `layers/counts` 为原始计数。官方预处理采用 `sc.pp.normalize_total(adata)` 默认目标再 log1p，[本地固定官方源码](/home/yyf/archive/external/PerturBench/src/perturbench/analysis/preprocess.py:111)与其 [GitHub 固定提交](https://github.com/altoslabs/perturbench/blob/c84038bc1ea409aa54f3832cfa6f34f5059adf0c/src/perturbench/analysis/preprocess.py)均可核查。官方 LinearModelPipelineControls 对 expression 只 ToDense/ToFloat，[DecoderOnly 的 MSE 与 forward](/home/yyf/archive/external/PerturBench/src/perturbench/modelcore/models/decoder_only.py:116)直接在该 X 尺度训练并输出。base 的额外 normalize/log1p 只对 count_based 输入分支生效。因此没有证据说本次 MC 原生输出需要再 CPM/log1p 或逆变换；那会改变原预测合同。

### 明确存在但尚未隔离的合同差别

1. [Public 构建](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/build_safeconf_common_gene_biology.py:305)先每个 guide 求 cell mean，再对 guide 等权平均；目标 truth 与 DecoderOnly 训练是 cell 等权。`CellWeighted` 方法名称指不同历史实验之间按 n_cells 加权，不代表历史实验内部也按 cells 加权。其距离可以同时混入 guide 组成变化与生物跨状态差别。
2. McFaline 原始官方 preprocessing 在 train/val/test 划分以前对整个 released dataset 做默认总量归一化及 HVG/DE 筛轴。这是官方现成资产的限定，不是本轮偷偷训练于 test error，但也不等于完全仅训练分区决定的部署预处理。
3. Source 与 MC 是相同 gene ID 的不同原生 normalization domains。源学习器标准化与一次 control-RMS 缩放失败，不能排除所有尺度/协变量迁移问题。该诊断 Learned 外部 ΔU20 为 −0.204123，CI [−0.284540, −0.070207]，只能否定这一条固定缩放规则的修复价值。
4. 当前外部对象是 25% DecoderOnly 加 75% TRAIN mean 的冻结混合预测。应按这个实际对象解释风险和 competence；独立发布架构不等于这个组合已有未干预的原生跨研究确认。

## 新增零拟合诊断：相对简单基线的 excess MSE

为了避免固定 20-cell 真值变噪带来的解释问题，我保持全部真值细胞、全部风险分数和全部上游输出，增加了一个已见数据诊断。令 `b` 为冻结 TRAIN state mean effect，`y` 为原完整 TEST effect。

`Δ_i = mean_g[(p_ig-y_ig)^2 - (b_ig-y_ig)^2] = mean_g[(p_ig-b_ig)*(p_ig+b_ig-2*y_ig)]`。

两边实算一致到 1e−12。`y²` 的同一项精确相消；这检查风险是否预测相对基线的模型差错，而不是绝对观测误差。它仍有对 y 的线性采样噪声，b 有训练测量不确定性，预测与真值对共同 control 的系数不同；不能称为无噪 biological error、latent oracle 或完整噪声扣除。alpha 小使混合预测贴近 b 也是需要明示的原因；我同时报告了 raw DecoderOnly 的相同差值。

| context | n | C 与 TRAINmean 的逐任务 MSE rho | C mean MSE | TRAINmean mean MSE | C 的 ΔMSE 均值 | C 比 baseline 更差的任务比例 |
|---|---:|---:|---:|---:|---:|---:|
| a172 | 189 | 0.998694 | 0.000766 | 0.000776 | −0.000010 | 0.058201 |
| t98g | 177 | 0.998262 | 0.000619 | 0.000626 | −0.000007 | 0.175141 |
| u87mg | 177 | 0.999580 | 0.001122 | 0.001142 | −0.000020 | 0.005650 |

对 calibrated excess MSE，Learned_hgb 的 U20 按 context 为 −0.230551/+0.173016/+0.123067，宏平均 0.021844；Learned history distance 为 −0.209787/+0.179573/+0.056932，宏平均 0.008906。raw DecoderOnly 的宏平均对应 0.030218/0.017845。原绝对误差上 0.695/0.730 的大分数没有在这个预测器差错对照中保留。这是描述性敏感性结果，不是无信号的统计证明。

逐任务、context 分数、macro、输入 SHA256 分别在 `EXCESS_MSE_TASK_DIAGNOSTICS.csv`、`EXCESS_MSE_SCORE_DIAGNOSTICS.csv`、`EXCESS_MSE_MACRO_DIAGNOSTICS.csv`、`EXCESS_MSE_INPUT_SHA256.csv`。上下文摘要在 `EXCESS_MSE_CONTEXT_SUMMARY.csv`。复算仅需要冻结 npy/npz 与旧风险 CSV，无新 modelcall、下游拟合或 random split。

## 只保留两项低成本优先动作

**动作 1，父任务执行：将 Public 内部平均方式对齐 cell mean，用固定 Manual 距离作一次零拟合对照。** 读取 `mcfaline23_gxe_processed.h5ad`、官方 split、共同轴 GENE_IDS，在原 public_memory 行序上只汇总 train+val treated cells；重建 `Ecell_j=mean_cells X_j−original_trainval_control_j`。保留原 n_cells、原 pooled control、原 exact context×treatment 排除、原外部 543 任务、原 TEST_CALIBRATED_EFFECTS 与原 error。使用固定 `w_j=n_j/sum n_j` 计算 `Dcell_i=sqrt(sum_j w_j*mean_g[(p_i−Ecell_j)^2])`，与原 guide mean 的 `Dguide` 配对比较并按同 gene 簇 bootstrap。无需 fit Source HGB、public transfer learner 或 CDF。

具体路径：`/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/public_mcfaline_trainval/{public_memory.parquet,effect_vectors.npy,control_vectors.npy}`；`.../common_gene_axis/{TEST_CALIBRATED_EFFECTS.npy,TEST_CONTROLS.npy,TEST_TRUE_EFFECTS.npy,GENE_IDS.json}`；`.../common_gene_axis/risk_cache/external_Manual.parquet`。`mc.build_pairs` 只用于得到合法 memory_row 与原 n_cells；距离不使用 source_conflict。也可复用 [manual_frames](/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/run_safeconf_public_mechanisms.py:31)到第 60 行；不要调用其 evaluate，因为那会 refit。不要调用 public_priors，因为该函数内部也会拟合 retrieval learner，混入不止一个因素的改变。

这个动作若有明显收益，只说明估计目标不匹配解释了部分损失；还必须相对 support-only 同任务判断。若不改善，直接记录 guide/cell mismatch 不是当前外部失败的主要解释，停止平均方式搜索。

**动作 2，已完成：保留原完整真值的 baseline loss contrast，检查模型特异风险。** 上述 excess MSE 是这项最小实验；它区别于 20-cell/guide-half/shuffle。对现阶段结果应停止将高绝对 U20 直接解释为 biological model failure。若后续要严格区分生物内容与测量精度，最低有价值的数据条件是独立实验重复或独立 batch 的稳定 cell/guide centroid、各任务测量精度足够且不与 history support 紧密共变，以及同任务冻结的至少一个有真实效应差异的强预测器。应在预先留出的研究/任务上同时报告原 error、相对简单基线的 loss contrast、强 support rule，并给目标 validation 资产相同预算。只有再加一个同family predictor、更多稀疏任务、或只把 TEST cells 固定到 20，都不会满足这个条件。此条件是未来资产接纳门槛，不是要求马上启动第三项重训练。

## 创新边界的原始来源

本次直接读取固定官方代码与原论文/官方摘要，没有使用综述替代原工作。

- [PertEMA 固定 README](https://github.com/OfficialBishal/PertEMA/blob/43c09a32e23d0ee2ae5dfbab21b2deeab27f1803/README.md)已讨论 OOF 错误监督、共同失败、mean baseline、噪声上限、路由边界；也明确新 screen 需重新拟合。它当前是软件，README 说尚无配套论文。故“发现共同困难/噪声主导”本身不构成新的科学结论。
- [Risk Advisor 原论文](https://link.springer.com/article/10.1007/s10994-022-06248-y)已有黑盒分类器外训练后置风险元模型。[ConfidNet 官方论文摘要](https://papers.nips.cc/paper_files/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html)已有学习模型置信度/失败预测。它们的分类设定不同，不能直接当生物回归基线，但已限制广义方法优先权。
- [FailureScope 原始摘要](https://arxiv.org/abs/2606.09878v1)提出按跨模型 pass/fail 结构在留出模型上诊断失败；[Fail-Fast, Restart-Smart 原始摘要](https://arxiv.org/abs/2608.03222v1)报告由单策略失败监督训练 monitor 并迁移其他策略。它们是 2026 预印本且任务域不同；这仅否定泛称“首次跨模型失败监督”，不能用作本方法的同任务性能基线。

建议任何后续主张限定到已验证的信息合同、测量对象与部署资产，并先获得明确增量或可复现且不同于既有研究的边界。当前 evidence 并未达到这个判断阈值。
