# WEB_RESEARCH_MAP

**版本**：v0.2（2026-09-28）
**问题**：如何为冻结的单细胞扰动预测器建立通用、可持续更新、可审计的 post-hoc reliability judge。

## 1. 研究边界与证据等级

- **A**：正式同行评审论文并有官方实现或官方数据；
- **B**：正式论文，无官方实现；
- **C**：preprint/会议稿并有代码；
- **D**：preprint/技术报告，无可核验实现；
- **E**：官方博客或产品页面，只作概念启发；
- **F**：二手材料，不作为方法依据。

主方法判断只使用 A–C。D 用于发现路线，E/F 不写成技术依据。

## 2. 九组路线的结论

### A. 单细胞扰动预测与不确定性

1. **GEARS** 是冻结上游预测器的合理黑盒输入来源，但官方仓库明确指出其跨 cell type、组合扰动和数据规模存在适用边界；它还提供 uncertainty tutorial，说明上游不确定性与外挂 judge 是两个可分离接口。SafeConf 应接收其 canonical effect，而不把 GEARS 内部结构写进风险器。
2. **CPA** 原生建模 drug、dose、time 和组合扰动，并有自身 uncertainty 估计；它是化学线的重要上游，但其 uncertainty 属于模型内输出，不能代替外部 post-hoc judge。
3. **scGPT/scFoundation** 证明 foundation-model 上游会改变预测尺度、输出形式和泛化场景；这加强了 universal adapter 的必要性，而没有证明一个 raw magnitude 可以跨模型直接比较。
4. **PRESCRIBE** 是目前最接近“预测时同时给出 uncertainty”的新路线：它把预测和不确定性放在同一个 evidential/Bayesian 上游模型中，并用 pseudo E-distance 评价。它不是 black-box post-hoc 方法，因此 SafeConf 的差异只能放在冻结上游、跨模型、外部历史和冷启动。
5. **PertEMA** 是最直接的相邻方法：冻结扰动预测器，再利用 prediction-time features 和 OOF errors 训练 post-hoc reliability estimator。它提高了 SafeConf 的 novelty 门槛；SafeConf 必须以同合同复现/对照，并证明公共历史、冷启动或跨扰动类型确有额外价值。
6. **GPerturb** 用 Gaussian process 直接建模扰动效应并传播 Bayesian uncertainty，属于内生生成/效应模型，不是外部 judge。它提示测量噪声和生物反应离散度不能混成单一模型错误。

### B. 通用 post-hoc error prediction

监督 error predictor 的通用模式是：冻结主模型，保存预测时可见的 evidence，用 held-out/OOF label 训练一个小的 correctness/error model。成熟经验有两点：

- 预测时可见的 logits/embedding/ensemble disagreement 可以成为 error evidence，但不能把训练内误差当作历史标签；
- error judge 需要和主模型的任务、split、输出误差定义绑定，跨模型转移不能默认成立。

这支持 SafeConf 的 `Prediction Evidence -> small risk learner` 起点，也说明复杂历史结构必须先证明 incremental value。

### C. Selective prediction / abstention

Selective prediction 把输出定义成 `(predictor, selection function)`，核心对象是 coverage 与 selective risk 的曲线；AURC 是常见汇总指标。对 SafeConf 而言：

- `risk_score` 是连续排序量；
- `REVIEW` 是固定预算下的高风险选择；
- `ABSTAIN` 是证据不足或校准不可靠时的拒绝；
- `PASS` 只能在 calibration 合同下使用，不能把低风险分数直接解释为概率正确。

因此主评价应包含 risk-coverage/AURC、Utility@review budget 和 error capture，不能只报告相关系数。

### D. Calibration / conformal

Conformal Risk Control 可以对单调损失的期望风险进行校准，并扩展到 distribution shift、quantile 和 adversarial risk control。它适合放在 SafeConf raw risk 后面做 calibration/interval/abstention 规则，不应被包装成 SafeConf 的核心表示创新。

分布改变时，coverage 结论需要交换性、权重或明确的 shift 假设。基因跨 donor、化学跨 cell line 的 held-out 场景不能自动继承 IID guarantee。

### E. Retrieval / memory

TabR 证明 tabular retrieval 可以把训练对象的近邻上下文和标签引入预测，且检索模块可以在不完全重训时接入新候选；但论文也指出大规模动态检索的训练开销和上下文冻结问题。对 SafeConf，先用 deterministic exact-condition retrieval 和固定 Top-K，后续才考虑 learned similarity。

SafeConf 的历史检索应该记录：

```text
relevance（是否同扰动/剂量/时间/相似背景）
quality（细胞数、重复、测量稳定性、来源数）
coverage（有多少合法历史）
```

历史内容和历史质量必须分开消融。

### F. Set learning

Deep Sets 给出无序集合的 permutation-invariant 结构；Set Transformer 通过 attention 处理集合成员间交互。两者解决的是表示问题，不证明 record-level history encoder 在当前小样本任务上必要。SafeConf 应先与 mean/weighted mean/handcrafted summary 比较；只有非参数汇总有稳定 residual signal，才允许升级 Deep Sets，再考虑 Set Transformer。

### G. Continual learning / memory update

Experience replay、Meta-Experience Replay 和 2024–2026 的 memory replay 工作支持“外部 memory 先更新，参数周期更新”的工程范式，但它们主要研究主任务的持续学习，不直接证明 error memory 的风险增量。对 SafeConf 最小可审计策略是：

1. 新模型接入时 `E=0`，只用 `P` 或 `P+H`；
2. 新的 OOF/held-out 误差记录进入 append-only memory；
3. 先固定 risk learner，只更新 memory；
4. 当 memory 达到预设量级再做周期性 retrain；
5. 用固定 0/5/10/25/50/100% 曲线找饱和点。

不直接采用每条反馈都在线改参数，避免时间顺序、泄漏和灾难性遗忘难以审计。

### H. 多域、多任务和缺失证据

多域适配工作的共同经验是共享特征抽取与 domain/task-specific adapter 比“所有域完全共享参数”更稳健，尤其当域的输入字段、尺度和缺失模式不同。对基因/化学：

- effect vector、context、control、risk objective 共用；
- dose/time、guide/mechanism、compound descriptors 用 typed optional fields + mask；
- calibration 可按 perturbation type 分开；
- evidence dropout 只在 V0 证明缺失退化有问题后加入。

当前数据量支持共享小主干，不支持两个大 encoder 或一个大多域 Transformer。

### I. Judge / verifier / critic

Verifier/critic 文献证明“生成者负责产生候选，judge 负责检查结果”是有价值的分工；过程监督和 outcome supervision 的差别也说明 judge 的标签来源决定可信度。但 LLM judge 的产品性方法和无公开结构的系统不能作为 SafeConf 技术依据。SafeConf 只借鉴角色分工：冻结 predictor 产出 effect，独立 judge 估计 error/risk；不引入 LLM judge。

## 2.5 2026 年新增证据与直接影响

这一节把截止 2026-09-28 能检索到的更新放在决策链中，避免只围绕已有 SafeConf 组件寻找支持。

1. **测量可靠性成为独立威胁变量。** Wang 等人的 2026 bioRxiv 预印本在 29 个数据集、7,170 个扰动上报告了大量扰动测量不可靠，并显示用小规模 pilot 可以预测完整 screen 的可靠性。这一结果支持我们记录 `cell_count / replicate / split-half noise / source`，同时要求把“实验测量质量”与“冻结上游模型的预测错误”分开建模；SafeConf 不能把质量标签直接写成模型错误的证据。
2. **评测协议本身需要审计。** scArchon 提供了可复用的单细胞扰动预测 benchmark pipeline，提醒我们把 dataset、split、metric、环境和上游模型版本写入 provenance ledger。它是评测基础设施，不是 SafeConf 的风险学习方法。
3. **检索历史错误再做校准已有先例。** HopCPT 通过相似历史 error regime 加权 conformal interval，说明“检索 + 历史误差 + 校准”不是孤立的新颖性来源。SafeConf 必须把领域差异放在扰动任务、公共实验历史质量和黑盒跨模型协议上，并做 history shuffle、no-history 和 cold-start 检验。
4. **轻量表格模型应先于注意力模型。** TabM 的参数高效 ensemble 适合作为 V0 的简单强基线；TabPFN 依赖预训练表格先验，当前样本量与生物任务分布不保证匹配，因此不把它列为主线。
5. **PertEMA 的证据级别必须准确标记。** PertEMA 官方仓库是软件和可复现材料，但当前没有可核验的正式同行评审论文；它可以作为同合同近邻实现和基线，不能被引用成已经完成的正式科学结论。
6. **LLM/产品型 judge 不提供实现依据。** 2026 年出现的 decision-only judge 预印本和产品说明只支持“接受/升级复核”的角色分工；内部训练、标签和校准都未公开，SafeConf 不采用 LLM judge。

## 2.6 研究结论的可证伪边界

- 若 E259 式的质量控制后 `Delta_H` 仍接近零，公共历史内容从主方法中删除，只保留质量/支持审计。
- 若合法 OOF `Delta_E` 不稳定，错误记忆只保留为未来反馈协议，不能写成当前贡献。
- 若简单拼接、残差和集合编码没有稳定差异，采用最简单的 post-hoc judge；复杂模块不因结构“看起来合理”而保留。
- 若基因和化学分线方向相反，统一的是输入/误差/校准协议；性能结论按类型收缩。

## 2.6.1 已有失败/负结果如何改变设计

这些结果和正向论文同等重要，直接进入停止门：

- **E201/E234**：当前 risk 与 error 有信号，但没有证明单一 ranker 已经稳定胜过幅度基线；主指标必须是 review-budget 下的风险捕获，而不是相关系数。
- **E258/E259**：控制 source/cell/split-half noise 后，若干数据线的 raw history 增益消失或变弱；`H` 必须做 quality-controlled residual test，不能从表面相关直接升级为生物历史机制。
- **E208**：Jiang24 上游 competence gate 没有形成可用的 test-truth 行；SafeConf 不得把缺少真实标签的线写成 external confirmation。
- **E273/E274**：当前历史和 error-memory 结果仍是 development-only，E274 的 0–100% memory 曲线没有逐行 OOF 证明，不能直接作为最终测试结论。
- **外部 benchmark**：近期跨方法比较显示效果依赖扰动效应大小、指标和 split；必须同时报告 effect scale、cold split 和 quality strata，避免“复杂模型在随机 split 上赢”被误读为通用可靠性。

## 2.7 可核验来源

- [GEARS 官方仓库](https://github.com/snap-stanford/GEARS)
- [CPA 官方仓库](https://github.com/facebookresearch/CPA)
- [PRESCRIBE 论文页面](https://papers.nips.cc/paper_files/paper/2025/hash/d6383e7643415842b48a5077a1b09c98-Abstract-Conference.html)
- [GPerturb 论文](https://www.nature.com/articles/s41467-025-61165-7)
- [PertEMA 官方仓库](https://github.com/OfficialBishal/PertEMA)
- [Reliable single-cell perturbations 预印本](https://www.biorxiv.org/content/10.64898/2026.08.11.744177v1)
- [scArchon 官方仓库](https://github.com/hdsu-bioquant/scArchon)
- [HopCPT 论文](https://papers.neurips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html)
- [TabR](https://arxiv.org/abs/2307.14338)、[TabM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)、[TabPFN](https://www.nature.com/articles/s41586-024-08328-6)
- [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc3924fd16-Abstract-Conference.html)

## 3. 外部研究给出的约束

1. SafeConf 与 PertEMA 的重叠风险高：不能把“post-hoc error prediction”本身写成新颖贡献。
2. SafeConf 与 PRESCRIBE 的边界必须清楚：PRESCRIBE 是模型内 uncertainty；SafeConf 是黑盒、外接、可跨模型的风险层。
3. Retrieval 和 set encoder 不是默认组件；只有在 `P+Q+H` 已证明内容增量后才升级。
4. Calibration/conformal 应放在 raw risk 后面，负责风险尺度和可控拒答，不作为风险证据来源。
5. Continual memory 的第一版必须 append-only、OOF、可追溯；不做无记录 online parameter update。
6. 基因/化学统一的合理形式是共享主干 + type embedding/typed masks + 小型 type-specific calibrator。

## 4. 直接影响 SafeConf 的可复用零件

| 零件 | 最小实现 | 复杂实现 | 当前是否需要 |
|---|---|---|---|
| 基础风险学习 | magnitude + small Ridge/GBDT | small MLP/monotone model | 需要，第一步 |
| 历史检索 | exact perturbation/dose/time + fixed Top-K | learned retrieval/TabR-like | 需要最小版 |
| 历史质量 | source count/cell count/replicate/noise/provenance | learned quality judge | 需要 |
| 历史集合聚合 | mean/weighted mean/summary | Deep Sets/Set Transformer | 先用最小版 |
| 残差修正 | P 基础风险 + H/E residual | gated residual | V0 后决定 |
| Error memory | append-only OOF records | continual parameter update | 需要 memory，暂不在线改参 |
| 缺失证据 | availability mask + fallback | evidence dropout/gating | mask 必须，dropout 待验证 |
| calibration | isotonic/CRC on held-out calibration | adaptive/shift-aware CRC | 后置需要 |
| abstention | fixed review budget + support threshold | learned defer policy | 需要可审计版 |
