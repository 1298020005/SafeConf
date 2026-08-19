# E184 原始文献索引

初次检索截止：2026-07-24；补充检索：2026-07-29。

| 主题 | 原始来源 | 本轮只采用的事实 |
|---|---|---|
| GEARS uncertainty | Roohani, Huang & Leskovec. *Nature Biotechnology* 2024. [DOI](https://doi.org/10.1038/s41587-023-01905-6) | GEARS 增加逐基因 log-variance 头，并通过误差相关损失学习不确定性代理 |
| PRESCRIBE | Cheng et al. NeurIPS 2025. [正式论文](https://papers.nips.cc/paper_files/paper/2025/file/d6383e7643415842b48a5077a1b09c98-Paper-Conference.pdf) | 多变量证据回归；联合 epistemic / aleatoric 置信度；用相关、ECE 和过滤评价 |
| GPerturb | Xing & Yau. *Nature Communications* 2025. [DOI](https://doi.org/10.1038/s41467-025-61165-7) | Gaussian process 扰动效应模型及基因效应的不确定性估计 |
| CPA uncertainty | Lotfollahi et al. *Molecular Systems Biology* 2023. [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/) | 以测试组合到训练扰动/协变量嵌入的距离提供模型相关 uncertainty score |
| 集成歧义分解 | Krogh & Vedelsby. NIPS 1994. [会议论文](https://proceedings.neurips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html) | 平方损失下，成员平均误差等于集成均值误差加预测多样性 |
| 现代集成分解综述 | Wood et al. JMLR 2023. [论文](https://www.jmlr.org/papers/v24/23-0041.html) | ensemble diversity 属于已有系统理论，不能包装成 SafeConf 独创 |
| split conformal regression | Lei et al. JASA 2018. [DOI](https://doi.org/10.1080/01621459.2017.1307116) | 任意固定回归器上的分布无关有限样本边际覆盖及其交换性前提 |
| conformalized quantile regression | Romano, Patterson & Candès. NeurIPS 2019. [会议论文](https://papers.nips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) | 以分位数回归改善异方差场景的区间自适应性，同时保留 conformal 覆盖 |
| 层级数据 conformal | Dunn, Wasserman & Ramdas. [arXiv:1809.07441](https://arxiv.org/abs/1809.07441) | 同一群组内重复观测破坏普通逐观测交换性，需要按层级设计校准 |
| 单细胞简单基线风险 | Ahlmann-Eltze et al. *Nature Biotechnology* 2025. [Systema](https://doi.org/10.1038/s41587-025-02777-8) | 简单基线可达到或超过复杂模型，评价必须隔离 systematic variation |
| 大规模扰动预测基准 | *Nature Methods* 2025. [文章](https://doi.org/10.1038/s41592-025-02980-0) | 27 种方法、29 套数据的泛化差异强调跨 setting 评价 |
| PerturBench | NeurIPS 2025 Datasets & Benchmarks. [正式页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8aee537279a66ced96319dfca3c00002-Abstract-Datasets_and_Benchmarks_Track.html) | 遗传与化学扰动的多场景、多指标评价，含 rank 与塌缩风险 |
| scPertEval | bioRxiv 2026. [预印本](https://www.biorxiv.org/content/10.64898/2026.07.23.740433v1) | 将 representation、transform、metric、reporting 分层并提供多类指标实现 |
| 在线选择后 conformal | Bao et al. *JMLR* 2025. [CAP](https://jmlr.org/beta/papers/v26/24-0452.html) | 选择后覆盖需要专门校准与 FCR 控制，不能从普通 split conformal 自动继承 |
| intervention selective conformal | 2026 预印本. [arXiv:2603.02204](https://arxiv.org/abs/2603.02204) | 在 intervention 场景及 Replogle K562 上研究有效的 selective conformal inference；未同行评议 |

## 仓库内部的直接证据

| 问题 | 路径 |
|---|---|
| PRESCRIBE 双冻结面板 | `docs/实验结果/E96_prescribe_native_comparison_20260713/reports/E96_REPORT.md` |
| 自适应上界基线开发比较 | `docs/实验结果/E179_nested_uq_baseline_benchmark_20260723/reports/E179_REPORT.md` |
| 自适应上界一次性确认失败 | `docs/实验结果/E180_xucao_fresh_guide_certificate_20260723/final_evaluation/reports/E180_FINAL_REPORT.md` |
| 注册家族证书定义 | `docs/实验结果/E181_registered_family_hilbert_certificate_20260724/reports/E181_REPORT.md` |
| 新公开研究前瞻评价 | `docs/实验结果/E182_gse225807_registered_family_20260724/final_evaluation/reports/E182_FINAL_REPORT.md` |
| 四项研究描述性合并 | `docs/实验结果/E183_all_study_family_synthesis_20260724/reports/E183_SYNTHESIS_REPORT.md` |

## 引用纪律

- 文献提供方法定义，不替代本仓库的冻结实验；
- PRESCRIBE、GEARS uncertainty 和 GPerturb 的输出对象不同，不能直接比较跨预测器的相关系数；
- conformal 覆盖必须同时写明交换性、校准单位和覆盖单位；
- 经典数学来源必须引用，SafeConf 的贡献集中在证书对象、组合、协议和实证。
