# E143 决策与实验室交接

## 当前能否开做

计算端已经准备完毕，但正式湿实验尚不能开做。缺少的不是代码，而是实验室外部条件：新细胞背景、dCas9-KRAB 稳定性、慢病毒/CRISPRi 许可、平台档期、预算和负责人。在这些信息确认前擅自给正式候选命名，会把“前瞻验证”变成事后挑选。

## 推荐规模

以真实相关 ρ=0.40、双侧 α=0.05 的 Fisher 近似计算，80% 功效至少需要 47 个独立基因。因此主方案采用 48 个基因，而不是把数万个细胞误当成独立样本。每基因 2 条独立 sgRNA，另加 6 条 non-targeting guide；2 个背景×3 个独立培养/转导批次，共 6 个 10x 文库。每 guide 每批目标 100 个 QC 后细胞、最低 50 个。

## 两阶段执行

第一阶段用 Nadig 的 24 基因面板调通感染、guide 捕获、Day 7 收样、qPCR 和活率流程。该面板来自真值解封前的风险分数，但 Nadig 真值已经被模型开发读取，所以只能做技术预实验。

第二阶段至少加入一个全新背景。首选实验室已有、转导稳定的 A549、Huh7 或 Hep3B，而不是为了论文临时选择最容易成功的细胞。先做 STR、支原体和三批 baseline；再填候选模板、冻结分数与清单，最后开始扰动。若实验室已有两个可靠的新背景，两个都用新背景比“新背景+Jurkat”更强。

## 必须由实验室确认的九项输入

1. 可用细胞系、是否已有 dCas9-KRAB 稳定株；
2. 慢病毒与 CRISPRi 的生物安全审批；
3. sgRNA 载体、包装体系、筛选标记和既往滴度；
4. qPCR、流式、10x 3′+guide capture、bulk RNA-seq 的可用平台；
5. 预算上限和最晚完成时间；
6. 三批独立 baseline 的提供方式；
7. 测序平台最低细胞量、上机规格和批次安排；
8. 湿实验负责人及独立盲法映射保管人；
9. STR、支原体、培养、转导、排除和偏差记录模板。

## 机制深入的触发条件

主实验完成后，仅在“背景交互稳定、两条 guide 一致、校正 viability 后仍存在”的目标中按冻结规则选 2 个。再做 Day 3/Day 7、独立 guide、sgRNA-resistant cDNA rescue、qPCR/Western/flow 和必要的 ATAC。现阶段高风险候选集中在核糖体/核仁、线粒体翻译/ISR、基础转录与 RNA 加工；具体通路节点必须等待主实验后再冻结，不能现在先写好一个机制故事再找支持证据。

## 文献依据

- Nadig 数据的官方 GEO 说明了双 sgRNA、低感染率、Day 3 FACS 和 Day 7 10x 3′流程：[GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667)。
- 原研究与跨细胞背景差异：[Nature Genetics 2025](https://www.nature.com/articles/s41588-025-02169-3)。
- 最新模型审计强调 batch-matched control、Pearson Δ、retrieval、强均值基线及 split-half 实验重复性：[TxPert, Nature Biotechnology 2026](https://www.nature.com/articles/s41587-026-03113-4)。
- 多细胞背景泛化仍是普遍难题：[Nature Methods benchmark](https://www.nature.com/articles/s41592-025-02980-0)。
