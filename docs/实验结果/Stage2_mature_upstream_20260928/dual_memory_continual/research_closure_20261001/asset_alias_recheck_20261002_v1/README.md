# 外部候选名称与资产身份复核

本次只核对作者数据库元数据和已有实验合同，未下载表达矩阵、读取新真值、启动模型或重试失败资产。

- **Gladstone CD4 / GWCD4i / genome-scale-tcell-perturb-seq**对应已使用的E168/E170等原研究，不能换名称当新确认。证据为既有SOURCE_LOCK，以及[作者仓库](https://github.com/emdann/GWT_perturbseq_analysis_2025)；其当前README的GEO链接为GSE314342，第三方搜索结果GSE278572未作为下载依据。
- **TF-ome / Southard / RPE1-CRISPRa-TFs / Zenodo15213619**对应已经停止的E247研究。作者元数据实际提供29.749GB full population、11.687GB singlets normalized log1p、1.898GB mean population及0.275GB guide矩阵。最小mean population由过载droplet回归模型重建，不能为了省下载而直接作为真实独立实验响应。
- 原文提到两个细胞类型，不代表这个RPE1数据记录提供了两背景；本次没有验证另一个背景的独立数据文件。论文为[Southard等Nature Genetics2025](https://www.nature.com/articles/s41588-025-02284-1)。

**选择：上述两者均不作为新增确认资产，不重开E247，不重复下载。**本次搜索没有证明全球不存在新候选，只完成这两个实际身份映射。当前优先补齐原授权的Source DEV PublicBiology重训与防遗忘闭环，而非从别名重新消费已有数据或增加第三个上游。
