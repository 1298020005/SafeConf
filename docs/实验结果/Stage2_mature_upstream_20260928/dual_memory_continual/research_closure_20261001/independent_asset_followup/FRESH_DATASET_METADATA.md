# 下一独立遗传研究：元数据审计（2026-10-02，上海时间）

**具体可保留的下一候选是 Huang et al. 2025 X-Atlas/Orion 的官方双背景 Parquet 子集。** 官方现有分片使旧“559.7 GB 整包超预算”的理由失效。固定 HCT116、HEK293T 各 Batch1–20，只需 **18,137,568,035 bytes（18.138 GB）**；原始基因空间与 Source3352 精确共有 **3,285 genes、98.0012%**。本轮仅用标签已经证明所选子集至少有 **4,663 个两背景各 ≥30 cells 的独立 gene clusters**。这是可供主理人决定的具体候选，**尚不是已登记 SEALED_CONFIRMATION，也不是获准训练的新 upstream attempt**。

本结论按公开性、独立研究、真实背景、规模、基因空间与资源选择，没有按 SafeConf 可能涨幅或新效果挑选。不能宣称已满足上游 competence、Source-error 外部增量或独立 biological replicate 保证。

## 1. 访问边界及审计偏差

- H5AD 使用 `h5py.File(...,'r')`；只读 root/obs/var schema、`var` 原始 gene identifiers、`obs/.../categories`。未读取 H5AD 的 X/layers/raw expression 数值、测试 effect/prediction vectors；未新增本地 test 解封事件、模型调用、GPU或训练。
- Orion 用 HTTP `Range` 精确读取 Parquet footer，以及 `gene_target`/首批的 `guide_target`、`sample` 压缩列。自定义 reader 只允许这些元数据列的 byte intervals，禁止其它列；**未读取 `gene_expression` 或 `gene_token_id` 数据列**。gene_metadata 是纯 ID 映射表。
- 必须保留两项访问偏差：①误把 `/home/yyf/data/scperturbench/all_dataset_genetic.csv` 当元数据，实际是已经公开的 benchmark 结果大表，输出被截断；不以其效果筛选。②浏览 Orion 官方 HF 主页面时，工具自动返回 dataset viewer 的少量原始 count 示例；没有将其分配为 test，也未用于任何设计、分布或效果计算，但因此**不能声称本轮完全没有接触表达示例**。这不等于重新封存已见结果。
- 主审直接可计量 metadata/code HTTP payload 为约 **7.363 MB**；协助代理另读官方 HTML/API，其中若干响应没有记录 payload bytes，估计约1.3–1.6 MB，**全链路严格 ≤10 MB 的 meter 未得到验证**。这项审计限额的记录缺口不能包装成“通过”。没有主动下载表达文件。用户的总新下载≤100 GB、新训练≤96 GPUh授权没有被10 MB审计限额替代。
- 仅写本 Markdown 和同名 CSV；没有改 DATA_ROLE_REGISTRY、论文、主代码、原始数据或训练/解封合同。

## 2. 本地优先候选的原始轴与历史角色

Source ID 唯一来源：`/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201/gene_ids.json`，3,352 unique symbols；SHA256 `3a6c0ef75eef65b0f82286ee142bbb6c02a8b128cff8d3110b38661f74201c08`。本轮只按完整字符串 exact 匹配，不做大小写转换、mouse-human替换、表达筛选、补零或事后 synonym 扩展。70%覆盖实际上要求至少 **2,347** 个共同Source genes，故 common≥2,000 仍不足以单独过门槛。

| 本地文件 | cells ×原始var | Source exact common | Source覆盖 | 实际规模/背景与历史角色 | 判定 |
|---|---:|---:|---:|---|---|
| official_generalization/Schmidt |60,572×5,005|1,224|36.52%|148类别含control；精确拆解为67 underlying genes、67 singleton、80 combo。无 donor/stimulation字段，不能从文件恢复多背景 |低于100 gene clusters且共同轴不达标 |
| official_scperturb/TianKampmann2019_iPSC / day7neuron |275,708 /182,790×33,752|3,351 /3,351|99.97%|真实iPSC与day7 neuron；guide multiplet类别不能当独立target；E168 candidate note明确历史SafeTrans/E129/E153已见 |只能SEEN桥接，不能新的untouched study |
| TianKampmann2021_CRISPRa / CRISPRi |21,193 /32,300×33,538|3,349 /3,349|99.91%|同一iPSC-induced neuron背景；2/4 batches不是2/4生物背景。101/185 perturb categories、203/372 guide categories；CRISPRi E129完成 |研究已见，不能重叫SEALED |
| official_generalization/TianActivation / TianInhibition |20,169×5,045 /31,815×5,152|640 /516|19.09% /15.39%|93/181 categories；缩减轴无raw/var备用全轴；E57等历史设计 |SEEN且共同轴不达标 |
| official_scperturb/WesselsSatija2023 |30,707×21,052|3,317|98.96%|THP-1单背景、CRISPR-Cas13；187 categories含组合；E165于2026-07-16 18:36:27+08不可逆解封 |SEEN，guide/combo不等于独立背景 |
| official_generalization/Wessels |19,775×5,020|1,133|33.80%|128类别、同一研究缩减轴；Phase是扰动后cell-cycle状态 |SEEN且共同轴不达标 |
| JoungZhang2023_atlas / combinatorial |1,145,823×37,528 /167,947×37,302|3,332 /3,332|99.40%|hESC ORF overexpression；atlas3,369 ORF类别须映射splice isoform→gene；只有1真实背景，2/9 batches不可冒充背景；组合文件57类别 |可元数据候选但不满足本次多背景要求；未证明fresh角色 |
| DatlingerBock2021 |39,194×25,904|3,129|93.35%|Jurkat，stimulated/unstimulated；41 guide类别，上界也<100 genes |规模淘汰；384 sample wells不增加独立target |
| SchraivogelSteinmetz2020 chromosome11 /8 |120,310×3,185 /112,260×4,191|33 /28|0.98% /0.84%|K562单背景；TAP-seq targeted readout，3,105/4,115 categories多为enhancer/guide coordinate |共同轴淘汰；不能按var数计Source覆盖 |
| XuCao2023 |98,315×59,429|3,313|98.84%|HEK293单背景；204 target类别、489 guides；E180已有预注册与最终评估目录 |SEEN；cell-cycle phase不能当预设context |
| SunshineHein2023 |90,380×33,538|3,349|99.91%|Calu-3单背景；24,195 multiplet labels；E177/E179历史资产 |SEEN；multiplet labels不能直接计独立genes |

Schmidt来源与“148类别”没有矛盾：官方 [Schmidt 2022研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC9307090/)筛选约70个基因，本地剩67；147个非control条件包含80个组合，不能作为147个独立基因。Joung [官方论文](https://doi.org/10.1016/j.cell.2022.11.026)以human TF splice isoforms构建hESC atlas；原始ORF类别数不能替代gene-level去重。

角色核查不仅依赖当前 registry 中是否恰有该名字：Tian见 `E168.../CANDIDATE_SELECTION_NOTE.md`、`E129.../RUN_STATUS.json`；Wessels见 `E165.../TEST_TRUTH_UNSEAL_EVENT.json`；XuCao见 `E180.../PREREG_ANALYSIS_PLAN.md`及final_evaluation目录；Sunshine见 `E177.../RUN_STATUS.json`和E179。`Stage2.../DATA_ROLE_REGISTRY.csv`还明确 Jiang E208/E216、Feng E258 等训练/验证已SEEN；虽然部分 test仍sealed，也不能把整个研究改叫新的设计独立研究。

## 3. 具体候选：Orion官方固定双背景子集

### 官方研究、数据空间和元数据证据

[Huang et al. 2025 preprint](https://doi.org/10.1101/2025.06.11.659105)及[作者官方发布](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/blob/53a5bc98d49247bcf967500292575c3d3602de31/README.md)：独立HCT116、HEK293T两种cell lines，genome-wide CRISPRi/FiCS Perturb-seq，官方target设计18,903 protein-coding genes。双guide指同一target的guide pair；不当作双基因干预。`sample` 明确为GEM batch。未确认独立culture/transduction biological replicate标识，不能将109/223 batches当生物重复。这个限制应在最终uncertainty/replicate合同中保留。

- 固定release：`53a5bc98d49247bcf967500292575c3d3602de31`。
- [官方gene_metadata.parquet](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/53a5bc98d49247bcf967500292575c3d3602de31/metadata/gene_metadata.parquet)：881,915bytes，SHA256 `33900a0dbaeafb84601c7c8ba7c6642c440774fc3f49613dab11cf1f2bf0fdbf`；schema `ensembl_id:string, gene_name:string, gene_token_id:int64`；38,606 rows、38,584 unique symbols。3,285 Source共同symbols在Orion内均唯一，不需要歧义mapping，覆盖98.0012%。来自GRCh38 2024-A官方reference。
- Parquet schema有`gene_token_id`与`gene_expression`的sparse lists，以及cell_barcode/sample/guide_target/gene_target/pass_guide_filter；本轮只读标签列。不同背景实际共享output空间，future preprocessing应保留raw轴用于published LM train-only embedding，只在endpoint取冻结3,285 common genes。
- 官方全部HCT116：109分片、46,576,484,789bytes；HEK293T：223分片、79,683,336,248bytes；表达全量126,259,821,037bytes，**全量仍超100GB**。旧Figshare H5AD两文件209,354,246,272+350,164,035,901bytes，亦超预算。可使用官方Parquet分片，不能据旧整包大小一概淘汰。

### 已固定子集和实际target规模下界

40-file aggregate标签审计之前预定两背景各Batch1–20；此时Batch1标签pilot已经见过，选择依据官方大小及metadata可行性，不应声称早于任何标签访问。不根据效果或网络成败换选：HCT116 **8,228,998,855bytes** +HEK293T **9,908,569,180bytes**。这是两个生物背景；GEM分片是测量批次，不是背景。

40个预定文件中36个footer/gene_target元数据读成功；HEK293T Batch5/7/9/10超时，没有替换、没有追加分片。以下为36分片**保守下界**，不是完整40文件最终计数：

| context |成功分片| metadata cells |Non-Targeting cells|非control distinct targets|≥20cells targets|≥30cells targets|≥50cells targets|
|---|---:|---:|---:|---:|---:|---:|---:|
| HCT116 |20/20|534,745|26,174|18,200|10,899|6,956|2,353|
| HEK293T |16/20|447,642|21,533|18,135|9,287|5,225|1,375|

两背景各≥30cells的共同target genes下界 **4,663**。新增四个未读metadata文件只能增加这一细胞数筛选下界。4,663是独立target gene簇，9,326以上context×gene pair不能当独立cluster。可以在metadata冻结之后按gene划60/20/20，理论test至少932簇；具体split与guide跨分区检查仍须在训练前落盘，不能把本审计的理论量改写成已完成test manifest。

首批额外guide metadata：HCT116 Batch1 18,549cells、10,075 noncontrol targets、11,091 distinct guide-pair strings、891NT cells；HEK293T Batch1 22,731cells、10,796 targets、11,897 guide-pair strings、1,079NT cells。两个首批均无≥20cells target，故**不能只取一个小分片便宣称足够任务测量重复**。

36个成功footer及行数/大小有聚合证据hash `02e62abbbb05ac9bb259f08405e27136ea4961518fc2bb48349cdeb86a28095f`（JSON rows按line/batch排序，字段line,batch,rows,bytes,footer_sha256）。该hash是审计36文件元数据集合，不冒充完整40-file expression manifest或整文件hash。

### 当前fresh角色、成本及下一合同边界

本地仓库与 `/home/yyf/proj` 中Orion命中只见E62/E63获取说明和download脚本；data目录仅旧download_logs，记录0 complete，未找到Orion训练/预测/解封manifest。它属于 **METADATA_ONLY_IDENTIFIED_PENDING_FREEZE**：以前已经被识别为候选；没有已知结果参与SafeConf设计的证据，**并不等于可证明从未被任何人用于设计**。本轮HF自动表达preview偏差亦须带入后续access ledger，不能直接承诺pristine。若主理人要求绝对零表达接触，Orion应标为受该偏差影响的候选，不能以新名字抹掉它。

可review的低成本upper候选是 [Ahlmann-Eltze et al. Nature Methods 2025](https://doi.org/10.1038/s41592-025-02772-6) 的作者官方[linear pretrained model代码](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/bfa6eeea2bd145a1af2ec0127a2e808cc38456a9/benchmark/src/run_linear_pretrained_model.R)。已核实有`solve_y_axb`、train-only PCA、双侧ridge、pseudobulk及gene-name输出；官方defaults PCA10、ridge0.1。真实published线性家族可与现有神经家族区分；不能把一个自写ridge代理包装成其官方模型。

下一合同应固定：上述release/40文件、完整raw IDs→3,285 endpoint、gene-level split、跨guide不泄漏、source/training-only PCA与controls、validation competence门槛、validation-only校准、模型和预测持久化、test一次性解封。先检查官方脚本在此inductive任务的可查询性与最强简单baseline；审计没有替它通过competence。不能因失败换分片、终点或新开第三大型上游。

预计新表达获取18.138GB，ID metadata0.882MB；CPU流式pseudobulk及官方LM可安排一个≤24CPUh的预算合同、目标0GPUh（这是计划上限，**尚无本数据实测wall-time/RAM证据**）。3,285×4,663 float32伪bulk数组单背景约61.3MB；官方PCA若保留38,606原始features，矩阵需约720MB单背景，额外copies/全部training targets会更大，应流式聚合并限制内存。不能据小最终预测矩阵忽略约百万cells的I/O成本。此候选可以在用户100GB/96GPUh总约束内review；没有开始下载表达或训练，也没有批准第三upstream。

## 4. 更fresh但公开访问尚不成立：Xu/Tan胃癌研究

[最新官方v3论文](https://www.biorxiv.org/content/10.1101/2025.04.16.649236v3.full)实际条件为GES1/SNU719/NUGC3 ×DMSO/SAHA/5-AZA/JQ1/GSK126，并有SNU1750/HGC27/LMSU的SAHA扩展；>200 GC genes、998 guides、4 guides/target，625,866 total cells。独立biology replicate数未核实；这些药物是预设环境，背景结构成立。local docs/tools/raw paths没有找到该研究实验记录，故比既有Tian/Wessels更fresh，但不能以未找到命中证明绝对未用。

v3 Data availability仍写 **SRA XXXXX**；v1提PRJNA1219803，[官方ENA read_run查询](https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA1219803&result=read_run&fields=run_accession,study_accession,sample_title,fastq_bytes,fastq_ftp&format=tsv)只返回表头。未找到公开processed release或可见raw runs，不能给可下载bytes、feature IDs共同覆盖、raw split或可执行成本。当前角色应为 **PUBLIC_ACCESS_UNCONFIRMED_NEEDS_METADATA**，不能用“论文公开”替代“研究数据公开可获取”，也不能报为立即可行确认。

## 5. 主理人可用的决策

保留Orion固定双背景18.138GB子集及published官方LM为**下一可review候选**；其规模与共同轴已有具体metadata数字，不必新开truth才能判断。实际fresh/access偏差、independent biological replicates、完整四个未读metadata文件、最终split和CPU成本仍须诚实写在合同里。如果主确认要求严格无任何表达preview和verified biological replicates，则本轮没有一个无条件合格的新主资产；不能把候选资格改写成确认已经成功。

现有本地Schmidt/Tian/Wessels/Datlinger/Schraivogel/Joung不满足规模、fresh、多背景或common轴组合；胃癌候选等待公开access。不要再把Source errors已失败的外部增量问题解释为“换一组support数据即可解决”。
