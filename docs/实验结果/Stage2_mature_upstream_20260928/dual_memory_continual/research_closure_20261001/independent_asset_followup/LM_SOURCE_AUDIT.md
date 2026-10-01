# Orion固定40文件及官方LM来源审计

日期：2026-10-02（上海）。本轮只审metadata/code/environment；未读新的表达、truth或prediction vectors，未训练、未安装R环境。Orion已经占用本轮唯一新公开遗传study名额，不继续扩候选。

**下载清单已就绪；官方LM尚未就绪。** `ORION_CANDIDATE_FILES.csv/.json`锁定40路径、大小和whole-file LFS SHA256；原官方LM的数学核心支持inductive gene-task，但原数据入口包含test truth，并缺少Orion双背景及raw-count预处理，当前机器还没有可用Rscript。因此不能直接执行原脚本，更不能把自写Python/R Ridge叫作该官方LM。

## 1. 固定文件manifest

- Dataset：`Xaira-Therapeutics/X-Atlas-Orion`；release `53a5bc98d49247bcf967500292575c3d3602de31`。
- 两个真实contexts：HCT116、HEK293T；各numeric Batch1–20。**不因超时、支持不足或效果换分片**。
- 精确expression bytes：HCT116 `8,228,998,855`；HEK293T `9,908,569,180`；总 `18,137,568,035`。另官方gene metadata881,915bytes。
- 40条manifest都含path/bytes/LFS SHA256/git blob oid/pinned download URL。LFS oid是**整个Parquet对象SHA256**；不能用footer或schema哈希替代文件校验。
- 官方来源：[pinned tree API](https://huggingface.co/api/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/53a5bc98d49247bcf967500292575c3d3602de31?recursive=true&limit=1000)。本次API payload104,939bytes；manifest本身保存其SHA256。
- 最终JSON SHA256：`d16cea664ca508659b97dfedbe165ec285609102a0fcb429349335c093a014f4`；CSV/JSON已核验40唯一path、size合计和64hex LFS hashes。
- Parent已选择实际下载；本子审计只生成清单，没有下载这18.138GB。下载后的合法阶段仅整文件hash、Parquet schema及行组/header；不要调用dataset viewer或读取`gene_expression`列。

## 2. 已见preview与pilot的精确范围

实际URL是 **`https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion`**：`web.open`主页面自动返回dataset viewer示例；没有直接调用viewer rows API，不能编造其后台URL。主页面使用main，**当时preview所属revision未知**；不能断言它已被pinned SHA覆盖。工具输出明确见到以下10行的字段与部分/截断raw-count lists；完整表达vector未见。

| cell_barcode | gene_target | sample |
|---|---|---|
|AAACCAAAGACATGTT-HCT116_Batch1|ST14|HCT116_Batch1|
|AAACCAAAGACCCAAC-HCT116_Batch1|SIGLEC5|HCT116_Batch1|
|AAACCAAAGAGGTACG-HCT116_Batch1|VSNL1|HCT116_Batch1|
|AAACCAAAGCGATTAT-HCT116_Batch1|KCNK7|HCT116_Batch1|
|AAACCAAAGGCTTAAT-HCT116_Batch1|APOA4|HCT116_Batch1|
|AAACCAAAGGGCTTGT-HCT116_Batch1|Non-Targeting|HCT116_Batch1|
|AAACCAAAGGTCCTTT-HCT116_Batch1|SBF2|HCT116_Batch1|
|AAACCAGCAAAGCTAG-HCT116_Batch1|SLC39A8|HCT116_Batch1|
|AAACCAGCAAGCTGGG-HCT116_Batch1|CDK5RAP2|HCT116_Batch1|
|AAACCAGCACTGCGTA-HCT116_Batch1|CHKA|HCT116_Batch1|

暴露非control targets为9个：`APOA4,CDK5RAP2,CHKA,KCNK7,SBF2,SIGLEC5,SLC39A8,ST14,VSNL1`。后续如要保留未见的gene-cluster评价单位，保守合同应使这9genes在**两个contexts都不进入test/calibration**；10已见cell barcodes从任何fresh truth估计中剔除，其中已见NT cell也从fresh control估计剔除。只删cell但仍将同gene簇声明untouched不够保守。它们可保留SEEN/development角色。最终应只称**独立外部复核**，不称整个study pristine；排除规则必须在truth前写入最终split，当前manifest只是建议，没有实际修改split。

选择时序纠正：40-file aggregate计数之前已经见过Batch1标签pilot及网页preview；文件合同依据大小和metadata可行性，不应声称它在任何标签访问之前选定。Parent在本轮明确选定了最终40文件，之后不换分片。

元数据pilot范围：HCT116 Batch1–20；HEK293T Batch1–20中除了5/7/9/10，其余16个footer和`gene_target`列已读；首批两背景还读`guide_target/sample`列。标签计数已用于规模判断，属于seen metadata；HTTP Range reader未取表达列。前审计误读公开benchmark metrics表的偏差另在FRESH报告保留，那个表不是Orion的新truth资产。E62旧历史只查到候选识别、下载失败记录、0 complete，没有找到Orion模型/结果。不存在证据支持补称“从未任何设计使用”。

36已读分片证明双背景各≥30cells共有target下界4,663；排除上述9genes后最保守仍≥4,654，不需要用效果挑目标。GEM `sample`不是独立culture/transduction replicate；已知guide pair亦不等于独立生物重复。

## 3. Published官方LM锁定与数学契约

官方论文：[Ahlmann-Eltze et al., Nature Methods 2025](https://doi.org/10.1038/s41592-025-02772-6)。作者repo锁定commit `bfa6eeea2bd145a1af2ec0127a2e808cc38456a9`。

| 官方文件 | bytes | SHA256 |
|---|---:|---|
|benchmark/src/run_linear_pretrained_model.R|7,850|49bda2a46b92c2b0b5bf263b957a3890dbdeff2d4837a94e6b0a397e0e38a6f8|
|benchmark/src/extract_pert_embedding_pca.R|1,989|21bcb3905da12fad7dcc56bf1423a2642135dfa5c6b2aa60ec9174ee2788d4ac|
|benchmark/renv.lock|61,706|9365c447bbc0fcd4747d69ecd9990671843509ffd4b14aa3f72eadcde4d8eb97|
|benchmark/src/prepare_perturbation_data.py|5,775|5fe1b9541ee2a711bc20dfebc63c354a1b98ced9784c992ed3c185a1a43b5c51|
|benchmark/src/run_transfer_perturbation_prediction.R|5,561|afbbd42c21325a41dac6444633f13f7fb28a80600ffb4023c81fa7e51aa6cf08|

[官方LM脚本](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/bfa6eeea2bd145a1af2ec0127a2e808cc38456a9/benchmark/src/run_linear_pretrained_model.R)定义：

- L32–59 `solve_y_axb`；L38–39用**training Y每gene均值**center，拟合双侧ridge矩阵；L150–151调用。默认PCA10、ridge0.1。预测L160为`gene_emb_sub %*% K %*% pert_emb_all + center + baseline`。保留intercept；不能擅自令ctrl预测强制等于baseline。
- L91先取得`train_data`。L95–105与L115–126两次training_data PCA都仅使用genes×training-condition **mean X矩阵**，不是control-only PCA，也不是delta-PCA；`pca$x`是gene-row embedding。
- `pert_emb=t(pca$x)`使embedding columns按gene名标识，因此未在训练conditions出现的目标gene，若存在full readout rows，已有仅由训练扰动得到的embedding。这个机制支持单基因inductive task；它不是“未见gene没有embedding所以填零”。
- L139–149做training conditions/genes匹配；L154–157按全部query clean_condition取embedding。目标gene不在feature rows时match=NA，官方没有合法missing-target fallback。必须在metadata阶段拒绝缺失/重复/非finite映射；不补零或偷偷删失败query。
- 可读external TSV embeddings，但gene_embedding读后转置、pert_embedding不转置；官方不检查external来源泄漏。外部embedding需独立固定source及完整manifest，否则不能靠入口可用推定合法。

**control PCA→newgenes不能称官方默认LM。** 单一control mean向量也不足PCA10。可运行control-cell PCA的模型会是另一项预先命名的适配；本轮没有提出用它替换published默认。保持38,606 raw feature空间供官方train-only gene-row embedding，再只在输出端选择冻结3,285 Source common genes，才不会因先缩输出轴而丢失大量target embeddings。

## 4. 原脚本输入为何不能直接用于当前sealed流程

- 原脚本L63读取`data/gears_pert_data/<dataset>/perturb_processed.h5ad`，需要`X`、obs `condition`、var `gene_name`及split JSON train/val/test条件名。官方prepare脚本仅支持已有GEARS数据入口，没有Orion Parquet适配或cell metadata split。
- L69按**全部**split条件保留cells；L87 pseudobulk全部condition；结尾L167–176将observed与predicted值比较并输出summary。即使模型系数只fit train，直接执行也会打开test表达。允许的test只下载/hash/schema，无法用原脚本直接达成。
- L64–66把ctrl放入train，但L84 baseline仍按所有`condition=='ctrl'`cells平均，没有context过滤。脚本pseudobulk key也没有context。把两个Orion cell lines直接合入该脚本会pool控制并合并同名目标，不能算真实双背景模型。必须显式每context/每fold独立处理，并冻结哪些control cells获准访问。
- [extract_pert_embedding_pca.R](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/bfa6eeea2bd145a1af2ec0127a2e808cc38456a9/benchmark/src/extract_pert_embedding_pca.R)L27–44对输入dataset**全部conditions**取pseudo-bulk/PCA，没有train/test mask。不能对同一个目标Orion整体运行再声称inductive；可仅用明确合法的独立source数据。
- 官方transfer脚本需要reference perturbed profiles，并对reference全部conditions取PCA；这不是仅用control可获得的transfer embedding，不能借其名字绕过test边界。

可行盲adapter应只替换loader、metadata query与输出端统计，保留经SHA锁定的`solve_y_axb`、train-only PCA、ridge和intercept数值核心；对照其官方函数进行小型合成输入数学一致性验证后，才可称**官方LM的Orion盲输入适配**。原脚本逐字节未改不能同时满足盲query流程。当前没有实现adapter，也没有把未经验证的代理登记为官方LM。

## 5. X的估计量与依赖版本

官方LM不执行normalize/log1p。它处理的`X`已经由输入数据定义；不能直接把Orion raw counts丢进去，再将结果解释成原论文processed-X endpoint。

renv pins **glmGamPoi1.16.0**。匹配版本[作者pseudobulk源码](https://github.com/const-ae/glmGamPoi/blob/50d5e6f/R/pseudobulk.R#L37)默认`counts='rowSums2', .default='rowMeans2'`，L176–187按assay名称选聚合。LM的assay名称是 **X**，故为mean，绝非sum。因此如合同使用cell-level logCP10k，应计算 **mean(log1p(10000×cellcount/librarysize))**；`log1p(normalize(sum counts))`是另一估计量。Orionraw→processed-X步骤及control policy仍需在表达访问前冻结，不能把当前audit当已经验证了预处理。

[renv.lock](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/bfa6eeea2bd145a1af2ec0127a2e808cc38456a9/benchmark/renv.lock)锁定R4.4.1、Bioconductor3.19；主要依赖：SingleCellExperiment1.26.0、Matrix1.7-0、MatrixGenerics1.16.0、glmGamPoi1.16.0、irlba2.3.5.1、zellkonverter1.14.1、tidyverse2.0.0、argparser0.7.2、rjson0.2.21、data.table1.15.4、tidylog1.1.0、skimr2.1.5；basilisk1.17.2来自GitHub commit `0a851fa47fb5d64fa5a20dcfc38b76c74cc76f4a`。

本地read-only检查PATH及/usr/bin、/usr/local/bin、/opt和常见conda/env目录，没有发现R/Rscript；不能在无运行时证据时说这些R packages已可用。官方脚本L3还硬编码EMBL BASILISK_EXTERNAL_CONDA路径，在本机无对应环境。`renv`是依赖锁文件，不是一个已经建好的runtime；官方提供的7个conda环境主要为神经模型，没有现成Orion LM容器。没有安装或启动任何环境。

## 6. 低成本判断的实际边界

1e4 target conditions×38,606gene伪bulk为386,060,000entries：double **3,088,480,000bytes（2.876GiB）**，float32 1,544,240,000bytes。R的`as.matrix`通常用double；change、train subsets、PCA工作区和centered Y会产生额外copies。两contexts同时持有多份矩阵不能按最终3285gene output262.8MB估内存。建议串行contexts及32–64GB RAM预算；这是工程估计，不是已测peak RAM。

10维PCA一次gene×condition×k乘法约7.72GFLOPs，但IRLBA要多轮，官方默认还调用两次PCA；ridge矩阵计算也扫描大Y。瓶颈更可能是约百万cells的18GB Parquet解码、cell-level normalization与聚合。必须逐batch/小cell chunk处理，不能dense展开1e6cells×38,606gene（double约309GB）。

官方LM本身可为0GPU；前审计“≤24CPUh”是可设预算，**没有Orion实测throughput保证**。当前阶段可确认数据获取成本和矩阵量级；环境恢复、盲adapter、估计量验证、validation competence均未完成，不能登记TRAINING_READY/CONFIRMATION_READY，也不启动任何第三upstream。

原脚本L165–173的observed/predicted `pivot_longer`还会产生386M行长表；cond/training/gene三个字符串指针列加obs/pred两个double列，仅基础向量约**15.44GB**，未含copies和分组开销。这段统计既读取truth又引入额外峰值，盲adapter应移除它，合法验证另做有限规模流式统计。保留官方数学核心不等于保留这段全量报告I/O；不得以关闭报告为由改变ridge、PCA或intercept。

## 7. Quality只能proxy

已确认`sample`是GEMbatch、guide_target是guide/pair标识；没有可证的独立culture/transduction biological replicate字段。cells、guide count、NT数、technical batch coverage可以作measurement-quality proxies。不得补造highQuality标签或把≥30cells自动命名为高生物重复质量；Quality本轮只应登记`proxy/biological_replicate_unknown`。是否达到competence只能由合法train/validation gate以后决定，不能从Source覆盖或细胞数反推。
