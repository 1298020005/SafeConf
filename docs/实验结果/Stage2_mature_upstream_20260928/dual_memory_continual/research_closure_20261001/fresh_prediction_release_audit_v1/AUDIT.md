# 新鲜预测发布可用性：六个作者官方发布族的限时元数据审计

## 结论

**没有识别到可直接确认不变Source3285核心的、尚未触及数值的新遗传研究预测包。**这是对本次六个官方发布族的限定负结果，不证明其他发布不存在。最接近需求的研究是Frangieh2021，但下面三个基准均未闭合“现成冻结预测＋完整基因轴＋匹配控制＋合格拆分”链条。未因此开始新推理、训练、适配或真值访问；本方向在此停止。

Source基准轴为 `/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_source_core_20261002_v1/GENE_IDS.json`：3285 genes，SHA `b7f80db0c0e46dd7667cb83f39153c4add31948ca00e5391c210b9c6a18d7c68`。不变参数核心要求**全部3285基因**及同一效应/控制估计量；一般记忆合同的≥2000/≥70%不能替代此要求。本次没有任何候选公开轴列表可据以精确相交，因此各候选missing IDs/count均记录为**未知**，不是0，也不把HVG5000/8192的容量当作全覆盖证明。

## 六个官方发布族及具体阻碍

| 作者发布族 | 官方元数据实际确认 | 对本次要求的判定 |
|---|---|---|
| **Systema / EPFL Brbic lab** | 支持Frangieh control、co-culture、IFN三种真实设置。代码生成本地 `{dataset}_{seed}_gears_post-pred.csv`，列取processed `gene_name`。Zenodo15746995只列46,427,312-byte软件ZIP；GitHub两个release资产仅source ZIP/TAR。 | 未识别已发布prediction CSV/gene-axis/control-vector manifest。≥100合格held-out gene、完整3285轴、norm和matched-control均未知。三个设置存在，不等于三设置上的合格预测已发布。见[官方项目](https://brbiclab.epfl.ch/projects/systema/)、[prediction writer](https://github.com/mlbio-epfl/systema/blob/main/src/run_gears.py)、[Zenodo文件元数据](https://zenodo.org/api/records/15746995)。 |
| **bm2 scPerturBench** | Frangieh页报告212 perturbations/110188 cells；代码为HVG5000、normalize10000/log1p、seed1–3模拟拆分。本地输出路径为 `…/hvg5000/{method}/savedModels{seed}/result.h5ad`。官方Zenodo提供669.4MB输入 `Frangieh.h5ad.gz`；结果目录是metric汇总。 | 未识别公共预测向量/轴/held-out ID manifest。212总perturbations不证明≥100合格gene clusters；三seed不算三生物strata。CP10000 pipeline不等于冻结CP4000。见[官方Frangieh页](https://bm2-lab.github.io/scPerturBench-reproducibility/Task_Genetic_perturbation_Dataset_Frangieh_Accuracy.html)、[预处理代码](https://raw.githubusercontent.com/bm2-lab/scPerturBench/main/Perturbation_generalization/Genetic/myUtil1.py)、[官方数据发布](https://zenodo.org/records/14638780)。 |
| **PerturbArena / TianGzlab scPerturbBench** | 官网/仓库提供24指标的benchmark结果与 `Model_predict_code` 执行入口；README要求配置环境并生成prediction。 | 已识别的是metric表/运行代码，不是带冻结轴/拆分/controls的prediction release；维度、合格gene/strata与norm未知。见[作者pipeline README](https://github.com/TianGzlab/scPerturbBench)、[作者结果网站仓库](https://github.com/LuyiTian/PerturbArena)。 |
| **SBB benchmark** | 官方pull脚本下载processed input h5ad、通用scGPT args/vocab/checkpoint及Docker模型镜像。Frangieh预处理明确筛 `perturbation_2=Control`，固定cell_type=melanoma，CP10000/log1p，HVG8192∪targets，五个condition外折。 | 没有已冻结Frangieh prediction包；该预处理保留一个背景，五fold不等于三生物strata。完整3285 membership未知，norm也不与CP4000相同。见[数据/模型资源pull脚本](https://raw.githubusercontent.com/michavol/sbb-perturbation-benchmark/main/scripts/pull_all_datasets.sh)、[Frangieh处理规则](https://raw.githubusercontent.com/michavol/sbb-perturbation-benchmark/main/data/frangieh21/get_data.py)。 |
| **scLAMBDA** | 作者README提供Norman训练例，以及Replogle/Nadig四cell-line训练例；prediction API要求已训练model。 | 本次观察到的例子属于已排除旧Norman或Source研究；未识别新独立研究的完整预测发布，不能因换model名称变成新研究。见[作者README](https://github.com/gefeiwang/scLAMBDA)。 |
| **CellOT** | 官方发布说明覆盖药物、IFN-beta患者外推和发育轨迹；提供预处理数据与训练/评估脚本。 | 未识别本次≥100 genetic clusters/三strata的独立遗传预测包。论文背景提及genetic perturbation不等于发布的实验满足此条件。见[作者README](https://github.com/bunnech/cellot)。 |

同一作者发布族的README、代码和release-list只是证据跟踪；共六个发布族，没有扩展到第七个候选。没有复用Arc/H1/E198、E170/GWTCD4、E247/TFome/Jiang、MC/Orion或旧Norman作为fresh确认。未读benchmark数字metric CSV来选择model，也未按可能的SafeConf收益筛候选。

## 需要哪些现成材料才能改变本次判定

最接近的Frangieh路线仍缺：带SHA的官方prediction/control文件清单；显式基因轴证明全部3285成员；每个固定upstream checkpoint的query/validation角色与≥100合格gene、≥3真实生物设置身份；精确mean-cell log1p(CP4000)及TRAIN matched-NTC控制/效应定义；合法历史证据及排除当前query/replicate的provenance。不能把raw h5ad、软件ZIP、公共pretraining checkpoint、scalar metric表或本地输出文件名当成这些材料。

以上轴/估计量字段未知时，本审计不做HVG重投影、补零、norm换算、伪造第三stratum或偷偷重训Source。若未来得到合格prediction release，其注册适配/能力验证仍可能构成第三个正式upstream尝试；当前计数保持2，需Root明确批准attempt cap例外及新访问合同后才能执行，不因“不训练上游”自动获准。

## 访问和预算记录

只读源码、README、网页/JSON文件列表以及本地基因ID轴。未下载预测、表达、真值、参数或数值结果数组；46MB软件ZIP、669MBh5ad及4.28MBmetric CSV均未下载。列入 `METADATA_SOURCES.json` 的16个成功HTTP元数据响应合计 **1,773,575 bytes**；SHA是原始文本/JSON/HTML字节，GitHub HTML不是raw Python字节。网页文本解析工具不提供底层传输字节，因此该数字是已精确核验的HTTP部分，不冒称完整网络流量账本。没有任何大型资产请求，远低于10MB显式metadata下载预算。

GitHub无账户API因rate-limit返回403；只继续作者公开文本，不创建账户/授权绕行。确有timeout/403的部分记录为未核验，未推断其隐藏内容。Source参数/合同/模型未改，0fit、0upstream call、0新增formal attempt；只新增本AUDIT和元数据URL/SHA回执。
