# SafeConf下一阶段实验工作单

执行负责人：当前Codex主会话。更新：2026-10-08。

目标：完成可支撑TCBB／CCF-B及计算机方向二区投稿的核心证据。研究问题仍为：公共真实实验和旧预测器错误，分别怎样帮助一个新预测器审核结果。

## 一、执行顺序与已启动工作

| 工作 | 当前状态 | 解决的问题 |
|---|---|---|
| A0：重复端点的配对统计 | 已完成 | Public超过资料数量规则的差值是否稳定 |
| A1：严格支持匹配的内容对照 | 下一项 | 正确历史内容在相近资料量下是否仍有增量 |
| B：Adamson跨研究公共参照 | 缓存已核到，待效应合同验收 | 相同干预类型、换研究与预测器后是否有效 |
| C：共享风险显式保留公共总分 | 来源训练资产已核到，待实施 | 来源学习器是否因输入表达而损失强公共判断 |

三项实验各自留完整结果。普通技术选择、排障和一次有依据的修复由负责人完成。某项结束后接续下一项。

## 二、A线：公共内容与测量影响

### A0：已执行

实际输入：

- `data_model_feedback_20261003_v1/measurement/simple_risk_sensitivity_v1/FROZEN_SIMPLE_RISKS.csv.gz`：542任务、377基因，3个背景。
- `data_model_feedback_20261003_v1/measurement/DEV_REPEAT_TASK_ERRORS.csv.gz`：原主误差、guide/plate/cell两半误差及既有WMSE诊断。
- 同目录资格依赖表：542行，禁止实验交集均为0。

上述相对路径位于当前工作树的`docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/`下。

固定公共分数、支持量规则及幅度，按每种端点的共同有效任务比较；按背景等权汇总U20。进行5,000次配对基因簇bootstrap，每次同步保留同基因的所有背景。8种端点分别报告，不据诊断端点更换主误差。

已取得：主端点Public减支持量为0.0446，95%区间[−0.0092,0.1241]；两个guide端点分别为0.0283[−0.0261,0.1141]和0.0381[−0.0753,0.1099]。8项Public减支持量的区间均跨零，8项Public减幅度的区间均为正。旧点估计逐项复现。零拟合，约53.5秒CPU。

可复现命令：

```bash
/home/miniconda/bin/python tools/scripts/run_safeconf_nextphase_measurement_v1.py \
  --output /home/yyf/runtime_artifacts/safeconf_nextphase_20261008_v1/measurement_paired_rerun
```

### A1：固定为一次更严格的内容实验

使用同一542开发任务、原三折基因划分、2,840维预测和公共效应。历史来自：

```text
/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/public_mcfaline_trainval/public_memory.parquet
/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy
```

每外折删除全部该折查询实验；每条查询继续执行原资格和同背景优先规则。先复现A0的真实公共分数，再固定以下对照：幅度、全部合法历史支持量、实际入选历史支持量、真实公共内容、五个严格匹配置乱内容。

置乱兼容条件固定为研究、背景、处理条件、效应合同与输出轴相同，细胞数按`floor(log(n_cells)/log(1.1))`分箱。交换兼容的扰动块；五个种子为20260930至20260934，接收记录的元数据和权重保持。每组不能有效置换时保留原记录并登记；使用历史中实际移动比例低于60%的折，不承担有效内容负对照结论，不事后放宽箱宽追求差值。

主比较是真实内容对两个支持量规则及置乱内容，分别报告原误差和已有重复端点的配对区间。当前查询的细胞数、重复真值只用于评分和诊断，不进入可部署特征。全部支持匹配规则在读本轮差值前冻结。

如果内容增量主要在原始端点存在，结合guide/plate结果定位测量依赖；如果重复端点也保留增量，固定其有效任务范围，再接B线。该实验本身不新增评分网络。

## 三、B线：修正跨研究资产选择

原先提出Norman＋GWPS时，漏核了干预方向：Norman主要是CRISPRa(激活)，GWPS是CRISPRi(抑制)。它们不进入同类历史主比较，也不通过直接把效应取负来修补。Norman资产保留，作为干预资格检查的对照；本轮主路线改为Adamson与GWPS。

原论文依据：[Norman](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/)、[GWPS](https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/)、[Adamson](https://pmc.ncbi.nlm.nih.gov/articles/PMC5315571/)。

已核到的实际预测目录：

```text
/home/yyf/proj/docs/实验结果/E65_scgpt_formal_fixed_panel_20260711/
/home/yyf/proj/docs/实验结果/E76a_adamson_scgpt_panel2_20260711/
```

每个目录有24任务、GEARS与scGPT两种预测、48条预测记录；每组为原登记512基因轴。使用各自的`arrays/predicted_effects.npz`、`tables/PREDICTION_RECORDS.csv`和`tables/E65_GENE_PANEL.csv`或`E76a_GENE_PANEL.csv`。两个面板及各预测器单列，独立扰动并集由实际身份表计算；模型和种子不增加生物样本量。

公共来源为本机`/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ReplogleWeissman2022_K562_gwps.h5ad`。

执行动作：

1. 核准相同CRISPRi方向、对照、表达处理、输出基因与原始预测的训练暴露；不能仅凭基因重合率进入计算。先读取原处理代码与训练侧元数据，登记效应合同，再生成公共效应。
2. 原生误差端点保持。共同基因轴用于公共距离及相同轴幅度对照，缺失基因不补零。跨面板不拼原始风险分数。
3. 固定比较：幅度、支持量、公共距离、五个合法内容置乱、已有同任务模型分歧。每个预测器使用自己的错误，另列既有原生不确定性分数中确实存在的部分。
4. 每面板至少20个有效任务才计算U20；覆盖不足保留全量状态并报告固定幅度回退，融合尺度来自允许训练区。分别统计每模型、每面板及两个面板等权汇总。
5. 使用旧验证记录检验上游能力。预测输出完整不等于能力通过；能力门沿用原合同，不因公共分数高而放宽。

这组资料已在旧研究中评价，属于跨研究复现。数据合格即完成全部固定比较；新方法正式确认另用完整配置冻结后才评分的资产。效应不相容时先修输入合同，无法闭合的记录不进主比较；A、C线继续。

## 四、C线：共享风险的一次明确表达修复

当前来源HGB输入为六项预测特征加七项公共特征，没有显式输入强公共规则的最终分数。本轮只比较一个固定改动：增加`simple_history_risk = sqrt(D²+V)`这一列，不改网络、损失或标签定义。

合法输入已核到：

```text
/home/yyf/runtime_artifacts/safeconf_research_20261003/sams_v1/DecoderOnly_RISK_FEATURES.parquet
/home/yyf/runtime_artifacts/safeconf_research_20261003/sams_v1/SAMS_VAE_RISK_FEATURES.parquet
```

每个来源只读取原反馈池331任务、228基因，用自身错误作Source监督。原212任务、152基因保持固定评价。分别执行DecoderOnly→SAMS和反向；不混合两模型原始错误。

固定学习器为200轮、学习率0.05、深度3、最小叶20、L2=10的HGB；Ridge等旧结果按合同复用。来源训练池按基因哈希三折，比较预测特征、原预测＋公共特征、再增加公共总分三组。每折单独拟合Source错误CDF、预处理和簇权重。公共特征先复核禁止实验关系；不一致时重建受影响部分。

候选开发与选择只使用该方向来源模型的错误；目标模型错误不参与配置选择。五个整簇标签置乱同样完成训练、比较和选择，保留不启用来源学习的原公共规则。零目标反馈不拟合Target通道。

真实开发18次拟合、完整五个置乱流程90次，必要最终拟合不超过6次，合计上限120次。只有来源侧开发通过原采用门，才冻结候选并对原212目标任务生成一次新分数。未通过则保留已验证公共规则，记录这个表达修复的结果，不继续搜索新特征。

## 五、采用、预算和实际交付

开发采用门沿用：U20点增量至少0.005、95%区间下界不低于−0.005、至少60%有效分层不降、AURC恶化不超过5%、高风险漏检率恶化不超过0.02、有效分层至少80%。同时面对最强简单对照；实际门和统计证据分别解释。

本阶段新增GPU训练0、大型上游训练0、下载0；CPU最多4线程、目标内存16GiB，总CPU额度24核时，计入既有资源账本。A0已计约0.015核时。A1先跑一个完整外折测时，B先测公共效应提取，随后更新真实耗时；不把等待时间算作实验工作量。

运行产物使用`/home/yyf/runtime_artifacts/safeconf_nextphase_20261008_v1/`；摘要与合同放本目录。保留输入哈希、逐任务分数、信息账本、配对统计、负结果、采用决定和下一动作。仅提交本轮所属文件。

本阶段验收是：内容归因、同干预类型的跨研究比较、一次有依据的共享风险修复各有实际结果，并给出固定完整配置。论文正文与PDF按用户要求继续暂停。
