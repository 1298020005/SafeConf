# Source normalization audit — DEV/SEEN

现有 E201 Source true/pred effect 使用官方 **log1p CP4k** 坐标；不能称为 logCP10k，也不能与未来采用 logCP10k 的 Orion 效应直接计算、解释 RawRMSE。基因重叠98%只解决轴对齐问题。此轮仅查代码、封存记录、HDF5 schema 和官方说明；新增 Source/Orion/test X、released vectors 的表达值读取及采样次数均为0，未构造 scaler。

**可证。** 固定 TxPert 提交 `08d82eea86746b044cf7531f4ec8c5f60e1cb73f` 当前干净。[Methods4.2.2，PDF第13页](https://www.valencelabs.com/wp-content/uploads/2025/05/TxPert.pdf) 明确采用 `log(1+4000*c/sum(c))`，效应为减去同cell-line/batch的对照均值。[固定README第42–76行](https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/README.md) 说明四线各自同程序归一化后合并；[固定onboarding notebook，cell11](https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/data/onboard_dataset.ipynb) 执行 `normalize_total(target_sum=4000)` 再 `log1p`，之后才选HVG。

`main.py:28` 从官方Zenodo15420279取cache；`gspp/data/datamodule.py:185` 直接读cached X，`:611` 将X变为训练tensor；`gspp/predictor.py:129` 的target是batch.x，`:143` 的sample_inference直接返回model输出；`gspp/models/txpert.py:222` 的线性decoder直接预测该表达坐标。SafeConf `run_e201_txpert_sealed_prediction.py:448` / `:493` 直接写模型输出，`release_e201_target_truth.py:329` 直接写同cache.X；`run_safeconf_research_closure.py:60` 将冻结均值减去冻结对照得到source效应。核对的路径无CP10k转换、逆标准化或表达再归一化。此前测量审计已复算所有1808个Source主任务并与冻结真值精确吻合，本轮未重扫它们。

**保留的原合同限制。** README声明先合并，再取test-cell-line HVGs与训练线存在基因的交集；E201沿用K562_cross_cell_lines3352gene官方轴。它包含held-context表达参与HVG选择的事实，不能重标为TRAIN-defined轴。新的Orion3285gene重叠仅为父任务报告的候选元数据，这里未独立复验其列表。

**元数据与可确认的数据。** 原cache.X为632488×3352CSR；obs只有barcode索引、batch、cell_line、condition、condition_name、control、dose、gene_name；var只有索引；无raw、counts layer、library-size列或scale-factor列。uns.log1p为空dict，另外保存DE信息；它不能单独证明4000，更不能证明10000。原文件SHA256由E200/E201 seal记录为 `1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8`，此轮只核对路径/schema/字节数，未重算大文件hash。

[Replogle官方release](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387) 明确区分raw_singlecell与gemgroup Z-normalized单细胞/伪bulk文件；`normalized`文件名并不等于logCP10k。[Nadig官方GEO样本](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8225184) 列出jurkat_raw_singlecell_01.h5ad，并说明Cell Ranger4.0.0的gene-count处理。固定TxPert onboarding直接示范从hepg2_raw_singlecell读取后做CP4k；不能用原研究的其他Z-normalized或TRADE衍生效应冒充该cache。

本地现有四个候选原始对应物：`/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/` 中的 `ReplogleWeissman2022_K562_essential.h5ad`（310385×8563）、`ReplogleWeissman2022_rpe1.h5ad`（247914×8749）、`NadigOConner2024_hepg2.h5ad`（145473×9624）、`NadigOConner2024_jurkat.h5ad`（262956×8882）。其schema保留UMI_count、ncounts、guide_id、cell_barcode、batch与基因ID；Replogle还保留core_scale_factor。它们可以支持后续获准的TRAIN/control-only cell映射、整数counts与pre-HVG分母重构核对。本轮未读取其X；尚未证明这些本地别名与原始官方count文件逐值等价，也未重建精确过滤/分母链。不能从3352或3285保留基因的总和恢复完整pre-HVG library分母。

**Source–Orion直接RawRMSE的必要条件。**

- 先冻结共享的输出坐标、log base、per-cell library分母/基因域、效应均值与对照匹配口径；按现有Source证据，共享输出若采用CP4k最直接。Orion truth和model output都须指向此坐标；选择Input.raw或logCP10k输入本身不证明输出相容。
- 在完整获准raw基因域上逐细胞归一化，再投影到TRAIN-defined common axis；保持signed effect，核对gene ID/order，不补0。所有学习式处理只用TRAIN，新增held-out truth仍在预测冻结后释放。
- 如果保留Orion输出为CP10k，则需要另行冻结可验证的Source变换合同；不能把旧centroid或delta直接乘2.5。若分母一致，细胞表达的数学变换是 `f(x)=log1p(2.5*expm1(x))`，而 `mean(f(x)) != f(mean(x))`；delta也不能直接套f。冻结模型的线性decoder还可能输出非count-domain的负表达，任意clipping会改变方法。

物理坐标相容仍不消除跨研究的cell/context/batch、时间、过滤和对照参照差异。CDF标签、control-RMS scaling与MC默认median normalization均不能代替Source尺度证据。此轮结论只关闭“Source已是logCP10k”的错误前提；未启动任何修复式新scaler或转换实验。

代码SHA256、完整schema、证据URL、可证/未证字段及gate结果见同目录SOURCE_NORMALIZATION_AUDIT.json。
