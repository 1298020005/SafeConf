# Orion metadata与盲输入准备

本目录只放轻量摘要；完整cell metadata、gene manifest、split和身份哈希保存在runtime：

`/home/yyf/data/safeconf_orion_frozen40_20261002/metadata_preparation_20261002_v1/`

实施脚本为 `tools/scripts/prepare_safeconf_orion_metadata_agent.py`。本轮没有启动上游训练、没有读取真实expression列或新prediction/truth vectors，未修改运行中的下载注册或session2643。Orion整体不称pristine。

## 冻结与访问身份

最终候选manifest `d16cea664ca508659b97dfedbe165ec285609102a0fcb429349335c093a014f4`已作为新exclusive-create、只读快照保存。旧注册manifest `997bdbf33c079746457646cd5fe433e3812063ebc1aefcb6f590376086f15900`用已知文字字段精确重构，SHA完全匹配；旧快照同样保留，active `DOWNLOAD_REGISTRATION.json`仍为旧hash。两版本40路径、size、LFS SHA和URL逐项完全一致，操作tuple SHA为 `f6134ea76ed658bd1f07e6e565291cf05d9f3896ff501e31cc5d29ba9597fd59`。

`NON_OPERATIONAL_AMENDMENT.json`明确这是选择时序说明的非操作修订，不把新bytes冒充旧bytes。只读文件模式是0444并有exclusive-create/内容一致检查；这提供可审计的不可覆盖约束，**不是OS root级防篡改保证**。

每个原Parquet只读取footer及非expression metadata列的精确压缩区间；代码检查metadata envelope不与`gene_token_id/gene_expression`列区间重叠。保存完整compressed envelope及各metadata列hash、footer hash、logical Arrow IPC hash、派生metadata Parquet hash，以及原cell_barcode、context、row index和row group。部分原文件身份来自已验证下载；其它来自pinned remote ranges，其whole-object LFS验证仍由正在运行的downloader完成，不能把metadata hash冒充整文件hash。

40/40 metadata共1,104,270 rows；metadata envelope42,800,596bytes。完整明细仅在runtime `row_metadata/*.parquet`和`file_identity/*.json`；repo摘要没有表达或逐cell明细。

## 固定分区和Source轴

UTF-8精确prefix为 `SafeConf-Orion-20261002-v1|`，无前导空格；对原始gene label求SHA256，以256-bit整数阈值分配60% TRAIN、20% VALIDATION、20% TEST。没有按性能选择。每gene在两个contexts和所有models始终同角色。

九个preview genes `APOA4,CDK5RAP2,CHKA,KCNK7,SBF2,SIGLEC5,SLC39A8,ST14,VSNL1` 强制TRAIN。十个已见cell_barcode全数实际匹配，单独为`DEV_EXPOSED_CELL`，不进入训练/校准/test truth估计，其中已见NTC也不进入fresh control估计。

| metadata角色 |genes或rows|
|---|---:|
|TRAIN genes|10,926|
|VALIDATION genes|3,614|
|TEST genes|3,745|
|TRAIN rows|629,838|
|VALIDATION rows|205,652|
|TEST rows（仅labels已读）|215,133|
|TRAIN_CONTROL_SOURCE_SCOPE rows|53,507|
|DEV_EXPOSED_CELL rows|10|
|missing/ambiguous target ID excluded rows|130|

两个contexts各保留≥30cells的共同gene clusters共5,948：TRAIN3,576、VALIDATION1,129、TEST1,243。这个计数是metadata规模资格，不是已生成预测、已通过competence或Quality为高。

`GENE_MANIFEST.csv`有3,285共同genes，严格按Source3352 ID顺序，保存Source index、Orion gene token及Ensembl ID。Source symbols在Orion均唯一；不做synonym、大小写或缺失补零。`GENE_AXIS_IDENTITY.json`记录67个Source缺失IDs和来源hash。默认published LM的gene-row embeddings仍需保留full raw gene space；3285是endpoint axis，不能提前裁掉full feature rows后给轴外target补embedding。

## 一次性metadata QC和controls范围

已冻结：`pass_guide_filter==1 AND num_features==2 AND finite positive total_counts`；没有表达阈值，没有test效果驱动的修改。pct_counts_mt等只记录为proxy，不新增拟合阈值。非NTC biologic unit为`originalstudy|gene|cellline`；独立重采样cluster为gene，同步两个cell lines。

各context模型只使用自己context的合法NTC作为TRAIN_CONTROL_SOURCE_SCOPE：HCT11626,173、HEK293T27,334；不pool跨context controls。当前是两个contexts内的gene holdout合同，不自动授权新的heldout-context control访问。完整范围在`CONTROLS_SCOPE.json`；现在只有metadata，control expression也尚未获准读取。

未证实独立culture/transduction biological replicate。`sample`是GEMbatch，guide pair也是技术/干预标识；Quality始终为`proxy/biological_replicate_unknown`，不补造highQuality标签。

Orion prospective estimator固定为 **mean(log1p(4000×cellUMI/full_library_total))**，对应parent核实的TxPert Source `normalize_total(target_sum=4000)→log1p`量纲。不是logsum，不拟合test median，denominator不在3285 projected axis上重算。官方metadata `total_counts`作为full-library total；一致性只可先在合法TRAIN rows核查。方法参数、competence与validation/calibration子合同尚未冻结，不消耗新attempt。

## 后续表达读取接口与严格private边界

`iter_authorized_expression_batches(...)`当前没有在真实数据上运行。它要求一个独立expression permit，绑定实际frozen method contract文件SHA和本次policy SHA；必须指定单一authorized context及允许的TRAIN/VALIDATION/TRAIN_CONTROL角色。TEST、DEV请求直接拒绝。

接口先读取已冻结、校验hash的row metadata，根据gene role、固定QC与seen-cell排除构建允许gene/cell集合；在创建Arrow Scanner时使用`gene_target.isin(allowed_genes) & cell_barcode.isin(allowed_cells)` filter，并投影表达列。只把通过filter的rows返回，不先dense读取全部test表达再mask。readahead/pre-buffer关闭；每batch再检查其cell集合。这是 **RETURNED_ROWS_BLIND**，不等于private expression从未被native decoder解析。

PyArrow24会从选中mixed rowgroups/pages读取和解码投影列，再过滤RecordBatches：[官方Parquet扫描源码](https://github.com/apache/arrow/blob/apache-arrow-24.0.0/cpp/src/arrow/dataset/file_parquet.cc#L638)、[Scanner filter源码](https://github.com/apache/arrow/blob/apache-arrow-24.0.0/cpp/src/arrow/dataset/scanner.cc#L458)。压缩字节经过读取、native表达值被解析、表达rows返回给训练caller，必须分开登记；不能用后者为零推出前两者为零。

若合同要求 **PRIVATE_ROWS_NEVER_PARSED**，接口在创建expression Scanner前检查metadata，只要原文件含任何非允许角色就拒绝。当前40原文件全部混合，严格模式不能启动。准确缺口是没有物理纯允许files/rowgroups/pages：需要供应方提供隔离资产，或另行明确授权独立custodian处理混合源；后者custodian本身会解析private值，必须独立记录，不能宣称custodian也未解析。普通Scanner、减小batch或read-then-mask均不能补足严格保证；本轮没有擅自执行custodian隔离或test解封。

合成Parquet验证（仅人工创建数据）已证明：普通接口仅返回TRAIN；同gene的DEV cell与TEST cell均排除；严格mixed模式在任何expression Scanner构造之前拒绝；TEST角色请求拒绝。另验证九preview genes全TRAIN、Source3285顺序、metadata字节reader拒绝非metadata区间、旧注册与两个snapshot hash。没有用真实表达验证接口。

完整runtime身份集合SHA：`89901ca8afbb6fe86664c54c5f6c2bab27b21d77f75563395e25c3dd0d9659f6`；policy SHA：`594a5aef77684dc99666022c76666100a5c9791082532d00b7c70ddfe1fac3b7`；gene split SHA：`3bda5a9e0674cd7379b01e798d1c1b8c337efec86f44aceb41d4d74c73df52be`。
