# Arc H1 元数据资格：停止作为新的独立确认资产

**状态：STOP_PREVIOUSLY_OPENED_EXPRESSION。** 本地文件与 E198 已使用的表达文件完全相同，不能再称未查看结果的新研究。

本次只读流式 SHA 核验得到 `f85862ecc9d9c34c1765b395105c74af08480471ade27ce8466d742d73c1449e`，大小 `4,377,281,643` bytes；与 E198 输入登记精确一致。绝对路径为 `/home/yyf/data/scperteval_official_20260730/arch1_processed_complete.h5ad`，不是同名候选的推断。

[E198_STATUS.json](/home/yyf/proj/docs/实验结果/E198_arch1_protocol_calibration_20260801/E198_STATUS.json) 于2026-08-02登记 `COMPLETE_EXTERNAL_PROTOCOL_CALIBRATION`、`dataset_opened_as_anndata=true`，150个扰动、1800条基于表达的评价协议记录，并选择 mse、pearson_pert、rank、energy_distance 和 de_auprc 作为后续终点。[输入哈希表](/home/yyf/proj/docs/实验结果/E198_arch1_protocol_calibration_20260801/tables/E198_INPUT_HASHES.csv) 与[runner的固定路径](/home/yyf/proj/tools/scripts/run_e198_arch1_protocol_calibration.py:46)共同确认文件身份；历史独立审计通过。

E198没有使用模型预测，也没有验证SafeConf；这不意味着表达结果仍未见。真实数据已经用于协议校准和终点选择。现在重新划分gene-hash TRAIN/VALIDATION/TEST不能撤销该历史。官方TRAIN标签本身不是淘汰理由；决定性理由是既有数值结果使用。

发现这一记录后，按Root指令停止共同轴、细胞门槛和历史支持覆盖的进一步探索，没有创建30/20/100划分。150扰动、221273细胞、38176对照为已有记录中的数值，不是本次新计算的合格任务数。Source3285交集、Public2070覆盖和≥30cells的独立基因数均未认证；无需为了使一个已失去新鲜性的候选满足数量条件继续读取。

实际已完成的一次H5AD元数据访问包括root/obs/var schema及151个扰动类别名。没有读取obs codes值、细胞ID值、var基因字符串、X/layers/raw数值。定位覆盖资产时还读取Source基因ID JSON、GENE_MANIFEST表头和Public bank清单/Parquet schema，未计算交集。随后只读取不透明整文件字节用于SHA确认。详细节点、输入证据和访问界限见READINESS.json。

新增下载、模型推理、拟合、bootstrap及TEST解封均为0；正式upstream尝试计数保持2。当前Arc确认提案停止，Source-scale/均值修复方向不自动重启，未授予新Source版本权限。研究目标仍未完成；此结论仅关闭这项资产资格分支。
