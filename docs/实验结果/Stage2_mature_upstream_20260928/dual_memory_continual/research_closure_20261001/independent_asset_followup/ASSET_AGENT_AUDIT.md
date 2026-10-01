# 独立资产资格审计：2026-10-01

本次审计根据既有 checkpoint、原生维度、合法 validation、数据独立性和运行成本盘点资产。未按 SafeConf 得分筛选候选；最终资产选择由 root 决定。新增表达数据/模型权重下载 0 bytes，新模型训练 0 GPUh，未经注册的新 test expression/truth 访问 0 行。

在当前“一天内、复用缓存、单次下载不超过 10 MB、不增加第三个新上游训练”的约束内，E195 是可立即执行的独立 Norman 原生 GEARS-UQ 次级源；本轮已完成其 validation competence 实算。已审计候选中没有同时具备新独立数据、已验证现成强模型、至少 100 个独立目标基因和新盲测资格的主确认替代品。

## 数字化盘点

| 资产 | 原生输出 genes | 现成模型/预测 | 已登记任务和簇 | 一天内合法复用 | 资格缺口 |
|---|---:|---|---|---|---|
| E195 Norman GEARS-UQ | 5,025 | 6 checkpoint；模型、预测、真值和锁的 60 条 registry 文件均存在 | 48 个独立单基因任务；每 panel 20 validation conditions，union 24 | 已实算六模型 validation；可复用旧预测 | 真值已在旧实验打开；condition holdout 含双扰动历史；48 小于 100；不能当新的盲测 |
| 旧 GEARS 7 面板 | Adamson 5,043 / Norman 5,025 / Frangieh 3,000 | 20 个 seed 预测缓存；旧 raw 目录未保存 model.pt；E78a seed 33 status 缺失 | 167 dataset-condition，166 个去重基因；逐 dataset 71 / 48 / 48 | 可复用已有预测审计 | 七面板合并不构成单 dataset ≥100 簇；旧真值已见；无可用 checkpoint 做新 query；E78a 非完整三 seed family |
| E201 TxPert STRING-GAT | 3,352 | 16 last checkpoint 和既有封存预测；16 个 checkpoint SHA 均重新验证 | 2,008 总任务、1,808 主任务；目标 566 / 416 / 405 / 421 | source validation 日志已盘点，可直接复用模型/预测 | 与当前 TxPert/Replogle/Nadig 源重合，新增独立源为 0；既有目标结果已打开 |
| E190 GEARS–scGPT | 512 | 六成员既有结果 release；本轮没有重复原始架构审计 | 692 查询只对应 47 个目标基因；54 source validation tasks | 可复用已有 release | 原生 512、47 基因，不能以 guide/batch 记录数充当独立扰动数 |
| ARC VCC2025 / scPertEval arch1 | 未下载检查 | 公开训练表达文件，未核实可直接推理的正式 checkpoint/预测矩阵 | 150 training perturbations；221,273 cells，其中 control 38,176 | 当前约束下不可直接实跑现成强模型 | training copy 不含官方 public/private test；H5AD 4,377,281,643 bytes；主 GCS 需订阅项目 |
| Altos PRiMeFlow | 18,001 | 官方代码、训练数据、split 和生成指令 CSV 已公开；未验证权重或已生成预测矩阵 | VCC private-test instruction 100 条件，当前无已登记结果 | 不符合现成资产的一天约束 | HF dataset tree 无 checkpoint/预测矩阵；Altos HF models API 返回空；GitHub 无 releases；官方 pretrain 12 GPU，finetune 4 GPU × 200 epochs |
| scPertEval-models | 8,192 HVG union perturbed genes | 六模型训练/推理 scaffold；未验证已训练发布工件 | 当前无已登记模型结果 | 不能作为现成输出直接实跑 | 每 fold 要执行 train_predict；公共 bucket listing 仅 processed expression 和 DE 工件；Replogle 来源又与已有源重合 |

完整字段和路径见 `ASSET_AGENT_AUDIT.candidates.csv`、`ASSET_AGENT_AUDIT.json`。以上候选缺口来自资产与验证资格，不能用其风险排序结果调整入选规则。

## E195 合法 validation 实算

输入来自既有 `E195_SEED_STATUS.json` 的 actual validation 条件。P1/P2 各 20 条，union 24；P1 包含 8 single / 12 double，P2 包含 5 single / 15 double。所有六 checkpoint 都纳入，12 个 model/config SHA 对登记表一致。没有重新训练、调整超参数或删除不利 seed。

| panel | seed | mean task-centroid RMSE | control RMSE | 胜 control 任务率 | mean effect Pearson |
|---|---:|---:|---:|---:|---:|
| P1 | 11 | 0.06687 | 0.06053 | 0.30 | 0.431 |
| P1 | 22 | 0.06988 | 0.06053 | 0.40 | 0.425 |
| P1 | 33 | 0.05606 | 0.06053 | 0.70 | 0.412 |
| P2 | 11 | 0.06241 | 0.06262 | 0.55 | 0.482 |
| P2 | 22 | 0.06362 | 0.06262 | 0.60 | 0.459 |
| P2 | 33 | 0.05640 | 0.06262 | 0.70 | 0.464 |

| panel | 3-member centroid RMSE | control RMSE | ratio-of-mean RMSE improvement | 任务胜率 | mean effect Pearson |
|---|---:|---:|---:|---:|---:|
| P1 | 0.05937 | 0.06053 | 1.92% | 0.60 | 0.469 |
| P2 | 0.05717 | 0.06262 | 8.70% | 0.65 | 0.504 |

这里是所有 5,025 genes 的 validation task-centroid 指标，不是细胞级训练日志 MSE。任务级 relative RMSE gain 均值和 ratio-of-means 是不同统计量；主表仅用后者。模型产生可测 Δ 信号，但 family 相对零效应对照的平均改善温和，成员间不稳定；该复算不支持把它事先称作稳定强模型。此结论不使用 test 选模，也不使用 SafeConf 结果。

推理输入固定为前 32 个已见 train control cells；输出减去同一批 control 均值，真值为对应既有 validation 全细胞均值减去全体已见 train controls 均值。只对允许行读取 CSR 值：2,000 control rows、9,794 distinct validation rows，0 test rows；没有物化完整 X，也没有打开 test NPZ。第二次运行为保存 family validation vectors 和复算 family competence；总执行仍为秒级，训练用量为 0。

具体文件：

- checkpoint：`/home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/{P1,P2}/raw_gears/seed_{11,22,33}/norman/seed_{11,22,33}/model/model.pt`
- config：同目录 `config.pkl`
- validation 登记：每个 `raw_gears/seed_*/E195_SEED_STATUS.json` 中 `child_status.actual_condition_sets.val`
- validation/control 表达：`/home/yyf/data/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad` 的对应许可行
- 输出：`ASSET_AGENT_AUDIT.validation_tasks.csv`、`.validation_summary.csv`、`.validation_family_tasks.csv`、`.validation_family_summary.csv`、`.validation_vectors.npz`、`.validation_checkpoint_hashes.csv`、`.validation_access.json`

## TxPert 既有 validation 的资格

16 个 last checkpoint 均重新哈希匹配 `E201_FAMILY_SEAL.json`。每个模型只使用 source-context validation；这里只读取既有日志，不重开目标表达。最后一次 logged Pearson Δ 如下，所有 seed 均报告。

| 留出 target | source train rows | source validation rows | 4 seeds last validation Pearson Δ 范围 |
|---|---:|---:|---:|
| K562 | 294,951 | 80,340 | 0.4469–0.4555 |
| RPE1 | 273,003 | 76,950 | 0.4025–0.4101 |
| HepG2 | 314,391 | 89,117 | 0.4759–0.4866 |
| Jurkat | 282,132 | 78,682 | 0.4683–0.4730 |

该记录是已有预测器能力和来源隔离的证据，不能将这些日志当作与 E195 同终点、同任务的预测精度比较。公开原始 K562 GAT 位于 `/home/yyf/data/txpert_official_20260802/cache/checkpoints/K562_unseen_cell_gat.ckpt`；E201 已重训练 last 位于 `/home/yyf/data/txpert_official_20260802/e201/formal/{target}/seed_{1..4}/checkpoints/last.ckpt`；合法 validation 日志位于同 run 的 `logs/txpert/version_0/metrics.csv`。输入 split 是 `/home/yyf/data/txpert_official_20260802/cache/E201_blind_{target}/splits/train_test_split.pkl`，blind training view 是同目录 `de_adata_test.h5ad`，已隔离目标 treatment；文件名含 test 不改变其注册语义。详情和 hashes 见 `.txpert_validation_inventory.csv`。

## 外部官方资产核实

[scPertEval 数据文档](https://scperteval.readthedocs.io/en/latest/user-guide/datasets.html) 明确 arch1 是 150 扰动 training split，并给出 221,273 cells / 38,176 control 的统计。HTTPS HEAD 返回 H5AD 长度 4,377,281,643 bytes；仅请求 metadata，未下载表达。公共 bucket listing 当前 17 项，只有 processed H5AD、README 和 DE outputs，未包含预测矩阵。[Arc 官方 atlas 仓库](https://github.com/ArcInstitute/arc-virtual-cell-atlas) 当前要求订阅的 billing project；本次没有创建账户、开户或付费操作。

[Altos 官方 PRiMeFlow 仓库](https://github.com/altoslabs/primeflow) 和 [HF 数据目录](https://huggingface.co/datasets/altoslabs/primeflow-vcc-datasets/tree/main) 已公开，故不能再写成“胜者代码尚未公开”。数据目录里的 `prediction_dataframe_vcc_test_multifile_h5.csv` 是生成条件说明，不能当作模型预测。该 HF tree 没有 weights 或已生成 predictions；[GitHub releases](https://github.com/altoslabs/primeflow/releases) 无发布。只能表述“本次检查的官方发布渠道没有已验证现成 weights/prediction asset”，不能绝对声称作者没有在任何地方公开。

[scPertEval 官方模型文档](https://scperteval.readthedocs.io/en/latest/benchmark/models.html) 链接的 [scPertEval-models](https://github.com/Virtual-Cell-Research-Community/scPertEval-models) 是 train/predict scaffold。现成 scGPT human foundation checkpoint 是预训练编码器，不能替代已训练扰动预测器；生成指令和教程生成的伪预测均不符合正式模型资产资格。

本轮没有继续修复 E208/E247/E258/Jiang，没有消耗第三个新上游训练名额，没有生成论文正文，没有提交 git commit。
