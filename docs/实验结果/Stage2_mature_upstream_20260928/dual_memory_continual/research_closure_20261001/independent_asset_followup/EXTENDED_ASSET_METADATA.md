# 扩展遗传资产元数据审计（2026-10-01）

本审计继续检查 E98 / E106 / E107 / E157，未停在 E195。结论：**这四项现存资产没有一项同时满足真实 published 模型、足够独立扰动簇、足够 common gene 输出轴和可登记为新增主验证的条件。** 最有价值的现存补充资产是 **Frangieh E106+E107 缓存**：189 个扰动、3 个背景、真实 TransformerGenerator/GEARS_Model；它只能提供已见历史、512 基因的规模/背景敏感性。不得把 E98 的 3,000 基因轴与 E106/E107 的真实架构合并成一个从未训练或冻结过的资产。

## 范围和访问边界

- 仅阅读 run status、合同、split/task manifest、源码、checkpoint 字节哈希及路径/大小；NPZ 只检查 ZIP entry name 和 NPY header。E98 NPZ 的 `genes` 与 H5AD `var/gene_name` 仅用于 gene-name 元数据交集。
- 未载入任何测试表达 X、测试 effect 数值或预测 vector 数值；未调用模型、GPU、训练、网络下载；未新增或重启 upstream attempt。Checkpoint 只串流 SHA256，没有反序列化。
- 资格由架构、持久化、输出轴、已登记扰动簇、数据角色决定；不使用任何 SafeConf 效果或潜在涨幅筛选。
- 资源原授权 100 GB / 96 GPU hours 没有被审计限制取代。本轮使用 0 GPU hours、0 下载。

## 资产数字

| 资产 | 实际输出轴 / 原生目标 | 登记规模与独立簇 | 背景 / validation | checkpoint 和 prediction | 资格缺口 |
|---|---|---|---|---|---|
| E98 Frangieh | 3,000 gene effect | 3,348 test 记录（3 fold × 4 train fraction × 279）；去重 567 context×perturb pair、189 perturb genes | 3 背景；每 fold 30 val pair，29/28/28 gene；跨 fold 90 pair、75 gene | 原始 effect cache 和 7,416 条双代理预测存在；whole-human embedding initializer 存在；无训练后神经 checkpoint | SourceEffect-scGPTKNN 与 ContextRidge 为代理预测器，非 scGPT/GEARS end-to-end 架构 |
| E106 scGPT | **512** gene effect；Transformer d_model 512 是另一概念 | 每 fold 279 test，共 837；去重 567 pair、189 gene | 同 E97：3 背景；258 train / 30 val / 279 test 每 fold | 3 个预测 NPZ 各有 567 entry，train258/val30/test279，header 全为 `(512,)`；129 pretrained tensors 已登记；best_state 只保存内存，无训练后 checkpoint 文件或序列化代码 | 减少的 512 gene 输出轴；不能把 initializer 当 fold-trained checkpoint |
| E107 GEARS | **512** gene effect；hidden64 | 同 E106，同一任务合同，不能把两模型相加为更多独立簇 | 同 E97；heldout context 不参与 coexpression 图 | 3 个预测 NPZ header 同上；GO/coexpression CSV 存在；best_state 只在内存，无序列化选定 checkpoint | 真实 GEARS_Model 类，训练改为 context-mean task 和 MSE；512 gene，无可重放训练后 checkpoint |
| E157 PRESCRIBE P3/P4 | 原始 Norman 33,694 gene；**模型 feature axis 2,044 gene；native PCA10/reconstruction2,044** | P3/P4 各 24，test 不重叠，共 **48 独立单基因簇** | 1 背景；共享 64 train condition（含 ctrl）、20 val condition；20 val 涉及 23 underlying genes | 2 best Lightning + 2 slim state checkpoints 均存在且 SHA 重验匹配；2 label-only CSV 各24 task；只保存10维 PCA预测 | 双24task panel，共享20val；PCA10原生真值与 raw log1p effect 不同；后续 E158 已不可逆解封 |

独立性解释：E97 189 基因在三个背景复用，三 fold 也复用同一矩阵。837 是模型/fold task 记录，567 是生物学 context×perturb pair，189 是 perturbation cluster。只有 3 个背景，不宜把背景泛化的不确定性写成拥有大量独立背景。E98 的四档训练量没有增加新的生物学样本。

E97 每 fold 测试设置固定为：context+perturbation unseen 30；context unseen row 159；perturbation unseen column 60；random missing pair 30。每 fold validation 为源背景随机 pair，不能直接宣称它提供了 heldout-context/double-unseen calibration guarantee。E106/E107 的 val predictions 存在，可以用于原注册校准复核；这些历史已用过的 validation/test 不能改名为新确认。

## 架构真实性和持久化

E98 RUN_STATUS 和 report 明确登记两种代理：SourceEffect_scGPTKNN / scGPTEmbedding_ContextRidge。scGPT embedding 的预训练 checkpoint 为 `/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human/best_model.pt`，205,385,258 bytes；存在 initializer 不能证明本数据上的神经扰动预测器已训练。

E106 通过 E65 `load_model` 实例化真实 `scgpt.model.TransformerGenerator`：12 层、8 heads、d_model512，迁移 encoder/value_encoder/transformer_encoder，129 matched tensors。当前调用使用普通 TransformerEncoder 路径。输入适配为同背景 control mean+perturbation flag，进行 MSE finetuning。E107 直接调用 `gears.model.GEARS_Model`，hidden64、GO与gene GNN各1层、decoder16、uncertainty=False；GO为外部先验，共表达图只用源背景对照。两者属于真实 published 架构的任务适配，当前 manifest 未提供足以证明整个本地模型包逐字节官方原版的 upstream commit。本轮不把这一缺口隐藏为“全官方原协议”。

E106/E107 的 `best_state` 在选择后 `load_state_dict`，之后写 prediction NPZ/status/history；两个 runner 没有 `torch.save` 或 model-save。各 fold 目录没有 `.pt/.ckpt`。已登记路径可恢复 prediction-cache replay，不能恢复新的模型 query。若补存或重训需要新的上游工作，这份审计没有为其添加 attempt 资格。

E157 的 PRESCRIBE 上游 commit 为 `6f7264a205aaff654a9594863c5c10b656f88ebe`（`https://github.com/Bunnybeibei/PRESCRIBE`）。对 SOURCE_MANIFEST 里的 model/distribution/nn/flow/loss **24 个核心文件**重新 SHA256 并与该 commit blob 比较：全部一致。数据 loader 有明确 Norman patch；模型架构和 native loss 没有暗改。native ListMLE 的排列不变性已由原合同登记，不能宣称 train E-distance 有效施加排序监督。

重新核对的 E157 四个文件：

| panel | 资产 | bytes | SHA256 |
|---|---|---:|---|
| P3 | best-00.ckpt |145,752,572|fc75c02201cea11c5da439628157e3a20a2b92173e69803946e6c270b575a31c|
| P3 | E157_LOCKED_NATIVE_STATE.pt |131,773,264|44d40c2e29a4d2e6169205f5f064cdb8f37095418faa6c964e9c89e1e4dd83d1|
| P4 | best-00.ckpt |145,752,572|f74425ef4462045998960c6abdfae118e7ca4a5fb6e4d83c30015165a6787b58|
| P4 | E157_LOCKED_NATIVE_STATE.pt |131,773,264|dd165be659758ceeb7bd12b4d4b942c123a92da076cbc0df1e24b50e715adcfd|

best 路径在 `/home/yyf/proj/docs/实验结果/E157_prescribe_norman_p3p4_native_20260714/norman_p{3,4}_formal_seed3407/checkpoints/main/`；slim 路径在 `/home/yyf/data/safeconf_e157_locked_models/norman_p{3,4}/`。E157 的历史 STATUS 表明当时 label-only freeze 未读 test truth，但这不是当前 fresh 资格：`E158/attempt_001/UNSEAL_EVENT.json` 已在 **2026-07-14T20:33:09+08:00** 登记 irreversible test unseal，唯一当前 RUN_STATUS 是 `failed_after_irreversible_test_unseal_preserve_attempt`。审计没有读 raw Norman 或其 test truth。

## 与现有 Source 的 gene-name 元数据交集

Source 的 gene IDs 来自 `/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201/gene_ids.json`，3,352 gene。只按精确 gene name 比较，没有缺失基因补零、同义名事后扩展或测试效应选择。

| 资产轴 | native gene | common gene | Source 覆盖 | 资产轴覆盖 |
|---|---:|---:|---:|---:|
| E98 Frangieh |3,000|818|24.40%|27.27%|
| E106/E107 Frangieh |512|219|6.53%|42.77%|
| E157 PRESCRIBE |2,044|568|16.95%|27.79%|

以上均未满足当前 common≥2,000 / Source coverage≥70% 门槛。E157 的 2,044 不能按 Norman raw33,694 或旧 native GEARS5,025 计数；E98 的3000不能移植给E106/E107。

## 下一资产建议与明确缺口

1. **现存、可立即复核的最有价值补充：E106+E107 Frangieh 缓存。** 189 个独立扰动簇超过24task panel，3背景、同一fold合同、合法源背景30val；适合作为历史真实模型的 reduced-axis sensitivity。没有新下载/训练需求。统计应按扰动基因聚类并保留fold，不能按837记录独立重采样。当前审计只给元数据结论，不启动结果向量读取或新校准。
2. **新的主验证资产仍然缺失。** 已存Frangieh代理3000维和真实模型512维分别满足部分条件，但没有真实published架构、至少100独立簇、持久化模型/输出和足够common轴的同一个新主资产。实现native3000+真实模型需新的固定模型合同与attempt，该项不在本轮添加。
3. **E157适合既有官方native架构/持久化证据，不能扩成新的大规模确认。** 两panel48test基因/shared20val、PCA10目标、Source共同基因568和历史已解封为实际缺口，checkpoint存在性不是缺口。

CSV 为并排可机读登记；所有候选 `main_confirmatory_qualified=False`、`new_upstream_attempt_authorized_by_audit=False`。结果选择不参考 SafeConf 得分。

## 可追溯证据

- `/home/yyf/proj/docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/{RUN_STATUS.json,manifests/E97_TASK_MANIFEST.csv}`：source110188×3000、189扰动、3contexts、split/setting；只读manifest重新计数。
- `/home/yyf/proj/docs/实验结果/E98_frangieh_gene_cartesian_predictions_20260713/{RUN_STATUS.json,reports/E98_REPORT.md}`；`/home/yyf/data/safeconf_e98_frangieh_cartesian/E98_EFFECT_ASSETS.npz` header及gene names；预测NPZ entry/header。
- `/home/yyf/proj/docs/实验结果/E106_frangieh_context_scgpt_20260713/folds/*/{RUN_STATUS.json,predicted_effects.npz}` 与 E107 同结构；`tools/scripts/run_e106_frangieh_context_scgpt.py`、`run_e107_frangieh_context_gears.py`、`run_e65_scgpt_formal_fixed_panel.py`；H5AD 只读结构/var gene names。
- E108 status 已登记837test、1674PredictionRecord、strict issues0、validation校准；属于旧评估的事实，没有据其分数挑选本次资产。
- `/home/yyf/proj/docs/实验结果/E155_prescribe_norman_p3p4_contract_20260714/{RUN_STATUS.json,manifests/*_SPLIT.csv}`；E156 RUN_STATUS 的2044gene及train-only角色；E157 ANALYSIS_CONTRACT/SOURCE_MANIFEST/INPUT_MANIFEST/STATUS；四checkpoint哈希；E158 UNSEAL_EVENT与RUN_STATUS（未读test vector）。
