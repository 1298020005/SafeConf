# E155｜PRESCRIBE Norman P3/P4 前瞻任务合同

冻结日期：2026-07-14。当前阶段只完成任务、切分和评价规则冻结；没有开始预处理、训练或测试。

## 1. 数据访问边界

冻结脚本只读取 Norman AnnData 的 `obs`、`var_names` 和文件形状，未访问 `X`。它读取 scGPT 扰动词表来排除官方模型不能编码的扰动，并读取 P1/P2 的旧切分任务名。以下内容在冻结阶段禁止打开：E95 的逐细胞预测、任务误差和 checkpoint；E96/E145 的逐任务结果表；P3/P4 的任何表达真值、预测或误差。

PRESCRIBE 论文：<https://papers.nips.cc/paper_files/paper/2025/file/d6383e7643415842b48a5077a1b09c98-Paper-Conference.pdf>。上游 Git commit 固定为 `6f7264a205aaff654a9594863c5c10b656f88ebe`。源码/数据哈希表为 `manifests/E155_SOURCE_HASHES.csv`，其 CSV 内容 SHA256 为 `63444e7b9470a97b4edaac7c74e45d5ded546d6a05845ca5330fb89c94da2915`。工作树含为 Norman frozen split 添加的本地兼容补丁，合同将它与上游源码分别记录，不能把补丁称为官方实现。

## 2. 任务资格与固定选择

测试候选必须同时满足：Norman 单基因 `gene+ctrl`；`obs` 中至少 200 个细胞；扰动基因存在于表达 `var_names` 和 PRESCRIBE 使用的 scGPT embedding；没有在 P1/P2 作为测试任务出现。合格候选共 49 个。

候选按 `SHA256("E155|Norman|P3P4|strict-gene-holdout|20260714|v1" + TAB + condition)` 升序排列。前 24 个固定为 P3，后 24 个固定为 P4，余下 1 个只作预先登记的 reserve，不因后续结果替换任务。P3/P4 任务重叠为 0，基因重叠为 0。

## 3. 严格基因留出切分

P3/P4 的 48 个测试基因组成共同 held-out gene set。任何包含其中一个基因的单扰动或组合扰动都不能进入任一面板的 train/val。两面板共享同一开发集：train 64 个条件（含 control），val 20 个条件；每个面板 test 24 个任务，另一面板的 24 个测试任务在本面板中保持 excluded。验证集按 `SHA256("E155|Norman|P3P4|shared-validation|20260714|v1" + TAB + condition)` 固定，control 强制留在 train。

该规则比 P1/P2 更严格：不仅测试任务本身不能进训练，含同一测试基因的组合扰动也被排除。这样 P3/P4 测试回答的是未见扰动基因的泛化，不是已见基因的新组合。

## 4. 固定模型与随机性

- PRESCRIBE native architecture/loss；上游 commit `6f7264a205aaff654a9594863c5c10b656f88ebe`，本地 loader patch 必须按哈希原样保留并单列披露。
- P3/P4 主分析不原样复用 E93：上游 Step1 在切分前用全数据选 HVG 和 DE metadata，会让测试表达参与特征预处理。E156 必须只用 train（含 control）拟合 HVG、PCA 和训练期 E-distance 排序标签；val/test 只做固定变换，测试表达只能在模型和分数全部锁定后用于评价。上游原样的 transductive preprocessing 若补跑，只能列为敏感性分析。
- 模型随机种子：3407；预处理随机种子同为 3407。
- formal：50 epochs，warmup 5 epochs，batch size 4096，deterministic training，与 E95 P1/P2 一致。
- P3/P4 不因 smoke 或 formal 中间结果修改任务、切分、种子、epoch、分数方向和评价终点。
- P1/P2 与 P3/P4 的预处理和基因留出强度不同，四个面板不能伪装成同一设计直接合并；P3/P4 先按本合同独立给出结果，再与 P1/P2 并列讨论。

## 5. 固定终点

对每个任务先求预测与真实平均表达，并减去同一个 control 平均表达，形成预测和真实 log-normalized 扰动效应向量。

1. **主要准确度终点**：`pearson_effect_accuracy`，即预测效应与真实效应向量的 Pearson 相关，越高越好；这是 PRESCRIBE 论文默认的 perturbation accuracy 口径。
2. **方向确认终点**：`frac_correct_direction_all`，逐基因预测效应与真实效应同号的比例，越高越好；`frac_correct_direction_top20_de` 为补充。
3. **误差补充终点**：`rmse_effect_error`，两个任务平均效应向量的 RMSE，越低越好。
4. **置信度**：任务内取 `epistemic_confidence` 和 `aleatoric_confidence` 均值；官方组合固定为 `combined_confidence = 2 * epistemic + aleatoric`。不拟合新权重。
5. **部署可见基线**：`predicted_magnitude_rms`。真实 magnitude 只可作诊断，不能参与风险排序。

主统计量是在 P3、P4 内分别计算 `combined_confidence` 与 Pearson accuracy 的 Spearman 相关，再取两面板等权宏平均；预期方向为正。外层 Pearson 关联作为敏感性分析。方向准确度预期正相关，RMSE 预期负相关。每个面板按任务做 10,000 次 bootstrap；宏平均在两个面板内分别重采样后等权合并。相同 bootstrap 索引用于 combined confidence 与 magnitude 的配对 Δρ。固定 coverage 为 90%、95% 和 50%–100%（每 5% 一档）。

## 6. 通过与边界

- **论文口径信号确认**：P3、P4 的 combined→Pearson Spearman 都大于 0，且双面板宏平均任务-bootstrap 95% CI 下界大于 0。
- **增量优势确认**：在上一条通过的基础上，combined 相对 predicted magnitude 的宏平均配对 Δρ 95% CI 下界大于 0。
- **方向生物学确认**：P3、P4 的 combined→direction accuracy 都大于 0，且宏平均 CI 下界大于 0。
- 第一条通过而增量优势不通过时，只能称为有可靠性排序信号，不能称为超越 magnitude 或具有独立增益。
- 第一条不通过时，P3/P4 复现失败；不得用 RMSE、某个单面板、reserve 替换或事后更改 coverage 来挽救主要结论。
- 任一测试任务缺失、测试基因进入 train/val、输出非有限或 checkpoint/任务不匹配时，先判合同失败；只能修复实现并保留审计记录，不能改任务。

## 7. 后续运行命令与资源预算

本次实际执行：

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e155_prescribe_norman_p3p4_contract.py
```

下阶段需先新增并审查 p3/p4 专用 adapter；当前 E93/E95 CLI 只接受 p1/p2，下面是冻结的拟运行接口，不表示脚本已经存在：

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e156_prescribe_norman_p3p4_preprocess.py --panel p3
/home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e156_prescribe_norman_p3p4_preprocess.py --panel p4
CUDA_VISIBLE_DEVICES=0 /home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e157_prescribe_norman_p3p4.py --panel p3 --mode formal --seed 3407
CUDA_VISIBLE_DEVICES=1 /home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e157_prescribe_norman_p3p4.py --panel p4 --mode formal --seed 3407
```

环境固定为 `/home/yyf/.conda/envs/prescribe_env`（Python 3.9.25）。服务器有两块 Quadro RTX 6000 24GB；预处理主要使用 CPU，训练每面板使用一块 GPU。E95 P1/P2 的 50-epoch 训练实测约 0.55 和 1.72 小时；E155 开发集更小，训练预算按每面板 0.5–2 小时、双 GPU 并行 0.5–2 小时估计。上游 10,000-run E-test 预处理耗时未被 E93 单独计时，完整预处理和训练保守预留 6–12 小时。该时间是资源预算，不是完成记录。
