# E201 正式核心评价冻结

冻结日期：2026-08-02

## 目的

本评价只处理四目标、四种子 TxPert STRING-GAT 的核心任务级结果。它回答三个
分开的命题：

1. 四种子分歧是否构成 family RMS error 的确定性下界；
2. 预测前固定的 SafeConf 风险能否把高 family error 任务排到前面；
3. SafeConf 在 predicted magnitude 之外是否仍有增量。

代数证书通过不自动等于经验路由通过，经验路由通过也不自动等于相对 magnitude
有增量。三个结论分别裁决。

## 运行前硬门

`run_e201_formal_core_evaluation.py` 只能在下列文件已提交、工作区无该文件改动、
本地 HEAD 与 Gitee/GitHub 远程分支实际哈希一致时运行：

- 本冻结文档和评价程序；
- `E201_PRETRUTH_RISK_STATUS.json`；
- `tables/E201_PRETRUTH_RISK_FEATURES.csv`；
- `E201_OFFICIAL_GENERAL_BASELINE_STATUS.json`；
- `E201_TARGET_TRUTH_RELEASE_STATUS.json`。

程序重新验证风险表、四份预测前模型 centroid 向量、official-general-baseline
centroid、四份 target truth、四份 truth manifest 和 observation 顺序的字节数与
SHA-256。真值释放状态必须证明 general baseline 在 target truth 之前封存。正式
输出目录或真值 centroid 目录已存在时拒绝覆盖。

## 固定任务与向量顺序

- 全部任务：2,008；
- 主分析：1,808 个 `primary_ge30` 任务；
- 低细胞数敏感性：200 个 `sensitivity_10_29` 任务；
- 任务键：`target cell line × perturbation condition`；
- target 顺序：K562、RPE1、HepG2、Jurkat；
- seed 顺序：1、2、3、4；
- 基因数：3,352；
- 风险表的 `source_mean_delta_row` 必须严格等于 0–2,007，所有 centroid 向量按
  这一顺序读取。

experimental batch 只用于预测阶段的 matched control。正式任务 centroid 按细胞
聚合，不把 batch 小组当独立任务。

## 固定误差

设四个 seed 的任务预测为 `p_s`，family centroid 为 `p_bar`，目标真值 centroid
为 `y`：

- `seed_s_rmse = RMS_g(p_s-y)`；
- `family_centroid_rmse = RMS_g(p_bar-y)`；
- `family_rms_error = sqrt(mean_s(seed_s_rmse²))`；
- `worst_seed_error = max_s(seed_s_rmse)`；
- `family_disagreement = RMS_s,g(p_s-p_bar)`；
- `control_error = RMS_g(control-y)`；
- `official_general_baseline_error = RMS_g(general_baseline-y)`；
- `source_transfer_error = RMS_g(source_prediction-y)`。

核心证书为：

`family_rms_error² = family_centroid_rmse² + family_disagreement²`

程序用同一组 float32 封存向量转为 float64 后重算各项。恒等式绝对残差容差固定
为 `1e-10`；`family_rms_error < family_disagreement-1e-12` 记为下界违反。预测前
表中 disagreement、magnitude 和 model-source gap 的重算容差固定为 `5e-6`。

## 固定风险与比较对象

主要风险量：`safeconf_e201_risk`。

主要简单对照：`predicted_magnitude`。

模型 family 内部对照：`family_disagreement`。

预测器强基线：TxPert official-`MeanBaseline` 等价 centroid、batch-matched control
和等权 source-transfer。official-general-baseline 在 source context 间按扰动细胞数
加权；source-transfer 对 source context 等权，两者不混称。

主要结局：`family_rms_error`。`family_centroid_rmse` 和 `worst_seed_error` 作完整
描述，不替换主结局。

## 统计单位

- 复核预算固定为 20%；
- bootstrap 固定 5,000 次；
- pooled bootstrap 以 perturbation condition 为簇；
- 同一 condition 在不同 target 的任务一起进入或离开一次重采样；
- 每次抽取与原始 unique condition 数相同的簇，有放回；
- 四个 target 另行逐一计算点估计和区间；
- 低细胞数敏感性只给描述性关联，不进入主门。

同一 bootstrap draw 同时用于 SafeConf、magnitude、partial correlation、两种 20%
复核效用和 paired baseline 差，保证增量比较是配对的。随机种子由固定前缀
`E201::<scope>` 的 SHA-256 决定。

## 正式裁决

### 证书门

- 恒等式最大绝对残差 ≤ `1e-10`；
- 下界违反任务数为 0；
- 预测前 disagreement/magnitude/model-source gap 重算残差 < `5e-6`。

### 经验路由门

- pooled SafeConf–family RMS Spearman 的 95% cluster-bootstrap CI 下限 > 0；
- pooled SafeConf 20% oracle-normalized review utility 的 95% CI 下限 > 0。

### magnitude 增量门

满足任意一项：

- 控制 predicted magnitude 后的 pooled partial Spearman 95% CI 下限 > 0；
- SafeConf 减 magnitude 的配对 20% review-utility 增量 95% CI 下限 > 0。

四个 target 的结果无论方向如何全部输出。某个科学门未通过时，程序执行状态仍可
为 `PASS`；对应科学状态写成 `NOT_SUPPORTED`，不把负结果误报成软件失败，也不
回改风险权重、任务、种子或置信区间单位。

## 固定输出

代码仓库输出：

- `formal_core_evaluation/tables/E201_CORE_INPUT_HASHES.csv`；
- `formal_core_evaluation/tables/E201_TASK_METRICS.csv`；
- `formal_core_evaluation/tables/E201_SEED_TASK_ERRORS.csv`；
- `formal_core_evaluation/tables/E201_TARGET_ERROR_SUMMARY.csv`；
- `formal_core_evaluation/tables/E201_RISK_ASSOCIATIONS.csv`；
- `formal_core_evaluation/tables/E201_PARTIAL_ASSOCIATIONS.csv`；
- `formal_core_evaluation/tables/E201_REVIEW_UTILITY.csv`；
- `formal_core_evaluation/tables/E201_INCREMENTAL_TESTS.csv`；
- `formal_core_evaluation/tables/E201_BASELINE_COMPARISONS.csv`；
- `formal_core_evaluation/tables/E201_SENSITIVITY_ASSOCIATIONS.csv`；
- `formal_core_evaluation/tables/E201_MAIN_CLUSTER_BOOTSTRAP_DRAWS.csv`；
- `formal_core_evaluation/tables/E201_FORMAL_GATES.csv`；
- `formal_core_evaluation/reports/E201_CORE_REPORT.md`；
- `formal_core_evaluation/figures/E201_core_audit.png` 和 `.pdf`；
- `formal_core_evaluation/E201_CORE_FINAL_STATUS.json`。

目标真值 centroid 是约 `2,008 × 3,352` 的 float32 数组，只写数据盘并在状态文件
登记相对路径、形状、字节数和 SHA-256，不进入 Git。scPertEval 五端点另用独立
程序运行，避免核心 RMSE/证书因补充评价依赖失败而丢失。
