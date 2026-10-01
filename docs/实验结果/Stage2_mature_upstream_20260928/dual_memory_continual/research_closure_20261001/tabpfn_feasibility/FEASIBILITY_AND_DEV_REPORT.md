# 固定 TabPFN V2 保险检查：可行性与完整 DEV 证据

Built with PriorLabs-TabPFN。此项是既有 72h 计划中的算法稳健性诊断，不构成新增方法创新，也不用于根据已见外部结果替换主方法。

20 个既定 DEV 检查已经全部完成：两个方向 × Manual/Learned × 五个 outer folds，共 20 次 TabPFN、20 次主加权 HGB、20 次等样本权重 HGB。每个方向/参考的完整 OOF cohort 都有 1,808 个 query、575 个 gene，三种学习器完整配对。两方向/两参考的 TabPFN−加权 HGB Primary Utility@20 的 95% CI 均跨零；该诊断没有支持“主 HGB 明显偏弱”的稳定证据。

## 固定设置与公平性

- 使用官方原始 V2 回归权重 `tabpfn-v2-regressor.ckpt`，SHA256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`；Python 包固定 `tabpfn==9.0.0`，通过 `TabPFNRegressor.create_default_for_version(ModelVersion.V2)` 显式锁定 V2，实际默认 8 个 estimators。
- 所有 fit 使用 seed `20260930`。只覆写本地 `model_path`、seed 和运行设备，其余是官方 V2 默认值；没有参数搜索、模型选择或样本重采样。
- 每个 outer fold 使用同一冻结训练/query cohort、相同 13 个 P+PUBLIC 输入和已有 `NumericPreprocessor`（26 列，含缺失指示），同一 source-only error-rank CDF。训练 gene 与 query gene 严格隔离，Manual/Learned 的训练 rank labels 经 SHA 重算证明相同。
- TabPFN 的 `fit(X, y)` API **不支持 sample_weight**，采用等样本权重；主 HGB 使用 gene 等权 sample_weight。因此同时报告同输入、同 labels 的等样本权重 HGB。这个限制不能隐去，也不能将 TabPFN 与主 HGB 称为“同训练权重”。
- dataset/upstream/model_version/target/fold/gene/task_id 只用于冻结 split、CDF 分组或评估，不进入数字风险特征。输入文件 SHA 已在运行前后复核。
- 加权 HGB 的四个完整 OOF Primary Utility@20 与已归档主矩阵逐项复算一致（容差 1e-12），见 `CANONICAL_HGB_REPLAY_AUDIT.json`；只读取既有 macro 摘要完成此核验。

## 完整 OOF 的 Primary Utility@20

以下为按目标 context 聚合完整五个 folds 后的 macro，而非逐 fold 分数简单平均。CI 使用既有 `bootstrap_u20`，5,000 次共享 gene 配对抽样、seed `20260930`；没有缺失 query。

| 方向 | 参考 | 加权 HGB | 等样本 HGB | 固定 V2 | V2−加权 HGB | 配对 gene 95% CI |
|---|---|---:|---:|---:|---:|---|
| Exphormer_to_GAT | Learned | 0.783134 | 0.782712 | 0.792937 | +0.009802 | [-0.014704, +0.030700] |
| Exphormer_to_GAT | Manual | 0.780548 | 0.781429 | 0.781621 | +0.001072 | [-0.014761, +0.036531] |
| GAT_to_Exphormer | Learned | 0.783743 | 0.781223 | 0.780262 | -0.003481 | [-0.016401, +0.023287] |
| GAT_to_Exphormer | Manual | 0.786008 | 0.782025 | 0.783769 | -0.002239 | [-0.019955, +0.018961] |

V2 的 Spearman 在四个方向/参考组合都有小幅改善，AURC 也有改善；这些次要指标不等同于 Primary Utility@20 获胜。全部次要指标、逐 fold 分数及对两种 HGB 的配对区间见 CSV。

## 运行与下载资源

20 次 V2 的 fit 合计 9.518 秒，predict 合计 13.430 秒；保守以 fit/predict 墙钟记 GPU 使用 0.006375 GPU-hours。峰 allocated 208.5 MiB，reserved 302.0 MiB。使用 Quadro RTX 6000 24GB。

本地研究/base 环境无 TabPFN 缓存，原 PyTorch 2.11+cu130 与驱动不兼容。独立 probe 环境 `/home/yyf/.venvs/safeconf-tabpfn-20261001` 复用已存在、已验证 CUDA 可用的 TxPert PyTorch 2.6+cu124，没有更改主研究环境或下载新 GPU 库。公开权重+license 共 44,402,460 字节；将 24 个 PyPI wheels（含既有 cache）保守全额计入后，本项累计记账 58,987,083 字节（0.058987 GB）。各项公开 URL、SHA256、字节数在 `DOWNLOAD_LEDGER.json`。

## 官方访问与许可证据

官方 Hugging Face API 已核验 V2-reg `private=false`、`gated=false`、`disabled=false`；下载固定 revision `4972a65a1b30806315c6f92499959ffbfc69a673`，没有使用登录、凭证或账号，也没有执行许可点击。官方 V2 权重使用 Prior Labs License 1.1（Apache 2.0 衍生，加 attribution 条款）；许可区分内部测试与分发，后续若公开分发模型/代码/衍生物需遵守原文。

- [官方 V2 回归仓库](https://huggingface.co/Prior-Labs/TabPFN-v2-reg)
- [固定 revision 的标准权重](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/4972a65a1b30806315c6f92499959ffbfc69a673/tabpfn-v2-regressor.ckpt)
- [官方权重许可原文](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt)
- [官方版本和容量说明](https://docs.priorlabs.ai/models)
- [官方 V2 默认加载接口](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/regressor.py)

## 可复核文件与边界

- `FEASIBILITY.json`、`DOWNLOAD_LEDGER.json`：本地环境、公开获取和累计下载记账。
- `DEV_FULL_MACRO_RESULTS.csv`、`DEV_FULL_CONTEXT_RESULTS.csv`、`DEV_FOLD_MACRO_RESULTS.csv`、`DEV_FOLD_CONTEXT_RESULTS.csv`：完整 OOF 与逐 fold 证据。
- `DEV_PAIRED_CLUSTER_BOOTSTRAP.csv`：完整 cohort 的八组配对区间。
- `DEV_FULL_RESOURCE_COSTS.csv`、`DEV_FULL_INPUT_AUDIT.json`、`DEV_FULL_STATUS.json`：60 次 fit 的耗时、输入 SHA、CDF/特征/权重隔离审计。
- `CANONICAL_HGB_REPLAY_AUDIT.json`：主矩阵既有加权 HGB 的精确复算证据。
- 逐 query 风险仅存服务端 `/home/yyf/runtime_artifacts/safeconf_research_20261001/tabpfn_v2_probe/DEV_FULL_TASK_PREDICTIONS.csv.gz`。

没有加载新 external/test truth，没有 external matrix 推断，没有新增 upstream 调用或大模型训练。外部保险诊断需由负责人基于此完整 DEV 证据决定；本报告不生成论文正文。
