# E202 冻结协议：控制扰动幅度后的模型失败诊断

冻结日期：2026-08-02

## 为什么做

E200 在 K562 整体背景留出中得到两个同时成立的事实：SafeConf 的
`transfer_risk` 与 TxPert-GAT 质心 RMSE 正相关，但 `predicted_magnitude`
明显更强。RMSE 会随预测效应幅度自然增大，因此 E202 不再重复检验“哪一个任务
绝对误差大”，而是检验训练背景之间的效应分歧能否解释 GAT 相对简单基线多犯的
错误。

E202 是看到 E200 结果后登记的诊断分析，不能追溯称为 E200 的事前确认性结果。

## 固定输入

| 输入 | SHA-256 |
| --- | --- |
| `../E200_txpert_cross_context_k562_20260802/formal_evaluation/tables/E200_TASK_METRICS.csv` | `91fc71c767ed4742bc794c1d55e6fd00dbfe77294f59201717233c702aff9062` |
| `../E200_txpert_cross_context_k562_20260802/formal_evaluation/tables/E200_SCPERTEVAL_TASK_METRICS.csv` | `5003410d21eef7f7fa3ffac53131a6ddf49885187435f33e94cc35c0f60f83a3` |

只使用 E200 的 566 个 `primary_ge30` 任务，不改任务门槛，不删除离群值，不按结果
筛基因。

## 主问题

- 主风险量：`training_delta_dispersion`；
- 固定混杂量：`predicted_magnitude`；
- 主结局：
  `gat_excess_rmse_vs_general = gat_centroid_rmse - general_baseline_centroid_rmse`；
- 方向：值越大表示 GAT 相对 general baseline 越差，预期风险方向为正。

主统计量为 partial Spearman。先分别把风险量和主结局转为平均秩，再各自对
`predicted_magnitude` 的平均秩做带截距的一元线性残差化，最后计算两组残差的
Pearson 相关。95% 区间使用 5,000 次任务级有放回 bootstrap，随机种子由
`SHA256("E202::<analysis label>")` 固定生成。

主结论只有两种：

- `SUPPORTED`：主 partial Spearman 的 95% 区间下界大于 0；
- `NOT_SUPPORTED`：其余所有情况。

不会用任一次要终点替换主门槛。

## 固定次要分析

1. 对主结局报告未控制的 Spearman，帮助解释控制幅度前后的变化。
2. 对 `transfer_risk`、`model_baseline_gap`、`negative_log_train_cells` 和
   `support_context_deficit` 运行相同 partial Spearman；只作成分诊断。
3. 将主结局替换为
   `gat_centroid_rmse - control_centroid_rmse`，检查参照基线的敏感性。
4. 在 scPertEval 的 `mse`、`pearson_pert`、`rank`、
   `energy_distance_pca_k=50`、`de_auprc` 五个定向误差上，计算
   `GAT oriented_error - general baseline oriented_error`，再运行相同的
   partial Spearman。所有 `oriented_error` 已在 E200 中固定为越大越差。
5. 将任务按 `predicted_magnitude` 的固定五分位分层，在每层内中心化风险量和
   主结局，报告合并后的描述性 Spearman及每层中位数；该结果不参与主门槛。

所有项目完整输出，不按显著性选择展示。bootstrap 区间是任务重抽样的不确定性
描述；单一 K562 数据集不能据此声称跨数据集泛化。

## 完整性要求

- 两个输入哈希必须完全匹配；
- 主表必须恰有 566 个唯一任务；
- 五端点表必须包含 566 × 3 × 5 行，并且每个
  `(task_id, predictor, endpoint)` 唯一；
- 运行脚本和本冻结文件必须已进入当前 Git commit，且两者无未提交修改；
- 正式输出目录已存在时拒绝覆盖；
- 输出表、报告、PNG/PDF 和状态文件全部生成后再发布。

## 能回答与不能回答

E202 能区分“风险量只是在识别大效应任务”与“控制效应幅度后仍能识别
GAT 相对基线的额外失败”。它不能替代其他目标细胞系、多随机种子、其他模型家族、
跨实验室数据或前瞻湿实验。
