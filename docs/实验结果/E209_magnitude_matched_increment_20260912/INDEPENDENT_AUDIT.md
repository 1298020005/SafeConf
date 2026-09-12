# E209 独立复核

复核时间：2026-09-12

## 复核结论

正式输出通过独立点估计重算与自助表完整性检查。

独立代码没有调用 E209 的 `partial_spearman`、`batch_metrics` 或聚合函数，而是直接读取 E153 冻结输入，用 SciPy 平均秩和 NumPy 最小二乘逐折重算。八个研究的最大绝对差为 `5.56e-17`，八研究等权均值均为 `0.16456935081048552`。

| 研究 | 控制幅度后的偏 Spearman |
|---|---:|
| Frangieh | 0.220660 |
| Lara_exvivo | 0.393707 |
| Liang | 0.207976 |
| Nadig_two_cellline | 0.106419 |
| Replogle_two_cellline | 0.174997 |
| Santinha | 0.048701 |
| Shifrut | 0.039706 |
| Tian_CRISPRi | 0.124390 |

自助表含 4 个分析，每个分析恰有 2,000 个不同 replicate，编号均为 0–1,999；两项统计量全部为有限数。直接从 8,000 行原始 draws 重算 2.5%、50% 和 97.5% 分位数，与正式汇总的最大绝对差为 `1.11e-16`。逐批次表不存在重复的 `analysis × study × batch` 主键。

## 正确解释

E153 八研究结果满足本实验的回顾性判据：`8/8` 研究为正，95% 簇自助区间为 `[0.1179, 0.2076]`。它支持“SafeConf 含有预测幅度之外的风险信息”。

E209 使用已解封历史数据，且合同前的诊断已经观察到方向，因此不能写成新的盲测或外部确认。E201 此处 `0.2827` 是四个目标分别计算后的宏平均；原正式报告的 `0.2503` 是 1,808 个任务合并统计，两种口径都保留，不能互换。

E187 有部分任务缺少可计算的 SafeConf 输入，正式程序仅按该输入缺失排除，不读取误差决定去留。E187 与 E153 还共享历史来源，因此只作为压力测试，不增加独立数据集数量。

## 关键文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `E209_REPORT.md` | `97a92965ba7cc2255284664294f41e7090c731917a652e97cd371858ecac4260` |
| `RUN_STATUS.json` | `4dfac5142e5414f0fe97bb39c73c14d87cdafceb119f3be6a233b9d82bb41cf4` |
| `E209_BATCH_RESULTS.csv` | `32032f427869393a9e7c5fdfb7a27f5cb0aecb92939fa563cd37f98d6f61f3cf` |
| `E209_BOOTSTRAP_DRAWS.csv` | `d4454ea4f170e8de49c1ad4c8576518c471406349c35b1ac2a180d03cabec2ba` |
| `E209_BOOTSTRAP_SUMMARY.csv` | `9cc3e26ea3af307130e75b55189f38ee0fb21f05f72273b6a1c1b3b3c8371549` |
| `E209_OVERALL_SUMMARY.csv` | `9edfd93e8dcb2ab547f244f2af949ad049a05f74d31ce2d393e41c3c5da4ff05` |
| `E209_STUDY_SUMMARY.csv` | `47fc21f8c7938856e78f1792d678b62ebb128d2d28bc6a88ed11a65a80a186bc` |
