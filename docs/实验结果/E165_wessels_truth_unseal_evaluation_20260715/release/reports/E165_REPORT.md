# E165 Wessels一次性test truth评价报告

## 访问边界

- 不可逆事件：`../TEST_TRUTH_UNSEAL_EVENT.json`；event在任何raw文件open之前落盘。
- 唯一expression读取：9,902 test rows × 前20,631 endogenous columns。
- 421 engineered/guide/barcode columns、train/validation/excluded rows读取数均为0。
- 归一化：每cell `log1p(10000 × endogenous count / endogenous library)`，再取冻结2,023 selected axis。
- baseline arm：已执行；PRESCRIBE arm：`True`。

## 五级baseline

- control_no_change: mean PCA10 RMSE=0.1198; raw RMSE=0.1322; centroid accuracy=0.5
- cell_weighted_perturbed_mean: mean PCA10 RMSE=0.0829; raw RMSE=0.09588; centroid accuracy=0.5
- condition_balanced_perturbed_mean: mean PCA10 RMSE=0.08195; raw RMSE=0.09464; centroid accuracy=0.5
- matching_single_mean: mean PCA10 RMSE=0.05484; raw RMSE=0.07333; centroid accuracy=0.8449
- single_additive: mean PCA10 RMSE=0.05074; raw RMSE=0.08655; centroid accuracy=0.9783

H1固定为 `mean(RMSE_cellweighted - RMSE_matching)`，observed=0.02806，通过=True。

H2 observed rho=0.0761，通过=False，解释层级=`confirmatory_H1_passed`。H1未通过时，H2只保留描述性结果。

## PRESCRIBE raw score

PRESCRIBE主种子3407：raw_log_prob与PCA10 own-model Pearson accuracy的Spearman rho=-0.2097，task 95% CI [-0.4782, 0.08653]，component-gene 95% CI [-0.4713, 0.05851]，预注册门通过=False。

PCA10 truth是E160主要口径；raw selected-gene truth敏感性、task bootstrap、component-gene bootstrap和LOGO均强制留存。

## Systema、SBB与split-half

Systema perturbed reference使用train condition centroids等权平均的冻结profile；官方centroid accuracy是正确centroid距离严格小于其余47个距离的比例，另报告更严格的nearest-centroid hit，tie在两者中都失败。语义核对来源为Systema官方代码commit `aaf5b5353993b48b78543f2f93b3e18ca65df515`。

split-half按condition+obs_name SHA确定性分半，平均raw-effect Pearson为0.8742。它是reproducibility benchmark/reference，不是upper bound：每半样本数小于完整truth。

Top20仅按truth effect绝对值选择，作为SBB signal-sensitive诊断；五级baseline hierarchy提供性能地板。除预注册H1/H2和native主终点外，Systema/SBB扩展均为contextual/descriptive。

## 方法来源

- [Systema](https://doi.org/10.1038/s41587-025-02777-8)
- [SBB principles](https://doi.org/10.64898/2026.04.20.719650)
- [TxPert](https://doi.org/10.1038/s41587-026-03113-4)

所有有利与不利结果均保留；本实验不保证期刊录用。
