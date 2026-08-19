# E198｜arch1 评价协议校准

分析性质：`EXTERNAL_PROTOCOL_CALIBRATION`。本实验只校准评价协议，没有模型预测，
不能写成 SafeConf 外部验证。

## 完整性

- Git freeze：`dfed0154f5ba45df6613f6f22e734f8be336e870`；
- 12 个固定协议 × 150 个扰动；
- integrity gates：14/14；
- official DRF 与独立公式最大差：`0.0`；
- 总协议运行时间：116.0 秒；
- PRIMARY / SECONDARY / REJECT：12 /
  0 / 0。

## 逐协议结果

| protocol | axis | BDS | BDS 95% lower | median DRF | DRF 95% lower | decision |
|---|---|---:|---:|---:|---:|---|
| `pearson` | direction | 1.0000 | 0.9750 | 0.6726 | 0.6211 | `PRIMARY_ELIGIBLE` |
| `pearson_ctrl` | direction | 1.0000 | 0.9750 | 0.7656 | 0.7291 | `PRIMARY_ELIGIBLE` |
| `pearson_pert` | direction | 0.9400 | 0.8899 | 0.7878 | 0.6889 | `PRIMARY_ELIGIBLE` |
| `mse` | absolute | 1.0000 | 0.9750 | 0.6711 | 0.6224 | `PRIMARY_ELIGIBLE` |
| `wmse_exp2` | absolute | 1.0000 | 0.9750 | 0.9964 | 0.9949 | `PRIMARY_ELIGIBLE` |
| `rank` | retrieval | 0.9933 | 0.9632 | 1.0000 | 1.0000 | `PRIMARY_ELIGIBLE` |
| `transpose_rank` | retrieval | 0.9933 | 0.9632 | 1.0000 | 1.0000 | `PRIMARY_ELIGIBLE` |
| `energy_distance_pca_k=50` | population | 1.0000 | 0.9750 | 1.0000 | 1.0000 | `PRIMARY_ELIGIBLE` |
| `unbiased_mmd_median_pca_k=50` | population | 1.0000 | 0.9750 | 1.0000 | 1.0000 | `PRIMARY_ELIGIBLE` |
| `de_auprc` | de | 0.9867 | 0.9527 | 0.4560 | 0.3949 | `PRIMARY_ELIGIBLE` |
| `de_auroc` | de | 0.9800 | 0.9429 | 0.6686 | 0.6264 | `PRIMARY_ELIGIBLE` |
| `de_overlap_k=50` | de | 0.9733 | 0.9334 | 0.6667 | 0.5683 | `PRIMARY_ELIGIBLE` |

## 事前优先级选出的后续端点

- absolute: `mse`
- direction: `pearson_pert`
- retrieval: `rank`
- population: `energy_distance_pca_k=50`
- de: `de_auprc`

没有通过的 axis 留空，E199 不得依据模型表现临时更换协议。
