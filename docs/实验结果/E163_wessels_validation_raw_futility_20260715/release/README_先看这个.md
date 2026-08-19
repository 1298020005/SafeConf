# E163 Wessels validation-only raw-score futility diagnostic

运行决策：`allow_new_test_label_only_preregistration`。

E163 只使用 E161 train/validation development 资产和 E162 已锁定的 validation 分数。Wessels test label、test expression、test truth 与 raw Wessels H5AD 均未访问。

## 主要关联

| seed | raw_log_prob → PCA10 own-model Pearson accuracy rho | estimable | task bootstrap 95% CI | component-gene bootstrap 95% CI | LOGO min / median / max |
|---:|---:|:---:|:---:|:---:|:---:|
| 3407 | 0.1617 | True | [-0.2469, 0.5302] | [-0.1636, 0.4224] | -0.1702 / 0.1767 / 0.2874 |
| 3408 | 0.0426 | True | [-0.4172, 0.4732] | [-0.2826, 0.3078] | -0.3368 / 0.0464 / 0.2158 |
| 3409 | -0.2261 | True | [-0.6376, 0.2574] | [-0.5429, 0.0713] | -0.3955 / -0.2366 / -0.0241 |

Authorization gate 固定要求三个主要关联均可估计、3407 rho>0、且至少两个种子 rho>0。本次 gate 为 `True`。bootstrap、cluster bootstrap 和 LOGO 只描述不确定性及共享组分依赖，不参与 gate。

## 强制 raw selected-gene truth 敏感性

| seed | raw_log_prob → raw-truth Pearson rho | estimable | raw_log_prob → raw-truth RMSE rho | estimable |
|---:|---:|:---:|---:|:---:|
| 3407 | -0.1522 | True | -0.0930 | True |
| 3408 | -0.1417 | True | 0.0061 | True |
| 3409 | -0.3748 | True | 0.2235 | True |

raw truth 结果是预先强制的敏感性。它与 PCA10 主真值方向不一致时也必须保留，不能据此替换主 endpoint。

## 结论边界

E163 是看见 E162 prediction collapse 和 raw-score variation 后进行的 validation-informed 去留诊断。即使 gate 通过，也只允许另行提交一个新的 test-label-only 预注册步骤；不能写成外部验证、确认成功、模型预测非退化或 SafeConf 已在 Wessels 得到支持。
