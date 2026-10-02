# Target-control 强对照：独立实际核验

**结论：实际执行／固定队列／保存draw配对核验 PASS。当前结果不能排除control解释，也没有证明Public或学习型方法相对control的稳定增量；不升级原Gate或Source迁移主张。**

## 核验范围和完整性

实际 `COMPLETE_FIXED_SEEN_TARGET_CONTROL_COMPARISON`，97.388010秒，原232 tasks／144 gene clusters，44 methods／43 comparators，924单方法行／903配对行，0新拟合／模型调用／bootstrap draw。执行代码SHA为`3858b4c111635f648b70ab5a7c74d92ea4047529a8e5320146a5bb4131eebd94`。

本次只读取已有CSV、manifest、identity列及保存的统计draws：

- 回执两份CSV的SHA／size一致；相关已有draw、score和identity输入的SHA／size与实际回执一致。
- 原primary identity、封存的新control identity逐行相同，且与Control33的完整task集合相同；HCT116107／HEK293T125，232唯一query／144 genes，没有缩到87个target在3285 readout内的任务。
- Control33 `5000×3×33×7`、StrongRef4 `5000×3×4×7`、本次6原规则＋control `5000×3×7×7`的别名／轴顺序按执行代码和原seal精确解析；saved gene indices为`5000×144`、均在0–143范围，canonical sorted gene集合一致。
- 从上述已有数组逐项复现全部**903配对CI和924单方法CI，精确一致**；所有配对点差精确等于point_a−point_control。未重新调用指标函数、采样、拟合或读取缓存／raw truth数值。
- 901配对项和922单方法项有5000 valid draws；`P_only_hgb`的HEK293T及macro Spearman各有4352 valid draws，其NaN及有效率被保留。不是所有指标都5000有效；全部U20项为5000有效。本次没有填补未定义draw或按其删任务。

实际回执也记录原LearnedHGB每个saved draw、context／macro anchor复算门通过；本次未再次读取errors重复该指标复算。

## 固定结果与不确定性

Control唯一规则为高own-context TRAIN target-gene mean-cell-logCP4000表达＝高风险。初始CP10000文字已由Root更正，未执行换算／反号／参数选择。

| U20 | 点估计 | 名义95% gene-bootstrap CI |
|---|---:|---|
| Control HCT116 | 0.153833 | [-0.093869, 0.450002] |
| Control HEK293T | 0.002775 | [-0.178082, 0.251466] |
| Control macro | 0.078304 | [-0.092418, 0.305068] |

43个原／Target方法−control的macro U20，**没有一个CI下界大于0**。仅NegativeSourceHistorySupport−control区间全负：`-0.234587 [-0.504832,-0.018513]`，是名义比较，不能当成多重比较校正后的普遍结论。全部43比较保留，不选预算或方法。

| 固定方法−control：macro U20 | 点差 | 名义95% paired CI |
|---|---:|---|
| Learned DirectRMSE | +0.188762 | [-0.171296, +0.461058] |
| 原Learned HGB | +0.126532 | [-0.216172, +0.416231] |
| 100% PublicTarget Ridge | +0.200771 | [-0.126186, +0.424130] |

Control的点估计低于以上Public参照／学习器，因此这个单标量在观察到的点估计层面不足以重现全部Public表现。但配对区间允许零及负增量，**尚未排除control，也未证明Public具有稳定的额外贡献**。不能将CI跨零写成“等价”“无效”，更不能将正点差写成“已超越强control”。Control自身的macro与两context区间也跨零，不能把它称为稳定的全部误差解释。

## 信息与科学边界

- Control使用已获准的完整38606 TRAIN NTC域，包含145个target位于原3285 readout之外的任务；它是额外可用生物输入的强对照，不是原P6＋PUBLIC7相同feature信息预算。0额外调用／新标签不等于0生物信息。
- 30个Target learners原已使用2790 validation tasks／1661 genes；这里0fit只是新增统计成本。它们是同一published C family两个context checkpoint的目标研究监督补充，不是新增独立上游家族或单checkpoint版本化ErrorAdapter证据。
- Control与Public参照是在同一已见队列上的预测前输入关联比较，未分解target响应、library／control测量误差或潜在生物真值。不能将其当作测量机制的因果消除，也不能以control自身不显著来排除该机制。
- 原13主方法、Source／LM参数、NC和主安全规则完整保留。当前比较不修复原HCT gate失败、不建立Source-only错误监督的稳定迁移、不产生新确认资格。不试反号、参数、额外分层或事后子集。

**可用结论句：**“固定TRAIN target-gene control表达代理的macro排序点估计为0.0783。Public参照和部分学习器的点估计更高，但在完整232已见任务上，它们相对control的配对区间仍包含零；现有证据尚不足以证明Public内容或Source监督在该额外生物信息强对照之外具有稳定增量。”

## 本次精确证据绑定

- 实际`RESULT_MANIFEST.json`：`cc2c82bfa3b6084b931ecedfce4a8d611098e2cb814baa69d0bbe139402928be`。
- `ALL_FIXED43_METHODS_MINUS_TARGET_CONTROL.csv`：`1e81ce1a46f3577483197e152a6ab27e68005b1df2ff867b704043f4bc0f959c`。
- `ALL_FIXED44_METHOD_METRIC_INTERVALS.csv`：`d66685fe10faf31263dfe067987a65faa3dfa36f15a0a4b8b03507c8498e93bf`。
- runtime`SIX_ORIGINAL_PLUS_NEW_CONTROL_METRIC_DRAWS.npy`：`cf57a78771e3b4e74bc7e70ae485ac31b05e870e26df6ef6d7eba487ca04e976`。
- 原5000 gene indices：`be533a498413bf284e187faf2d8c3d103949f865ba9ef0765d71ef7925b9479c`。

仅新增本审阅文件；未执行第二个统计job、改变已封存输入／方法或追加模型数值实验。
