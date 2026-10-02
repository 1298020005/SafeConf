# E170 原确认的强规则补充（SEEN）

## 信息和执行边界

这是旧E170揭盲后的固定强对照补齐，不是新确认或重训练。原六种预测、Gate A和输入SHA均保留；补充分数先封存再解析缓存误差，无新原始表达读取。

原监督模型用了1920条目标ensemble validation errors。validation donor CE0008678，test donor CE0010866；known640gene在validation有误差监督，属于同研究held-out donor，非Source-only未见gene迁移。四panel分别512gene轴、guide pseudobulk log1p(CP10000)效应，不是当前2840/3285 CP4000核心确认。

full2400tasks/800genes，known1920/640，missing480/160，各5000次整gene/三state成对抽样，固定种子20261002/3/4，不宣称原确认CI逐值复现。

## 固定规则和结果

DirectHistoryDistance=-negative_model_source_gap；DistancePlusHistoryDispersion=sqrt(gap^2+source_delta_dispersion^2)；NegativeLogHistorySupport=-log1p(n_source_cells)。前两者480无历史保留NA，不填零效应。数量规则允许观测零历史数作为额外full控制。known每panel/state支持量恒定，内部无排序信息、Spearman保持NA。

| cohort | method | tasks | gene_clusters | utility20 |
| --- | --- | --- | --- | --- |
| FULL_ORIGINAL_2400 | Magnitude_raw | 2400 | 800 | 0.165509 |
| FULL_ORIGINAL_2400 | Ridge_U | 2400 | 800 | 0.183584 |
| FULL_ORIGINAL_2400 | Ridge_US | 2400 | 800 | 0.183584 |
| FULL_ORIGINAL_2400 | Ridge_USR | 2400 | 800 | 0.201032 |
| FULL_ORIGINAL_2400 | Ridge_USRCH | 2400 | 800 | 0.215895 |
| FULL_ORIGINAL_2400 | V2_nested | 2400 | 800 | 0.215895 |
| FULL_ORIGINAL_2400 | NegativeLogHistorySupport | 2400 | 800 | 0.055194 |
| KNOWN_HISTORY_1920 | Magnitude_raw | 1920 | 640 | 0.219760 |
| KNOWN_HISTORY_1920 | Ridge_U | 1920 | 640 | 0.235520 |
| KNOWN_HISTORY_1920 | Ridge_US | 1920 | 640 | 0.235520 |
| KNOWN_HISTORY_1920 | Ridge_USR | 1920 | 640 | 0.271966 |
| KNOWN_HISTORY_1920 | Ridge_USRCH | 1920 | 640 | 0.272226 |
| KNOWN_HISTORY_1920 | V2_nested | 1920 | 640 | 0.272010 |
| KNOWN_HISTORY_1920 | DirectHistoryDistance | 1920 | 640 | 0.242532 |
| KNOWN_HISTORY_1920 | DistancePlusHistoryDispersion | 1920 | 640 | 0.247730 |
| KNOWN_HISTORY_1920 | NegativeLogHistorySupport | 1920 | 640 | -0.056398 |
| NO_HISTORY_480 | Magnitude_raw | 480 | 160 | 0.045065 |
| NO_HISTORY_480 | Ridge_U | 480 | 160 | 0.027796 |
| NO_HISTORY_480 | Ridge_US | 480 | 160 | 0.027796 |
| NO_HISTORY_480 | Ridge_USR | 480 | 160 | 0.015855 |
| NO_HISTORY_480 | Ridge_USRCH | 480 | 160 | 0.015011 |
| NO_HISTORY_480 | V2_nested | 480 | 160 | 0.015011 |
| NO_HISTORY_480 | NegativeLogHistorySupport | 480 | 160 | 0.066721 |

### 固定U20差值

| cohort | method_a | method_b | metric | point_difference | ci95_lower | ci95_upper | valid_draws | total_draws |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FULL_ORIGINAL_2400 | V2_nested | Magnitude_raw | utility20 | 0.050387 | -0.026048 | 0.115683 | 5000 | 5000 |
| FULL_ORIGINAL_2400 | V2_nested | Ridge_U | utility20 | 0.032311 | -0.019282 | 0.083357 | 5000 | 5000 |
| FULL_ORIGINAL_2400 | V2_nested | Ridge_US | utility20 | 0.032311 | -0.019282 | 0.083357 | 5000 | 5000 |
| FULL_ORIGINAL_2400 | V2_nested | Ridge_USR | utility20 | 0.014863 | -0.017810 | 0.040741 | 5000 | 5000 |
| FULL_ORIGINAL_2400 | V2_nested | Ridge_USRCH | utility20 | -0.000000 | -0.008282 | 0.005499 | 5000 | 5000 |
| FULL_ORIGINAL_2400 | V2_nested | NegativeLogHistorySupport | utility20 | 0.160701 | 0.028941 | 0.290751 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | V2_nested | Magnitude_raw | utility20 | 0.052250 | -0.032127 | 0.132240 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | V2_nested | Ridge_U | utility20 | 0.036490 | -0.019743 | 0.103061 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | V2_nested | Ridge_US | utility20 | 0.036490 | -0.019743 | 0.103061 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | V2_nested | Ridge_USR | utility20 | 0.000044 | -0.029516 | 0.043188 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | V2_nested | Ridge_USRCH | utility20 | -0.000215 | -0.007619 | 0.004183 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | V2_nested | DirectHistoryDistance | utility20 | 0.029478 | -0.018512 | 0.090661 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | V2_nested | DistancePlusHistoryDispersion | utility20 | 0.024281 | -0.028890 | 0.084104 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | V2_nested | NegativeLogHistorySupport | utility20 | 0.328408 | 0.217423 | 0.447275 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | DirectHistoryDistance | Magnitude_raw | utility20 | 0.022771 | -0.070604 | 0.096167 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | DistancePlusHistoryDispersion | DirectHistoryDistance | utility20 | 0.005198 | -0.037357 | 0.056680 | 5000 | 5000 |
| KNOWN_HISTORY_1920 | NegativeLogHistorySupport | Magnitude_raw | utility20 | -0.276159 | -0.403205 | -0.161242 | 5000 | 5000 |
| NO_HISTORY_480 | V2_nested | Magnitude_raw | utility20 | -0.030054 | -0.180912 | 0.183269 | 5000 | 5000 |
| NO_HISTORY_480 | V2_nested | Ridge_U | utility20 | -0.012786 | -0.103396 | 0.099804 | 5000 | 5000 |
| NO_HISTORY_480 | V2_nested | Ridge_US | utility20 | -0.012786 | -0.103396 | 0.099804 | 5000 | 5000 |
| NO_HISTORY_480 | V2_nested | Ridge_USR | utility20 | -0.000845 | -0.063014 | 0.054792 | 5000 | 5000 |
| NO_HISTORY_480 | V2_nested | Ridge_USRCH | utility20 | 0.000000 | -0.001178 | 0.025474 | 5000 | 5000 |
| NO_HISTORY_480 | V2_nested | NegativeLogHistorySupport | utility20 | -0.051711 | -0.326664 | 0.164406 | 5000 | 5000 |

## 判断与复现

旧Gate A点门通过保留；本轮full V2−Magnitude +0.050387[-0.026048,0.115683]，原CI也跨零。known V2−Direct +0.029478和−DistancePlusDispersion +0.024281均跨零；missing V2−Magnitude −0.030054[-0.180912,0.183269]。这不证明门控普遍优于强规则、安全回退或Source-only跨家族有效。数量规则弱，不据此宣称主要增量。

23macro/276strata/161pairs完整保留，158pairs为5000有效抽样，3个constant-Support Spearman差值保持NA。原full六方法42点重现。耗时28.047s，CPU增量99.872s，峰值343154688B，0fit/upstream/GPU。INDEPENDENT_ACTUAL_REVIEW.md独立核原输入、代码、数组SHA及所有CI/点。完整scores/counts/draws位于RESULT_MANIFEST列出的runtime目录，不进入Git。

复现脚本tools/scripts/evaluate_safeconf_e170_fixed_strong_references.py拒绝覆盖完成输出；脚本的输出根为固定常量，完整重新运行需在隔离副本中登记新的REPORT/RUNTIME路径及代码哈希；常规数值复核直接使用已存数组，不覆盖原结果。停止继续变规则/挑子组或重选旧确认winner。
