# Core result tables

## Current fixed results (2026-10-02)

The historical tables below this section use the earlier 512-gene contract and must not be mixed with current common2840 results. The canonical Source/McFaline points come from `common_gene_axis/results/MATRIX_MACRO_RESULTS.csv`; three registered training seeds are repeated fits, not independent datasets. Current values below use the first registered seed20260930. Raw RMSE/AURC values across different gene contracts are not pooled.

| Existing line (common2840) | Fixed method | U20 |
|---|---|---:|
| Exphormer_to_GAT | Learned_WeightedHistoryDistance | 0.593518 |
| Exphormer_to_GAT | Learned_hgb | 0.783134 |
| Exphormer_to_GAT | Prediction_hgb | 0.758953 |
| GAT_to_Exphormer | Learned_WeightedHistoryDistance | 0.592336 |
| GAT_to_Exphormer | Learned_hgb | 0.783743 |
| GAT_to_Exphormer | Prediction_hgb | 0.752714 |
| TxPert_to_McFaline | Learned_WeightedHistoryDistance | 0.729672 |
| TxPert_to_McFaline | Learned_hgb | 0.695253 |
| TxPert_to_McFaline | Prediction_hgb | -0.008533 |

### New Orion prospective truth-blind replication (3285 genes)

The primary history-supported cohort is232tasks/144geneclusters, out of2993eligibleTESTtasks/1750genes; its task coverage is7.75%. Every method uses the same232primary tasks. Models,13candidate IDs and scores were frozen before once-only TEST reading. TEST metadata was SEEN; no full-method pristine confirmation or full-task fallback is claimed.

| Frozen method | HCT116 U20 | HEK293T U20 | Macro U20 |
|---|---:|---:|---:|
| Magnitude | 0.115290 | 0.038942 | 0.077116 |
| P_only_ridge | 0.144158 | 0.038942 | 0.091550 |
| P_only_hgb | 0.070719 | 0.110517 | 0.090618 |
| Manual_ridge | 0.100288 | 0.198982 | 0.149635 |
| Manual_hgb | 0.196304 | 0.277732 | 0.237018 |
| Learned_ridge | 0.078153 | 0.198982 | 0.138568 |
| Learned_hgb | 0.160524 | 0.249146 | 0.204835 |
| Manual_WeightedHistoryDistance | 0.218687 | 0.154635 | 0.186661 |
| Learned_WeightedHistoryDistance | 0.236258 | 0.193517 | 0.214887 |
| Uniform_DirectRMSE | 0.274660 | 0.249581 | 0.262120 |
| Manual_DirectRMSE | 0.286916 | 0.201145 | 0.244031 |
| Learned_DirectRMSE | 0.252302 | 0.281829 | 0.267066 |
| NegativeSourceHistorySupport | -0.189015 | -0.123552 | -0.156284 |

Fixed primary Learned_hgb−Learned_WeightedHistoryDistance: ΔU20=−0.010052, paired nominal95%CI[−0.271236,0.191339]. HCT116 error@10 worsens7.598%, high-risk miss-rate increases0.090909; the registered primary gate is NOT_ESTABLISHED. The positive comparison with support alone (Δ0.361119,CI[0.081521,0.632134]) does not replace the primary comparison. Manual secondary and Public-versus-P-only intervals cross zero. Full399metric rows,1764paired contrasts,coverage and immutable completion receipts are retained under `orion_preparation/actual_fixed_result_tables_v1/`.

## Historical earlier-contract tables


| line | method | utility20 | spearman | aurc |
|---|---|---|---|---|
| Exphormer_to_GAT | Learned_WeightedHistoryDistance | 0.605786 | 0.599227 | 0.053083 |
| Exphormer_to_GAT | Learned_hgb | 0.793723 | 0.753558 | 0.048675 |
| Exphormer_to_GAT | Magnitude | 0.768216 | 0.741977 | 0.048831 |
| Exphormer_to_GAT | Manual_WeightedHistoryDistance | 0.589739 | 0.566887 | 0.053591 |
| Exphormer_to_GAT | Manual_hgb | 0.784847 | 0.744611 | 0.048874 |
| Exphormer_to_GAT | Prediction_hgb | 0.762670 | 0.734398 | 0.049132 |
| GAT_to_Exphormer | Learned_WeightedHistoryDistance | 0.626475 | 0.603109 | 0.052629 |
| GAT_to_Exphormer | Learned_hgb | 0.795018 | 0.749545 | 0.048570 |
| GAT_to_Exphormer | Magnitude | 0.758726 | 0.727634 | 0.048927 |
| GAT_to_Exphormer | Manual_WeightedHistoryDistance | 0.565115 | 0.570039 | 0.053188 |
| GAT_to_Exphormer | Manual_hgb | 0.786514 | 0.737704 | 0.048815 |
| GAT_to_Exphormer | Prediction_hgb | 0.764206 | 0.723861 | 0.048996 |
| TxPert_to_McFaline | Learned_WeightedHistoryDistance | 0.614206 | 0.639245 | 0.019812 |
| TxPert_to_McFaline | Learned_hgb | 0.551331 | 0.510874 | 0.020515 |
| TxPert_to_McFaline | Magnitude | -0.141916 | -0.118309 | 0.023005 |
| TxPert_to_McFaline | Manual_WeightedHistoryDistance | 0.604652 | 0.628957 | 0.019845 |
| TxPert_to_McFaline | Manual_hgb | 0.583540 | 0.271321 | 0.021314 |
| TxPert_to_McFaline | Prediction_hgb | 0.042859 | -0.011514 | 0.022245 |

| line | method_a | method_b | delta_utility20 | ci95_lower | ci95_upper |
|---|---|---|---|---|---|
| Exphormer_to_GAT | Manual_hgb | Manual_WeightedHistoryDistance | 0.195109 | 0.151362 | 0.270840 |
| Exphormer_to_GAT | Learned_hgb | Learned_WeightedHistoryDistance | 0.187937 | 0.127291 | 0.247440 |
| GAT_to_Exphormer | Manual_hgb | Manual_WeightedHistoryDistance | 0.221400 | 0.142230 | 0.266120 |
| GAT_to_Exphormer | Learned_hgb | Learned_WeightedHistoryDistance | 0.168543 | 0.117654 | 0.232377 |
| TxPert_to_McFaline | Manual_hgb | Manual_WeightedHistoryDistance | -0.021111 | -0.149550 | 0.031432 |
| TxPert_to_McFaline | Learned_hgb | Learned_WeightedHistoryDistance | -0.062875 | -0.124213 | 0.046060 |

## SEEN Orion validation reuse versus fixed direct rule

| 方法 | U20 | 相对同学习式直接距离的差值 | 名义95%配对CI |
|---|---:|---:|---|
| B100_TargetOnly_ridge | 0.029795 | -0.237271 | [-0.501003, 0.097587] |
| B100_TargetOnly_hgb | 0.073119 | -0.193947 | [-0.496712, 0.109124] |
| B100_PublicTarget_ridge | 0.279075 | +0.012009 | [-0.188223, 0.179268] |
| B100_PublicTarget_hgb | 0.168386 | -0.098679 | [-0.356386, 0.173309] |
| B100_SharedTarget_ridge | 0.243019 | -0.024047 | [-0.192460, 0.187497] |
| B100_SharedTarget_hgb | 0.069851 | -0.197214 | [-0.397087, 0.121020] |

All150fixed macroU20 rule comparisons have nominal95%paired CIs crossing zero; no primary promotion. See `orion_preparation/validation_strong_reference_completion_v1/`.

## Current2840 Public target-coverage growth, SEEN

| 迁移线 | 100%−10% U20增量 | 配对名义95%CI |
|---|---:|---|
| GAT_to_Exphormer | +0.033246 | [-0.001294, 0.049115] |
| Exphormer_to_GAT | +0.021191 | [-0.013601, 0.039759] |
| TxPert_to_McFaline | +0.680116 | [0.522410, 0.786253] |

Allfixedtasks HGB; sameSource labels100%, fivefixedorders. Coverage+representation/refit, not samegene content improvement; no PublicBiology update. FullMC risk remains below strongdistance.

## SEEN Orion actual-versus-physical-history null

| 原固定规则 | 真实U20 | 五次置换U20范围 | 名义正区间数／5 |
|---|---:|---|---:|
| Uniform_DirectRMSE | 0.262120 | [-0.095331, 0.130880] | 2/5 |
| Manual_DirectRMSE | 0.244031 | [-0.176300, -0.007733] | 2/5 |
| Manual_WeightedHistoryDistance | 0.186661 | [-0.039078, 0.179600] | 0/5 |

All5seeds retained;15macro point differencespositive but4nominal CIs>0, weighted5allcrosszero. Not newconfirmation or originalgate repair.

## SEEN full TRAIN target-control expression comparator

| 固定方法 | U20 | 相对control点差 | 名义95%配对CI |
|---|---:|---:|---|
| Magnitude | 0.077116 | -0.001187 | [-0.139833, 0.136044] |
| Learned_DirectRMSE | 0.267066 | +0.188762 | [-0.171296, 0.461058] |
| Learned_hgb | 0.204835 | +0.126532 | [-0.216172, 0.416231] |
| B100_PublicTarget_ridge | 0.279075 | +0.200771 | [-0.126186, 0.424130] |
| B100_SharedTarget_ridge | 0.243019 | +0.164715 | [-0.120531, 0.423764] |

ControlU20 .078304; all43macro paired superiorityintervals unsupported. Full38606controlinputbeyondP13, no newprimary.

## Source DEV PublicBiology持续重训（三固定臂）

| 开发评价角色 | 初始A U20 | 新bank/errors＋旧Bio B | 新bank/errors＋重训Bio C | C−B配对95%CI |
|---|---:|---:|---:|---|
| old anchor，335tasks/112genes |0.725469|0.760510|0.783727|[-0.028035,0.043883]|
| new task gate，345tasks/115genes |0.759756|0.753450|0.745657|[-0.031081,0.038584]|

只C为候选；C−A newgate −0.014099、miss＋0.025460使原门失败，服务保留A。C−B生物重建RMSE anchor −0.000280[-0.000538,-0.000068]、newgate −0.000355[-0.000614,-0.000122]；不把微小生物重建增益写作最终risk收益。SourceDEV观察性操作，原外部主结果不变。全48/6/42行及代码/失败在continual_runtime_replay/public_biology_refit_v2/。

## Source同任务强幅度对照（SEEN）

| line | context | method_a | method_b | metric | point_a | point_b | difference_a_minus_b | ci95_lower | ci95_upper | valid_draws | saved_draws | new_RNG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exphormer_to_GAT | macro | Learned_hgb | Magnitude | utility20 | 0.783134 | 0.748808 | 0.034327 | -0.009285 | 0.069899 | 5000 | 5000 | 0 |
| Exphormer_to_GAT | macro | Manual_hgb | Magnitude | utility20 | 0.780548 | 0.748808 | 0.031741 | -0.017315 | 0.058492 | 5000 | 5000 | 0 |
| Exphormer_to_GAT | macro | Prediction_hgb | Magnitude | utility20 | 0.758953 | 0.748808 | 0.010145 | -0.036005 | 0.041538 | 5000 | 5000 | 0 |
| GAT_to_Exphormer | macro | Learned_hgb | Magnitude | utility20 | 0.783743 | 0.746253 | 0.037491 | -0.002974 | 0.077648 | 5000 | 5000 | 0 |
| GAT_to_Exphormer | macro | Manual_hgb | Magnitude | utility20 | 0.786008 | 0.746253 | 0.039755 | -0.003418 | 0.073826 | 5000 | 5000 | 0 |
| GAT_to_Exphormer | macro | Prediction_hgb | Magnitude | utility20 | 0.752714 | 0.746253 | 0.006461 | -0.020953 | 0.047175 | 5000 | 5000 | 0 |

## E170固定有历史强对照

| cohort | method_a | method_b | metric | point_difference | ci95_lower | ci95_upper | valid_draws | total_draws |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
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

原V2使用1920目标验证错误；旧Gate原样保留。
