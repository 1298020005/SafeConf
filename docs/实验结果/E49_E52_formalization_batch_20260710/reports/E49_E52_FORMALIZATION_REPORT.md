# E49-E52 第二阶段正式化实验

- 生成时间：2026-07-10T17:20:40
- Git：`020ddea1b4e5`
- 工作区 dirty：`True`

## 1. 已跑实验

- E49 OpenProblems DGE：官方 train→test、训练集内部 cell-type holdout、compound holdout。
- E50 sciplex3：1000 基因 cell-line holdout。
- E51 Norman：单基因到组合扰动，mean/sum 两种单基因组合预测。
- E52 TCDD：dose-aware 留出，用最近剂量和 log-dose 线性趋势两个预测器。

## E49 OpenProblems pooled

                 experiment_group                       risk_score_name                target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
           official_train_to_test risk_oracle_true_magnitude_diagnostic        error_drug_mean_rmse      255  0.923077       51          1.077373          1.705793
           official_train_to_test              risk_predicted_magnitude         error_additive_rmse      255  0.642040       51          0.953555          1.250931
           official_train_to_test              risk_predicted_magnitude error_pair_or_additive_rmse      255  0.642040       51          0.953555          1.250931
           official_train_to_test                        risk_op_formal        error_cell_mean_rmse      255  0.634122       51          1.054072          1.288973
internal_cell_type_holdout_pooled              risk_predicted_magnitude error_pair_or_additive_rmse      602  0.615358      121          1.041180          1.568583
internal_cell_type_holdout_pooled              risk_predicted_magnitude        error_drug_mean_rmse      602  0.615358      121          1.041180          1.568583
internal_cell_type_holdout_pooled              risk_predicted_magnitude         error_additive_rmse      602  0.615358      121          1.041180          1.568583
           official_train_to_test                     risk_disagreement        error_cell_mean_rmse      255  0.594663       51          0.975709          1.193147
internal_cell_type_holdout_pooled risk_oracle_true_magnitude_diagnostic        error_drug_mean_rmse      602  0.567066      121          1.114591          1.679180
internal_cell_type_holdout_pooled risk_oracle_true_magnitude_diagnostic         error_additive_rmse      602  0.567066      121          1.114591          1.679180
internal_cell_type_holdout_pooled risk_oracle_true_magnitude_diagnostic error_pair_or_additive_rmse      602  0.567066      121          1.114591          1.679180
internal_cell_type_holdout_pooled risk_oracle_true_magnitude_diagnostic        error_cell_mean_rmse      602  0.526951      121          1.154530          1.803437

## E49 OpenProblems split-level

                               experiment                       risk_score_name                target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
internal_cell_type_holdout::Myeloid cells risk_oracle_true_magnitude_diagnostic error_pair_or_additive_rmse       15  0.971429        3          2.056672          2.266486
internal_cell_type_holdout::Myeloid cells risk_oracle_true_magnitude_diagnostic        error_drug_mean_rmse       15  0.971429        3          2.056672          2.266486
internal_cell_type_holdout::Myeloid cells risk_oracle_true_magnitude_diagnostic         error_additive_rmse       15  0.971429        3          2.056672          2.266486
      internal_cell_type_holdout::B cells risk_oracle_true_magnitude_diagnostic        error_drug_mean_rmse       15  0.939286        3          1.556860          2.159986
      internal_cell_type_holdout::B cells risk_oracle_true_magnitude_diagnostic         error_additive_rmse       15  0.939286        3          1.556860          2.159986
      internal_cell_type_holdout::B cells risk_oracle_true_magnitude_diagnostic error_pair_or_additive_rmse       15  0.939286        3          1.556860          2.159986
internal_cell_type_holdout::Myeloid cells              risk_predicted_magnitude        error_cell_mean_rmse       15  0.935714        3          3.081121          2.778967
internal_cell_type_holdout::Myeloid cells risk_oracle_true_magnitude_diagnostic        error_cell_mean_rmse       15  0.932143        3          3.081121          2.778967
                   official_train_to_test risk_oracle_true_magnitude_diagnostic        error_drug_mean_rmse      255  0.923077       51          1.077373          1.705793
internal_cell_type_holdout::Myeloid cells              risk_predicted_magnitude error_pair_or_additive_rmse       15  0.914286        3          2.056672          2.266486
internal_cell_type_holdout::Myeloid cells              risk_predicted_magnitude        error_drug_mean_rmse       15  0.914286        3          2.056672          2.266486
internal_cell_type_holdout::Myeloid cells              risk_predicted_magnitude         error_additive_rmse       15  0.914286        3          2.056672          2.266486

## E50 sciplex3

             dataset_name       setting                 risk_score_name          target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
sciplex3_cell_line_formal leave_context        risk_predicted_magnitude         error_v0_rmse      104  0.935842       21          0.221258          1.706680
sciplex3_cell_line_formal leave_context        risk_predicted_magnitude       error_mean_rmse      104  0.935458       21          0.220765          1.707994
sciplex3_cell_line_formal leave_context        risk_predicted_magnitude error_contextsim_rmse      104  0.935245       21          0.220425          1.709444
sciplex3_cell_line_formal leave_context             risk_safeconf_smoke error_contextsim_rmse      104  0.896057       21          0.212585          1.648640
sciplex3_cell_line_formal leave_context             risk_safeconf_smoke       error_mean_rmse      104  0.895641       21          0.212894          1.647097
sciplex3_cell_line_formal leave_context             risk_safeconf_smoke         error_v0_rmse      104  0.895342       21          0.213337          1.645586
sciplex3_cell_line_formal leave_context               risk_disagreement         error_v0_rmse      104  0.835842       21          0.222967          1.719863
sciplex3_cell_line_formal leave_context               risk_disagreement       error_mean_rmse      104  0.835757       21          0.222473          1.721209
sciplex3_cell_line_formal leave_context               risk_disagreement error_contextsim_rmse      104  0.835426       21          0.222133          1.722683
sciplex3_cell_line_formal leave_context risk_inverse_context_similarity error_contextsim_rmse      104 -0.037732       21          0.120361          0.933422
sciplex3_cell_line_formal leave_context risk_inverse_context_similarity       error_mean_rmse      104 -0.044767       21          0.120321          0.930889
sciplex3_cell_line_formal leave_context risk_inverse_context_similarity         error_v0_rmse      104 -0.049210       21          0.120283          0.927809

## E51 Norman

         experiment            risk_score_name           target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
Norman_combo_formal         risk_norman_formal  error_sum_single_rmse      125  0.615232       25          0.060404          1.285548
Norman_combo_formal risk_mean_single_magnitude  error_sum_single_rmse      125  0.615232       25          0.060404          1.285548
Norman_combo_formal risk_additive_disagreement  error_sum_single_rmse      125  0.615232       25          0.060404          1.285548
Norman_combo_formal  risk_sum_single_magnitude  error_sum_single_rmse      125  0.615232       25          0.060404          1.285548
Norman_combo_formal risk_mean_single_magnitude error_mean_single_rmse      125  0.364375       25          0.055092          1.200774
Norman_combo_formal         risk_norman_formal error_mean_single_rmse      125  0.364375       25          0.055092          1.200774
Norman_combo_formal  risk_sum_single_magnitude error_mean_single_rmse      125  0.364375       25          0.055092          1.200774
Norman_combo_formal risk_additive_disagreement error_mean_single_rmse      125  0.364375       25          0.055092          1.200774
Norman_combo_formal        risk_missing_single error_mean_single_rmse      125       NaN       25          0.049377          1.076213
Norman_combo_formal        risk_missing_single  error_sum_single_rmse      125       NaN       25          0.051924          1.105069

## E52 TCDD dose-aware

     experiment          risk_score_name       target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
TCDD_dose_aware     risk_tcdd_dose_aware    error_mean_rmse       48  0.712875       10          0.115037          1.461758
TCDD_dose_aware risk_predicted_magnitude error_nearest_rmse       48  0.693244       10          0.135704          1.560287
TCDD_dose_aware risk_predicted_magnitude    error_mean_rmse       48  0.677160       10          0.112114          1.424613
TCDD_dose_aware     risk_tcdd_dose_aware  error_linear_rmse       48  0.662288       10          0.103399          1.230066
TCDD_dose_aware     risk_tcdd_dose_aware error_nearest_rmse       48  0.638714       10          0.139367          1.602406
TCDD_dose_aware risk_predicted_magnitude  error_linear_rmse       48  0.612462       10          0.100327          1.193524
TCDD_dose_aware   risk_dose_disagreement  error_linear_rmse       48  0.437690       10          0.082905          0.986259
TCDD_dose_aware   risk_dose_disagreement error_nearest_rmse       48  0.398870       10          0.120098          1.380853
TCDD_dose_aware   risk_dose_disagreement    error_mean_rmse       48  0.382762       10          0.090748          1.153123
TCDD_dose_aware     risk_logdose_extreme    error_mean_rmse       48  0.248923       10          0.065018          0.826179
TCDD_dose_aware     risk_logdose_extreme  error_linear_rmse       48  0.191479       10          0.078939          0.939087
TCDD_dose_aware     risk_logdose_extreme error_nearest_rmse       48  0.103070       10          0.064604          0.742794

## 状态表


### OpenProblems splits

                                         experiment  n_train  n_test
                             official_train_to_test      602     255
                internal_cell_type_holdout::B cells      587      15
          internal_cell_type_holdout::Myeloid cells      587      15
               internal_cell_type_holdout::NK cells      458     144
           internal_cell_type_holdout::T cells CD4+      458     144
           internal_cell_type_holdout::T cells CD8+      462     140
     internal_cell_type_holdout::T regulatory cells      458     144
              internal_compound_holdout::Idelalisib      596       6
              internal_compound_holdout::Crizotinib      596       6
     internal_compound_holdout::Porcn Inhibitor III      596       6
               internal_compound_holdout::Foretinib      596       6
              internal_compound_holdout::LDN 193189      596       6
              internal_compound_holdout::CHIR-99021      596       6
              internal_compound_holdout::Dactolisib      596       6
internal_compound_holdout::O-Demethylated Adapalene      596       6
                internal_compound_holdout::MLN 2238      596       6
             internal_compound_holdout::Penfluridol      596       6
             internal_compound_holdout::Palbociclib      596       6
             internal_compound_holdout::Linagliptin      596       6
                    internal_compound_holdout::R428      596       6
    internal_compound_holdout::Oprozomib (ONX 0912)      597       5
               internal_compound_holdout::Alvocidib      597       5
      internal_compound_holdout::Mometasone Furoate      598       4
              internal_compound_holdout::Vorinostat      598       4
              internal_compound_holdout::Oxybenzone      598       4
      internal_compound_holdout::ABT-199 (GDC-0199)      598       4
              internal_compound_holdout::Trametinib      598       4
             internal_compound_holdout::Selumetinib      598       4
               internal_compound_holdout::Dasatinib      598       4
               internal_compound_holdout::Flutamide      598       4

### sciplex3 status

                  dataset                                                         path   n_obs  n_vars  n_genes_used context_col perturbation_col  n_tasks  n_contexts  n_perturbations status       setting heldout              dataset_name  n_train  n_test
sciplex3_cell_line_formal extra_official/cellular_context_generalization/sciplex3.h5ad 26046.0  5000.0        1000.0   cell_line     perturbation    104.0         3.0             36.0     ok           NaN     NaN                       NaN      NaN     NaN
                      NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    A549 sciplex3_cell_line_formal     68.0    36.0
                      NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    K562 sciplex3_cell_line_formal     69.0    35.0
                      NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    MCF7 sciplex3_cell_line_formal     71.0    33.0

### TCDD status

 n_tasks  n_celltypes  n_doses  n_genes
      48            6        8     1000