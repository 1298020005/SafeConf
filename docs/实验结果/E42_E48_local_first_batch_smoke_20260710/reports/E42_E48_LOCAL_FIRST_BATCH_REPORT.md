# E42-E48 本地第一批 smoke

- 生成时间：2026-07-10T16:57:19
- Git：`900abafd5dad`
- 工作区 dirty：`True`

## 1. 普通任务矩阵结果

覆盖 sciplex3、TCDD、KaggleCrossPatient、crossSpecies。每个任务先用轻量参考预测器出 error，再看风险分数排序。

任务构建：

                 dataset                                                                   path  n_obs  n_vars  n_genes_used context_col perturbation_col  n_tasks  n_contexts  n_perturbations status error
      sciplex3_cell_line           extra_official/cellular_context_generalization/sciplex3.h5ad  26046    5000           256   cell_line     perturbation      104           3               36     ok      
           TCDD_celltype               extra_official/cellular_context_generalization/TCDD.h5ad 103745    5000           256    celltype     perturbation        6           6                1     ok      
               TCDD_dose               extra_official/cellular_context_generalization/TCDD.h5ad 103745    5000           256    celltype             dose       48           6                8     ok      
KaggleCrossPatient_donor extra_official/cellular_context_generalization/KaggleCrossPatient.h5ad  25583    5000           256    donor_id     perturbation       30           3               10     ok      
    crossSpecies_species       extra_official/cellular_context_generalization/crossSpecies.h5ad 112903    5000           256  condition1     perturbation       12           4                3     ok      

Spearman 较高的项目：

            dataset_name       setting          risk_score_name          target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
      sciplex3_cell_line leave_context risk_predicted_magnitude error_contextsim_rmse      104  0.879216       21          0.204066          1.477704
      sciplex3_cell_line leave_context risk_predicted_magnitude       error_mean_rmse      104  0.872879       21          0.207612          1.485001
      sciplex3_cell_line leave_context risk_predicted_magnitude         error_v0_rmse      104  0.862605       21          0.212338          1.491126
      sciplex3_cell_line leave_context      risk_safeconf_smoke error_contextsim_rmse      104  0.826191       21          0.188310          1.363610
      sciplex3_cell_line leave_context      risk_safeconf_smoke       error_mean_rmse      104  0.816877       21          0.190837          1.365012
      sciplex3_cell_line leave_context      risk_safeconf_smoke         error_v0_rmse      104  0.802197       21          0.194189          1.363672
      sciplex3_cell_line leave_context        risk_disagreement error_contextsim_rmse      104  0.776281       21          0.203403          1.472899
      sciplex3_cell_line leave_context        risk_disagreement         error_v0_rmse      104  0.774708       21          0.212100          1.489455
      sciplex3_cell_line leave_context        risk_disagreement       error_mean_rmse      104  0.774692       21          0.207118          1.481469
KaggleCrossPatient_donor leave_context risk_predicted_magnitude         error_v0_rmse       30  0.541268        6          0.026599          1.179463
KaggleCrossPatient_donor leave_context risk_predicted_magnitude       error_mean_rmse       30  0.523026        6          0.026448          1.176846
KaggleCrossPatient_donor leave_context risk_predicted_magnitude error_contextsim_rmse       30  0.494105        6          0.026325          1.174154

## 2. Norman 单基因到组合

          dataset_name          risk_score_name               target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
Norman_single_to_combo  risk_norman_combo_smoke error_single_additive_rmse      125  0.426937       25          0.068132          1.267948
Norman_single_to_combo risk_predicted_magnitude error_single_additive_rmse      125  0.426937       25          0.068132          1.267948
Norman_single_to_combo      risk_missing_single error_single_additive_rmse      125       NaN       25          0.053455          0.994799
Norman_single_to_combo          risk_combo_size error_single_additive_rmse      125       NaN       25          0.053455          0.994799

## 3. Papalexi RNA-protein 一致性

        dataset_name  n_aligned_perturbations  spearman_rna_l2_vs_protein_l2  spearman_rna_l2_vs_protein_abs_mean  rna_tasks  protein_tasks
Papalexi_RNA_protein                       94                       0.501297                             0.537174         94             94

## 4. Gasperini 调控标签稀疏审计

                       field  n_unique                                                                                                                                                                                                                                   top10
                perturbation     16532                                                                                                     nan:4473 | control:369 | DCTPP1:38 | TOP1:37 | ATPIF1:37 | HES6:33 | ZNF593:33 | chr1:161359440-161359463:32 | FAM83A:31 | EWSR1:31
                        gene     16837                                                                     nan:4473 | DCTPP1_TSS:37 | ATPIF1_TSS:37 | TOP1_TSS:36 | ZNF593_TSS:33 | HES6_TSS:32 | TMEM165_TSS:31 | chr1:161359440-161359463:31 | FAM96A_TSS:31 | FAM83A_TSS:31
                      nperts        30                                                                                                                                                       1:24341 | 2:8992 | 3:3827 | 4:1662 | 5:837 | 6:460 | 0:369 | 7:284 | 8:159 | 9:97
                      sample         6 K1000_CRISPRi_cells_r1_SI-GA-F1:7230 | K1000_CRISPRi_cells_r4_SI-GA-F4:6981 | K1000_CRISPRi_cells_r3_SI-GA-F3:6868 | K1000_CRISPRi_cells_r5_SI-GA-F5:6848 | K1000_CRISPRi_cells_r2_SI-GA-F2:6737 | K1000_CRISPRi_cells_r6_SI-GA-F6:6620
            sample_directory         6 K1000_CRISPRi_cells_r1_SI-GA-F1:7230 | K1000_CRISPRi_cells_r4_SI-GA-F4:6981 | K1000_CRISPRi_cells_r3_SI-GA-F3:6868 | K1000_CRISPRi_cells_r5_SI-GA-F5:6848 | K1000_CRISPRi_cells_r2_SI-GA-F2:6737 | K1000_CRISPRi_cells_r6_SI-GA-F6:6620
perturbation_support_profile     16532                                                                                                                                                                         low_support_lt15=15969; coordinate_like=15225; control_like=214