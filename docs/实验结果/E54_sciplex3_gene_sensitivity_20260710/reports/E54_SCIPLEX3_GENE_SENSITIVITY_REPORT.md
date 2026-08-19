# E54 sciplex3 基因数敏感性

- 生成时间：2026-07-10T17:33:17
- Git：`25ac9cd66eb4`
- 工作区 dirty：`True`

## 1. 稳定性汇总

 n_genes_setting          risk_score_name  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
            1000        risk_disagreement      104  0.835757       21          0.222473          1.721209
            3000        risk_disagreement      104  0.848997       21          0.209768          1.723914
            5000        risk_disagreement      104  0.847930       21          0.176010          1.739155
            1000 risk_predicted_magnitude      104  0.935458       21          0.220765          1.707994
            3000 risk_predicted_magnitude      104  0.945481       21          0.207984          1.709253
            5000 risk_predicted_magnitude      104  0.945625       21          0.176010          1.739155
            1000      risk_safeconf_smoke      104  0.895641       21          0.212894          1.647097
            3000      risk_safeconf_smoke      104  0.915454       21          0.201591          1.656719
            5000      risk_safeconf_smoke      104  0.905755       21          0.167458          1.654658

## 2. Top summary

               dataset_name       setting          risk_score_name          target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment  n_genes_setting
sciplex3_cell_line_gene5000 leave_context risk_predicted_magnitude error_contextsim_rmse      104  0.945849       21          0.175799          1.739671             5000
sciplex3_cell_line_gene3000 leave_context risk_predicted_magnitude error_contextsim_rmse      104  0.945705       21          0.208052          1.708726             3000
sciplex3_cell_line_gene3000 leave_context risk_predicted_magnitude         error_v0_rmse      104  0.945705       21          0.207958          1.709761             3000
sciplex3_cell_line_gene5000 leave_context risk_predicted_magnitude       error_mean_rmse      104  0.945625       21          0.176010          1.739155             5000
sciplex3_cell_line_gene3000 leave_context risk_predicted_magnitude       error_mean_rmse      104  0.945481       21          0.207984          1.709253             3000
sciplex3_cell_line_gene5000 leave_context risk_predicted_magnitude         error_v0_rmse      104  0.945412       21          0.176303          1.738719             5000
sciplex3_cell_line_gene1000 leave_context risk_predicted_magnitude         error_v0_rmse      104  0.935842       21          0.221258          1.706680             1000
sciplex3_cell_line_gene1000 leave_context risk_predicted_magnitude       error_mean_rmse      104  0.935458       21          0.220765          1.707994             1000
sciplex3_cell_line_gene1000 leave_context risk_predicted_magnitude error_contextsim_rmse      104  0.935245       21          0.220425          1.709444             1000
sciplex3_cell_line_gene3000 leave_context      risk_safeconf_smoke error_contextsim_rmse      104  0.915731       21          0.201658          1.656205             3000
sciplex3_cell_line_gene3000 leave_context      risk_safeconf_smoke         error_v0_rmse      104  0.915475       21          0.201570          1.657243             3000
sciplex3_cell_line_gene3000 leave_context      risk_safeconf_smoke       error_mean_rmse      104  0.915454       21          0.201591          1.656719             3000
sciplex3_cell_line_gene5000 leave_context      risk_safeconf_smoke error_contextsim_rmse      104  0.906214       21          0.167269          1.655255             5000
sciplex3_cell_line_gene5000 leave_context      risk_safeconf_smoke       error_mean_rmse      104  0.905755       21          0.167458          1.654658             5000
sciplex3_cell_line_gene5000 leave_context      risk_safeconf_smoke         error_v0_rmse      104  0.905275       21          0.167714          1.654010             5000
sciplex3_cell_line_gene1000 leave_context      risk_safeconf_smoke error_contextsim_rmse      104  0.896057       21          0.212585          1.648640             1000
sciplex3_cell_line_gene1000 leave_context      risk_safeconf_smoke       error_mean_rmse      104  0.895641       21          0.212894          1.647097             1000
sciplex3_cell_line_gene1000 leave_context      risk_safeconf_smoke         error_v0_rmse      104  0.895342       21          0.213337          1.645586             1000
sciplex3_cell_line_gene3000 leave_context        risk_disagreement         error_v0_rmse      104  0.848997       21          0.209750          1.724494             3000
sciplex3_cell_line_gene3000 leave_context        risk_disagreement       error_mean_rmse      104  0.848997       21          0.209768          1.723914             3000

## 3. 构建状态

                    dataset                                                         path   n_obs  n_vars  n_genes_used context_col perturbation_col  n_tasks  n_contexts  n_perturbations status       setting heldout                dataset_name  n_train  n_test  n_genes_setting
sciplex3_cell_line_gene1000 extra_official/cellular_context_generalization/sciplex3.h5ad 26046.0  5000.0        1000.0   cell_line     perturbation    104.0         3.0             36.0     ok           NaN     NaN                         NaN      NaN     NaN             1000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    A549 sciplex3_cell_line_gene1000     68.0    36.0             1000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    K562 sciplex3_cell_line_gene1000     69.0    35.0             1000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    MCF7 sciplex3_cell_line_gene1000     71.0    33.0             1000
sciplex3_cell_line_gene3000 extra_official/cellular_context_generalization/sciplex3.h5ad 26046.0  5000.0        3000.0   cell_line     perturbation    104.0         3.0             36.0     ok           NaN     NaN                         NaN      NaN     NaN             3000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    A549 sciplex3_cell_line_gene3000     68.0    36.0             3000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    K562 sciplex3_cell_line_gene3000     69.0    35.0             3000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    MCF7 sciplex3_cell_line_gene3000     71.0    33.0             3000
sciplex3_cell_line_gene5000 extra_official/cellular_context_generalization/sciplex3.h5ad 26046.0  5000.0        5000.0   cell_line     perturbation    104.0         3.0             36.0     ok           NaN     NaN                         NaN      NaN     NaN             5000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    A549 sciplex3_cell_line_gene5000     68.0    36.0             5000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    K562 sciplex3_cell_line_gene5000     69.0    35.0             5000
                        NaN                                                          NaN     NaN     NaN           NaN         NaN              NaN      NaN         NaN              NaN     ok leave_context    MCF7 sciplex3_cell_line_gene5000     71.0    33.0             5000