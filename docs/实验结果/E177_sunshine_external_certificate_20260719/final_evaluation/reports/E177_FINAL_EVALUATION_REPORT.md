# E177 final evaluation report

E177 is the independent public processed-data check after E176. The run used 144 frozen targets and eight technical groups. Metadata freeze, F2 assets, pretruth gate and calibration were all committed before final evaluation truth was opened.

## Main result

- Evaluation targets/tasks: 50/400.
- Ensemble target-cluster coverage: 44/50 = 88.0%; exact 95% CI 75.69%–95.47%.
- Pair-mean target-cluster coverage: 44/50 = 88.0%; exact 95% CI 75.69%–95.47%.
- Task-level coverage: 98.25%.
- Pair-distance lower-bound violations: mean=0, max=0.

## Interpretation

The strongest transferable part is still the deterministic pair-distance certificate: `RMSE(p1,p2)/2` did not violate either pair-mean or pair-max RMSE on the 400 hidden evaluation tasks.

The calibrated upper bound is more mixed. The target-level point estimate is 88.0%, slightly below the nominal 90% target, while the exact confidence interval still includes 90%. This should be reported as a boundary result, not as a clean external pass.

Ranking diagnostics are weak. SafeConf risk has Spearman about 0.057 with ensemble RMSE and 0.059 with pair-mean RMSE. For this external dataset, the result supports the certificate/coverage protocol more than the idea of a strong practical risk-ranking router.

`gem_group` is only a technical-repeat label here. It must not be described as donor, patient or biological-context replication.
