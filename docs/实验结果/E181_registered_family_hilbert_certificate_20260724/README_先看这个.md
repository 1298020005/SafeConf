# E181｜先看这个

E181 是 E176、E177、E180 已冻结预测的跨研究方法整合，不是新的前瞻性实验。

主结果：scGPT 五个种子与 GEARS 五个种子组成的注册家族，在 2,393 个任务上出现 0 次家族 RMS 下界违反和 0 次最坏成员下界违反。原 conformal 参考质心上界通过可计算的质心距离搬移，得到严格的双侧误差证书。

入口：

- 完整说明：`reports/E181_REPORT.md`
- 方法图：`figures/F1_E181_METHOD.svg`
- 数据集汇总：`tables/E181_DATASET_SUMMARY.csv`
- 逐任务证书：`tables/E181_TASK_CERTIFICATES.csv`
- 原结果复算审计：`tables/E181_SOURCE_REPRODUCTION_AUDIT.csv`
- 输出完整性：`MANIFEST.sha256`

边界：E181 使用的是已经打开的评估真值，作用是理论整合、索引复核与定量审计。SafeConf 排序没有因本实验恢复为主张；E180 中失败的学习型上界也没有进入正式证书。
