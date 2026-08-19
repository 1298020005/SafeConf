# SafeConf supplement v0.2 小结

更新时间：2026-06-05

## 一句话

这轮不是新主表，只是把服务器已有的补充数据集拿来试跑冻结的 `protocol_v0_2`。结果是：5 个候选里 4 个跑通，1 个无法构造任务。

## 跑了哪些

| 数据集 | 中文说明 | 状态 | 备注 |
| --- | --- | --- | --- |
| XieHon2017 | 基因扰动数据集 | 跑通 | simple combined rho = 0.424，RC@80% = 16.85% |
| sciplex3_small | 小型化学扰动数据集 | 跑通 | simple combined rho = 0.515，RC@80% = 12.48% |
| SrivatsanTrapnell2020_sciplex4 | sci-Plex 化学扰动数据集 | 跑通 | simple combined rho = 0.734，但 test 只有 28 条，适合补充不适合主结论 |
| ShifrutMarson2018 | T cell 基因扰动数据集 | 跑通 | simple combined rho = 0.122，信号弱 |
| LaraAstiasoHuntly2023_leukemia | 白血病基因扰动数据集 | 未跑通 | 当前 `celltype + guide_id` 组合构造出 0 个 task，不硬凑 |

## 关键表

- `tables/SUPPLEMENT_RUN_STATUS.csv`：每个数据集跑没跑通。
- `tables/SUPPLEMENT_MAIN_PER_DATASET_SUMMARY.csv`：4 个跑通数据集的主要 rho 和 RC@80%。
- `tables/LaraAstiasoHuntly2023_leukemia_TASK_SUMMARY.csv`：说明 Lara leukemia 为什么失败。
- `reports/LaraAstiasoHuntly2023_leukemia.log`：失败日志。

## 怎么理解

这轮补充结果对论文是加分，但不能替代 7 主表。

能说：

- 补充数据里又有 3 个数据集显示正信号，尤其 `XieHon2017` 和 `sciplex3_small` 比较有用。
- `SrivatsanTrapnell2020_sciplex4` 数字很强，但样本太少，只能当补充。
- `ShifrutMarson2018` 信号弱，说明这个任务不是所有数据都能跑出好看数字。

不能说：

- 不能把 supplement 和主表混成一个大 pooled 结论。
- 不能把 Lara leukemia 写成失败实验，它目前只是数据结构不适合当前 `context × perturbation` 任务。
- 不能因为 sciplex4 rho 高就说化学线完全稳定，McFarland 主表仍然是明确失败边界。

## 下一步

1. 主线优先继续：Tahoe pseudobulk adapter（超大外部验证候选）。
2. supplement 可以继续扩，但要先做 eligibility，不要盲跑。
3. 如果 Claude/Qwen 复核，重点问：这些补充数据是否适合放 supplement，还是只放审计表。
