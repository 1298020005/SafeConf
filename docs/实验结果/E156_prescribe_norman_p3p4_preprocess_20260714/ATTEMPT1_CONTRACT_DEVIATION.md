# E156 早期尝试与正式重跑记录

## 第一次尝试：时间顺序偏差，已中止

2026-07-14 的第一次实现先用 train-only 基因轴和 PCA 对 P3/P4 测试 X 做了固定
transform，再把结果写入权限为 `000` 的文件。该尝试没有训练模型、生成预测、汇总测试效应或
计算任何终点；HVG、PCA、E-distance 等拟合参数也没有使用测试 X。

E155 合同规定测试表达只能在 checkpoint 和任务分数全部锁定后用于评价。提前做固定 transform
违反该时间顺序，因此第一次实现判为 aborted，不能作为 P3/P4 严格盲评链条的一部分。两个提前
生成文件的 SHA256 为：

- P3：`0e36e924026da12093b2deb89c6fbab1c71d58082509e4911fd3133bb4e6ab16`
- P4：`832a1f3e640ca424e2daa7a34d9c51e9e0e654d66dd02bd4877201ae802e4a7e`

完整快照位于
`/home/yyf/archive/research_attempts/safeconf/E156_attempt1_early_test_transform_20260714/`，
只用于审计，不供 E157/E158 使用。

## 第二次尝试：数值合格，执行脚本追溯不足

第二次运行只处理 train+val，数值与泄漏审计均通过；但运行结束后脚本文字又被修改，状态中也
没有记录执行脚本 SHA256，无法密码学证明仓库中的脚本正是当时运行版本。因此该次资产虽没有
科学或数据泄漏错误，仍不作为最终前瞻链条。快照位于
`/home/yyf/archive/research_attempts/safeconf/E156_attempt2_valid_assets_missing_runner_provenance_20260714/`。

## 正式 E156

最终 runner 先提交到 Git commit `b7270bafa7644912f99b398a4a4c69140faef6fb`，再于
2026-07-14 18:48–18:51 从 E155 冻结输入完整重跑。`RUN_STATUS.json` 记录了 runner SHA256，
并验证工作区脚本与该 Git HEAD blob 完全一致。原始 H5AD 用于整文件 SHA256 校验并以 backed
方式读取 `obs`；测试 X 行没有被索引、载入内存或执行变换。正式数据目录只包含 train+val，
E157/E158 只允许接受这次运行登记的新资产哈希。正式重跑与第二次尝试的 P3/P4 development
H5AD 已逐元素比较：obs/var 顺序、稀疏 X、PCA 坐标、PCA mean/components 以及
`y_n/y_d/y_s` 的最大差全部为 0，记录见 `tables/E156_RERUN_EQUIVALENCE_AUDIT.csv`。
