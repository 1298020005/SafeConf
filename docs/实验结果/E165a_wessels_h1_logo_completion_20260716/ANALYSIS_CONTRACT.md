# E165a｜Wessels H1 leave-one-component-gene-out 审计补充

冻结日期：2026-07-16。E165a 在 E165 主 release 已公开之后执行，目的只有一个：补齐 E164 解封前已写明、但 E165 主发布表中遗漏的 H1 paired-effect LOGO 记录。它不是新假设，不重新打开 Wessels raw H5AD，不读取单细胞表达，不修改 H1/H2/P1、bootstrap、task集合或任何通过门。

## 固定输入

只允许读取当前 Git HEAD 已提交的以下 E165 文件，并逐字节核对 SHA256：

- `release/RUN_STATUS.json`：`51ae8e62f9930c941138a2cfd6c099c24bb19b2a992aecdbf09a9f093c5646a7`；
- `release/RESULTS_SHA256.csv`：`e86b7a48aee7adbc508f4392592247fdad33cf30730f801c819e05c50fca443e`；
- `E165_PREDICTOR_TASK_METRICS.csv.gz`：`c37467db93d5f69ea33851ace13fc4c9c63747976bb8763f3e741813ea889ccd`；
- `E165_HYPOTHESIS_TESTS.csv`：`df5e631a2c965aa535defef60eed25aa10bde6e1db58b7132a0d3c5b6ce9d0b4`；
- `E165_TEST_TRUTH_TASKS.csv`：`a761c433814a5ffd5d464680c6f789743f0ecdccb289a2260c2d7189a618d0ba`。

E165 phase 必须为 `complete_one_time_test_truth_evaluation`。runner与本合同在 formal 时必须已提交且与 HEAD blob一致。

## 固定计算

对48个任务固定定义：

`delta_i = PCA10_RMSE_cell_weighted_perturbed_mean,i - PCA10_RMSE_matching_single_mean,i`。

完整均值必须逐位复核 E165 H1 point estimate。对每个test component gene，删除所有包含该gene的任务，在剩余任务上重新计算 `mean(delta_i)`；保存removed/remaining task数、均值、最小值、中位数、最大值和正值比例。正值仍表示matching更好。

该表只评价H1对共享component gene的敏感性。它不重新计算置信区间，不改变E165 H1 `passed=true`，也不能被称为新的独立验证。

## 访问与发布

- raw Wessels opened：false；
- expression rows/columns indexed：0；
- test truth profile重新读取：false；只读取已发布的task-level metrics；
- 输出先写 `.release.staging`，经allowlist、SHA256、无symlink和fsync检查后原子rename为`release/`；已有release拒绝覆盖。

固定命令：

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e165a_wessels_h1_logo_completion.py --mode preflight

/home/yyf/.conda/envs/prescribe_env/bin/python \
  tools/scripts/run_e165a_wessels_h1_logo_completion.py --mode formal
```
