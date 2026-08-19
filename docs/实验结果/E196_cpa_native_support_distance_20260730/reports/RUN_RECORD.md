# E196｜运行记录

- started：`2026-07-30T15:04:19+08:00`
- finished：`2026-07-30T15:10:50+08:00`
- elapsed seconds：`384.589`
- command：`/home/yyf/.conda/envs/cpa_runtime_env/bin/python tools/scripts/run_e196_cpa_native_support_distance.py --mode full --device cpu`
- executable：`/home/yyf/.conda/envs/cpa_runtime_env/bin/python`
- Python：`3.9.25`
- CPA：`0.8.8`
- torch：`2.1.2+cu118`
- CPA source commit：`fbd7c0250edc23eff003a10c99655579c53afd63`
- CPA worktree：`M cpa/__init__.py`
- CPA runtime source：`__init__.py` 的 Ray 可选导入兼容补丁与
  `_api.py/_model.py/_module.py` 均由 `E196_CODE_LOCK.json` 锁定；核心模型与
  uncertainty 实现文件没有未记录漂移
- model retraining：`false`
- covariate embedding：按冻结 `covars_encoder` 类别编号直接读取同一
  `module.covars_embeddings`；规避 CPA 0.8.8 scalar wrapper 的三维 AnnData 兼容缺陷
- synthetic smoke：`PASS`
- pseudo-test reproduction tolerance：`1e-05`
- manifest/task-key descriptive resampling：`10000`
- output hash index：`tables/E196_INPUT_HASHES.csv`

冻结 CPA builder 会把来源 h5ad 载入进程，用于重建训练输入、pseudo-control
预测复现和 control mean；target perturbed expression 与 error 列不进入距离函数。
全部九个模型的距离阶段完成并写入带哈希的 pre-outcome 文件后，程序才解析并连接
既有 RMSE。运行失败时 `E196_STATUS.json` 写为 `FAILED`，不得沿用部分表得出结论。
