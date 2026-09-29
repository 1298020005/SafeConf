# SafeConf v4 复现报告

## 运行合同

- 开发：4 contexts × 5 gene-disjoint folds；outer task 从未进入本折风险模型训练。
- V2：inner OOF 生成 rP/rPQ、outer-train 内尺度校准和单调 gate。
- 主指标：`ceil(0.2*n)`、risk 降序、task_id 字典序处理 ties。
- Bootstrap：5,000 次，perturbation gene cluster 为重采样单位。
- SEALED：Final Candidate 冻结前未读取 E170 test 结果、字段覆盖或 truth；四面板在授权提交远端后一次性打开。E216 后验识别为此前已揭盲并降级为 SEEN。
- PertEMA：官方软件 commit `43c09a32e23d0ee2ae5dfbab21b2deeab27f1803`，官方 XGB 超参数，同 outer split/label budget。

## 关键文件哈希

| 文件 | SHA-256 |
| --- | --- |
| docs/实验结果/Stage2_mature_upstream_20260928/FINAL_METHOD_CONFIG.json | 4dfdcea23d126f497a7b30d875187976bbef7cc63a1986d0f0fdf5371b0c48e1 |
| docs/实验结果/Stage2_mature_upstream_20260928/FINAL_CANDIDATE_FREEZE.json | 17349418afaf1366b6af78e247fc9ef3558c9deddc172b2283b31dba3e400c0e |
| docs/实验结果/Stage2_mature_upstream_20260928/FIELD_AVAILABILITY_MATRIX.csv | 71330392743d6dee8ca4bd424a80159fbea195f2c578d39927b8b7d6949ff578 |
| docs/实验结果/Stage2_mature_upstream_20260928/HISTORY_SOURCE_AUDIT.csv | 30dd67f2a38df5c9e047694509e0ac5d29c7d7c2ee6e6dbb6be27e476b8bd5a3 |
| docs/实验结果/Stage2_mature_upstream_20260928/UPSTREAM_COMPETENCE_V4.csv | 45b14ed93f8787d4216f73c7b0778ab813a8c3c2e5dcacfeb582b697c9cfe7e1 |
| docs/实验结果/Stage2_mature_upstream_20260928/EXPERIMENT_MASTER_TABLE.csv | 825445180b52f2f02bdbf7cca4547960f0ca3290d53ee773b8ccc1b2e54b0947 |
| docs/实验结果/Stage2_mature_upstream_20260928/pertema_fair_comparison/RUN_STATUS.json | 87c5fb27166d027108f1f301fe0a3033486756d8070950547a7886b2199e2cf9 |
| docs/实验结果/Stage2_mature_upstream_20260928/pertema_fair_comparison/SUMMARY.csv | 69d6b9b17f5d74903d3f1eefd8752f1d3a61be4b400a83523a90e6796e1cadca |
| docs/实验结果/Stage2_mature_upstream_20260928/safeconf_v4_development/e190_gears_crossfamily/RUN_STATUS.json | eeb5e87182dcebb9eeaa287601b2bdfcac07727f80ee682c5eea10ea58f0dbc4 |
| docs/实验结果/Stage2_mature_upstream_20260928/safeconf_v4_development/e190_gears_crossfamily/SUMMARY.csv | 5ba637966857d6fb5fb7acbcb4bf8f6fc3d3906c24acd950b1e796a4a88410de |
| docs/实验结果/Stage2_mature_upstream_20260928/E170_V4_CONFIRMATION_AUTHORIZATION.json | a31a4b79a3b1c8c7e9e1da7ce902c9e19a283149fe7539e9d8c3130455533755 |
| docs/实验结果/Stage2_mature_upstream_20260928/confirmation/e170_primary_cd4_four_panel/RUN_STATUS.json | 70359a27e3e3f22f5df40c0707b0a86434da069d20e20a507cba7086799f298c |
| docs/实验结果/Stage2_mature_upstream_20260928/confirmation/e170_primary_cd4_four_panel/SUMMARY.csv | 877342a86270c02a4e194acc8d93af06952eb735970bcd40a97b24a22a6f9f5a |
| docs/实验结果/Stage2_mature_upstream_20260928/confirmation/e170_primary_cd4_four_panel/GATE_A_RESULT.json | c743d944407e7b6f9ba8d32cee8ee5247d1d2d4388970d1302e6ea81e2eb461b |
| docs/实验结果/Stage2_mature_upstream_20260928/confirmation/e170_primary_cd4_four_panel/PAIRED_CLUSTER_BOOTSTRAP.csv | 9b689ee1f7b8eb8ded4b57f3873e8db1a2a0c03e5eed4e63c0524611006035e4 |
| docs/实验结果/Stage2_mature_upstream_20260928/error_memory_budget/RUN_STATUS.json | 33d684b07406ca5eb7aac7f9b28202d253cfc20260250476709319ab30fb83d5 |
| docs/实验结果/Stage2_mature_upstream_20260928/error_memory_budget/SUMMARY.csv | 60a5e8c0dbe0a152be7285c82477ab76c2ae09f7d9ea0134f9767b99f963248e |
| docs/实验结果/Stage2_mature_upstream_20260928/cpa_risk_batch_v2/RUN_STATUS.json | 5d203f90898807dae0760035d60225c14985e5705dd5ed0e36ccd1725ef1e6de |
| docs/实验结果/Stage2_mature_upstream_20260928/cpa_risk_batch_v2/SUMMARY.csv | 6befd8a5e9e8d649b4fb8cf9664b0e12f14f7cd9d7dc19dd5982a5905fad1757 |
| docs/实验结果/Stage2_mature_upstream_20260928/cpa_risk_batch_v2/INCREMENTAL_RESULTS.csv | d61d9ce917c6ac26c9f553dc9845ae8364b5dda4e7ae26b6a81bbbe13d6606a6 |

## 复跑命令

```bash
python tools/scripts/build_safeconf_governance_artifacts.py
python tools/scripts/run_safeconf_v4_development.py --upstream e201
python tools/scripts/run_safeconf_v4_development.py --upstream e205
python tools/scripts/audit_safeconf_competence_gate.py
python tools/scripts/run_e190_gears_safeconf_v4.py
/tmp/pertema-venv/bin/python tools/scripts/run_safeconf_pertema_fair_comparison.py
python tools/scripts/freeze_safeconf_final_candidate.py
python tools/scripts/register_e170_v4_confirmation.py
python tools/scripts/build_e170_v4_confirmation_truth.py
python tools/scripts/run_e170_safeconf_v4_confirmation.py
python tools/scripts/build_safeconf_v4_paper_package.py
```

结果脚本默认拒绝覆盖已经存在的 RUN_STATUS，正式复跑应使用新输出目录或先保留原证据快照。
