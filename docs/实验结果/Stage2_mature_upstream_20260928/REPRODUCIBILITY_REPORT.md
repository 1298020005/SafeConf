# SafeConf v4 复现报告

## 运行合同

- 开发：4 contexts × 5 gene-disjoint folds；outer task 从未进入本折风险模型训练。
- V2：inner OOF 生成 rP/rPQ、outer-train 内尺度校准和单调 gate。
- 主指标：`ceil(0.2*n)`、risk 降序、task_id 字典序处理 ties。
- Bootstrap：5,000 次，perturbation gene cluster 为重采样单位。
- SEALED：Final Candidate 冻结前未读取结果、字段覆盖或 truth；E216 后验识别为此前已揭盲并降级为 SEEN。
- PertEMA：官方软件 commit `43c09a32e23d0ee2ae5dfbab21b2deeab27f1803`，官方 XGB 超参数，同 outer split/label budget。

## 关键文件哈希

| 文件 | SHA-256 |
| --- | --- |
| docs/实验结果/Stage2_mature_upstream_20260928/FINAL_METHOD_CONFIG.json | 4dfdcea23d126f497a7b30d875187976bbef7cc63a1986d0f0fdf5371b0c48e1 |
| docs/实验结果/Stage2_mature_upstream_20260928/FINAL_CANDIDATE_FREEZE.json | 17349418afaf1366b6af78e247fc9ef3558c9deddc172b2283b31dba3e400c0e |
| docs/实验结果/Stage2_mature_upstream_20260928/FIELD_AVAILABILITY_MATRIX.csv | e04be0798877d4e387158fa8b42afbb2898bb0eaa9a61b99921560f4fb8a6087 |
| docs/实验结果/Stage2_mature_upstream_20260928/HISTORY_SOURCE_AUDIT.csv | 659f85e9587e9adb517293ec7c71abb8b64e8b2598be8a6c3b5015896d4eedc2 |
| docs/实验结果/Stage2_mature_upstream_20260928/UPSTREAM_COMPETENCE_V4.csv | fd7b920e9fdba16656f4490e77b84b24d671ee16ea22d3e296b18b8a95638e4c |
| docs/实验结果/Stage2_mature_upstream_20260928/EXPERIMENT_MASTER_TABLE.csv | 827ef08a015f3f59d739a862b8050ad1697e4148314519043d4807493038821a |
| docs/实验结果/Stage2_mature_upstream_20260928/pertema_fair_comparison/RUN_STATUS.json | 87c5fb27166d027108f1f301fe0a3033486756d8070950547a7886b2199e2cf9 |
| docs/实验结果/Stage2_mature_upstream_20260928/pertema_fair_comparison/SUMMARY.csv | 69d6b9b17f5d74903d3f1eefd8752f1d3a61be4b400a83523a90e6796e1cadca |

## 复跑命令

```bash
python tools/scripts/build_safeconf_governance_artifacts.py
python tools/scripts/run_safeconf_v4_development.py --upstream e201
python tools/scripts/run_safeconf_v4_development.py --upstream e205
python tools/scripts/audit_safeconf_competence_gate.py
/tmp/pertema-venv/bin/python tools/scripts/run_safeconf_pertema_fair_comparison.py
python tools/scripts/freeze_safeconf_final_candidate.py
python tools/scripts/build_safeconf_v4_paper_package.py
```

结果脚本默认拒绝覆盖已经存在的 RUN_STATUS，正式复跑应使用新输出目录或先保留原证据快照。
