# 复现入口

工作树：`/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921`。解释器：`/home/miniconda/bin/python`。

仅重算保存分数、重新生成表图及哈希（不训练、不下载、不再次读取原始确认数据）：

```bash
cd /home/yyf/runtime_worktrees/e220_reviewer_closure_20260921
/home/miniconda/bin/python tools/scripts/build_safeconf_evidence_freeze_v21.py --output /home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/研究推进/20261009_投稿证据_v21/evidence_freeze_v1
```

原单路线入口（用于独立环境复现整个登记协议，不用于本机再开新确认）：

```bash
/home/miniconda/bin/python tools/scripts/run_safeconf_kolf_panel_v21.py --phase complete
```

原输入与保存分数路径见`ARTIFACT_MANIFEST.json`；所有部署排序见`KOLF_ADOPTED_SYSTEM_RANKING.parquet`。
图上均值为实际运行平均，不是一个可部署的平均排序。实际采用系统固定为PublicRule。
环境：Python3.12.4 / NumPy2.4.4 / pandas2.3.3 / matplotlib3.10.8。
