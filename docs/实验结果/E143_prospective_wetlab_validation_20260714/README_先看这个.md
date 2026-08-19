# E143 前瞻湿实验验证包

这不是已经完成的湿实验。服务器端已完成候选规则、功效、盲法、质控、样本布局、分析门槛和交接模板；物理实验还需要新细胞背景、CRISPRi 条件、平台、预算和实验负责人。

1. 先读 `reports/E143_DECISION_AND_HANDOFF.md`。
2. 与实验室确认条件后填 `templates/FORMAL_CANDIDATE_INPUT.csv`。
3. 正式候选必须在任何扰动后表达读出产生前运行 `python tools/scripts/freeze_e143_formal_wetlab_candidates.py --input <填好的CSV>` 冻结和哈希；风险映射会写入仓库外的私有目录。
4. `tables/E143_NADIG_TECHNICAL_PILOT.csv` 只能调流程，不能作为新增独立验证。
