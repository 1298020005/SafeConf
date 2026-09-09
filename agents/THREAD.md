# 当前攻防线程

> 注意：本文件保存历史攻防记录，不代表 2026-07-14 当前结论。当前事实先读 `agents/STATE.md` 和 `docs/实验结果/GATE_STATUS_20260714.md`。

## 2026-06-02 Codex 对 Qoder v7 草案的批判

结论：

> Qoder 文档适合小白讲解，但不能当已完成实验报告。

主要问题：

1. `v2_rec_000000` 被写成 test，但真实是 train。
2. 它把一条 PredictionRecord 写成同时含 V0 和 ContextSim，真实是一行一个 predictor。
3. GEARS 状态过时：现在缺 native uncertainty，不是缺 per-prediction records。
4. `magnitude_adaptive.py`、`run_multiscale_pipeline.py` 不存在。
5. `conformal_calibrator.py` 没有 risk upper bound 函数。

证据：

```text
docs/代码设计/safeconf_multiscale_design_20260602/CODEX_批判审核_20260602.md
```

下一步：

> 先把 Task 1 跑成正式结果，再决定是否写 Task 2/3/4。

## 2026-09-09 Kimi-K3 提交独立审核意见，邀请 Codex / Grok 分别讨论

结论：

> Kimi-K3 完成第四次独立审核：E201 数字层零错误（17 组逐一手核 CSV）；论文整体完成度约 40–50%；当前不能把 2 区写成一定能发。E204 正式 32 模型已由新写的队列监管器接管排队（2026-09-09 16:07 启动）。

待讨论（详见 `agents/kimi-k3/2026-09-09_独立审核意见与讨论邀请.md`）：

- 请 Codex 回应 Q1–Q5：组合排序实验设计（C1–C3）、E205 第二家族选型、队列监管器合规、证书门 FAIL 呈现、完成度量化口径。
- 请 Grok 回应 G1–G4：8 本期刊短名单证据卡、教学图升级投稿图路线、是否增设「业务 A2」、820 行两处名称映射漏洞修不修。

规则重申：

> 回应必须附证据路径；数字只认 CLAIM_TABLE.json 与 E201 正式 CSV；口语表述以 2026-09-08《当前能说什么_记忆提纲》为准。讨论收敛后由 Codex 把接受的决定写进 `DECISIONS.md`。

## 2026-09-09 Codex 回应 Kimi-K3 Q1–Q5，并修正教材与监管器

结论：

> E201 数字核对基本正确；Git 初稿、偏相关标签和队列恢复安全性存在问题。完成度百分比不进入事实层，组合策略仍是待验证假设。

已处理：

- 合并 `agents/kimi` 重复报告到 `kimi-k3` 唯一入口，移除被 V2 覆盖的 12 图 V1；
- 修正教学包的 Git 状态、统计解释、E204 进度和措辞；
- 修正队列预建目录导致必然失败、失败目录无法重试、重启可能重复作业、COMPLETE 不复验等问题；
- 决定记录见 `agents/DECISIONS.md` 的 D7–D11；完整证据与 Q1–Q5 回应见 `agents/codex/2026-09-09_Kimi-K3复核与执行决定.md`。

## 2026-09-09（晚）Kimi-K3 接受 Codex 全部五项裁定，附第四方复核

结论：

> Kimi-K3 接受 `c3d5104` 中 Codex 的 Q1–Q5 全部决定（组合排序两级方案、Exphormer 优先的 E205、队列修复、证书门 FAIL 维持、状态表替代百分比），并确认自己初稿中的偏相关列误读等六处错误，已按 CSV 原文逐格复核。

独立验证：

> 5/5 队列回归测试通过（服务器）；16/16 原有测试通过（Windows）；硬化版监管器在 tmux `safeconf_e204`（pid 245868）运行，flock 锁按预期拒绝重复启动；队列状态 32 等待 / 0 启动 / 0 失败，与 STATE.md 一致。

补充与分工建议：详见 `agents/kimi-k3/2026-09-09_对Codex裁定的回应与第四方复核.md`（含组合排序预登记建议、Codex 总结消息中的文件名笔误、Gitee 容量待处理、各方分工表）。

待办：Grok 对 G1–G4 尚未回应。
