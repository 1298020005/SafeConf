# 决策记录

## 2026-07-14

### D3. 外部 Agent 唯一根入口

决定：

> `START_HERE_FOR_AGENTS.md` 是所有外部 Agent 的唯一根入口；详细学习材料统一放入 `docs/学习导航/`。

### D4. 当前事实入口

决定：

> 发生冲突时，以 `docs/实验结果/GATE_STATUS_20260714.md` 为最高事实入口，E131/E132 为当前主证据。

### D5. 论文主线升级

决定：

> 新论文以六套正式 scGPT–GEARS 结果为主线。2026-06-16 的七数据集/V0-ContextSim Methods/Results 草稿只作素材，不再视作当前正文。

### D6. 停止事后路由器调参

决定：

> E126/E130 未通过后，不再在已解封的六数据集上更换风险模型。新方法必须事前冻结，并用全新数据确认。

## 2026-06-02

### D1. 多 agent 交流入口

决定：

> 新建 `proj/agents/` 作为 Cursor、Qoder、Codex 的共享入口。

理由：

- `docs/` 应该放定稿设计，不适合放来回争论。
- `discuss/` 太重、太长，适合作归档，不适合作当前入口。
- `agents/` 短、好找、适合三方协作。

### D2. INDEX 放置

决定：

> `/home/yyf/INDEX.md` 作为全局入口；`/home/yyf/proj/INDEX.md` 作为项目入口。

用法：

- 新 AI 不懂服务器结构：先读 `/home/yyf/INDEX.md`。
- 已经进入项目：先读 `/home/yyf/proj/INDEX.md`。
- 要参与讨论：读 `/home/yyf/proj/agents/README.md`。

## 2026-07-12

### D-Grok-01. 周四后主线以周老师三 setting 为准

决定：

> 发文门槛先按周老师原话：小矩阵 + 整行/整列 + 跨数据集。E74 pair-risk 可作为方法升级，但不能替代三 setting 收口。课程论文不进科研主线。

证据：

- `agents/grok/2026-07-12_周老师后_Codex主线判断.md`
- 聊天原文与 `workspace/group_meeting_20260709_MAINLINE_WHITE/`

## 2026-07-12

### D-Grok-02. Codex 互审以 REVIEW_PACK 为法

决定：

> Codex/其他 AI 互审以 `agents/CODEX_ADVERSARIAL_REVIEW_PACK_20260712.md` 为准。
> 优先级：周老师 Z7–Z9 > panel 复现；稳定二区=Gate Q2-A–D；一区另议。
> 主 claim 二选一写清，禁止混 L1/L2/L3。

