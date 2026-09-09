# 决策记录

## 2026-09-09

### D7. 业务 A 与业务 B 分开裁决

决定：

> E201 只支持预测后风险关联；E204 单独检验训练加权。E204 即使成功，也不能自动证明 SafeConf 相对幅度的复核排序增益。

### D8. 组合策略尚未成立

决定：

> `partial Spearman=0.2503` 只表示控制幅度后的剩余关联。任何“幅度 + SafeConf”组合必须先限制候选、在 E201 内开发并冻结，再到未参与开发的数据一次性评价；在此之前不写成可用策略。

### D9. E204 完成定义

决定：

> 32 个训练结束不是实验完成。完整链条是：训练验收 → 盲态预测 → 输入与顺序审计 → checkpoint/预测/评分/哈希封存 → GitHub/Gitee 留痕 → 解盲 → 同 target 同 seed 配对分析 → 全部失败与边界报告。

### D10. Kimi-K3 材料的地位

决定：

> `agents/kimi-k3` 保留角色意见与产物索引；相同审核报告只在学习包保留一份。其数字核验可参考，Git 旧快照、完成度百分比和未验证的组合/选刊判断不进入事实层。

### D11. E201 证书门不改判

决定：

> 预设阈值下继续报告 FAIL，不事后放宽。可补做 float64 诊断以解释数量级，但诊断结果只能作为敏感性说明，不能覆盖原门判定。

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
