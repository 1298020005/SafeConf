## 2026-07-12 Grok：稳定录用版 Codex 粘贴指令

- 文件：`agents/PASTE_TO_CODEX_STABLE_ACCEPT_20260712.md`
- 复制范围：文中 `===BEGIN===` 至 `===END===`
- 定位：按周老师三 setting 做录用最小集合（M1–M8 / Gate0–8），二区稳定优先

## 2026-07-12 Grok：Codex 互审包（周老师+二区/一区）

- 主文件：`agents/CODEX_ADVERSARIAL_REVIEW_PACK_20260712.md`
- 含：idea 锁死、Z1–Z11 验收、Codex 对错想法、实验规格、期刊匹配、A1–A10 对抗题、KILL 规则、启动指令
- 目标：按周老师三 setting；稳定二区底线；一区仅 Gate Q2 后评估

## 2026-07-12 Grok：数据集与分区判断

- 存档：`agents/grok/2026-07-12_数据集与分区判断.md`
- 数据大体选对；当前不能稳定二区；不能发一区。

## 2026-07-12 Grok：周老师后 Codex 主线判断

- 已存档：`agents/grok/2026-07-12_周老师后_Codex主线判断.md`
- 结论：Codex 半对；老师三 setting 未收口；E74 科学更硬但 claim 侧移；不宜继续无脑 panel。
- 课程论文线忽略。

## 2026-06-07 Desktop-GPT Claude评审回应与最低投稿版调整

- 做了什么：完成 Run02 v0.3 router 审计、Run03 GEARS 三 seed smoke 与统一评估；根据 Claude 评审把 Run03 降级为 supplement probe，并定义最低投稿版。
- 详细记录：`/home/yyf/archive/safeconf/project_history/repository_archive_20260617/协作记录/桌面GPT/2026-06-07_二区实验推进/14_回应Claude评审_调整后的最低投稿路线.md`
- 给 Claude 复制：`/home/yyf/archive/safeconf/project_history/repository_archive_20260617/协作记录/桌面GPT/2026-06-07_二区实验推进/给Claude复制_最低投稿版与后续实验复核.md`
- 下一步：Claude 复核后，优先锁最低投稿版和论文骨架。

## 2026-06-14: Fold-safe E1-E4 + Claude 验收

- 发现 learned LOPO 跨 outer-fold task-label 泄漏，提交 `f2e2c9d` 修复。
- E1-E4 以 fold-safe 设计重跑，全部 gate pass，提交 `efa9555`。
- Claude 验收确认：泄漏修复完整，E2 是当前最强证据，E4 seed
  一致是当前 HistGBT 配置的确定性行为。
- E1 LODO×LOPO 补充分析：disagreement 从 LOPO 的 2/7 上升到
  LODO×LOPO 的 4/7，但不能称为唯一或最稳定信号。
- McFarland 继续作为 frozen v0.2 failure boundary；learned LOPO
  partial rho = 0.331，E2 isotonic residual partial rho = 0.226。
- Tahoe 已完成 100/1026 shards sampled smoke，不是“未启用”。
- 下一步由 Claude 主导科学判断，Codex 执行 scPerturBench aggregate-error
  identifier/granularity feasibility audit；对齐失败立即停止，不硬凑。

## 2026-06-14: scPerturBench E8b Phase 2 对齐审计

- 从官方固定 commit 下载 genetic 与 chemical aggregate CSV，共约 67MB，
  放在 `/home/yyf/data/scperturbench/`，未进入 Git。
- Frangieh：benchmark 74 个基因，全部能与 SafeConf 211 个 perturbations
  精确匹配；benchmark 缺 condition，只能做 perturbation-only 聚合关联。
- sciplex3：benchmark 为 3 cell lines × 75 drugs × 4 doses；SafeConf 为
  3 contexts × 188 drug-only tasks。需要显式 context 映射、drug alias 和
  dose 聚合，只能作为 sensitivity。
- 发现 Claude 原 Phase 3 的 combined score 是 confidence 方向，若与 error
  做正相关必须取负；chem_robust frozen 公式也不含 context。
- Phase 3 未启动，等待 Claude 冻结 metric、score、mapping 和 bootstrap gate。

## 2026-06-15: E8b 正式关联实验完成

- Claude 批准预注册后完成 Frangieh primary、nuisance controls、metric/DEG
  sensitivity 和 sciplex3 sensitivity。
- Frangieh MSE DEG=5000：median rho=0.584，B=1000 CI [0.393, 0.726]，
  14/15 methods 为正，预注册 gate PASS。
- shuffled-risk null median=0.007，经验单侧 p=1/201。
- 样本量基线更强（rho=0.764），因此增加明确标注为 post hoc 的控制分析：
  控制 log(Nstimulated) 后 median partial rho=0.335，
  CI [0.047, 0.538]，15/15 methods 为正。
- sensitivity 显示结论依赖指标与基因面板：小 DEG 的 MSE 信号弱，
  pearson_distance DEG=5000 反向。不能声称跨所有指标普遍成立。
- sciplex3 仅使用 60 exact+alias drugs；A549/K562/MCF7/pooled median rho
  为 0.458/0.469/0.425/0.384，只作 sensitivity。
- frozen v0.2 未修改，原始 scPerturBench CSV 未进入 Git。
