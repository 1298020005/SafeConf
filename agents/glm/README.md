# GLM 工作区（ZCode / GLM-5.3 主控）

建立：2026-08-17 凌晨（Asia/Shanghai）
身份：用户通过 ZCode 调用的 GLM 智能体，接替 GPT/Claude/Grok/Codex 的审核与执行工作。
用户本回合的授权（原话摘要，完整见 `03_THINKING_LOG.md`）：

1. 先按交接包（`~/.zcode/tmp/prompt-attachments/.../01-GLM-_SafeConf-_20260817.md`，
   服务器内副本见 `00_AUDIT_REPORT_20260817.md` 附录）第 8 节交付审核报告；
2. 审核之后，**一步步亲手完成实验**（尤其是 E201 评估链），不许盲跑、不许无人核验地批处理；
3. 最终按 Nature 格式、用现有内容产出中英文双份论文稿，图必须做好；
4. 所有输出写到本目录 `proj/agents/glm/`，详细记录想法与报告，**这部分是写给后续 AI 看的**；
5. 后续用户会继续提要求，遇到问题解决问题。

## 对"不能借助脚本"的解释（GLM 的理解，已写入思考日志）

E201 的六个评估脚本是 2026-08-02 预注册冻结协议的一部分
（`docs/实验结果/E201_txpert_multitarget_retraining_20260802/TARGET_RELEASE_AND_EVALUATION_FREEZE.md`），
替换它们会摧毁预注册本身。因此本工作区的执行纪律是：

- **脚本 = 冻结协议的仪器**。每一步运行前，人工核对前置条件；运行后，人工核验
  输出物（行数、哈希、状态 JSON、表格摘要），并记录在 `STATE.md`；
- 绝不为了"结果好看"改种子、改公式、改任务集、换指标；负结果原样保留；
- 任何一步核验失败 → 立即停链，写清失败原因，等待用户或下一步指令；
- 开 target 真值之前，封存物必须已经双远程（GitHub + Gitee）提交——这是协议的
  硬顺序，本工作区的自动化同样遵守。

## 硬规则（继承交接包第 0 节，全部继续有效）

1. 开真值前不得以任何形式读取 E201 target 扰动表达；
2. 旧稿 `submission/SafeConf_current/` 不是投稿正文；
3. 不写"首个单细胞扰动不确定性"（PRESCRIBE 已做）；
4. 不写"普遍优于 predicted magnitude"（E200/chemical/双未见有反例）；
5. 课程/考试材料不进科研主线；
6. 不删负结果、不挑细胞系、不改种子；
7. 分区口径分开（中科院大类/小类、JCR），录用不保证；
8. agents/ 目录材料不是实验事实入口，事实以 `docs/实验结果/` 冻结报告为准。

## 文件索引

| 文件 | 内容 | 状态 |
|---|---|---|
| `README.md` | 本章程 | 定稿 |
| `STATE.md` | E201 链执行状态机（唯一当前状态入口） | 持续更新 |
| `00_AUDIT_REPORT_20260817.md` | 交接包 §8 要求的完整审核报告（第一交付物） | 定稿 |
| `01_LITERATURE_VERIFY_20260817.md` | 联网文献核查：近邻论文、刮查结论、期刊分区事实 | 定稿 |
| `02_JOURNAL_FACTS_20260817.md` | 期刊分区/IF/政策事实表（与 01 互补） | 定稿 |
| `03_THINKING_LOG.md` | GLM 持续思考日志（写给 AI 看，按时间追加） | 持续更新 |
| `04_E201_RUNBOOK.md` | 16/16 完成后评估链的逐步执行手册（**规范命令已写死**） | 定稿 |
| `05_E201_RESULTS.md` | E201 三门裁决与四 target 结果（自动化在 STAGE_7 后生成） | 待生成 |
| `06_EVIDENCE_AUDIT.md` | 独立复算记录 + E189/E191 标签错误更正 + 复核提示 | 定稿 |
| `07_给作者的九点后说明书.md` | 零术语版：额度到期后会发生什么、BLOCKED 怎么办、汇报话术 | 定稿 |
| `paper/` | 中英文正文、五张图、图脚本、cover letters、README | 进行中 |
| `../../code/safeconf_audit/` | 最小可安装审计包（12.6s 一键复算 E199/E200，13 项全过） | v0.1.0，未提交 |

## 给下一个 AI 的阅读顺序

```text
proj/START_HERE_FOR_AGENTS.md
proj/agents/glm/README.md          ← 你在这里
proj/agents/glm/STATE.md           ← 先看当前到哪一步
proj/agents/glm/04_E201_RUNBOOK.md ← 再看下一步怎么做
proj/agents/glm/00_AUDIT_REPORT_20260817.md
proj/docs/实验结果/CURRENT_RESEARCH_STATUS_20260815.md
```

冲突时的权威顺序：`docs/实验结果/` 冻结报告 > `GATE_STATUS_20260729.md` > 本目录。
