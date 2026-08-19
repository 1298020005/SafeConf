# agents 协作入口

> Codex、Claude、Cursor、Qoder、Grok 都先读这里。
> 目标：互相找问题，不让用户反复手动解释。

进入角色目录前，必须先读：

```text
START_HERE_FOR_AGENTS.md
docs/学习导航/README.md
docs/学习导航/03_Agent学习任务与验收.md
```

`agents/` 保存协作状态和原始意见，不是当前实验事实入口。发生冲突时以 `docs/实验结果/GATE_STATUS_20260714.md` 为准。

## 文件怎么用

| 文件 | 谁写 | 用途 |
|---|---|---|
| `STATE.md` | 主要由 Codex 更新，其他 AI 可指出错误 | 当前事实，只写已确认内容 |
| `THREAD.md` | 三个 AI 都能写 | 当前攻防讨论，按时间追加 |
| `TASKS.md` | Codex/Cursor 更新 | 下一步要做什么，别散 |
| `DECISIONS.md` | Codex 更新 | 已接受的决定，避免反复吵 |
| `PROMPT.md` | 给外部 AI 复制 | 如果某个 AI 不能读文件，就复制这份 |

## 角色目录

| 目录 | 用途 |
|---|---|
| `codex/` | Codex 执行记录和服务器侧主控说明 |
| `claude/` | Claude 评审口径、验收清单、只读复核说明 |
| `cursor/` | Cursor 本地编辑/侦察任务交接 |
| `qoder/` | Qoder 原始输出和图解草稿 |
| `grok/` | Grok 原始输出和复制工具 |

根目录下的 `STATE.md`、`TASKS.md`、`DECISIONS.md` 是共享状态；角色目录只放各自材料。

## 规则

1. 不许默认同意任何 AI。
2. 不许把未实现设计写成已完成结果。
3. 每次批评必须给证据路径。
4. 每次提出新实验，必须写输入、输出、通过标准。
5. 用户是小白，英文第一次出现要带中文解释。

## Codex 互审包（2026-07-12）

强制阅读：`CODEX_ADVERSARIAL_REVIEW_PACK_20260712.md`（周老师验收 + 二区关卡 + 对抗题）。

## 当前短命令

```bash
cd /home/yyf/proj
cat agents/README.md
cat agents/STATE.md
cat agents/TASKS.md
```

## Windows 粘贴

对话后输入 `/copy`，再对输出的纯文本执行 Grok 内置 `/copy`，粘贴到本机。
详见 `agents/grok/README.md`。
