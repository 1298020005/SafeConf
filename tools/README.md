# SafeConf tools

这里放“会执行，但不是 SafeConf 核心论文代码”的工具。

## 目录

| 路径 | 用途 |
|---|---|
| `scripts/` | 数据下载、正式实验复跑、补充实验复跑脚本 |
| `codex-switch/` | 服务器 Codex 账号和网络环境文档 |
| `build_safeconf_resource_inventory.py` | 生成服务器资源清单 |

## 和 code/ 的区别

```text
code/  = SafeConf 正式代码
tools/ = 管理、复跑、同步、环境辅助
```

运行任何长时间脚本前，先读脚本头部说明。

依赖旧桌面资源的 `build_safeconf_desktop_portal.ps1` 已移入
`/home/yyf/archive/safeconf/tool_history/`，不再作为当前同步方式。
