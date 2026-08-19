# codex-switch — 远程 lab-168 Codex 账号与网络

服务器 `10.11.52.168`（SSH 名 `lab-168`）上，Cursor 远程 Codex 面板 + 348/350 双账号切换 + 统一本地数据。

| 文档 | 内容 |
|:-----|:-----|
| [操作手册.md](./操作手册.md) | 日常：切号、自检、Reload |
| [目录与架构.md](./目录与架构.md) | 路径、脚本、当前架构 |
| [网络与代理.md](./网络与代理.md) | 7897 / 17897、Windows & 服务器配置 |
| [迭代记录.md](./迭代记录.md) | 历代方案变更时间线 |
| [踩坑记录.md](./踩坑记录.md) | 故障现象 → 原因 → 处理 |

**一键切号**

```bash
codex-switch 350   # 或 348
# Cursor → Reload Window
```

**自检**

```bash
~/bin/proxy-healthcheck.sh
~/bin/codex-data-status.sh
```

脚本实体在 `~/bin/`，本目录只放文档。
