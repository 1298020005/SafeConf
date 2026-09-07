# 2026-09-06 论文审核与从零教学

先读顺序：

1. [完成度与 GPT 误导对照](./01_完成度与GPT误导对照.md)
2. [从零 Nature 图解教学](./02_从零Nature图解教学.md)
3. [一区二区投稿就绪报告](./03_一区二区投稿就绪报告.md)
4. [五成分与封存流程精讲](./04_五成分与封存流程精讲.md)
5. [期刊对照表](./05_期刊对照表.md)

若 Cursor 报找不到 04：工作区请打开 `/home/yyf/proj`，或直接粘贴

```text
/home/yyf/proj/docs/学习导航/20260906_论文审核与从零教学/04_五成分与封存流程精讲.md
```

数字表：[CLAIM_TABLE.json](./CLAIM_TABLE.json)，由 `python3 -m safeconf_audit.paper_pack` 从官方 CSV 生成。图在 [figures/](./figures/)。

重新生成图并自检：

```bash
cd /home/yyf/proj
PYTHONPATH=code/safeconf_audit python3 -m safeconf_audit.paper_pack --repo /home/yyf/proj
PYTHONPATH=code/safeconf_audit python3 -m pytest code/safeconf_audit/tests/test_paper_pack.py
```
