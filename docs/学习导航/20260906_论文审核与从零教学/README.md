# 2026-09-06 论文审核与从零教学

这套材料同时包含科研事实、图解教学和期刊初筛。第一次阅读不要直接钻进 4 MB 的期刊长表；先看第 7 项弄清哪些来自实验、哪些只是规则初筛，再看第 8 项：Codex 审核里哪些该听、哪些说过头。

先读顺序：

1. [完成度与 GPT 误导对照](./01_完成度与GPT误导对照.md)
2. [从零 Nature 图解教学](./02_从零Nature图解教学.md)
3. [一区二区投稿就绪报告](./03_一区二区投稿就绪报告.md)
4. [五成分与封存流程精讲](./04_五成分与封存流程精讲.md)
5. [期刊对照表](./05_期刊对照表.md)（8 本放大）
6. [820 本期刊宽口径初筛](./06_全球相关期刊逐本评价.md)（每行分开标记业务 A/B；45 行人工画像，775 行刊名/类别规则；不是 820 本官网精读）
7. [Grok 会话与 820 期刊产物详细审核](./07_Grok会话与820期刊产物详细审核_20260907.md)（Codex 对 Grok 会话和 820 行产物的审核；主判断可用）
8. [对 Codex 审核的复核](./08_对Codex审核Grok期刊产物的复核_20260907.md)（留存：07 里哪些该采纳，哪些是加重指控或把图注修正说成图结构修正）

若 Cursor 报找不到 04：工作区请打开 `/home/yyf/proj`，或直接粘贴

```text
/home/yyf/proj/docs/学习导航/20260906_论文审核与从零教学/04_五成分与封存流程精讲.md
```

数字表：[CLAIM_TABLE.json](./CLAIM_TABLE.json)，由 `python3 -m safeconf_audit.paper_pack` 从官方 CSV 生成。图在 [figures/](./figures/)。

重新生成图并自检：

```bash
cd /home/yyf/proj
PYTHONPATH=code/safeconf_audit python3 -m safeconf_audit.paper_pack --repo /home/yyf/proj
PYTHONPATH=code/safeconf_audit python3 code/safeconf_audit/tests/test_paper_pack.py
```
