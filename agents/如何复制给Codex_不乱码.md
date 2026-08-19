# 怎么粘贴给 Codex 不乱码

乱码原因：中文 UTF-8 在 Windows/某些终端被当成 Latin-1/GBK 解读。

## 方法 A（最稳，推荐）

复制**纯英文**指令：

```bash
cat /home/yyf/proj/agents/PASTE_TO_CODEX_EN.txt
```

或纯 ASCII（路径里的中文会变成 `\uXXXX`，Codex 仍可读）：

```bash
cat /home/yyf/proj/agents/PASTE_TO_CODEX_EN_ASCII.txt
```

在 Codex 里粘贴上述英文全文即可。内容完整，含周老师要求与 Gate 计划。

## 方法 B：浏览器复制中文

```bash
# 本机用浏览器打开（Remote 则端口转发或下载 html）
xdg-open /home/yyf/proj/agents/PASTE_TO_CODEX.html
# 或
firefox /home/yyf/proj/agents/PASTE_TO_CODEX.html
```

页面上点 **「复制英文版」** 或 **「复制中文版」**。

## 方法 C：Windows 记事本

用 **UTF-8 BOM** 中文文件：

`/home/yyf/proj/agents/PASTE_TO_CODEX_ZH_UTF8BOM.txt`

用 VS Code / 记事本「UTF-8」打开，再全选复制。不要用会强制 GBK 的旧工具直接 type。

## 方法 D：让 Codex 自己读文件（最好，零粘贴）

在 Codex 只发这一句英文：

```text
Read and obey this file completely:
/home/yyf/proj/agents/PASTE_TO_CODEX_EN.txt
Start at Step A. Do not skip gates.
```

这样完全不经过剪贴板，不会乱码。
