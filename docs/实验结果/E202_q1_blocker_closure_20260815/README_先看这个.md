# E202：补上一区审稿会问的对照，而不是再训一个模型

冻结日期：2026-08-15

## 要关上的洞

没有一张表同时回答：

1. PRESCRIBE 官方置信度在困难 OOD 上是否还能用；
2. predicted magnitude 何时更强；
3. SafeConf / 家族分歧在哪些 setting 有复核收益；
4. 什么时候应该写 `ABSTAIN`，而不是报一个数。

E158 已经解封且官方分数饱和。E202 **禁止**用同一 P3/P4 再跑一遍当确认。

## 分两截

### E202a — 现在做，不占 GPU，不碰 E201 真值

只读已经存在的正式表，生成论文用总表：

| 行 | 数据来源 | 允许写入的内容 |
|---|---|---|
| PRESCRIBE 官方分数 × 严格未见基因 | E158/E159 | 饱和、主统计不可估计、应 abstain |
| 公开 TxPert × K562 未见扰动 | E199 | 分歧相关与 20% utility；magnitude CI 跨 0 |
| 公开 TxPert × K562 整背景留出 | E200 | magnitude 更强；图模型仍优于 general baseline |
| 小矩阵 / 双未见 | E189 | 随机容易、双未见可负 |
| 跨研究 RPE1 | E192 | 事前 gate 为 ABSTAIN |
| E201 四背景 | 无 | 本表留空，写“盲测未释放” |

输出：

- `tables/E202A_SETTING_COMPARISON.csv`
- `reports/E202A_REPORT.md`
- 一张白底总图（若数字齐全）

### E202b — E201 正式评价之后，且必须换新数据

只有当阶段 III 决定冲大类一区、或 Briefings 审稿明确点名 PRESCRIBE 头对头
时才启动。必须满足：

- 新的未解封扰动集合，不能是 Norman P3/P4；
- 官方 combined 分数在验证折上先通过“非饱和”门槛（任务间标准差 > 0）；
- 与 SafeConf、magnitude 使用同一任务键和同一 20% 预算；
- 不占用正在跑 E201 的 GPU1。

## 禁止

- 打开 E201 target 表达；
- 把 E159 的 post-hoc raw log_prob 相关写进主结果；
- 把饱和写成 ρ=0 或“PRESCRIBE 无效所以我们更好”。
