# E200 prepare 报告

- 生成时间：`2026-08-02T04:18:29+08:00`
- 数据：632,488 个细胞，3,352 个基因，5 个数据标签背景；模型训练背景固定为 RPE1、HepG2、Jurkat，目标为 K562。
- 严格 context-only：580 个任务；主分析 566 个，低细胞数敏感性 14 个。
- 官方 `unseen_cell` 中另有 validation-only 202 个和 train/validation 均未见 33 个，已从主分析剥离。
- 同时未见背景和扰动：272 个，单列压力测试。
- prepare gates：38/38。

公开 checkpoint 只有跨 K562 的单个 GAT。E200 先做单模型与 general baseline 的可复核审计；多模型家族结论保持未回答。
