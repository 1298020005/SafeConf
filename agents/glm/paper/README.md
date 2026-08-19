# 论文工作区（GLM v1，2026-08-17；科学批判与多版本图更新于同日）

## 图的版本清单（全部白底 300dpi，png+pdf 各一份）

| 图 | 内容 | 版本 |
|---|---|---|
| Fig1 概念+三门+足迹 | 三栏示意 | **V1** `Fig1_concept_contract`；**V2** `Fig1V2_horizontal_flow`（横向单行流程，含真值时序带） |
| Fig2 跨场景信号有效性 | 森林图 | **V1** `Fig2_setting_forest`；**V2** `Fig2V2_footprint_heatmap`（判定热图，含 ρ 值与图例分级） |
| Fig3 翻转 | 双面板条形 | **V1** `Fig3_utility_flip`；**V2** `Fig3V2_dumbbell_flip`（哑铃图，一图看懂赢家互换） |
| Fig4 E201 设计 | 占位面板 | **V1** `Fig4_E201_design`；**V2** `Fig4V2_E201_complete_preview`（c 面板改为条件合同模拟预览，无假数据、明确标注） |
| **Fig5 逐分量足迹**（新） | 两个场景全部信号的 ρ+CI+效用 | `Fig5_component_footprint` |
| **Fig6 增量/机制/组合陷阱**（新） | 配对 Δ、偏相关轴切换、组合稀释 | `Fig6_increment_mechanism` |
| **Fig7 条件合同模拟**（新） | 九场景×三策略矩阵 + 平均/最差对比 | `Fig7_conditional_contract` |
| FigS1 弃用台账（补充） | E192/E189+E191/E158 | `FigS1_abstention_ledger` |

选择建议：主刊投稿推荐组合 = Fig1V2 + Fig2V2 + Fig3V2 + Fig4V2 + Fig5–7（科学主线
最清晰）；组会汇报推荐 = Fig1（V1 三栏）+ Fig3V2 + Fig7。

## 占位符纪律（给下一个 AI）

1. 正文里所有 `[[E201-PLACEHOLDER-*]]` 只能在评估链 STAGE_7 完成、
   `STATE.md` 记录三门裁决后替换；
2. 摘要 A/B 两版按冻结三叉选择，禁止出数前预先选定"好听"版本；
3. 替换时**四个 target 全部报告**，方向不一致者不许删；
4. 图 4c 面板出数后重画（脚本里留了函数结构）；
5. 投稿前自查禁语：first / universally outperform magnitude /
   gene+chemical 统一成功 / 任何分区承诺。

## 已知待办（出数前后）

- [ ] E201 出数：替换全部占位符；表 1 补 E201 行；图 4c 重画
- [ ] ConfPert 全文人工复核（OpenReview 验证墙）
- [ ] pip 最小审计包 + 一条复现命令（E199/E200 主数字重算）
- [ ] references.bib 整理（英文稿参考文献当前为编号列表）
- [ ] cover letter（Briefings）；导师确认作者顺序/通讯/基金/致谢
- [ ] 学院分区口径书面确认（决定 Briefings 定位）
