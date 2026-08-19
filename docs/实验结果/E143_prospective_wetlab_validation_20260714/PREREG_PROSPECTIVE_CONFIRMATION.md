# E143 预注册草案｜双背景前瞻 CRISPRi 验证

## 假设与设计

在一个从未进入七数据开发/评价的新细胞背景，以及 Jurkat 锚点或第二个新背景中，先取得三批未扰动 baseline，再生成 scGPT、GEARS 预测和 Directional-SafeConf 分数。任何扰动后表达数据产生前，冻结 48 个目标、96 条 sgRNA、预测向量、风险分数、代码提交与 SHA256。

48 个目标由 8 个双背景高风险、8 个双背景低风险、16 个背景反差和 16 个中间风险组成。高低风险组在 predicted magnitude、模型分歧、baseline 表达、STRING degree、DepMap fitness、guide 质量与功能类别上匹配。另冻 10 个有顺序的技术替补，只能因未表达、合成失败或不满足 guide 规则替换。

## 主终点

每个 guide×batch 先在同批 non-targeting control 上形成 pseudobulk Δ；再在冻结的 512 基因轴上计算两个预测器平均 centered-Pearson error、centered-cosine error，以及二者秩均值。统计独立单位是 gene；同一基因的两个背景、两条 guide、三批实验作为一个 cluster。

主 gate：两个背景的 Pearson 与 cosine 风险相关宏平均均为正；复合方向误差的 gene-cluster bootstrap 95% CI 下界>0；相对 predicted magnitude 的 Δρ 点估计>0。增强主张还要求 Δρ 的 95% CI 下界>0。背景反差组至少 12/16 个基因的风险差方向与真实误差差方向一致。

## 次要终点

RMSE、top-25% 对 bottom-25% 错误富集、AURC、guide 一致性、敲低效率、活率、凋亡、细胞周期、通路 PROGENy 误差和 cell-line×perturbation 交互。所有次要结果明确标注，不替换失败的主终点。

## 盲法与排除

湿实验人员和测序方只见随机样本/guide 编码，不见风险分层和预期方向。风险映射由独立保管人保存。完成 cell QC、guide QC、pseudobulk、排除日志与哈希后才能解盲。排除只按 `tables/E143_QC_RULES.csv`，不因预测不准、方向不符或 P 值不显著而删样本。

## 证据边界

若只重做 HepG2/Jurkat Nadig 面板，属于技术复现；若改用 siRNA，属于跨干预模态探索；若只做 qPCR 小面板，不能证明全转录组方向风险。正式确认至少一个细胞背景必须全新。
