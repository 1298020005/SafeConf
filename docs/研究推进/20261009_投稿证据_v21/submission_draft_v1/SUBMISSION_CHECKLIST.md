# SafeConf 投稿执行清单

## 主文必须回答

- [x] 新预测器没有自身错误记录时，公共实验能否启动风险审核。
- [x] 公共审核是否优于预测幅度、支持量和历史能量基线。
- [x] 两种主预测器是否在同一任务合同上复现。
- [x] Target feedback 与 Public 的关系是否按相同预算、公平输入和两类统计端点报告。
- [x] 跨模型压力证据是否区分 qualified evaluation 与 competence-failed stress test。
- [x] KOLF 是否在能力门、风险冻结、evaluation truth 时序和混合历史覆盖下闭合。

## 主图

- [x] Figure 1：Public evidence risk-audit framework。
- [x] Figure 2：McFaline 20% review severe-error retrieval。
- [x] Figure 3：Target feedback budget，global 与 context-macro 分开。
- [x] Figure 4：KOLF frozen independent evaluation。
- [x] Figure 5：Frangieh cross-family stress evidence。

## 审稿人最可能检查的公平性

- [x] PublicRule 不读取当前研究 error labels。
- [x] KOLF predictor competence 在 evaluation truth 之前完成。
- [x] KOLF risk scores 在 evaluation truth 之前冻结并保存 hash。
- [x] 无历史任务进入完整排序并使用固定幅度回退。
- [x] Source/Target/公共规则保留相同任务、误差合同和生物簇 bootstrap。
- [x] 反馈顺序和 learner seed 不计为独立生物样本。
- [x] Frangieh predictor competence failure 公开报告，不冒充外部确认。
- [x] 1,400 输出轴与实际 1,366 公共共同轴的差异显式记录，不补零。

## 补充材料

- [x] 所有训练/评价/确认 split、信息账本和输入 hash。
- [x] PertEMA 当前适配与官方完整 conformal 流程的范围区别。
- [x] DeepSets、背景映射、Source gate、Target label-equivalent 负结果。
- [x] KOLF 失败回执、备用 KNN 条件和未触发记录。
- [x] 每张图的 source data、重算命令和环境版本。

## 写作顺序

1. 先用 `MANUSCRIPT_DRAFT.md` 固定主线和结果顺序。
2. 将 `REFERENCES_TO_INSERT.md` 的原始论文和官方代码补入参考文献。
3. 用 `FIGURE_LEGENDS.md` 统一图注，保持 global 与 macro 术语一致。
4. 将 Supplement 索引中的表格和审计文件挂到正文主张。
5. [x] 生成完整主文 PDF、补充材料 PDF 和投稿包；生成前再次核对 `ARTIFACT_MANIFEST.json` 的输入 hash。
6. [ ] 套用目标期刊官方模板并填写作者、通讯作者、基金和数据声明；这一步不改变冻结结果。
