# SafeConf 当前唯一投稿文件夹

这里是当前唯一有效的论文稿件目录。旧课程论文、早期风险排序稿和内部 E 编号报告都不是投稿正文。

## 直接打开

1. `SafeConf_manuscript.docx`：可直接在 Word 中修改的英文主文。
2. `SafeConf_manuscript.pdf`：主文固定版式预览。
3. `SafeConf_manuscript.md`：主文源文件，数字、公式、图注和声明均在这里。
4. `SafeConf_supplement.docx`：补充材料 Word 版。
5. `SafeConf_supplement.pdf`：补充材料固定版式预览。
6. `SafeConf_supplement.md`：补充材料源文件。
7. `figures/`：五张英文主图，含 600 dpi PNG、PDF 和 SVG。
8. `tables/`：2 张主表和 14 张补充表。
9. `SafeConf_source_data.zip`：可直接作为 Additional file 2 上传的完整 CSV 数据包。
10. `Cover_letter_BMC_Bioinformatics.docx`：投稿信 Word 版。
11. `Cover_letter_BMC_Bioinformatics.md`：投稿信源文件。
12. `SUBMISSION_CHECKLIST.md`：提交前仅剩的作者信息与上传检查。

## 稿件定位

- 当前题目：**SafeConf: registered-family error certificates for auditable single-cell perturbation prediction**
- 稿件类型：计算生物学方法类 Research Article。
- 当前格式：BMC Bioinformatics 风格的结构化摘要、Methods、Results、Discussion 和 Declarations；内容保持可迁移。
- 主张边界：SafeConf 的贡献是注册模型家族的误差证书对象、参考质心上界搬移、靶点簇校准和 fail-closed 真值访问协议。平方误差分解、三角不等式和 split conformal 均明确归于经典方法。
- 关键负结果：GSE225807 的家族上界覆盖为 16/20，低于事前要求的 17/20，正文保留为 `FAIL`，没有事后改阈值。
- 新增压力测试：4 个多背景数据集 × 4 种训练比例 × 4 种缺失结构，共 8,196 个任务；另有 581 个跨数据集任务。8,777 个任务的确定性下界均无违例，跨数据集部分明确不声称上界覆盖。
- 对照与边界：正文已加入 scGPT–GEARS 模型特异性分析、50 次完整靶点重分割的上界基线比较，以及 PRESCRIBE 原生终点对照。

## 投稿前只需人工确认的内容

这些信息无法从代码或实验记录中推断，不能代填。除此之外，科学正文、实验、图表、补充材料、投稿信和机器审计已收口：

1. 全部作者及顺序；
2. 通讯作者姓名和邮箱；
3. 基金名称及编号；
4. 导师对目标期刊、作者贡献和 AI 辅助披露文字的最终确认。

主文中的方括号仅用于标记这四类信息。科学内容、主要数字、方法、图表和参考文献已写入。

## 重建

在仓库根目录运行：

```bash
python submission/SafeConf_current/scripts/build_figures.py
bash submission/SafeConf_current/scripts/build_documents.sh
```

仓库主数字可独立复核：

```bash
python tools/scripts/validate_current_certificate_release.py
python submission/SafeConf_current/scripts/audit_submission.py
```

预期输出：

```text
SafeConf release validation: PASS (12033 checks, 0 failed)
SafeConf submission audit: PASS (86 checks, 0 failed)
```

图表采用白底、统一字体和低饱和蓝灰配色；红色仅表示事前门槛未通过。图均由已提交 CSV 表格确定性生成，不含生成式图片。
