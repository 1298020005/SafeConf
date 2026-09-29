# SafeConf：9 月 30 日开题与进展汇报入口

更新日期：2026-09-29。交付对象：导师/学校的完整研究进展和开题报告。

## 当前论文判断

SafeConf v4 已完成开发、冻结、PertEMA 公平比较和 E170 一次性确认，当前进入：

> **Route A：SafeConf 计算方法论文。**

主方法是 nested evidence-aware shrinkage：先从 Universal Prediction Evidence 建立 prediction-only risk，再根据历史 Support、Relevance、Conflict 和 Missingness 决定历史修正的使用强度。Quality 因合法字段不足没有进入最终确认模型。

E170 四面板包含 2,400 个任务、800 个新 perturbation cluster、一个 held-out donor 和 12 个预注册 strata。冻结 V2 相对 Magnitude 的 ΔUtility@20 为 +0.0504，10/12 strata 非负，风险退化护栏全部通过，Gate A PASS。该证据属于同研究新扰动/留出供体确认，不称 external-study confirmation。

## 汇报文件

1. 开题报告确认版：[Markdown](开题报告与进展.md)｜[Word](SafeConf_开题报告与进展_20260930.docx)｜[PDF](SafeConf_开题报告与进展_20260930.pdf)
2. [12 页汇报提纲与答辩问答](汇报提纲.md)
3. [从实验到方法的数学推导](方法推导.md)
4. [实验与投稿路线](实验与投稿路线.md)
5. [导师汇报结果总表](../../实验结果/Stage2_mature_upstream_20260928/SAFEConf_导师汇报结果_20260930.md)
6. [论文结果包](../../实验结果/Stage2_mature_upstream_20260928/PAPER_RESULT_TABLES.md)
7. [一次性确认报告](../../实验结果/Stage2_mature_upstream_20260928/confirmation/e170_primary_cd4_four_panel/REPORT.md)

## 明天优先讲的四个结论

1. Magnitude 是强基线，不能用弱基线制造提升。
2. TxPert 两结构中 V2 的 U20 为 0.7963/0.7918，均高于 Magnitude 和同合同 PertEMA 适配。
3. 方法冻结后，E170 一次性 confirmation 的 U20 为 0.2159，对比 Magnitude 0.1655，并通过预注册 Gate A。
4. 有合法历史时收益明确；无历史任务接近无信息，跨 family 小样本 gate 和弱化学上游作为失败边界完整保留。

## 论文主张边界

可以写：

- 统一黑盒 post-hoc risk auditing 接口；
- Support/Relevance-aware historical evidence；
- nested evidence shrinkage；
- 同研究 held-out donor/new-perturbation confirmation；
- PertEMA 同合同比较；
- Error Memory 周期重训扩展。

当前不写：

- Quality-aware 已验证；
- 广泛 model-agnostic；
- external-study confirmation；
- 化学扰动主确认；
- 完全零历史泛化；
- 在线逐样本自学习。

## 复现与版本

论文包状态、代码、结果表、主图、确认图、失败修复日志和文件哈希均位于：

```text
docs/实验结果/Stage2_mature_upstream_20260928/
```

核心测试：

```bash
python tests/test_safeconf_v4_evidence_contract.py
```

开题 Word 生成：

```bash
python tools/scripts/build_safeconf_proposal_docx.py \
  --source docs/方法设计/20260928_证据驱动开题/开题报告与进展.md \
  --output /tmp/SafeConf_开题报告与进展_20260930.docx
```
