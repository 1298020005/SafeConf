# SafeConf 科学示意图与组会修订 v2

本版重画方法图和演变图；原有三张结果图原样保留。参考了本地 scGPT、scFoundation、C.Origami 的 Fig. 1，使用生物对象、效应向量、局部机制与 a–c panel 组织方法。论文截图只用于本机参考，没有复制进输出。

## 方法图

- **英文科学示意图：**[PNG](METHOD_OVERVIEW.png) / [可编辑 SVG](METHOD_OVERVIEW.svg)
- **中文组会版本：**[PNG](METHOD_OVERVIEW_ZH.png) / [可编辑 SVG](METHOD_OVERVIEW_ZH.svg)

图分三部分：新预测器输出，公共参照与可选风险学习，有限复核与后续反馈。绿色实线突出公共参照主干；蓝色、橙色虚线分别代表可选的共享风险与目标个性化。树形小图标表示已训练风险学习器，避免将历史错误画成当前查询直接读取的数据。家族证书作为灰色独立研究线放在底部。

当前 McFaline 仍采用 PublicRule。虚线是经开发验收后可以启用的候选，不表示三路必须相加，也不表示 Source/Target 已普遍优于公共规则。

细胞、培养皿、效应条和曲线是原创示意元素；曲线不是实际实验数据，绿色带表示历史变化的视觉示意，不是置信区间。各 panel 的含义和算法对应见 [图注](FIGURE_CAPTIONS.md)。

## 演变图

[实验驱动的转折图 PNG](EVIDENCE_DRIVEN_EVOLUTION.png) / [SVG](EVIDENCE_DRIVEN_EVOLUTION.svg)

按四个研究问题组织：困难设置中的分歧、强幅度基线、SafeConf-M 的相关与复核收益差别、核对 PertEMA 后的信息迁移问题。每个转折都有对应证据和实际改动。证书从 7 月起并行，不画成 E201 之后才产生。

## 原三图与补充强对照

以下文件与上一版字节一致：

- [Source 结果](03_SOURCE_RESULT.png)
- [Target / PertEMA 加 Public](04_TARGET_RESULT.png)
- [固定 20% 复核收益](05_REVIEW_RESULT.png)
- [原三 panel 合图](MEETING_RESULTS.png)

新补 [强基线图](APPENDIX_STRONG_BASELINES.png) / [SVG](APPENDIX_STRONG_BASELINES.svg)。数据来自已发布逐任务结果的必要复算，没有重训。

| 当前合同 | 方法 | U20 |
|---|---|---:|
| Exphormer → GAT | Magnitude / Source HGB | 0.749 / 0.780 |
| GAT → Exphormer | Magnitude / Source HGB | 0.746 / 0.786 |
| McFaline 212 任务 | 历史支持量规则 / PublicRule | 0.795 / 0.838 |

Source 减幅度的点增量分别为 0.031、0.040；5,000 次配对基因簇 bootstrap 区间为 [-0.016, 0.059] 和 [-0.001, 0.076]。真实 Source 超过公共与置乱对照的结论保留，相对幅度的额外收益尚未获得稳定区间证据。支持量规则不等于实验质量，PublicRule 与它的点差也不能全部归因于纯生物内容。

## 汇报使用顺序

1. 实验驱动的演变图：讲哪个结果改变了哪个判断。
2. 新方法图：讲当前计算关系与监督来源。
3. 固定复核收益：22 对 4，对应当前采用的 PublicRule。
4. Source 结果：同时讲强幅度对照；需要展示数字时使用强基线图。
5. Target / PertEMA：Public 增强错误学习器，但当前反馈候选未替换 PublicRule。

[修订后的完整讲稿](TALK_TRACK_V2.md) 保留上一版演变和结果内容，补齐强对照与 SAMS 能力范围。对网页建议的独立核对见 [审阅记录](WEB_REVIEW_RESOLUTION.md)。

## 重画

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python tools/scripts/audit_safeconf_meeting_strong_baselines_v2.py
/home/miniconda/bin/python tools/scripts/build_safeconf_scientific_figures_v2.py
```

第一条只复算已开放分数、核对相同真值并做配对统计。第二条画原创矢量图并复制已保存结果图。未新增训练、下载、永久 TEST 读取、服务修改或正文/PDF工作。来源与输出哈希分别保存在 STRONG_BASELINE_AUDIT.json 和 FIGURE_MANIFEST_V2.json。
