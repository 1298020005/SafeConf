# SafeConf 组会：研究演变与当前架构

这次汇报接续周老师 7 月 9 日的问题，说明哪些实验推动了方法变化，再汇报当前真实结果。老师问答已回到当时聊天原文核对；不重新介绍单细胞和扰动预测基础。

## 建议的五页主体

| 页 | 图 | 这一页要回答什么 |
|---|---|---|
| 1 | [研究演变](01_RESEARCH_EVOLUTION.png) / [SVG](01_RESEARCH_EVOLUTION.svg) | 原来想做什么，老师的追问和哪些结果改变了它？ |
| 2 | [当前架构](02_CURRENT_ARCHITECTURE.png) / [SVG](02_CURRENT_ARCHITECTURE.svg) | Public、Source、Target 谁训练谁；新预测器来了怎样评分？ |
| 3 | [Source 结果](03_SOURCE_RESULT.png) / [SVG](03_SOURCE_RESULT.svg) | 其他预测器的错误经验能否迁移？在哪个范围成立？ |
| 4 | [Target / PertEMA 适配](04_TARGET_RESULT.png) / [SVG](04_TARGET_RESULT.svg) | 现有错误学习器加入公共信息，额外获得什么？ |
| 5 | [固定复核收益](05_REVIEW_RESULT.png) / [SVG](05_REVIEW_RESULT.svg) | 最终选中的系统，在相同复核数量下实际帮助多少？ |

原有三 panel [结果图](MEETING_RESULTS.png) / [SVG](MEETING_RESULTS.svg) **原样复制，SHA-256 一致**。第 3—5 页只是从同一组已保存 CSV 放大呈现，没有重训或改变数值。

## 按追问使用的两页备份

- [E201 转折实验](APPENDIX_E201_TURNING_POINT.png) / [SVG](APPENDIX_E201_TURNING_POINT.svg)：幅度、旧分与条件关联，另列固定复核收益。E201 的误差目标是四个 GAT 种子的 family RMS。
- [困难设置](APPENDIX_HARD_SETTINGS.png) / [SVG](APPENDIX_HARD_SETTINGS.svg)：老师要求的整行、整列、双未见；展示 E189 中分歧与家族误差的实际关系。范围来自不同支持预算，不是 CI。

## 开场可以直接这样讲

> 老师，上次您问的是这个分数对应哪个模型的误差、分歧是不是只反映任务难度，以及整行整列和跨数据集是否还成立。我后面沿着这些问题做了验证。几个结果让方法发生了实质变化：幅度比原来的固定分数强；家族下界和经验排序需要分开；看到 PertEMA 后，自身错误学习已有直接近邻，所以我把重点放到公共实验参照和旧预测器错误向新预测器迁移。今天我按这些转折讲，再汇报目前真正采用的配置。

完整 [口头讲稿](TALK_TRACK.md) 按约 10—12 分钟组织，可依讨论压缩。网页建议的独立核对在 [审阅说明](REVIEW_OF_WEB_PLAN.md)，逐项数据来源在 [证据表](EVIDENCE_BINDINGS.csv)。

## 当前采用决定

- **McFaline 当前默认：PublicRule。** Source/Target 候选分别保留，配置通过对应开发验收才升级。
- **TxPert Source DEV：双向 Source HGB 有明确增量。** 条件 gate 弱于始终 Source，未额外采用 gate。
- **PertEMA：当前真值下的官方树配方及原生输入适配。** 增加 Public 有明确增量；没有把它标成完整官方校准 / conformal 流程。
- **家族证书：独立并行研究线。** 不作为当前公共距离的概率保证。

不同阶段的模型、任务、误差合同和汇总方式各自保留；这些图不表示同一队列逐层上涨。

## 重画与复核

在研究工作树运行：

```bash
/home/miniconda/bin/python tools/scripts/build_safeconf_meeting_evolution_v1.py
```

脚本只读取已发布表格。没有新增训练、永久 TEST 读取或服务更改。图与输入的哈希在 [FIGURE_MANIFEST.json](FIGURE_MANIFEST.json)。PNG 适合直接放幻灯片；SVG 可以编辑，跨平台缺字时使用 PNG。
