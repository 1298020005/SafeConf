# 固定 physical-content null 正式器独立代码审阅

**结论：PASS（限定静态代码审阅），未发现需要停止当前健康任务的实质数学、配对或条件化错误。**

审阅文件：`tools/scripts/run_safeconf_orion_fixed_physical_content_null.py`。
SHA256：`9dcfe3ee508da55058a2b64556d36311a2ea16f7b6c091c5e69f722204a04150`。
同时只读核对固定 preparation、原 `8b69` weighted-prior 数学及原 `dec8` metric helper。审阅未运行正式器，未读取表达、预测数组或真值数值，未修改既有代码、回执或结果。

## 核对结果

- **身份及行对齐。** preparation 输入／输出绑定先校验；bank 行号及实验 ID 使用已验证准备包；固定 query 左侧一对一合并保持行序，`row_scores` 使用相同 query 顺序，原三规则先执行 bitwise 复算门。统计阶段再次核对 232 个 query 的完整身份集合和 canonical sorted 144 gene 顺序。
- **实际效应置换。** 每个固定种子按排序后的冻结 study/context/扰动类型/effect contract/gene-space/control/n_batches/cell-support-quartile 组生成循环 derangement。只改 pair 的 `memory_row`，真实 recipient ID、资格、`log_source_cells` 和查询历史链接保持；Manual 权重仍由 recipient cells 计算，Uniform 权重仍为均匀权重。移动数及组身份逐项校验，移动记录不得保留同 target。这里交换实际 effect vectors，没有 shuffle 距离列。
- **全集保留。** 5200 个可移动 bank rows 每个种子全部移动；singleton rows 留在原位，3 个无可移动历史的 query 仍保留。全 232 tasks、全部 18 分数必须 finite；未按结果删任务或挑种子。
- **封存与误差使用。** 原 3 分数及 15 null 分数、donor assignments 先落盘并写 score seal，再数值解析既有 SEEN cached errors。无 learner/model call、参数拟合、Source/CDF 更新或新 TEST opening。
- **固定配对统计。** 原 5000×144 saved gene indices 被共享于全部规则、种子、两个 context；每个 gene block 同时带入所有现存 context 行。调用原 metric helper，context 少于 20 时保留未定义结果，macro 对两个 context 等权且传播未定义。每项差值来自同一 draw 的 actual−null，不相减独立 CI。378 metric rows、315 paired rows 全部保留并报告 valid draws。
- **实现与输入保持。** preparation／数学／指标代码 pins、原三规则精确 score codec、运行前后输入 SHA 检查均保留；当前原 13 主比较和 Source 参数不变。

## 必须保留的解释边界

1. 支持匹配是预定 group 加 cell-count quartile，recipient support/weights 不变；donor cell count 不是逐值相等，亦未精确匹配测量方差。结果只能描述该条件化负对照下的效应对应关系，不能分离纯生物内容的因果效应。
2. 五个种子全部报告，不能事后选种子或缩到 229 可移动-history queries。这是固定 232 已见任务上的辅助诊断，不构成新的独立确认，也不改变主方法。
3. 封存前 `before` 完整性绑定包含 cached truth parquet 的流式 SHA 读取。因此准确表述应为“全部分数封存后才数值解析／使用 cached errors”，不应写“封存前真值文件完全未打开”。这些 bytes 仅用于固定输入完整性，未参与置换选择或分数计算。

本结论针对上述精确代码版本及既有 preparation 约束；不替代实际运行结果与输出 artifact 的终态核验，没有追加数值复算或统计运行。
