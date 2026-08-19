# E179 靶点级嵌套不确定性基线比较

E179 是上界方法开发实验，不是新的外部确认。

- [完整报告](reports/E179_REPORT.md)
- [方法汇总](tables/E179_METHOD_SUMMARY.csv)
- [逐轮结果](tables/E179_REPEAT_RESULTS.csv)
- [配对效率差](tables/E179_PAIRED_REDUCTIONS.csv)
- [输入哈希](tables/E179_INPUT_HASHES.csv)

当前冻结候选是 `extra_trees_vector`：它使用预测向量形状、两模型关系和五个随机种子波动形成误差基线，再按完整扰动靶点做 split conformal 校准。该候选已经写入 E180 的预注册锁；能否成立只看 E180 新评价真值。
