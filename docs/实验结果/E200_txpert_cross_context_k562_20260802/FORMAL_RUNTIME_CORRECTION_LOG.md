# E200 正式评价运行时修正

## 2026-08-02：attempt 1 图形阶段停止

正式评价已完成数据完整性、任务误差、scPertEval、bootstrap 和裁决表计算，在制图时因 `yerr=decile.sem` 停止。`decile` 是 DataFrame，属性访问返回 Pandas 的 `sem()` 方法，而不是同名列。

修正锁定为：

```python
yerr=decile["sem"]
```

本次修正只改变图形误差条的列取值方式。不改变预注册任务、特征、误差、评价端点、bootstrap 种子、复核预算、统计判据或结论解析。首次部分输出保存在 `DATA/txpert_official_20260802/e200/failed_attempts/formal_attempt_001/`，哈希见 `E200_FORMAL_ATTEMPT_001_HASHES.csv`。
