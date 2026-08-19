# E185｜当前证书最小复现验证

一条命令：

```bash
python tools/scripts/validate_current_certificate_release.py
```

本轮实际运行了 12,033 项检查，失败 0。验证器只读 Git 已提交的证书表、状态文件和哈希清单，不读取原始 h5ad、评价真值数组或模型检查点。

详细结果：

- [E185 报告](./reports/E185_REPORT.md)
- [重算主数字](./tables/E185_MAIN_NUMBERS.csv)
- [根目录复现说明](../../../REPRODUCE_CURRENT_RELEASE.md)

E185 的 `PASS` 表示发布物内部一致、主数字可由任务级表重算、E182 的失败裁决未被覆盖。它不代表从原始数据重新训练模型，也不代表期刊录用。
