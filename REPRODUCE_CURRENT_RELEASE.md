# SafeConf 当前证书最小复现

这一入口用于外部审稿、远程 Codex 和新服务器快速复核当前主数字。它不训练 scGPT/GEARS，不读取原始表达矩阵、评价真值数组或模型检查点，只使用 Git 已提交的任务级证书、靶点簇表、状态文件和哈希清单。

## 一条命令

在仓库根目录运行：

```bash
python tools/scripts/validate_current_certificate_release.py
```

只依赖 Python 标准库，支持 Python 3.9 及以上。默认输出：

```text
runtime/current_release_audit/
├── VALIDATION.json
├── VALIDATION_REPORT.md
└── CURRENT_RELEASE_MAIN_NUMBERS.csv
```

## 它重新检查什么

1. E181 发布目录的 manifest 哈希；
2. E183 所有输入文件的字节数与 SHA-256；
3. 2,433 个任务的家族 RMS 下界和最坏成员下界；
4. 每个任务的上界覆盖标志是否与数值一致；
5. guide 任务按研究和靶点聚合后是否恰为 737 个靶点簇；
6. 四项研究的任务数、靶点数、下界违反和上界覆盖是否与发布表一致；
7. E182 是否仍为 `FAIL`、是否仍是 16/20；
8. `Beta(18,2)`–beta-binomial 参考概率是否重算为约 18.68%。

成功时终端应显示：

```text
SafeConf release validation: PASS
```

核心预期值：

```text
studies = 4
tasks = 2433
target_clusters = 737
family_lower_violations = 0
worst_lower_violations = 0
family_upper_targets_covered = 666
worst_upper_targets_covered = 688
E182 status = FAIL
```

## 自定义输出目录

```bash
python tools/scripts/validate_current_certificate_release.py \
  --output-dir /tmp/safeconf_release_check
```

此验证器复核的是已发布证书的一致性与来源完整性，不替代从原始 h5ad 重新训练模型。完整训练链仍由 E176、E177、E180 和 E182 各自的冻结合同、运行脚本和访问审计承担。
