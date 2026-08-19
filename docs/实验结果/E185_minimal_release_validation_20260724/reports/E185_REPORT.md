# E185｜SafeConf 当前证书发布物最小复现

## 结果

`tools/scripts/validate_current_certificate_release.py` 在全新输出目录中运行完成：

```text
SafeConf release validation: PASS (12033 checks, 0 failed)
```

标准库验证器重算出：

| 指标 | 重算结果 |
|---|---:|
| 研究数 | 4 |
| 评价任务 | 2,433 |
| 靶点簇 | 737 |
| 家族 RMS 下界违反 | 0 |
| 最坏成员下界违反 | 0 |
| 家族上界任务覆盖 | 2,331/2,433 = 95.81% |
| 家族上界靶点簇同时覆盖 | 666/737 = 90.37% |
| 最坏成员上界靶点簇同时覆盖 | 688/737 = 93.35% |
| 最大 Hilbert 恒等式残差 | 9.682e-17 |

E182 被单独锁定复核：

```text
status = FAIL
family target coverage = 16/20
worst-member target coverage = 19/20
beta-binomial reference P(K <= 16) = 0.186813
```

18.68% 的有限校准参考概率只解释波动，不改变注册门槛的 `FAIL`。

## 具体检查

1. E181 的 19 个发布文件逐个通过 `MANIFEST.sha256`；
2. E183 的 6 个直接输入逐个通过字节数与 SHA-256；
3. 2,433 行任务证书逐行重算两类下界和两类上界覆盖标志；
4. 按 `(study, target_cluster)` 重新聚合出 737 个靶点簇；
5. 每个靶点的同时覆盖标志与 E183 靶点表逐项一致；
6. 四项研究的任务数、靶点数、违反数和覆盖数与 E183 study summary 一致；
7. 所有重算全局数字与 E183 `RUN_STATUS.json` 一致；
8. 直接按 `Beta(18,2)`–beta-binomial 公式重算 E182 有限样本参考概率。

验证器与直接输入的字节数、SHA-256 固定在 `tables/INPUT_HASHES.csv`。相同输入下连续两次运行生成的 `VALIDATION_REPORT.md` 和 `CURRENT_RELEASE_MAIN_NUMBERS.csv` 哈希完全一致。

## 复现边界

这个入口验证“发布证书能否从已提交任务级结果独立重算”。它特意不依赖 NumPy、Pandas、PyTorch、GPU、原始表达矩阵和未公开模型检查点，因此适合审稿人和远程 Agent 快速核对。

完整端到端训练仍由 E176、E177、E180 和 E182 各自的冻结合同、模型输出、真值访问记录和运行脚本承担。E185 不冒充第二次模型训练。
