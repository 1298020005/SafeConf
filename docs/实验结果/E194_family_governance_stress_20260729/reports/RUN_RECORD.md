# E194 运行记录

运行日期：2026-07-29

运行环境：`/home/yyf/.conda/envs/scgpt_env/bin/python`（Python 3.9）

## 版本顺序

1. `1ddde45`：冻结原始分析方案；
2. `e1c4567`：在尚未提交实现、尚未运行结果前完成独立数学审阅修订；
3. `bd1dee3`：提交 E194 实现；
4. 首次执行在构造审计阶段因 Python 3.9 不支持 `zip(strict=True)` 退出，未形成
   结果表；
5. `29f649b`：只修复运行时语法兼容，不改分析定义；
6. 重新完整执行并生成最终结果。

所有上述提交均在下一阶段开始前推送到 GitHub 与 Gitee 的
`exp/task-risk-audit-20260611` 分支。

## 成功运行

```text
/usr/bin/time -v \
  /home/yyf/.conda/envs/scgpt_env/bin/python \
  tools/scripts/run_e194_family_governance_stress.py
```

- wall time：1 分 55.35 秒；
- user time：122.79 秒；
- system time：7.72 秒；
- maximum resident set size：1,349,016 KB；
- exit status：0。

## 最终完整性

- 状态：PASS；
- 任务行：134,385；
- 场景摘要：310；
- family member 审计行：1,608；
- family / worst 下界违例：0 / 0；
- 最大平方恒等式残差：`6.661338147750939e-16`；
- 治理、E193 复现、方差分解与攻击解析式：492/492 通过。

首次兼容性失败没有被隐藏；它是代码执行失败，不是统计 gate 失败，也没有产生可供
选择的中间结果。
