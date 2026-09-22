# E224：场景门控的幅度—SafeConf 路由

E223 之后固定的假设：只缺细胞背景时不追加 SafeConf，其余三种场景允许补充幅度以上的风险。

- 嵌套选择后的平均 top-20% 效用差值：**+0.02063**。
- 外层正向研究：**3/4**。
- 所有 8196 个任务均保留；外层研究答案没有参与规则选择。
- 仍属于历史开发证据；需用 E208 或新研究按冻结规则做外部确认。

```bash
python tools/scripts/run_e224_scenario_gate_router.py --input <E187_CARTESIAN_TASK_CERTIFICATES.csv> --output <new_output_dir> --bootstrap 10000
```
