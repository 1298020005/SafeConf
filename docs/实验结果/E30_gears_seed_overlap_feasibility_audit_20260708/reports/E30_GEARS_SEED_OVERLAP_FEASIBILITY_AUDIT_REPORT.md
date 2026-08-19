# E30 GEARS seed-overlap feasibility audit

生成时间：2026-07-08T13:04:59

## 先看结论

E30 检查了 E25 strict GEARS 包能否直接支持 seed/ensemble uncertainty。

- E25 records：54
- unique task groups：47
- repeat ≥2 tasks：5
- repeat ≥3 tasks：2
- singleton tasks：42
- repeated-task true effect max diff：0.0

结论：当前 E25 包可以证明 true effect 在重复任务内一致，但不能支撑正式 seed-ensemble uncertainty，因为大多数任务只出现一次。seed disagreement 只能在 5 个重复任务上做 exploratory check。
