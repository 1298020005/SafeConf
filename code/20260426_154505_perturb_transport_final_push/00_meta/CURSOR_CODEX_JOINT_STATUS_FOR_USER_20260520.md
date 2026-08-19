# Cursor + Codex 联合状态说明

时间：2026-05-20

## 先说人话版

Cursor 这次主要像“质检员”和“审稿人模拟器”：

- 它没有主要负责发明新模型。
- 它做的是：把我们到底够不够强，变成可检查的硬标准。
- 它补了一个很重要的对照：`ContextSimBaseline`，意思是“只靠细胞环境相似度能不能做到同样效果？”

我这边主要像“实验工程”和“方法改造”：

- 我继续改模型，让它别只跟着平均误差跑。
- 我加了 `EffectBlendV2` 和 `TopRankGraftV2`，用来测试一个想法：
  - `V0` 很稳，别轻易破坏；
  - `V2` 的整体数值不稳，但有时关键基因排名有用；
  - 所以尝试把 `V2` 的关键基因排名 graft 到 `V0` 上。

合起来看：

- Cursor 补的是“怎么证明这东西不是自嗨”。
- 我补的是“怎么让模型真的有机会超过强基线”。
- 这两部分可以互补，不冲突。

## Cursor 具体做了什么

### 1. 写了硬标准

文件：

`00_meta/Q1_TOP_Q2_MASTER_STANDARD.md`

作用：

- 把“强二区 / 一区潜力”拆成具体门槛。
- 例如：必须赢 `V0`、赢 `V2`、赢 `ContextSimBaseline`，还要有 external validation 和 risk-coverage。

这件事有用，因为以后不能只说“感觉还行”，必须看表。

### 2. 写了自动评分器

文件：

`03_code/evaluate_q1_readiness.py`

作用：

- 自动读结果表；
- 判断当前是：
  - `NOT_READY`
  - `Q2_CANDIDATE_NEEDS_CONFIRMATION`
  - `Q2_TOP_READY`
  - `Q1_READY_CANDIDATE`

我已经修了两个点：

- 支持 `--primary-model`，这样可以评估 `TopRankGraftV2` 这种 probe；
- 修正了一个逻辑：一区必须真的有 `ContextSimBaseline`，不能没跑也放过。

### 3. 加了 ContextSimBaseline

文件：

`03_code/transport_models.py`

模型名：

`ContextSimBaseline`

通俗解释：

它是一个很朴素的对照：

“如果两个细胞环境看起来很像，那我就把相似环境里的扰动效果拿过来。”

如果我们的模型赢不了它，那审稿人会说：

“你这个不就是简单相似度加权吗？”

所以这个对照很重要。

### 4. 跑了 CPU 宽证据

路径：

`46_q1_cpu_push_20260520/`

它跑了：

- `V0`
- `V2`
- `ContextSimBaseline`
- `SafeTransPT`
- `NetworkSafeTransPT`
- `PolicySafeTransPT`

结果已经完成。

最新自动评分：

`NOT_READY`

但有一个有用信号：

`PolicySafeTransPT` 相对 `ContextSimBaseline` 在 held-out perturbation 上明显更好。

## 我这边做了什么

### 1. 改了 GPU 模型目标

我发现单纯用 MSE 训练太保守。

所以我在 `run_deep_gpu_transport.py` 里加入：

- effect-aware loss；
- rank loss；
- cosine direction loss；
- top-effect sign loss。

意思是：

模型不能只追求平均误差低，还要抓住真正变化最大的关键基因。

### 2. 加了 EffectBlendV2

想法：

`V0` 稳，`V2` 有时能抓关键基因。

所以让模型自己在验证集上决定：

“这次要不要混一点 V2？”

### 3. 加了 TopRankGraftV2

想法：

不要把整个 `V2` 预测都拿过来，因为它整体数值可能很烂。

只拿 `V2` 认为最重要的 top genes，插到 `V0` 的预测里。

这就像：

`V0` 负责稳住大局，`V2` 负责提醒“哪些基因可能最关键”。

## 最新结果怎么理解

### CPU 主线

路径：

`46_q1_cpu_push_20260520/results/Q1_READINESS_REPORT.json`

标签：

`NOT_READY`

主要原因：

- `PolicySafeTransPT` 还没有稳定赢 `V0`；
- risk-coverage 没有达到要求；
- unsafe / safe 区分还不够稳定。

好消息：

- `ContextSimBaseline` 已经补上；
- `PolicySafeTransPT` 能赢 `ContextSimBaseline`，说明不是简单相似度就能解释完。

### GPU / graft 线

路径：

`43_gpu_effect_objective_main_20260520/`

整体结果：

- `TopRankGraftV2` 比 `V2` 稳；
- 个别 setting 上 top20 / DEG 有提升；
- 但还没有稳定超过 `V0`。

路径：

`48_gpu_graft_tian_20260520/`

外部 Tian 数据：

- `DeepCalibratedSafeTransport` 的 Pearson / RMSE 比 `V0` 略好；
- 但 top20 / DEG 不是全面赢；
- `TopRankGraftV2` 作为关键基因 probe 有价值，但不能直接当主方法。

## 现在两边怎么配合

最合理分工：

| 部分 | 谁负责 | 用处 |
|---|---|---|
| 硬标准和自动评分 | Cursor | 防止乱吹 |
| `ContextSimBaseline` | Cursor | 审稿对照 |
| CPU 宽证据 | Cursor / 后台 | 判断主方法是否稳定 |
| GPU effect objective | Codex | 找突破口 |
| `EffectBlendV2` / `TopRankGraftV2` | Codex | 方法 probe / 消融 |
| 最终整合和讲法 | Codex | 给你汇报和写稿 |

## 当前最重要结论

现在不能说已经达到一区或强二区。

但我们比之前清楚很多：

1. 问题是明确的：跨细胞环境的扰动效应迁移不总是安全。
2. `V0` 是非常强的基线，必须正面承认。
3. `PolicySafeTransPT` 能赢简单相似度 baseline，这是一个有用信号。
4. `TopRankGraftV2` 提示：未来突破口可能不是整体预测，而是“稳定整体 + 关键基因排名修正”。
5. 下一步应该把 graft 思想合并回 `PolicySafeTransPT` 的路由里，而不是另开一个方向。

## 下一步

1. 把 `TopRankGraftV2` 的关键基因 graft 思想，合并成 `PolicySafeTransPT` 的一个专家选项。
2. 对 `ContextSimBaseline`、`V0`、`V2` 做同 setting 的正式比较图。
3. 写 `GEARS_HEAD_TO_HEAD.md`，说明 GEARS 和我们任务定义是否完全一致。
4. 如果要更像论文，需要补真实 pathway / GO / Reactome prior，而不是继续只靠 hash prior。

