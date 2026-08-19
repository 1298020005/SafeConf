# SafeTrans-PT：一区门槛 + 二区 Top 达标标准

更新：2026-05-20

这份是**可执行的投稿门槛**，不是安慰性自评。  
项目内旧的 `Q2_READY_WITH_FOCUSED_CLAIMS` **不等于**期刊二区，更不等于一区。

---

## 目标期刊档（按您要求）

| 档位 | 代表期刊 | 我们要达到的状态 |
|------|----------|------------------|
| **二区 Top** | Bioinformatics、NAR Methods、Briefings in Bioinformatics 上沿 | 主文可投，审稿人挑不出「没基线 / 没外部验证」 |
| **一区门槛** | Genome Biology、Nature Communications（方法）、Nat Methods 边缘 | 问题 + 方法 + **稳定 SOTA 信号** + 生物学解释 |

---

## 主方法（冻结，不再发散）

**唯一主模型：** `PolicySafeTransPT`  
**唯一 GPU 深模型：** `DeepCalibratedSafeTransport`（effect-objective 校准门）  
**必打对照：** `V0`、`V2`、`ContextSimBaseline`、正式 **GEARS**（同数据集多种子）

其他变体（SafeTransPT、Network、no_abstain）只进 **消融 / 附录**。

---

## 硬性 Pass/Fail（自动评分见 `evaluate_q1_readiness.py`）

### A. 二区 Top（`Q2_TOP_READY`）— 全部满足

1. **Main held-out perturbation**：`PolicySafeTransPT` 相对 `V0` 在 **≥60%** setting 上 effect 指标胜出（top20 或 DEG ≥ +0.01，且 program↑ 或 RMSE↓），且 **≥3** 个数据集。
2. **External held-out**：**≥2** 个独立数据集，胜率 **≥50%**。
3. **Risk–coverage**：80% coverage 时 RMSE 比全量回答 **≥3%** 更好。
4. **Safe vs unsafe**：**≥50%** setting 上 `unsafe_rmse > safe_rmse`。
5. **GEARS 表**：Norman + Adamson 正式跑通（已有），主文必须有一张 **同任务定义** 的对比表或诚实说明任务差异。

### B. 一区门槛（`Q1_READY_CANDIDATE`）— 在 A 之上全部满足

1. Main held-out 胜率 **≥75%**（相对 V0）。
2. Held-out 相对 **V2** 胜率 **≥55%**。
3. Held-out 相对 **ContextSimBaseline** 胜率 **≥70%**（证明不是「相似度加权」就能做）。
4. **≥3** 个 external held-out，胜率 **≥65%**。
5. Leave-context：**program consistency** 相对 V0 胜出比例 **≥50%**（不强求 Pearson 全面赢，用拒判叙事补）。
6. **消融完整**：no_abstain / no_pathway / ContextSim / V2 至少 4 张图。
7. **生物学解释**：≥1 个数据集有 module/pathway 图（network module 或 Reactome/GO enrichment）。
8. **手稿叙事**：leave-context 作为 **unsafe boundary**，不是失败。

---

## 证据包清单（投稿前必须齐）

| # | 产物 | 路径模式 | 状态 |
|---|------|----------|------|
| 1 | 宽口径 CPU 安全证据 | `46_q1_cpu_push_*/results/` | 进行中 |
| 2 | GPU 校准主跑 | `39–41_gpu_*_20260520/` | 部分完成 |
| 3 | Q1 自动评分 | `results/Q1_READINESS_REPORT.json` | 每轮跑完自动生成 |
| 4 | GEARS 正式基线 | `FORMAL_GEARS_FINAL_SUMMARY.csv` | 已有 |
| 5 | 主文 4 图 | 效应热图、risk-coverage、unsafe 对比、GEARS 汇总 | 待刷新 |
| 6 | 方法稿 story | `08_论文定位和后续路线_CN.md` | 已有，随结果更新 |

---

## 与 Codex 分工（并行）

| 执行方 | 任务 |
|--------|------|
| **本机 CPU** | `run_q1_cpu_master_20260520.sh` → 宽数据集 PolicySafeTrans + ContextSim + risk-coverage |
| **GPU 队列** | `queue_gpu_effect_objective_20260520.sh` → DeepCalibrated + effect-objective |
| **Codex** | 读 `~/.codex/AGENTS.md`：补 GEARS 同 split 表、Reactome prior、手稿 Results 段 |
| **你** | 确认毕业/学院硬性分区文件；有则放进 `00_meta/GRADUATION_REQUIREMENTS.md` |

---

## 诚实预期

- 达到 **Q2_TOP_READY**：再跑 **1–2 周** 集中实验 + 1 周写稿，**可行**。
- 达到 **Q1_READY_CANDIDATE**：通常还需 **1–2 月**（外部验证 + 生物学图 + 稳定赢 V2/GEARS），除非 GPU 校准线接下来全面超过 V0。

每轮实验结束运行：

```bash
python 03_code/evaluate_q1_readiness.py --results-dir <RUN>/results --write-md
```

以 `Q1_READINESS_REPORT.json` 的 `label` 为准，不靠感觉。
