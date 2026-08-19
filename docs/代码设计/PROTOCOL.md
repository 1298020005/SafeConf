# Cross-Context Perturbation Prediction Confidence Scoring · 评估协议（v0.2）

| 项 | 内容 |
|:---|:---|
| 状态 | **已冻结（2026-05-23）** — 新数据集 blind 不得改公式 |
| 用途 | 论文 Methods 骨架；代码 `safetrans_confidence` 的行为规格 |
| 依据 | 导师要求 + `01-研究方案/SafeTrans-confidence-scoring-方案.md` |
| 实证基线 | `outputs/confidence_task_mvp_v2_1/` + `outputs/benchmark_protocol_v0_2_pkg/` |
| 主分数 | `protocol_v0_2_family_confidence` |

---

## 1. 协议要解决什么问题

在单细胞 **cross-context perturbation effect prediction** 中，已有 predictor（V0、ContextSim、GEARS 等）会对每个 `(cellular context c, perturbation p)` 输出一个预测效应向量 `ŷ`。本协议回答：

> **这一次 prediction 有多可信？** 即：给出一个标量 **confidence score**，使得 **分数高 ↔ 真实误差低**。

本协议 **不** 训练新的深度 predictor，**不** 以「预测 RMSE 是否 SOTA」为主结论。

---

## 2. 基本对象

### 2.1 一条 PredictionRecord（评估单元）

**定义**：一次 **(dataset, fold, split, context c, perturbation p, predictor π)** 上的 prediction。

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `record_id` | str | 全局唯一 |
| `dataset_name` | str | 如 Haber |
| `fold_id` | int | 0 … K-1 |
| `split` | enum | `train` / `val` / `test` |
| `context` | str | target context |
| `perturbation` | str | target perturbation |
| `predictor_name` | str | 如 V0StrongBaseline |
| `predicted_effect` | vec ∈ ℝ^G | predictor 输出（NPZ 存） |
| `true_effect` | vec ∈ ℝ^G | 该 context 内观测效应（NPZ 存） |
| `true_error_rmse` | float | **主 ground truth 误差** |

**G（基因维）**：默认 **5000**（与项目 46_q1 一致）；写入 `config` 不可 silent 改成 1000。

### 2.2 Confidence score（方法输出）

| 名称 | 方向 | 说明 |
|:-----|:-----|:-----|
| `confidence_score` | 越大越可信 | **论文主输出** |
| `risk_score` | 越大越不可信 | 可选，= 单调变换的 confidence 或预测误差回归值 |

**评估时**：若方法输出 risk，则 aligned correlation 对 risk 取正号，等价于 confidence 与误差负相关。

---

## 3. 数据与任务构建

### 3.1 效应定义

对每个 `(c, p)`：

```text
effect(c,p) = mean(expression | c, p) - mean(expression | c, control)
```

control 判定：non-targeting / ctrl / vehicle 等（与 `build_context_splits.py` 一致）。

### 3.2 数据集族（论文分线）

| 族 | 数据集（计划） | 论文角色 |
|:---|:---------------|:---------|
| **gene_main** | Haber, Parekh, Norman,（+ Adamson） | **主文 Table** |
| **chem_robust** | KaggleCrossCell, KaggleCrossPatient | **Robustness 附表**（标注同源） |
| **external** | PapalexiSatija 或 Frangieh（后期） | 一区 external |

**禁止**：把 chemical 与 genetic 混在一个 pooled Spearman 当唯一 headline。

### 3.3 当前已跑通（MVP v2_1 + 包复现）

Haber, Parekh, KaggleCrossCell；test records = **154**（两 predictor 合计）。

### 3.4 主分数公式 v0.2（冻结）

在 **fold train** 上对特征 z-score，再组合（**禁止**用 test 标签调权重）：

| 族 | 公式 |
|:---|:-----|
| **gene_main** | `z(context_similarity) + z(log_support) − z(model_disagreement)` |
| **chem_robust** | `z(log_support) − z(model_disagreement)`（stability 权重 = 0） |

实现：`safetrans_confidence/scoring/protocol_v0_2.py`；配置：`config/scoring/protocol_v0_2.yaml`。

---

## 4. Split 协议（核心）

### 4.1 选用：held-out (context, perturbation) pair

对每个数据集，全体 task 为 `(c,p)`。划分单位是 **pair**，不是单独 held-out context 或 held-out perturbation。

**直觉**：predictor 在 train 中见过 `c` 和 `p`，但没见过组合 `(c,p)` → 测的是 **组合迁移** 下的可靠性。

### 4.2 每 fold 划分

```text
全部 eligible (c,p) pairs
  → 按 perturbation 分层 K-fold（默认 K=5；极小数据集 K=3）
  → 对 fold f:
       test_f     = 该 fold 的 held-out pairs
       trainval_f = 其余 pairs
       val_f      = trainval 的 10%（分层）
       train_f    = trainval 的 90%
```

### 4.3 必须满足的泄漏约束（自动化断言）

| ID | 约束 |
|:--:|:-----|
| L1 | `(c,p) ∈ test_f` ⇒ `(c,p) ∉ train_f` 且 `(c,p) ∉ val_f` |
| L2 | 每个 test 的 `p` 在 train_f 中存在某 `c'≠c` 使 `(c',p)∈train_f` |
| L3 | 每个 test 的 `c` 在 train_f 中存在某 `p'≠p` 使 `(c,p')∈train_f` |
| L4 | 计算 feature 时 **禁止** 使用 test 的 `true_effect` 或 test 标签 |

### 4.4 Split 文件发布格式（建议）

```json
{
  "dataset": "Haber",
  "fold_id": 2,
  "seed": 5201,
  "train_pairs": [["ctxA","pert1"], ...],
  "val_pairs": [...],
  "test_pairs": [...]
}
```

路径：`splits/{dataset}/fold_{k}.json` — 与 OpenML task 文件类似，便于复现。

---

## 5. Predictor 协议

### 5.1 角色

Predictor 是 **黑盒**：只消费 train fold 的 mask，对 val+test 的 task 输出 `predicted_effect`。

### 5.2 MVP / 二区 必选

| name | 说明 |
|:-----|:-----|
| `V0StrongBaseline` | 强 lookup 基线 |
| `ContextSimBaseline` | 上下文相似迁移 |

### 5.3 一区 扩展

| name | 说明 |
|:-----|:-----|
| `GEARS`（或 CPA） | 用已有 checkpoint / `gears_formal_baselines_v2`，不重训 |

### 5.4 训练规则

```text
对每个 (dataset, fold_f, predictor π):
  π.fit(tasks, train_mask = train_f)
  ŷ = π.predict(tasks, task_ids = val_f ∪ test_f)
```

**禁止**：在 test 上调参；禁止用 test error 选 predictor。

---

## 6. Confidence features（仅允许用 fold train 统计）

| feature | 直觉 | 缺失处理 |
|:--------|:-----|:---------|
| `context_similarity_max` | target context 与 train 中 control 最相似 | 必填 |
| `perturbation_support_count` | train 中同 p 的 context 数 | 必填 |
| `perturbation_effect_stability` | train 中同 p 各 context 效应一致性 | **允许 NaN**；禁止用 test 填假值 |
| `prediction_magnitude_deviation` | ‖ŷ‖ 相对 train 效应尺度 | 必填 |
| `model_disagreement` | 多 predictor 的 RMSE 分歧 | ≥2 predictor 时 |
| `ood_nearest_distance` | (c,p) 在 train 表征空间距离 | 必填 |
| `historical_residual` | train 内 LOO context 预测残差（按 p 聚合） | 必填 |

**化学线（KCC）**：`stability` 缺失率可 >80%；**主文 combined 应使用 chem 专用权重或置零 stability 项**（待 GPT 讨论定稿）。

---

## 7. Scoring 方法（benchmark 必跑清单）

| score_name | 类型 | 论文角色 |
|:-----------|:-----|:---------|
| `random` | baseline | 应 ≈0 相关 |
| 单 feature → score | baseline | 每个 feature 单独 |
| `simple_combined` | **主方法候选** | 规则组合 + fold val 调权 |
| `historical_residual` | 强 baseline | 单独一行 |
| `learned_histgb` | ablation | 同 fold train+val；**小样本不得作唯一 headline** |
| `model_disagreement` | 诊断 | 与 learned 区分是否独立 |

### 7.1 simple_combined（v0.1 算法）

```text
对每个 (dataset, fold, predictor):
  在 train_f 上算 z-score 参考（median/MAD 或 mean/std）
  在 val_f 上网格搜索权重 w（固定符号：+sim,+stab,+support,-mag,-dis,-ood）
  目标：最大化 aligned Spearman(confidence, rmse) on val_f
  固定 w 应用于 test_f
```

### 7.2 learned_histgb（v0.1 算法）

```text
对每个 (dataset, fold, predictor):
  Train pool = train_f ∪ val_f  （禁止其他 fold）
  HistGradientBoostingRegressor → y = true_error_rmse
  Predict test_f only
```

---

## 8. 评估指标（论文主文 vs 附录）

### 8.1 主文（必须）

| 指标 | 定义 | 聚合 |
|:-----|:-----|:-----|
| **ρ_aligned** | Spearman(score, RMSE)；confidence 取负对齐 | **per (dataset, predictor)**；再报 median across gene_main |
| **RC@80%** | 保留 top 80% confidence 样本的平均 RMSE vs 全量 RMSE 的相对下降 | per dataset |
| **ΔRMSE_high-low** | 最高 20% vs 最低 20% confidence 的 mean RMSE 差 | per dataset + bootstrap 95% CI |

### 8.2 附录（建议）

- Pearson
- AURC / AUGRC（selective prediction 社区）
- AUROC failure detection（failure = RMSE > q80）
- Calibration buckets

### 8.3 禁止作为主结论

- **pooled 全数据集** 的单一 Spearman（除非明确标为 supplementary）
- 只报 learned 不报 combined
- 不标 split 泄漏检查的实验

---

## 9. 当前实证（v2_1，供讨论时引用）

**simple_combined aligned ρ（test）**：

| dataset | ρ |
|:--------|--:|
| Haber | 0.48 |
| Parekh | 0.43 |
| KaggleCrossCell | 0.12 |

**已知局限**：

- learned 在同 fold 协议下 **未** 稳定优于 disagreement
- KCC stability 缺失 82.5%
- test n 仍偏小（154 total）

这些 **不否定协议**，但决定主文叙事应偏 **benchmark + 可解释 combined**，而非 learned SOTA。

---

## 10. 论文贡献与协议的对应（写 Introduction 用）

| 贡献 | 协议条款 |
|:-----|:---------|
| C1 Task 定义 | §2 |
| C2 Held-out pair 协议 | §4 |
| C3 Feature + scoring 族 | §6–7 |
| C4 系统 benchmark | §3、§5、§8 + 多数据集 |
| C5 生物解释（后期） | 协议外 post-hoc analysis |

---

## 11. 复现最小集（benchmark suite 发布）

他人复现需：

1. 本 `PROTOCOL.md` + `config/*.yaml`
2. `splits/*.json`
3. `python -m safetrans_confidence.cli run --config configs/gene_main.yaml`
4. 输出：`prediction_records.parquet`, `eval_summary.csv`, `figures/`

环境：`scgpt_env` + `datasets/singlecell_perturbation_atlas` 路径（或文档说明如何获取 h5ad）。

---

## 12. v0.2 已定稿项

- [x] chem 线 stability：**置零**（chem_robust 公式不含 stability）
- [x] 主方法：`protocol_v0_2_family_confidence`；learned 仅 ablation
- [x] historical_residual：仅 baseline score
- [ ] 主文是否加 AUGRC（附录可选）
- [ ] split JSON 随 supplementary 发布（Phase 1 工程项）

---

## 13. 版本历史

| 版本 | 日期 | 变更 |
|:-----|:-----|:-----|
| v0.2 | 2026-05-23 | 冻结 family 公式；`safetrans_confidence` 包启动 |
| v0.1 | 2026-05-21 | 初稿；吸收 mvp_v2_1 教训 |
