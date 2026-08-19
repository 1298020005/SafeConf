# 给单细胞大牛小导师的故事迭代（毕业 = 一区）

更新：2026-05-20

小导师看重的不是「又做了一个预测模型」，而是三件事。下面按他的视角重写，并标明**证据要补什么**。

---

## 1. 是否真的解决了问题？

### 领域里的真问题（不是我们的自嗨）

湿实验里真正痛的是：

> 我在细胞系 A 里看到扰动 X 有效，到细胞系 B / 病人背景里还要不要再做一遍？  
> 如果模型说「有效」，我敢不敢信？

现有工具（GEARS、CPA、CellOT、scGen）默认**必须给答案**，很少回答**「该不该搬、搬了会不会翻车」**。

### 我们声称的解法（一句话）

**SafeTrans-PT / PolicySafeTransPT**：把跨背景扰动效应预测改成**带可迁移性判断的决策**——能搬才搬，不能搬就拒判或回退强基线。

### 怎样算「真的解决了」（小导师会认的标准）

| 标准 | 含义 | 当前状态 |
|------|------|----------|
| **硬 split 上有效** | held-out perturbation 上 effect 指标稳定优于强基线 | 部分数据集有信号，未全面赢 V0 |
| **难边界被识别** | leave-context 上 unsafe 样本误差更高 | 有 risk-coverage / safe-unsafe 对比，需加强 |
| **不是换指标刷分** | top20、DEG、program 与 Pearson 同向 | 周末 DeepSafe 有，Policy 需 46 号宽跑确认 |
| **外部可复现** | ≥3 独立队列、多种子 | 数据够，证据在补 |
| **生物学可解释** | 为何能搬/不能搬（pathway、module） | 有 network 雏形，需 1 张硬图 |

**对他汇报时别说**：「我们全面超越 GEARS」。  
**要说**：「我们补了 industry 里缺的 **transportability decision** 这一层，并在 held-out perturbation 上有正向证据；leave-context 我们用来定义 unsafe boundary。」

---

## 2. 是否真的讲好一个故事？

### 故事骨架（三区，15 分钟组会版）

**第一幕 — 痛点（2 min）**  
单细胞扰动很贵。大家在做「预测表达」，但实验决策者问的是：**这个效应在另一个背景还成立吗？**

**第二幕 — 方法（5 min）**  
不是黑箱端到端，而是三步：

1. **检索**：历史上类似 context / perturbation 任务  
2. **路由**：基线 V0 vs 程序迁移 V2 vs 网络先验  
3. **拒判**：transportability 低 → 不硬搬（risk-coverage 可展示）

**第三幕 — 证据与边界（5 min）**  
- 主结果：**held-out perturbation**（未见扰动）  
- 边界：**leave-context**（未见背景）→ 支持「需要 safety」的动机  
- 对照：V0、V2、**ContextSim**（证明不是相似度加权就能做）、GEARS（同任务或诚实说明任务差）

**收尾（1 min）**  
对行业的用处（见下节）+ 下一步：外部验证 + pathway 图 + 手稿

### 故事上必须避免的坑

- 把 leave-context 失败说成「模型不行」→ 应说成 **「这正是我们要检测的 unsafe transport」**  
- 堆 8 个模型名 → 只讲 **PolicySafeTransPT** 一条主线  
- 只报 Pearson → 一定带 **top20 + program + risk-coverage 一张图**

---

## 3. 是否真的对行业有用？

### 对实验组（湿实验）

- **少做无效迁移**：在不适合的背景上少浪费 CRISPR/药物筛选  
- **优先做**：模型标为 safe + 高 transportability 的 perturbation × context 组合  

### 对计算组（方法/benchmark）

- 提出新任务：**transportability-aware perturbation effect prediction**  
- 新评价：**risk-coverage + safe/unsafe contrast**，而不只是 MSE/Pearson  
- 可接 scPerturb 生态：统一 split + 指标（我们已在用多数据集）

### 对「大牛小导师」最有杀伤力的 utility 句

> 当虚拟细胞/扰动预测进入临床前筛选时，**「敢不敢信跨背景预测」** 比 **「平均 Pearson 高 0.02」** 更决定能不能指导实验。我们给的是前者。

---

## 4. 毕业（一区）与这个故事的对齐

| 学院硬性 | 故事要求 | 实验要闭合的环 |
|----------|----------|----------------|
| 1 篇一区 | 新问题 + 可信方法 + 硬证据 | Policy 赢 V0/V2/ContextSim（held-out）；GEARS 表；3 外部 |
| 小导师认可 | 真问题 + 好故事 + 有用 | 上文三节 + 1 个生物学 case figure |

**自动进度**：每轮跑完看 `results/Q1_READINESS_REPORT.json` 的 `label`，不要凭感觉说「够一区了」。

---

## 5. 下一轮故事迭代（做完 46 号跑后填）

- [ ] held-out main 胜率：___%  
- [ ] 能否展示 risk-coverage 单调改善：是 / 否  
- [ ] 最能打动导师的 1 个数据集案例：___  
- [ ] 需要补的 1 张生物学图：___  

（由 `monitor_q1_push.sh` 自动更新检查项时可挂接此处。）
