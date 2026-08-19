# 方向判断：有没有论文潜力

一句话先说：
**有潜力，但不是“随便堆资源就能上”的那种潜力。**
它更像一个方法论文：关键看我们能不能证明“安全迁移”这件事真的比“直接预测”更重要。

## 1. 这件事到底值不值得做

值。

原因很简单：单细胞 perturbation 里，**同一个扰动在不同细胞状态、不同队列、不同实验背景里，效果不一定能直接搬过去**。  
所以真正的问题不只是“预测 effect”，而是：

- 这个 effect 能不能迁移到新 context
- 不能迁移的时候，模型能不能自己停下来
- 迁移时该信谁，基线、程序模型还是网络模型

这个问题比单纯预测更像一个“决策问题”。

## 2. 现在别人已经做了什么

相关方向很多，但大多是“预测”而不是“安全决策”：

- [scGen](https://www.nature.com/articles/s41592-019-0494-8)：学扰动后的表达变化
- [CPA](https://www.nature.com/articles/s41587-022-01495-w)：把 perturbation / cell state 拆开建模
- [CellOT](https://www.nature.com/articles/s41587-022-01496-9)：用 optimal transport 做跨状态映射
- [GEARS](https://www.nature.com/articles/s41592-023-01739-0)：做基因扰动，强调图结构与组合泛化
- [scPerturb benchmark / systema 类工作](https://www.nature.com/articles/s41587-024-02249-0)：更系统地比较扰动预测方法
- [PertAdapt](https://www.biorxiv.org/) / [TxPert](https://arxiv.org/) 这类近期工作：更强调外推、预训练、图知识或基础模型适配

它们都在解决“怎么预测”，但**很少把“什么时候不该预测”当成主问题**。

## 3. 我们现在的想法，不只是资源整合

我们现在做的不是把几个模型拼起来，而是把问题改写成：

### Safe transport / 安全迁移

模型先判断：

- 这个任务像不像训练过的历史任务
- 哪个专家更可信
- 这次是不是该拒绝迁移

然后才输出预测。

现在代码里的核心是：

- `V0`：稳一点的基线
- `V1`：程序空间 transport
- `V2`：加 pathway / graph prior
- `Network`：网络先验版本
- `PolicySafeTransPT`：负责检索、路由、混合、拒判

这不是简单 ensemble，因为它多了一层**决策逻辑**：

1. 先检索相似历史任务  
2. 再路由到不同专家  
3. 再做校准和 abstention  

这更接近现在计算机里常见的：

- retrieval-augmented prediction
- mixture-of-experts
- selective prediction / abstention
- calibrated routing

## 4. 为什么它不是纯资源整合

因为我们改的不是“多放几个模型名字”，而是：

- 输出目标变了
  - 不只给 effect
  - 还给 `transportability_score`
  - `selected_expert`
  - `unsafe_flag`

- 评价标准变了
  - 不只看 Pearson / Spearman
  - 还看 top20 overlap、DEG precision、program consistency、risk-coverage

- 问题定义变了
  - 从“尽量预测”
  - 变成“该不该迁移、迁移谁、什么时候停”

这就是方法感，而不是资源拼接感。

## 5. 现在的代码落地情况

已经落地的东西：

- `03_code/safetrans_models.py`
  - 新增 `PolicySafeTransPT`
  - 带检索 prior、专家路由、置信度拒判

- `03_code/run_full.py`
  - 已接入 `PolicySafeTransPT`

- `03_code/run_safety_abstention_evidence.py`
  - 也会记录 `PolicySafeTransPT`

- `03_code/run_policy_router_focus.sh`
  - 第一轮聚焦实验启动脚本

## 6. 现在最关键的实验要看什么

如果它真有论文潜力，必须至少看到这些信号：

- 在 hard split 上，`PolicySafeTransPT` 比 `V2` 更稳
- 不是只涨 Pearson/Spearman
- top20 / DEG / program consistency 里至少有一两个明显变好
- external 数据上方向一致
- unsafe flag 真的能把高风险任务筛掉

如果这些都做不到，那就说明它还只是一个“看起来合理”的想法，不够成论文主线。

## 7. 我的判断

目前这条线是**可以继续冲**的。

但要把它从“像一个方法”变成“像一篇论文”，还差最后一段证据：

- 更稳的 OOD 结果
- 更清楚的拒判收益
- 更像真实生物学问题的解释图

换句话说：
**方向是对的，成不成，接下来这轮实验最关键。**

