# SafeTrans-PT 方法升级说明

一句话：
不是只问“这个扰动能不能迁移”，而是先去历史里找像不像，再决定信谁，最后不确定就别硬猜。

## 现在的核心思路

1. 先找相似场景
   - 看这个新任务和训练集中哪些 perturbation + context 最像。
   - 这一步像“先查字典 / 先检索案例”。

2. 再让多个专家模型竞争
   - `V0`：稳一点的平均基线
   - `V1`：程序空间运输
   - `V2`：加路径/图先验的运输
   - `Network`：共表达网络版本

3. 最后做路由和拒判
   - 不是强行选一个答案。
   - 如果模型自己都不太确定，就回退到基线，或者标成 unsafe transport。

## 为什么这比只做一个打分器更像“新方法”

- 以前更像：给每个任务打一个“能不能迁移”的分数。
- 现在更像：一个小型决策系统。
  - 先检索
  - 再路由
  - 再校准置信度
  - 不稳就拒绝

这更接近现在计算机领域常见的：

- retrieval-augmented learning
- mixture-of-experts
- calibrated abstention
- selective prediction

## 对这个课题真正有用的点

- 不是所有 perturbation effect 都该硬迁移。
- 真正有价值的是：
  - 哪些能迁移
  - 哪些不能迁移
  - 什么时候该停手

这就把问题从“单纯预测”往“安全迁移决策”上推了一步。

## 现在代码里做了什么

- `PolicySafeTransPT`
  - 负责在多个专家之间选路
  - 同时混入检索 prior
  - 再根据置信度决定是否 abstain

- `run_full.py`
  - 已经把这个新模型接进评估链

- `run_safety_abstention_evidence.py`
  - 也会输出这个模型和 baseline 的对比结果

## 后面最值得继续补的方向

- 让 retrieval 更聪明
  - 不只是按 control / prior 找相似
  - 还能加 gene-program embedding

- 让校准更严格
  - 用 conformal prediction 做更正式的拒判

- 让专家更分工
  - 哪类任务交给哪类专家
  - 不再只靠一个统一模型硬扛

