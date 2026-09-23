# 与现有可靠性方法如何做同合同对照

核查时间：2026-09-23。外部 PertEMA 源码以提交 `43c09a32e23d0ee2ae5dfbab21b2deeab27f1803` 稀疏检出，保存在 `archive/external/PertEMA/`，不纳入 SafeConf Git 大仓库。

## 已确认的事实

- [PertEMA 官方仓库](https://github.com/OfficialBishal/PertEMA)也是**预测后的可靠性层**，因此 SafeConf 不能自称首个给扰动预测打风险分的方法。PertEMA 使用由折外误差训练的梯度提升树、校准与区间；其 README 明确说发给用户的冻结模型是 CD4 T 细胞筛查示例，换筛查数据要用该筛查的折外误差重新训练。对应代码：外部 `pertema/scoring.py`。
- PertEMA 发布版 `pertema/featurize.py` 把上下文固定为 `Rest/Stim8hr/Stim48hr`；它不能未经适配直接接收 E201 四细胞系或 Jiang24 的 `cell_type×treatment`。把其演示权重硬套到新数据并报低分，是不公平的对照。
- [PRESCRIBE 论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6383e7643415842b48a5077a1b09c98-Abstract-Conference.html)是带不确定性估计的**上游响应预测模型**。它可以作为另一种预测器/不确定性输出做共同任务比较，但不能在输入模型、任务或误差定义不同的情况下直接拿文中一个数字与 SafeConf 风险效用相减。

## 可实施的比较合同

分两条信息预算，不混表：

1. **零目标误差标签部署**：SafeConf候选、幅度、分歧、单项历史和固定历史组合，只使用预测、目标对照及官方训练来源历史。E235 的 Jiang24 224 任务属于本栏。
2. **允许历史误差标签的校准部署**：复现 PertEMA 风格的学习型误差评分器，统一给两边相同数量、同一基因分组的折外历史错误标签。可用官方源码适配新上下文；适配部分及训练/测试基因隔离需单独登记、测试。此栏的胜负不能证明零标签方案劣/优，因为输入资源不同。

最终相关工作只能写可核对的差异：若 E235 通过，SafeConf 在**不使用目标任务真实误差训练一个评分器**的条件下，利用训练来源的跨背景效应离散度做预测后排序；PertEMA使用折外误差训练元模型，PRESCRIBE把不确定性融入预测器。若 E235 失败，就不能把这个差异等同于实际性能优势。
