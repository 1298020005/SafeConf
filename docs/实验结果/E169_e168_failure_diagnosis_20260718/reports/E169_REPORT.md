# E169｜E168 未确认结果的可复现诊断

## 结论

E168 的正式判定保持 **NO_CONFIRMATION**，本轮没有改公式、换终点或重写门槛。全 200 targets 的 ΔAURC 为 0.001061；seen 160 为 0.002045。两个区间均跨 0，不能写成独立确认。

## 为什么没有形成增量

在 seen-160 和 column-unseen-40 各自内部，SafeConf 与 disagreement 的弱序完全一致：**True**。同一 test donor、同一状态内，context similarity 只有 1 个取值；分层后 support 也只有 1 个取值。因此这两个结构特征只能区分 seen/unseen 两层，不能在每层内部给 160 或 40 个基因排序。真正承担排序的只剩模型分歧。

seen-160 的 context/support 可区分层级范围为 1–1。这不是代码退化：24 个预测器证书、6 个风险证书和三种子稳定证书都已通过；问题在于可部署特征的信息量。

## 三状态结果

| stratum | state | SafeConf AURC | magnitude AURC | Δ |
|---|---|---:|---:|---:|
| all_200 | Rest | 0.118709 | 0.119010 | +0.000301 |
| all_200 | Stim8hr | 0.122518 | 0.125491 | +0.002974 |
| all_200 | Stim48hr | 0.121495 | 0.121403 | -0.000093 |
| seen_160 | Rest | 0.118574 | 0.120877 | +0.002304 |
| seen_160 | Stim8hr | 0.121981 | 0.123196 | +0.001215 |
| seen_160 | Stim48hr | 0.120794 | 0.123410 | +0.002616 |
| column_unseen_40_descriptive | Rest | 0.114995 | 0.110445 | -0.004550 |
| column_unseen_40_descriptive | Stim8hr | 0.127420 | 0.125305 | -0.002115 |
| column_unseen_40_descriptive | Stim48hr | 0.113412 | 0.111488 | -0.001924 |

column-unseen-40 在三个状态的 Δ 都为负；seen-160 三个状态点估计都为正，但效应很小，层级推断未通过。

## 另外两项检查

validation donor 的历史误差、两个 train donors 的平均误差和跨 donor 标准差全部原样检查，没有只保留最好的组合。这些线索在已解封 160 个 seen targets 上信号偏弱，只能用于下一轮开发集特征筛选，不能拿来重算 E168 主结论。

guide 复现审计覆盖 600 个任务。guide 间差异是真值解封后才能得到的实验噪声指标，可解释测量不稳定性，但不能作为部署分数输入。

## 冻结方向风险头的二级结果

E135 在 E168 之前冻结的 Directional-SafeConf 未重拟合。其三状态宏平均方向 rank ρ=0.110，bootstrap 95% CI [-0.015, 0.232]；magnitude=0.139，disagreement=0.149。

这项分析是在 E168 absolute 主结果解封后提出，只能标为事后二级审计。即使相关为正，也不能替代失败的主终点，更不能冒充第二次前瞻确认。

## 下一步实验约束

1. E168 已解封 200 targets 永久退出模型选择和阈值调整；
2. 从 5,310 个尚未读取 targeting X 的合格 targets 中，按预先冻结哈希建立多个不重叠 200-target 面板；
3. 先冻结所有面板、代码、分数和联合统计，再统一解封，检验当前微弱正效应能否靠更大样本得到精确结论；
4. perturbation similarity、历史跨 donor 误差和模型特异风险只能在 train/validation 或其他开发数据中开发，之后另找未解封面板确认；
5. 同一供体的多面板属于靶点复制，不冒充多供体或多研究复制。真正冲一区仍需要新研究/新背景，优先是 E143 前瞻湿实验。

## 图

- `figures/E169_FIG1_STATE_DELTAS.png`：absolute 主终点；
- `figures/E169_FIG2_FEATURE_RESOLUTION.png`：结构特征分辨率；
- `figures/E169_FIG3_DIRECTIONAL_SECONDARY.png`：方向风险二级审计。
