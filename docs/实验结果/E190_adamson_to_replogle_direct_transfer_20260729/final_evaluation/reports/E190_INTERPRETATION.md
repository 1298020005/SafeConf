# E190 结果怎么回答周老师

## 跨数据集实验已经是“真预测器迁移”

模型只在 Adamson 2016 K562 CRISPRi 的 216 个训练伪重复和 54 个验证伪重复上
拟合。Replogle 2022 的输入只有 48 个 batch 的 non-targeting control 和扰动身份。
692 个目标任务的预测在提交 `75a71fa` 双远程之后，才读取 4,959 个目标扰动细胞。
这不是把风险分数从一个数据集搬到另一个数据集。

## 预测性能

任务加权平均 RMSE：

- scGPT：0.25826；
- GEARS：0.23850；
- 六模型 family：0.24919；
- 直接搬用 Adamson 同基因效应：0.23706；
- zero-effect：0.25828。

六模型 family 在 59.8% 的任务上优于 zero-effect，source-effect baseline 为
62.4%。但 47 个基因等权后，family 相对 zero 的 RMSE 差为 -0.00254，
95% CI [-0.00675, 0.00110]；source-effect 为 -0.00734，
95% CI [-0.01631, 0.00028]。两者都没有形成基因层面的明确优势。任务平均看起来
较好，部分原因是不同基因拥有的目标 batch 数不同。

## 分歧能说明什么

六成员多样性下界与 family RMS error 的 Spearman 为 0.424，
95% CI [0.135, 0.632]；直径一半与 worst-member error 为 0.507，
95% CI [0.240, 0.696]。跨研究 setting 中，分歧确实携带风险信号。

predicted magnitude 的相关为 0.420，source-effect magnitude 为 0.419，均与
分歧接近。因此，本实验不能写成“分歧优于幅度”。它支持的是确定性下界具有严格
含义，并能在部分 setting 反映难度。

## 证书与预测器必须分开

family RMS 和 worst-member 的确定性下界均为 0 违例。这个结论不依赖预测器优于
zero-effect。E189 显示 GEARS 会拖累 Primary CD4 family；E190 中 GEARS 反而比
scGPT 更适合直接迁移。SafeConf 评估的是当前模型家族的风险边界，不保证家族成员
本身准确，也不替代模型选择。

## 能写到什么程度

老师要求的基因侧直接跨研究预测已经完成，并保留了统计边界。可以写“跨研究任务
层面观察到平均改善和稳定的下界—误差关联”，不能写“跨研究预测显著优于简单
基线”或“disagreement 是普适置信度”。下一项计算实验应转向证书紧致度和复核
预算收益，继续堆相似数据集不会解决这个核心问题。
