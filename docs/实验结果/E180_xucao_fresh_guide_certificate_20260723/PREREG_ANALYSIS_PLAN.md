# E180 预注册分析计划：XuCao2023 guide 复现确认

## 研究问题

在一个未进入 E176–E179 正式结果的新研究中，scGPT 与 GEARS 的预测距离能否继续给出零违例的确定性误差下界；E179 冻结的向量特征上界能否在保持靶点级覆盖的同时，比常数 split conformal 更窄。

## 数据与任务

- 官方处理文件：`XuCao2023.h5ad`，98,315 个细胞、59,429 个表达变量。
- 仅纳入基因名同时存在于表达轴和 scGPT 词表、至少有 2 个 guide、且每个 guide 至少 20 个细胞的靶点。
- 主任务是一条 guide 相对 pooled control 的 512 维表达效应。
- 同一基因的全部 guide 始终留在同一分区；评价覆盖事件要求该基因所有 guide 同时被上界覆盖。
- `Cell_cycle_phase` 是扰动后观测标签，不能冒充预先给定的细胞背景；只做敏感性分析。

## 冻结分区

采用只依赖靶点名、细胞数和 guide 数的确定性哈希分层，比例为 40% supervised train、20% model validation、20% conformal calibration、20% prospective evaluation。表达矩阵和真值误差不参与选择。

## 模型和证书

1. scGPT 与 GEARS 使用五个固定随机种子；
2. 确定性证书：`pair_lower = RMSE(p_scGPT-p_GEARS)/2`；
3. 上界比较：常数、预测幅度、`max(预测幅度, pair_lower)`、E179 冻结的 ExtraTrees 向量特征基线；
4. ExtraTrees 只用 model-validation 靶点的预测特征和误差拟合；
5. conformal correction 只用 calibration 靶点，按基因簇最大残差校准；
6. evaluation 真值只允许在预测释放和校准模型同时推送到 GitHub/Gitee 后打开一次。

## 不允许的修改

评价真值打开后，不换靶点、guide、特征、模型、seed、阈值或 endpoint。若主结果失败，保留失败并进入新实验编号。
