# E180 元数据冻结报告

E180 已在不读取表达矩阵数值的条件下完成新研究冻结。

- 数据：XuCao2023，98,315 个细胞，59,429 个表达变量；
- control：2,758 个细胞；
- 合格靶点：153 个；
- 合格 guide 任务：411 个；
- 分区：conformal_calibration=29, model_validation=32, prospective_evaluation=27, supervised_train=65；
- 靶点选择使用表达真值或既往误差：否；
- X 数值读取：0；
- 主分组单位：基因；同一基因的 guide 不跨分区。

下一步必须先提交并推送本冻结文件，再构建 F2 资产。E179 的自适应上界规则已经写入 `MODEL_INPUT_LOCK.json`，后续不得依据 E180 评价结果改动。
