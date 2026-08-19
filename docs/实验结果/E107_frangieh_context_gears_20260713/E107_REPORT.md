# E107｜Frangieh 同背景输入 GEARS 正式实验

三个外层 fold 各自重建共表达图，只读取源背景对照细胞；留出背景不参与该图。GO 图属于外部先验。任务输入为同背景 control mean 和扰动位点，验证集选择 checkpoint，测试标签在选择结束后才用于误差。逐 fold 图边、输入来源、训练曲线和结果位于 `folds/`。
