# 最终合同完整性审查

只读审查；只在本目录写表。没有运行模型、读取新的 sealed/test、修改其他目录或执行 Git。数值与文件绑定见 `ACTUAL_AUDIT.json`；`audit_snapshot.py` 可零拟合重建。

| 合同项 | 当前审查结论 | 文件证据与范围 |
|---|---|---|
| A 四臂、全部基因外折、三神经种子 | 已完成 | `full_v2_resume/TASK_PREDICTIONS.csv.gz`：Source 每个臂1808任务/575基因，McFaline 每个臂542任务/377基因；共同覆盖无删任务。 |
| A 全部臂同轴/同效应估计版本 | 已完成 | `run_safeconf_publicset_v1.py:load_domain`：McFaline 四臂统一 cell-weighted、完整2840轴；old guide 单列 `AlignmentControl_OldGuideMean`，不能当架构增益。 |
| A 训练成本及基线训练次数 | 已完成计数，最终总账待更新 | 三run 24+51+45=120唯一神经stage，60早停+60重拟合，原型24包含在120里。B0/强简单规则0fit；B1 CPU实际19次保存fit audit，10唯一fold范围，恢复作业重拟合也必须计成本。 |
| A 生物层统计 | 已完成，架构未通过 | `statistics_v1/RUN_STATUS.json`：5000共同gene抽样、种子指标均值、fold×context宏、60显式复制语义检查；STOP仅是生物层替换候选决定。 |
| A 合法 Source 教师与 Public 时序 | 静态/既有资格审查通过 | `risk_followup_v1/TEACHER_CONTEXT_PROVENANCE.csv` 核32旧teacher；新risk仅使用当前context、其他architecture的outer-train错误与CDF；Public outer/inner拟合/PCA排除相应query基因。合法其他context生物历史已知，不能称全pipeline未见研究。 |
| A 新固定HGB读出与信息包比较 | pending 产物 | `risk_followup_v1/registered_universal_v1/REGISTRATION.json`及新版代码修复Universal P；旧relative sparsity160fits保留。主统计统一20context×fold，pooled四context仅supp；静态复核通过，需等新版400fits与65088rows及5000统计。 |
| A Real/Support/物理内容置乱 | pending | 新信息包比较尚未最终完成，旧content null不能替代新构建器版本。需要合法fit-bank供体、排除当前parent、重算effect相关输入/冲突/权重/μ/dispersion/discrepancy，完整保存固定置乱次序。 |
| A 历史结构完整交付表 | pending 小补表 | `full_v2_resume/HISTORY_SET_AUDIT.csv`只有n_history与排除scope；须合并已有元数据/权重/参照结果，补每query去重实验单元、背景/来源、effect相同与可选择空间、有效权重来源/缺失。不能用细胞数冒充独立来源。 |
| B native512双向跨家族比较 | 已完成 SEEN 压力测试 | `frangieh_cross_family_v1/FIT_LEDGER.csv`：120总riskfits=60HGB+60Ridge，全部原始3fold×5riskfold及双向；原生512轴不补零。 |
| B 上游资格 | 已完成负结果 | `UPSTREAM_COMPETENCE.csv` 六模型全部FAIL（相对TRAIN-OOF所选均值基线误差差距8.13%–14.10%）；不能把已执行压力测试升为合格主验证，不要求为修复此资格另起上游训练。 |
| B 每失败的诊断修复 | 已完成限定修复 | Ridge clipping使外推排序塌缩；`RidgeScoreRepair`固定arctan单调输出修复，0新fit，保留原120模型及primary结果，六资格FAIL不变。 |
| C 反馈版本闭合 | pending 最终保留版本决定 | 旧曲线来自旧external_Learned/Public/Shared。附件§6允许hash完全相同版本复用；最终若保持原冻结Public与Shared则可复用；若接入newcell/newB1/NN/newrisk，受影响组合应同feedback/eval split及0/10/25/50/75/100%预算重评，未变化的Target-only等可复用。 |
| C 论文、表图、supp与repro | 初稿已生成，最终pending | PDF实际编译成功、旧22来源SHA一致、已展示57数字cell通过。仍把四臂称prototype running，未导入新A实际结果；最终需刷新结果、停止/保留范围、最多3主图及复现入口。 |
| C 数值验证回执 | 需小修 | 当前 `NUMERIC_CHECKS.json`仍 `latex_compilation=FAILED`，而 `BUILD_STATUS.json`是COMPILED；最终编译后重跑验证使回执一致。 |
| C 实际Git交付 | pending root核验 | 本审查未执行Git；baseline_commit不能替代最终实际提交。 |

目前没有发现可以据此停止整个合同的技术或预算障碍。生物四臂架构阴性、native512资格失败都是已完成的限定结果；最终方法决定须等待新版嵌套读出与信息对照，并绑定最终保留的 Public/risk 版本。该审查表不宣称整个目标成功。
