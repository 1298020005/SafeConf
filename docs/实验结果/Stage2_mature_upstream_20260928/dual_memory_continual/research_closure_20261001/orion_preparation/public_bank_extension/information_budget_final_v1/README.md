# Frozen Orion method information budget

13行追加表对应同一232条common任务、144个生物gene clusters（HCT116107、HEK293T125）。六个Source监督风险方法各继承3616条prediction-error records、1808个生物tasks、575个gene clusters、2个predictors但仅1个TxPert graph family；本阶段未增加拟合。各simple/history规则不使用Source prediction-error风险监督；Learned reference规则继承Source Public biology模型（4737个biological-transfer pairs），不能将这种生物学监督记为prediction errors或C labels。Support-only只使用预测时的历史Source bank支持。

所有13方法的C risk/feedback/CDF、C validation biology及C upstream calibration列均为0。共享upstream过程已使用2790个C validation errors、1661个gene clusters做固定competence gate，且只计一次。它们未用于风险训练、CDF、Public learner、output calibration或模型选择；现有ledger无competence字段，因此单独写在AUDIT.json，不误填calibration列。此处不能声称零C信息，也不是SafeConf效果PASS。

四个主输入B74/C8/47c76/b89逐SHA绑定。Source历史为原已知TRAIN eligibility，不宣称risk outer-train-only；Source风险错误在训练参数中编码，推理不逐条查源错误。只准备追加表和列义审计，不编辑共享ledger，也未读取predictions/risk/原始表达/TEST truth。
