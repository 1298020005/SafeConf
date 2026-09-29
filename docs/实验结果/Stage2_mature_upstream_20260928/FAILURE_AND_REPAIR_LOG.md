# SafeConf v4 失败与修复日志

| 资产/环节 | 失败诊断 | 一次修复或替换 | 结果 | 决策 |
| --- | --- | --- | --- | --- |
| E208 Jiang24 | Latent/Linear 验证 MSE 分别为 no-change 的 53.31/32.91 倍 | E233 修复 LinearAdditive 的训练—推理 control 合同，并比较 softplus/linear | 最佳 linear 仍差 54.97%，0/12 context 不劣 | 停止该修复；测试真值继续封存 |
| E245 Jiang24 来源迁移 | 成熟上游缺失 | 用合法 train 历史做质量门控来源迁移 | 最佳仅改善 1.25%，未达 2% 门 | 不生成测试预测 |
| E216 资源模型 | 上游未超过 no-change | 结果完整保留 | SafeConf-M 外部确认失败 | 登记为 SEEN 负结果，不复用为 confirmation |
| E247 Kaden CRISPRa | GEARS 虽略胜 no-change，却弱于最强 train-mean-effect 基线 | 两 seed ensemble + 图特征阶段 | 相对最强简单基线差 1.70%，CI 稳定为负 | 测试表达继续封存 |
| E258 Feng | 学习型上游没有实质超过收缩历史均值 | residual MLP、历史距离、分半噪声诊断 | 最佳 MLP 只多 0.071%；历史增量可被支持度/噪声解释 | 测试供体真值继续封存 |
| V4 risk learner | 复杂模型可能过拟合 1,808 tasks | 同 split 比 Ridge/HGB/small MLP | Ridge 稳定最好，MLP 明显落后 | 不扩大型网络 |
| Quality evidence | 无合法 replicate/split-half 字段 | 不用 conflict proxy 冒充 Quality | eligible coverage 0 | 收缩主张为 Support/Relevance-aware |
| E190 GEARS V2 | 独立 family 仅 47 个 gene clusters，gate 在一折退回 prediction-only | 核验输入哈希、同尺度校准、inner/outer 隔离与 fold gate；未发现执行错误 | V1 点增量为正但区间宽；V2−V1 为负 | 保留为跨家族失败边界，不修改冻结候选 |
| E170 truth builder | 旧 E168 Git ancestry 在当前快照分支不可证明，构建器在读取 test X 前 fail-closed | 在揭盲前登记一次代码合同修复：改为核验冻结 RUN_STATUS、wrapper/helper 与 F2 manifest 的精确字节哈希 | 42GB 源哈希一致，四面板同开，未在修复时读取 truth | 合法修复；授权与修复均先提交远端 |
| E170 confirmation | 冻结 V2 需要一次性确认 | 所有 2,400 tasks、12 strata 完整运行，不删无历史列 | Gate A PASS；ΔU20=+0.0504，10/12 strata 非负；bootstrap CI 跨 0 | Route A 成立，同时保留统计不精确和 no-history 边界 |
