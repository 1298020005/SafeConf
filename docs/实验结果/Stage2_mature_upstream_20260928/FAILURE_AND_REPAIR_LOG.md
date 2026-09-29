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
| E190 GEARS V2 | 逐折定位显示混合异常：1/5 fold 校准异常、1/5 history branch 表现下降、2/5 gate misrouting；“47 clusters 导致 gate 普遍失稳”不成立 | 按预注册最小诊断比较 frozen affine、identity、正斜率 affine、isotonic；不修改冻结 v4 | raw `rPQ` pooled U20=0.3652，经 frozen fold-wise affine 后为0.0276；identity/isotonic 的 V2 点估计恢复至0.3113/0.3287，但相对 frozen 的95% CI均跨0 | 记录 cross-fold calibration 为主要可疑机制；因 gate misrouting 未达3/5 folds，停止 cluster 下采样；诊断变体不得替换冻结结果 |
| V2 learned gate 机制 | learned gate 是否真正优于简单固定混合尚未单独验证 | 保持完全相同的 calibrated `rP/rPQ`，比较 w=0/0.25/0.5/0.75/1 与 learned gate；DEV/SEEN-only，5000 次 gene-cluster bootstrap | learned−best-fixed：GAT +0.0015 `[-0.0154,+0.0211]`；Exphormer -0.0035 `[-0.0245,+0.0211]`；GEARS -0.0390 `[-0.2508,+0.1021]`；正确分支路由仅55%–60% | gate 暂不具备主创新证据；冻结 v4 保持不变用于预注册外部检验，论文主线优先落在合法历史证据的风险审计 |
| McFaline 上游预检 | 两个官方配置均在第一个 batch 前出现 DataLoader worker `Killed`；GPU 尚未执行，属于并行 worker 复制 HDF5/AnnData 的主机内存问题 | 唯一一次工程修复：`num_workers 12→0`；官方 batch=8000、模型、优化器、split 均不变 | LatentAdditive 193M 与 DecoderOnly 62.8M 均完成一轮真实 train/validation batch，GPU 24GB 可运行 | 两个预注册上游正式 validation 训练已启动；test truth 保持封存 |
| E170 truth builder | 旧 E168 Git ancestry 在当前快照分支不可证明，构建器在读取 test X 前 fail-closed | 在揭盲前登记一次代码合同修复：改为核验冻结 RUN_STATUS、wrapper/helper 与 F2 manifest 的精确字节哈希 | 42GB 源哈希一致，四面板同开，未在修复时读取 truth | 合法修复；授权与修复均先提交远端 |
| E170 confirmation | 冻结 V2 需要一次性确认 | 所有 2,400 tasks、12 strata 完整运行，不删无历史列 | Gate A PASS；ΔU20=+0.0504，10/12 strata 非负；bootstrap CI 跨 0 | Route A 成立，同时保留统计不精确和 no-history 边界 |
