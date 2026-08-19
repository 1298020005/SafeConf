# SafeConf 当前任务

更新时间：2026-07-14

## 已完成

- [x] 六套正式 scGPT–GEARS 数据验证，30 folds、2,953 个测试任务
- [x] 5,906 条 strict PredictionRecord，合同问题数 0
- [x] E131 六数据集元分析
- [x] E132 六数据集分诊效用
- [x] E111 预测器专项机制审计
- [x] E114 保守 split-conformal 误差上界
- [x] E116 背景新颖度与生物机制审计
- [x] E117 紧上界失败记录
- [x] E118 chemical 正式边界
- [x] E126/E130 学习型路由器负结果
- [x] Tian 合同与门槛事前冻结后解封真值
- [x] 当前 gate 与投稿总账更新到 2026-07-14
- [x] 统一外部 Agent 学习入口与学习导航

## 当前最高优先级

### A. 统一论文正文

- [ ] 以 E131/E132 为主线建立新 manuscript outline
- [ ] 重写 Methods：六数据合同、正式 scGPT–GEARS、calibrated/frozen 定义和统计方法
- [ ] 重写 Results：主统计、异质性、分诊、机制、误差界和失败边界
- [ ] 重写 Discussion：实际收益大小、预测器依赖、chemical 和生物机制限制
- [ ] 建立每个正文数字到 CSV/report/commit 的映射表

### B. 投稿工程

- [ ] 统一主图、补图和表格编号
- [ ] 建立 paper 级一键复现 manifest
- [ ] 建立数据与代码可用性声明
- [ ] 进行独立统计和 claim 审核

## 条件启动

- [ ] 若冲更高门槛，优先寻找真正未见的新数据、合作或湿实验验证
- [ ] 任何新风险模型必须先冻结，再到第七个新数据确认

## 明确不做

- [x] 不在已经解封的六数据集上继续试路由器硬救
- [x] 不删除 Santinha、Shifrut、Tian 负 fold 或 chemical 负结果
- [x] 不把 fixed top-20% 小增益写成稳定节省大量湿实验
- [x] 不把 GEARS 较强信号写成所有预测器通用
- [x] 不把旧 Methods/Results 草稿直接当当前论文正文
- [x] 不把 Agent 原始意见、smoke 或 archive 当当前正式结论
