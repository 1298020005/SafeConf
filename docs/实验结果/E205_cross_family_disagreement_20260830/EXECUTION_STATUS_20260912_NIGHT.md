# E205 夜间执行状态

记录时间：2026-09-12 21:55（Asia/Shanghai）

## 当前进度

- K562 / seed 1：80 轮完成；
- RPE1 / seed 1：80 轮完成；
- HepG2 / seed 1：GPU 0 训练中；
- Jurkat / seed 1：GPU 1 训练中；
- 其余 12 项：由同一监管器按固定顺序排队；
- 目标真实扰动结果：未授权读取。

单个 Exphormer 作业占用约 14–15 GB 显存。每张 Quadro RTX 6000 为 24 GB，因此当前一张卡一个作业、两卡共两个作业是安全的最大并行度；在同一卡增加第三项会产生显存不足风险。

## 21:10 队列停止的原因与处理

下午运行期间，仓库产生新的 E207/E206 提交，但新 HEAD 尚未同步到 GitHub 与 Gitee。训练适配器的版本门要求本地 HEAD 与两个远程跟踪引用完全一致，后续 14 项因此在进入数据训练前拒绝启动。该事件不是模型、CUDA 或数据失败，也没有产生可用的部分结果。

已将 E206 数值复核结果提交并同步到双远程，随后以第三次工程尝试恢复队列。K562 和 RPE1 的完整结果未覆盖；14 个启动前失败目录与日志保留。HepG2、Jurkat 启动后显存约为 14.8 GB 和 14.1 GB，GPU 利用率约为 80%。

## 证据入口

- 队列状态：`/home/yyf/data/txpert_official_20260802/e205/formal/E205_FORMAL_QUEUE_STATUS.json`；
- 监管器：tmux 会话 `safeconf_e205_formal`；
- 本轮日志：`.../_queue_logs/hepg2_seed1_attempt3.log` 与 `jurkat_seed1_attempt3.log`；
- 已完成状态：`.../K562/seed_1/E205_RUN_STATUS.json` 与 `.../RPE1/seed_1/E205_RUN_STATUS.json`。

绝对路径只用于服务器执行审计，不写入论文图表。

## 训练后第一道门

已新增 `tools/scripts/seal_e205_exphormer_checkpoint_family.py`。16 项全部完成后，该入口逐项检查：80 轮计数、模型结构、训练与验证行数、零目标真值访问、检查点 SHA-256、参数有限性和 16 项状态张量结构一致性。只有全部通过才生成 E205 checkpoint family seal；任何一项失败都会阻止后续预测。
