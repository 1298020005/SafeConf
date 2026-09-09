# SafeConf 当前权威状态

更新时间：2026-09-09
当前分支：`exp/task-risk-audit-20260611`

本文件只做当前事实摘要。科研数字以正式 CSV/JSON 为准；运行状态以数据盘状态文件和进程检查为准；Agent 教材与聊天均不能覆盖它们。

## 1. 当前研究分成两条业务

### 业务 A：预测后风险审计

TxPert 已经产生扰动预测后，SafeConf 尝试在不知道真实答案时，把更容易出错的任务排到前面。

E201 已完成四个细胞系、四个随机种子的整背景留出评价：

- 主任务 1,808，敏感性任务 200；
- SafeConf 与 family RMS error 的合并 Spearman：`0.4082 [0.3506, 0.4621]`；
- 预测幅度：`0.6189`，是更强的单一排序器；
- 控制预测幅度后的偏 Spearman：`0.2503 [0.2021, 0.2980]`；
- 20% 复核效用：SafeConf `0.3200`，幅度 `0.5943`，差值 `-0.2743`。

因此能说“SafeConf 有幅度之外的关联”，不能说“SafeConf 优于幅度”，也不能把“幅度主排序、SafeConf 补充”写成已经验证的用法。

证据：

```text
docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md
docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/tables/
docs/学习导航/20260906_论文审核与从零教学/CLAIM_TABLE.json
```

### 业务 B：训练时加权

E204 用 source-only difficulty（只从训练侧支持度、背景覆盖和效应离散度得到的难度）调整训练权重。

- 四个 target 的一轮工程 profile 已通过；
- 32 个正式训练项已经登记：16 个 `risk_weighted`、16 个 `dispersion_only`；
- 2026-09-09 服务器复核时，两张 GPU 被其他作业占用，32 项均尚未启动；
- 当前没有 80 轮效果结果，不能说加权训练让模型变准。

证据：

```text
docs/实验结果/E204_risk_guided_training_20260830/ANALYSIS_FREEZE.md
docs/实验结果/E204_risk_guided_training_20260830/PROFILE_ACCEPTANCE_20260905.md
/home/yyf/data/txpert_official_20260802/e204/formal/E204_QUEUE_STATUS.json
/home/yyf/data/txpert_official_20260802/e204/formal/E204_QUEUE_SUPERVISOR.log
```

## 2. 必须保留的边界

- SafeConf 单独排序弱于预测幅度；
- K562 上 TxPert 四种子质心误差 `0.0540`，差于 batch-matched control 的 `0.0523`；
- RPE1 的家族分歧相关为 `0.0397`，区间跨 0；
- E201 确定性证书门因 `3.61e-10 > 1e-10` 保持 FAIL，下界违反数为 0；
- E202 主检验失败；
- E201 四个预测成员是同一 TxPert-GAT 架构的四个随机种子，不是四种模型；
- E205 只有冻结协议，没有跨架构正式结果；
- 没有冻结后的全新外部确认，不能把既有开发数据重新包装成新外部验证。

## 3. Git 与协作状态

- Kimi-K3 提交 `b6991c1` 已进入服务器、GitHub 和 Gitee 同名实验分支；
- `agents/kimi/` 的手工副本与教学包报告逐字相同，已并入 `agents/kimi-k3` 的索引并去除重复；
- Kimi 初稿所写“远程停在六月、无共同祖先、27 个提交未推送”已经作废；
- Windows 本地是否同步无法从服务器独立证明，只能确认相同提交已进入服务器和双远程。

## 4. 当前事实优先级

```text
正式 CSV / JSON / 运行状态文件
  > 当前实验正式报告与冻结协议
  > START_HERE_FOR_AGENTS.md 和本 STATE
  > 学习材料
  > Agent 审核意见
  > 聊天转述
```
