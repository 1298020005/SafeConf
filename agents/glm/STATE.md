# E201 评估链状态机（GLM 维护）

> 本文件是 E201 链的**唯一当前状态入口**。每完成或核验一步，立即更新。
> 规则：先核验、再推进；任何 FAIL → 停链并在"事件日志"记录。

## 当前阶段

**STAGE_3：零真值预测已启动**（Grok 2026-08-19 核查后接管）

> 训练 16/16 已于 2026-08-17 13:04:19 完成。family seal 于 2026-08-19 01:34:21
> 完成并双远程提交 `cd1779e`。GLM 值班 cron 未跑（runCount=0）。Grok 独立核查
> 见 `docs/实验结果/NEXT_PHASE_AFTER_GLM_20260819.md`。现启动 K562 seed 1
> 零真值预测。target 真值仍未打开。

| 项 | 值 |
|---|---|
| 最后更新 | 2026-08-19 01:40（Asia/Shanghai）Grok |
| 训练 | **16/16 完成**（2026-08-17 13:04:19） |
| 当前步骤 | STAGE_3 预测（K562 seed 1） |
| target 真值访问 | 0 行（保持到 STAGE_6） |
| 基线 commit | cd1779e（本地=origin=github 三方一致） |

## 阶段定义与判据

| 阶段 | 内容 | 进入条件 | 退出判据（人工核验） | 状态 |
|---|---|---|---|---|
| STAGE_0 | 等待 jurkat/seed_4 跑完 | — | 队列日志出现 `COMPLETE jurkat/seed_4`，16/16 全 COMPLETE | 完成 |
| STAGE_1 | family seal（封存 16 个 checkpoint） | STAGE_0 完成 | seal JSON/CSV 含 4×4 笛卡尔积、哈希齐全；脚本自检通过 | 完成 |
| STAGE_2 | 双远程提交 seal 与代码 | STAGE_1 核验通过 | GitHub、Gitee、本地 HEAD 均为 `cd1779e` | 完成 |
| STAGE_3 | 16 份零真值预测（每 target 先 seed_1 封共享文件，再 seed_2–4） | STAGE_2 完成 | 每个 target × seed 的 `predictions.npy` 存在且哈希入册；启动自检（commit/工作区/seal 哈希）全过 | 进行中 |
| STAGE_4 | 预测前风险特征 + official general baseline + E200 等价性检查 | STAGE_3 完成 | 2,008 任务风险状态 PASS；baseline centroid 封存；K562 580 任务 E200 等价性最大绝对残差 ≤ 5e-6 | 未开始 |
| STAGE_5 | 风险表 + baseline + 哈希双远程提交 | STAGE_4 核验通过 | 双远程 HEAD 一致 | 未开始 |
| STAGE_6 | 释放 target 真值（不可逆点） | STAGE_5 完成 | `release_e201_target_truth.py` 重算哈希通过后才打开 | 未开始 |
| STAGE_7 | 冻结正式评价（三门分判 + 四 target 全报） | STAGE_6 完成 | 证书门/路由门/magnitude 增量门各自出数；负结果保留 | 未开始 |
| STAGE_8 | 结果写入论文与报告 | STAGE_7 完成 | 论文 E201 占位符全部替换；四个 target 全报 | 未开始 |

## 事件日志（倒序追加）

- 2026-08-19 01:40 Grok 核查 GLM 产出：同意审核与翻转叙事；纠正 STATE 过时；
  确认 seal 已提交；启动 STAGE_3（`CUDA_VISIBLE_DEVICES=1`，K562 seed 1）。
  详见 `docs/实验结果/NEXT_PHASE_AFTER_GLM_20260819.md`。
- 2026-08-17 03:3x 科学批判批次（用户点名要的核心）：新增 `08_SCIENCE_CRITIQUE_AND_FIXES.md`
  （中心问题=固定分数必然顾此失彼：E199 增量 +0.299 vs E200 增量 −0.456；四弱点
  W1 组合稀释/W2 种子方差≠架构分歧/W3 足迹事后性/W4 效用增量功效不足；三层解决
  方案 + E203 预注册草案）。本轮新计算并入文：偏相关轴切换（E199: 0.386 vs
  0.005；E200: 0.875 vs 0.301）、组合排除实验（E200 任何组合稀释幅度）、九场景
  条件合同模拟（合同均值 0.232/下限 0；固定分下限 −0.127/−0.080；E192 双信号
  双双跨 0 → 弃用正确）。新图：Fig5/6/7（科学三图）+ Fig1–4 的 V2 变体，共 12
  个图形文件；双语正文新增"翻转由什么构成"小节与图 5–7 图注；图 7 视觉复核
  layout clean。全部新增数字来源与脚本注释可追溯。
- 2026-08-17 02:4x GLM 凌晨加班批次完成：
  ① runbook 六脚本规范命令全部写死（数据根=/home/yyf/data，盲视图/SHA/目录布局
  逐一核实自脚本源码）；② cron 提示词升级（长任务 nohup 跨班核验）；
  ③ **独立复算 E199/E200 全部头条数字并通过**（263/566 任务口径、五个相关系数
  4 位小数一致、恒等式残差 2.6e-18）；
  ④ 最小审计包 `code/safeconf_audit/`（pip 安装，12.6 秒 ALL PASS，13 项检查）；
  ⑤ **审核抓获标签错误**：E202a 与交接包把 E189 双未见 Spearman 区间
  （−0.349~−0.241）误标为 utility；E191 效用实为分歧 −0.127/幅度 −0.080；
  已修正双语正文、审核报告、图 1c/2/S1 并重新生成；
  ⑥ 补充图 S1（弃用台账：E192 ABSTAIN/E189+E191 负结果/E158 饱和）；
  ⑦ 两封 cover letter（Briefings 主 + Bioinformatics 备）；
  ⑧ 作者说明书 07（九点后发生什么）。ConfPert 全文复核：本环境无浏览器后端，
  留给作者人工（已写入 07）。用户授权默认：主投 Briefings；STAGE_6 按预注册
  协议自动推进。
- 2026-08-17 01:35 GLM 部署定时自动化 `automation-d53114c1-e577-43ae-bbfb-34575693a1f3`
  （每 45 分钟）：检查队列；16/16 后每次最多推进一个 runbook 阶段，逐步核验并回写
  本文件；任何失败 → BLOCKED；红线写死（STAGE_6 前不读 target 真值）。结果报告将
  写入 `05_E201_RESULTS.md`。
- 2026-08-17 01:30 论文 v1 完成：英文/中文 Nature 格式正文（E201 全部占位）+
  四张主图（300dpi 白底）+ 制图脚本，见 `paper/`。数字全部来自冻结 CSV。
- 2026-08-17 01:15 GLM 建立本状态机。E201 为 15/16；jurkat/seed_4 于 01:08:36 启动。
  队列监督 PID 1965937 正常；GPU1 12.3GB/68%。未触碰任何 target 真值。
- 2026-08-17 01:10 复核：hepg2/seed_4 COMPLETE（01:08:30）。历史单任务耗时
  11.5–13.5h，jurkat/seed_4 预计 13:00–14:30 完成。

## 已冻结判据速记（详见 04_E201_RUNBOOK.md 与 TARGET_RELEASE_AND_EVALUATION_FREEZE.md）

- 主分析任务：1,808 个 ≥30 细胞的 context–perturbation 任务（全部 2,008 任务另有敏感性层 200 个 10–29 细胞）。
- 路由门：`safeconf_e201_risk` 对 family RMS error 的 pooled Spearman CI 下限 > 0 **且** 20% review utility CI 下限 > 0。
- magnitude 增量门：控制 magnitude 的 partial Spearman CI 下限 > 0，**或** 配对 utility 增量 CI 下限 > 0。
- 证书门：`family_RMS² = centroid_RMSE² + disagreement²` 残差在数值容差内（这是等权 family 的恒等式，作完整性检查报告，不作科学发现）。
- bootstrap：按 perturbation condition 整簇 5,000 次；同一扰动跨 target 一起重采样。
- 四个 target 点估计与区间全部报告，不许删方向不一致者。
