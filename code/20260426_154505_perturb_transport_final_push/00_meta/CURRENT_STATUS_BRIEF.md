# 2026-05-20 13:09 方法修改记录

- 问题：`V0` 强基线很稳，直接深度残差不容易全面超过。
- 已做代码修改：
  - 深度模型训练目标加入 effect-aware loss：cosine、rank、top-effect sign loss。
  - blend 选择从 MSE 改成 effect objective，直接优化 top20 / DEG / program consistency 相关目标。
  - 新增 `EffectBlendV2`：用内层验证集决定 `V0` 和 `V2` 的混合比例。
  - 新增 `TopRankGraftV2`：保留 `V0` 的整体稳定性，只把 `V2` 的 top gene ranking 注入到关键基因上。
- 快速 smoke：
  - `47_top_rank_graft_smoke_fast_20260520`
  - 在 `KaggleCrossCell` 小任务上，`EffectBlendV2` 相对 `V0` 提升了 Pearson / Spearman / RMSE / top20 / DEG。
- 新后台：
  - `gpu_graft_tian_20260520` 正在跑新版 `TopRankGraftV2` 外部验证，输出到 `48_gpu_graft_tian_20260520/`。
  - `gpu_graft_main_20260520` 正在跑新版 main 批次，输出到 `43_gpu_effect_objective_main_20260520/`。

# 2026-05-20 12:30 状态更新

- 昨晚 `30/31/34/35/36` 批次已正常结束，tmux 旧会话已退出。
- GPU 深度对照已产出：
  - `35_gpu_deep_gpu0_20260519`: `TianKampmann2019`, `n_rows=7164`
  - `36_gpu_deep_gpu1_20260519`: `KaggleCrossCell/Haber/Parekh`, `n_rows=840`
- 当前结论：
  - `V0` 强基线仍然最硬，深度残差模型没有全面超过它。
  - `PolicySafeTransPT` 相对 `V2` 有稳定小提升，但相对 `V0` 主要是接近或持平。
  - 这说明方向有证据，但还不能包装成“全面胜出”的最终稿。
- 已追加代码改动：
  - `run_deep_gpu_transport.py` 新增 `DeepCalibratedSafeTransport`
  - 逻辑：验证集判断残差 transport 是否真的有用；没把握就回退 `V0`，防止 unsafe transport 拖坏结果。
- 新后台任务已启动：
  - `gpu_calibrated_main_20260520` -> `39_gpu_calibrated_main_20260520/logs/run_gpu_calibrated_main.log`
  - `gpu_calibrated_tian_20260520` -> `40_gpu_calibrated_tian_20260520/logs/run_gpu_calibrated_tian.log`

# 2026-05-19 GPU 新进展

- `run_deep_gpu_transport.py` 已补上缺失参数 `--v2-blend`，并增加进度打印。
- smoke 已跑通：`37_gpu_deep_smoke_fix_20260519/`，`KaggleCrossCell`，`seed=11`，`n_rows=52`。
- 正式后台任务已启动：
  - `gpu_deep_tian_20260519` -> `35_gpu_deep_gpu0_20260519/`
  - `gpu_deep_main_20260519` -> `36_gpu_deep_gpu1_20260519/`
- 当前观察：
  - GPU1 已开始占用显存，进入训练段。
  - GPU0 还在前处理，但进程正常，没有挂死。

# 当前状态速记

时间：2026-05-19 11:43 CST

## 已完成的一批

- 方向：safe cross-context perturbation transport
- 已跑通的批次：`29_policy_router_refresh_20260519`
- 数据集：
  - main: `KaggleCrossCell`, `kangCrossCell`, `Haber`
  - external: `KaggleCrossPatient`, `McFarland`
- 模型：
  - `V0`
  - `V2`
  - `SafeTransPT`
  - `SafeTransPT_no_abstain`
  - `SafeTransPT_no_pathway`
  - `NetworkSafeTransPT`
  - `PolicySafeTransPT`

## 当前最有用的信号

- `PolicySafeTransPT` 在这批里是最均衡的一个：
  - Pearson / Spearman 和 `V0` 接近
  - program consistency 比 `V0` 和 `V2` 更好
  - risk-coverage 里 top20 / DEG 也比传统 safe 版本稳

### 平均指标

- `V0`
  - Pearson 0.4220
  - Spearman 0.2909
  - RMSE 0.0864
  - top20 0.3203
  - DEG 0.3844
  - program consistency 0.6109

- `PolicySafeTransPT`
  - Pearson 0.4219
  - Spearman 0.2901
  - RMSE 0.0859
  - top20 0.3156
  - DEG 0.3848
  - program consistency 0.5824

## 现在正在跑的三条线

- `policy_router_wide_20260519`
  - 更大范围 main + external
  - 目标：把证据面铺宽

- `policy_full_20260519`
  - full summary / external validation
  - 目标：生成明天直接可讲的总表

- `policy_router_soft_20260519`
- `policy_router_hybrid_20260519`
  - 路由方式对照
  - 目标：看 hard / soft / hybrid 哪个更适合

- `policy_tian_ext_20260519`
  - 外部验证专线
  - 目标：把 `TianKampmann2019` 作为更有分量的外部证据拉进来

## 明天汇报可用的句子

“我们不是只在做一个扰动预测器，而是在做一个安全迁移决策框架：先检索，再路由，再拒判。现在已有结果显示，这条线至少能把一些程序一致性和风险控制做出来，后面会继续用更宽的数据集和路由对照把它夯实。”
