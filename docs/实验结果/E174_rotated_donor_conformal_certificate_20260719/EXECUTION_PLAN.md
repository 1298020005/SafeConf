# E174 execution plan

1. `F1/F1B` 已冻结 800 个新靶点、轮换供体角色、20%/80% 分区和旧数据方法回退；此时 E174 expression X 读取数为 0。
2. 用 base Python 3.12 分别构建 R01–R04 `F2_pretruth`；只允许 control、train seen 与 validation seen 行。
3. 用 `scgpt_env` 在两块 GPU 上运行四面板、两个模型族、三个 seed；四个 RIAG snapshot 均 PASS 后整体提交 GitHub 与 Gitee。
4. 以该 pretruth gate commit 构建四个 `F3A_calibration`，每面板只读 240 个预注册校准 guide rows；最终评价读取数保持 0。
5. 运行 joint calibration，固定 160-target cluster conformal quantile；提交 calibration snapshot 到两个远程。
6. 以 calibration gate commit 构建四个 `F4_evaluation`，每面板只读剩余 960 个 evaluation guide rows。
7. 运行一次 final evaluator；640 个目标为唯一主要总体。输出无论正负都完整提交，不重算分位数、不换模型、不删面板。

每次 truth access 前都必须验证当前代码已提交、当前分支同时存在于 GitHub/Gitee，且上一步 snapshot 的本地字节与 gate commit 完全一致。
