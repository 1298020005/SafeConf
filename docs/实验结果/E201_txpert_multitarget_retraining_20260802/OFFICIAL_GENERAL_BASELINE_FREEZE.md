# E201 TxPert general baseline 预测前冻结

冻结日期：2026-08-02

## 目的

E201 不能只把 SafeConf 与 predicted magnitude 比较，也要保留 TxPert 公开代码中
的 `MeanBaseline`。四目标正式评价使用其单扰动分支的任务 centroid 等价实现，
不读取 target 扰动表达。

## 与公开 `MeanBaseline` 的对应

公开实现对每个 source cell line 分别计算：

`source perturbation expression - batch-matched source control`

然后按该扰动在各 source cell line 的训练细胞数加权，得到跨 source delta；最后
将这个 delta 加到每个 target 细胞的 batch-matched average control 上。

E201 已固定为单基因扰动，因此不涉及公开实现的 double-perturbation 或 unseen-
perturbation global fallback 分支。任务级等价预测固定为：

`target matched-control centroid + Σ(n_source_context × source_delta) / Σn_source_context`

`build_e201_official_general_baseline.py` 调用已经封存的 source-evidence 读取函数，
只从对应 target 的物理盲训练 H5AD 读取三个 source 背景。target 扰动行在这些文件
中为 0；target control centroid 来自零真值预测视图的封存 control 向量。

## 独立等价性检查

K562 已有 E200 通过 TxPert 公开 `MeanBaseline` 类实际生成的细胞级 general-
baseline prediction 和 batch-matched control。E201 程序只打开这两份预测前产物，
不打开 E200 truth，按 580 个相同任务聚合后检查：

`official prediction centroid - official control centroid`

与任务级加权 source delta 的最大绝对差。容差在 E201 target truth 打开前固定为
`5e-6`；580 个任务必须全部存在，prediction/control 行序和任务标签必须一致。
这项检查先以只读 preflight 独立运行并写入
`E201_OFFICIAL_GENERAL_BASELINE_PREFLIGHT.json`。正式 baseline 生成时重复整项计算，
最大残差和 RMS 残差必须与 preflight 在 `1e-15` 绝对容差内一致。

## 运行门

正式 baseline 只在以下文件已提交、干净且本地/Gitee/GitHub 实际远程哈希一致时
生成：

- 本冻结文档和 baseline 程序；
- 已封存的任务底表、source support、任务底表状态和 source-evidence 构建程序；
- 已封存的 E201 风险状态、风险表和 control centroid 向量。

输出前要求：

- 2,008 个任务、1,808 个主任务；
- 5,238 条 source-context support 与原底表逐项一致；
- target 扰动表达访问 0 行；
- target truth 尚未物化；
- K562 的 E200 公开代码等价性检查通过；
- weighted delta、general-baseline centroid 全部有限；
- 输出目录不存在，不覆盖已有产物。

## 输出

数据盘：

- `E201_OFFICIAL_GENERAL_BASELINE_WEIGHTED_DELTAS.npy`；
- `E201_OFFICIAL_GENERAL_BASELINE_CENTROIDS.npy`。

代码仓库：

- `E201_OFFICIAL_GENERAL_BASELINE_STATUS.json`；
- `E201_OFFICIAL_GENERAL_BASELINE_PREFLIGHT.json`；
- `tables/E201_OFFICIAL_GENERAL_BASELINE_SUPPORT_AUDIT.csv`；
- `tables/E201_OFFICIAL_GENERAL_BASELINE_ACCESS_AUDIT.csv`；
- `OFFICIAL_GENERAL_BASELINE_REPORT.md`。

所有数组只在状态中记录 DATA 相对路径、形状、字节数和 SHA-256，不进入 Git。
该状态必须双远程提交后，`release_e201_target_truth.py` 才允许打开 E201 target truth。
