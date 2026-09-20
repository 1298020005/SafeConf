# E216 真值释放前封存记录

## 封存目的

E216 用独立公开数据 Jiang24 检查 SafeConf-M 能否在新的数据来源、细胞背景和上游预测器上继续排序高误差任务。为避免看过测试答案后再挑公式，本目录先封存全部任务分数、标准化参数和证书阈值；完成 Git 提交及双远程推送后，才允许读取测试扰动表达。

## 本次封存内容

- 正式任务：224 个，覆盖 12 个“细胞状态 × 刺激条件”组合。
- 上游预测成员：4 个 LatentAdditive 随机种子和 1 个 LinearAdditive 基线。
- 跨结构加权：LatentAdditive 家族合计 0.5，LinearAdditive 合计 0.5，避免四个同结构随机种子在数量上压过单个线性结构。
- 已冻结输出：预测幅度、SafeConf、固定 4:1 组合 SafeConf-M、跨结构预测下界，以及 9 个预设证书阈值。
- 测试扰动真值读取量：0 行。
- 目标真值参与特征、权重或阈值选择：否。

## 可审计文件

| 文件 | 作用 |
|---|---|
| `E216_PRETRUTH_STATUS.json` | 任务数、成员、权重和真值访问状态 |
| `E216_PRETRUTH_TASK_SCORES.csv` | 224 个任务的全部真值无关分数 |
| `E216_PRETRUTH_STATE_STANDARDIZATION.csv` | 12 个状态内的中心与尺度参数 |
| `E216_CERTIFICATE_THRESHOLDS.csv` | 测试前冻结的 9 个证书阈值 |
| `E216_PRETRUTH_MANIFEST.csv` | 输入、预测及产物的逐文件 SHA-256 |

用于正式评价的跨结构完整预测张量因体积较大，不重复写入 Git；原始文件保存在服务器：

`/home/yyf/data/perturbench_e216/pretruth_20260920/E216_ARCHITECTURE_BALANCED_FAMILY.npz`

- 文件大小：13,934,979 字节。
- SHA-256：`c00f9d5ffaefa0281ae04e6c636e30fa5794b0802183420da7346f518eb5bb29`。

## 下一道门

本目录提交并同时推送到 GitHub、Gitee 后，另建“测试真值释放授权”提交，逐项引用本次封存提交与文件哈希。正式评价只能按预注册脚本执行，不能根据结果修改任务、权重、指标或证书阈值。
