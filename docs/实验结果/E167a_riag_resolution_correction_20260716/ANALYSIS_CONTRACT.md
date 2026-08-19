# E167a 分析合同｜RIAG v2 批次级分辨率修正

冻结日期：2026-07-16

## 1. 身份与边界

E167a 是 E167 v1 正式失败后的公开协议修正。它不覆盖 E167 的 `FAIL`，不属于独立外部确认，也不为 SafeConf 增加新的效果结论。全部分析只使用已解封历史资产；TianKampmann2019、DatlingerBock2021 和其他候选确认数据的表达矩阵保持不读。

E167 v1 把“预测向量不塌缩”和“风险分数具有高分辨率”写成同一个参考保留条件，并在研究层面合并不同 fold。正式结果显示，Replogle 和 Santinha 的预测向量正常变化，但若干实际排序批次的风险分数存在大规模并列。E167a 修正判定对象和状态语义，不把并列分数改成连续分数。

## 2. 冻结输入与批次

输入及 SHA-256 固定在 `SOURCE_LOCK.csv`。runner、合同和 source lock 必须先提交，正式运行只接受与当前 Git HEAD 完全一致的文件。历史资产已在一次性转换步骤中拆成：不含 loss 的 unit registry、任务分数和预测向量；单独的 postgate truth 与 viability 文件。转换前实际复核 E167 的 30 个来源哈希；转换脚本、构建证明、29 个预测数组与 8,380 条 `predictor row → task_id` 映射均留存。CSV 往返没有改变任何 `1e-6` operational label，预测矩阵逐元素完全一致。

`batch_id` 表示同一次实际排序队列和同一校准尺度：

- E153 八研究按原 `fold_id` 分批，共 34 个批次；
- Norman P1–P4、Wessels 各 seed、E87 和 E89 各自作为一个已登记批次；
- 不根据分数分布或真值重新拆分批次。

正式 runner 先一次读取并锁定 source-lock bytes，只验证 `PRETRUTH` 文件；实际计算也直接使用同一批已验证 bytes，避免 verify/read 时间窗口。随后生成 batch、predictor、coverage、unit、synthetic 和 pretruth gate 表并写入 `PRETRUTH_GATE_SNAPSHOT.json`。若 pretruth 回归失败，流程立即停止，两个 truth 文件保持未打开。通过时才按最初冻结的 source lock 读取并验证 `POSTGATE_TRUTH` bytes。完成前再次核验全部输入与快照哈希。证书函数的数据类型不含 loss 字段。历史损失只用于门后 tie-aware 指标审计，不得覆盖门状态。

## 3. 固定数值精度

沿用 E167 v1 的数值尺度：

\[
\delta_s=10^{-6},\qquad \delta_m=10^{-6},\qquad \delta_p=10^{-6}.
\]

候选风险、magnitude 和预测向量分别使用上述精度。量化采用 IEEE/NumPy 的 round-to-nearest、ties-to-even。`1e-12` 抖动不得制造新的有效等级。

`1e-6` 是本组已登记 provider 输出上的工程精度，不是跨方法通用理论常数，也不是单位变换不变量。`E167A_UNIT_REGISTRY.csv` 冻结每个候选的原始 transformation 和 numeric unit；正式运行禁止 rescale。以后接入新 provider 时必须在看见真值前登记其输出语义与有效精度。

## 4. G2：分数非退化与操作分辨率

### G2a｜数值非退化

每个 batch 必须满足：全部分数有限、量化后至少 2 个等级、原分数 population std 大于 `1e-6`。失败状态为 `ABSTAIN_SCORE_SATURATION`。

唯一值达到任务数一半不再作为硬门。至少两个可区分等级只保证候选分数一侧的秩方差非零；完整 Spearman 还要求门后 truth 非恒定。该条件也不保证 top-k 集合唯一。

### G2b｜coverage 边界

固定两类操作：接受最低风险 20% 的任务；优先复核最高风险 20% 的任务。风险—覆盖曲线沿用 coverage `0.20, 0.25, ..., 0.95, 1.00`，其中 `1.00` 只参与积分，不参与非平凡边界通过率。

对方向确定后的第 \(k=\lceil cn\rceil\) 个阈值，记严格越过阈值的任务数为 \(a\)，阈值并列组大小为 \(t\)，尚需位置为 \(m=k-a\)：

- `t == m`：集合唯一，`EXACT_SET`；
- `t > m`：并列组跨过 cutoff，`TIEBREAK_REQUIRED`；
- 可选集合数量以 \(\log_{10}\binom{t}{m}\) 报告。

同时报告连续秩分辨率：

\[
R=1-\frac{\sum_j(t_j^3-t_j)}{n^3-n},
\]

其中 \(t_j\) 是各分数等级的任务数。该量只作诊断，不设置事后通过线。

## 5. ties 下的 RC/AURC

普通稳定排序会让跨 cutoff 的并列任务按 CSV 行顺序进入集合。E167a 固定使用 tie-average：

\[
\operatorname{Risk}(k)=
\frac{\sum_{s_i<q}\ell_i+(m/t)\sum_{s_i=q}\ell_i}{k}.
\]

另报同一并列组中选择损失最小或最大的合法 AURC 上下界。候选风险与 magnitude 均按各自冻结精度计算 tie-aware RC/AURC，再报告差值。tie-average 与上下界必须在任意行排列下保持不变。真实部署如需唯一名单，必须在真值前登记次级规则或随机 seed；禁止加 jitter。

## 6. G3–G5

### G3a｜预测任务依赖

每个 predictor、每个 batch 分别满足：全部坐标有限、量化后至少 2 个预测向量、至少一个坐标的跨任务 population std 大于 `1e-6`。多预测器单元要求每个预测器均通过。

唯一预测向量比例、最大重复向量比例和向量分辨率继续报告，但不再用“必须过半”区分完全塌缩与低分辨率。

- `MODEL_UQ` 的 G3a 失败：`ABSTAIN_PREDICTOR_COLLAPSE`；
- `STRUCTURAL_RISK` 的 G3a 失败：只授权 `STRUCTURAL_RISK_ONLY`，不得称模型内生 uncertainty。

### G4｜重复稳定性

Wessels 三 seed 沿用 E167 的 Kendall W、两两 Spearman 中位数和 bootstrap 判定。失败不得被测试相关性覆盖。没有三重复时只能写 `G4_NOT_EVALUATED`，不能获得部署授权。

### G5｜magnitude 同序

G5 使用实际进入 G2b/RC 的量化 weak order：

\[
\operatorname{rank}(\tilde s)=\operatorname{rank}(\tilde m).
\]

只有 operational weak order 完全相同时才标记 `BASELINE_EQUIVALENT`。raw float weak order另作诊断，不能覆盖量化排序。这样避免“原始值正仿射、量化后的 ties 却不同”时错误宣称 AURC 等价。

### 门后 endpoint 与上游预测检查

loss 解封后必须检查 endpoint truth 至少有两个不同值；否则 Spearman 不可解释，状态为 `ENDPOINT_TRUTH_DEGENERATE`。已有简单基线比较时，另报预测器优于 simple/no-change 的任务比例；比例不超过 0.5 标记 `UPSTREAM_PREDICTOR_INVALID`。这些门后状态不能反向修改 G2–G5，但会阻止效果或部署结论。E87 是预留反例。

## 7. 批次状态顺序

1. G2a 失败：`ABSTAIN_SCORE_SATURATION`；
2. `MODEL_UQ` 的 G3a 失败：`ABSTAIN_PREDICTOR_COLLAPSE`；
3. `STRUCTURAL_RISK` 的 G3a 失败：`STRUCTURAL_RISK_ONLY_PREDICTOR_COLLAPSE`；
4. G4 已评价且失败：`ABSTAIN_UNSTABLE`；
5. G5 operational 同序：`EVALUABLE_BASELINE_EQUIVALENT`；
6. G4 未评价且 cutoff 跨 tie：`EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED`；
7. G4 未评价且 cutoff 唯一：`EVALUABLE_SELECTIVE_RANKING_G4_NOT_EVALUATED`；
8. G4 通过且 cutoff 跨 tie：`EVALUABLE_ASSOCIATION_TIE_AWARE_G4_PASSED`；
9. G4 通过且 cutoff 唯一：`EVALUABLE_SELECTIVE_RANKING_G4_PASSED`。

批次状态逐项保留。一个高分辨率 fold 不得覆盖另一个低分辨率 fold。E167a 全部历史资产的 `evidence_scope=RETROSPECTIVE_DEVELOPMENT`，`deployment_authorized=False`；`EVALUABLE` 只表示某类统计评价可继续，不表示效果通过或可部署。

## 8. 预设回归测试

E167a 的开发回归通过条件为：

1. E167 v1 的 `FAIL`、正式 HEAD 和结果 manifest 哈希保持锁定；
2. Norman P3/P4 official 固定为 `ABSTAIN_SCORE_SATURATION`，raw 与 Wessels 固定为 `ABSTAIN_PREDICTOR_COLLAPSE`，且 Wessels reason 保留 `G4_UNSTABLE`；
3. 所有历史非塌缩参考的预测批次通过 G3a；
4. exact constant、`1e-12` score/prediction jitter、prediction collapse 和 magnitude clone 得到预期判定；
5. 两个各自为常数但均值不同的批次不能被合并方差掩盖；
6. 20 个真实分数等级和 20 个预测表型不再因少于任务数一半被误判为完全塌缩；跨 cutoff 的 score ties 必须触发 `TIEBREAK_REQUIRED`；
7. 高分辨率和粗分辨率批次的状态保持分离；
8. tied 表格随机重排 100 次，17 个 coverage 的 tie-average RC、best/worst 边界和 AURC 在 `1e-12` 容差内不变；
9. magnitude operational 同序复制在全部 coverage 上得到相同 RC/AURC；raw 正仿射但量化 ties 不同的反例不得被判 G5 等价；
10. 非有限 score 输出 `ABSTAIN_SCORE_SATURATION`，不能在 G2b 量化时使整次运行崩溃；
11. pretruth 证书快照先于 postgate truth 文件验证；E87 在门后明确标为 `UPSTREAM_PREDICTOR_INVALID`；所有历史批次保持 `deployment_authorized=False`。

E167a 通过只说明修正规则内部一致。该规则随后必须原样冻结到新的外部确认任务，才可能形成确认性证据。
