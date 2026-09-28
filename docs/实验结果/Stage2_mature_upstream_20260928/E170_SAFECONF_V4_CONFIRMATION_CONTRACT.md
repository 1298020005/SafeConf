# E170 SafeConf-v4 一次性确认合同

状态：**已冻结、尚未打开 test-donor targeting truth**。

本合同在 SafeConf-v4 Final Candidate 冻结后登记。E170 的四个 200-target 面板及其 scGPT/GEARS pretruth predictions 已于 2026-07-18 同时冻结；旧 E170 因 legacy seed-risk stability gate 只有 P03 通过而停止，四个面板的 test truth 均未读取。本次不继承旧 SafeConf score，也不只挑 P03；四个面板必须整体、一次性打开和报告。

## 1. 证据角色

- role：`SEALED_CONFIRMATION`；
- 确认层级：同一公开研究、同一 held-out donor、全新的 800 个 target genes；
- 任务：4 panels × 200 genes × 3 cell states = 2400 tasks；
- 独立 clusters：800 个 `panel_id::gene`；
- evaluation strata：4 panels × 3 states = 12；
- 不能声称：新 donor、新 study 或外部数据 confirmation。

旧 E168/E172 的已开封 target genes 不进入 E170；E170 target selection 是不读取 targeting X 的哈希选择。旧 E170 pretruth field distribution 可以在 Final Candidate 冻结后审计，但不得反向修改 V1/V2。

## 2. 上游能力门

上游固定为：

```text
ensemble_seed_family_mean
= mean(scGPT seeds 3407/3408/3409,
       GEARS seeds 3407/3408/3409)
```

能力门只使用 1920 个 validation tasks，不读 test donor truth。strongest simple baseline 固定为相同 gene×state 下两个 train donor effect 的均值。

必须同时满足 v4 competence rule：

- relative macro RMSE gap ≤ +2%；
- ≥60% panel×state validation strata non-inferior；
- target-cluster bootstrap 不支持稳定劣势。

四面板合并验证结果已登记为：relative gap `-0.0781519`，12/12 strata non-inferior，95% cluster-bootstrap CI `[-0.086189, -0.070394]`。因此允许进入确认；此判定没有使用 test truth 或 SafeConf confirmation 涨幅。

## 3. Prediction/Error Contract

- output contract：`E170_primary_CD4_direct_effect_512gene_v1`；
- prediction：冻结的 `ensemble_seed_family_mean` direct-effect vector；
- truth：每个 guide 先在 512-gene panel 上做 raw-count sum、library-size 1e4 scaling、`log1p`，减 matched donor×state NTC control，再对注册 guides 等权平均；
- gene order：每 panel 冻结的 512-gene `GENE_PANEL.csv` 顺序；
- label：task-level RMSE on aligned effect vector；
- 不跨 panel 混合 raw coordinates；只使用每条 task 的 panel 内标量特征和 RMSE。

## 4. History Eligibility

风险模型只用 validation-donor task errors 拟合。test-donor truth 不参加拟合、标准化、校准或 gate。

对 validation/test task，合法历史只包括同 panel、同 gene、同 cell state 的两个 train-donor effects：

- Support：可用 train donor/context 数；
- Relevance：prediction-source cosine 与 negative model-source RMSE gap；
- Conflict proxy：两个 train-donor effect 的 RMS dispersion；
- Content：train-donor mean effect magnitude；
- Missingness：column-unseen target 没有历史，明确编码为 missing；
- Quality：不进入冻结候选。

禁止使用 validation task 自身、test donor task、同一 test task guide、或 test truth 派生统计作为 history。

## 5. 冻结方法

不得修改：

- Universal P 六字段；
- V1=`Ridge(P+Support+Relevance), alpha=10`；
- V2 的 inner OOF 同尺度校准、单调 sigmoid evidence shrinkage；
- Support/Relevance/Conflict/Missingness 的系数方向；
- Utility@20、risk-coverage、AURC、miss-rate 和 bootstrap 规则。

风险模型 fitting set：四 panel 的 1920 个 validation tasks。inner cross-fitting group=`panel_id::gene`。confirmation query：2400 个 test tasks。column-unseen tasks 不删除，作为无历史 fallback 压力层完整报告。

## 6. 一次性评价

Primary comparison：`Frozen V2 vs Magnitude`。

同时完整报告：

- Magnitude；
- Universal P；
- Frozen V1（预指定 secondary benchmark）；
- Frozen V2（Final Candidate）；
- seen-history 160 targets/panel；
- column-unseen 40 targets/panel；
- 12 个 panel×state strata；
- 5000 次 panel-stratified target-cluster paired bootstrap。

Gate A 严格使用 `FINAL_METHOD_CONFIG.json` 中预注册阈值。confirmation 结果打开后，不允许在 V1/V2 之间换主方法，不允许修改字段或重新选择面板。

## 7. 失败解释

- V2 通过：支持同研究新 perturbation/held-out donor 层的冻结确认；仍需在论文中区分外部数据泛化。
- V1 正、V2 负：记录 V2 独立确认失败；V1 只能作为预指定 secondary benchmark，不能在同一 confirmation 上事后升为主方法。
- 两者均负：进入 Gate C 边界分析，重点检验 history availability、donor shift 与 legacy model disagreement 的失效条件；不得调参后重开同一结果。
