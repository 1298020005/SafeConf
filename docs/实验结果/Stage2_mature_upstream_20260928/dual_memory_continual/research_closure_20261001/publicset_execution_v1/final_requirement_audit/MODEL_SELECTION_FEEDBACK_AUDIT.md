# 当前模型取舍与反馈复用的独立审查

仅消费已完成Source统计、保存bootstrap和旧反馈身份/信息账本；不运行模型、不打开MC TEST数组、不做论文/PDF工作。精确SHA、22门限及复算误差见 `MODEL_SELECTION_FEEDBACK_AUDIT.json`，全门限附 `ACTUAL_MAIN_RISK_GATES.csv`。

新版 `risk_followup_v1/registered_universal_v1` 已400真实HGB fits、65088预测行、5000共同gene draws。保存observed与seed-mean macro最大差3.34e−16，全部gate点差2.02e−16、CI9.72e−17以内；两个方向使用完全相同5000×575的基因次数。150份神经prior/model/weights与query合同完整性回执PASS。基线广播至三个seed槽不增加真实fit数。

| 采用问题 | Exphormer→GAT：ΔU20 [95% CI] | GAT→Exphormer：ΔU20 [95% CI] | 判断 |
|---|---|---|---|
| B2−B1 | +0.020529 [−0.010745,0.047469] | −0.006273 [−0.028400,0.029345] | 两门均FAIL；不支持B2整体替换 |
| B3−B2 | +0.001219 [−0.014924,0.017265] | +0.005545 [−0.012496,0.021195] | 非负分层50%/45%，两结构门FAIL；不扩DeepSets |
| B2−P-only | +0.042697 [0.001166,0.082781] | +0.035990 [0.006762,0.083117] | 两实用门PASS，支持当前完整Public包的限定增量 |
| B2−P+纯Support | +0.048719 [0.007996,0.084718] | +0.021183 [−0.000566,0.073290] | 两登记实用门PASS，但第二向名义CI仍含0 |
| B1−P+纯Support | +0.028189 [−0.012815,0.070236] | +0.027456 [−0.004890,0.076443] | 仅第二向通过；不能称B1内容增量两向稳定成立 |

已有fixed-reference统计也不支持全局替换：B2/B3均未跨所有Source/MC强基线门，B3−B2结构门失败。保留B1是预登记的incumbent策略，**不证明B1显著最优或神经方案无效**。B2−B1改变了representation和生物训练目标；只有B3−B2控制神经训练目标而改变set context。当前null只有B1的600/1200风险fit，完整生物内容机制结论仍pending。不能用Source-only新reader推导新MC迁移/反馈收益。

## 旧反馈的确定版本

- 缓存：`/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache/external_Learned.parquet`，SHA `fa1e373edc04cb18bce0b4a993c57877643c33fe623ec1dc75973f44062c8988`；543任务/380基因，冻结alpha025 predictor、2840输出合同。当前Source新reader不改这份外部缓存。
- 原Public来源：`tools/scripts/build_safeconf_common_gene_biology.py` 中 `public_effects.append(mean(guides))`；`tools/scripts/run_safeconf_common_axis_closure.py:target_features` 用 `common_gene_axis/public_mcfaline_trainval/effect_vectors.npy` 加 `public_priors` 生成旧guide参照。该bank的effect SHA是 `36212d51411fbe01234bbe82e1fb938a84b6f569ab540b64a30cab9a26cc67b8`。
- 反馈脚本：`tools/scripts/run_safeconf_research_closure.py:run_feedback`；shared从 `common_gene_axis/results/MATRIX_TASK_PREDICTIONS.csv.gz` 的 `TxPert_to_McFaline/Learned_hgb/20260930` 取值。
- 固定分割：`common_gene_axis/results/FEEDBACK_SPLIT_CONTRACT.json`；331记录/228基因反馈池，212记录/152基因永久评价池。10/25/50/75/100%使用37/87/164/248/331记录，23/57/114/171/228基因；所有budget CDF、Residual内部CDF/kappa等消耗同一允许记录并集。
- `STRICT_FEEDBACK_INFORMATION_LEDGER.csv` 明确各预算0新增validation risk-CDF错误，但upstream校准仍有542validation记录、Public生物训练仍有542validation记录。0%仅表示零feedback-pool错误，不能称零目标研究信息。

## 若最终改为新的cell-support prior

必须冻结新Public版本，并在**原543任务**构建新内容相关PUBLIC字段，再在同一331/212池评价。Public变化包括prior magnitude、prediction-prior RMSE/cosine、dispersion、history conflict；如同时改聚合权重则effective sources也变。metadata支持数与P保持原定义时可保持相同。

| 已有组合 | 新μ/Public改变时的处理 |
|---|---|
| TargetOnly_Ridge/HGB、PertEMA_P_adapted | P、标签、分割、seed均相同可复用，无须重fit |
| NativeControl_HGB、PertEMA_Native_adapted | Native/P字段及合同相同可复用 |
| PublicTarget_HGB | 同0/10/25/50/75/100%预算重评发生变化的组合；正预算同原行重fit |
| SharedTarget_HGB、ResidualHGB | 新Public字段即需重fit；Residual合法内部CDF/kappa随新版本重新执行，所有错误仍计原预算并集 |
| PertEMA_Public/Shared、PertEMA_Native_Public/Shared | 重fit受影响Public组合；原P/native-only可复用 |
| Shared | 仅固定共享map与其输入/PUBLIC/输出SHA全不变时可复用；若新μ进入shared map，必须先封存新score再重评；不能给旧score换新标签 |
| 无反馈参照与support强规则 | μ相关direct/weighted rules需重算；support/P-only本身可复用；一律在相同212任务比较，不能将543/542主表与212反馈表相减 |

PertEMA入口为 `tools/scripts/run_safeconf_official_pertema_feedback.py`，支持显式 `--runtime-root`、`--result-root` 新版本；native Public字段在新cache的 `native_control_reference/TASK_FEATURES.parquet` 同步更新，原native列可保留。官方适配权威配对统计是 `PAIRED_BOOTSTRAP_REPAIRED.csv`，原float-error join统计不能复用。

`tools/scripts/run_safeconf_common_axis_closure.py --phase feedback` 可消费冻结cache，但**禁止在旧默认结果目录重跑**：完成保护按phase文件，旧all完成不一定挡住feedback覆盖。需要新版本时同时给新 `--result-root` / `--risk-cache-root`，绑定必要冻结输入；不运行all/fit/evaluate，这些分支会打开MC TEST数组。`run_safeconf_reference_estimand_diagnostic.py` 与 `run_safeconf_estimand_risk_replay.py` 也会加载TEST预测/对照数组，当前只读审查不调用它们。

最终可复用旧反馈的条件是明确保留旧guide/Public/Shared全版本；采用新cell-support方案就不能声称旧反馈已经验收该方案。用户已延后论文/PDF，本审查只给实验取舍和版本合同。
