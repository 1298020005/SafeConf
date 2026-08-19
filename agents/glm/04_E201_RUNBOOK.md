# E201 评估链逐步执行手册（GLM，2026-08-17）

依据：`docs/实验结果/E201_txpert_multitarget_retraining_20260802/TARGET_RELEASE_AND_EVALUATION_FREEZE.md`
（2026-08-02 冻结）+ `EXECUTION_CHECKPOINT_20260811.md` §6 + 脚本本体。
**总原则：一步一核验，核验不过不进下一步；每步完成后立即更新 `STATE.md` 事件日志。**

公共环境（**规范值已由 GLM 于 2026-08-17 逐脚本核实并固定**）：

```bash
PY=/home/yyf/.venvs/txpert-08d82eea/bin/python
DATA_ROOT=/home/yyf/data                 # 状态文件里 "DATA/..." 前缀相对它解析
RUNS=/home/yyf/data/txpert_official_20260802/e201/formal
TXPERT=/home/yyf/archive/external/TxPert
SEAL_JSON=/home/yyf/proj/docs/实验结果/E201_txpert_multitarget_retraining_20260802/E201_FAMILY_SEAL.json
SEAL_CSV=/home/yyf/proj/docs/实验结果/E201_txpert_multitarget_retraining_20260802/tables/E201_FAMILY_SEAL.csv
PRED_ROOT=$RUNS/predictions                          # 布局: $PRED_ROOT/<target>/shared 与 /seed_<s>
PRETRUTH_VEC=/home/yyf/data/txpert_official_20260802/e201/pretruth_vectors   # 运行前必须不存在
EVAL_VEC=/home/yyf/data/txpert_official_20260802/e201/evaluation_vectors     # 运行前必须不存在
cd /home/yyf/proj && git status --short --branch   # 必须干净
```

已核实的脚本内建校验（供核验时对照）：
- seal 脚本内嵌 4 target 的期望行数/train_batches/val_batches/盲视图 SHA-256、
  TxPert commit `08d82eea…`、adapter SHA `27434447…`；
- 预测脚本内嵌盲 H5AD SHA `85f93d1b…`（位于 $TXPERT/cache/E201_prediction_blind/
  de_adata_test.h5ad，140,792,831 字节）与 manifest SHA；
- 风险脚本期望 $PRED_ROOT/<target>/shared/E201_SHARED_TARGET_MANIFEST.json、
  controls.npy、observations.csv，seed 目录内 E201_PREDICTION_RUN.json 记录
  family_seal_sha256 与 shared manifest SHA 链；
- baseline/评价脚本共享向量文件名 E201_SEED/FAMILY/CONTROL/SOURCE_TRANSFER/
  OFFICIAL_GENERAL_BASELINE_CENTROIDS.npy；评价脚本自校验
  `--self-test` 可在任何时刻安全运行（不打开结果）。

双远程一致性检查（每个提交点都要跑）：

```bash
git fetch origin github 2>/dev/null || git fetch origin && git fetch github
git rev-parse HEAD origin/exp/task-risk-audit-20260611 github/exp/task-risk-audit-20260611
# 三个哈希必须完全一致
```

---

## STAGE_1 family seal（封存 16 个 checkpoint）

前置：STAGE_0 完成——队列日志出现 `COMPLETE jurkat/seed_4`，16/16 全 COMPLETE，
监督进程退出，GPU1 释放（`nvidia-smi` 确认无 E201 残留进程）。

```bash
$PY tools/scripts/seal_e201_txpert_checkpoint_family.py \
  --runs-root $RUNS --data-root $DATA_ROOT \
  --output-json $SEAL_JSON --output-csv $SEAL_CSV
```

人工核验清单：
- [ ] seal JSON 含 4 target × 4 seed 笛卡尔积（16 项），无缺
- [ ] 每项含 last.ckpt 与 best 的 SHA-256、epoch=80、step 数
- [ ] `target_perturbed_cells_accessed` 全 0
- [ ] 脚本自检（行数/batch 数/有限值/参数结构）通过，无 WARN
- [ ] 失败即停：任何条件不符 → 停链，记录，勿手工"修复"seal

## STAGE_2 seal + 代码双远程提交

- [ ] `git add` seal JSON/CSV 及相关文档；commit 信息风格沿用仓库惯例（`e201: seal 16-checkpoint family`）
- [ ] push 到 origin（Gitee）与 github 两端；三方哈希一致
- [ ] 大文件检查：seal 产物应只是小 JSON/CSV；`git count-objects -vH` 无异常增长

## STAGE_3 十六份零真值预测

顺序（冻结）：每个 target 先 seed_1（写共享 `controls.npy`/`observations.csv`），
再 seed_2–4（复核共享文件哈希后只写各自 `predictions.npy`）。batch-size 16。

```bash
for T in K562 RPE1 hepg2 jurkat; do
  $PY tools/scripts/run_e201_txpert_sealed_prediction.py \
    --txpert-repo $TXPERT --data-root $DATA_ROOT \
    --family-seal $SEAL_JSON \
    --target $T --seed 1 \
    --output-dir $PRED_ROOT/$T/seed_1 --shared-dir $PRED_ROOT/$T/shared --batch-size 16
  for S in 2 3 4; do
    $PY tools/scripts/run_e201_txpert_sealed_prediction.py \
      --txpert-repo $TXPERT --data-root $DATA_ROOT \
      --family-seal $SEAL_JSON \
      --target $T --seed $S \
      --output-dir $PRED_ROOT/$T/seed_$S --shared-dir $PRED_ROOT/$T/shared --batch-size 16
  done
done
```

注意：output-dir 已存在时脚本会拒绝（防覆盖）；必须先 seed_1 后 seed_2–4。

人工核验（每个 target×seed 跑完即查，不是最后一起查）：
- [ ] 启动自检全过：TxPert commit 固定、SafeConf 双远程一致、seal 哈希一致、
      盲 H5AD SHA-256 一致
- [ ] 首 batch 的 0→1 dummy X 不变性测试通过（脚本内建）
- [ ] 每个 batch `batch.x` 非零数 = 0 的审计行存在
- [ ] 目录中出现且仅出现预期的 `predictions.npy`（无 `truth.npy`！）
- [ ] 显存/磁盘正常；GPU1 独占，无并发训练

## STAGE_4 预测前风险特征 + general baseline + E200 等价性

```bash
$PY tools/scripts/run_e201_pretruth_risk_features.py \
  --data-root $DATA_ROOT --family-seal $SEAL_JSON \
  --prediction-root $PRED_ROOT --vector-output-dir $PRETRUTH_VEC

$PY tools/scripts/build_e201_official_general_baseline.py \
  --data-root $DATA_ROOT --vector-output-dir $PRETRUTH_VEC
```

（$PRETRUTH_VEC 运行前必须不存在；输出表/状态固定写在
docs/实验结果/E201_*/tables/E201_PRETRUTH_RISK_FEATURES.csv 与对应 JSON。）

已知事实（供比对）：风险主分数 = 5 个 z 分量等权平均
（family_disagreement、model_source_gap、source_delta_dispersion、
negative_log_source_cells、support_context_deficit）；predicted_magnitude **不进入**
风险分数，保留为强基线。preflight 时 E200 等价性最大绝对残差 2.7865171e-6
（容差 5e-6），正式运行应同量级。

人工核验：
- [ ] 2,008 任务风险状态全 PASS；目标表达访问仍 0
- [ ] 2,008 个 baseline centroid 封存；5,238 条 source support 审计
- [ ] K562 580 任务 E200 等价性 max|residual| ≤ 5e-6
- [ ] 风险分量标准化参数只用主任务估计（脚本自检）

## STAGE_5 风险表 + baseline + 哈希双远程提交

同 STAGE_2 流程；这是**开真值前的最后承诺点**。全部核验记录写入 STATE.md。

## STAGE_6 释放 target 真值（不可逆点）

前置全部满足后才运行：

```bash
$PY tools/scripts/release_e201_target_truth.py \
  --data-root $DATA_ROOT --txpert-repo $TXPERT --prediction-root $PRED_ROOT
```

- [ ] 脚本重算所有文件哈希通过后才打开官方 target X（顺序内建，勿绕过）
- [ ] 生成各 target `truth.npy` + release manifest
- 从这一刻起 E201 进入已解盲状态：后续任何修改预测/风险/评价定义都是违规。

## STAGE_7 冻结正式评价

```bash
$PY tools/scripts/run_e201_formal_core_evaluation.py \
  --data-root $DATA_ROOT --evaluation-vector-dir $EVAL_VEC
```

（$EVAL_VEC 运行前必须不存在；输出固定在 docs/实验结果/E201_*/formal_core_evaluation/
下的 tables/reports/figures 与 E201_CORE_FINAL_STATUS.json。）

人工核验与判读（对照冻结门）：
- [ ] 证书门：恒等式残差在数值容差内、family RMS < disagreement 的任务数 = 0
- [ ] 路由门：safeconf_e201_risk 对 family RMS 的 pooled Spearman CI 下限 > 0
      **且** 20% utility CI 下限 > 0（cluster bootstrap 5,000，按 condition 整簇）
- [ ] magnitude 增量门：partial Spearman CI 下限 > 0 **或** 配对 utility 增量
      CI 下限 > 0
- [ ] 四个 target 各自的点估计+区间完整输出；方向不一致的 target 保留
- [ ] 强基线（official general baseline、batch-matched control、predicted
      magnitude）与 E198 五项补充指标（mse/pearson_pert/rank/energy_distance_pca50/
      de_auprc）都在表里
- [ ] 输出任务级明细；负结果原样写入报告

## STAGE_8 结果入文

- 替换论文所有 `[[E201-PLACEHOLDER-*]]`；按三叉选摘要版本（见 00_AUDIT_REPORT §5）；
- 四 target 表 + 森林图更新；validation footprint 表加 E201 行；
- 向周老师汇报用第 10 节语气版本。

## 停链条件（任何一条触发 → 停止并记录，不尝试"修好再跑"）

1. 任何脚本自检/哈希/等价性失败；
2. 出现 `truth.npy` 早于 STAGE_6；
3. 双远程推送失败（绝不带着未承诺的封存开真值）；
4. GPU/磁盘异常；
5. 数字与冻结定义不符（如任务数不是 2,008/1,808）。
