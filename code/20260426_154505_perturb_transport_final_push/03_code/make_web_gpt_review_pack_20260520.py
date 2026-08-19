#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

import pandas as pd


HOME = Path("/home/yyf")
ROOT = HOME / "codex_cout" / "20260426_154505_perturb_transport_final_push"
OUT_ROOT = HOME / "codex_cout" / "20260520_SAFE_TRANS_WEB_GPT_REVIEW_20"
PACK = OUT_ROOT / "WEB_GPT_REVIEW_20"
ZIP_PATH = HOME / "codex_cout" / "20260520_SAFE_TRANS_WEB_GPT_REVIEW_20.zip"


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=8)
    except Exception as exc:
        return f"COMMAND_FAILED: {cmd!r}\n{exc!r}\n"


def read_text(path: Path, max_chars: int | None = None) -> str:
    if not path.exists():
        return f"[missing] {path}\n"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if max_chars is None else text[:max_chars]


def write(name: str, body: str) -> None:
    path = PACK / name
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")


def csv_preview(path: Path, max_rows: int = 30) -> str:
    if not path.exists():
        return f"[missing] {path}"
    try:
        df = pd.read_csv(path)
        return df.head(max_rows).to_csv(index=False)
    except Exception as exc:
        return f"[failed to read {path}] {exc!r}"


def json_pretty(path: Path) -> str:
    if not path.exists():
        return f"[missing] {path}"
    try:
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)
    except Exception:
        return read_text(path, 6000)


def result_snapshot() -> pd.DataFrame:
    rows: list[dict] = []
    candidates = [
        ("strict_before_fix", ROOT / "46_q1_cpu_push_20260520" / "results"),
        ("policy_fix_smoke", ROOT / "54_policy_calibrated_smoke3_20260520" / "results"),
        ("gpu_main_previous", ROOT / "43_gpu_effect_objective_main_20260520" / "results"),
        ("gpu_tian_previous", ROOT / "48_gpu_graft_tian_20260520" / "results"),
    ]
    for label, res in candidates:
        q1 = res / "Q1_READINESS_REPORT.json"
        if q1.exists():
            data = json.loads(q1.read_text(encoding="utf-8"))
            checks = data.get("checks", {})
            rows.append({
                "run_label": label,
                "results_dir": str(res),
                "readiness_label": data.get("label"),
                "primary_model": data.get("primary_model"),
                **{f"check_{k}": v for k, v in checks.items()},
            })
        for summary_name in ["SAFETY_SUMMARY.csv", "GPU_DEEP_SUMMARY.csv"]:
            summary = res / summary_name
            if summary.exists():
                df = pd.read_csv(summary)
                for model in sorted(df["model"].dropna().astype(str).unique()):
                    sub = df[df["model"].astype(str) == model]
                    held = sub[sub.get("split_type", "").astype(str).eq("heldout_perturbation")] if "split_type" in sub else sub
                    if held.empty:
                        held = sub
                    rows.append({
                        "run_label": label,
                        "results_dir": str(res),
                        "summary_file": summary_name,
                        "model": model,
                        "n_rows": len(sub),
                        "heldout_top20_mean": held["top20_overlap_mean"].mean() if "top20_overlap_mean" in held else None,
                        "heldout_deg_mean": held["deg_precision_top50_mean"].mean() if "deg_precision_top50_mean" in held else None,
                        "heldout_program_mean": held["program_shift_consistency_mean"].mean() if "program_shift_consistency_mean" in held else None,
                        "heldout_rmse_mean": held["rmse_mean"].mean() if "rmse_mean" in held else None,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    PACK.mkdir(parents=True, exist_ok=True)

    tmux = run(["bash", "-lc", "tmux ls || true"])
    gpu = run(["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true"])
    ps = run(["bash", "-lc", "ps -u yyf -o pid,stat,pcpu,pmem,etime,cmd --sort=-pcpu | sed -n '1,20p'"])

    snapshot = result_snapshot()
    snapshot.to_csv(PACK / "19_CURRENT_RESULT_TABLES.csv", index=False)

    opus = read_text(HOME / "SafeTrans-Opus7-主审报告.md")
    strict = read_text(HOME / "SafeTrans-严格审核清单.md", 12000)
    handoff = read_text(HOME / "SafeTrans-OPUS交接.md", 12000)
    status = read_text(HOME / "SafeTrans-进展总览.md", 12000)
    policy_fix = read_text(ROOT / "00_meta" / "POLICY_FIX_RUN_20260520.md")

    write("01_README_FOR_WEB_GPT.md", f"""
    # SafeTrans-PT Web GPT Review Pack

    这个包是给网页版本 GPT / Opus / 其他模型做交叉审查用的。

    你要让网页 GPT 明白三件事：

    1. 我们研究的问题是什么：跨 cellular context 的单细胞扰动效应迁移。
    2. Opus7 主审指出了什么问题：现在证据还不支持强方法论文。
    3. Codex 已经怎么改：从简单路由改成专家选择 + 误差校准 + rank-preserving graft，并重启后台实验。

    当前诚实状态：

    - 方向有意义。
    - 当前严格 Q1/Q2-top evaluator 仍未通过。
    - 新一轮修代码后的长实验正在后台跑。
    - 这个包不是最终投稿包，是“严审 + 复盘 + 下一步设计包”。

    当前后台：

    ```text
    {tmux.strip()}
    ```

    GPU 状态：

    ```text
    {gpu.strip()}
    ```
    """)

    write("02_OPUS7_MAIN_REVIEW_FULL.md", opus)

    write("03_STRICT_REVIEW_CHECKLIST_AND_HANDOFF.md", f"""
    # Strict Review Checklist + Opus Handoff

    ## 严格审核清单节选

    {strict}

    ## Opus 交接节选

    {handoff}

    ## 进展总览节选

    {status}
    """)

    write("04_PLAIN_CHINESE_EXPLANATION_FOR_USER.md", """
    # 给小白看的解释

    ## 我们到底在做什么？

    我们在研究一个问题：

    **一个基因/药物扰动，在 A 细胞环境里测过以后，能不能迁移到 B 细胞环境里预测？**

    举例：

    - 在 K562 细胞里敲掉某个基因，细胞表达会变。
    - 那么换成 T cell、iPSC、肿瘤细胞，变化还能不能猜？
    - 如果能猜，哪些基因变化最可靠？
    - 如果不能猜，模型能不能主动说“这个我不敢预测”？

    ## 为什么这个问题有价值？

    因为真实实验不可能把所有扰动、所有细胞类型、所有病人状态都测一遍。

    如果模型能判断“哪些扰动效应可以迁移”，就可以：

    - 减少实验量；
    - 帮助挑选值得验证的 perturbation；
    - 给药物筛选、CRISPR 筛选、疾病机制分析提供候选方向；
    - 避免模型在完全不靠谱的细胞背景上硬猜。

    ## 现在卡在哪里？

    简单说：V0 这个朴素 baseline 太强。

    V0 的做法像这样：

    > 这个扰动以前平均造成什么表达变化，我就照着猜。

    它很土，但很稳。我们的复杂方法目前还没有稳定超过它。

    ## Opus7 为什么批得重？

    因为论文审稿人会问：

    - 你真的比最简单方法强吗？
    - 你的安全拒判真的能筛掉错误预测吗？
    - 你的 external validation 真独立吗？
    - 你和 GEARS / CPA / CellOT 是不是同题比较？

    当前答案还不够硬。

    ## Codex 现在改了什么？

    我把模型从“一个模型自己猜”改成：

    - 多个专家一起出意见；
    - 路由器判断该信谁；
    - 误差校准器估计这次预测大概会不会错；
    - safety score 决定这条结果是否适合汇报；
    - rank graft 尽量保住 V0 的 top20 强项，同时吸收复杂模型在 program / DEG 上的优势。
    """)

    write("05_PROBLEM_AND_PAPER_STORY.md", """
    # Scientific Story

    ## 题目

    SafeTrans-PT: safe cross-context transport of single-cell perturbation effects.

    中文：

    **单细胞扰动效应在不同细胞背景之间的安全迁移。**

    ## 不是在做什么

    不是单纯做 cell type annotation。

    不是只做一个普通预测器。

    不是说所有扰动都能从一个细胞类型迁移到另一个细胞类型。

    ## 真正在解决什么

    以前很多 perturbation prediction 方法关注：

    - 给定 perturbation，预测表达变化；
    - 给定 drug/gene，预测 treated cell state；
    - 在已有数据分布附近做泛化。

    我们更关注：

    - 一个 effect 到底能不能跨 context 迁移；
    - 哪些 effect 迁移后仍然可信；
    - 哪些 context shift 会让模型不该相信自己；
    - 如何把“不安全预测”从结果里识别出来。

    ## 潜在论文定位

    当前最稳妥的定位不是“全面打败 SOTA”，而是：

    **一个面向 cross-context perturbation transport 的安全评估和选择性预测框架。**

    如果后续重跑证明新 Policy 稳定超过 V0/V2，才可以升级成强方法论文。
    """)

    write("06_CURRENT_STRICT_STATUS.md", f"""
    # Current Strict Status

    ## 旧主结果，严格 evaluator

    ```json
    {json_pretty(ROOT / "46_q1_cpu_push_20260520" / "results" / "Q1_READINESS_REPORT.json")}
    ```

    ## 新代码 smoke 结果

    ```json
    {json_pretty(ROOT / "54_policy_calibrated_smoke3_20260520" / "results" / "Q1_READINESS_REPORT.json")}
    ```

    ## 解释

    - 旧结果：`NOT_READY`，主要因为 Policy 没有稳定赢 V0，risk-coverage 也没过。
    - 新 smoke：risk-coverage 从负数变成正数，但 held-out main 仍未赢 V0。
    - 这说明新 safety score 有一点改善，但方法性能还需要完整长跑验证。

    结论：

    **现在不能说稳 Q1，也不能说稳 Q2-top。可以说方向和问题定义成立，但方法证据还在补强。**
    """)

    write("07_CODE_CHANGELOG_20260520.md", f"""
    # Code Changelog - 2026-05-20

    ## 修改过的关键文件

    - `{ROOT}/03_code/transport_models.py`
    - `{ROOT}/03_code/safetrans_models.py`
    - `{ROOT}/03_code/run_policy_calibrated_push_20260520.sh`
    - `{ROOT}/03_code/run_gpu_main_policy_fix_20260520.sh`
    - `{ROOT}/03_code/run_gpu_external_policy_fix_20260520.sh`
    - `{ROOT}/00_meta/POLICY_FIX_RUN_20260520.md`

    ## 关键修复

    1. `RidgeRegressor` 增加 target intercept。
       - 原来只有 `X @ coef`。
       - 现在会中心化 `y`，预测时加回 `y_mean`。
       - 这对 expert utility / RMSE calibration 很重要。

    2. `PolicySafeTransPT` 增加内部专家：
       - `V0`
       - `V1`
       - `V2`
       - `Safe`
       - `Network`
       - `ContextSim`

    3. 增加 out-of-fold expert utility prediction。

    4. 增加 out-of-fold expert RMSE prediction。

    5. `transportability_score` 改成校准风险分：
       - predicted RMSE
       - context similarity
       - perturbation consistency
       - expert agreement
       - retrieval confidence

    6. 增加 rank-preserving graft：
       - 保留 V0 的 top20 强项；
       - 在 top50 区域尝试吸收 V2/Safe/Network/ContextSim 的一致信号。

    ## 本轮修复记录

    {policy_fix}
    """)

    write("08_METHOD_DESIGN_CURRENT.md", """
    # Current Method Design

    ## 名字

    主方法仍然叫：

    `PolicySafeTransPT`

    不新增奇怪的新模型名，避免故事发散。

    ## 方法结构

    输入：

    - control expression：未扰动细胞表达；
    - perturbation identity：扰动名称；
    - context identity：细胞类型、病人、batch 或 cell line；
    - historical effects：训练集中已有扰动效应；
    - pathway / program features：基因程序或通路层面的低维表示。

    内部专家：

    - `V0`：强朴素 baseline；
    - `V1`：program-level transport；
    - `V2`：graph/pathway prior transport；
    - `Safe`：保守 safety-aware transport；
    - `Network`：hdWGCNA-inspired module transport；
    - `ContextSim`：只在 context 相似时迁移。

    核心逻辑：

    1. 每个专家都给一个 effect prediction。
    2. Router 预测哪个专家更适合当前 task。
    3. RMSE calibrator 估计当前 task 可能错多大。
    4. Safety score 判断这条预测是否适合信任。
    5. Rank graft 尽量同时保住 top genes 和 program consistency。

    ## 这比普通模型多了什么？

    普通模型说：

    > 我给你一个预测。

    SafeTrans-PT 想说：

    > 我先判断这次迁移靠不靠谱，再决定怎么预测；不靠谱就标记为 unsafe。
    """)

    write("09_DATASETS_AND_SPLITS.md", f"""
    # Data and Splits

    ## 已使用/正在使用的数据来源

    当前 atlas 路径：

    `/home/yyf/datasets/singlecell_perturbation_atlas/`

    主要数据包括：

    - KaggleCrossCell
    - Haber
    - Parekh
    - Wessels
    - NormanWeissman2019
    - DixitRegev2016
    - AdamsonWeissman2016
    - KaggleCrossPatient
    - McFarland
    - TianKampmann2019
    - PapalexiSatija2021
    - Frangieh
    - SrivatsanTrapnell2020

    ## 主要 split

    - held-out perturbation：训练时没见过某个扰动；
    - leave-context：训练时没见过某个 cellular context；
    - external：尝试在独立数据集上验证方向。

    ## 当前正在跑的选择表

    ### CPU policy main selected

    ```csv
    {csv_preview(ROOT / "51_policy_calibrated_q1_20260520" / "results" / "SAFETY_MAIN_SELECTED.csv", 20)}
    ```

    ### CPU policy external selected

    ```csv
    {csv_preview(ROOT / "51_policy_calibrated_q1_20260520" / "results" / "SAFETY_EXTERNAL_SELECTED.csv", 20)}
    ```

    ### GPU main selected

    ```csv
    {csv_preview(ROOT / "52_gpu_policy_fix_main_20260520" / "results" / "GPU_DEEP_SELECTED_DATASETS.csv", 20)}
    ```

    ### GPU external selected

    ```csv
    {csv_preview(ROOT / "53_gpu_policy_fix_external_20260520" / "results" / "GPU_DEEP_SELECTED_DATASETS.csv", 20)}
    ```
    """)

    write("10_METRICS_EXPLAINED.md", """
    # Metrics Explained

    ## Pearson

    预测 effect vector 和真实 effect vector 的线性相关。

    通俗说：

    > 整体变化趋势像不像？

    缺点：

    > 可能整体像，但关键基因没猜对。

    ## Spearman

    排名相关。

    通俗说：

    > 哪些基因变化更大，排序像不像？

    ## RMSE

    平均误差。

    通俗说：

    > 猜得离真实值有多远？

    越低越好。

    ## top20 overlap

    真实变化最大的 20 个基因，预测有没有找回来。

    这是生物解释里很重要的指标。

    ## DEG precision

    DEG 是 differentially expressed genes，差异表达基因。

    DEG precision 看预测出来的重要差异基因有多少是真的。

    ## program consistency

    不看单个基因，而看基因程序/通路层面变化是否一致。

    通俗说：

    > 不要求每个字都一样，但这段话表达的意思是不是差不多。

    ## risk-coverage

    按模型自信程度排序，只保留最可信的一部分预测。

    如果安全机制真的有用：

    - coverage 降低一点；
    - RMSE 应该下降；
    - unsafe 样本应该比 safe 样本更差。
    """)

    write("11_OPUS_GAPS_AND_FIX_MAPPING.md", """
    # Opus7 Gaps and Fix Mapping

    ## Gap 1: 没有稳定超过 V0

    Opus7 意思：

    > 简单 baseline 已经很强，你的复杂模型没有稳定赢它。

    Codex 修复：

    - 加专家选择；
    - 加 Safe / Network / ContextSim 专家；
    - 加 rank-preserving graft，尽量保住 V0 的 top20，同时吸收复杂模型的 program signal。

    当前状态：

    - smoke 未证明已解决；
    - 长跑正在验证。

    ## Gap 2: risk-coverage 失败

    Opus7 意思：

    > 模型拒判以后，结果没有变好。

    Codex 修复：

    - confidence 不再只是 router probability；
    - 改成 predicted RMSE + context similarity + expert agreement 的校准分。

    当前状态：

    - smoke 里 risk RMSE gain 从旧结果的负数变成正数；
    - unsafe contrast 还不够，需要长跑确认。

    ## Gap 3: GEARS 比较不够同题

    当前未完全解决。

    后续需要：

    - 同 dataset；
    - 同 gene space；
    - 同 split；
    - 同 effect-level metrics。

    ## Gap 4: external validation 不够硬

    后续需要：

    - 把 KaggleCrossPatient / crossPatient 这类近源 external 区分开；
    - 把 Tian / Papalexi / Frangieh / Srivatsan 作为更强 external attempt；
    - 每个 external 记录为什么能用或为什么不能用。
    """)

    write("12_RUNNING_EXPERIMENTS_STATUS.md", f"""
    # Running Experiments Status

    ## tmux

    ```text
    {tmux.strip()}
    ```

    ## GPU

    ```text
    {gpu.strip()}
    ```

    ## Top processes

    ```text
    {ps.strip()}
    ```

    ## Running dirs

    - CPU policy: `{ROOT}/51_policy_calibrated_q1_20260520/`
    - GPU main: `{ROOT}/52_gpu_policy_fix_main_20260520/`
    - GPU external: `{ROOT}/53_gpu_policy_fix_external_20260520/`

    ## Commands to inspect

    ```bash
    tmux attach -t safetrans_policy_fix_20260520
    tail -f {ROOT}/51_policy_calibrated_q1_20260520/logs/run_safety.log
    tail -f {ROOT}/52_gpu_policy_fix_main_20260520/logs/run_gpu_main.log
    tail -f {ROOT}/53_gpu_policy_fix_external_20260520/logs/run_gpu_external.log
    ```
    """)

    write("13_COMPUTER_SCIENCE_IDEAS_TO_BORROW.md", """
    # Computer Science Ideas Worth Borrowing

    下面不是空想，而是可以落到 SafeTrans-PT 代码里的方向。

    ## 1. Counterfactual routing analysis

    来源思路：

    近期 MoE 论文在问：router 选的专家真的是最优的吗？如果换一个同等计算量的专家，会不会更好？

    对我们的启发：

    - 不要只看 Policy 选了哪个专家；
    - 要反事实比较：同一个 task 下 V0、V2、Safe、Network、ContextSim 哪个真的好；
    - 用 out-of-fold 结果训练 router；
    - 这正是本轮 Policy 改造的核心。

    可落地代码：

    - `expert_utility_regressor`
    - `expert_rmse_regressor`
    - `selected_expert` audit table

    ## 2. Conformal / selective prediction

    来源思路：

    conformal prediction 关心覆盖率和不确定性保证。

    对我们的启发：

    - 不要只输出 prediction；
    - 输出“这条 prediction 是否安全”；
    - risk-coverage 曲线必须变好；
    - unsafe 组误差应该高于 safe 组。

    可落地代码：

    - calibration split；
    - nonconformity score = predicted RMSE / expert disagreement / context distance；
    - coverage-controlled abstention。

    ## 3. Conditional optimal transport

    来源思路：

    Conditional Monge Gap 这类工作强调把 treatment/context covariates 放进 transport map。

    对我们的启发：

    - 不能只迁移 effect；
    - 要条件化到 context、dose、drug/gene、cell type；
    - 我们可以把 `control_mean + context embedding + perturbation prior` 作为 transport condition。

    可落地代码：

    - context-conditioned residual map；
    - OT-like barycentric proxy baseline；
    - effect distribution not only mean vector。

    ## 4. Virtual cell benchmark thinking

    来源思路：

    Virtual Cell Challenge 强调 biologically meaningful generalization across perturbations and cell types。

    对我们的启发：

    - 论文不要只讲模型；
    - 要讲 benchmark / split / failure boundary；
    - leave-context 是真正难点，不要硬吹。

    可落地代码：

    - benchmark protocol；
    - held-out perturbation；
    - leave-context；
    - external dataset audit。

    ## 5. Agentic model design

    来源思路：

    CellForge 类工作强调自动分析任务、设计方法、生成实验代码。

    对我们的启发：

    - 可以把 SafeTrans-PT 包装成“可审计自动建模流程”；
    - 但不要把 agent 当主创新；
    - agent 只作为实验组织工具，主创新仍是 safe transport。

    ## 6. hdWGCNA / network modules

    对我们的启发：

    - 单个基因太嘈杂；
    - 可以用共表达模块解释哪些 effect 可迁移；
    - NetworkSafeTransPT 应该成为 biological explanation branch，而不是主模型硬拼性能。
    """)

    write("14_LITERATURE_MAP_WITH_LINKS.md", """
    # Literature Map with Links

    ## Perturbation response prediction

    - GEARS: graph-enhanced perturbation prediction baseline.
    - CPA: compositional perturbation autoencoder.
    - CellOT: optimal transport for single-cell perturbations.
    - scGen / scGPT / scFoundation: foundation-model or latent-space perturbation prediction families.

    ## Recent related papers checked

    1. Virtual Cell Challenge, Cell, 2025.
       - Link: https://www.sciencedirect.com/science/article/pii/S0092867425006750
       - Relevant point: community now cares about perturbation response prediction and context/cell-type generalization benchmarks.

    2. Conditional Monge Gap, arXiv 2025.
       - Link: https://arxiv.org/abs/2504.08328
       - Relevant point: conditional optimal transport for unseen treatments and covariates.

    3. Distribution-informed Online Conformal Prediction, arXiv 2025/2026.
       - Link: https://arxiv.org/abs/2512.07770
       - Relevant point: uncertainty/coverage under distribution shift.

    4. Counterfactual Routing Analysis in MoE, arXiv 2026.
       - Link: https://arxiv.org/abs/2605.07260
       - Relevant point: router decisions can be wrong even when experts contain useful capability; evaluate routes directly.

    5. CellForge, 2025.
       - Link: https://arxiv.org/abs/2508.02276
       - Relevant point: agentic design of virtual cell models; useful as automation inspiration, not as our main claim.

    6. Closing the loop: Teaching single-cell foundation models to learn from perturbations, bioRxiv 2025/2026.
       - Link: https://www.biorxiv.org/content/10.1101/2025.07.08.663754v1
       - Relevant point: perturbation data should be used iteratively to improve foundation models.

    ## What is still missing from our paper story

    - A formal, fair, same-task GEARS / CPA / CellOT head-to-head.
    - A stronger biological explanation: which programs are transferable and why.
    - A convincing uncertainty result: unsafe predictions must be visibly worse than safe predictions.
    - A clean claim boundary: held-out perturbation may work; leave-context is harder.
    """)

    write("15_NEXT_EXPERIMENT_PLAN.md", """
    # Next Experiment Plan

    ## Immediate

    1. Let current tmux runs finish.
    2. Re-run strict evaluator:

    ```bash
    cd /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push
    conda activate scgpt_env
    python 03_code/evaluate_q1_readiness.py --results-dir 51_policy_calibrated_q1_20260520/results --write-md
    ```

    3. Compare:

    - old 46 results;
    - new 51 CPU Policy results;
    - new 52 GPU main results;
    - new 53 GPU external results.

    ## If results improve

    Push these:

    - held-out perturbation stable improvement;
    - safety score improves risk-coverage;
    - external direction consistent;
    - rank-graft preserves top20 while improving program-level signal.

    ## If results do not improve

    Do not fake it.

    Shift paper positioning to:

    **benchmark/protocol + safety boundary paper**.

    Then implement:

    - fair GEARS/CPA/CellOT comparator;
    - conformal abstention;
    - stronger network module explanation;
    - external-only case study.

    ## What would make it more Q1-like

    - Same-task wins over V0/V2/GEARS on at least two hard OOD settings.
    - Clear unsafe transport detection.
    - Biological explanation at gene-program/pathway level.
    - External validation with direction consistency.
    """)

    write("16_Q1_Q2_READINESS_CHECKLIST.md", """
    # Q1 / Q2 Readiness Checklist

    ## Must pass before saying Q2-top

    - [ ] PolicySafeTransPT wins V0 on >=60% main held-out perturbation settings.
    - [ ] External held-out win fraction >=50%.
    - [ ] Risk-coverage RMSE improves by >=3% at about 80% coverage.
    - [ ] Unsafe group has higher RMSE than safe group in >=50% settings.
    - [ ] Results are saved as CSV/JSON/logs.
    - [ ] Claims do not say “fully solved”.

    ## Must pass before saying Q1 candidate

    - [ ] PolicySafeTransPT wins V0 on >=75% main held-out settings.
    - [ ] Beats V2 on >=55% held-out perturbation settings.
    - [ ] Beats ContextSimBaseline on >=70% held-out settings.
    - [ ] Leave-context program consistency improves in >=50%.
    - [ ] At least 3 external held-out datasets.
    - [ ] Fair GEARS/CPA/CellOT head-to-head or honest limitation.
    - [ ] Biological program explanation is clear.
    - [ ] Safety/abstention result is not cosmetic.

    ## Current honest label

    `NOT_READY` under strict evaluator.

    This is not a death sentence. It means:

    > Good problem, insufficient evidence so far.
    """)

    write("17_PROMPT_FOR_WEB_GPT_REVIEW.md", f"""
    # Prompt to Paste into Web GPT

    请你作为严格但建设性的审稿人，阅读我上传的 20 个文件。

    项目路径：

    `{ROOT}`

    研究方向：

    **Safe cross-context single-cell perturbation effect transport**

    中文：

    **跨细胞背景的单细胞扰动效应安全迁移。**

    请你重点判断：

    1. 这个问题是否真的有论文价值？
    2. 现在的方法是不是只是资源整合，还是有方法创新？
    3. Opus7 的批评是否合理？
    4. Codex 这次改代码的方向是否正确？
    5. 还可以借鉴哪些计算机领域方法？
    6. 如果目标是稳定 Q2、尝试 Q1，下一步最该补什么？
    7. 哪些说法不能在论文/PPT里讲？
    8. 如果你是导师，你会建议继续推进、收缩题目，还是换方向？

    禁止：

    - 不要因为故事好听就给高评价。
    - 不要说“稳一区”这种没有证据的话。
    - 不要只看 Pearson/Spearman。

    请输出：

    - 一段通俗结论；
    - 严格审稿意见；
    - 方法创新评分；
    - 实验可信度评分；
    - 最短补强路线；
    - 给学生汇报用的版本。
    """)

    write("18_FILE_PATHS_AND_COMMANDS.md", f"""
    # File Paths and Commands

    ## Main project

    `{ROOT}`

    ## Code

    `{ROOT}/03_code/`

    ## Important code files

    - `{ROOT}/03_code/safetrans_models.py`
    - `{ROOT}/03_code/transport_models.py`
    - `{ROOT}/03_code/run_safety_abstention_evidence.py`
    - `{ROOT}/03_code/run_deep_gpu_transport.py`
    - `{ROOT}/03_code/evaluate_q1_readiness.py`

    ## Current run scripts

    - `{ROOT}/03_code/run_policy_calibrated_push_20260520.sh`
    - `{ROOT}/03_code/run_gpu_main_policy_fix_20260520.sh`
    - `{ROOT}/03_code/run_gpu_external_policy_fix_20260520.sh`

    ## Check experiments

    ```bash
    tmux ls
    tmux attach -t safetrans_policy_fix_20260520
    ```

    ## Re-score strict readiness

    ```bash
    cd {ROOT}
    conda activate scgpt_env
    python 03_code/evaluate_q1_readiness.py --results-dir 51_policy_calibrated_q1_20260520/results --write-md
    python 03_code/evaluate_q1_readiness.py --results-dir 52_gpu_policy_fix_main_20260520/results --primary-model DeepCalibratedSafeTransport --write-md
    python 03_code/evaluate_q1_readiness.py --results-dir 53_gpu_policy_fix_external_20260520/results --primary-model DeepCalibratedSafeTransport --write-md
    ```
    """)

    # 19_CURRENT_RESULT_TABLES.csv is already written.

    manifest_rows = []
    for path in sorted(PACK.iterdir()):
        if path.is_file():
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size})
    manifest_text = "\n".join([f"- {r['file']} ({r['bytes']} bytes)" for r in manifest_rows])
    write("20_MANIFEST.md", f"""
    # Manifest

    This zip contains exactly 20 review files under `WEB_GPT_REVIEW_20/`.

    {manifest_text}

    ## Zip path

    `{ZIP_PATH}`
    """)

    # Update manifest after file 20 creation.
    files = sorted(PACK.iterdir())
    if len([p for p in files if p.is_file()]) != 20:
        raise RuntimeError(f"Expected 20 files, got {len([p for p in files if p.is_file()])}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACK.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT_ROOT))

    print(json.dumps({
        "zip_path": str(ZIP_PATH),
        "folder": str(OUT_ROOT),
        "n_files": len([p for p in PACK.iterdir() if p.is_file()]),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
