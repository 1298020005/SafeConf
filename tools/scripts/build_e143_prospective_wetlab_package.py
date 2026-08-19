#!/usr/bin/env python3
"""Build the E143 prospective, blinded wet-lab validation handoff package.

This script does not claim to execute physical experiments.  It converts the
frozen Nadig pre-truth scores into a technical-pilot panel, calculates formal
correlation power, and creates the preregistration, QC, sample-layout and input
templates required before a genuinely new-cell-context experiment can start.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E143_prospective_wetlab_validation_20260714"
TABLES, REPORTS, TEMPLATES, FIGURES = OUT / "tables", OUT / "reports", OUT / "templates", OUT / "figures"
NADIG_SCORES = ROOT / "docs/实验结果/E139_nadig_directional_confirmation_20260714/tables/E139_DIRECTIONAL_SCORES_BEFORE_TRUTH.csv"
SEED = 202607143


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation_power(n: int, true_rho: float, alpha: float = 0.05) -> tuple[float, float, float]:
    standard_error = 1 / np.sqrt(n - 3)
    mean = np.arctanh(true_rho) / standard_error
    critical = norm.ppf(1 - alpha / 2)
    power = norm.sf(critical - mean) + norm.cdf(-critical - mean)
    lower = np.tanh(np.arctanh(true_rho) - critical * standard_error)
    upper = np.tanh(np.arctanh(true_rho) + critical * standard_error)
    return float(power), float(lower), float(upper)


def power_table() -> pd.DataFrame:
    rows = []
    for n_genes in [16, 24, 32, 40, 48, 56, 64]:
        for true_rho in [0.30, 0.40, 0.50, 0.60]:
            power, lower, upper = correlation_power(n_genes, true_rho)
            rows.append({"independent_perturbation_genes": n_genes, "assumed_true_rho": true_rho,
                         "approximate_two_sided_power_alpha_0.05": power,
                         "expected_fisher_95ci_low": lower, "expected_fisher_95ci_high": upper,
                         "unit_is_gene_not_cell": True})
    return pd.DataFrame(rows)


def minimum_n_table() -> pd.DataFrame:
    rows = []
    for true_rho in [0.30, 0.40, 0.50, 0.60]:
        for target_power in [0.80, 0.90]:
            minimum = next(n for n in range(8, 401) if correlation_power(n, true_rho)[0] >= target_power)
            rows.append({"assumed_true_rho": true_rho, "target_power": target_power,
                         "minimum_independent_genes_fisher_approximation": minimum})
    return pd.DataFrame(rows)


def technical_pilot_candidates() -> pd.DataFrame:
    scores = pd.read_csv(NADIG_SCORES)
    required = {"perturbation", "context", "directional_risk_frozen", "fold_id", "frozen_model_sha256"}
    if not required.issubset(scores.columns):
        raise RuntimeError(f"missing score-only columns: {sorted(required - set(scores.columns))}")
    pivot = scores.pivot_table(index="perturbation", columns="context", values="directional_risk_frozen", aggfunc="mean")
    if not {"HepG2", "Jurkat"}.issubset(pivot.columns):
        raise RuntimeError("Nadig score snapshot must contain HepG2 and Jurkat")
    pivot = pivot.dropna(subset=["HepG2", "Jurkat"]).copy()
    pivot["mean_risk"] = pivot[["HepG2", "Jurkat"]].mean(axis=1)
    pivot["risk_delta_HepG2_minus_Jurkat"] = pivot["HepG2"] - pivot["Jurkat"]
    used: set[str] = set()
    selected = []

    def take(frame: pd.DataFrame, stratum: str, count: int):
        for gene, row in frame.iterrows():
            if gene in used:
                continue
            used.add(gene)
            selected.append({"gene": gene, "pilot_stratum": stratum, "risk_HepG2": row.HepG2,
                             "risk_Jurkat": row.Jurkat, "mean_risk": row.mean_risk,
                             "risk_delta_HepG2_minus_Jurkat": row.risk_delta_HepG2_minus_Jurkat,
                             "selection_used_expression_or_error_truth": False,
                             "independent_confirmation_eligible": False,
                             "purpose": "technical workflow calibration only"})
            if sum(item["pilot_stratum"] == stratum for item in selected) == count:
                break

    take(pivot.sort_values("mean_risk", ascending=False), "dual_context_high_risk", 6)
    take(pivot.sort_values("mean_risk", ascending=True), "dual_context_low_risk", 6)
    take(pivot.sort_values("risk_delta_HepG2_minus_Jurkat", ascending=False), "HepG2_higher_risk", 6)
    take(pivot.sort_values("risk_delta_HepG2_minus_Jurkat", ascending=True), "Jurkat_higher_risk", 6)
    result = pd.DataFrame(selected)
    result.insert(0, "pilot_id", [f"PILOT-{index:02d}" for index in range(1, len(result) + 1)])
    result["source_score_snapshot_sha256"] = sha256(NADIG_SCORES)
    return result


def formal_candidate_slots() -> pd.DataFrame:
    strata = (["both_contexts_high"] * 8 + ["both_contexts_low"] * 8 +
              ["new_context_high_anchor_low"] * 8 + ["new_context_low_anchor_high"] * 8 +
              ["middle_risk_coverage"] * 16)
    rows = []
    for index, stratum in enumerate(strata, 1):
        rows.append({"formal_slot": f"FORMAL-{index:02d}", "required_stratum": stratum, "gene": "TBD_AFTER_BASELINE",
                     "new_context": "TBD_ONE_OF_A549_HUH7_HEP3B_OR_LAB_VALIDATED_ALTERNATIVE",
                     "anchor_context": "Jurkat_or_second_new_context", "guide_1": "TBD", "guide_2": "TBD",
                     "risk_frozen_before_perturbation_readout": True, "replacement_allowed_only_for_prefrozen_technical_failure": True})
    return pd.DataFrame(rows)


def formal_input_template() -> pd.DataFrame:
    columns = ["gene", "new_context_name", "anchor_context_name", "risk_new_context", "risk_anchor_context",
               "predicted_magnitude_new", "predicted_magnitude_anchor", "model_disagreement_new", "model_disagreement_anchor",
               "control_expression_new", "control_expression_anchor", "string_degree_v12_score700",
               "depmap_gene_effect_new", "depmap_gene_effect_anchor", "guide_1_sequence", "guide_1_on_target_score",
               "guide_2_sequence", "guide_2_on_target_score", "maximum_predicted_off_target_score",
               "copy_number_warning", "gene_family_multimapping_warning", "eligible_before_perturbation_truth"]
    return pd.DataFrame([{column: "" for column in columns}])


def sample_layout() -> pd.DataFrame:
    rows = []
    for context_code, context in [("NEW", "new_context_TBD"), ("ANC", "anchor_context_TBD")]:
        for batch in range(1, 4):
            digest = hashlib.sha256(f"E143|{context_code}|{batch}|{SEED}".encode()).hexdigest()[:8].upper()
            rows.append({"blinded_library_id": f"WL-{digest}", "context_code": context_code,
                         "biological_batch": batch, "collection_day": 7, "library_type": "10x_3prime_plus_guide_capture",
                         "target_qc_cells_per_guide": 100, "minimum_qc_cells_per_guide": 50,
                         "planned_targeting_guides": 96, "planned_non_targeting_guides": 6,
                         "target_total_qc_cells": 10200, "suggested_loaded_cells_allowing_20pct_loss": 12750,
                         "risk_stratum_visible_to_wetlab": False})
    return pd.DataFrame(rows)


def qc_rules() -> pd.DataFrame:
    rows = [
        ("cell_identity", "Before perturbation", "STR identity matches expected line", "quarantine line; do not replace based on outcome"),
        ("mycoplasma", "Before perturbation and harvest", "negative", "repeat test; positive batch excluded before unblinding"),
        ("baseline_replicates", "Before candidate freeze", "three independently cultured controls; pairwise pseudobulk Pearson >=0.90", "resolve batch drift before scoring"),
        ("guide_design", "Before ordering", "two independent guides per gene; prespecified on/off-target filters", "use only ranked technical alternate"),
        ("guide_assignment", "Cell QC", "primary analysis retains one confidently assigned guide; doublets removed", "record all assignment rates"),
        ("cells_per_guide_batch", "Cell QC", ">=50 after QC; target 100-150", "failed guide-batch remains in CONSORT-style flow table"),
        ("knockdown", "Before label unblinding", ">=50% reduction by target RNA/qPCR for at least one guide; report both", "technical failure, never outcome-based deletion"),
        ("candidate_completion", "Experiment QC", ">=75% candidates meet prespecified knockdown and cell-count QC in both contexts", "otherwise label study technically incomplete"),
        ("cell_filter", "Cell QC", "library-specific robust MAD thresholds for genes/UMI/mitochondrial fraction frozen before risk unblinding", "no risk-stratum-specific thresholds"),
        ("viability", "Harvest", "record live fraction, Annexin V/PI and cell-cycle proxy", "adjust/explain; do not silently remove high-death targets"),
        ("batch_balance", "Library design", "all risk strata represented in every culture batch and lane", "rerandomize before wet work"),
        ("analysis_lock", "Before unblinding", "QC report, pseudobulk matrix, exclusion log and hashes complete", "only then release risk codebook"),
    ]
    return pd.DataFrame(rows, columns=["qc_item", "timepoint", "prefrozen_rule", "failure_action"])


def configure_plotting() -> None:
    cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(cjk_font).exists():
        font_manager.fontManager.addfont(cjk_font)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10,
    })


def save_figure(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_power_figure() -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    palette = {0.30: "#7A9E9F", 0.40: "#2C7FB8", 0.50: "#31A354", 0.60: "#756BB1"}
    counts = np.arange(10, 121)
    for true_rho, color in palette.items():
        values = [correlation_power(int(count), true_rho)[0] for count in counts]
        ax.plot(counts, values, lw=2, color=color, label=f"真实 ρ={true_rho:.1f}")
    ax.axhline(.8, color="#555555", lw=1, ls="--")
    ax.axvline(48, color="#C44E52", lw=1.2, ls=":")
    ax.text(49.5, .16, "正式方案：48 基因", color="#C44E52", fontsize=9, rotation=90, va="bottom")
    ax.set(xlim=(10, 120), ylim=(0, 1.02), xlabel="独立扰动基因数（不是细胞数）", ylabel="双侧检验功效",
           title="E143 前瞻验证：相关性检验的样本量")
    ax.set_yticks(np.linspace(0, 1, 6)); ax.grid(axis="y", color="#E6E6E6", lw=.7)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout(); save_figure(fig, "E143_FIG1_CORRELATION_POWER")


def draw_design_figure() -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    steps = [
        ("1", "新背景准备", "STR + 支原体\n3 批未扰动 baseline"),
        ("2", "计算冻结", "scGPT / GEARS\n风险分数与预测向量"),
        ("3", "匹配候选", "48 基因\n高 / 低 / 背景反差 / 中间"),
        ("4", "盲法扰动", "2 guides / gene\n2 背景 × 3 独立批次"),
        ("5", "转录组读出", "6 个 10x 文库\nDay 7 + guide capture"),
        ("6", "锁定后解盲", "先 QC 和 pseudobulk\n再检验风险—误差关系"),
    ]
    x_positions = np.linspace(.25, 10.25, len(steps))
    for index, ((number, title, body), x) in enumerate(zip(steps, x_positions)):
        box = patches.FancyBboxPatch((x, 1.25), 1.55, 2.5, boxstyle="round,pad=0.04,rounding_size=0.08",
                                     facecolor="#F7FAFC", edgecolor="#6B8798", linewidth=1.2)
        ax.add_patch(box)
        ax.add_patch(patches.Circle((x + .25, 3.42), .18, facecolor="#2C7FB8", edgecolor="none"))
        ax.text(x + .25, 3.42, number, color="white", ha="center", va="center", weight="bold", fontsize=9)
        ax.text(x + .78, 3.00, title, ha="center", va="center", weight="bold", fontsize=11, color="#243746")
        ax.text(x + .78, 2.22, body, ha="center", va="center", fontsize=9.2, color="#425563", linespacing=1.5)
        if index < len(steps) - 1:
            ax.annotate("", xy=(x_positions[index + 1] - .10, 2.50), xytext=(x + 1.65, 2.50),
                        arrowprops={"arrowstyle": "-|>", "color": "#7A9E9F", "lw": 1.4})
    ax.text(6, 4.55, "E143 双细胞背景前瞻 CRISPRi 验证", ha="center", va="center", fontsize=15, weight="bold", color="#243746")
    ax.text(6, .55, "统计单位：48 个独立基因；同一基因的背景、guide 和批次整体聚类 bootstrap",
            ha="center", va="center", fontsize=10, color="#555555")
    fig.tight_layout(); save_figure(fig, "E143_FIG2_PROSPECTIVE_DESIGN")


def write_documents(pilot: pd.DataFrame, power: pd.DataFrame, minimum: pd.DataFrame) -> None:
    min_r04_80 = int(minimum[(minimum.assumed_true_rho == .4) & (minimum.target_power == .8)].minimum_independent_genes_fisher_approximation.iloc[0])
    (OUT / "README_先看这个.md").write_text(
        "# E143 前瞻湿实验验证包\n\n"
        "这不是已经完成的湿实验。服务器端已完成候选规则、功效、盲法、质控、样本布局、分析门槛和交接模板；"
        "物理实验还需要新细胞背景、CRISPRi 条件、平台、预算和实验负责人。\n\n"
        "1. 先读 `reports/E143_DECISION_AND_HANDOFF.md`。\n"
        "2. 与实验室确认条件后填 `templates/FORMAL_CANDIDATE_INPUT.csv`。\n"
        "3. 正式候选必须在任何扰动后表达读出产生前运行 `python tools/scripts/freeze_e143_formal_wetlab_candidates.py --input <填好的CSV>` 冻结和哈希；风险映射会写入仓库外的私有目录。\n"
        "4. `tables/E143_NADIG_TECHNICAL_PILOT.csv` 只能调流程，不能作为新增独立验证。\n"
    )
    (OUT / "PREREG_PROSPECTIVE_CONFIRMATION.md").write_text(
        "# E143 预注册草案｜双背景前瞻 CRISPRi 验证\n\n"
        "## 假设与设计\n\n"
        "在一个从未进入七数据开发/评价的新细胞背景，以及 Jurkat 锚点或第二个新背景中，"
        "先取得三批未扰动 baseline，再生成 scGPT、GEARS 预测和 Directional-SafeConf 分数。"
        "任何扰动后表达数据产生前，冻结 48 个目标、96 条 sgRNA、预测向量、风险分数、代码提交与 SHA256。\n\n"
        "48 个目标由 8 个双背景高风险、8 个双背景低风险、16 个背景反差和 16 个中间风险组成。"
        "高低风险组在 predicted magnitude、模型分歧、baseline 表达、STRING degree、DepMap fitness、"
        "guide 质量与功能类别上匹配。另冻 10 个有顺序的技术替补，只能因未表达、合成失败或不满足 guide 规则替换。\n\n"
        "## 主终点\n\n"
        "每个 guide×batch 先在同批 non-targeting control 上形成 pseudobulk Δ；再在冻结的 512 基因轴上计算"
        "两个预测器平均 centered-Pearson error、centered-cosine error，以及二者秩均值。"
        "统计独立单位是 gene；同一基因的两个背景、两条 guide、三批实验作为一个 cluster。\n\n"
        "主 gate：两个背景的 Pearson 与 cosine 风险相关宏平均均为正；复合方向误差的 gene-cluster bootstrap "
        "95% CI 下界>0；相对 predicted magnitude 的 Δρ 点估计>0。增强主张还要求 Δρ 的 95% CI 下界>0。"
        "背景反差组至少 12/16 个基因的风险差方向与真实误差差方向一致。\n\n"
        "## 次要终点\n\n"
        "RMSE、top-25% 对 bottom-25% 错误富集、AURC、guide 一致性、敲低效率、活率、凋亡、细胞周期、"
        "通路 PROGENy 误差和 cell-line×perturbation 交互。所有次要结果明确标注，不替换失败的主终点。\n\n"
        "## 盲法与排除\n\n"
        "湿实验人员和测序方只见随机样本/guide 编码，不见风险分层和预期方向。风险映射由独立保管人保存。"
        "完成 cell QC、guide QC、pseudobulk、排除日志与哈希后才能解盲。排除只按 `tables/E143_QC_RULES.csv`，"
        "不因预测不准、方向不符或 P 值不显著而删样本。\n\n"
        "## 证据边界\n\n"
        "若只重做 HepG2/Jurkat Nadig 面板，属于技术复现；若改用 siRNA，属于跨干预模态探索；"
        "若只做 qPCR 小面板，不能证明全转录组方向风险。正式确认至少一个细胞背景必须全新。\n"
    )
    (REPORTS / "E143_DECISION_AND_HANDOFF.md").write_text(
        "# E143 决策与实验室交接\n\n"
        "## 当前能否开做\n\n"
        "计算端已经准备完毕，但正式湿实验尚不能开做。缺少的不是代码，而是实验室外部条件："
        "新细胞背景、dCas9-KRAB 稳定性、慢病毒/CRISPRi 许可、平台档期、预算和负责人。"
        "在这些信息确认前擅自给正式候选命名，会把“前瞻验证”变成事后挑选。\n\n"
        "## 推荐规模\n\n"
        f"以真实相关 ρ=0.40、双侧 α=0.05 的 Fisher 近似计算，80% 功效至少需要 {min_r04_80} 个独立基因。"
        "因此主方案采用 48 个基因，而不是把数万个细胞误当成独立样本。每基因 2 条独立 sgRNA，"
        "另加 6 条 non-targeting guide；2 个背景×3 个独立培养/转导批次，共 6 个 10x 文库。"
        "每 guide 每批目标 100 个 QC 后细胞、最低 50 个。\n\n"
        "## 两阶段执行\n\n"
        "第一阶段用 Nadig 的 24 基因面板调通感染、guide 捕获、Day 7 收样、qPCR 和活率流程。"
        "该面板来自真值解封前的风险分数，但 Nadig 真值已经被模型开发读取，所以只能做技术预实验。\n\n"
        "第二阶段至少加入一个全新背景。首选实验室已有、转导稳定的 A549、Huh7 或 Hep3B，"
        "而不是为了论文临时选择最容易成功的细胞。先做 STR、支原体和三批 baseline；"
        "再填候选模板、冻结分数与清单，最后开始扰动。若实验室已有两个可靠的新背景，"
        "两个都用新背景比“新背景+Jurkat”更强。\n\n"
        "## 必须由实验室确认的九项输入\n\n"
        "1. 可用细胞系、是否已有 dCas9-KRAB 稳定株；\n"
        "2. 慢病毒与 CRISPRi 的生物安全审批；\n"
        "3. sgRNA 载体、包装体系、筛选标记和既往滴度；\n"
        "4. qPCR、流式、10x 3′+guide capture、bulk RNA-seq 的可用平台；\n"
        "5. 预算上限和最晚完成时间；\n"
        "6. 三批独立 baseline 的提供方式；\n"
        "7. 测序平台最低细胞量、上机规格和批次安排；\n"
        "8. 湿实验负责人及独立盲法映射保管人；\n"
        "9. STR、支原体、培养、转导、排除和偏差记录模板。\n\n"
        "## 机制深入的触发条件\n\n"
        "主实验完成后，仅在“背景交互稳定、两条 guide 一致、校正 viability 后仍存在”的目标中按冻结规则选 2 个。"
        "再做 Day 3/Day 7、独立 guide、sgRNA-resistant cDNA rescue、qPCR/Western/flow 和必要的 ATAC。"
        "现阶段高风险候选集中在核糖体/核仁、线粒体翻译/ISR、基础转录与 RNA 加工；具体通路节点必须等待主实验后再冻结，"
        "不能现在先写好一个机制故事再找支持证据。\n\n"
        "## 文献依据\n\n"
        "- Nadig 数据的官方 GEO 说明了双 sgRNA、低感染率、Day 3 FACS 和 Day 7 10x 3′流程："
        "[GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667)。\n"
        "- 原研究与跨细胞背景差异：[Nature Genetics 2025](https://www.nature.com/articles/s41588-025-02169-3)。\n"
        "- 最新模型审计强调 batch-matched control、Pearson Δ、retrieval、强均值基线及 split-half 实验重复性："
        "[TxPert, Nature Biotechnology 2026](https://www.nature.com/articles/s41587-026-03113-4)。\n"
        "- 多细胞背景泛化仍是普遍难题：[Nature Methods benchmark](https://www.nature.com/articles/s41592-025-02980-0)。\n"
    )


def main() -> None:
    for directory in [OUT, TABLES, REPORTS, TEMPLATES, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    pilot = technical_pilot_candidates()
    power = power_table()
    minimum = minimum_n_table()
    slots = formal_candidate_slots()
    input_template = formal_input_template()
    layout = sample_layout()
    qc = qc_rules()
    pilot.to_csv(TABLES / "E143_NADIG_TECHNICAL_PILOT.csv", index=False)
    power.to_csv(TABLES / "E143_CORRELATION_POWER.csv", index=False)
    minimum.to_csv(TABLES / "E143_MINIMUM_GENE_COUNT.csv", index=False)
    slots.to_csv(TEMPLATES / "FORMAL_CANDIDATE_SLOTS.csv", index=False)
    input_template.to_csv(TEMPLATES / "FORMAL_CANDIDATE_INPUT.csv", index=False)
    layout.to_csv(TEMPLATES / "BLINDED_LIBRARY_LAYOUT.csv", index=False)
    qc.to_csv(TABLES / "E143_QC_RULES.csv", index=False)
    write_documents(pilot, power, minimum)
    draw_power_figure()
    draw_design_figure()
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "RUN_STATUS.json")
    status = {
        "experiment": "E143_prospective_wetlab_validation", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "server_side_package_complete_physical_experiment_awaits_external_inputs",
        "n_technical_pilot_genes": int(pilot.gene.nunique()), "n_formal_candidate_slots": len(slots),
        "planned_guides_per_gene": 2, "planned_non_targeting_guides": 6, "planned_contexts": 2,
        "planned_independent_batches_per_context": 3, "planned_10x_libraries": len(layout),
        "nadig_score_snapshot_sha256": sha256(NADIG_SCORES),
        "package_files_sha256": {str(path.relative_to(OUT)): sha256(path) for path in files},
        "physical_wetlab_executed": False, "external_inputs_required": True,
        "independent_confirmation_claim_allowed_now": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print("\nTechnical pilot:\n", pilot[["gene", "pilot_stratum", "risk_HepG2", "risk_Jurkat"]].to_string(index=False))
    print("\nMinimum gene counts:\n", minimum.to_string(index=False))


if __name__ == "__main__":
    main()
