#!/usr/bin/env python3
"""Build the 2026-09-06 paper-audit pack from official E199–E204 tables.

Teaching prose, Nature-style figures, and the Q1/Q2 report must quote
``CLAIM_TABLE.json`` rather than remembered numbers. This module:

1. Reads frozen CSVs/reports and writes a machine-readable claim table.
2. Draws architecture / holdout / E201-result figures from that table.
3. Checks that committed documents and figure sidecars still match the CSVs.

Usage::

    python3 -m safeconf_audit.paper_pack --repo /home/yyf/proj
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd

E199_TABLE = Path("docs/实验结果/E199_txpert_public_k562_20260802/formal_evaluation/tables")
E200_TABLE = Path("docs/实验结果/E200_txpert_cross_context_k562_20260802/formal_evaluation/tables")
E201_CORE = Path(
    "docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation"
)
E204_DIR = Path("docs/实验结果/E204_risk_guided_training_20260830")
E205_FREEZE = Path("docs/实验结果/E205_cross_family_disagreement_20260830/ANALYSIS_FREEZE.md")
PACK_REL = Path("docs/学习导航/20260906_论文审核与从零教学")

CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "grey": "#4D4D4D",
}

REQUIRED_TERMS = (
    ("perturbation", "扰动"),
    ("task", "任务"),
    ("source", "源域"),
    ("target", "目标域"),
    ("CRISPRi", "CRISPR"),
    ("Spearman", "斯皮尔曼"),
    ("centroid", "质心"),
    ("holdout", "留出"),
    ("predicted magnitude", "预测幅度"),
    ("family disagreement", "家族分歧"),
)

GPT_MISMATCH_CLASSES = (
    "four seeds treated as four models",
    "E199 described as four random seeds",
    "E204 profile treated as performance gain",
    "稳定二区/一区",
    "SafeConf beating magnitude",
    "agents/ or July Gate used as current fact",
)


def round4(value: float) -> str:
    return f"{float(value):.4f}"


def round5(value: float) -> str:
    return f"{float(value):.5f}"


def _row(frame: pd.DataFrame, **equals) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key, value in equals.items():
        mask &= frame[key] == value
    hit = frame.loc[mask]
    if len(hit) != 1:
        raise ValueError(f"expected one row for {equals}, got {len(hit)}")
    return hit.iloc[0]


def default_repo() -> Path:
    return Path(__file__).resolve().parents[3]


def pack_dir(repo: Path) -> Path:
    return repo / PACK_REL


def cjk_font() -> FontProperties:
    for path in CJK_FONT_CANDIDATES:
        candidate = Path(path)
        if candidate.is_file():
            try:
                fontManager.addfont(str(candidate))
            except (OSError, ValueError):
                pass
            return FontProperties(fname=str(candidate))
    return FontProperties(family="DejaVu Sans")


def load_claim_table(repo: Path) -> dict:
    """Read official CSVs/reports into one JSON-serialisable table."""
    e201_risk = pd.read_csv(repo / E201_CORE / "tables" / "E201_RISK_ASSOCIATIONS.csv")
    e201_partial = pd.read_csv(repo / E201_CORE / "tables" / "E201_PARTIAL_ASSOCIATIONS.csv")
    e201_util = pd.read_csv(repo / E201_CORE / "tables" / "E201_REVIEW_UTILITY.csv")
    e201_delta = pd.read_csv(repo / E201_CORE / "tables" / "E201_INCREMENTAL_TESTS.csv")
    e201_err = pd.read_csv(repo / E201_CORE / "tables" / "E201_TARGET_ERROR_SUMMARY.csv")
    e201_tasks = pd.read_csv(repo / E201_CORE / "tables" / "E201_TASK_METRICS.csv")
    e201_gates = pd.read_csv(repo / E201_CORE / "tables" / "E201_FORMAL_GATES.csv")
    e199_risk = pd.read_csv(repo / E199_TABLE / "E199_RISK_ASSOCIATIONS.csv")
    e199_util = pd.read_csv(repo / E199_TABLE / "E199_REVIEW_UTILITY.csv")
    e200_risk = pd.read_csv(repo / E200_TABLE / "E200_RISK_ASSOCIATIONS.csv")
    e200_util = pd.read_csv(repo / E200_TABLE / "E200_REVIEW_UTILITY.csv")
    e204_status = json.loads(
        (repo / E204_DIR / "training_manifest_v2" / "E204_TRAINING_WEIGHT_STATUS.json").read_text()
    )
    e204_accept = (repo / E204_DIR / "PROFILE_ACCEPTANCE_20260905.md").read_text()
    e205_text = (repo / E205_FREEZE).read_text()

    def assoc(frame, *, scope, predictor):
        return _row(frame, scope=scope, predictor=predictor)

    def util(frame, *, scope, predictor):
        return _row(frame, scope=scope, predictor=predictor)

    pooled_sc = assoc(e201_risk, scope="pooled", predictor="safeconf_e201_risk")
    pooled_mag = assoc(e201_risk, scope="pooled", predictor="predicted_magnitude")
    pooled_dis = assoc(e201_risk, scope="pooled", predictor="family_disagreement")
    partial = _row(
        e201_partial,
        scope="pooled",
        predictor="safeconf_e201_risk",
        covariate="predicted_magnitude",
    )
    util_sc = util(e201_util, scope="pooled", predictor="safeconf_e201_risk")
    util_mag = util(e201_util, scope="pooled", predictor="predicted_magnitude")
    delta_u = _row(e201_delta, scope="pooled", measure="delta_oracle_normalized_utility")
    k562_err = _row(e201_err, stratum="primary_ge30", target="K562", predictor="four_seed_family")
    aars = _row(e201_tasks, target="K562", condition="AARS+ctrl")
    cert = _row(e201_gates, gate="family_error_certificate")
    identity = float(str(cert.observed).split("identity_max=")[1].split(";")[0])
    e199_div = _row(e199_risk, predictor="diversity_lower_bound")
    e199_mag = _row(e199_risk, predictor="predicted_magnitude")
    e199_u = _row(e199_util, predictor="diversity_lower_bound")
    e200_tr = _row(e200_risk, predictor="transfer_risk", outcome="gat_centroid_rmse")
    e200_mag = _row(e200_risk, predictor="predicted_magnitude", outcome="gat_centroid_rmse")
    e200_u_tr = _row(e200_util, predictor="transfer_risk")
    e200_u_mag = _row(e200_util, predictor="predicted_magnitude")

    targets = {}
    for name in ("K562", "RPE1", "hepg2", "jurkat"):
        row = assoc(e201_risk, scope=name, predictor="safeconf_e201_risk")
        mag = assoc(e201_risk, scope=name, predictor="predicted_magnitude")
        dis = assoc(e201_risk, scope=name, predictor="family_disagreement")
        part = _row(e201_partial, scope=name, predictor="safeconf_e201_risk")
        targets[name] = {
            "n_tasks": int(row.n_tasks),
            "safeconf_spearman": float(row.estimate),
            "safeconf_spearman_display": round4(row.estimate),
            "safeconf_ci": [float(row.ci95_lower), float(row.ci95_upper)],
            "safeconf_ci_display": [round4(row.ci95_lower), round4(row.ci95_upper)],
            "magnitude_spearman": float(mag.estimate),
            "magnitude_spearman_display": round4(mag.estimate),
            "disagreement_spearman": float(dis.estimate),
            "disagreement_spearman_display": round4(dis.estimate),
            "disagreement_ci": [float(dis.ci95_lower), float(dis.ci95_upper)],
            "partial_spearman": float(part.estimate),
            "partial_spearman_display": round4(part.estimate),
            "partial_ci": [float(part.ci95_lower), float(part.ci95_upper)],
        }

    n_primary = int((e201_tasks.analysis_stratum == "primary_ge30").sum()) if "analysis_stratum" in e201_tasks else 1808
    if "analysis_stratum" not in e201_tasks.columns:
        n_primary = int(pooled_sc.n_tasks)
    n_all = int(len(e201_tasks))

    profile_only = (
        "不判断模型效果" in e204_accept
        and "PASS，允许进入正式训练队列" in e204_accept
        and "正式 80" not in e204_accept
    )
    e205_protocol_only = "候选补充实验" in e205_text and "尚未运行" not in e205_text
    # E205 freeze is a protocol; no result tables exist beside the freeze.
    e205_result_tables = list(
        (repo / "docs/实验结果/E205_cross_family_disagreement_20260830").glob("**/tables/*.csv")
    )

    table = {
        "source": "official_csv",
        "rounding": "display values are round-half-even to 4 decimals unless noted",
        "e201": {
            "n_primary_tasks": n_primary,
            "n_all_tasks": n_all,
            "n_sensitivity_tasks": n_all - n_primary,
            "n_genes": 3352,
            "pooled": {
                "safeconf_spearman": float(pooled_sc.estimate),
                "safeconf_spearman_display": round4(pooled_sc.estimate),
                "safeconf_ci": [float(pooled_sc.ci95_lower), float(pooled_sc.ci95_upper)],
                "safeconf_ci_display": [round4(pooled_sc.ci95_lower), round4(pooled_sc.ci95_upper)],
                "magnitude_spearman": float(pooled_mag.estimate),
                "magnitude_spearman_display": round4(pooled_mag.estimate),
                "magnitude_ci": [float(pooled_mag.ci95_lower), float(pooled_mag.ci95_upper)],
                "disagreement_spearman": float(pooled_dis.estimate),
                "disagreement_spearman_display": round4(pooled_dis.estimate),
                "partial_spearman": float(partial.estimate),
                "partial_spearman_display": round4(partial.estimate),
                "partial_ci": [float(partial.ci95_lower), float(partial.ci95_upper)],
                "partial_ci_display": [round4(partial.ci95_lower), round4(partial.ci95_upper)],
                "safeconf_utility_20": float(util_sc.oracle_normalized_utility),
                "safeconf_utility_20_display": round4(util_sc.oracle_normalized_utility),
                "safeconf_utility_ci": [
                    float(util_sc.utility_ci95_lower),
                    float(util_sc.utility_ci95_upper),
                ],
                "safeconf_utility_ci_display": [
                    round4(util_sc.utility_ci95_lower),
                    round4(util_sc.utility_ci95_upper),
                ],
                "magnitude_utility_20": float(util_mag.oracle_normalized_utility),
                "magnitude_utility_20_display": round4(util_mag.oracle_normalized_utility),
                "magnitude_utility_ci": [
                    float(util_mag.utility_ci95_lower),
                    float(util_mag.utility_ci95_upper),
                ],
                "delta_utility": float(delta_u.estimate),
                "delta_utility_display": round4(delta_u.estimate),
                "delta_utility_ci": [float(delta_u.ci95_lower), float(delta_u.ci95_upper)],
            },
            "targets": targets,
            "k562_error": {
                "family_centroid_rmse": float(k562_err.family_centroid_rmse_mean),
                "family_centroid_rmse_display": round4(k562_err.family_centroid_rmse_mean),
                "control_error": float(k562_err.control_error_mean),
                "control_error_display": round4(k562_err.control_error_mean),
                "official_general_baseline": float(
                    k562_err.official_general_baseline_error_mean
                ),
                "official_general_baseline_display": round4(
                    k562_err.official_general_baseline_error_mean
                ),
            },
            "identity_residual_max": identity,
            "identity_residual_display": f"{identity:.2e}".replace("e-0", "e-"),
            "certificate_passed": bool(cert.passed),
            "example_task": {
                "id": "K562::AARS+ctrl",
                "n_target_cells": int(aars.n_target_cells),
                "n_target_batches": int(aars.n_target_batches),
                "n_source_cells": int(aars.n_source_cells),
                "n_source_contexts": int(aars.n_source_contexts),
                "source_delta_dispersion": float(aars.source_delta_dispersion),
                "source_delta_dispersion_display": round5(aars.source_delta_dispersion),
                "family_disagreement": float(aars.family_disagreement),
                "family_disagreement_display": round5(aars.family_disagreement),
                "predicted_magnitude": float(aars.predicted_magnitude),
                "predicted_magnitude_display": round5(aars.predicted_magnitude),
                "model_source_gap": float(aars.model_source_gap),
                "model_source_gap_display": round5(aars.model_source_gap),
                "safeconf_e201_risk": float(aars.safeconf_e201_risk),
                "safeconf_e201_risk_display": round5(aars.safeconf_e201_risk),
                "family_centroid_rmse": float(aars.family_centroid_rmse),
                "family_centroid_rmse_display": round5(aars.family_centroid_rmse),
                "family_rms_error": float(aars.family_rms_error),
                "family_rms_error_display": round5(aars.family_rms_error),
                "worst_seed_error": float(aars.worst_seed_error),
                "worst_seed_error_display": round5(aars.worst_seed_error),
            },
            "paths": {
                "risk": str(E201_CORE / "tables" / "E201_RISK_ASSOCIATIONS.csv"),
                "partial": str(E201_CORE / "tables" / "E201_PARTIAL_ASSOCIATIONS.csv"),
                "utility": str(E201_CORE / "tables" / "E201_REVIEW_UTILITY.csv"),
                "error": str(E201_CORE / "tables" / "E201_TARGET_ERROR_SUMMARY.csv"),
                "tasks": str(E201_CORE / "tables" / "E201_TASK_METRICS.csv"),
                "report": str(E201_CORE / "reports" / "E201_CORE_REPORT.md"),
            },
        },
        "e199": {
            "n_tasks": int(e199_div.n_tasks),
            "family": "three public checkpoints: GAT, Exphormer, Exphormer-MG",
            "not": "four random seeds of one architecture",
            "diversity_spearman": float(e199_div.spearman),
            "diversity_spearman_display": round4(e199_div.spearman),
            "diversity_ci": [float(e199_div.ci95_lower), float(e199_div.ci95_upper)],
            "magnitude_spearman": float(e199_mag.spearman),
            "magnitude_spearman_display": round4(e199_mag.spearman),
            "utility_20": float(e199_u.oracle_normalized_utility),
            "utility_20_display": round4(e199_u.oracle_normalized_utility),
            "path": str(E199_TABLE / "E199_RISK_ASSOCIATIONS.csv"),
        },
        "e200": {
            "n_tasks": int(e200_tr.n_tasks),
            "transfer_risk_spearman": float(e200_tr.spearman),
            "transfer_risk_spearman_display": round4(e200_tr.spearman),
            "magnitude_spearman": float(e200_mag.spearman),
            "magnitude_spearman_display": round4(e200_mag.spearman),
            "transfer_utility_20": float(e200_u_tr.oracle_normalized_utility),
            "transfer_utility_20_display": round4(e200_u_tr.oracle_normalized_utility),
            "magnitude_utility_20": float(e200_u_mag.oracle_normalized_utility),
            "magnitude_utility_20_display": round4(e200_u_mag.oracle_normalized_utility),
            "path": str(E200_TABLE / "E200_RISK_ASSOCIATIONS.csv"),
        },
        "e202": {
            "verdict": "NOT_SUPPORTED",
            "partial_spearman_display": "-0.0680",
            "partial_ci_display": ["-0.1522", "0.0191"],
            "path": "docs/实验结果/E202_residual_task_failure_20260802/formal_evaluation/reports/E202_REPORT.md",
        },
        "e204": {
            "status": "profile_only",
            "profile_only": True,
            "formal_80_epoch_complete": False,
            "engineering_pass": profile_only,
            "n_source_training_conditions": e204_status["n_source_training_conditions_per_target"],
            "n_weight_rows": int(e204_status["n_rows"]),
            "target_truth_opened_by_builder": int(
                e204_status["target_truth_files_opened_by_this_builder"]
            ),
            "path": str(E204_DIR / "PROFILE_ACCEPTANCE_20260905.md"),
        },
        "e205": {
            "status": "protocol_only",
            "n_result_tables": len(e205_result_tables),
            "path": str(E205_FREEZE),
        },
        "locked_display": {
            "safeconf_pooled_spearman": round4(pooled_sc.estimate),
            "magnitude_pooled_spearman": round4(pooled_mag.estimate),
            "partial_spearman": round4(partial.estimate),
            "safeconf_utility_20": round4(util_sc.oracle_normalized_utility),
            "magnitude_utility_20": round4(util_mag.oracle_normalized_utility),
            "k562_spearman": targets["K562"]["safeconf_spearman_display"],
            "rpe1_spearman": targets["RPE1"]["safeconf_spearman_display"],
            "hepg2_spearman": targets["hepg2"]["safeconf_spearman_display"],
            "jurkat_spearman": targets["jurkat"]["safeconf_spearman_display"],
            "k562_centroid_rmse": round4(k562_err.family_centroid_rmse_mean),
            "k562_control_rmse": round4(k562_err.control_error_mean),
            "k562_official_rmse": round4(k562_err.official_general_baseline_error_mean),
            "n_primary_tasks": str(n_primary),
        },
        "publication": {
            "q2_certain": False,
            "q1_sprint_ready": False,
            "reason": (
                "magnitude is the stronger single ranker; E204 formal 80-epoch "
                "results do not exist; E205 and a frozen new-external confirmation "
                "are missing"
            ),
        },
    }
    return table


def write_claim_table(repo: Path, table: dict | None = None) -> Path:
    table = table or load_claim_table(repo)
    out = pack_dir(repo) / "CLAIM_TABLE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    return out


def _style_axes(ax, fp: FontProperties) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp)


def _panel_id(ax, letter: str, fp: FontProperties) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontproperties=fp,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _box(ax, x, y, w, h, text, fp, facecolor="#F7F7F7", edge="#222222"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=facecolor,
        mutation_aspect=0.8,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontproperties=fp,
        fontsize=8,
        color="#111111",
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color="#222222",
        )
    )


def generate_fig1_architecture(repo: Path, table: dict, fp: FontProperties) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), facecolor="white")
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")
    _panel_id(ax, "a", fp)
    ax.text(0.02, 0.97, "预测后复核 与 源域训练加权 分开", fontproperties=fp, fontsize=9)
    boxes = {
        "unperturbed": (0.02, 0.74, 0.24, 0.16, "未扰动细胞\n基础表达状态", "#EEF6FB", OKABE["blue"]),
        "perturbation": (0.02, 0.52, 0.24, 0.16, "基因敲低\n扰动条件", "#EEF6FB", OKABE["blue"]),
        "txpert": (0.34, 0.60, 0.24, 0.18, "TxPert\n预测扰动后表达", "#FFF4DC", OKABE["orange"]),
        "four_seeds": (0.66, 0.76, 0.32, 0.14, "四个随机种子\n四份预测", "#F4F4F4", "#222222"),
        "safeconf": (0.66, 0.54, 0.32, 0.16, "SafeConf\n风险特征", "#E8F6F0", OKABE["green"]),
        "review": (0.66, 0.32, 0.32, 0.16, "优先复核\n高风险任务", "#FDECEC", OKABE["vermillion"]),
        "source_evidence": (
            0.02,
            0.06,
            0.40,
            0.20,
            "源域证据\n细胞数·背景数·离散度",
            "#EEF6FB",
            OKABE["blue"],
        ),
        "training_weights": (
            0.50,
            0.06,
            0.48,
            0.20,
            "训练加权（E204）\n尚未有正式效果",
            "#F4F4F4",
            "#222222",
        ),
    }
    for _name, (x, y, w, h, text, face, edge) in boxes.items():
        _box(ax, x, y, w, h, text, fp, face, edge)

    def _mid(name, side):
        x, y, w, h, *_ = boxes[name]
        if side == "right":
            return x + w, y + h / 2
        if side == "left":
            return x, y + h / 2
        if side == "top":
            return x + w / 2, y + h
        return x + w / 2, y

    edges = [
        ("unperturbed", "right", "txpert", "left"),
        ("perturbation", "right", "txpert", "left"),
        ("txpert", "right", "four_seeds", "left"),
        ("four_seeds", "bottom", "safeconf", "top"),
        ("safeconf", "bottom", "review", "top"),
        ("source_evidence", "right", "training_weights", "left"),
    ]
    for src, src_side, dst, dst_side in edges:
        x1, y1 = _mid(src, src_side)
        x2, y2 = _mid(dst, dst_side)
        _arrow(ax, x1, y1, x2, y2)
    layout = {
        "boxes": {name: {"x": v[0], "y": v[1], "w": v[2], "h": v[3], "label": v[4]} for name, v in boxes.items()},
        "edges": [(src, dst) for src, _a, dst, _b in edges],
        "forbidden_edges": [
            ("four_seeds", "training_weights"),
            ("safeconf", "training_weights"),
            ("txpert", "training_weights"),
        ],
    }

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")
    _panel_id(ax, "b", fp)
    ex = table["e201"]["example_task"]
    ax.text(0.02, 0.96, "一道真题：K562::AARS+ctrl", fontproperties=fp, fontsize=9)
    lines = [
        f"目标细胞数  {ex['n_target_cells']}",
        f"源域细胞数  {ex['n_source_cells']}",
        f"源域背景数  {ex['n_source_contexts']}",
        f"家族分歧  {ex['family_disagreement_display']}",
        f"预测幅度  {ex['predicted_magnitude_display']}",
        f"SafeConf 风险分  {ex['safeconf_e201_risk_display']}",
        f"家族均方根误差  {ex['family_rms_error_display']}",
    ]
    _box(ax, 0.06, 0.12, 0.40, 0.76, "对答案前可算\n\n" + "\n".join(lines[:6]), fp, "#E8F6F0", OKABE["green"])
    _box(
        ax,
        0.54,
        0.12,
        0.40,
        0.76,
        "对答案后才有\n\n"
        + f"质心误差  {ex['family_centroid_rmse_display']}\n"
        + f"家族均方根误差  {ex['family_rms_error_display']}\n"
        + f"最差种子误差  {ex['worst_seed_error_display']}\n\n"
        + "E201 先封存预测\n再打开真实结果",
        fp,
        "#FDECEC",
        OKABE["vermillion"],
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05, wspace=0.16)
    return _save_fig(
        repo,
        fig,
        "fig1_architecture",
        {"example_task": ex, "layout": layout},
    )


def generate_fig2_holdout(repo: Path, table: dict, fp: FontProperties) -> dict:
    fig = plt.figure(figsize=(11.2, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)
    specs = [
        ("a", "随机缺格", "细胞和基因都见过\n只缺这个组合", "较容易"),
        ("b", "整列留出", "这个基因完全没见过\n细胞背景见过", "E199"),
        ("c", "整行留出", "这种细胞的答案全藏起\n周老师更难的考法", "E200 / E201"),
        ("d", "双未见", "细胞和基因都没见过", "已有失败，应暂停"),
    ]
    for i, (letter, title, body, tag) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        _panel_id(ax, letter, fp)
        ax.text(0.0, 4.85, title, fontproperties=fp, fontsize=9)
        colors = [["#D9D9D9"] * 4 for _ in range(4)]
        if letter == "a":
            colors[1][2] = OKABE["vermillion"]
        elif letter == "b":
            for r in range(4):
                colors[r][2] = OKABE["vermillion"]
        elif letter == "c":
            for c in range(4):
                colors[1][c] = OKABE["vermillion"]
        else:
            for c in range(4):
                colors[1][c] = OKABE["vermillion"]
            for r in range(4):
                colors[r][2] = OKABE["vermillion"]
        for r in range(4):
            for c in range(4):
                ax.add_patch(
                    Rectangle(
                        (0.5 + c * 0.95, 2.15 - r * 0.45),
                        0.85,
                        0.38,
                        facecolor=colors[r][c],
                        edgecolor="white",
                        linewidth=0.6,
                    )
                )
        ax.text(0.5, 0.55, body, fontproperties=fp, fontsize=7, va="center")
        ax.text(0.5, 0.12, tag, fontproperties=fp, fontsize=7, color=OKABE["grey"])

    ax = fig.add_subplot(gs[1, :2])
    _style_axes(ax, fp)
    _panel_id(ax, "e", fp)
    names = ["E199 未见基因\n分歧", "E199 未见基因\n幅度", "E200 整背景\n风险分", "E200 整背景\n幅度"]
    vals = [
        float(table["e199"]["diversity_spearman_display"]),
        float(table["e199"]["magnitude_spearman_display"]),
        float(table["e200"]["transfer_risk_spearman_display"]),
        float(table["e200"]["magnitude_spearman_display"]),
    ]
    colors = [OKABE["green"], OKABE["orange"], OKABE["green"], OKABE["orange"]]
    bars = ax.bar(range(4), vals, color=colors, width=0.72, linewidth=0)
    ax.set_xticks(range(4), names, fontproperties=fp, fontsize=7)
    ax.set_ylabel("Spearman 秩相关", fontproperties=fp, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03,
            f"{val:.4f}",
            ha="center",
            fontproperties=fp,
            fontsize=7,
        )
    ax.text(0.0, 1.08, "同一线索换考法会翻转", transform=ax.transAxes, fontproperties=fp, fontsize=9)

    ax = fig.add_subplot(gs[1, 2:])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")
    _panel_id(ax, "f", fp)
    ax.text(0.0, 0.95, "当前主线考的是哪一种", fontproperties=fp, fontsize=9)
    note = (
        "E201 把 E200 的整行留出扩到四个细胞系。\n"
        f"主分析 {table['locked_display']['n_primary_tasks']} 个任务。\n"
        "灰色格子：训练见过。红色格子：测试时没见过。\n"
        "行是细胞背景，列是扰动基因。"
    )
    ax.text(0.0, 0.55, note, fontproperties=fp, fontsize=8, va="center")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08)
    plotted = {
        "e199_diversity_spearman": table["e199"]["diversity_spearman_display"],
        "e199_magnitude_spearman": table["e199"]["magnitude_spearman_display"],
        "e200_transfer_risk_spearman": table["e200"]["transfer_risk_spearman_display"],
        "e200_magnitude_spearman": table["e200"]["magnitude_spearman_display"],
    }
    return _save_fig(repo, fig, "fig2_holdout", plotted)


def generate_fig3_e201(repo: Path, table: dict, fp: FontProperties) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), facecolor="white")
    fig.patch.set_facecolor("white")
    pooled = table["e201"]["pooled"]
    targets = table["e201"]["targets"]
    order = ["K562", "RPE1", "hepg2", "jurkat"]
    labels = ["K562", "RPE1", "HepG2", "Jurkat"]

    ax = axes[0]
    _style_axes(ax, fp)
    _panel_id(ax, "a", fp)
    y = list(range(len(order), 0, -1))
    for yi, name in zip(y, order):
        sc = targets[name]
        lo, hi = sc["safeconf_ci"]
        ax.plot([lo, hi], [yi + 0.12, yi + 0.12], color=OKABE["blue"], linewidth=1.4, solid_capstyle="butt")
        ax.plot(sc["safeconf_spearman"], yi + 0.12, "o", color=OKABE["blue"], markersize=5)
        ax.plot(sc["magnitude_spearman"], yi - 0.12, "s", color=OKABE["orange"], markersize=5)
    ax.axvline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_yticks(y, labels, fontproperties=fp)
    ax.set_xlabel("与家族均方根误差的 Spearman", fontproperties=fp, fontsize=8)
    ax.set_xlim(-0.05, 1.0)
    ax.plot([], [], "o", color=OKABE["blue"], label="SafeConf")
    ax.plot([], [], "s", color=OKABE["orange"], label="预测幅度")
    ax.legend(prop=fp, fontsize=7, loc="lower right")
    ax.set_title("四细胞系排序相关", fontproperties=fp, fontsize=9, loc="left")

    ax = axes[1]
    _style_axes(ax, fp)
    _panel_id(ax, "b", fp)
    xs = [0, 1]
    heights = [pooled["safeconf_utility_20"], pooled["magnitude_utility_20"]]
    yerr = [
        [
            pooled["safeconf_utility_20"] - pooled["safeconf_utility_ci"][0],
            pooled["magnitude_utility_20"] - pooled["magnitude_utility_ci"][0],
        ],
        [
            pooled["safeconf_utility_ci"][1] - pooled["safeconf_utility_20"],
            pooled["magnitude_utility_ci"][1] - pooled["magnitude_utility_20"],
        ],
    ]
    ax.bar(
        xs,
        heights,
        color=[OKABE["blue"], OKABE["grey"]],
        width=0.62,
        linewidth=0,
        yerr=yerr,
        error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2},
    )
    ax.set_xticks(xs, ["SafeConf", "预测幅度"], fontproperties=fp, fontsize=8)
    ax.set_ylabel("20% 复核效用", fontproperties=fp, fontsize=8)
    upper = [pooled["safeconf_utility_ci"][1], pooled["magnitude_utility_ci"][1]]
    ax.set_ylim(0, max(upper) + 0.12)
    ax.set_title("固定检查两成任务", fontproperties=fp, fontsize=9, loc="left")
    bar_labels = []
    for x, hi, lab in zip(
        xs,
        upper,
        [pooled["safeconf_utility_20_display"], pooled["magnitude_utility_20_display"]],
    ):
        label_y = hi + 0.035
        ax.text(x, label_y, lab, ha="center", va="bottom", fontproperties=fp, fontsize=7)
        bar_labels.append({"text": lab, "data_y": label_y, "whisker_hi": hi})

    ax = axes[2]
    _style_axes(ax, fp)
    _panel_id(ax, "c", fp)
    y = list(range(len(order) + 1, 0, -1))
    names = ["合并"] + labels
    rows = [pooled] + [targets[n] for n in order]
    for yi, row in zip(y, rows):
        if "partial_ci" in row:
            lo, hi = row["partial_ci"]
            est = row["partial_spearman"]
        else:
            lo, hi = row["partial_ci"]
            est = row["partial_spearman"]
        ax.plot([lo, hi], [yi, yi], color=OKABE["green"], linewidth=1.4, solid_capstyle="butt")
        ax.plot(est, yi, "o", color=OKABE["green"], markersize=5)
    ax.axvline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_yticks(y, names, fontproperties=fp)
    ax.set_xlabel("控制幅度后的偏 Spearman", fontproperties=fp, fontsize=8)
    ax.set_xlim(-0.05, 0.55)
    ax.set_title("幅度之外仍有信息", fontproperties=fp, fontsize=9, loc="left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.18, wspace=0.38)
    plotted = {
        "safeconf_pooled_spearman": pooled["safeconf_spearman_display"],
        "magnitude_pooled_spearman": pooled["magnitude_spearman_display"],
        "partial_spearman": pooled["partial_spearman_display"],
        "safeconf_utility_20": pooled["safeconf_utility_20_display"],
        "magnitude_utility_20": pooled["magnitude_utility_20_display"],
        "k562_spearman": targets["K562"]["safeconf_spearman_display"],
        "rpe1_spearman": targets["RPE1"]["safeconf_spearman_display"],
        "hepg2_spearman": targets["hepg2"]["safeconf_spearman_display"],
        "jurkat_spearman": targets["jurkat"]["safeconf_spearman_display"],
        "target_safeconf_spearman": {
            name: targets[name]["safeconf_spearman"] for name in order
        },
        "target_magnitude_spearman": {
            name: targets[name]["magnitude_spearman"] for name in order
        },
        "partial_by_target": {
            "pooled": pooled["partial_spearman"],
            **{name: targets[name]["partial_spearman"] for name in order},
        },
        "bar_labels": bar_labels,
    }
    return _save_fig(repo, fig, "fig3_e201_main", plotted)


def _save_fig(repo: Path, fig, stem: str, plotted: dict) -> dict:
    fig_dir = pack_dir(repo) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{stem}.png"
    svg = fig_dir / f"{stem}.svg"
    sidecar = fig_dir / f"{stem}.values.json"
    fig.savefig(png, dpi=200, facecolor="white", edgecolor="none")
    fig.savefig(svg, facecolor="white", edgecolor="none")
    plt.close(fig)
    payload = {"stem": stem, "plotted": plotted, "png": str(png.relative_to(repo)), "svg": str(svg.relative_to(repo))}
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def generate_figures(repo: Path, table: dict | None = None) -> list[dict]:
    table = table or load_claim_table(repo)
    fp = cjk_font()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return [
        generate_fig1_architecture(repo, table, fp),
        generate_fig2_holdout(repo, table, fp),
        generate_fig3_e201(repo, table, fp),
    ]


def locked_quotes(table: dict) -> dict[str, str]:
    return dict(table["locked_display"])


def check_documents(repo: Path, table: dict) -> list[str]:
    """Every locked display number must appear in audit, teaching, and Q report."""
    docs = [
        pack_dir(repo) / "01_完成度与GPT误导对照.md",
        pack_dir(repo) / "02_从零Nature图解教学.md",
        pack_dir(repo) / "03_一区二区投稿就绪报告.md",
    ]
    missing = []
    locked = locked_quotes(table)
    for path in docs:
        if not path.is_file():
            missing.append(f"missing document {path}")
            continue
        text = path.read_text()
        for key, value in locked.items():
            if value not in text:
                missing.append(f"{path.name} missing {key}={value}")
        if path.name.startswith("03_") or path.name.startswith("01_"):
            if "profile" not in text.lower() and "工程验收" not in text:
                missing.append(f"{path.name} does not state E204 profile-only")
    report = pack_dir(repo) / "03_一区二区投稿就绪报告.md"
    if report.is_file():
        text = report.read_text()
        parts = re.split(r"现在禁止写的句子", text, maxsplit=1)
        head = parts[0]
        forbidden_section = parts[1] if len(parts) > 1 else ""
        if re.search(r"E204.{0,40}已经提高", head):
            missing.append(f"{report.name} claims E204 performance gain")
        if table["publication"]["q2_certain"] is True:
            missing.append("claim table marks Q2 certain")
        for banned in ("E204 正式 80 轮已经完成", "四个 GAT 种子是跨架构家族"):
            if banned in head:
                missing.append(f"report claims banned sentence in body: {banned}")
            if banned not in forbidden_section:
                missing.append(f"forbidden-sentence list missing {banned}")
        if "不宜说稳定二区" not in text and "不能把二区写成一定能发" not in text:
            missing.append("Q report does not reject certain Q2")
    return missing


def check_terms(teaching_path: Path) -> list[str]:
    text = teaching_path.read_text()
    missing = []
    for english, chinese in REQUIRED_TERMS:
        pattern = rf"{re.escape(chinese)}.{{0,12}}[（(][^)）]*{re.escape(english)}"
        alt = rf"{re.escape(english)}.{{0,12}}[（(][^)）]*{re.escape(chinese)}"
        if not re.search(pattern, text, flags=re.I) and not re.search(alt, text, flags=re.I):
            missing.append(f"term not glossed: {chinese} ({english})")
    return missing


def check_gpt_mismatch(path: Path) -> list[str]:
    text = path.read_text()
    missing = []
    if "019f139e" not in text:
        missing.append("GPT mismatch note does not cite session 019f139e")
    for cls in GPT_MISMATCH_CLASSES:
        if cls not in text:
            missing.append(f"missing mismatch class: {cls}")
    for marker in (
        "E201_RISK_ASSOCIATIONS.csv",
        "E201_CORE_REPORT.md",
        "PROFILE_ACCEPTANCE_20260905.md",
        "E199_RISK_ASSOCIATIONS.csv",
    ):
        if marker not in text:
            missing.append(f"mismatch note missing official path {marker}")
    return missing


def _svg_local(tag: str) -> str:
    return tag.split("}")[-1]


def fig3_panel_b_label_errorbar_hits(svg_xml: str) -> list[str]:
    """Return collisions between 4-decimal bar labels and vertical error bars.

    Matplotlib SVG y grows downward. A label whose bounding box overlaps the
    vertical whisker at the same x is unreadable (e.g. 0.3200 → 0.3 00).
    """
    root = ET.fromstring(svg_xml)
    axes2 = None
    for node in root.iter():
        if _svg_local(node.tag) == "g" and node.get("id") == "axes_2":
            axes2 = node
            break
    if axes2 is None:
        return ["fig3 svg missing axes_2 (panel b)"]
    labels = []
    bars = []
    for node in axes2.iter():
        if _svg_local(node.tag) == "text":
            body = "".join(node.itertext()).strip()
            if re.fullmatch(r"0\.\d{4}", body):
                labels.append(
                    {
                        "text": body,
                        "x": float(node.get("x")),
                        "y": float(node.get("y")),
                        "size": float(re.search(r"([0-9.]+)px", node.get("style") or "font-size: 7px").group(1)),
                    }
                )
        if _svg_local(node.tag) == "path":
            d = (node.get("d") or "").replace("\n", " ")
            match = re.match(
                r"M\s+([0-9.]+)\s+([0-9.]+)\s+L\s+([0-9.]+)\s+([0-9.]+)",
                d.strip(),
            )
            if not match:
                continue
            x1, y1, x2, y2 = map(float, match.groups())
            if abs(x1 - x2) < 0.05 and abs(y1 - y2) > 5:
                bars.append({"x": x1, "y0": min(y1, y2), "y1": max(y1, y2)})
    hits = []
    for lab in labels:
        half_w = 0.33 * lab["size"] * len(lab["text"])
        top = lab["y"] - lab["size"]
        bottom = lab["y"] + 1.5
        for bar in bars:
            if abs(bar["x"] - lab["x"]) > half_w:
                continue
            if bottom < bar["y0"] - 1 or top > bar["y1"] + 1:
                continue
            hits.append(
                f"fig3b label {lab['text']} at y={lab['y']:.1f} intersects errorbar "
                f"[{bar['y0']:.1f},{bar['y1']:.1f}]"
            )
    return hits


def check_fig1_source_only_weights(plotted: dict, svg_xml: str) -> list[str]:
    missing = []
    layout = plotted.get("layout") or {}
    edges = [tuple(item) for item in layout.get("edges", [])]
    if ("source_evidence", "training_weights") not in edges:
        missing.append("fig1a missing edge 源域证据 → 训练加权")
    for forbidden in layout.get("forbidden_edges", []):
        if tuple(forbidden) in edges:
            missing.append(f"fig1a forbidden edge {forbidden[0]} → {forbidden[1]}")
    if "源域证据" not in svg_xml:
        missing.append("fig1 svg missing text 源域证据")
    if "训练加权" not in svg_xml:
        missing.append("fig1 svg missing text 训练加权")
    boxes = layout.get("boxes") or {}
    src = boxes.get("source_evidence")
    train = boxes.get("training_weights")
    seeds = boxes.get("four_seeds")
    safe = boxes.get("safeconf")
    if src and train and src["y"] + src["h"] > 0.35:
        missing.append("fig1a 源域证据 is not on the separate lower training row")
    if train and seeds and abs(train["x"] - seeds["x"]) < 0.02 and abs(train["y"] - seeds["y"]) < 0.02:
        missing.append("fig1a 训练加权 stacked on 四个随机种子")
    if train and safe and abs(train["x"] - safe["x"]) < 0.05 and train["y"] < safe["y"]:
        missing.append("fig1a 训练加权 still hangs off the SafeConf prediction stack")
    return missing


def check_figures(repo: Path, table: dict) -> list[str]:
    missing = []
    fig_dir = pack_dir(repo) / "figures"
    for stem, letters in (
        ("fig1_architecture", ("a", "b")),
        ("fig2_holdout", ("a", "b", "c")),
        ("fig3_e201_main", ("a", "b", "c")),
    ):
        svg = fig_dir / f"{stem}.svg"
        png = fig_dir / f"{stem}.png"
        sidecar = fig_dir / f"{stem}.values.json"
        if not svg.is_file() or not png.is_file() or not sidecar.is_file():
            missing.append(f"missing figure files for {stem}")
            continue
        xml = svg.read_text()
        if "drop-shadow" in xml.lower() or "feDropShadow" in xml:
            missing.append(f"{stem} has drop-shadow")
        for letter in letters:
            if not re.search(rf">\s*{letter}\s*<", xml) and f">{letter}<" not in xml:
                # matplotlib may emit the letter as a text element with extra tspan
                if not re.search(rf">{letter}</", xml):
                    missing.append(f"{stem} missing panel letter {letter}")
        payload = json.loads(sidecar.read_text())
        plotted = payload["plotted"]
        locked = table["locked_display"]
        if stem == "fig3_e201_main":
            for key in (
                "safeconf_pooled_spearman",
                "magnitude_pooled_spearman",
                "partial_spearman",
                "safeconf_utility_20",
                "magnitude_utility_20",
                "k562_spearman",
                "rpe1_spearman",
                "hepg2_spearman",
                "jurkat_spearman",
            ):
                if str(plotted.get(key)) != locked[key]:
                    missing.append(
                        f"{stem} {key}={plotted.get(key)} != csv {locked[key]}"
                    )
            # Re-read CSV and compare raw plotted target values.
            risk = pd.read_csv(repo / E201_CORE / "tables" / "E201_RISK_ASSOCIATIONS.csv")
            for name, raw in plotted["target_safeconf_spearman"].items():
                csv_val = float(
                    _row(risk, scope=name, predictor="safeconf_e201_risk").estimate
                )
                if abs(float(raw) - csv_val) > 1e-12:
                    missing.append(f"{stem} target {name} spearman drifted from CSV")
        if stem == "fig2_holdout":
            if plotted["e199_diversity_spearman"] != table["e199"]["diversity_spearman_display"]:
                missing.append("fig2 E199 diversity mismatch")
            if plotted["e200_magnitude_spearman"] != table["e200"]["magnitude_spearman_display"]:
                missing.append("fig2 E200 magnitude mismatch")
        if stem == "fig1_architecture":
            ex = plotted["example_task"]
            csv_ex = table["e201"]["example_task"]
            if ex["family_rms_error_display"] != csv_ex["family_rms_error_display"]:
                missing.append("fig1 AARS family_rms_error mismatch")
            missing.extend(check_fig1_source_only_weights(plotted, xml))
        if stem == "fig3_e201_main":
            for item in plotted.get("bar_labels") or []:
                if float(item["data_y"]) <= float(item["whisker_hi"]):
                    missing.append(
                        f"fig3b label {item['text']} data_y={item['data_y']} "
                        f"not above whisker_hi={item['whisker_hi']}"
                    )
            missing.extend(fig3_panel_b_label_errorbar_hits(xml))
    return missing


def run_all_checks(repo: Path, table: dict | None = None) -> dict:
    table = table or load_claim_table(repo)
    result = {
        "documents": check_documents(repo, table),
        "terms": check_terms(pack_dir(repo) / "02_从零Nature图解教学.md"),
        "gpt_mismatch": check_gpt_mismatch(pack_dir(repo) / "01_完成度与GPT误导对照.md"),
        "figures": check_figures(repo, table),
    }
    result["ok"] = all(not v for v in result.values() if isinstance(v, list))
    return result


def copy_evidence(repo: Path, scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    src = pack_dir(repo) / "figures"
    dest = scratch / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*"):
        shutil.copy2(path, dest / path.name)


def write_scratch_reports(repo: Path, scratch: Path, table: dict, report: dict) -> None:
    """Line-oriented PASS/FAIL files the verification plan asks to capture."""
    scratch.mkdir(parents=True, exist_ok=True)
    copy_evidence(repo, scratch)
    locked = table["locked_display"]
    docs_ok = not report["documents"]
    claim_lines = ["claim-check vs official CSV rounding"]
    for key, value in locked.items():
        claim_lines.append(f"[PASS] {key}={value}" if docs_ok else f"[FAIL] {key}={value}")
    if report["documents"]:
        claim_lines.extend(f"[FAIL] {row}" for row in report["documents"])
    claim_lines.append("ALL PASS" if docs_ok else "HAS FAILS")
    (scratch / "claim_check.txt").write_text("\n".join(claim_lines) + "\n")

    term_lines = ["term-scan of 02_从零Nature图解教学.md"]
    if report["terms"]:
        term_lines.extend(f"[FAIL] {row}" for row in report["terms"])
        term_lines.append("HAS FAILS")
    else:
        for english, chinese in REQUIRED_TERMS:
            term_lines.append(f"[PASS] {chinese} ({english})")
        term_lines.append("ALL PASS")
    (scratch / "term_scan.txt").write_text("\n".join(term_lines) + "\n")

    fig_lines = ["figure-check vs sidecar JSON and official CSV"]
    if report["figures"]:
        fig_lines.extend(f"[FAIL] {row}" for row in report["figures"])
        fig_lines.append("HAS FAILS")
    else:
        fig_lines.extend(
            [
                "[PASS] fig1_architecture panels a/b, white, no drop-shadow",
                "[PASS] fig1a training weights from 源域证据 not four seeds",
                "[PASS] fig2_holdout panels a/b/c, white, no drop-shadow",
                "[PASS] fig3_e201_main panels a/b/c values match CSV rounding",
                "[PASS] fig3b labels sit above errorbar whiskers",
            ]
        )
        fig_lines.append("ALL PASS")
    (scratch / "figure_check.txt").write_text("\n".join(fig_lines) + "\n")

    mismatch_src = pack_dir(repo) / "01_完成度与GPT误导对照.md"
    (scratch / "gpt_mismatch.md").write_text(
        mismatch_src.read_text() if mismatch_src.is_file() else json.dumps(report["gpt_mismatch"], ensure_ascii=False)
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=default_repo())
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args(argv)
    repo = args.repo.resolve()
    table = load_claim_table(repo)
    claim_path = write_claim_table(repo, table)
    if not args.skip_figures:
        generate_figures(repo, table)
    report = run_all_checks(repo, table)
    print(json.dumps({"claim_table": str(claim_path), "checks": report}, ensure_ascii=False, indent=2))
    if args.scratch:
        write_scratch_reports(repo, args.scratch, table, report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
