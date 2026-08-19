#!/usr/bin/env python3
"""Develop an ensemble-aligned seed-stability gate from E170 pretruth data.

This program may read the E170 F2 seen validation effects, but it refuses to
run if any E170 F3 directory exists.  It never opens the source H5AD and never
uses E170 test truth.  All results are development evidence for a later fresh
target experiment, not a post-hoc override of E170's registered abort.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E170 = ROOT / "docs/实验结果/E170_primary_cd4_multipanel_precision_20260718"
OUT = ROOT / "docs/实验结果/E171_seed_ensemble_gate_development_20260718"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
ISOLATED = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/E170")
PRETRUTH_HELPER = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
POSTGATE_HELPER = ROOT / "tools/scripts/run_e168_primary_cd4_postgate.py"
PANELS = ("P01", "P02", "P03", "P04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
SEEDS = (3407, 3408, 3409)
WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.0)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed_and_pushed(paths: list[Path]) -> tuple[str, str, dict[str, str]]:
    head = git_text("rev-parse", "HEAD")
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise RuntimeError("E171 development requires a named branch")
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        if committed != path.read_bytes():
            raise RuntimeError(f"development input/code differs from HEAD: {relative}")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        subprocess.run(
            ["git", "fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}"],
            cwd=ROOT, check=True,
        )
        remote_head = git_text("rev-parse", fetched)
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head], cwd=ROOT
        ).returncode:
            raise RuntimeError(f"development HEAD absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return head, branch, remote_heads


def zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center = float(np.median(reference))
    scale = float(np.percentile(reference, 75) - np.percentile(reference, 25))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(reference, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return (np.asarray(values, float) - center) / scale


def build_loo_risks(score: pd.DataFrame, predictions: dict[str, np.ndarray]) -> np.ndarray:
    train = score.donor_role.eq("train").to_numpy()
    fixed = -score.z_context_train960.to_numpy(float) - score.z_log_support_train960.to_numpy(float)
    risks = []
    for omitted in SEEDS:
        retained = [seed for seed in SEEDS if seed != omitted]
        scgpt = np.mean([predictions[f"scGPT_seed{seed}"] for seed in retained], axis=0)
        gears = np.mean([predictions[f"GEARS_seed{seed}"] for seed in retained], axis=0)
        disagreement = np.sqrt(np.mean((scgpt - gears) ** 2, axis=1))
        risks.append(fixed + zscore(disagreement, disagreement[train]))
    return np.stack(risks)


def panel_development(
    panel: str, pretruth: Any, postgate: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    release = E170 / "pretruth_release" / panel
    snapshot = json.loads((release / "PRETRUTH_GATE_SNAPSHOT.json").read_text())
    if snapshot.get("test_targeting_x_values_read") != 0:
        raise RuntimeError(f"{panel} snapshot reports test truth access")
    score = pd.read_csv(release / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False)
    with np.load(release / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        predictions = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    loo = build_loo_risks(score, predictions)
    original = pd.read_csv(release / "tables/G4_SEED_STABILITY.csv", keep_default_na=False)
    rows = []
    for state in STATES:
        for stratum, mask in {
            "all_200": score.donor_role.eq("test") & score.culture_condition.eq(state),
            "seen_160": score.donor_role.eq("test") & score.culture_condition.eq(state)
            & score.target_stratum.eq("DONOR_UNSEEN_ONLY"),
        }.items():
            old = original.loc[
                original.culture_condition.eq(state) & original.stratum.eq(stratum)
            ].iloc[0]
            new = pretruth.g4_stability(
                loo[:, mask.to_numpy()],
                pretruth.stable_seed("E171_LOO_G4", panel, state, stratum),
            )
            rows.append(
                {
                    "panel_id": panel,
                    "culture_condition": state,
                    "stratum": stratum,
                    "n_tasks": int(mask.sum()),
                    "original_single_seed_pair_median_spearman": float(old.median_pairwise_spearman),
                    "original_single_seed_pair_ci95_lower": float(old.bootstrap_ci95_lower),
                    "original_single_seed_pair_passed": str(old.passed).strip().lower() == "true",
                    "loo_two_seed_family_mean_median_spearman": float(new["median_pairwise_spearman"]),
                    "loo_two_seed_family_mean_ci95_lower": float(new["bootstrap_ci95_lower"]),
                    "loo_two_seed_family_mean_passed": bool(new["passed"]),
                    "gate_threshold_median_spearman": 0.5,
                    "gate_threshold_ci95_lower_exclusive": 0.0,
                }
            )

    scgpt_mean = np.mean([predictions[f"scGPT_seed{seed}"] for seed in SEEDS], axis=0)
    gears_mean = np.mean([predictions[f"GEARS_seed{seed}"] for seed in SEEDS], axis=0)
    ensemble = (scgpt_mean + gears_mean) / 2.0
    train = score.donor_role.eq("train").to_numpy()
    z_magnitude = zscore(score.predicted_magnitude.to_numpy(float), score.predicted_magnitude.to_numpy(float)[train])
    z_disagreement = zscore(score.model_disagreement_rmse.to_numpy(float), score.model_disagreement_rmse.to_numpy(float)[train])
    effects_path = ISOLATED / panel / "F2_pretruth/SEEN_TARGET_EFFECTS.npz"
    with np.load(effects_path, allow_pickle=False) as archive:
        effects = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    validation = score.index[
        score.donor_role.eq("validation") & score.target_stratum.eq("DONOR_UNSEEN_ONLY")
    ]
    validation_rows = []
    candidate_rows = []
    for state in STATES:
        indices = validation[score.loc[validation, "culture_condition"].eq(state)]
        loss = np.asarray(
            [
                np.sqrt(np.mean((ensemble[index] - effects[str(score.task_id.iloc[index])]) ** 2))
                for index in indices
            ],
            dtype=float,
        )
        safeconf = score.safeconf_risk.iloc[indices].to_numpy(float)
        magnitude = score.predicted_magnitude.iloc[indices].to_numpy(float)
        safe_aurc = postgate.tie_aware_aurc_value(safeconf, loss)
        magnitude_aurc = postgate.tie_aware_aurc_value(magnitude, loss)
        validation_rows.append(
            {
                "panel_id": panel,
                "culture_condition": state,
                "n_validation_seen_tasks": len(indices),
                "safeconf_aurc": safe_aurc,
                "magnitude_aurc": magnitude_aurc,
                "delta_magnitude_minus_safeconf": magnitude_aurc - safe_aurc,
                "safeconf_error_spearman": postgate.spearman(safeconf, loss),
                "magnitude_error_spearman": postgate.spearman(magnitude, loss),
            }
        )
        candidate_scores = {"magnitude_only": z_magnitude[indices], "safeconf_only": z_disagreement[indices]}
        for weight in WEIGHTS:
            candidate_scores[f"zmag_plus_{weight:g}_zdis"] = z_magnitude[indices] + weight * z_disagreement[indices]
        for name, candidate in candidate_scores.items():
            candidate_aurc = postgate.tie_aware_aurc_value(candidate, loss)
            candidate_rows.append(
                {
                    "panel_id": panel,
                    "culture_condition": state,
                    "candidate": name,
                    "candidate_aurc": candidate_aurc,
                    "magnitude_aurc": magnitude_aurc,
                    "delta_magnitude_minus_candidate": magnitude_aurc - candidate_aurc,
                    "candidate_error_spearman": postgate.spearman(candidate, loss),
                    "development_only": True,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(validation_rows), pd.DataFrame(candidate_rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: Any, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    svg_path = FIGURES / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    # Matplotlib writes trailing spaces in multi-line SVG path data.  Strip
    # them mechanically so Git's whitespace audit remains clean.
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def make_figures(stability: pd.DataFrame, validation: pd.DataFrame) -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    order = stability.reset_index(drop=True)
    x = np.arange(len(order))
    for index, row in order.iterrows():
        ax.plot(
            [index, index],
            [row.original_single_seed_pair_median_spearman, row.loo_two_seed_family_mean_median_spearman],
            color="#B8C2CC", linewidth=1.0, zorder=1,
        )
    ax.scatter(x - 0.08, order.original_single_seed_pair_median_spearman, s=24, color="#7A8B99", label="single-seed pair", zorder=2)
    ax.scatter(x + 0.08, order.loo_two_seed_family_mean_median_spearman, s=28, color="#176B87", label="leave-one-out family mean", zorder=3)
    ax.axhline(0.5, color="#C94C4C", linestyle="--", linewidth=1.2, label="registered threshold 0.5")
    labels = [f"{row.panel_id}\n{row.culture_condition.replace('Stim','S')}\n{'all' if row.stratum=='all_200' else 'seen'}" for row in order.itertuples()]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("median pairwise Spearman")
    ax.set_title("Seed-stability estimator alignment (test truth unread)", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center")
    ax.set_ylim(-0.05, 1.02); fig.tight_layout(); save_figure(fig, "Fig1_seed_stability_original_vs_loo")

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    table = validation.copy(); table["unit"] = table.panel_id + " / " + table.culture_condition
    colors = np.where(table.delta_magnitude_minus_safeconf > 0, "#176B87", "#B45A5A")
    y = np.arange(len(table))
    ax.barh(y, table.delta_magnitude_minus_safeconf, color=colors, height=0.68)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(table.unit, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("validation Δ AURC (magnitude − SafeConf); positive favors SafeConf")
    ax.set_title("Validation performance remains inconclusive", loc="left", fontweight="bold")
    fig.tight_layout(); save_figure(fig, "Fig2_validation_delta_safeconf_vs_magnitude")


def write_manifest() -> str:
    hashes = {
        path.relative_to(OUT).as_posix(): sha256_file(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    payload = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    (OUT / "MANIFEST.sha256").write_text(payload, encoding="utf-8")
    return sha256_file(OUT / "MANIFEST.sha256")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"append-only E171 development output exists: {OUT}")
    for directory in (OUT, TABLES, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    f3 = sorted(ISOLATED.glob("*/F3_postgate"))
    if f3:
        raise RuntimeError(f"E170 F3 truth exists; pretruth-only development refused: {f3}")
    pretruth_status = json.loads((E170 / "PRETRUTH_RUN_STATUS.json").read_text())
    if pretruth_status.get("decision") != "PRETRUTH_ABORTED" or pretruth_status.get("test_targeting_x_values_read") != 0:
        raise RuntimeError("E170 pretruth abort contract changed")
    code_inputs = [
        Path(__file__).resolve(), PRETRUTH_HELPER, POSTGATE_HELPER,
        E170 / "PRETRUTH_RUN_STATUS.json", E170 / "reports/E170_PRETRUTH_MULTIPANEL_REPORT.md",
    ]
    head, branch, remote_heads = require_committed_and_pushed(code_inputs)
    pretruth = import_module("e171_pretruth_helper", PRETRUTH_HELPER)
    postgate = import_module("e171_postgate_helper", POSTGATE_HELPER)
    stability_tables, validation_tables, candidate_tables = [], [], []
    for panel in PANELS:
        stability, validation, candidates = panel_development(panel, pretruth, postgate)
        stability_tables.append(stability); validation_tables.append(validation); candidate_tables.append(candidates)
    stability = pd.concat(stability_tables, ignore_index=True)
    validation = pd.concat(validation_tables, ignore_index=True)
    candidates = pd.concat(candidate_tables, ignore_index=True)
    stability.to_csv(TABLES / "E171_SEED_STABILITY_COMPARISON.csv", index=False, float_format="%.17g")
    validation.to_csv(TABLES / "E171_VALIDATION_SAFE_CONF_VS_MAGNITUDE.csv", index=False, float_format="%.17g")
    candidates.to_csv(TABLES / "E171_VALIDATION_CANDIDATE_GRID.csv", index=False, float_format="%.17g")
    candidate_summary = (
        candidates.groupby("candidate", as_index=False)
        .agg(
            equal_unit_mean_delta=("delta_magnitude_minus_candidate", "mean"),
            positive_units=("delta_magnitude_minus_candidate", lambda values: int((values > 0).sum())),
            mean_error_spearman=("candidate_error_spearman", "mean"),
        )
        .sort_values(["equal_unit_mean_delta", "candidate"], ascending=[False, True])
    )
    candidate_summary.to_csv(TABLES / "E171_VALIDATION_CANDIDATE_SUMMARY.csv", index=False, float_format="%.17g")
    make_figures(stability, validation)
    loo_pass = int(stability.loo_two_seed_family_mean_passed.sum())
    old_pass = int(stability.original_single_seed_pair_passed.sum())
    mean_delta = float(validation.delta_magnitude_minus_safeconf.mean())
    positive = int((validation.delta_magnitude_minus_safeconf > 0).sum())
    best = candidate_summary.iloc[0]
    status = {
        "schema": "safeconf_e171_pretruth_development_v1",
        "experiment": "E171_seed_ensemble_gate_development",
        "stage": "PRETRUTH_DEVELOPMENT_ONLY",
        "status": "COMPLETE",
        "git_head": head,
        "git_branch": branch,
        "remote_heads": remote_heads,
        "e170_test_targeting_x_values_read": 0,
        "e170_f3_directories_present": 0,
        "original_g4_units_passed": old_pass,
        "registered_g4_units": len(stability),
        "loo_family_mean_g4_units_passed": loo_pass,
        "validation_magnitude_minus_safeconf_mean_delta": mean_delta,
        "validation_safeconf_positive_units": positive,
        "validation_units": len(validation),
        "best_exploratory_candidate": str(best.candidate),
        "best_exploratory_candidate_mean_delta": float(best.equal_unit_mean_delta),
        "performance_rescue_claim_supported": False,
        "recommended_gate_for_fresh_protocol": "three_leave_one_seed_out_two_seed_family_mean_risks",
        "recommended_deployed_score_changed": False,
        "deployment_authorized": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    report = f"""# E171｜seed-ensemble gate 开发审计

E170 test truth 仍未读取，F3 目录数为 0。本审计只使用 pretruth 预测与已允许的 validation donor seen-target effects。

原 G4 比较三个单-seed scGPT–GEARS 配对，24 个 panel×state×stratum 单元通过 {old_pass}/24。改为三组 leave-one-seed-out family means 后通过 {loo_pass}/24；最终部署分数仍使用三个 seeds 的完整 family mean，SafeConf 公式、目标和阈值没有变化。这个修正让 gate 检查的估计器更接近实际部署估计器，可以进入新的未读目标协议，但不能回头解封 E170。

validation donor 的 SafeConf 相对 magnitude 平均 Δ(AURC_magnitude−AURC_SafeConf)={mean_delta:.6g}，仅 {positive}/12 单元为正。固定开发网格中最好的候选是 `{best.candidate}`，平均 Δ={best.equal_unit_mean_delta:.6g}；这些结果没有形成稳定的 performance rescue。因此下一次 fresh-target 实验可修正 seed gate，但不得宣称已经解决 magnitude 基线，也不得以 validation 结果筛掉负面板。
"""
    (REPORTS / "E171_DEVELOPMENT_REPORT.md").write_text(report, encoding="utf-8")
    manifest_sha = write_manifest()
    print(json.dumps({**status, "manifest_sha256": manifest_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
