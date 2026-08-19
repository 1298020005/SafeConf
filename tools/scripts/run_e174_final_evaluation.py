#!/usr/bin/env python3
"""Evaluate E174 once on the 640 still-sealed rotated-donor targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr

from e174_conformal_common import (
    MODEL_SPECS,
    add_pair_columns,
    apply_cluster_upper,
    load_npz_vectors,
    predict_ridge,
    rmse_rows,
    target_key,
)
import run_e174_joint_calibration as calibration_helper


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
CALIBRATION_RELEASE = EXPERIMENT / "calibration_release"
CALIBRATION_SNAPSHOT = CALIBRATION_RELEASE / "CALIBRATION_GATE_SNAPSHOT.json"
OUT = EXPERIMENT / "final_evaluation"
STAGING = EXPERIMENT / ".final_evaluation.staging"
PANELS = ("R01", "R02", "R03", "R04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026071903
SCRIPT = Path(__file__).resolve()
ASSET_WRAPPER = ROOT / "tools/scripts/build_e174_rotated_donor_panel_assets.py"
F4_ALLOWLIST = {
    "EVALUATION_TARGET_EFFECTS.npz",
    "EVALUATION_GUIDE_EFFECTS.npz",
    "EVALUATION_GUIDE_EFFECT_INDEX.csv",
    "EVALUATION_TASKS.csv",
    "ROW_ACCESS_AUDIT.csv",
    "ACCESS_ATTESTATION.json",
    "MANIFEST.sha256",
}


class IntegrityFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode().strip()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def strict_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.lower()
    if not values.isin({"true", "false"}).all():
        raise IntegrityFailure("non-boolean frozen task flag")
    return values.eq("true")


def verify_calibration_gate(gate_commit: str, branch: str) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    remote_heads, code_hashes = calibration_helper.verify_code_freeze(gate_commit, branch)
    relative = CALIBRATION_SNAPSHOT.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{gate_commit}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure("calibration snapshot absent from gate commit") from exc
    if committed != CALIBRATION_SNAPSHOT.read_bytes():
        raise IntegrityFailure("calibration snapshot differs from gate commit")
    snapshot = json.loads(committed)
    required = {
        "schema": "safeconf_e174_joint_calibration_snapshot_v1",
        "status": "PASS",
        "n_calibration_targets": 160,
        "n_calibration_tasks": 480,
        "evaluation_targeting_x_values_read": 0,
        "method_frozen_before_evaluation_truth": True,
    }
    mismatches = {
        key: {"expected": value, "observed": snapshot.get(key)}
        for key, value in required.items()
        if snapshot.get(key) != value
    }
    if mismatches:
        raise IntegrityFailure(f"calibration snapshot changed: {mismatches}")
    status = json.loads((CALIBRATION_RELEASE / "RUN_STATUS.json").read_text())
    manifest = CALIBRATION_RELEASE / "MANIFEST.sha256"
    if sha256_file(manifest) != status.get("manifest_sha256"):
        raise IntegrityFailure("calibration release manifest changed")
    for path in (manifest, CALIBRATION_RELEASE / "RUN_STATUS.json"):
        relative_path = path.relative_to(ROOT).as_posix()
        try:
            committed_path = git("show", f"{gate_commit}:{relative_path}").stdout
        except subprocess.CalledProcessError as exc:
            raise IntegrityFailure(f"calibration control file absent from gate commit: {relative_path}") from exc
        if committed_path != path.read_bytes():
            raise IntegrityFailure(f"calibration control file differs from gate commit: {relative_path}")
    for line in manifest.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if sha256_file(CALIBRATION_RELEASE / rel) != digest:
            raise IntegrityFailure(f"calibration payload hash changed: {rel}")
        path = CALIBRATION_RELEASE / rel
        try:
            frozen = git("show", f"{gate_commit}:{path.relative_to(ROOT).as_posix()}").stdout
        except subprocess.CalledProcessError as exc:
            raise IntegrityFailure(f"calibration payload absent from gate commit: {rel}") from exc
        if frozen != path.read_bytes():
            raise IntegrityFailure(f"calibration payload differs from gate commit: {rel}")
    code_hashes.extend(
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (CALIBRATION_SNAPSHOT, manifest, CALIBRATION_RELEASE / "RUN_STATUS.json")
    )
    return snapshot, remote_heads, code_hashes


def load_evaluation_panel(
    panel: str, gate_commit: str, snapshot: dict[str, Any], helper: Any
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    pretruth_snapshot, release = helper.verify_pretruth_release(panel, gate_commit)
    isolated = DATA_ROOT / "isolated/E174" / panel
    f2 = isolated / "F2_pretruth"
    f3 = isolated / "F3A_calibration"
    f4 = isolated / "F4_evaluation"
    _f2, f2_manifest = helper.parse_flat_manifest(f2, calibration_helper.F2_ALLOWLIST)
    _f3, f3_manifest = helper.parse_flat_manifest(f3, calibration_helper.F3_ALLOWLIST)
    _f4, f4_manifest = helper.parse_flat_manifest(f4, F4_ALLOWLIST)
    if snapshot["f2_manifest_sha256"].get(panel) != f2_manifest:
        raise IntegrityFailure(f"{panel} F2 differs from calibration gate")
    if snapshot["f3_calibration_manifest_sha256"].get(panel) != f3_manifest:
        raise IntegrityFailure(f"{panel} F3A differs from calibration gate")
    f4_att = json.loads((f4 / "ACCESS_ATTESTATION.json").read_text())
    required = {
        "experiment": f"E174_rotated_donor_conformal_certificate::{panel}",
        "stage": f"E174_{panel}_F4_EVALUATION_TRUTH_BUILD",
        "status": "PASS",
        "gate_commit": gate_commit,
        "f2_manifest_sha256": f2_manifest,
        "f3_calibration_manifest_sha256": f3_manifest,
        "source_full_sha256": snapshot["source_full_sha256"],
        "calibration_targeting_x_values_read": 0,
        "evaluation_targeting_x_values_read": 960,
        "other_truth_partition_x_values_read": 0,
        "n_evaluation_target_effects": 480,
        "n_evaluation_guide_effects": 960,
        "test_performance_metrics_computed": 0,
    }
    mismatches = {
        key: {"expected": value, "observed": f4_att.get(key)}
        for key, value in required.items()
        if f4_att.get(key) != value
    }
    if mismatches:
        raise IntegrityFailure(f"{panel} F4 attestation failed: {mismatches}")
    if f4_att.get("builder_sha256") != sha256_file(ASSET_WRAPPER):
        raise IntegrityFailure(f"{panel} F4 builder changed")
    if pretruth_snapshot.get("source_full_sha256") != snapshot["source_full_sha256"]:
        raise IntegrityFailure(f"{panel} pretruth source differs from calibration gate")

    scoring = pd.read_csv(release / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False)
    locked = pd.read_csv(
        EXPERIMENT / f"manifests/{panel}/E174_{panel}_TASK_MANIFEST.csv",
        keep_default_na=False,
    )
    evaluation_tasks = locked.loc[strict_flag(locked.evaluation_test_task)].copy()
    evaluation_ids = evaluation_tasks.task_id.astype(str).tolist()
    if len(evaluation_ids) != 480:
        raise IntegrityFailure(f"{panel} evaluation task count changed")
    score_index = scoring.set_index("task_id")
    metadata = score_index.loc[evaluation_ids].reset_index()
    if metadata.heldout_donor_partition.astype(str).ne("EVALUATION_80PCT").any():
        raise IntegrityFailure(f"{panel} calibration target entered final evaluation")
    with np.load(release / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        row = {task: index for index, task in enumerate(scoring.task_id.astype(str))}
        prediction = {
            name: np.stack([np.asarray(archive[name][row[task]], dtype=float) for task in evaluation_ids])
            for name in ("scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean")
        }
    truth = load_npz_vectors(f4 / "EVALUATION_TARGET_EFFECTS.npz", 480)
    if set(truth) != set(evaluation_ids):
        raise IntegrityFailure(f"{panel} evaluation truth/query keys differ")
    truth_matrix = np.stack([truth[task] for task in evaluation_ids])
    sc, ge = prediction["scGPT_seed_mean"], prediction["GEARS_seed_mean"]
    ensemble = prediction["ensemble_seed_family_mean"]
    disagreement = rmse_rows(sc, ge)
    if not np.allclose(
        disagreement, metadata.model_disagreement_rmse.to_numpy(float), atol=2e-7, rtol=2e-6
    ):
        raise IntegrityFailure(f"{panel} disagreement differs from sealed pretruth score")
    sc_error, ge_error = rmse_rows(sc, truth_matrix), rmse_rows(ge, truth_matrix)
    metrics = metadata.copy()
    metrics["panel_id"] = panel
    metrics["ensemble_rmse"] = rmse_rows(ensemble, truth_matrix)
    metrics["scgpt_rmse"] = sc_error
    metrics["gears_rmse"] = ge_error
    metrics["pair_mean_rmse"] = (sc_error + ge_error) / 2.0
    metrics["pair_max_rmse"] = np.maximum(sc_error, ge_error)
    metrics["pair_lower_bound_rmse"] = disagreement / 2.0
    metrics["pair_mean_mse"] = (sc_error**2 + ge_error**2) / 2.0
    metrics["decomposition_rhs_mse"] = metrics.ensemble_rmse**2 + disagreement**2 / 4.0
    metrics["decomposition_abs_residual"] = np.abs(
        metrics.pair_mean_mse - metrics.decomposition_rhs_mse
    )
    metrics["nochange_rmse"] = np.sqrt(np.mean(truth_matrix**2, axis=1))
    hashes = [
        {"panel_id": panel, "path": str(path), "sha256": sha256_file(path)}
        for path in (
            f2 / "MANIFEST.sha256",
            f3 / "MANIFEST.sha256",
            f4 / "MANIFEST.sha256",
            release / "PRETRUTH_GATE_SNAPSHOT.json",
        )
    ]
    return metrics, {"f4_manifest_sha256": f4_manifest}, hashes


def clopper_pearson(success: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if success == 0 else float(beta.ppf(alpha / 2, success, total - success + 1))
    upper = 1.0 if success == total else float(beta.ppf(1 - alpha / 2, success + 1, total - success))
    return lower, upper


def coverage_and_efficiency(metrics: pd.DataFrame, snapshot: dict[str, Any]) -> pd.DataFrame:
    rows = []
    populations = {
        "all_640": np.ones(len(metrics), dtype=bool),
        "seen_512": metrics.target_stratum.eq("DONOR_UNSEEN_ONLY").to_numpy(),
        "column_unseen_128": metrics.target_stratum.eq("COLUMN_UNSEEN").to_numpy(),
    }
    for outcome in ("ensemble_rmse", "pair_mean_rmse"):
        for spec in MODEL_SPECS:
            upper_column = f"upper_{outcome}__{spec}"
            for population, mask in populations.items():
                block = metrics.loc[mask].copy()
                block["covered"] = block[outcome] <= block[upper_column] + 1e-12
                block["target_key"] = target_key(block)
                target_covered = block.groupby("target_key").covered.all()
                success, total = int(target_covered.sum()), len(target_covered)
                low, high = clopper_pearson(success, total)
                lower = (
                    block.pair_lower_bound_rmse.to_numpy(float)
                    if outcome == "pair_mean_rmse"
                    else np.zeros(len(block), dtype=float)
                )
                rows.append(
                    {
                        "outcome": outcome,
                        "model_spec": spec,
                        "population": population,
                        "n_targets": total,
                        "n_tasks": len(block),
                        "targets_all_states_covered": success,
                        "target_simultaneous_coverage": success / total,
                        "exact_binomial_ci95_lower": low,
                        "exact_binomial_ci95_upper": high,
                        "task_marginal_coverage": float(block.covered.mean()),
                        "mean_upper": float(block[upper_column].mean()),
                        "median_upper": float(block[upper_column].median()),
                        "mean_interval_width": float(np.mean(block[upper_column].to_numpy(float) - lower)),
                        "selected_primary_model": snapshot["selected_model_spec"][outcome] == spec,
                    }
                )
    return pd.DataFrame(rows)


def certificate_audit(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    populations = {"ALL_640": metrics}
    populations.update({panel: metrics.loc[metrics.panel_id.eq(panel)] for panel in PANELS})
    for name, block in populations.items():
        rows.append(
            {
                "population": name,
                "n_targets": target_key(block).nunique(),
                "n_tasks": len(block),
                "pair_mean_bound_violations": int(
                    (block.pair_lower_bound_rmse > block.pair_mean_rmse + 1e-10).sum()
                ),
                "pair_max_bound_violations": int(
                    (block.pair_lower_bound_rmse > block.pair_max_rmse + 1e-10).sum()
                ),
                "decomposition_max_abs_residual": float(block.decomposition_abs_residual.max()),
                "median_bound_tightness_pair_mean": float(
                    np.median(block.pair_lower_bound_rmse / block.pair_mean_rmse)
                ),
                "mean_bound_tightness_pair_mean": float(
                    np.mean(block.pair_lower_bound_rmse / block.pair_mean_rmse)
                ),
            }
        )
    return pd.DataFrame(rows)


def ranking_audit(metrics: pd.DataFrame, postgate: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, aurc_rows = [], []
    scores = {
        "fixed_safeconf": "safeconf_risk",
        "magnitude": "predicted_magnitude",
        "disagreement_pair_lower": "pair_lower_bound_rmse",
    }
    outcomes = ("ensemble_rmse", "pair_mean_rmse", "pair_max_rmse")
    for panel in PANELS:
        for state in STATES:
            block = metrics.loc[
                metrics.panel_id.eq(panel) & metrics.culture_condition.eq(state)
            ].copy()
            if len(block) != 160:
                raise IntegrityFailure(f"{panel}/{state} evaluation count changed")
            for outcome in outcomes:
                for score_name, score_column in scores.items():
                    rho = spearmanr(
                        block[score_column].to_numpy(float), block[outcome].to_numpy(float)
                    ).statistic
                    rows.append(
                        {
                            "panel_id": panel,
                            "culture_condition": state,
                            "outcome": outcome,
                            "score_name": score_name,
                            "n_tasks": len(block),
                            "spearman": float(rho),
                        }
                    )
                    _, summary = postgate.tie_aware_curve(
                        block[score_column].to_numpy(float), block[outcome].to_numpy(float)
                    )
                    aurc_rows.append(
                        {
                            "panel_id": panel,
                            "culture_condition": state,
                            "outcome": outcome,
                            "score_name": score_name,
                            "n_tasks": len(block),
                            **summary,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(aurc_rows)


def cluster_bootstrap(metrics: pd.DataFrame, ranking: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    target = metrics.assign(target_key=target_key(metrics))
    groups = {
        key: np.asarray(index, dtype=int)
        for key, index in target.groupby(["panel_id", "target_stratum"]).groups.items()
    }
    result = np.empty((BOOTSTRAP_DRAWS, 4), dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = []
        for indices in groups.values():
            keys = target.loc[indices].groupby("target_key").groups
            names = np.asarray(list(keys), dtype=object)
            sample = rng.choice(names, size=len(names), replace=True)
            chosen.extend(np.concatenate([np.asarray(keys[name], dtype=int) for name in sample]))
        block = target.loc[chosen]
        result[draw, 0] = float(
            np.mean(
                block["upper_pair_mean_rmse__magnitude"].to_numpy(float)
                - block["upper_pair_mean_rmse__state_stratum_constant"].to_numpy(float)
            )
        )
        result[draw, 1] = float(
            np.mean(
                block["upper_pair_mean_rmse__magnitude_plus_pair_lower"].to_numpy(float)
                - block["upper_pair_mean_rmse__magnitude"].to_numpy(float)
            )
        )
        macro = []
        macro_mag = []
        for panel in PANELS:
            for state in STATES:
                unit = block.loc[block.panel_id.eq(panel) & block.culture_condition.eq(state)]
                macro.append(
                    spearmanr(unit.pair_lower_bound_rmse, unit.pair_mean_rmse).statistic
                )
                macro_mag.append(
                    spearmanr(unit.predicted_magnitude, unit.pair_mean_rmse).statistic
                )
        result[draw, 2] = float(np.mean(macro))
        result[draw, 3] = float(np.mean(macro_mag))
    names = (
        "pair_upper_magnitude_minus_constant",
        "pair_upper_composite_minus_magnitude",
        "pair_lower_vs_pair_mean_macro_spearman",
        "magnitude_vs_pair_mean_macro_spearman",
    )
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "estimand": name,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_mean": float(result[:, index].mean()),
                "bootstrap_ci95_lower": float(np.quantile(result[:, index], 0.025)),
                "bootstrap_ci95_upper": float(np.quantile(result[:, index], 0.975)),
            }
        )
    return pd.DataFrame(rows), result


def save_figures(metrics: pd.DataFrame, coverage: pd.DataFrame, ranking: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = STAGING / "figures"
    figures.mkdir(exist_ok=True)

    selected = coverage.loc[
        coverage.population.eq("all_640") & coverage.selected_primary_model
    ].copy()
    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    x = np.arange(len(selected))
    ax.bar(x, selected.target_simultaneous_coverage, color=["#176B87", "#3CAEA3"])
    ax.axhline(0.90, color="#C94C4C", linestyle="--", linewidth=1.1, label="nominal 0.90")
    ax.set_xticks(x, ["Ensemble RMSE", "Pair-mean RMSE"])
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Three-state target coverage")
    ax.set_title("Rotated-donor conformal coverage on 640 hidden targets", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "F1_conformal_coverage.png", dpi=240, facecolor="white")
    fig.savefig(figures / "F1_conformal_coverage.svg", facecolor="white")
    plt.close(fig)

    sample = metrics.sample(min(900, len(metrics)), random_state=174)
    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    ax.scatter(
        sample.pair_mean_rmse,
        sample.pair_lower_bound_rmse,
        s=9,
        alpha=0.35,
        color="#176B87",
        edgecolors="none",
    )
    limit = max(float(sample.pair_mean_rmse.max()), float(sample.pair_lower_bound_rmse.max()))
    ax.plot([0, limit], [0, limit], color="#C94C4C", linewidth=1, linestyle="--")
    ax.set_xlabel("Observed pair-mean RMSE")
    ax.set_ylabel("Certified lower bound d(p1,p2)/2")
    ax.set_title("Exact lower certificate; low disagreement is not safety", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "F2_pair_lower_certificate.png", dpi=240, facecolor="white")
    fig.savefig(figures / "F2_pair_lower_certificate.svg", facecolor="white")
    plt.close(fig)

    macro = ranking.loc[ranking.outcome.eq("pair_mean_rmse")].groupby("score_name").spearman.mean()
    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    order = ["fixed_safeconf", "disagreement_pair_lower", "magnitude"]
    ax.bar(np.arange(3), [macro[name] for name in order], color=["#6B7280", "#176B87", "#C57A2B"])
    ax.axhline(0, color="#D1D5DB", linewidth=1)
    ax.set_xticks(np.arange(3), ["Fixed SafeConf", "Pair lower", "Magnitude"])
    ax.set_ylabel("Equal panel×state Spearman")
    ax.set_title("Risk ranking is reported without post-truth switching", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "F3_ranking_audit.png", dpi=240, facecolor="white")
    fig.savefig(figures / "F3_ranking_audit.svg", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 2.5))
    ax.axis("off")
    boxes = [
        (0.02, "Freeze 800 targets\nand all predictions", "#E8F1F5"),
        (0.27, "Open 160 targets\nfor calibration", "#FFF3E3"),
        (0.52, "Commit conformal\nquantiles", "#E7F3EC"),
        (0.77, "Open 640 hidden\nevaluation targets", "#E8F1F5"),
    ]
    for x, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, 0.34), 0.19, 0.38, facecolor=color, edgecolor="#667085", linewidth=1.1))
        ax.text(x + 0.095, 0.53, label, ha="center", va="center")
    for x in (0.21, 0.46, 0.71):
        ax.annotate("", xy=(x + 0.055, 0.53), xytext=(x, 0.53), arrowprops={"arrowstyle": "->", "color": "#667085"})
    ax.set_title("E174 physical truth-access sequence", loc="left", fontweight="bold")
    fig.savefig(figures / "F4_access_sequence.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(figures / "F4_access_sequence.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_manifest() -> str:
    files = sorted(
        path for path in STAGING.rglob("*") if path.is_file() and path.name not in {"MANIFEST.sha256", "RUN_STATUS.json"}
    )
    text = "".join(
        f"{sha256_file(path)}  {path.relative_to(STAGING).as_posix()}\n" for path in files
    )
    atomic_bytes(STAGING / "MANIFEST.sha256", text.encode())
    return sha256_file(STAGING / "MANIFEST.sha256")


def run(gate_commit: str, branch: str) -> dict[str, Any]:
    started = time.time()
    if OUT.exists() or STAGING.exists():
        raise IntegrityFailure("E174 final output is append-only")
    snapshot, remote_heads, code_hashes = verify_calibration_gate(gate_commit, branch)
    helper = calibration_helper.import_joint_helper()
    postgate = helper.import_postgate_helper()
    metrics_list, input_hashes, panel_meta = [], [], {}
    for panel in PANELS:
        metrics, metadata, hashes = load_evaluation_panel(panel, gate_commit, snapshot, helper)
        metrics_list.append(metrics)
        panel_meta[panel] = metadata
        input_hashes.extend(hashes)
    metrics = add_pair_columns(pd.concat(metrics_list, ignore_index=True))
    if len(metrics) != 1920 or target_key(metrics).nunique() != 640:
        raise IntegrityFailure("final population must be 640 targets / 1,920 tasks")
    frozen_targets = pd.read_csv(EXPERIMENT / "manifests/E174_ALL_SELECTED_TARGETS.csv")
    expected_eval = set(
        frozen_targets.loc[
            frozen_targets.heldout_donor_partition.eq("EVALUATION_80PCT"), "ensembl_core"
        ].astype(str)
    )
    if set(metrics.perturbed_gene_id.astype(str)) != expected_eval:
        raise IntegrityFailure("final targets differ from frozen evaluation partition")

    for outcome in ("ensemble_rmse", "pair_mean_rmse"):
        for spec in MODEL_SPECS:
            model = snapshot["models"][outcome][spec]
            rule = snapshot["calibration_rules"][outcome][spec]
            base = predict_ridge(metrics, model)
            metrics[f"base_{outcome}__{spec}"] = base
            metrics[f"upper_{outcome}__{spec}"] = apply_cluster_upper(metrics, base, rule)
    certificate = certificate_audit(metrics)
    overall_certificate = certificate.loc[certificate.population.eq("ALL_640")].iloc[0]
    if (
        overall_certificate.pair_mean_bound_violations != 0
        or overall_certificate.pair_max_bound_violations != 0
        or overall_certificate.decomposition_max_abs_residual > 1e-8
    ):
        raise IntegrityFailure("final certificate numeric audit failed")

    coverage = coverage_and_efficiency(metrics, snapshot)
    ranking, aurc = ranking_audit(metrics, postgate)
    bootstrap, draws = cluster_bootstrap(metrics, ranking)
    selected_coverage = coverage.loc[
        coverage.population.eq("all_640") & coverage.selected_primary_model
    ].set_index("outcome")
    ensemble_cov = float(selected_coverage.loc["ensemble_rmse", "target_simultaneous_coverage"])
    pair_cov = float(selected_coverage.loc["pair_mean_rmse", "target_simultaneous_coverage"])
    empirical_coverage_at_nominal = ensemble_cov >= 0.90 and pair_cov >= 0.90
    decision = (
        "CERTIFICATE_AND_CONFORMAL_EMPIRICAL_AUDIT_PASS"
        if empirical_coverage_at_nominal
        else "CERTIFICATE_PASS_WITH_EMPIRICAL_COVERAGE_SHORTFALL"
    )

    for sub in ("tables", "arrays", "figures", "reports"):
        (STAGING / sub).mkdir(parents=True, exist_ok=False)
    atomic_csv(STAGING / "tables/EVALUATION_TASK_METRICS.csv", metrics)
    atomic_csv(STAGING / "tables/CERTIFICATE_AUDIT.csv", certificate)
    atomic_csv(STAGING / "tables/CONFORMAL_COVERAGE_EFFICIENCY.csv", coverage)
    atomic_csv(STAGING / "tables/RANKING_SPEARMAN.csv", ranking)
    atomic_csv(STAGING / "tables/LEGACY_AURC_DIAGNOSTIC.csv", aurc)
    atomic_csv(STAGING / "tables/CLUSTER_BOOTSTRAP_SUMMARY.csv", bootstrap)
    atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(code_hashes + input_hashes))
    with (STAGING / "arrays/CLUSTER_BOOTSTRAP_DRAWS.npz").open("wb") as handle:
        np.savez_compressed(handle, draws=draws)
    save_figures(metrics, coverage, ranking)
    report = f"""# E174 final evaluation

正式状态：**{decision}**。E174 使用轮换测试供体 CE0008678 和 800 个全新靶点；160 个靶点用于一次 conformal 校准，主要评价只含随后开放的 640 个隐藏靶点、1,920 个任务。

模型对下界在 pair mean 与 pair max 上的违例均为 0；平方误差分解最大残差为 {overall_certificate.decomposition_max_abs_residual:.3g}。这证明高分歧能够证实模型对至少存在相应风险，但小分歧仍不能证明安全。

冻结选择的 magnitude 基础模型在三个状态同时覆盖的 target-level 经验覆盖率为：ensemble RMSE {ensemble_cov:.3f}，pair-mean RMSE {pair_cov:.3f}，目标值为 0.90。完整 exact binomial 区间、constant/composite 效率对照和 10,000 次 target-cluster bootstrap 均在 tables 中；任何覆盖不足都原样保留，不重新调分位数。

固定 SafeConf、magnitude 与 disagreement 的排序/AURC 只作为次要诊断。E168/E172 已否定的“固定 SafeConf 稳定优于 magnitude”不会因 E174 改写。E174 仍属于同一 Primary CD4 研究，不代表独立研究、湿实验、临床部署或期刊录用保证。
"""
    atomic_bytes(STAGING / "reports/E174_FINAL_REPORT.md", report.encode())
    manifest_sha = write_manifest()
    status = {
        "schema": "safeconf_e174_final_evaluation_status_v1",
        "experiment": "E174_rotated_donor_conformal_certificate",
        "stage": "F4_FINAL_HIDDEN_EVALUATION",
        "status": "COMPLETE",
        "decision": decision,
        "calibration_gate_commit": gate_commit,
        "calibration_gate_remote_heads": remote_heads,
        "n_calibration_targets_excluded_from_primary": 160,
        "n_hidden_evaluation_targets": 640,
        "n_hidden_evaluation_tasks": 1920,
        "pair_mean_bound_violations": 0,
        "pair_max_bound_violations": 0,
        "squared_error_decomposition_max_abs_residual": float(
            overall_certificate.decomposition_max_abs_residual
        ),
        "ensemble_rmse_target_simultaneous_coverage": ensemble_cov,
        "pair_mean_rmse_target_simultaneous_coverage": pair_cov,
        "empirical_coverage_at_or_above_nominal_for_both": empirical_coverage_at_nominal,
        "fixed_safeconf_increment_claim_supported": False,
        "evaluation_truth_used_to_select_model_or_recalibrate_quantile": False,
        "same_study_rotated_donor_not_independent_study": True,
        "deployment_authorized": False,
        "manifest_sha256": manifest_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "wall_seconds": time.time() - started,
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    os.replace(STAGING, OUT)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-commit", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    result = run(args.gate_commit, args.branch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
