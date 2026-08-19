#!/usr/bin/env python3
"""Open E180 evaluation guides once and produce the frozen final certificate audit."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E180_xucao_fresh_guide_certificate_20260723"
RELEASE = OUT / "final_evaluation"
STAGING = OUT / f".final_evaluation.staging.{os.getpid()}"
PRETRUTH = OUT / "pretruth_release"
CALIBRATION = OUT / "calibration_release"
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
MODEL_LOCK = OUT / "MODEL_INPUT_LOCK.json"
STAT_LOCK = OUT / "STATISTICAL_ANALYSIS_LOCK.json"
TASKS = OUT / "manifests/E180_GUIDE_TASK_MANIFEST.csv"
BUILDER = ROOT / "tools/scripts/build_e180_xucao_pretruth_assets.py"
PRETRUTH_RUNNER = ROOT / "tools/scripts/run_e180_xucao_pretruth.py"
CALIBRATION_RUNNER = ROOT / "tools/scripts/run_e180_xucao_calibration.py"
F2_ROOT = Path("/home/yyf/data/safeconf_e180_external/isolated/F2_pretruth")
N_GENES = 512
EPS = 1e-9
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 2026072401
METHOD_BASES = {
    "constant": "constant_base",
    "predicted_magnitude": "magnitude_base",
    "magnitude_plus_pair_lower": "magnitude_plus_lower_base",
    "extra_trees_vector": "extra_trees_vector_base",
}
METHOD_LABELS = {
    "constant": "Constant",
    "predicted_magnitude": "Magnitude",
    "magnitude_plus_pair_lower": "Magnitude + lower",
    "extra_trees_vector": "ExtraTrees",
}

BLUE = "#3B6FB6"
TEAL = "#2A9D8F"
ORANGE = "#E6863B"
RED = "#C84C4C"
GREY = "#6B7280"
LIGHT = "#E8EEF6"


class IntegrityError(RuntimeError):
    """E180 final-evaluation integrity failure."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{key: np.asarray(value, np.float32) for key, value in sorted(arrays.items())},
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    local = path.read_bytes()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityError(f"uncommitted E180 final input: {relative}") from exc
    if local != committed:
        raise IntegrityError(f"E180 final input differs from HEAD: {relative}")
    return {"path": relative, "bytes": len(local), "sha256": hashlib.sha256(local).hexdigest()}


def verify_remotes(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                remote,
                f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
            ],
            cwd=ROOT,
            check=False,
        )
        if fetched.returncode:
            raise IntegrityError(f"cannot verify E180 final freeze on {remote}")
        remote_head = git_text("rev-parse", f"refs/remotes/{remote}/{branch}")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise IntegrityError(f"E180 final code absent from {remote}")
        remote_heads[remote] = remote_head
    return branch, remote_heads


def formal_audit() -> tuple[str, str, dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    head = git_text("rev-parse", "HEAD")
    branch, remotes = verify_remotes(head)
    paths = [
        RUNNER,
        BUILDER,
        PRETRUTH_RUNNER,
        CALIBRATION_RUNNER,
        SOURCE_LOCK,
        MODEL_LOCK,
        STAT_LOCK,
        TASKS,
        PRETRUTH / "PRETRUTH_GATE_SNAPSHOT.json",
        PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv",
        PRETRUTH / "arrays/PRETRUTH_PREDICTIONS.npz",
        CALIBRATION / "ACCESS_ATTESTATION.json",
        CALIBRATION / "CALIBRATION_MODEL.json",
    ]
    hashes = [require_committed(path, head) for path in paths]
    pretruth = json.loads((PRETRUTH / "PRETRUTH_GATE_SNAPSHOT.json").read_text())
    calibration_access = json.loads((CALIBRATION / "ACCESS_ATTESTATION.json").read_text())
    calibration_model = json.loads((CALIBRATION / "CALIBRATION_MODEL.json").read_text())
    if (
        pretruth.get("status") != "PASS"
        or pretruth.get("evaluation_target_x_rows_read") != 0
        or calibration_access.get("status") != "PASS"
        or calibration_access.get("evaluation_target_x_rows_read") != 0
        or calibration_model.get("status") != "FROZEN"
        or calibration_model.get("evaluation_truth_opened")
    ):
        raise IntegrityError("E180 pre-evaluation freeze is invalid")
    return head, branch, remotes, hashes, calibration_model


def aggregate_evaluation_truth(
    source: Path,
    tasks: pd.DataFrame,
    panel: pd.DataFrame,
    control: np.ndarray,
    builder: Any,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    evaluation = tasks[tasks["target_split"].eq("prospective_evaluation")].copy()
    allowed = {
        (str(row.perturbation), str(row.guide_id)): str(row.task_id)
        for row in evaluation.itertuples(index=False)
    }
    adata = ad.read_h5ad(source, backed="r")
    try:
        obs = adata.obs.copy()
        obs["perturbation"] = obs["perturbation"].astype(str)
        obs["guide_id"] = obs["guide_id"].astype(str)
        mask = np.asarray(
            [
                (perturbation, guide) in allowed
                for perturbation, guide in zip(
                    obs["perturbation"], obs["guide_id"], strict=True
                )
            ],
            dtype=bool,
        )
        rows = obs.loc[mask, ["perturbation", "guide_id"]].copy()
        rows["source_row_index"] = np.flatnonzero(mask)
        rows = rows.sort_values("source_row_index").reset_index(drop=True)
        panel_columns = panel["source_column_index"].to_numpy(np.int64)
        sums: defaultdict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(N_GENES, np.float64)
        )
        counts: defaultdict[str, int] = defaultdict(int)
        access: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            block = rows.iloc[start : start + batch_size]
            matrix = builder.read_rows(
                adata,
                block["source_row_index"].astype(int).tolist(),
                panel_columns,
            ).toarray()
            for meta, vector in zip(block.itertuples(index=False), matrix, strict=True):
                task_id = allowed[(str(meta.perturbation), str(meta.guide_id))]
                sums[task_id] += vector
                counts[task_id] += 1
                access.append(
                    {
                        "source_row_index": int(meta.source_row_index),
                        "task_id": task_id,
                        "truth_access_phase": "F4_EVALUATION_ONLY",
                    }
                )
    finally:
        adata.file.close()
    truth = {
        task_id: (sums[task_id] / counts[task_id] - control).astype(np.float32)
        for task_id in sorted(sums)
    }
    if set(truth) != set(evaluation["task_id"].astype(str)):
        raise IntegrityError("E180 evaluation truth task set incomplete")
    return truth, pd.DataFrame(access)


def exact_binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def spearman(a: pd.Series, b: pd.Series) -> float:
    av = a.to_numpy(float)
    bv = b.to_numpy(float)
    if np.unique(av).size < 2 or np.unique(bv).size < 2:
        return float("nan")
    return float(
        np.corrcoef(rankdata(av, method="average"), rankdata(bv, method="average"))[
            0, 1
        ]
    )


def build_results(
    scores: pd.DataFrame,
    truth: dict[str, np.ndarray],
    calibration_model: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluation = scores[scores["target_split"].eq("prospective_evaluation")].copy()
    with np.load(PRETRUTH / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        scgpt = np.asarray(archive["scGPT_seed_mean"], np.float64)
        gears = np.asarray(archive["GEARS_seed_mean"], np.float64)
    index = {task: row for row, task in enumerate(scores["task_id"].astype(str))}
    rows: list[dict[str, Any]] = []
    for item in evaluation.itertuples(index=False):
        task_id = str(item.task_id)
        row_index = index[task_id]
        target = np.asarray(truth[task_id], np.float64)
        sc_error = float(np.sqrt(np.mean((scgpt[row_index] - target) ** 2)))
        ge_error = float(np.sqrt(np.mean((gears[row_index] - target) ** 2)))
        disagreement = float(np.sqrt(np.mean((scgpt[row_index] - gears[row_index]) ** 2)))
        result: dict[str, Any] = {
            "task_id": task_id,
            "perturbation": str(item.perturbation),
            "guide_id": str(item.guide_id),
            "n_guide_cells": int(item.n_guide_cells),
            "scgpt_rmse": sc_error,
            "gears_rmse": ge_error,
            "pair_mean_rmse": (sc_error + ge_error) / 2.0,
            "pair_max_rmse": max(sc_error, ge_error),
            "model_disagreement_rmse": disagreement,
            "pair_lower_bound": disagreement / 2.0,
            "predicted_magnitude": float(item.predicted_magnitude),
            "extra_trees_vector_base": float(item.extra_trees_vector_base),
        }
        for method, base_column in METHOD_BASES.items():
            base = float(getattr(item, base_column))
            correction = float(
                calibration_model["methods"][method]["conformal_correction"]
            )
            upper = max(result["pair_lower_bound"], base + correction)
            result[f"base__{method}"] = base
            result[f"upper__{method}"] = upper
            result[f"covered__{method}"] = result["pair_mean_rmse"] <= upper + EPS
        result["pair_lower_mean_violation"] = (
            result["pair_lower_bound"] > result["pair_mean_rmse"] + EPS
        )
        result["pair_lower_max_violation"] = (
            result["pair_lower_bound"] > result["pair_max_rmse"] + EPS
        )
        rows.append(result)
    tasks = pd.DataFrame(rows)

    cluster_rows: list[dict[str, Any]] = []
    for perturbation, block in tasks.groupby("perturbation", sort=True):
        item: dict[str, Any] = {
            "perturbation": perturbation,
            "n_guides": len(block),
            "max_pair_mean_rmse": float(block["pair_mean_rmse"].max()),
            "mean_pair_mean_rmse": float(block["pair_mean_rmse"].mean()),
            "max_pair_lower_bound": float(block["pair_lower_bound"].max()),
        }
        for method in METHOD_BASES:
            item[f"cluster_covered__{method}"] = bool(
                block[f"covered__{method}"].all()
            )
            item[f"mean_upper__{method}"] = float(
                block[f"upper__{method}"].mean()
            )
        cluster_rows.append(item)
    clusters = pd.DataFrame(cluster_rows)

    efficiency_rows: list[dict[str, Any]] = []
    for method in METHOD_BASES:
        covered = int(clusters[f"cluster_covered__{method}"].sum())
        lower, upper = exact_binomial_ci(covered, len(clusters))
        efficiency_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_targets": len(clusters),
                "n_tasks": len(tasks),
                "targets_all_guides_covered": covered,
                "target_simultaneous_coverage": covered / len(clusters),
                "exact_binomial_ci95_lower": lower,
                "exact_binomial_ci95_upper": upper,
                "task_marginal_coverage": float(
                    tasks[f"covered__{method}"].mean()
                ),
                "mean_upper": float(tasks[f"upper__{method}"].mean()),
                "median_upper": float(tasks[f"upper__{method}"].median()),
                "mean_width_above_pair_lower": float(
                    (
                        tasks[f"upper__{method}"]
                        - tasks["pair_lower_bound"]
                    ).mean()
                ),
                "selected_primary_method": method == "extra_trees_vector",
            }
        )
    efficiency = pd.DataFrame(efficiency_rows)
    ranking = pd.DataFrame(
        [
            {
                "score": score,
                "outcome": "pair_mean_rmse",
                "n_tasks": len(tasks),
                "spearman": spearman(tasks[score], tasks["pair_mean_rmse"]),
            }
            for score in (
                "pair_lower_bound",
                "predicted_magnitude",
                "extra_trees_vector_base",
            )
        ]
    )
    return tasks, clusters, efficiency, ranking


def bootstrap_efficiency(tasks: pd.DataFrame) -> pd.DataFrame:
    genes = np.asarray(sorted(tasks["perturbation"].unique()), dtype=object)
    grouped = {gene: tasks[tasks["perturbation"].eq(gene)] for gene in genes}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for draw in range(BOOTSTRAPS):
        take = rng.choice(genes, size=len(genes), replace=True)
        constant_values = np.concatenate(
            [grouped[gene]["upper__constant"].to_numpy(float) for gene in take]
        )
        adaptive_values = np.concatenate(
            [
                grouped[gene]["upper__extra_trees_vector"].to_numpy(float)
                for gene in take
            ]
        )
        rows.append(
            {
                "draw": draw,
                "adaptive_minus_constant_mean_upper": float(
                    adaptive_values.mean() - constant_values.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    for suffix in (".png", ".svg"):
        fig.savefig(
            directory / f"{name}{suffix}",
            dpi=240 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def make_figures(
    tasks: pd.DataFrame,
    efficiency: pd.DataFrame,
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
    directory: Path,
) -> None:
    configure_plots()
    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        (0.02, "65 train genes\n174 guide tasks"),
        (0.27, "32 validation genes\nfreeze adaptive base"),
        (0.52, "29 calibration genes\nfreeze 90% correction"),
        (0.77, "27 evaluation genes\n73 untouched guides"),
    ]
    for x, label in labels:
        ax.add_patch(
            plt.Rectangle(
                (x, 0.32), 0.19, 0.35, facecolor="white", edgecolor=BLUE, linewidth=1.6
            )
        )
        ax.text(x + 0.095, 0.495, label, ha="center", va="center")
    for x in (0.215, 0.465, 0.715):
        ax.annotate(
            "",
            xy=(x + 0.045, 0.495),
            xytext=(x, 0.495),
            arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.5},
        )
    ax.text(
        0.5,
        0.12,
        "All guides from one gene stay together; evaluation truth is opened once.",
        ha="center",
        color=GREY,
    )
    save_figure(fig, directory, "F1_E180_FROZEN_DESIGN")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    block = efficiency.sort_values("mean_upper")
    colors = [
        TEAL if method == "extra_trees_vector" else BLUE if method == "constant" else LIGHT
        for method in block["method"]
    ]
    ax.barh(block["method_label"], block["mean_upper"], color=colors)
    ax.set_xlabel("Mean calibrated upper bound (RMSE)")
    ax.set_title("Evaluation efficiency: smaller is sharper", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    save_figure(fig, directory, "F2_E180_UPPER_EFFICIENCY")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(efficiency))
    ax.errorbar(
        efficiency["target_simultaneous_coverage"],
        y,
        xerr=np.vstack(
            [
                efficiency["target_simultaneous_coverage"]
                - efficiency["exact_binomial_ci95_lower"],
                efficiency["exact_binomial_ci95_upper"]
                - efficiency["target_simultaneous_coverage"],
            ]
        ),
        fmt="o",
        color=BLUE,
        capsize=3,
    )
    ax.axvline(0.90, color=RED, linestyle="--", linewidth=1)
    ax.set_yticks(y, efficiency["method_label"])
    ax.set_xlim(0.55, 1.01)
    ax.set_xlabel("Target-simultaneous coverage (95% exact CI)")
    ax.set_title("All guides must be covered", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    save_figure(fig, directory, "F3_E180_COVERAGE_FOREST")

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(
        tasks["pair_mean_rmse"],
        tasks["pair_lower_bound"],
        s=24,
        color=TEAL,
        alpha=0.75,
    )
    limit = max(tasks["pair_mean_rmse"].max(), tasks["pair_lower_bound"].max()) * 1.05
    ax.plot([0, limit], [0, limit], color=RED, linestyle="--", linewidth=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Observed pair-mean RMSE")
    ax.set_ylabel("Deterministic pair lower bound")
    ax.set_title("Triangle-inequality certificate", fontweight="bold")
    ax.grid(color="#E5E7EB", linewidth=0.7)
    save_figure(fig, directory, "F4_E180_LOWER_CERTIFICATE")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].scatter(tasks["scgpt_rmse"], tasks["gears_rmse"], s=24, color=BLUE, alpha=0.7)
    axes[0].set_xlabel("scGPT RMSE")
    axes[0].set_ylabel("GEARS RMSE")
    axes[0].set_title("Shared task difficulty")
    axes[0].grid(color="#E5E7EB", linewidth=0.7)
    values = bootstrap["adaptive_minus_constant_mean_upper"].to_numpy()
    axes[1].hist(values, bins=35, color=ORANGE, alpha=0.85)
    axes[1].axvline(0, color=RED, linestyle="--", linewidth=1)
    axes[1].set_xlabel("ExtraTrees − constant mean upper")
    axes[1].set_ylabel("Target-cluster bootstrap draws")
    axes[1].set_title("Adaptive-base efficiency")
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, directory, "F5_E180_DIFFICULTY_AND_EFFICIENCY")


def main() -> None:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityError("E180 final release is append-only")
    audit = formal_audit()
    head, branch, remotes, hashes, calibration_model = audit
    source_lock = json.loads(SOURCE_LOCK.read_text())
    source = Path(source_lock["source_path"])
    if sha256_file(source) != source_lock["source_sha256"]:
        raise IntegrityError("E180 source changed before final evaluation")
    builder = import_script("e180_builder_for_final", BUILDER)
    panel = pd.read_csv(F2_ROOT / "GENE_PANEL.csv")
    with np.load(F2_ROOT / "CONTROL_PROFILES.npz", allow_pickle=False) as archive:
        control = np.asarray(archive["GLOBAL"], np.float32)
    tasks_manifest = pd.read_csv(TASKS, keep_default_na=False)
    scores = pd.read_csv(PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv")
    truth, access = aggregate_evaluation_truth(
        source, tasks_manifest, panel, control, builder, batch_size=1024
    )
    tasks, clusters, efficiency, ranking = build_results(
        scores, truth, calibration_model
    )
    if tasks["pair_lower_mean_violation"].astype(bool).any() or tasks[
        "pair_lower_max_violation"
    ].astype(bool).any():
        raise IntegrityError("E180 deterministic lower-bound violation")
    bootstrap = bootstrap_efficiency(tasks)
    primary = efficiency.set_index("method").loc["extra_trees_vector"]
    constant = efficiency.set_index("method").loc["constant"]
    gates = {
        "pair_lower_bound_violations": int(
            tasks["pair_lower_mean_violation"].sum()
            + tasks["pair_lower_max_violation"].sum()
        )
        == 0,
        "evaluation_target_simultaneous_coverage_minimum_0_85": bool(
            primary["target_simultaneous_coverage"] >= 0.85
        ),
        "adaptive_mean_upper_not_above_constant": bool(
            primary["mean_upper"] <= constant["mean_upper"] + 1e-12
        ),
        "truth_access_or_split_violation_zero": bool(
            set(access["task_id"].astype(str))
            == set(
                tasks_manifest.loc[
                    tasks_manifest["target_split"].eq("prospective_evaluation"),
                    "task_id",
                ].astype(str)
            )
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    sc_ge_spearman = spearman(tasks["scgpt_rmse"], tasks["gears_rmse"])
    primary_minus_constant = (
        float(primary["mean_upper"]) - float(constant["mean_upper"])
    )
    boot_low, boot_high = bootstrap[
        "adaptive_minus_constant_mean_upper"
    ].quantile([0.025, 0.975])

    try:
        for sub in ("tables", "arrays", "figures", "reports"):
            (STAGING / sub).mkdir(parents=True, exist_ok=False)
        atomic_csv(STAGING / "tables/EVALUATION_TASK_RESULTS.csv", tasks)
        atomic_csv(STAGING / "tables/EVALUATION_TARGET_RESULTS.csv", clusters)
        atomic_csv(STAGING / "tables/COVERAGE_EFFICIENCY.csv", efficiency)
        atomic_csv(STAGING / "tables/RANKING_DIAGNOSTICS.csv", ranking)
        atomic_csv(STAGING / "tables/CLUSTER_BOOTSTRAP_DRAWS.csv", bootstrap)
        atomic_csv(STAGING / "tables/EVALUATION_ROW_ACCESS_AUDIT.csv", access)
        atomic_npz(STAGING / "arrays/EVALUATION_TRUE_EFFECTS.npz", truth)
        input_hashes = hashes + [
            {
                "path": str(F2_ROOT / name),
                "bytes": (F2_ROOT / name).stat().st_size,
                "sha256": sha256_file(F2_ROOT / name),
            }
            for name in ("GENE_PANEL.csv", "CONTROL_PROFILES.npz", "MANIFEST.sha256")
        ]
        atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(input_hashes))
        make_figures(tasks, efficiency, ranking, bootstrap, STAGING / "figures")
        summary = {
            "schema": "safeconf_e180_final_summary_v1",
            "experiment": "E180_xucao_fresh_guide_certificate",
            "status": status,
            "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "git_head_before_truth_open": head,
            "git_branch": branch,
            "remote_heads_before_truth_open": remotes,
            "n_evaluation_targets": int(clusters.shape[0]),
            "n_evaluation_guide_tasks": int(tasks.shape[0]),
            "evaluation_x_rows_read": int(access.shape[0]),
            "pair_lower_mean_violations": int(
                tasks["pair_lower_mean_violation"].sum()
            ),
            "pair_lower_max_violations": int(
                tasks["pair_lower_max_violation"].sum()
            ),
            "median_pair_lower_tightness": float(
                (tasks["pair_lower_bound"] / tasks["pair_mean_rmse"]).median()
            ),
            "scgpt_gears_error_spearman": sc_ge_spearman,
            "primary_target_simultaneous_coverage": float(
                primary["target_simultaneous_coverage"]
            ),
            "constant_target_simultaneous_coverage": float(
                constant["target_simultaneous_coverage"]
            ),
            "primary_mean_upper": float(primary["mean_upper"]),
            "constant_mean_upper": float(constant["mean_upper"]),
            "primary_minus_constant_mean_upper": primary_minus_constant,
            "bootstrap_ci95_primary_minus_constant_mean_upper": [
                float(boot_low),
                float(boot_high),
            ],
            "prospective_success_gates": gates,
            "evaluation_truth_opened_once": True,
            "posttruth_method_or_target_change": False,
        }
        atomic_json(STAGING / "E180_FINAL_SUMMARY.json", summary)
        report = f"""# E180 XuCao2023 新研究最终评价

## 结论

E180 在 {len(clusters)} 个从未用于训练或校准的基因、{len(tasks)} 个 guide 任务上完成了一次性评价。确定性两模型证书继续保持 **0 个违例**；scGPT 与 GEARS 的任务误差 Spearman 为 **{sc_ge_spearman:.3f}**，再次说明共享任务难度强于单纯模型分歧排序。

主 ExtraTrees 上界的靶点同时覆盖率为 **{primary['target_simultaneous_coverage']:.3f}**，常数 split conformal 为 **{constant['target_simultaneous_coverage']:.3f}**。主方法平均上界为 **{primary['mean_upper']:.4f}**，常数上界为 **{constant['mean_upper']:.4f}**，差值为 **{primary_minus_constant:+.4f} RMSE**；靶点簇 bootstrap 的 95% 区间为 **[{boot_low:+.4f}, {boot_high:+.4f}]**。

预注册总状态：**{status}**。该状态由四个冻结门槛共同决定，失败项不会在本实验编号内通过换模型或换靶点修补。

## 证据表

- [任务级结果](../tables/EVALUATION_TASK_RESULTS.csv)
- [靶点级结果](../tables/EVALUATION_TARGET_RESULTS.csv)
- [覆盖与效率](../tables/COVERAGE_EFFICIENCY.csv)
- [最终摘要](../E180_FINAL_SUMMARY.json)

## 图

![冻结设计](../figures/F1_E180_FROZEN_DESIGN.png)

![上界效率](../figures/F2_E180_UPPER_EFFICIENCY.png)

![覆盖率](../figures/F3_E180_COVERAGE_FOREST.png)

![确定性下界](../figures/F4_E180_LOWER_CERTIFICATE.png)

![任务难度与效率](../figures/F5_E180_DIFFICULTY_AND_EFFICIENCY.png)

## 解释

确定性下界回答“两个预测器不可能同时都比它更准”，不依赖校准数据。上界回答“在与 calibration 靶点可交换的前提下，新的完整基因簇有多大概率被覆盖”。两者分别承担不可违背的几何证书和带条件的统计覆盖，不能互相替代。
"""
        atomic_bytes(STAGING / "reports/E180_FINAL_REPORT.md", report.encode())
        os.replace(STAGING, RELEASE)
    except Exception:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
