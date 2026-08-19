#!/usr/bin/env python3
"""Synthesize E168/E172 and formalize a falsification-aware pair-risk certificate.

This is a post-truth audit.  It does not claim a new independent confirmation,
retrain a predictor, or modify the frozen SafeConf score.  Its purpose is to
separate the failed ensemble-error routing claim from the deterministic risk
statement that is actually identifiable from two prediction vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
E168 = ROOT / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
E171 = ROOT / "docs/实验结果/E171_seed_ensemble_gate_development_20260718"
E172 = ROOT / "docs/实验结果/E172_primary_cd4_fresh_targets_20260718"
OUT = ROOT / "docs/实验结果/E173_falsification_aware_pair_certificate_20260719"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
PANELS = ("E168", "Q01", "Q02", "Q03", "Q04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
SCORES = {
    "SafeConf": "safeconf_risk",
    "disagreement": "model_disagreement_rmse",
    "magnitude": "predicted_magnitude",
}
OBJECTIVES = {
    "ensemble_rmse": "ensemble_rmse",
    "pair_mean_rmse": "pair_mean_rmse",
    "pair_max_rmse": "pair_max_rmse",
}
BOOTSTRAPS = 3000
BOOTSTRAP_SEED = 2026071901
BOUND_TOLERANCE = 1.0e-9
IDENTITY_TOLERANCE = 1.0e-7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def require_committed(path: Path, head: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing or symlinked E173 input: {path}")
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{head}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"uncommitted E173 input: {relative}") from exc
    if committed != path.read_bytes():
        raise RuntimeError(f"E173 input differs from HEAD: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_release_manifest(release: Path) -> dict[str, str]:
    manifest_path = release / "MANIFEST.sha256"
    rows: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in rows:
            raise RuntimeError(f"invalid release manifest entry: {line}")
        rows[relative] = digest
    for relative, expected in rows.items():
        path = release / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"release artifact changed: {path}")
    return rows


def verify_code_and_inputs() -> tuple[str, str, dict[str, str], pd.DataFrame]:
    head = git_text("rev-parse", "HEAD")
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise RuntimeError("E173 requires a named Git branch")
    inputs = [
        SCRIPT,
        E168 / "postgate_release/MANIFEST.sha256",
        E168 / "postgate_release/RUN_STATUS.json",
        E168 / "postgate_release/tables/TASK_METRICS.csv",
        E168 / "postgate_release/tables/PREDICTION_RECORDS.csv",
        E168 / "postgate_release/tables/PRIMARY_INFERENCE.csv",
        E168 / "manifests/E168_SELECTED_TARGETS.csv",
        E171 / "RUN_STATUS.json",
        E172 / "postgate_release/MANIFEST.sha256",
        E172 / "postgate_release/RUN_STATUS.json",
        E172 / "postgate_release/tables/TASK_METRICS.csv",
        E172 / "postgate_release/tables/PREDICTION_RECORDS.csv",
        E172 / "postgate_release/tables/JOINT_PRIMARY_INFERENCE.csv",
        E172 / "manifests/E172_ALL_SELECTED_TARGETS.csv",
    ]
    hashes = pd.DataFrame([require_committed(path, head) for path in inputs])
    verify_release_manifest(E168 / "postgate_release")
    verify_release_manifest(E172 / "postgate_release")
    remote_heads: dict[str, str] = {}
    for remote in ("github", "origin"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git(
            "fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False
        )
        if result.returncode:
            raise RuntimeError(f"cannot verify E173 code on {remote}")
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", head, remote_head, check=False).returncode:
            raise RuntimeError(f"E173 HEAD absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return head, branch, remote_heads, hashes


def rho(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    if np.std(left_rank) <= 0 or np.std(right_rank) <= 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def load_experiment(experiment: Path, default_panel: str | None) -> pd.DataFrame:
    tables = experiment / "postgate_release/tables"
    metrics = pd.read_csv(tables / "TASK_METRICS.csv", keep_default_na=False)
    records = pd.read_csv(tables / "PREDICTION_RECORDS.csv", keep_default_na=False)
    if "panel_id" not in metrics:
        if default_panel is None:
            raise RuntimeError("panel identity missing")
        metrics["panel_id"] = default_panel
    wanted = records.loc[
        records.predictor_name.isin(
            ["scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean"]
        ),
        ["task_id", "predictor_name", "true_error_rmse"],
    ]
    pivot = wanted.pivot(
        index="task_id", columns="predictor_name", values="true_error_rmse"
    )
    if set(pivot.columns) != {
        "scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean"
    }:
        raise RuntimeError(f"formal family-mean predictors missing: {experiment}")
    frame = metrics.set_index("task_id").join(pivot).reset_index()
    frame["ensemble_rmse"] = frame.ensemble_seed_family_mean.astype(float)
    frame["pair_mean_rmse"] = (
        frame.scGPT_seed_mean.astype(float) + frame.GEARS_seed_mean.astype(float)
    ) / 2.0
    frame["pair_max_rmse"] = frame[
        ["scGPT_seed_mean", "GEARS_seed_mean"]
    ].astype(float).max(axis=1)
    frame["pair_lower_bound_rmse"] = frame.model_disagreement_rmse.astype(float) / 2.0
    frame["pair_mean_mse"] = (
        frame.scGPT_seed_mean.astype(float) ** 2
        + frame.GEARS_seed_mean.astype(float) ** 2
    ) / 2.0
    frame["decomposition_rhs_mse"] = (
        frame.ensemble_rmse.astype(float) ** 2
        + frame.model_disagreement_rmse.astype(float) ** 2 / 4.0
    )
    frame["decomposition_abs_residual"] = np.abs(
        frame.pair_mean_mse - frame.decomposition_rhs_mse
    )
    frame["bound_tightness_pair_mean"] = (
        frame.pair_lower_bound_rmse / frame.pair_mean_rmse
    )
    return frame


def validate_population(frame: pd.DataFrame) -> None:
    if len(frame) != 3000 or frame.task_id.nunique() != 3000:
        raise RuntimeError("E168+E172 must contain 3,000 unique test tasks")
    if set(frame.panel_id.astype(str)) != set(PANELS):
        raise RuntimeError("five frozen target panels are not all present")
    for panel in PANELS:
        block = frame.loc[frame.panel_id.eq(panel)]
        if len(block) != 600 or block.perturbed_gene_id.nunique() != 200:
            raise RuntimeError(f"{panel} is not a 200-target three-state panel")
    if not np.isfinite(
        frame[list(SCORES.values()) + list(OBJECTIVES.values())].to_numpy(float)
    ).all():
        raise RuntimeError("non-finite score or objective")
    earlier = set(
        pd.read_csv(E168 / "manifests/E168_SELECTED_TARGETS.csv").ensembl_core.astype(str)
    )
    later = set(
        pd.read_csv(E172 / "manifests/E172_ALL_SELECTED_TARGETS.csv").ensembl_core.astype(str)
    )
    if earlier & later or len(earlier) != 200 or len(later) != 800:
        raise RuntimeError("fresh-target panels overlap or changed size")


def unit_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        for state in STATES:
            block = frame.loc[
                frame.panel_id.eq(panel) & frame.culture_condition.eq(state)
            ]
            if len(block) != 200:
                raise RuntimeError(f"unit count changed: {panel}/{state}")
            for objective_name, objective_column in OBJECTIVES.items():
                for score_name, score_column in SCORES.items():
                    rows.append(
                        {
                            "panel_id": panel,
                            "culture_condition": state,
                            "objective": objective_name,
                            "score_name": score_name,
                            "n_tasks": len(block),
                            "spearman": rho(
                                block[score_column].to_numpy(float),
                                block[objective_column].to_numpy(float),
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    comparator = result.loc[result.score_name.eq("magnitude")].rename(
        columns={"spearman": "magnitude_spearman"}
    )[
        ["panel_id", "culture_condition", "objective", "magnitude_spearman"]
    ]
    return result.merge(
        comparator, on=["panel_id", "culture_condition", "objective"], how="left"
    ).assign(
        delta_vs_magnitude=lambda value: value.spearman - value.magnitude_spearman
    )


def build_panel_arrays(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    columns = list(SCORES.values()) + list(OBJECTIVES.values())
    for panel in PANELS:
        block = frame.loc[frame.panel_id.eq(panel)].copy()
        genes = sorted(block.perturbed_gene_id.astype(str).unique())
        arrays: dict[str, np.ndarray] = {}
        for column in columns:
            pivot = block.pivot(
                index="perturbed_gene_id", columns="culture_condition", values=column
            ).loc[genes, list(STATES)]
            if pivot.shape != (200, 3):
                raise RuntimeError(f"panel array failed: {panel}/{column}/{pivot.shape}")
            arrays[column] = pivot.to_numpy(float)
        result[panel] = arrays
    return result


def bootstrap_macro(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_arrays = build_panel_arrays(frame)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_pairs = [
        (objective_name, score_name, objective_column, score_column)
        for objective_name, objective_column in OBJECTIVES.items()
        for score_name, score_column in SCORES.items()
    ]
    observed_rows = []
    for objective_name, score_name, objective_column, score_column in metric_pairs:
        values = []
        for panel in PANELS:
            for state_index in range(3):
                values.append(
                    rho(
                        panel_arrays[panel][score_column][:, state_index],
                        panel_arrays[panel][objective_column][:, state_index],
                    )
                )
        observed_rows.append(
            {
                "objective": objective_name,
                "score_name": score_name,
                "observed_equal_panel_state_spearman": float(np.mean(values)),
                "positive_panel_state_units": int(np.sum(np.asarray(values) > 0)),
                "n_panel_state_units": len(values),
            }
        )
    observed = pd.DataFrame(observed_rows)
    draws = np.empty((BOOTSTRAPS, len(metric_pairs)), dtype=np.float64)
    for draw in range(BOOTSTRAPS):
        sampled = {
            panel: rng.integers(0, 200, size=200) for panel in PANELS
        }
        for metric_index, (
            _objective_name, _score_name, objective_column, score_column
        ) in enumerate(metric_pairs):
            values = []
            for panel in PANELS:
                take = sampled[panel]
                for state_index in range(3):
                    values.append(
                        rho(
                            panel_arrays[panel][score_column][take, state_index],
                            panel_arrays[panel][objective_column][take, state_index],
                        )
                    )
            draws[draw, metric_index] = float(np.nanmean(values))
    summary_rows = []
    draw_rows = []
    for metric_index, (
        objective_name, score_name, _objective_column, _score_column
    ) in enumerate(metric_pairs):
        values = draws[:, metric_index]
        low, high = np.quantile(values, [0.025, 0.975])
        record = observed.loc[
            observed.objective.eq(objective_name) & observed.score_name.eq(score_name)
        ].iloc[0].to_dict()
        summary_rows.append(
            {
                **record,
                "bootstrap_draws": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_ci95_lower": float(low),
                "bootstrap_ci95_upper": float(high),
                "bootstrap_fraction_gt_zero": float(np.mean(values > 0)),
            }
        )
        for draw, value in enumerate(values):
            draw_rows.append(
                {
                    "draw": draw,
                    "objective": objective_name,
                    "score_name": score_name,
                    "equal_panel_state_spearman": float(value),
                }
            )
    summary = pd.DataFrame(summary_rows)
    draw_frame = pd.DataFrame(draw_rows)
    for objective_name in OBJECTIVES:
        dis = draws[:, metric_pairs.index(
            (objective_name, "disagreement", OBJECTIVES[objective_name], SCORES["disagreement"])
        )]
        mag = draws[:, metric_pairs.index(
            (objective_name, "magnitude", OBJECTIVES[objective_name], SCORES["magnitude"])
        )]
        delta = dis - mag
        low, high = np.quantile(delta, [0.025, 0.975])
        dis_observed = summary.loc[
            summary.objective.eq(objective_name) & summary.score_name.eq("disagreement"),
            "observed_equal_panel_state_spearman",
        ].iloc[0]
        mag_observed = summary.loc[
            summary.objective.eq(objective_name) & summary.score_name.eq("magnitude"),
            "observed_equal_panel_state_spearman",
        ].iloc[0]
        summary_rows.append(
            {
                "objective": objective_name,
                "score_name": "disagreement_minus_magnitude",
                "observed_equal_panel_state_spearman": float(dis_observed - mag_observed),
                "positive_panel_state_units": int(
                    (
                        unit_ranking(frame).query(
                            "objective == @objective_name and score_name == 'disagreement'"
                        ).delta_vs_magnitude
                        > 0
                    ).sum()
                ),
                "n_panel_state_units": 15,
                "bootstrap_draws": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_ci95_lower": float(low),
                "bootstrap_ci95_upper": float(high),
                "bootstrap_fraction_gt_zero": float(np.mean(delta > 0)),
            }
        )
        for draw, value in enumerate(delta):
            draw_rows.append(
                {
                    "draw": draw,
                    "objective": objective_name,
                    "score_name": "disagreement_minus_magnitude",
                    "equal_panel_state_spearman": float(value),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def feature_identifiability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        for state in STATES:
            for stratum in ("DONOR_UNSEEN_ONLY", "COLUMN_UNSEEN"):
                block = frame.loc[
                    frame.panel_id.eq(panel)
                    & frame.culture_condition.eq(state)
                    & frame.target_stratum.eq(stratum)
                ]
                expected = 160 if stratum == "DONOR_UNSEEN_ONLY" else 40
                if len(block) != expected:
                    raise RuntimeError(f"stratum changed: {panel}/{state}/{stratum}")
                safe = block.safeconf_risk.to_numpy(float)
                disagreement = block.model_disagreement_rmse.to_numpy(float)
                rows.append(
                    {
                        "panel_id": panel,
                        "culture_condition": state,
                        "target_stratum": stratum,
                        "n_tasks": len(block),
                        "context_similarity_unique_1e_6": int(
                            np.unique(np.rint(block.context_similarity_max / 1e-6)).size
                        ),
                        "support_count_unique": int(
                            block.perturbation_support_count.astype(int).nunique()
                        ),
                        "safeconf_disagreement_spearman": rho(safe, disagreement),
                        "safeconf_disagreement_rank_identical": bool(
                            np.array_equal(
                                rankdata(safe, method="average"),
                                rankdata(disagreement, method="average"),
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def certificate_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for panel in [*PANELS, "ALL_1000_TARGETS"]:
        block = frame if panel == "ALL_1000_TARGETS" else frame.loc[frame.panel_id.eq(panel)]
        rows.append(
            {
                "panel_id": panel,
                "n_tasks": len(block),
                "pair_mean_bound_violations": int(
                    (
                        block.pair_lower_bound_rmse
                        > block.pair_mean_rmse + BOUND_TOLERANCE
                    ).sum()
                ),
                "pair_max_bound_violations": int(
                    (
                        block.pair_lower_bound_rmse
                        > block.pair_max_rmse + BOUND_TOLERANCE
                    ).sum()
                ),
                "squared_error_decomposition_max_abs_residual": float(
                    block.decomposition_abs_residual.max()
                ),
                "bound_tightness_pair_mean_median": float(
                    block.bound_tightness_pair_mean.median()
                ),
                "bound_tightness_pair_mean_q10": float(
                    block.bound_tightness_pair_mean.quantile(0.10)
                ),
                "bound_tightness_pair_mean_q90": float(
                    block.bound_tightness_pair_mean.quantile(0.90)
                ),
            }
        )
    thresholds = []
    for tolerance in (0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050):
        certified = frame.pair_lower_bound_rmse > tolerance
        thresholds.append(
            {
                "error_tolerance_rmse": tolerance,
                "n_tasks": len(frame),
                "n_certified_pair_mean_above_tolerance": int(certified.sum()),
                "certified_fraction": float(certified.mean()),
                "empirical_false_certificates_pair_mean": int(
                    (certified & (frame.pair_mean_rmse <= tolerance)).sum()
                ),
                "empirical_false_certificates_pair_max": int(
                    (certified & (frame.pair_max_rmse <= tolerance)).sum()
                ),
                "guarantee_source": "triangle_inequality_not_empirical_calibration",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def confirmation_outcomes() -> pd.DataFrame:
    e168 = pd.read_csv(
        E168 / "postgate_release/tables/PRIMARY_INFERENCE.csv"
    ).loc[lambda x: x.stratum.eq("all_200")].iloc[0]
    e172 = pd.read_csv(
        E172 / "postgate_release/tables/JOINT_PRIMARY_INFERENCE.csv"
    ).loc[lambda x: x.stratum.eq("all_800")].iloc[0]
    return pd.DataFrame(
        [
            {
                "experiment": "E168",
                "n_fresh_targets": 200,
                "n_panel_state_units": 3,
                "endpoint": "AURC_magnitude_minus_SafeConf",
                "estimate": float(e168.observed_mean_state_delta),
                "ci95_lower": float(e168.bootstrap_ci95_lower),
                "ci95_upper": float(e168.bootstrap_ci95_upper),
                "permutation_p_one_sided": float(e168.permutation_p_one_sided),
                "formal_decision": "NO_CONFIRMATION",
            },
            {
                "experiment": "E172",
                "n_fresh_targets": 800,
                "n_panel_state_units": 12,
                "endpoint": "AURC_magnitude_minus_SafeConf",
                "estimate": float(e172.observed_equal_panel_state_delta),
                "ci95_lower": float(e172.bootstrap_ci95_lower),
                "ci95_upper": float(e172.bootstrap_ci95_upper),
                "permutation_p_one_sided": float(e172.permutation_p_one_sided),
                "formal_decision": "NO_TARGET_REPLICATION",
            },
        ]
    )


def configure_plots() -> None:
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
            "font.size": 9,
        }
    )


def save_figure(fig: Any, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    svg = FIGURES / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def make_figures(
    frame: pd.DataFrame,
    bootstrap: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> None:
    configure_plots()
    colors = {"SafeConf": "#6B7280", "disagreement": "#176B87", "magnitude": "#C47A34"}
    objectives = ["ensemble_rmse", "pair_mean_rmse", "pair_max_rmse"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), sharey=True)
    for ax, objective in zip(axes, objectives, strict=True):
        block = bootstrap.loc[
            bootstrap.objective.eq(objective)
            & bootstrap.score_name.isin(["SafeConf", "disagreement", "magnitude"])
        ].set_index("score_name").loc[["SafeConf", "disagreement", "magnitude"]]
        x = np.arange(3)
        y = block.observed_equal_panel_state_spearman.to_numpy(float)
        low = block.bootstrap_ci95_lower.to_numpy(float)
        high = block.bootstrap_ci95_upper.to_numpy(float)
        ax.errorbar(
            x, y, yerr=np.vstack([y - low, high - y]), fmt="none",
            ecolor="#9CA3AF", capsize=3, linewidth=1.1,
        )
        ax.scatter(x, y, s=38, color=[colors[name] for name in block.index], zorder=3)
        ax.axhline(0, color="#D1D5DB", linewidth=0.9)
        ax.set_xticks(x, ["SafeConf", "Disagree", "Magnitude"], rotation=25, ha="right")
        ax.set_title(objective.replace("_", " "), loc="left", fontweight="bold")
    axes[0].set_ylabel("Equal panel×state Spearman")
    fig.suptitle("Objective separation on 1,000 disjoint fresh targets", x=0.07, ha="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "F1_objective_separation")

    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    y = np.arange(len(confirmation))
    estimates = confirmation.estimate.to_numpy(float)
    ax.errorbar(
        estimates, y,
        xerr=np.vstack([
            estimates - confirmation.ci95_lower.to_numpy(float),
            confirmation.ci95_upper.to_numpy(float) - estimates,
        ]),
        fmt="o", color="#176B87", ecolor="#6B7280", capsize=4,
    )
    ax.axvline(0, color="#C94C4C", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, ["E168 · 200 targets", "E172 · 800 targets"])
    ax.set_xlabel("Δ AURC (magnitude − SafeConf); positive favors SafeConf")
    ax.set_title("Two frozen fresh-target tests did not confirm increment", loc="left", fontweight="bold")
    save_figure(fig, "F2_fresh_target_falsification")

    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    for panel, color in zip(PANELS, ["#173F5F", "#20639B", "#3CAEA3", "#F6D55C", "#C94C4C"], strict=True):
        values = np.sort(
            frame.loc[frame.panel_id.eq(panel), "bound_tightness_pair_mean"].to_numpy(float)
        )
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=panel, color=color, linewidth=1.3)
    ax.set_xlabel("Certified lower bound / observed pair-mean RMSE")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title("Certificate validity is exact; tightness is limited", loc="left", fontweight="bold")
    save_figure(fig, "F3_pair_certificate_tightness")

    fig, ax = plt.subplots(figsize=(9.4, 2.5))
    ax.axis("off")
    boxes = [
        (0.02, "Two frozen\nprediction vectors", "#E8F1F5"),
        (0.27, "RIAG\nG2/G3/G4", "#E8F1F5"),
        (0.52, "Pair certificate\nd(p1,p2)/2", "#E7F3EC"),
        (0.77, "Validation increment\nvs magnitude?", "#FFF3E3"),
    ]
    for x, text_value, color in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.19, 0.38, facecolor=color, edgecolor="#667085", linewidth=1.1))
        ax.text(x + 0.095, 0.54, text_value, ha="center", va="center", fontsize=9)
    for x in (0.21, 0.46, 0.71):
        ax.annotate("", xy=(x + 0.055, 0.54), xytext=(x, 0.54), arrowprops={"arrowstyle": "->", "color": "#667085"})
    ax.text(0.865, 0.22, "PASS → calibrated router", ha="center", color="#176B87", fontsize=9)
    ax.text(0.865, 0.10, "FAIL → ABSTAIN; retain certificate + magnitude", ha="center", color="#C94C4C", fontsize=9)
    ax.set_title("Falsification-aware SafeConf-Cert decision path", loc="left", fontweight="bold")
    save_figure(fig, "F4_fail_closed_decision_path")


def write_manifest() -> str:
    files = sorted(
        path
        for path in OUT.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.sha256", "RUN_STATUS.json"}
    )
    payload = "".join(
        f"{sha256_file(path)}  {path.relative_to(OUT).as_posix()}\n" for path in files
    )
    atomic_bytes(OUT / "MANIFEST.sha256", payload.encode())
    return sha256_file(OUT / "MANIFEST.sha256")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"append-only E173 output exists: {OUT}")
    head, branch, remote_heads, input_hashes = verify_code_and_inputs()
    e168_status = json.loads((E168 / "postgate_release/RUN_STATUS.json").read_text())
    e171_status = json.loads((E171 / "RUN_STATUS.json").read_text())
    e172_status = json.loads((E172 / "postgate_release/RUN_STATUS.json").read_text())
    if e172_status.get("decision") != "NO_TARGET_REPLICATION":
        raise RuntimeError("E172 formal decision changed")
    if e171_status.get("performance_rescue_claim_supported") is not False:
        raise RuntimeError("E171 validation boundary changed")
    frame = pd.concat(
        [load_experiment(E168, "E168"), load_experiment(E172, None)],
        ignore_index=True,
    )
    validate_population(frame)
    units = unit_ranking(frame)
    bootstrap, draws = bootstrap_macro(frame)
    identifiability = feature_identifiability(frame)
    certificate, thresholds = certificate_tables(frame)
    confirmation = confirmation_outcomes()
    overall = certificate.loc[certificate.panel_id.eq("ALL_1000_TARGETS")].iloc[0]
    if (
        overall.pair_mean_bound_violations != 0
        or overall.pair_max_bound_violations != 0
        or overall.squared_error_decomposition_max_abs_residual > IDENTITY_TOLERANCE
    ):
        raise RuntimeError("pair certificate numeric validation failed")
    for directory in (OUT, TABLES, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_csv(TABLES / "E173_TASK_OBJECTIVES.csv", frame)
    atomic_csv(TABLES / "E173_UNIT_RANKING.csv", units)
    atomic_csv(TABLES / "E173_CLUSTER_BOOTSTRAP_SUMMARY.csv", bootstrap)
    atomic_csv(TABLES / "E173_CLUSTER_BOOTSTRAP_DRAWS.csv", draws)
    atomic_csv(TABLES / "E173_FEATURE_IDENTIFIABILITY.csv", identifiability)
    atomic_csv(TABLES / "E173_PAIR_CERTIFICATE_AUDIT.csv", certificate)
    atomic_csv(TABLES / "E173_CERTIFICATE_THRESHOLD_CURVE.csv", thresholds)
    atomic_csv(TABLES / "E173_FRESH_CONFIRMATION_OUTCOMES.csv", confirmation)
    atomic_csv(TABLES / "E173_INPUT_HASHES.csv", input_hashes)
    make_figures(frame, bootstrap, confirmation)
    pair_mean = bootstrap.loc[
        bootstrap.objective.eq("pair_mean_rmse")
        & bootstrap.score_name.eq("disagreement")
    ].iloc[0]
    pair_max = bootstrap.loc[
        bootstrap.objective.eq("pair_max_rmse")
        & bootstrap.score_name.eq("disagreement")
    ].iloc[0]
    delta_pair_mean = bootstrap.loc[
        bootstrap.objective.eq("pair_mean_rmse")
        & bootstrap.score_name.eq("disagreement_minus_magnitude")
    ].iloc[0]
    identical_seen = identifiability.loc[
        identifiability.target_stratum.eq("DONOR_UNSEEN_ONLY"),
        "safeconf_disagreement_rank_identical",
    ]
    report = f"""# E173｜E172 失败后的可证伪方法收缩

## 正式结论先写清楚

E168 的 200 个新目标未确认 SafeConf 相对 magnitude 的 AURC 增量；E172 又在完全不重叠的 800 个新目标上得到 `NO_TARGET_REPLICATION`。因此，固定 absolute-RMSE SafeConf 不能再写成稳定超过 predicted magnitude 的主方法。E171 validation 也没有提前支持 performance rescue，这条负证据不是统计偶然或事后挑面板可以解决的问题。

## 失败来自什么

同一 state、同一 target stratum 内，context similarity 与 support 均为常数。seen strata 的 {int(identical_seen.sum())}/{len(identical_seen)} 个 panel×state 单元中，SafeConf 与 disagreement 的任务排序完全一致。固定公式在这些单元没有提供独立于模型分歧的新排序信息；巨大 context z 值只改变单元间位置，不改善单元内 AURC。

## 仍然成立的模型对证书

对任意两个预测向量 `p1,p2` 和未知真值 `y`，`d(p1,p2)/2` 同时下界 pair mean RMSE 与 pair max RMSE。E168+E172 共 1,000 个互斥目标、3,000 个任务的数值核验中，mean/max 下界违例均为 0；平方误差分解最大绝对残差为 {overall.squared_error_decomposition_max_abs_residual:.3g}。

分歧对 pair mean RMSE 的 15 单元等权 Spearman 为 {pair_mean.observed_equal_panel_state_spearman:.3f}，target-cluster bootstrap 95% CI [{pair_mean.bootstrap_ci95_lower:.3f}, {pair_mean.bootstrap_ci95_upper:.3f}]；对 pair max RMSE 为 {pair_max.observed_equal_panel_state_spearman:.3f}，CI [{pair_max.bootstrap_ci95_lower:.3f}, {pair_max.bootstrap_ci95_upper:.3f}]。相对 magnitude 的 pair-mean Δrho 仅 {delta_pair_mean.observed_equal_panel_state_spearman:.3f}，CI [{delta_pair_mean.bootstrap_ci95_lower:.3f}, {delta_pair_mean.bootstrap_ci95_upper:.3f}]，不能宣称排序增量。

证书的价值不依赖相关性：当 `d/2 > tau` 时，可在没有目标真值的情况下证明两模型平均误差和至少一个模型误差超过 `tau`。它不能指出哪一个模型错，也不能把小分歧解释为安全。

## 修正后的系统定义

`SafeConf-Cert` 分成两个输出。第一层始终报告模型对下界并运行 RIAG 的预测/分数可识别性检查。第二层经验路由必须在 validation 上相对 magnitude 通过预先冻结的增量 gate；未通过就输出 `ABSTAIN_INCREMENTAL_ROUTING`，保留 certificate 和 magnitude 参考，不进入测试性能宣传。按这条规则，E171 会在 test truth 前拒绝 E172 的增量路由，而不会事后把失败解释成成功。

E173 是已解封数据上的方法收缩和二级审计，不是新的独立确认。下一项正式实验需要换 test donor，并使用全新目标；其主要对象应为模型对证书与 fail-closed 决策，而不是再次检验已被两次否定的固定公式优势。
"""
    atomic_bytes(REPORTS / "E173_REPORT.md", report.encode())
    atomic_bytes(
        OUT / "README_先看这个.md",
        b"# E173\n\nRead `reports/E173_REPORT.md` first. E173 is a post-truth falsification audit, not independent confirmation.\n",
    )
    manifest_sha = write_manifest()
    status = {
        "schema": "safeconf_e173_falsification_aware_pair_certificate_v1",
        "stage": "POST_TRUTH_METHOD_CONTRACTION",
        "status": "COMPLETE",
        "git_head": head,
        "git_branch": branch,
        "remote_heads": remote_heads,
        "python": sys.version,
        "platform": platform.platform(),
        "n_disjoint_fresh_targets": 1000,
        "n_tasks": len(frame),
        "n_panels": 5,
        "n_panel_state_units": 15,
        "e168_formal_decision": e168_status.get("decision"),
        "e172_formal_decision": e172_status.get("decision"),
        "fixed_safeconf_stable_increment_vs_magnitude_supported": False,
        "pair_mean_bound_violations": int(overall.pair_mean_bound_violations),
        "pair_max_bound_violations": int(overall.pair_max_bound_violations),
        "squared_error_decomposition_max_abs_residual": float(
            overall.squared_error_decomposition_max_abs_residual
        ),
        "pair_mean_disagreement_macro_spearman": float(
            pair_mean.observed_equal_panel_state_spearman
        ),
        "pair_mean_disagreement_ci95": [
            float(pair_mean.bootstrap_ci95_lower), float(pair_mean.bootstrap_ci95_upper)
        ],
        "pair_mean_disagreement_minus_magnitude": float(
            delta_pair_mean.observed_equal_panel_state_spearman
        ),
        "pair_mean_disagreement_minus_magnitude_ci95": [
            float(delta_pair_mean.bootstrap_ci95_lower),
            float(delta_pair_mean.bootstrap_ci95_upper),
        ],
        "revised_method_state": "PAIR_CERTIFICATE_PLUS_FAIL_CLOSED_EMPIRICAL_ROUTER",
        "incremental_router_state_on_e171_validation": "ABSTAIN",
        "independent_confirmation_claim": False,
        "source_expression_opened": False,
        "predictors_retrained": False,
        "safeconf_score_modified": False,
        "manifest_sha256": manifest_sha,
        "deployment_authorized": False,
    }
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
