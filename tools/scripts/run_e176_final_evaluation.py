#!/usr/bin/env python3
"""Evaluate E176 once on 640 still-sealed targets across four held-out donors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
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
import run_e176_donor_specific_calibration as calibration


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
CALIBRATION = EXPERIMENT / "calibration_release"
CALIBRATION_SNAPSHOT = CALIBRATION / "CALIBRATION_GATE_SNAPSHOT.json"
OUT = EXPERIMENT / "final_evaluation"
STAGING = EXPERIMENT / ".final_evaluation.staging"
PANELS = ("H01", "H02", "H03", "H04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026071906
SCRIPT = Path(__file__).resolve()
TRUTH_BUILDER = ROOT / "tools/scripts/build_e176_truth_assets.py"
PRETRUTH_BUILDER = ROOT / "tools/scripts/build_e176_four_donor_panel_assets.py"
POSTGATE_HELPER = ROOT / "tools/scripts/run_e168_primary_cd4_postgate.py"


class IntegrityFailure(RuntimeError):
    pass


def import_postgate() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e176_postgate_math", POSTGATE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import postgate math: {POSTGATE_HELPER}")
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


def verify_calibration_gate(
    commit: str, branch: str
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    remote_heads, hashes = calibration.verify_code_freeze(commit, branch)
    hashes.extend([
        calibration.require_committed(SCRIPT, commit),
        calibration.require_committed(POSTGATE_HELPER, commit),
        calibration.require_committed(CALIBRATION_SNAPSHOT, commit),
        calibration.require_committed(CALIBRATION / "RUN_STATUS.json", commit),
        calibration.require_committed(CALIBRATION / "MANIFEST.sha256", commit),
    ])
    snapshot = json.loads(CALIBRATION_SNAPSHOT.read_text())
    required = {
        "schema": "safeconf_e176_donor_specific_calibration_gate_v1",
        "status": "PASS",
        "n_calibration_targets": 160,
        "n_calibration_tasks": 480,
        "n_calibration_targets_each_donor": 40,
        "finite_sample_order_rank_each_donor": 37,
        "evaluation_targeting_x_values_read": 0,
        "method_frozen_before_evaluation_truth": True,
        "final_evaluator_authorized": True,
    }
    changed = {
        key: {"expected": value, "observed": snapshot.get(key)}
        for key, value in required.items() if snapshot.get(key) != value
    }
    if changed:
        raise IntegrityFailure(f"calibration gate changed: {changed}")
    status = json.loads((CALIBRATION / "RUN_STATUS.json").read_text())
    if sha256_file(CALIBRATION / "MANIFEST.sha256") != status.get("manifest_sha256"):
        raise IntegrityFailure("calibration release manifest changed")
    for line in (CALIBRATION / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = CALIBRATION / relative
        hashes.append(calibration.require_committed(path, commit))
        if sha256_file(path) != expected:
            raise IntegrityFailure(f"calibration payload changed: {relative}")
    return snapshot, remote_heads, hashes


def load_panel(
    panel: str, commit: str, snapshot: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    release = EXPERIMENT / "pretruth_release" / panel
    pretruth_snapshot = json.loads((release / "PRETRUTH_GATE_SNAPSHOT.json").read_text())
    scoring_path = release / "tables/PRETRUTH_SCORING_INTERFACE.csv"
    prediction_path = release / "arrays/PRETRUTH_PREDICTIONS.npz"
    hashes = [
        calibration.require_committed(release / "PRETRUTH_GATE_SNAPSHOT.json", commit),
        calibration.require_committed(scoring_path, commit),
        calibration.require_committed(prediction_path, commit),
    ]
    isolated = DATA_ROOT / "isolated/E176" / panel
    f2, f3, f4 = (
        isolated / "F2_pretruth",
        isolated / "F3_calibration",
        isolated / "F4_evaluation",
    )
    f2_manifest = sha256_file(f2 / "MANIFEST.sha256")
    f3_manifest = sha256_file(f3 / "MANIFEST.sha256")
    f4_manifest = sha256_file(f4 / "MANIFEST.sha256")
    if snapshot["f2_manifest_sha256"].get(panel) != f2_manifest:
        raise IntegrityFailure(f"{panel} F2 differs from calibration gate")
    if snapshot["f3_calibration_manifest_sha256"].get(panel) != f3_manifest:
        raise IntegrityFailure(f"{panel} F3 differs from calibration gate")
    f4_att = json.loads((f4 / "ACCESS_ATTESTATION.json").read_text())
    required_f4 = {
        "experiment": f"E176_four_donor_fresh_confirmation::{panel}",
        "stage": f"E176_{panel}_F4_EVALUATION_TRUTH_BUILD",
        "status": "PASS",
        "gate_commit": commit,
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
    changed = {
        key: {"expected": value, "observed": f4_att.get(key)}
        for key, value in required_f4.items() if f4_att.get(key) != value
    }
    if changed:
        raise IntegrityFailure(f"{panel} F4 attestation failed: {changed}")
    if f4_att.get("builder_sha256") != sha256_file(TRUTH_BUILDER):
        raise IntegrityFailure(f"{panel} F4 builder changed")
    if pretruth_snapshot.get("source_full_sha256") != snapshot["source_full_sha256"]:
        raise IntegrityFailure(f"{panel} pretruth source differs from calibration gate")

    scoring = pd.read_csv(scoring_path, keep_default_na=False)
    locked = pd.read_csv(
        EXPERIMENT / f"manifests/{panel}/E176_{panel}_TASK_MANIFEST.csv",
        keep_default_na=False,
    )
    evaluation = locked.loc[strict_flag(locked.evaluation_test_task)].copy()
    ids = evaluation.task_id.astype(str).tolist()
    if len(ids) != 480:
        raise IntegrityFailure(f"{panel} evaluation task count changed")
    score_index = scoring.set_index("task_id")
    metadata = score_index.loc[ids].reset_index()
    if metadata.heldout_donor_partition.astype(str).ne("EVALUATION_80PCT").any():
        raise IntegrityFailure(f"{panel} calibration target entered final evaluation")
    with np.load(prediction_path, allow_pickle=False) as archive:
        row = {task: index for index, task in enumerate(scoring.task_id.astype(str))}
        predictions = {
            name: np.stack([np.asarray(archive[name][row[task]], dtype=float) for task in ids])
            for name in ("scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean")
        }
    truth = load_npz_vectors(f4 / "EVALUATION_TARGET_EFFECTS.npz", 480)
    if set(truth) != set(ids):
        raise IntegrityFailure(f"{panel} evaluation truth/query keys differ")
    truth_matrix = np.stack([truth[task] for task in ids])
    sc, ge, ensemble = (
        predictions["scGPT_seed_mean"],
        predictions["GEARS_seed_mean"],
        predictions["ensemble_seed_family_mean"],
    )
    disagreement = rmse_rows(sc, ge)
    if not np.allclose(
        disagreement, metadata.model_disagreement_rmse.to_numpy(float),
        atol=2e-7, rtol=2e-6,
    ):
        raise IntegrityFailure(f"{panel} frozen disagreement changed")
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
    for path in (f2 / "MANIFEST.sha256", f3 / "MANIFEST.sha256", f4 / "MANIFEST.sha256"):
        hashes.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return metrics, {"f4_manifest_sha256": f4_manifest}, hashes


def clopper_pearson(success: int, total: int) -> tuple[float, float]:
    lower = 0.0 if success == 0 else float(beta.ppf(0.025, success, total - success + 1))
    upper = 1.0 if success == total else float(beta.ppf(0.975, success + 1, total - success))
    return lower, upper


def coverage_table(metrics: pd.DataFrame, snapshot: dict[str, Any]) -> pd.DataFrame:
    rows = []
    populations = {
        "ALL_640": np.ones(len(metrics), dtype=bool),
        "SEEN_512": metrics.target_stratum.eq("DONOR_UNSEEN_ONLY").to_numpy(),
        "COLUMN_UNSEEN_128": metrics.target_stratum.eq("COLUMN_UNSEEN").to_numpy(),
    }
    populations.update({panel: metrics.panel_id.eq(panel).to_numpy() for panel in PANELS})
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
                lower_bound = (
                    block.pair_lower_bound_rmse.to_numpy(float)
                    if outcome == "pair_mean_rmse" else np.zeros(len(block))
                )
                rows.append({
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
                    "mean_interval_width_above_lower": float(
                        np.mean(block[upper_column].to_numpy(float) - lower_bound)
                    ),
                    "selected_primary_model": snapshot["selected_model_spec"][outcome] == spec,
                })
    return pd.DataFrame(rows)


def certificate_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    populations = {"ALL_640": metrics}
    populations.update({panel: metrics.loc[metrics.panel_id.eq(panel)] for panel in PANELS})
    for name, block in populations.items():
        ratio = block.pair_lower_bound_rmse.to_numpy(float) / block.pair_mean_rmse.to_numpy(float)
        rows.append({
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
            "median_bound_tightness_pair_mean": float(np.median(ratio)),
            "mean_bound_tightness_pair_mean": float(np.mean(ratio)),
            "median_pair_lower_bound_rmse": float(block.pair_lower_bound_rmse.median()),
            "p90_pair_lower_bound_rmse": float(block.pair_lower_bound_rmse.quantile(0.90)),
        })
    return pd.DataFrame(rows)


def ranking_tables(metrics: pd.DataFrame, postgate: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = {
        "fixed_safeconf": "safeconf_risk",
        "magnitude": "predicted_magnitude",
        "disagreement_pair_lower": "pair_lower_bound_rmse",
    }
    correlations, curves = [], []
    for panel in PANELS:
        for state in STATES:
            block = metrics.loc[
                metrics.panel_id.eq(panel) & metrics.culture_condition.eq(state)
            ]
            if len(block) != 160:
                raise IntegrityFailure(f"{panel}/{state} evaluation count changed")
            for outcome in ("ensemble_rmse", "pair_mean_rmse", "pair_max_rmse"):
                for score_name, score_column in scores.items():
                    rho = spearmanr(block[score_column], block[outcome]).statistic
                    correlations.append({
                        "panel_id": panel, "culture_condition": state,
                        "outcome": outcome, "score_name": score_name,
                        "n_tasks": len(block), "spearman": float(rho),
                    })
                    _, summary = postgate.tie_aware_curve(
                        block[score_column].to_numpy(float),
                        block[outcome].to_numpy(float),
                    )
                    curves.append({
                        "panel_id": panel, "culture_condition": state,
                        "outcome": outcome, "score_name": score_name,
                        "n_tasks": len(block), **summary,
                    })
    return pd.DataFrame(correlations), pd.DataFrame(curves)


def cluster_bootstrap(metrics: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    target = metrics.assign(target_key=target_key(metrics)).reset_index(drop=True)
    groups = []
    for _, block in target.groupby(["panel_id", "target_stratum"], sort=True):
        groups.append({
            key: np.asarray(indices, dtype=int)
            for key, indices in block.groupby("target_key").groups.items()
        })
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP_DRAWS, 4), dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen: list[int] = []
        for group in groups:
            names = np.asarray(list(group), dtype=object)
            sample = rng.choice(names, size=len(names), replace=True)
            chosen.extend(np.concatenate([group[name] for name in sample]).tolist())
        block = target.iloc[chosen]
        draws[draw, 0] = float(np.mean(
            block["upper_pair_mean_rmse__magnitude"]
            - block["upper_pair_mean_rmse__state_stratum_constant"]
        ))
        draws[draw, 1] = float(np.mean(
            block["upper_pair_mean_rmse__magnitude_plus_pair_lower"]
            - block["upper_pair_mean_rmse__magnitude"]
        ))
        pair_rho, mag_rho = [], []
        for panel in PANELS:
            for state in STATES:
                unit = block.loc[
                    block.panel_id.eq(panel) & block.culture_condition.eq(state)
                ]
                pair_rho.append(spearmanr(
                    unit.pair_lower_bound_rmse, unit.pair_mean_rmse
                ).statistic)
                mag_rho.append(spearmanr(
                    unit.predicted_magnitude, unit.pair_mean_rmse
                ).statistic)
        draws[draw, 2] = float(np.nanmean(pair_rho))
        draws[draw, 3] = float(np.nanmean(mag_rho))
    names = (
        "pair_upper_magnitude_minus_constant",
        "pair_upper_composite_minus_magnitude",
        "pair_lower_vs_pair_mean_macro_spearman",
        "magnitude_vs_pair_mean_macro_spearman",
    )
    summary = pd.DataFrame([{
        "estimand": name,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_mean": float(draws[:, index].mean()),
        "bootstrap_ci95_lower": float(np.quantile(draws[:, index], 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws[:, index], 0.975)),
    } for index, name in enumerate(names)])
    return summary, draws


def save_figures(
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    ranking: pd.DataFrame,
    snapshot: dict[str, Any],
) -> None:
    figures = STAGING / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    colors = {"ensemble_rmse": "#3B6FB6", "pair_mean_rmse": "#C65D3A"}

    selected = coverage.loc[
        coverage.selected_primary_model
        & coverage.population.isin(PANELS)
    ].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(PANELS))
    for offset, outcome in zip((-0.16, 0.16), ("ensemble_rmse", "pair_mean_rmse")):
        block = selected.loc[selected.outcome.eq(outcome)].set_index("population").loc[list(PANELS)]
        y = block.target_simultaneous_coverage.to_numpy(float)
        low = block.exact_binomial_ci95_lower.to_numpy(float)
        high = block.exact_binomial_ci95_upper.to_numpy(float)
        ax.errorbar(
            x + offset, y, yerr=np.vstack([y - low, high - y]), fmt="o",
            capsize=3, color=colors[outcome], label=outcome.replace("_", " "),
        )
    ax.axhline(0.90, color="#666666", linestyle="--", linewidth=1)
    ax.set_xticks(x, PANELS)
    ax.set_ylim(0.72, 1.01)
    ax.set_ylabel("Target-level simultaneous coverage")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "F1_donor_specific_coverage.png", dpi=240, facecolor="white")
    fig.savefig(figures / "F1_donor_specific_coverage.svg", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    sample = metrics.sample(min(1500, len(metrics)), random_state=20260719)
    ax.scatter(
        sample.pair_lower_bound_rmse, sample.pair_mean_rmse,
        s=8, alpha=0.35, color="#3B6FB6", linewidths=0,
    )
    maximum = float(max(sample.pair_mean_rmse.max(), sample.pair_lower_bound_rmse.max()))
    ax.plot([0, maximum], [0, maximum], color="#C65D3A", linewidth=1)
    ax.set_xlabel("Exact pair lower bound (RMSE)")
    ax.set_ylabel("Observed pair-mean RMSE")
    fig.tight_layout()
    fig.savefig(figures / "F2_pair_lower_certificate.png", dpi=240, facecolor="white")
    fig.savefig(figures / "F2_pair_lower_certificate.svg", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rules = snapshot["calibration_rules"]
    width = 0.35
    for index, outcome in enumerate(("ensemble_rmse", "pair_mean_rmse")):
        q = [
            float(rules[outcome][panel][snapshot["selected_model_spec"][outcome]]["quantile"])
            for panel in PANELS
        ]
        ax.bar(
            x + (index - 0.5) * width, q, width=width,
            color=colors[outcome], label=outcome.replace("_", " "),
        )
    ax.set_xticks(x, PANELS)
    ax.set_ylabel("Donor-specific conformal padding")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "F3_calibration_padding.png", dpi=240, facecolor="white")
    fig.savefig(figures / "F3_calibration_padding.svg", facecolor="white")
    plt.close(fig)

    macro = ranking.loc[ranking.outcome.eq("pair_mean_rmse")].groupby(
        "score_name", as_index=False
    ).spearman.mean().sort_values("spearman")
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.barh(macro.score_name, macro.spearman, color=["#888888", "#3B6FB6", "#C65D3A"])
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Macro Spearman with pair-mean RMSE")
    fig.tight_layout()
    fig.savefig(figures / "F4_ranking_diagnostic.png", dpi=240, facecolor="white")
    fig.savefig(figures / "F4_ranking_diagnostic.svg", facecolor="white")
    plt.close(fig)


def write_manifest() -> str:
    files = sorted(path for path in STAGING.rglob("*")
                   if path.is_file() and path.name not in {"MANIFEST.sha256", "RUN_STATUS.json"})
    payload = "".join(
        f"{sha256_file(path)}  {path.relative_to(STAGING).as_posix()}\n" for path in files
    )
    atomic_bytes(STAGING / "MANIFEST.sha256", payload.encode())
    return sha256_file(STAGING / "MANIFEST.sha256")


def run(
    truth_access_gate_commit: str,
    implementation_repair_commit: str,
    branch: str,
) -> dict[str, Any]:
    started = time.time()
    if OUT.exists() or STAGING.exists():
        raise IntegrityFailure("E176 final evaluation is append-only")
    if calibration.git(
        "merge-base", "--is-ancestor",
        truth_access_gate_commit, implementation_repair_commit,
        check=False,
    ).returncode:
        raise IntegrityFailure("post-truth implementation repair does not descend from truth gate")
    snapshot, remote_heads, code_hashes = verify_calibration_gate(
        implementation_repair_commit, branch
    )
    metrics_list, input_hashes, panel_meta = [], [], {}
    for panel in PANELS:
        frame, metadata, hashes = load_panel(panel, truth_access_gate_commit, snapshot)
        metrics_list.append(frame)
        panel_meta[panel] = metadata
        input_hashes.extend(hashes)
    metrics = add_pair_columns(pd.concat(metrics_list, ignore_index=True))
    if len(metrics) != 1920 or target_key(metrics).nunique() != 640:
        raise IntegrityFailure("final population must be 640 targets / 1,920 tasks")
    frozen = pd.read_csv(EXPERIMENT / "manifests/E176_ALL_SELECTED_TARGETS.csv")
    expected = set(frozen.loc[
        frozen.heldout_donor_partition.eq("EVALUATION_80PCT"), "ensembl_core"
    ].astype(str))
    if set(metrics.perturbed_gene_id.astype(str)) != expected:
        raise IntegrityFailure("final targets differ from frozen evaluation partition")

    for outcome in ("ensemble_rmse", "pair_mean_rmse"):
        for spec in MODEL_SPECS:
            model = snapshot["models"][outcome][spec]
            for panel in PANELS:
                mask = metrics.panel_id.eq(panel)
                rule = snapshot["calibration_rules"][outcome][panel][spec]
                base = predict_ridge(metrics.loc[mask], model)
                metrics.loc[mask, f"base_{outcome}__{spec}"] = base
                metrics.loc[mask, f"upper_{outcome}__{spec}"] = apply_cluster_upper(
                    metrics.loc[mask], base, rule
                )
    certificate = certificate_table(metrics)
    overall = certificate.loc[certificate.population.eq("ALL_640")].iloc[0]
    if (
        overall.pair_mean_bound_violations != 0
        or overall.pair_max_bound_violations != 0
        or overall.decomposition_max_abs_residual > 1e-8
    ):
        raise IntegrityFailure("final certificate numeric audit failed")
    coverage = coverage_table(metrics, snapshot)
    ranking, aurc = ranking_tables(metrics, import_postgate())
    bootstrap, draws = cluster_bootstrap(metrics)
    selected_overall = coverage.loc[
        coverage.population.eq("ALL_640") & coverage.selected_primary_model
    ].set_index("outcome")
    ensemble_cov = float(selected_overall.loc[
        "ensemble_rmse", "target_simultaneous_coverage"
    ])
    pair_cov = float(selected_overall.loc[
        "pair_mean_rmse", "target_simultaneous_coverage"
    ])
    empirical_at_nominal = ensemble_cov >= 0.90 and pair_cov >= 0.90
    decision = (
        "CERTIFICATE_AND_EMPIRICAL_COVERAGE_AUDIT_PASS"
        if empirical_at_nominal
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
    save_figures(metrics, coverage, ranking, snapshot)
    report = f"""# E176 final four-donor evaluation

正式状态：**{decision}**。主要评价只包含 640 个从未参与开发或校准的靶点、1,920 个任务；四位供体各贡献 160 个靶点。

模型对下界在 pair mean 与 pair max 上的违例均为 0；平方误差分解最大残差为 {overall.decomposition_max_abs_residual:.3g}。冻结 magnitude 基础模型加供体专属 conformal 校准后，三个状态同时覆盖的总体 target-level 经验覆盖率为：ensemble RMSE {ensemble_cov:.3f}，pair-mean RMSE {pair_cov:.3f}，目标值 0.90。逐供体 Clopper–Pearson 区间和上界宽度均保留在表中，没有根据评价真值调整分位数。

fixed SafeConf、magnitude 与 disagreement 的 Spearman/AURC 仅作诊断。E176 属于同一 Primary CD4 研究的多供体内部确认，不能替代独立研究、湿实验或临床验证。
"""
    atomic_bytes(STAGING / "reports/E176_FINAL_REPORT.md", report.encode())
    manifest_sha = write_manifest()
    status = {
        "schema": "safeconf_e176_final_evaluation_status_v1",
        "experiment": "E176_four_donor_fresh_confirmation",
        "stage": "F4_FINAL_HIDDEN_EVALUATION",
        "status": "COMPLETE",
        "decision": decision,
        "calibration_gate_commit": truth_access_gate_commit,
        "posttruth_implementation_repair_commit": implementation_repair_commit,
        "calibration_gate_remote_heads": remote_heads,
        "n_calibration_targets_excluded_from_primary": 160,
        "n_hidden_evaluation_targets": 640,
        "n_hidden_evaluation_tasks": 1920,
        "n_heldout_donors": 4,
        "pair_mean_bound_violations": 0,
        "pair_max_bound_violations": 0,
        "squared_error_decomposition_max_abs_residual": float(
            overall.decomposition_max_abs_residual
        ),
        "ensemble_rmse_target_simultaneous_coverage": ensemble_cov,
        "pair_mean_rmse_target_simultaneous_coverage": pair_cov,
        "empirical_coverage_at_or_above_nominal_for_both": empirical_at_nominal,
        "legacy_fixed_safeconf_increment_claim_supported": False,
        "evaluation_truth_used_to_select_model_or_recalibrate_quantile": False,
        "same_study_multi_donor_not_independent_study": True,
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
    parser.add_argument("--calibration-gate-commit", required=True)
    parser.add_argument("--implementation-repair-commit", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(
            args.calibration_gate_commit,
            args.implementation_repair_commit,
            args.branch,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
