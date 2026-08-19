#!/usr/bin/env python3
"""E167: retrospective development of a truth-free risk identifiability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E167_risk_identifiability_certificate_20260716"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
SOURCE_LOCK = OUT / "SOURCE_LOCK.csv"
STAGING = OUT / ".release.staging"
RELEASE = OUT / "release"

SEED = 202607167
N_BOOT = 2000
SCORE_STD_MIN = 1e-6
VECTOR_STD_MIN = 1e-6
VECTOR_QUANTIZATION_DECIMALS = 6
NEAR_REDUNDANCY_RHO = 0.98

E153_TASKS = ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables/E153_ABSOLUTE_TASK_INPUT.csv"
E96_TASKS = ROOT / "docs/实验结果/E96_prescribe_native_comparison_20260713/tables/E96_PRESCRIBE_TASKS.csv"
E96_PREDS = ROOT / "docs/实验结果/E96_prescribe_native_comparison_20260713/arrays/predicted_effects.npz"
E145_METRICS = ROOT / "docs/实验结果/E145_prescribe_paper_endpoint_20260714/tables/E145_TASK_METRICS.csv"
E159_JOINED = ROOT / "docs/实验结果/E159_prescribe_saturation_forensics_20260714/tables/E159_POSTHOC_JOINED_TASKS.csv"
E164_RISK = ROOT / "docs/实验结果/E164_wessels_pretruth_lock_20260715/release/tables/E164_RISK_WIDE.csv"
E165_METRICS = ROOT / "docs/实验结果/E165_wessels_truth_unseal_evaluation_20260715/release/tables/E165_PREDICTOR_TASK_METRICS.csv.gz"
E87_TASKS = ROOT / "docs/实验结果/E87_sciplex_to_openproblems_cpa_20260712/tables/E87_TASK_SCORES.csv"
E87_PREDS = ROOT / "docs/实验结果/E87_sciplex_to_openproblems_cpa_20260712/arrays/predicted_effects.npz"
E89_TASKS = ROOT / "docs/实验结果/E89_sciplex3_to_sciplex4_cpa_20260712/tables/E89_TASK_SCORES.csv"
E89_PREDS = ROOT / "docs/实验结果/E89_sciplex3_to_sciplex4_cpa_20260712/arrays/predicted_effects.npz"

E153_SOURCES = {
    "Frangieh": (
        ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/PREDICTION_RECORDS.csv",
        ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/arrays/predicted_effects.npz",
    ),
    "Lara_exvivo": (
        ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Lara_exvivo/PREDICTION_RECORDS.csv",
        ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Lara_exvivo/arrays/predicted_effects.npz",
    ),
    "Santinha": (
        ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Santinha/PREDICTION_RECORDS.csv",
        ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Santinha/arrays/predicted_effects.npz",
    ),
    "Shifrut": (
        ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut/PREDICTION_RECORDS.csv",
        ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut/arrays/predicted_effects.npz",
    ),
    "Liang": (
        ROOT / "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang/PREDICTION_RECORDS.csv",
        ROOT / "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang/arrays/predicted_effects.npz",
    ),
    "Tian_CRISPRi": (
        ROOT / "docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi/PREDICTION_RECORDS.csv",
        ROOT / "docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi/arrays/predicted_effects.npz",
    ),
    "Nadig_two_cellline": (
        ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline/PREDICTION_RECORDS.csv",
        ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline/arrays/predicted_effects.npz",
    ),
    "Replogle_two_cellline": (
        ROOT / "docs/实验结果/E151_replogle_formal_dual_models_20260714/Replogle_two_cellline/PREDICTION_RECORDS.csv",
        ROOT / "docs/实验结果/E151_replogle_formal_dual_models_20260714/Replogle_two_cellline/arrays/predicted_effects.npz",
    ),
}

WESSELLS_PROFILES = {
    seed: ROOT / f"docs/实验结果/E164_wessels_pretruth_lock_20260715/release/profiles/E164_PRESCRIBE_TEST_POST_PROFILES_SEED{seed}.csv.gz"
    for seed in (3407, 3408, 3409)
}

ALLOWLIST = {
    ".E167_TRANSACTION.json",
    "RUN_STATUS.json",
    "RESULTS_SHA256.csv",
    "README_先看这个.md",
    "reports/E167_REPORT.md",
    "tables/E167_INPUT_HASHES.csv",
    "tables/E167_REAL_UNIT_CERTIFICATES.csv",
    "tables/E167_SYNTHETIC_CONTROLS.csv",
    "tables/E167_TASK_LEVEL.csv",
    "tables/E167_TRUTH_ASSOCIATION_AUDIT.csv",
    "tables/E167_REPLICATE_STABILITY.csv",
    "tables/E167_GATE_SUMMARY.csv",
    "tables/E167_THEOREM_CHECKS.csv",
    "figures/F1_real_unit_gate_matrix.svg",
    "figures/F2_score_prediction_resolution.svg",
}


class IntegrityFailure(RuntimeError):
    pass


@dataclass
class Unit:
    unit_id: str
    study_id: str
    lane: str
    endpoint_id: str
    perturbation_family: str
    candidate_name: str
    role: str
    task_ids: np.ndarray
    clusters: np.ndarray
    score: np.ndarray
    magnitude: np.ndarray
    loss: np.ndarray
    predictors: dict[str, np.ndarray]
    predictor_viability_fraction: float = float("nan")
    synthetic_kind: str = "none"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def committed_and_matching(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"Required file is not committed: {relative}") from exc
    observed = sha256_file(path)
    if hashlib.sha256(committed).hexdigest() != observed:
        raise IntegrityFailure(f"Working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": observed}


def verify_inputs(head: str) -> list[dict[str, Any]]:
    own = [committed_and_matching(path, head) for path in (RUNNER, CONTRACT, SOURCE_LOCK)]
    lock = pd.read_csv(SOURCE_LOCK, dtype=str)
    if len(lock) != 30 or set(lock.columns) != {"path", "sha256"} or lock.path.duplicated().any():
        raise IntegrityFailure("SOURCE_LOCK schema/count/uniqueness failed")
    rows = []
    for row in lock.itertuples(index=False):
        path = ROOT / row.path
        if not path.is_file() or path.is_symlink():
            raise IntegrityFailure(f"Missing or symlinked frozen input: {row.path}")
        observed = sha256_file(path)
        if observed != row.sha256:
            raise IntegrityFailure(f"Frozen input changed: {row.path}")
        committed_and_matching(path, head)
        rows.append({"path": row.path, "bytes": path.stat().st_size, "sha256": observed})
    return own + rows


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def spearman(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 4 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    value = np.corrcoef(rankdata(a, method="average"), rankdata(b, method="average"))[0, 1]
    return float(value) if math.isfinite(value) else float("nan")


def required_resolution(n_tasks: int) -> int:
    return max(12, int(math.ceil(n_tasks / 2)))


def validate_unit(unit: Unit) -> None:
    n = len(unit.task_ids)
    arrays = [unit.clusters, unit.score, unit.magnitude, unit.loss]
    if n < 12 or any(len(value) != n for value in arrays):
        raise IntegrityFailure(f"Unit length mismatch: {unit.unit_id}")
    if len(set(unit.task_ids.astype(str))) != n:
        raise IntegrityFailure(f"Duplicate task IDs: {unit.unit_id}")
    if not np.isfinite(unit.score).all() or not np.isfinite(unit.magnitude).all():
        raise IntegrityFailure(f"Non-finite pretruth score: {unit.unit_id}")
    if not unit.predictors:
        raise IntegrityFailure(f"No predictor matrices: {unit.unit_id}")
    for name, matrix in unit.predictors.items():
        if matrix.ndim != 2 or matrix.shape[0] != n or matrix.shape[1] < 2:
            raise IntegrityFailure(f"Predictor shape mismatch: {unit.unit_id}/{name}/{matrix.shape}")


def prediction_stats(predictors: dict[str, np.ndarray]) -> dict[str, Any]:
    unique_counts = []
    max_coordinate_stds = []
    all_finite = True
    dimensions = []
    for matrix in predictors.values():
        values = np.asarray(matrix, float)
        all_finite &= bool(np.isfinite(values).all())
        dimensions.append(values.shape[1])
        quantized = np.round(values, decimals=VECTOR_QUANTIZATION_DECIMALS)
        quantized[np.abs(quantized) < 0.5 * 10 ** (-VECTOR_QUANTIZATION_DECIMALS)] = 0.0
        fingerprints = {hashlib.sha256(row.tobytes()).digest() for row in quantized}
        unique_counts.append(len(fingerprints))
        max_coordinate_stds.append(float(np.max(np.std(values, axis=0, ddof=0))))
    return {
        "prediction_all_finite": all_finite,
        "n_predictors": len(predictors),
        "vector_dimensions": ";".join(str(value) for value in dimensions),
        "min_quantized_unique_vectors": min(unique_counts),
        "min_predictor_max_coordinate_std": min(max_coordinate_stds),
    }


def weak_order_identical(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(rankdata(a, method="average"), rankdata(b, method="average")))


def selective_aurc(score: np.ndarray, loss: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    ordered_loss = np.asarray(loss, float)[order]
    coverages = np.linspace(0.2, 1.0, 17)
    risks = []
    for coverage in coverages:
        k = max(1, int(math.ceil(coverage * len(loss))))
        risks.append(float(np.mean(ordered_loss[:k])))
    return float(np.trapezoid(risks, coverages) / (coverages[-1] - coverages[0]))


def certificate_row(unit: Unit, replicate: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_unit(unit)
    n = len(unit.task_ids)
    required = required_resolution(n)
    unique_score = int(pd.Series(unit.score).nunique(dropna=False))
    score_std = float(np.std(unit.score, ddof=0))
    g2 = bool(np.isfinite(unit.score).all() and unique_score >= required and score_std > SCORE_STD_MIN)
    pred = prediction_stats(unit.predictors)
    g3 = bool(
        pred["prediction_all_finite"]
        and pred["min_quantized_unique_vectors"] >= required
        and pred["min_predictor_max_coordinate_std"] > VECTOR_STD_MIN
    )
    baseline_rho = spearman(unit.score, unit.magnitude)
    exact_equivalent = weak_order_identical(unit.score, unit.magnitude)
    near_redundant = bool(math.isfinite(baseline_rho) and abs(baseline_rho) >= NEAR_REDUNDANCY_RHO)
    g4_evaluated = replicate is not None
    g4_passed = bool(replicate["g4_passed"]) if replicate is not None else False
    reasons = []
    if not g2:
        reasons.append("G2_SCORE_SATURATION")
    if not g3:
        reasons.append("G3_PREDICTOR_COLLAPSE")
    if g4_evaluated and not g4_passed:
        reasons.append("G4_UNSTABLE")
    if exact_equivalent:
        reasons.append("G5_BASELINE_EQUIVALENT")
    if not g2:
        status = "ABSTAIN_SCORE_SATURATION"
    elif not g3:
        status = "ABSTAIN_PREDICTOR_COLLAPSE"
    elif g4_evaluated and not g4_passed:
        status = "ABSTAIN_UNSTABLE"
    elif exact_equivalent:
        status = "ASSOCIATION_ONLY_BASELINE_EQUIVALENT"
    else:
        status = "ELIGIBLE_G2_G3_ONLY"
    return {
        "unit_id": unit.unit_id,
        "study_id": unit.study_id,
        "lane": unit.lane,
        "endpoint_id": unit.endpoint_id,
        "perturbation_family": unit.perturbation_family,
        "candidate_name": unit.candidate_name,
        "role": unit.role,
        "synthetic_kind": unit.synthetic_kind,
        "n_tasks": n,
        "required_resolution": required,
        "score_all_finite": bool(np.isfinite(unit.score).all()),
        "score_unique": unique_score,
        "score_unique_fraction": unique_score / n,
        "score_std": score_std,
        "G2_score_estimability_passed": g2,
        **pred,
        "prediction_unique_fraction_min": pred["min_quantized_unique_vectors"] / n,
        "G3_prediction_task_conditioning_passed": g3,
        "G4_replicate_stability_evaluated": g4_evaluated,
        "G4_replicate_stability_passed": g4_passed if g4_evaluated else np.nan,
        "score_vs_magnitude_spearman": baseline_rho,
        "G5_exact_weak_order_equivalent": exact_equivalent,
        "G5_near_redundant_abs_rho_ge_0_98": near_redundant,
        "predictor_viability_fraction_vs_simple": unit.predictor_viability_fraction,
        "authorization_status": status,
        "reason_codes": ";".join(reasons) if reasons else "NONE",
        "target_truth_used_for_G0_to_G5": False,
    }


def load_matrix_from_keys(npz: np.lib.npyio.NpzFile, keys: list[str]) -> np.ndarray:
    missing = [key for key in keys if key not in npz.files]
    if missing:
        raise IntegrityFailure(f"Missing vector key: {missing[0]}")
    matrix = np.vstack([np.asarray(npz[key], dtype=np.float64) for key in keys])
    return matrix


def load_e153_units() -> list[Unit]:
    tasks = pd.read_csv(E153_TASKS)
    units = []
    for dataset, block in tasks.groupby("dataset", sort=True):
        block = block.reset_index(drop=True)
        records_path, arrays_path = E153_SOURCES[dataset]
        records = pd.read_csv(records_path)
        records["task_id"] = records.task_id.astype(str)
        block["task_id"] = block.task_id.astype(str)
        key_frame = block[["fold_id", "task_id"]].copy()
        selected = records.merge(key_frame, on=["fold_id", "task_id"], how="inner", validate="many_to_one")
        if len(selected) != 2 * len(block) or selected.predictor_name.nunique() != 2:
            raise IntegrityFailure(f"E153 vector join failed: {dataset}/{len(selected)}/{len(block)}")
        predictors = {}
        with np.load(arrays_path) as archive:
            for predictor, pred_rows in selected.groupby("predictor_name", sort=True):
                indexed = pred_rows.set_index(["fold_id", "task_id"])
                keys = [str(indexed.loc[(row.fold_id, row.task_id), "predicted_effect_key"]) for row in block.itertuples()]
                predictors[str(predictor)] = load_matrix_from_keys(archive, keys)
        units.append(Unit(
            unit_id=f"E153::{dataset}",
            study_id=dataset,
            lane="STRUCTURAL_RISK",
            endpoint_id="two_predictor_mean_absolute_rmse",
            perturbation_family="genetic",
            candidate_name="safeconf_calibrated_pair_risk",
            role="real_noncollapsed_reference",
            task_ids=(block.fold_id.astype(str) + "::" + block.task_id).to_numpy(),
            clusters=block.perturbation.astype(str).to_numpy(),
            score=block.safeconf_calibrated_pair_risk.to_numpy(float),
            magnitude=block.baseline_predicted_magnitude.to_numpy(float),
            loss=block.error_two_predictor_mean_rmse.to_numpy(float),
            predictors=predictors,
        ))
    return units


def load_e96_units() -> list[Unit]:
    scores = pd.read_csv(E96_TASKS)
    metrics = pd.read_csv(E145_METRICS)
    joined = scores.merge(
        metrics[["panel", "task_id", "pearson_effect_accuracy"]],
        on=["panel", "task_id"], how="inner", validate="one_to_one",
    )
    if len(joined) != 48:
        raise IntegrityFailure("E96/E145 join failed")
    units = []
    with np.load(E96_PREDS) as archive:
        for panel, block in joined.groupby("panel", sort=True):
            block = block.reset_index(drop=True)
            keys = [f"E95::{panel}::{task_id}::pred" for task_id in block.task_id]
            matrix = load_matrix_from_keys(archive, keys)
            units.append(Unit(
                unit_id=f"E96::{panel}", study_id=panel, lane="MODEL_UQ",
                endpoint_id="prescribe_paper_pearson_accuracy_loss", perturbation_family="genetic",
                candidate_name="prescribe_official_combined_risk", role="real_noncollapsed_reference",
                task_ids=block.task_id.astype(str).to_numpy(), clusters=block.task_id.astype(str).to_numpy(),
                score=block.risk_combined.to_numpy(float), magnitude=-block.magnitude_pred_rms.to_numpy(float),
                loss=-block.pearson_effect_accuracy.to_numpy(float), predictors={"PRESCRIBE": matrix},
            ))
    return units


def load_e159_units() -> list[Unit]:
    joined = pd.read_csv(E159_JOINED)
    pca_columns = [f"predicted_pca_{index}" for index in range(10)]
    units = []
    for panel, block in joined.groupby("panel", sort=True):
        block = block.reset_index(drop=True)
        predictor = {"PRESCRIBE": block[pca_columns].to_numpy(float)}
        common = dict(
            study_id=panel, lane="MODEL_UQ", endpoint_id="prescribe_paper_pearson_accuracy_loss",
            perturbation_family="genetic", role="real_collapse_positive",
            task_ids=block.task_id.astype(str).to_numpy(), clusters=block.task_id.astype(str).to_numpy(),
            magnitude=-block.predicted_magnitude_rms.to_numpy(float),
            loss=-block.pearson_effect_accuracy.to_numpy(float), predictors=predictor,
        )
        units.append(Unit(
            unit_id=f"E159::{panel}::official", candidate_name="prescribe_official_combined_risk",
            score=-block.combined_confidence_official.to_numpy(float), **common,
        ))
        units.append(Unit(
            unit_id=f"E159::{panel}::raw_log_prob", candidate_name="prescribe_raw_log_prob_risk",
            score=-block.log_prob.to_numpy(float), **common,
        ))
    return units


def load_wessels_units() -> tuple[list[Unit], pd.DataFrame]:
    risk = pd.read_csv(E164_RISK)
    metrics = pd.read_csv(E165_METRICS)
    score_matrix = []
    units = []
    for seed in (3407, 3408, 3409):
        score_column = f"prescribe_raw_log_prob_seed{seed}_confidence"
        confidence = risk[score_column].to_numpy(float)
        score_matrix.append(-confidence)
        profile = pd.read_csv(WESSELLS_PROFILES[seed])
        profile = profile.set_index("condition").loc[risk.condition.astype(str)]
        gene_columns = [column for column in profile.columns if column != "seed"]
        matrix = profile[gene_columns].to_numpy(float)
        metric_block = metrics.loc[metrics.predictor.eq(f"prescribe_seed{seed}")].set_index("condition").loc[risk.condition.astype(str)]
        if len(metric_block) != 48:
            raise IntegrityFailure(f"Wessels metrics join failed: {seed}")
        prescribe_rmse = metric_block.pca10_rmse.to_numpy(float)
        control_rmse = metrics.loc[metrics.predictor.eq("control_no_change")].set_index("condition").loc[risk.condition.astype(str), "pca10_rmse"].to_numpy(float)
        viability = float(np.mean(prescribe_rmse < control_rmse))
        units.append(Unit(
            unit_id=f"E165::Wessels::PRESCRIBE_seed{seed}", study_id="Wessels", lane="MODEL_UQ",
            endpoint_id="pca10_pearson_accuracy_loss", perturbation_family="genetic_combinatorial",
            candidate_name="prescribe_raw_log_prob_risk", role="real_collapse_positive",
            task_ids=risk.condition.astype(str).to_numpy(), clusters=risk.condition.astype(str).to_numpy(),
            score=-confidence, magnitude=risk.prescribe_magnitude_raw_seed3407.to_numpy(float),
            loss=-metric_block.pca10_pearson.to_numpy(float), predictors={"PRESCRIBE": matrix},
            predictor_viability_fraction=viability,
        ))
    score_matrix_array = np.column_stack(score_matrix)
    pair_rows = []
    for left in range(3):
        for right in range(left + 1, 3):
            pair_rows.append({"seed_left": (3407, 3408, 3409)[left], "seed_right": (3407, 3408, 3409)[right], "spearman": spearman(score_matrix_array[:, left], score_matrix_array[:, right])})
    pairwise_median = float(np.median([row["spearman"] for row in pair_rows]))
    ranks = np.column_stack([rankdata(score_matrix_array[:, index], method="average") for index in range(3)])
    rank_sums = ranks.sum(axis=1)
    n, k = ranks.shape
    kendall_w = float(12 * np.sum((rank_sums - rank_sums.mean()) ** 2) / (k * k * (n**3 - n)))
    rng = np.random.default_rng(SEED + 165)
    draws = []
    for _ in range(N_BOOT):
        take = rng.integers(0, n, size=n)
        values = [spearman(score_matrix_array[take, left], score_matrix_array[take, right]) for left in range(3) for right in range(left + 1, 3)]
        finite = [value for value in values if math.isfinite(value)]
        draws.append(float(np.median(finite)) if finite else np.nan)
    finite_draws = np.asarray(draws, float)
    finite_draws = finite_draws[np.isfinite(finite_draws)]
    low, high = np.quantile(finite_draws, [0.025, 0.975])
    summary = {
        "replicate_group": "Wessels_PRESCRIBE_raw_log_prob",
        "n_tasks": n, "n_replicates": k, "kendall_w": kendall_w,
        "median_pairwise_spearman": pairwise_median,
        "bootstrap_ci95_low": float(low), "bootstrap_ci95_high": float(high),
        "g4_passed": bool(pairwise_median >= 0.5 and low > 0),
    }
    pair_table = pd.DataFrame(pair_rows)
    for key, value in summary.items():
        pair_table[key] = value
    return units, pair_table


def load_chemical_units() -> list[Unit]:
    units = []
    e87 = pd.read_csv(E87_TASKS)
    with np.load(E87_PREDS) as archive:
        predictors = {
            "CPA": load_matrix_from_keys(archive, [f"E87::{key}::CPA_0.8.8_RDKIT_cross_dataset::pred" for key in e87.task_key]),
            "ridge": load_matrix_from_keys(archive, [f"E87::{key}::inductive_ridge_cross_dataset_v1::pred" for key in e87.task_key]),
        }
    units.append(Unit(
        unit_id="E87::sciPlex3_to_OpenProblems", study_id="OpenProblems2023", lane="MODEL_UQ",
        endpoint_id="two_predictor_mean_absolute_rmse", perturbation_family="chemical",
        candidate_name="model_disagreement_rmse", role="real_noncollapsed_reference",
        task_ids=e87.task_key.astype(str).to_numpy(), clusters=e87.drug.astype(str).to_numpy(),
        score=e87.model_disagreement_rmse.to_numpy(float), magnitude=e87.predicted_magnitude_mean.to_numpy(float),
        loss=e87.pair_mean_rmse.to_numpy(float), predictors=predictors,
        predictor_viability_fraction=float(np.mean(np.minimum(e87.error_cpa_rmse, e87.error_ridge_rmse) < e87.zero_effect_rmse)),
    ))
    e89 = pd.read_csv(E89_TASKS)
    with np.load(E89_PREDS) as archive:
        predictors = {
            "CPA": load_matrix_from_keys(archive, [f"E89::{key}::CPA_0.8.8_RDKIT_sciPlex3::pred" for key in e89.task_key]),
            "interpolation": load_matrix_from_keys(archive, [f"E89::{key}::source_dose_interpolation_v1::pred" for key in e89.task_key]),
        }
    units.append(Unit(
        unit_id="E89::sciPlex3_to_sciPlex4", study_id="sciPlex4", lane="MODEL_UQ",
        endpoint_id="two_predictor_mean_absolute_rmse", perturbation_family="chemical",
        candidate_name="model_disagreement_rmse", role="real_noncollapsed_reference",
        task_ids=e89.task_key.astype(str).to_numpy(), clusters=e89.drug.astype(str).to_numpy(),
        score=e89.model_disagreement_rmse.to_numpy(float), magnitude=e89.predicted_magnitude_mean.to_numpy(float),
        loss=e89.pair_mean_rmse.to_numpy(float), predictors=predictors,
        predictor_viability_fraction=float(np.mean(np.minimum(e89.error_cpa_rmse, e89.error_interpolation_rmse) < e89.zero_effect_rmse)),
    ))
    return units


def make_synthetic_controls(base_units: list[Unit]) -> list[Unit]:
    controls = []
    for unit in base_units:
        center = float(np.median(unit.score))
        n = len(unit.score)
        controls.append(replace(
            unit, unit_id=f"SYN::{unit.study_id}::constant_score", role="synthetic_negative",
            candidate_name="synthetic_exact_constant", score=np.full(n, center), synthetic_kind="exact_constant",
        ))
        jitter = (np.arange(n, dtype=float) - (n - 1) / 2) * 1e-12
        controls.append(replace(
            unit, unit_id=f"SYN::{unit.study_id}::epsilon_jitter", role="synthetic_negative",
            candidate_name="synthetic_machine_epsilon_jitter", score=center + jitter, synthetic_kind="epsilon_jitter",
        ))
        collapsed = {name: np.repeat(matrix[:1], n, axis=0) for name, matrix in unit.predictors.items()}
        controls.append(replace(
            unit, unit_id=f"SYN::{unit.study_id}::prediction_collapse", role="synthetic_negative",
            candidate_name="synthetic_score_with_collapsed_prediction", predictors=collapsed, synthetic_kind="prediction_collapse",
        ))
        controls.append(replace(
            unit, unit_id=f"SYN::{unit.study_id}::magnitude_clone", role="synthetic_baseline_clone",
            candidate_name="synthetic_magnitude_clone", score=unit.magnitude.copy(), synthetic_kind="magnitude_clone",
        ))
    return controls


def bootstrap_association(unit: Unit) -> dict[str, Any]:
    observed_candidate = spearman(unit.score, unit.loss)
    observed_magnitude = spearman(unit.magnitude, unit.loss)
    observed_delta = observed_candidate - observed_magnitude if math.isfinite(observed_candidate) and math.isfinite(observed_magnitude) else np.nan
    cluster_values = np.asarray(sorted(set(unit.clusters.astype(str))))
    index = {cluster: np.flatnonzero(unit.clusters.astype(str) == cluster) for cluster in cluster_values}
    rng_seed = int(hashlib.sha256(f"{SEED}|{unit.unit_id}".encode()).hexdigest()[:16], 16) % (2**32 - 1)
    rng = np.random.default_rng(rng_seed)
    candidate_draws, delta_draws = [], []
    for _ in range(N_BOOT):
        selected = rng.choice(cluster_values, size=len(cluster_values), replace=True)
        take = np.concatenate([index[value] for value in selected])
        candidate = spearman(unit.score[take], unit.loss[take])
        magnitude = spearman(unit.magnitude[take], unit.loss[take])
        candidate_draws.append(candidate)
        delta_draws.append(candidate - magnitude if math.isfinite(candidate) and math.isfinite(magnitude) else np.nan)

    def interval(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, float)
        array = array[np.isfinite(array)]
        if not len(array):
            return float("nan"), float("nan")
        low, high = np.quantile(array, [0.025, 0.975])
        return float(low), float(high)

    candidate_low, candidate_high = interval(candidate_draws)
    delta_low, delta_high = interval(delta_draws)
    return {
        "unit_id": unit.unit_id,
        "candidate_vs_loss_spearman": observed_candidate,
        "candidate_ci95_low": candidate_low,
        "candidate_ci95_high": candidate_high,
        "magnitude_vs_loss_spearman": observed_magnitude,
        "delta_candidate_minus_magnitude": observed_delta,
        "delta_ci95_low": delta_low,
        "delta_ci95_high": delta_high,
        "candidate_aurc": selective_aurc(unit.score, unit.loss),
        "magnitude_aurc": selective_aurc(unit.magnitude, unit.loss),
        "truth_used_for_gate_override": False,
    }


def task_table(units: list[Unit]) -> pd.DataFrame:
    rows = []
    for unit in units:
        for index in range(len(unit.task_ids)):
            rows.append({
                "unit_id": unit.unit_id, "study_id": unit.study_id, "role": unit.role,
                "task_id": unit.task_ids[index], "cluster": unit.clusters[index],
                "candidate_score_pretruth": unit.score[index],
                "magnitude_score_pretruth": unit.magnitude[index],
                "registered_truth_loss_postgate": unit.loss[index],
                "truth_used_for_G0_to_G5": False,
            })
    return pd.DataFrame(rows)


def make_gate_matrix(real: pd.DataFrame, path: Path) -> None:
    frame = real.copy().sort_values(["role", "study_id", "candidate_name"]).reset_index(drop=True)
    matrix = np.full((len(frame), 4), np.nan)
    matrix[:, 0] = frame.G2_score_estimability_passed.astype(float)
    matrix[:, 1] = frame.G3_prediction_task_conditioning_passed.astype(float)
    matrix[:, 2] = np.where(frame.G4_replicate_stability_evaluated, frame.G4_replicate_stability_passed.astype(float), np.nan)
    matrix[:, 3] = (~frame.G5_exact_weak_order_equivalent).astype(float)
    display = np.nan_to_num(matrix, nan=-1)
    cmap = ListedColormap(["#D9D9D9", "#D55E00", "#4C956C"])
    fig, ax = plt.subplots(figsize=(7.8, max(5.2, 0.30 * len(frame) + 1.8)), facecolor="white")
    ax.set_facecolor("white")
    ax.imshow(display + 1, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    labels = frame.unit_id.str.replace("E153::", "", regex=False).str.replace("E165::Wessels::", "Wessels::", regex=False)
    ax.set_yticks(np.arange(len(frame)), labels, fontsize=8)
    ax.set_xticks(np.arange(4), ["G2 score", "G3 prediction", "G4 replicate", "G5 non-clone"], fontsize=9)
    for row in range(len(frame)):
        for column in range(4):
            value = matrix[row, column]
            symbol = "·" if np.isnan(value) else ("✓" if value == 1 else "×")
            ax.text(column, row, symbol, ha="center", va="center", color="#202020", fontsize=10, fontweight="bold")
    ax.set_title("RIAG v1: truth-free gate outcomes on real historical units", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_resolution_scatter(real: pd.DataFrame, path: Path) -> None:
    colors = real.role.map({"real_noncollapsed_reference": "#4C78A8", "real_collapse_positive": "#D55E00"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(7.6, 5.2), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(real.score_unique_fraction, real.prediction_unique_fraction_min, c=colors, s=48, edgecolor="white", linewidth=0.7)
    for row in real.itertuples(index=False):
        if row.role == "real_collapse_positive" or row.unit_id.startswith("E87"):
            label = row.unit_id.replace("E159::", "").replace("E165::", "")
            ax.annotate(label, (row.score_unique_fraction, row.prediction_unique_fraction_min), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(-0.03, 1.04); ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("Candidate score unique fraction")
    ax.set_ylabel("Minimum prediction-vector unique fraction")
    ax.set_title("Score variation cannot rescue collapsed predictions", loc="left", fontweight="bold")
    ax.grid(color="#E8E8E8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("NA" if not math.isfinite(value) else f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def analyze() -> dict[str, Any]:
    e153_units = load_e153_units()
    wessels_units, replicate_table = load_wessels_units()
    real_units = e153_units + load_e96_units() + load_e159_units() + wessels_units + load_chemical_units()
    if len(real_units) != 19:
        raise IntegrityFailure(f"Expected 19 real units, observed {len(real_units)}")
    replicate_summary = replicate_table.iloc[0].to_dict()
    real_rows = []
    for unit in real_units:
        replicate = replicate_summary if unit.study_id == "Wessels" else None
        real_rows.append(certificate_row(unit, replicate))
    real = pd.DataFrame(real_rows)

    synthetic_units = make_synthetic_controls(e153_units)
    synthetic_rows = []
    for unit in synthetic_units:
        row = certificate_row(unit)
        row["aurc_delta_vs_magnitude"] = selective_aurc(unit.score, unit.loss) - selective_aurc(unit.magnitude, unit.loss)
        synthetic_rows.append(row)
    synthetic = pd.DataFrame(synthetic_rows)

    truth = pd.DataFrame([bootstrap_association(unit) for unit in real_units])
    real = real.merge(truth, on="unit_id", how="left", validate="one_to_one")

    real_collapsed = real.role.eq("real_collapse_positive")
    real_reference = real.role.eq("real_noncollapsed_reference")
    synth_constant = synthetic.synthetic_kind.isin(["exact_constant", "epsilon_jitter"])
    synth_prediction = synthetic.synthetic_kind.eq("prediction_collapse")
    synth_clone = synthetic.synthetic_kind.eq("magnitude_clone")
    checks = {
        "real_collapse_detection": bool((~(real.loc[real_collapsed, "G2_score_estimability_passed"] & real.loc[real_collapsed, "G3_prediction_task_conditioning_passed"])).all()),
        "real_noncollapsed_reference_retention": bool((real.loc[real_reference, "G2_score_estimability_passed"] & real.loc[real_reference, "G3_prediction_task_conditioning_passed"]).all()),
        "synthetic_constant_and_jitter_detection": bool((~synthetic.loc[synth_constant, "G2_score_estimability_passed"]).all()),
        "synthetic_prediction_collapse_detection": bool((~synthetic.loc[synth_prediction, "G3_prediction_task_conditioning_passed"]).all()),
        "synthetic_magnitude_clone_equivalence": bool(synthetic.loc[synth_clone, "G5_exact_weak_order_equivalent"].all()),
        "synthetic_magnitude_clone_aurc_identity": bool(np.allclose(synthetic.loc[synth_clone, "aurc_delta_vs_magnitude"], 0, atol=1e-15, rtol=0)),
        "wessels_replicate_instability_detected": bool(not replicate_summary["g4_passed"]),
        "no_truth_gate_override": bool(real.target_truth_used_for_G0_to_G5.eq(False).all()),
    }
    strict_pass = bool(all(checks.values()))

    summary_rows = []
    for name, passed in checks.items():
        summary_rows.append({"check": name, "passed": passed})
    gate_summary = pd.DataFrame(summary_rows)
    theorem_checks = pd.DataFrame([
        {"theorem": "P1_constant_rank_not_estimable", "empirical_check": "constant+jitter G2 rejected", "passed": checks["synthetic_constant_and_jitter_detection"]},
        {"theorem": "P2_output_collapse_blocks_model_UQ", "empirical_check": "real+synthetic collapsed predictions rejected", "passed": checks["real_collapse_detection"] and checks["synthetic_prediction_collapse_detection"]},
        {"theorem": "P3_same_weak_order_same_AURC", "empirical_check": "magnitude clones exact zero delta AURC", "passed": checks["synthetic_magnitude_clone_aurc_identity"]},
        {"theorem": "P4_low_replicate_reliability_blocks_authorization", "empirical_check": "Wessels raw score G4 rejected", "passed": checks["wessels_replicate_instability_detected"]},
        {"theorem": "necessary_not_sufficient", "empirical_check": "E87 passes G2/G3 while predictor viability is zero", "passed": bool(real.loc[real.unit_id.eq("E87::sciPlex3_to_OpenProblems"), "predictor_viability_fraction_vs_simple"].iloc[0] == 0)},
    ])
    return {
        "real_units": real_units, "real": real, "synthetic": synthetic,
        "truth": truth, "replicate": replicate_table, "gate_summary": gate_summary,
        "theorem_checks": theorem_checks, "strict_pass": strict_pass,
    }


def write_release(result: dict[str, Any], input_hashes: list[dict[str, Any]], head: str) -> dict[str, Any]:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E167 release is append-only and already exists")
    (STAGING / "reports").mkdir(parents=True)
    (STAGING / "tables").mkdir(); (STAGING / "figures").mkdir()
    transaction = {"schema": "safeconf_e167_transaction_v1", "transaction_id": uuid.uuid4().hex, "created_at": now()}
    atomic_json(STAGING / ".E167_TRANSACTION.json", transaction)
    atomic_csv(STAGING / "tables/E167_INPUT_HASHES.csv", pd.DataFrame(input_hashes))
    atomic_csv(STAGING / "tables/E167_REAL_UNIT_CERTIFICATES.csv", result["real"])
    atomic_csv(STAGING / "tables/E167_SYNTHETIC_CONTROLS.csv", result["synthetic"])
    atomic_csv(STAGING / "tables/E167_TASK_LEVEL.csv", task_table(result["real_units"]))
    atomic_csv(STAGING / "tables/E167_TRUTH_ASSOCIATION_AUDIT.csv", result["truth"])
    atomic_csv(STAGING / "tables/E167_REPLICATE_STABILITY.csv", result["replicate"])
    atomic_csv(STAGING / "tables/E167_GATE_SUMMARY.csv", result["gate_summary"])
    atomic_csv(STAGING / "tables/E167_THEOREM_CHECKS.csv", result["theorem_checks"])
    make_gate_matrix(result["real"], STAGING / "figures/F1_real_unit_gate_matrix.svg")
    make_resolution_scatter(result["real"], STAGING / "figures/F2_score_prediction_resolution.svg")

    real = result["real"]
    collapse_view = real.loc[real.role.eq("real_collapse_positive"), [
        "unit_id", "score_unique", "min_quantized_unique_vectors", "authorization_status",
        "candidate_vs_loss_spearman",
    ]]
    reference_view = real.loc[real.role.eq("real_noncollapsed_reference"), [
        "unit_id", "score_unique_fraction", "prediction_unique_fraction_min",
        "score_vs_magnitude_spearman", "authorization_status",
    ]]
    wessels = result["replicate"].iloc[0]
    e87 = real.loc[real.unit_id.eq("E87::sciPlex3_to_OpenProblems")].iloc[0]
    report = (
        "# E167｜风险可识别性与适用性证书（RIAG v1）\n\n"
        f"预设开发 gate：`{'PASS' if result['strict_pass'] else 'FAIL'}`。"
        "本结果只证明证书能识别不可估计与基线等价情形，不证明通过后的分数一定可靠。\n\n"
        "## 真实塌缩案例\n\n" + markdown_table(collapse_view) + "\n\n"
        f"Wessels raw score 三 seed 的两两 Spearman 中位数为 `{wessels.median_pairwise_spearman:.4f}`，"
        f"bootstrap 95% CI `[{wessels.bootstrap_ci95_low:.4f}, {wessels.bootstrap_ci95_high:.4f}]`，"
        f"Kendall W=`{wessels.kendall_w:.4f}`；G4=`{'PASS' if wessels.g4_passed else 'FAIL'}`。\n\n"
        "## 非塌缩参考\n\n" + markdown_table(reference_view) + "\n\n"
        "## 必要条件不等于准确率保证\n\n"
        f"E87 的 score 和预测向量通过 G2/G3，但任一预测器优于 no-change 的任务比例为 "
        f"`{e87.predictor_viability_fraction_vs_simple:.4f}`。因此 RIAG 通过只授权评价流程继续，不能替代上游预测器有效性检查。\n\n"
        "## 解释边界\n\n"
        "Norman、Wessels 和其余历史资产真值均已解封，E167 属于方法开发。"
        "下一项确认必须在新数据 test truth 前生成证书；G2/G3/G4 失败后，任何测试相关性都不得覆盖拒绝状态。"
        "结构风险可以在 predictor collapse 后单独研究，但不能再称为模型内生 uncertainty。\n"
    )
    atomic_bytes(STAGING / "reports/E167_REPORT.md", report.encode())
    atomic_bytes(STAGING / "README_先看这个.md", b"# E167\n\nRead `reports/E167_REPORT.md` first.\n")

    manifest_rows = []
    for path in sorted(STAGING.rglob("*")):
        if path.is_symlink():
            raise IntegrityFailure("Symlink found in E167 staging")
        if path.is_file() and path.name not in {"RUN_STATUS.json", "RESULTS_SHA256.csv"}:
            manifest_rows.append({"relative_path": path.relative_to(STAGING).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_csv(STAGING / "RESULTS_SHA256.csv", pd.DataFrame(manifest_rows))
    status = {
        "schema": "safeconf_e167_riag_v1",
        "phase": "complete_retrospective_development_not_independent_confirmation",
        "completed_at": now(), "git_head_at_formal_run": head,
        "transaction_id": transaction["transaction_id"],
        "raw_expression_files_opened": 0, "new_candidate_truth_opened": False,
        "n_real_units": len(result["real"]), "n_synthetic_units": len(result["synthetic"]),
        "n_source_files": 30, "strict_development_gate_passed": result["strict_pass"],
        "all_gate_checks": {row.check: bool(row.passed) for row in result["gate_summary"].itertuples()},
        "results_manifest_sha256": sha256_file(STAGING / "RESULTS_SHA256.csv"),
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    observed = {path.relative_to(STAGING).as_posix() for path in STAGING.rglob("*") if path.is_file()}
    if observed != ALLOWLIST:
        raise IntegrityFailure(f"E167 allowlist mismatch: {sorted(observed ^ ALLOWLIST)}")
    os.replace(STAGING, RELEASE)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    head = git_head()
    hashes = verify_inputs(head)
    if args.mode == "preflight":
        print(json.dumps({"phase": "preflight_passed", "git_head": head, "n_locked_sources": 30, "raw_expression_files_opened": 0}, ensure_ascii=False, indent=2))
        return
    result = analyze()
    status = write_release(result, hashes, head)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
