#!/usr/bin/env python3
"""E167a: batch-aware, tie-aware correction of the RIAG development protocol."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E167a_riag_resolution_correction_20260716"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
SOURCE_LOCK = OUT / "SOURCE_LOCK.csv"
STAGING = OUT / ".release.staging"
RELEASE = OUT / "release"

SCORE_TOL = 1e-6
MAGNITUDE_TOL = 1e-6
PREDICTION_TOL = 1e-6
TARGET_COVERAGE = 0.20
RC_COVERAGES = np.linspace(0.20, 1.00, 17)
NONTRIVIAL_COVERAGES = RC_COVERAGES[:-1]
SEED = 2026071671

OLD_STATUS = ROOT / "docs/实验结果/E167_risk_identifiability_certificate_20260716/release/RUN_STATUS.json"
OLD_MANIFEST = ROOT / "docs/实验结果/E167_risk_identifiability_certificate_20260716/release/RESULTS_SHA256.csv"
ASSETS = OUT / "input_assets"
ASSET_ATTESTATION = ASSETS / "E167A_ASSET_BUILD_ATTESTATION.json"
UNIT_REGISTRY = ASSETS / "E167A_UNIT_REGISTRY.csv"
PRETRUTH_TASKS = ASSETS / "E167A_PRETRUTH_TASKS.csv"
PRETRUTH_PREDICTIONS = ASSETS / "E167A_PRETRUTH_PREDICTIONS.npz"
PREDICTION_TASK_MAP = ASSETS / "E167A_PREDICTION_TASK_MAP.csv"
POSTGATE_TRUTH = ASSETS / "E167A_POSTGATE_TRUTH.csv"
POSTGATE_VIABILITY = ASSETS / "E167A_POSTGATE_VIABILITY.csv"

ALLOWLIST = {
    ".E167A_TRANSACTION.json",
    "README_先看这个.md",
    "RESULTS_SHA256.csv",
    "RUN_STATUS.json",
    "PRETRUTH_GATE_SNAPSHOT.json",
    "reports/E167A_REPORT.md",
    "tables/E167A_INPUT_HASHES.csv",
    "tables/E167A_PRETRUTH_INPUT_HASHES.csv",
    "tables/E167A_PRETRUTH_GATE_SUMMARY.csv",
    "tables/E167A_BATCH_CERTIFICATES.csv",
    "tables/E167A_PREDICTOR_CERTIFICATES.csv",
    "tables/E167A_COVERAGE_BOUNDARIES.csv",
    "tables/E167A_TIE_AWARE_RC.csv",
    "tables/E167A_TIE_AWARE_AURC.csv",
    "tables/E167A_UNIT_SUMMARY.csv",
    "tables/E167A_POSTGATE_DIAGNOSTICS.csv",
    "tables/E167A_SYNTHETIC_TESTS.csv",
    "tables/E167A_GATE_SUMMARY.csv",
    "figures/F1_batch_score_resolution.svg",
    "figures/F2_tie_aware_aurc_intervals.svg",
    "figures/F3_riag_v2_decision_flow.svg",
}

BLUE = "#3B6FB6"
ORANGE = "#D97732"
GREEN = "#4F8A70"
GREY = "#A7A9AC"
DARK = "#252525"
GRID = "#E6E6E6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "svg.fonttype": "none",
})


class IntegrityFailure(RuntimeError):
    pass


@dataclass
class PretruthUnit:
    unit_id: str
    study_id: str
    lane: str
    endpoint_id: str
    perturbation_family: str
    candidate_name: str
    role: str
    score_transform: str
    score_numeric_unit: str
    score_registered_precision: str
    magnitude_transform: str
    magnitude_numeric_unit: str
    magnitude_registered_precision: str
    prediction_numeric_unit: str
    prediction_registered_precision: str
    task_ids: np.ndarray
    clusters: np.ndarray
    score: np.ndarray
    magnitude: np.ndarray
    predictors: dict[str, np.ndarray]


EXPECTED_UNIT_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    **{
        f"E153::{study}": (
            "STRUCTURAL_RISK",
            "real_noncollapsed_reference",
            ("GEARS_context_mean_trainonly_graphs", "scGPT_context_mean_finetuned"),
        )
        for study in (
            "Frangieh", "Lara_exvivo", "Liang", "Nadig_two_cellline",
            "Replogle_two_cellline", "Santinha", "Shifrut", "Tian_CRISPRi",
        )
    },
    "E96::Norman_P1": ("MODEL_UQ", "real_noncollapsed_reference", ("PRESCRIBE",)),
    "E96::Norman_P2": ("MODEL_UQ", "real_noncollapsed_reference", ("PRESCRIBE",)),
    "E159::Norman_P3::official": ("MODEL_UQ", "real_collapse_positive", ("PRESCRIBE",)),
    "E159::Norman_P3::raw_log_prob": ("MODEL_UQ", "real_collapse_positive", ("PRESCRIBE",)),
    "E159::Norman_P4::official": ("MODEL_UQ", "real_collapse_positive", ("PRESCRIBE",)),
    "E159::Norman_P4::raw_log_prob": ("MODEL_UQ", "real_collapse_positive", ("PRESCRIBE",)),
    "E165::Wessels::PRESCRIBE_seed3407": ("MODEL_UQ", "real_collapse_positive", ("PRESCRIBE",)),
    "E165::Wessels::PRESCRIBE_seed3408": ("MODEL_UQ", "real_collapse_positive", ("PRESCRIBE",)),
    "E165::Wessels::PRESCRIBE_seed3409": ("MODEL_UQ", "real_collapse_positive", ("PRESCRIBE",)),
    "E87::sciPlex3_to_OpenProblems": ("MODEL_UQ", "real_noncollapsed_reference", ("CPA", "ridge")),
    "E89::sciPlex3_to_sciPlex4": ("MODEL_UQ", "real_noncollapsed_reference", ("CPA", "interpolation")),
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def committed_payload(path: Path, head: str, expected_sha256: str | None = None) -> tuple[dict[str, Any], bytes]:
    relative = path.relative_to(ROOT).as_posix()
    if not path.is_file() or path.is_symlink():
        raise IntegrityFailure(f"Missing or symlinked frozen input: {relative}")
    payload = path.read_bytes()
    observed = sha256_bytes(payload)
    if expected_sha256 is not None and observed != expected_sha256:
        raise IntegrityFailure(f"Frozen input changed: {relative}")
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"Required file is not committed: {relative}") from exc
    if sha256_bytes(committed) != observed:
        raise IntegrityFailure(f"Working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": len(payload), "sha256": observed}, payload


def parse_source_lock(payload: bytes) -> pd.DataFrame:
    lock = pd.read_csv(BytesIO(payload), dtype=str)
    if len(lock) != 12 or set(lock.columns) != {"access_phase", "path", "sha256"} or lock.path.duplicated().any():
        raise IntegrityFailure("SOURCE_LOCK schema/count/uniqueness failed")
    if set(lock.access_phase) != {"PRETRUTH", "POSTGATE_TRUTH"}:
        raise IntegrityFailure("SOURCE_LOCK access phases failed")
    return lock


def verify_locked_phase(
    head: str,
    phase: str,
    frozen_lock: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    rows = []
    payloads = {}
    for row in frozen_lock.loc[frozen_lock.access_phase.eq(phase)].itertuples(index=False):
        path = ROOT / row.path
        metadata, payload = committed_payload(path, head, row.sha256)
        rows.append({"access_phase": phase, **metadata})
        payloads[row.path] = payload
    return rows, payloads


def verify_pretruth_inputs(head: str) -> dict[str, Any]:
    own = []
    own_payloads = {}
    for path in (RUNNER, CONTRACT, SOURCE_LOCK):
        metadata, payload = committed_payload(path, head)
        own.append({"access_phase": "PRETRUTH", **metadata})
        own_payloads[metadata["path"]] = payload
    source_lock_relative = SOURCE_LOCK.relative_to(ROOT).as_posix()
    frozen_lock = parse_source_lock(own_payloads[source_lock_relative])
    rows, payloads = verify_locked_phase(head, "PRETRUTH", frozen_lock)
    old_status_relative = OLD_STATUS.relative_to(ROOT).as_posix()
    old_manifest_relative = OLD_MANIFEST.relative_to(ROOT).as_posix()
    old_status = json.loads(payloads[old_status_relative])
    if old_status.get("strict_development_gate_passed") is not False:
        raise IntegrityFailure("E167 v1 FAIL state was not preserved")
    if old_status.get("results_manifest_sha256") != sha256_bytes(payloads[old_manifest_relative]):
        raise IntegrityFailure("E167 v1 manifest pointer changed")
    attestation = json.loads(payloads[ASSET_ATTESTATION.relative_to(ROOT).as_posix()])
    builder_relative = "tools/scripts/build_e167a_isolated_assets.py"
    if (
        attestation.get("builder_sha256") != sha256_bytes(payloads[builder_relative])
        or attestation.get("n_e167_sources_verified") != 30
        or attestation.get("source_hash_errors") != []
        or attestation.get("n_units") != 19
        or attestation.get("n_tasks") != 4334
        or attestation.get("n_prediction_arrays") != 29
        or attestation.get("n_prediction_task_map_rows") != 8380
        or attestation.get("score_operational_label_changes") != 0
        or attestation.get("magnitude_operational_label_changes") != 0
        or attestation.get("prediction_arrays_exact_roundtrip") is not True
    ):
        raise IntegrityFailure("Isolated asset build attestation failed")
    return {
        "hashes": own + rows,
        "payloads": {**own_payloads, **payloads},
        "frozen_lock": frozen_lock,
        "source_lock_sha256": sha256_bytes(own_payloads[source_lock_relative]),
    }


def verify_postgate_inputs(head: str, frozen_lock: pd.DataFrame) -> dict[str, Any]:
    rows, payloads = verify_locked_phase(head, "POSTGATE_TRUTH", frozen_lock)
    return {"hashes": rows, "payloads": payloads}


def revalidate_all_inputs(head: str, frozen_lock: pd.DataFrame, source_lock_sha256: str) -> None:
    source_metadata, source_payload = committed_payload(SOURCE_LOCK, head, source_lock_sha256)
    if source_metadata["sha256"] != source_lock_sha256:
        raise IntegrityFailure("Frozen source lock changed during formal run")
    if not parse_source_lock(source_payload).equals(frozen_lock):
        raise IntegrityFailure("Frozen source lock table changed during formal run")
    committed_payload(RUNNER, head)
    committed_payload(CONTRACT, head)
    for row in frozen_lock.itertuples(index=False):
        committed_payload(ROOT / row.path, head, row.sha256)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, dataframe_csv_bytes(frame))


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g").encode()


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def quantize(values: np.ndarray, tolerance: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise IntegrityFailure("Cannot quantize non-finite values")
    return np.rint(array / tolerance).astype(np.int64)


def rank_resolution(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    if n <= 1:
        return 0.0
    numerator = float(np.sum(counts**3 - counts))
    return float(1.0 - numerator / (n**3 - n))


def score_stats(score: np.ndarray) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64)
    finite = bool(np.isfinite(score).all())
    if not finite:
        return {
            "score_all_finite": False,
            "score_quantized_unique": 0,
            "score_unique_fraction": 0.0,
            "score_std": float("nan"),
            "score_max_tie_fraction": 1.0,
            "score_rank_resolution": 0.0,
            "G2a_score_nondegenerate_passed": False,
        }
    labels, counts = np.unique(quantize(score, SCORE_TOL), return_counts=True)
    std = float(np.std(score, ddof=0))
    passed = bool(len(labels) >= 2 and std > SCORE_TOL)
    return {
        "score_all_finite": True,
        "score_quantized_unique": int(len(labels)),
        "score_unique_fraction": float(len(labels) / len(score)),
        "score_std": std,
        "score_max_tie_fraction": float(np.max(counts) / len(score)),
        "score_rank_resolution": rank_resolution(counts),
        "G2a_score_nondegenerate_passed": passed,
    }


def predictor_certificate_rows(predictors: dict[str, np.ndarray], take: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for predictor_name, matrix in predictors.items():
        values = np.asarray(matrix[take], dtype=np.float64)
        finite = bool(np.isfinite(values).all())
        if not finite:
            rows.append({
                "predictor_name": predictor_name,
                "prediction_all_finite": False,
                "vector_dimension": values.shape[1],
                "quantized_unique_vectors": 0,
                "prediction_unique_fraction": 0.0,
                "prediction_max_repeat_fraction": 1.0,
                "prediction_rank_resolution": 0.0,
                "predictor_max_coordinate_std": float("nan"),
                "G3a_predictor_passed": False,
            })
            continue
        encoded = np.ascontiguousarray(quantize(values, PREDICTION_TOL))
        fingerprints = [hashlib.sha256(row.tobytes()).digest() for row in encoded]
        counts = np.asarray(list(Counter(fingerprints).values()), dtype=np.int64)
        max_std = float(np.max(np.std(values, axis=0, ddof=0)))
        rows.append({
            "predictor_name": predictor_name,
            "prediction_all_finite": True,
            "vector_dimension": values.shape[1],
            "quantized_unique_vectors": int(len(counts)),
            "prediction_unique_fraction": float(len(counts) / len(values)),
            "prediction_max_repeat_fraction": float(np.max(counts) / len(values)),
            "prediction_rank_resolution": rank_resolution(counts),
            "predictor_max_coordinate_std": max_std,
            "G3a_predictor_passed": bool(len(counts) >= 2 and max_std > PREDICTION_TOL),
        })
    return rows


def prediction_stats(predictors: dict[str, np.ndarray], take: np.ndarray) -> dict[str, Any]:
    details = predictor_certificate_rows(predictors, take)
    all_finite = bool(details and all(row["prediction_all_finite"] for row in details))
    passed = bool(details and all(row["G3a_predictor_passed"] for row in details))
    min_std = min((row["predictor_max_coordinate_std"] for row in details), default=float("nan"))
    return {
        "prediction_all_finite": all_finite,
        "n_predictors": len(details),
        "vector_dimensions": ";".join(str(row["vector_dimension"]) for row in details),
        "min_quantized_unique_vectors": min((row["quantized_unique_vectors"] for row in details), default=0),
        "prediction_unique_fraction_min": min((row["prediction_unique_fraction"] for row in details), default=0.0),
        "prediction_max_repeat_fraction_worst": max((row["prediction_max_repeat_fraction"] for row in details), default=1.0),
        "prediction_rank_resolution_min": min((row["prediction_rank_resolution"] for row in details), default=0.0),
        "min_predictor_max_coordinate_std": min_std,
        "G3a_prediction_task_dependence_passed": passed,
        "failed_predictor_names": ";".join(row["predictor_name"] for row in details if not row["G3a_predictor_passed"]) or "NONE",
    }


def batch_map(unit: PretruthUnit) -> dict[str, np.ndarray]:
    if unit.unit_id.startswith("E153::"):
        labels = np.asarray([str(value).split("::", 1)[0] for value in unit.task_ids], dtype=object)
    else:
        labels = np.repeat(unit.unit_id, len(unit.task_ids)).astype(object)
    return {str(label): np.flatnonzero(labels == label) for label in sorted(set(labels.astype(str)))}


def boundary_record(score: np.ndarray, coverage: float, direction: str) -> dict[str, Any]:
    labels = quantize(score, SCORE_TOL)
    n = len(labels)
    k = max(1, int(math.ceil(coverage * n)))
    if direction == "lowest_risk_accept":
        threshold = np.sort(labels)[k - 1]
        strict = int(np.sum(labels < threshold))
    elif direction == "highest_risk_review":
        threshold = np.sort(labels)[::-1][k - 1]
        strict = int(np.sum(labels > threshold))
    else:
        raise ValueError(direction)
    tied = int(np.sum(labels == threshold))
    slots = int(k - strict)
    exact = bool(tied == slots)
    log10_sets = float(
        (math.lgamma(tied + 1) - math.lgamma(slots + 1) - math.lgamma(tied - slots + 1))
        / math.log(10)
    )
    return {
        "coverage": float(coverage),
        "direction": direction,
        "selected_k": k,
        "strictly_selected": strict,
        "boundary_tie_size": tied,
        "boundary_slots_needed": slots,
        "boundary_exact_set": exact,
        "boundary_status": "EXACT_SET" if exact else "TIEBREAK_REQUIRED",
        "log10_possible_boundary_sets": log10_sets,
    }


def raw_weak_order_identical(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(rankdata(np.asarray(a, float), method="average"), rankdata(np.asarray(b, float), method="average")))


def operational_weak_order_identical(a: np.ndarray, b: np.ndarray) -> bool:
    score_labels = quantize(np.asarray(a, float), SCORE_TOL)
    magnitude_labels = quantize(np.asarray(b, float), MAGNITUDE_TOL)
    return bool(np.array_equal(rankdata(score_labels, method="average"), rankdata(magnitude_labels, method="average")))


def unevaluated_boundary(coverage: float, direction: str) -> dict[str, Any]:
    return {
        "coverage": float(coverage),
        "direction": direction,
        "selected_k": np.nan,
        "strictly_selected": np.nan,
        "boundary_tie_size": np.nan,
        "boundary_slots_needed": np.nan,
        "boundary_exact_set": False,
        "boundary_status": "NOT_EVALUATED_G2A_FAILED",
        "log10_possible_boundary_sets": np.nan,
    }


def risk_at_k(score: np.ndarray, loss: np.ndarray, k: int, tolerance: float) -> dict[str, float]:
    labels = quantize(score, tolerance)
    loss = np.asarray(loss, float)
    if not np.isfinite(loss).all():
        raise IntegrityFailure("Postgate loss contains non-finite values")
    threshold = np.sort(labels)[k - 1]
    strict_mask = labels < threshold
    tie_mask = labels == threshold
    strict_loss = loss[strict_mask]
    tied_loss = np.sort(loss[tie_mask])
    slots = int(k - len(strict_loss))
    strict_sum = float(np.sum(strict_loss))
    expected = (strict_sum + slots / len(tied_loss) * float(np.sum(tied_loss))) / k
    best = (strict_sum + float(np.sum(tied_loss[:slots]))) / k
    worst = (strict_sum + float(np.sum(tied_loss[-slots:]))) / k
    order = np.argsort(labels, kind="mergesort")
    deterministic = float(np.mean(loss[order[:k]]))
    return {
        "risk_tie_average": float(expected),
        "risk_best_legal_tie_order": float(best),
        "risk_worst_legal_tie_order": float(worst),
        "risk_mergesort_input_order": deterministic,
    }


def tie_aware_curve(score: np.ndarray, loss: np.ndarray, tolerance: float = SCORE_TOL) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for coverage in RC_COVERAGES:
        k = max(1, int(math.ceil(float(coverage) * len(score))))
        rows.append({"coverage": float(coverage), "selected_k": k, **risk_at_k(score, loss, k, tolerance)})
    curve = pd.DataFrame(rows)
    span = float(RC_COVERAGES[-1] - RC_COVERAGES[0])
    summary = {}
    mapping = {
        "aurc_tie_average": "risk_tie_average",
        "aurc_best_legal_tie_order": "risk_best_legal_tie_order",
        "aurc_worst_legal_tie_order": "risk_worst_legal_tie_order",
        "aurc_mergesort_input_order": "risk_mergesort_input_order",
    }
    for output, column in mapping.items():
        summary[output] = float(np.trapezoid(curve[column], curve.coverage) / span)
    summary["aurc_partial_identification_width"] = (
        summary["aurc_worst_legal_tie_order"] - summary["aurc_best_legal_tie_order"]
    )
    summary["max_pointwise_risk_width"] = float(
        np.max(curve.risk_worst_legal_tie_order - curve.risk_best_legal_tie_order)
    )
    return curve, summary


def certificate_for_batch(
    unit: PretruthUnit,
    batch_id: str,
    take: np.ndarray,
    replicate: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    score = unit.score[take]
    magnitude = unit.magnitude[take]
    score_summary = score_stats(score)
    pred_summary = prediction_stats(unit.predictors, take)
    predictor_rows = [
        {"unit_id": unit.unit_id, "batch_id": batch_id, **row}
        for row in predictor_certificate_rows(unit.predictors, take)
    ]
    g2 = bool(score_summary["G2a_score_nondegenerate_passed"])
    coverage_rows = []
    for coverage in RC_COVERAGES:
        low = boundary_record(score, float(coverage), "lowest_risk_accept") if g2 else unevaluated_boundary(float(coverage), "lowest_risk_accept")
        coverage_rows.append({"unit_id": unit.unit_id, "batch_id": batch_id, **low})
    high_target = boundary_record(score, TARGET_COVERAGE, "highest_risk_review") if g2 else unevaluated_boundary(TARGET_COVERAGE, "highest_risk_review")
    coverage_rows.append({"unit_id": unit.unit_id, "batch_id": batch_id, **high_target})
    low_target = boundary_record(score, TARGET_COVERAGE, "lowest_risk_accept") if g2 else unevaluated_boundary(TARGET_COVERAGE, "lowest_risk_accept")
    low_grid_exact = bool(g2 and all(row["boundary_exact_set"] for row in coverage_rows if row["direction"] == "lowest_risk_accept" and row["coverage"] < 1.0))
    registered_exact = bool(g2 and low_grid_exact and high_target["boundary_exact_set"])
    g4_evaluated = replicate is not None
    g4_passed = bool(replicate["g4_passed"]) if replicate is not None else False
    raw_g5_equivalent = bool(g2 and raw_weak_order_identical(score, magnitude))
    g5_equivalent = bool(g2 and operational_weak_order_identical(score, magnitude))
    g3 = bool(pred_summary["G3a_prediction_task_dependence_passed"])
    reasons = []
    if not g2:
        reasons.append("G2A_SCORE_SATURATION")
    if not g3:
        reasons.append("G3A_PREDICTOR_COLLAPSE")
    if g4_evaluated and not g4_passed:
        reasons.append("G4_UNSTABLE")
    if g5_equivalent:
        reasons.append("G5_OPERATIONAL_BASELINE_EQUIVALENT")
    if g2 and not registered_exact:
        reasons.append("G2B_TIEBREAK_REQUIRED")
    if not g2:
        status = "ABSTAIN_SCORE_SATURATION"
    elif unit.lane == "MODEL_UQ" and not g3:
        status = "ABSTAIN_PREDICTOR_COLLAPSE"
    elif unit.lane == "STRUCTURAL_RISK" and not g3:
        status = "STRUCTURAL_RISK_ONLY_PREDICTOR_COLLAPSE"
    elif g4_evaluated and not g4_passed:
        status = "ABSTAIN_UNSTABLE"
    elif g5_equivalent:
        status = "EVALUABLE_BASELINE_EQUIVALENT"
    elif not g4_evaluated and not registered_exact:
        status = "EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED"
    elif not g4_evaluated:
        status = "EVALUABLE_SELECTIVE_RANKING_G4_NOT_EVALUATED"
    elif not registered_exact:
        status = "EVALUABLE_ASSOCIATION_TIE_AWARE_G4_PASSED"
    else:
        status = "EVALUABLE_SELECTIVE_RANKING_G4_PASSED"
    row = {
        "unit_id": unit.unit_id,
        "study_id": unit.study_id,
        "batch_id": batch_id,
        "lane": unit.lane,
        "role": unit.role,
        "candidate_name": unit.candidate_name,
        "endpoint_id": unit.endpoint_id,
        "score_transform": unit.score_transform,
        "score_numeric_unit": unit.score_numeric_unit,
        "score_registered_precision": unit.score_registered_precision,
        "magnitude_transform": unit.magnitude_transform,
        "magnitude_numeric_unit": unit.magnitude_numeric_unit,
        "magnitude_registered_precision": unit.magnitude_registered_precision,
        "prediction_numeric_unit": unit.prediction_numeric_unit,
        "prediction_registered_precision": unit.prediction_registered_precision,
        "n_tasks": len(take),
        **score_summary,
        **pred_summary,
        "low_risk_20pct_boundary_status": low_target["boundary_status"],
        "low_risk_20pct_boundary_tie_size": low_target["boundary_tie_size"],
        "low_risk_20pct_boundary_slots_needed": low_target["boundary_slots_needed"],
        "high_risk_20pct_boundary_status": high_target["boundary_status"],
        "high_risk_20pct_boundary_tie_size": high_target["boundary_tie_size"],
        "high_risk_20pct_boundary_slots_needed": high_target["boundary_slots_needed"],
        "low_risk_rc_grid_all_exact": low_grid_exact,
        "G4_replicate_stability_evaluated": g4_evaluated,
        "G4_replicate_stability_passed": g4_passed if g4_evaluated else np.nan,
        "G5_raw_weak_order_equivalent_diagnostic": raw_g5_equivalent,
        "G5_operational_weak_order_equivalent": g5_equivalent,
        "registered_selection_exact": registered_exact,
        "evaluation_status": status,
        "reason_codes": ";".join(reasons) if reasons else "NONE",
        "evidence_scope": "RETROSPECTIVE_DEVELOPMENT",
        "deployment_authorized": False,
        "truth_used_for_G2_to_G5": False,
    }
    return row, coverage_rows, predictor_rows


def score_gate(values: np.ndarray) -> bool:
    return bool(score_stats(np.asarray(values, float))["G2a_score_nondegenerate_passed"])


def simple_prediction_gate(matrix: np.ndarray) -> tuple[bool, dict[str, Any]]:
    values = np.asarray(matrix, float)
    take = np.arange(len(values), dtype=int)
    stats = prediction_stats({"synthetic": values}, take)
    return bool(stats["G3a_prediction_task_dependence_passed"]), stats


def synthetic_tests() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    batch_a = np.zeros(50)
    batch_b = np.ones(50)
    pooled = np.concatenate([batch_a, batch_b])
    passed = (not score_gate(batch_a)) and (not score_gate(batch_b)) and score_gate(pooled)
    rows.append({
        "test_id": "S1_cross_batch_constant_masking",
        "expected": "each batch fails G2a although pooled vector varies",
        "observed": f"batchA={score_gate(batch_a)};batchB={score_gate(batch_b)};pooled={score_gate(pooled)}",
        "passed": passed,
    })

    counts = np.asarray([3, 6, 6, 6] + [5] * 15 + [4])
    coarse_score = np.repeat(np.arange(20, dtype=float), counts)
    coarse_low = boundary_record(coarse_score, TARGET_COVERAGE, "lowest_risk_accept")
    coarse_high = boundary_record(coarse_score, TARGET_COVERAGE, "highest_risk_review")
    passed = score_gate(coarse_score) and not coarse_low["boundary_exact_set"] and not coarse_high["boundary_exact_set"]
    rows.append({
        "test_id": "S2_twenty_score_levels",
        "expected": "G2a passes; 20% selection requires tie handling",
        "observed": f"G2a={score_gate(coarse_score)};unique={len(np.unique(coarse_score))};low={coarse_low['boundary_status']};high={coarse_high['boundary_status']}",
        "passed": passed,
    })

    phenotype = np.repeat(np.arange(20, dtype=float), 5)
    phenotype_matrix = np.column_stack([phenotype, phenotype**2])
    phenotype_pass, phenotype_stats = simple_prediction_gate(phenotype_matrix)
    rows.append({
        "test_id": "S3_twenty_prediction_phenotypes",
        "expected": "G3a passes with 20/100 unique vectors",
        "observed": f"G3a={phenotype_pass};unique={phenotype_stats['min_quantized_unique_vectors']}",
        "passed": phenotype_pass and phenotype_stats["min_quantized_unique_vectors"] == 20,
    })

    jitter = (np.arange(100, dtype=float) - 49.5) * 1e-12
    jitter_prediction = np.column_stack([jitter, -jitter])
    jitter_prediction_pass, _ = simple_prediction_gate(jitter_prediction)
    rows.append({
        "test_id": "S4_epsilon_jitter",
        "expected": "1e-12 score and prediction jitter both fail",
        "observed": f"score_G2a={score_gate(jitter)};prediction_G3a={jitter_prediction_pass}",
        "passed": (not score_gate(jitter)) and (not jitter_prediction_pass),
    })

    loss = 0.2 + 0.1 * np.sin(np.arange(100, dtype=float) / 7.0) + np.arange(100) / 10000
    reference_curve, reference = tie_aware_curve(coarse_score, loss)
    rng = np.random.default_rng(SEED)
    max_difference = 0.0
    for _ in range(100):
        order = rng.permutation(len(coarse_score))
        permuted_curve, permuted = tie_aware_curve(coarse_score[order], loss[order])
        for key in ("aurc_tie_average", "aurc_best_legal_tie_order", "aurc_worst_legal_tie_order"):
            max_difference = max(max_difference, abs(reference[key] - permuted[key]))
        for column in ("risk_tie_average", "risk_best_legal_tie_order", "risk_worst_legal_tie_order"):
            max_difference = max(max_difference, float(np.max(np.abs(reference_curve[column] - permuted_curve[column]))))
        max_difference = max(max_difference, abs(reference["max_pointwise_risk_width"] - permuted["max_pointwise_risk_width"]))
    rows.append({
        "test_id": "S5_tie_aware_permutation_invariance",
        "expected": "100 row permutations change all tie-aware RC points/bounds/AURC by <=1e-12",
        "observed": f"max_abs_difference={max_difference:.3e}",
        "passed": max_difference <= 1e-12,
    })

    high_resolution = np.arange(100, dtype=float)
    synthetic_prediction = np.column_stack([np.arange(200, dtype=float), np.arange(200, dtype=float) ** 2])
    synthetic_unit = PretruthUnit(
        unit_id="E153::SYN_BATCH_MAP", study_id="SYN_BATCH_MAP", lane="STRUCTURAL_RISK",
        endpoint_id="synthetic", perturbation_family="synthetic", candidate_name="synthetic",
        role="synthetic", score_transform="identity", score_numeric_unit="synthetic", score_registered_precision="1e-6",
        magnitude_transform="identity", magnitude_numeric_unit="synthetic", magnitude_registered_precision="1e-6",
        prediction_numeric_unit="synthetic", prediction_registered_precision="1e-6",
        task_ids=np.asarray([f"high::{i}" for i in range(100)] + [f"coarse::{i}" for i in range(100)]),
        clusters=np.asarray([str(i) for i in range(200)]),
        score=np.concatenate([high_resolution, coarse_score]),
        magnitude=np.mod(np.arange(200, dtype=float) * 37.0, 200.0),
        predictors={"synthetic": synthetic_prediction},
    )
    mapped = batch_map(synthetic_unit)
    high_status = certificate_for_batch(synthetic_unit, "high", mapped["high"], None)[0]["evaluation_status"]
    coarse_status = certificate_for_batch(synthetic_unit, "coarse", mapped["coarse"], None)[0]["evaluation_status"]
    rows.append({
        "test_id": "S6_batch_status_not_masked",
        "expected": "batch_map and certificate keep exact and tie-aware statuses separate",
        "observed": f"high={high_status};coarse={coarse_status}",
        "passed": high_status == "EVALUABLE_SELECTIVE_RANKING_G4_NOT_EVALUATED" and coarse_status == "EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED",
    })

    constant_prediction = np.ones((100, 3), dtype=float)
    constant_prediction_pass, _ = simple_prediction_gate(constant_prediction)
    structural_collapse_unit = PretruthUnit(
        unit_id="E153::SYN_STRUCTURAL_COLLAPSE", study_id="SYN", lane="STRUCTURAL_RISK",
        endpoint_id="synthetic", perturbation_family="synthetic", candidate_name="synthetic",
        role="synthetic", score_transform="identity", score_numeric_unit="synthetic", score_registered_precision="1e-6",
        magnitude_transform="identity", magnitude_numeric_unit="synthetic", magnitude_registered_precision="1e-6",
        prediction_numeric_unit="synthetic", prediction_registered_precision="1e-6",
        task_ids=np.asarray([f"single::{i}" for i in range(100)]), clusters=np.asarray([str(i) for i in range(100)]),
        score=high_resolution, magnitude=np.mod(np.arange(100) * 37, 100).astype(float),
        predictors={"synthetic": constant_prediction},
    )
    structural_status = certificate_for_batch(structural_collapse_unit, "single", np.arange(100), None)[0]["evaluation_status"]
    rows.append({
        "test_id": "S7_variable_score_constant_prediction",
        "expected": "variable score passes G2a; structural lane exposes predictor collapse without calling it MODEL_UQ",
        "observed": f"G2a={score_gate(high_resolution)};G3a={constant_prediction_pass};status={structural_status}",
        "passed": score_gate(high_resolution) and not constant_prediction_pass and structural_status == "STRUCTURAL_RISK_ONLY_PREDICTOR_COLLAPSE",
    })

    magnitude_clone = 2.0 * coarse_score + 3.0
    clone_left_curve, clone_left = tie_aware_curve(coarse_score, loss, SCORE_TOL)
    clone_right_curve, clone_right = tie_aware_curve(magnitude_clone, loss, MAGNITUDE_TOL)
    clone_difference = 0.0
    for column in ("risk_tie_average", "risk_best_legal_tie_order", "risk_worst_legal_tie_order"):
        clone_difference = max(clone_difference, float(np.max(np.abs(clone_left_curve[column] - clone_right_curve[column]))))
    clone_difference = max(clone_difference, abs(clone_left["aurc_tie_average"] - clone_right["aurc_tie_average"]))
    clone_unit = PretruthUnit(
        unit_id="E153::SYN_CLONE", study_id="SYN", lane="STRUCTURAL_RISK",
        endpoint_id="synthetic", perturbation_family="synthetic", candidate_name="synthetic",
        role="synthetic", score_transform="identity", score_numeric_unit="synthetic", score_registered_precision="1e-6",
        magnitude_transform="identity", magnitude_numeric_unit="synthetic", magnitude_registered_precision="1e-6",
        prediction_numeric_unit="synthetic", prediction_registered_precision="1e-6",
        task_ids=np.asarray([f"single::{i}" for i in range(100)]), clusters=np.asarray([str(i) for i in range(100)]),
        score=coarse_score, magnitude=magnitude_clone,
        predictors={"synthetic": np.column_stack([np.arange(100), np.arange(100) ** 2])},
    )
    clone_status = certificate_for_batch(clone_unit, "single", np.arange(100), None)[0]["evaluation_status"]
    rows.append({
        "test_id": "S8_magnitude_weak_order_clone",
        "expected": "same operational weak order and identical RC/bounds/AURC at all coverages",
        "observed": f"operational_weak_order={operational_weak_order_identical(coarse_score, magnitude_clone)};max_delta={clone_difference:.3e};status={clone_status}",
        "passed": operational_weak_order_identical(coarse_score, magnitude_clone) and clone_difference <= 1e-15 and clone_status == "EVALUABLE_BASELINE_EQUIVALENT",
    })

    affine_score = np.tile(np.asarray([0.0, 0.6e-6, 1.2e-6, 100e-6]), 25)
    affine_magnitude = 3.0 + 2.0 * affine_score
    _, affine_left = tie_aware_curve(affine_score, loss, SCORE_TOL)
    _, affine_right = tie_aware_curve(affine_magnitude, loss, MAGNITUDE_TOL)
    affine_unit = PretruthUnit(
        unit_id="E153::SYN_AFFINE", study_id="SYN", lane="STRUCTURAL_RISK",
        endpoint_id="synthetic", perturbation_family="synthetic", candidate_name="synthetic",
        role="synthetic", score_transform="identity", score_numeric_unit="synthetic", score_registered_precision="1e-6",
        magnitude_transform="identity", magnitude_numeric_unit="synthetic", magnitude_registered_precision="1e-6",
        prediction_numeric_unit="synthetic", prediction_registered_precision="1e-6",
        task_ids=np.asarray([f"single::{i}" for i in range(100)]), clusters=np.asarray([str(i) for i in range(100)]),
        score=affine_score, magnitude=affine_magnitude,
        predictors={"synthetic": np.column_stack([np.arange(100), np.arange(100) ** 2])},
    )
    affine_certificate = certificate_for_batch(affine_unit, "single", np.arange(100), None)[0]
    rows.append({
        "test_id": "S9_raw_affine_but_operationally_different",
        "expected": "raw affine order is not called G5-equivalent when frozen quantization changes ties",
        "observed": f"raw={raw_weak_order_identical(affine_score, affine_magnitude)};operational={affine_certificate['G5_operational_weak_order_equivalent']};status={affine_certificate['evaluation_status']};delta_AURC={affine_left['aurc_tie_average']-affine_right['aurc_tie_average']:.3e}",
        "passed": raw_weak_order_identical(affine_score, affine_magnitude) and not affine_certificate["G5_operational_weak_order_equivalent"] and affine_certificate["evaluation_status"] != "EVALUABLE_BASELINE_EQUIVALENT",
    })

    nonfinite_unit = PretruthUnit(
        unit_id="E153::SYN_NONFINITE", study_id="SYN_NONFINITE", lane="STRUCTURAL_RISK",
        endpoint_id="synthetic", perturbation_family="synthetic", candidate_name="synthetic",
        role="synthetic", score_transform="identity", score_numeric_unit="synthetic", score_registered_precision="1e-6",
        magnitude_transform="identity", magnitude_numeric_unit="synthetic", magnitude_registered_precision="1e-6",
        prediction_numeric_unit="synthetic", prediction_registered_precision="1e-6",
        task_ids=np.asarray([f"single::{i}" for i in range(12)]), clusters=np.asarray([str(i) for i in range(12)]),
        score=np.asarray([float(i) for i in range(11)] + [float("nan")]),
        magnitude=np.arange(12, dtype=float), predictors={"synthetic": np.column_stack([np.arange(12), np.arange(12) ** 2])},
    )
    nonfinite_status = certificate_for_batch(nonfinite_unit, "single", np.arange(12), None)[0]["evaluation_status"]
    rows.append({
        "test_id": "S10_nonfinite_score_abstains",
        "expected": "non-finite score produces ABSTAIN instead of crashing G2b",
        "observed": nonfinite_status,
        "passed": nonfinite_status == "ABSTAIN_SCORE_SATURATION",
    })
    return pd.DataFrame(rows)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if len(a) < 4 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    value = np.corrcoef(rankdata(a, method="average"), rankdata(b, method="average"))[0, 1]
    return float(value) if math.isfinite(value) else float("nan")


def load_pretruth_units(payloads: dict[str, bytes]) -> list[PretruthUnit]:
    registry_columns = {
        "unit_id", "study_id", "lane", "endpoint_id", "perturbation_family", "candidate_name", "role",
        "score_transform", "score_numeric_unit", "score_registered_precision",
        "magnitude_transform", "magnitude_numeric_unit", "magnitude_registered_precision",
        "prediction_numeric_unit", "prediction_registered_precision", "predictor_assets_json",
    }
    task_columns = {"unit_id", "task_id", "cluster", "candidate_score_pretruth", "magnitude_score_pretruth"}
    mapping_columns = {"unit_id", "predictor_name", "asset_key", "row_index", "task_id"}
    registry = pd.read_csv(BytesIO(payloads[UNIT_REGISTRY.relative_to(ROOT).as_posix()]), dtype=str)
    tasks = pd.read_csv(BytesIO(payloads[PRETRUTH_TASKS.relative_to(ROOT).as_posix()]), dtype={"unit_id": str, "task_id": str, "cluster": str})
    mapping = pd.read_csv(BytesIO(payloads[PREDICTION_TASK_MAP.relative_to(ROOT).as_posix()]), dtype={"unit_id": str, "predictor_name": str, "asset_key": str, "task_id": str})
    if set(registry.columns) != registry_columns or set(tasks.columns) != task_columns or set(mapping.columns) != mapping_columns:
        raise IntegrityFailure("Isolated pretruth exact-schema check failed")
    if len(registry) != 19 or registry.unit_id.duplicated().any() or set(registry.unit_id) != set(EXPECTED_UNIT_SPECS):
        raise IntegrityFailure("Isolated unit registry failed")
    if not np.isfinite(tasks[["candidate_score_pretruth", "magnitude_score_pretruth"]].to_numpy(float)).all():
        raise IntegrityFailure("Isolated pretruth scores contain non-finite values")
    if tasks[["unit_id", "task_id", "cluster"]].isna().any().any() or tasks.cluster.astype(str).str.len().eq(0).any():
        raise IntegrityFailure("Isolated pretruth identifiers failed")
    if not registry.score_transform.eq("identity_as_loaded_by_E167_v1").all() or not registry.magnitude_transform.eq("identity_as_loaded_by_E167_v1").all():
        raise IntegrityFailure("Frozen score/magnitude transform identity failed")
    if not registry[["score_registered_precision", "magnitude_registered_precision", "prediction_registered_precision"]].eq("1e-6").all().all():
        raise IntegrityFailure("Frozen provider precision registry failed")
    units = []
    registered_asset_keys: set[str] = set()
    with np.load(BytesIO(payloads[PRETRUTH_PREDICTIONS.relative_to(ROOT).as_posix()])) as archive:
        for row in registry.itertuples(index=False):
            expected_lane, expected_role, expected_predictors = EXPECTED_UNIT_SPECS[row.unit_id]
            if row.lane != expected_lane or row.role != expected_role:
                raise IntegrityFailure(f"Frozen unit lane/role failed: {row.unit_id}")
            block = tasks.loc[tasks.unit_id.eq(row.unit_id)].reset_index(drop=True)
            if len(block) < 12 or block.task_id.duplicated().any():
                raise IntegrityFailure(f"Isolated pretruth task block failed: {row.unit_id}")
            predictor_assets = json.loads(row.predictor_assets_json)
            if tuple(sorted(predictor_assets)) != tuple(sorted(expected_predictors)):
                raise IntegrityFailure(f"Frozen predictor identity failed: {row.unit_id}")
            predictors = {}
            for predictor_name, asset_key in predictor_assets.items():
                registered_asset_keys.add(asset_key)
                if asset_key not in archive.files:
                    raise IntegrityFailure(f"Missing isolated prediction asset: {row.unit_id}/{asset_key}")
                matrix = np.asarray(archive[asset_key], dtype=np.float64)
                if matrix.ndim != 2 or matrix.shape[0] != len(block):
                    raise IntegrityFailure(f"Prediction shape mismatch: {row.unit_id}/{predictor_name}/{matrix.shape}")
                map_block = mapping.loc[(mapping.unit_id == row.unit_id) & (mapping.predictor_name == predictor_name)].sort_values("row_index")
                if (
                    len(map_block) != len(block)
                    or not np.array_equal(map_block.row_index.to_numpy(int), np.arange(len(block)))
                    or not np.array_equal(map_block.task_id.astype(str).to_numpy(), block.task_id.astype(str).to_numpy())
                    or not map_block.asset_key.eq(asset_key).all()
                ):
                    raise IntegrityFailure(f"Prediction task-row map failed: {row.unit_id}/{predictor_name}")
                predictors[predictor_name] = matrix
            units.append(PretruthUnit(
                unit_id=row.unit_id,
                study_id=row.study_id,
                lane=row.lane,
                endpoint_id=row.endpoint_id,
                perturbation_family=row.perturbation_family,
                candidate_name=row.candidate_name,
                role=row.role,
                score_transform=row.score_transform,
                score_numeric_unit=row.score_numeric_unit,
                score_registered_precision=row.score_registered_precision,
                magnitude_transform=row.magnitude_transform,
                magnitude_numeric_unit=row.magnitude_numeric_unit,
                magnitude_registered_precision=row.magnitude_registered_precision,
                prediction_numeric_unit=row.prediction_numeric_unit,
                prediction_registered_precision=row.prediction_registered_precision,
                task_ids=block.task_id.astype(str).to_numpy(),
                clusters=block.cluster.astype(str).to_numpy(),
                score=block.candidate_score_pretruth.to_numpy(float),
                magnitude=block.magnitude_score_pretruth.to_numpy(float),
                predictors=predictors,
            ))
        if set(archive.files) != registered_asset_keys:
            raise IntegrityFailure("Unregistered or missing NPZ arrays detected")
    if len(units) != 19 or len(tasks) != sum(len(unit.task_ids) for unit in units):
        raise IntegrityFailure("Isolated pretruth asset coverage failed")
    if len(mapping) != sum(len(unit.task_ids) * len(unit.predictors) for unit in units):
        raise IntegrityFailure("Prediction task-map coverage failed")
    return units


def wessels_replicate_summary(units: list[PretruthUnit]) -> dict[str, Any]:
    wessels = sorted([unit for unit in units if unit.study_id == "Wessels"], key=lambda unit: unit.unit_id)
    if len(wessels) != 3:
        raise IntegrityFailure("Expected three Wessels pretruth replicates")
    reference_tasks = list(wessels[0].task_ids.astype(str))
    score_columns = []
    for unit in wessels:
        indexed = {task: value for task, value in zip(unit.task_ids.astype(str), unit.score)}
        if set(indexed) != set(reference_tasks):
            raise IntegrityFailure("Wessels replicate task alignment failed")
        score_columns.append(np.asarray([indexed[task] for task in reference_tasks], float))
    matrix = np.column_stack(score_columns)
    pairwise = [spearman(matrix[:, left], matrix[:, right]) for left in range(3) for right in range(left + 1, 3)]
    median_pairwise = float(np.median(pairwise))
    ranks = np.column_stack([rankdata(matrix[:, index], method="average") for index in range(3)])
    rank_sums = ranks.sum(axis=1)
    n, k = ranks.shape
    kendall_w = float(12 * np.sum((rank_sums - rank_sums.mean()) ** 2) / (k * k * (n**3 - n)))
    rng = np.random.default_rng(SEED + 165)
    draws = []
    for _ in range(2000):
        take = rng.integers(0, n, size=n)
        values = [spearman(matrix[take, left], matrix[take, right]) for left in range(3) for right in range(left + 1, 3)]
        finite = [value for value in values if math.isfinite(value)]
        draws.append(float(np.median(finite)) if finite else np.nan)
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "median_pairwise_spearman": median_pairwise,
        "kendall_w": kendall_w,
        "bootstrap_ci95_low": float(low),
        "bootstrap_ci95_high": float(high),
        "g4_passed": bool(median_pairwise >= 0.5 and low > 0),
    }


def analyze_pretruth(pretruth_payloads: dict[str, bytes]) -> dict[str, Any]:
    units = load_pretruth_units(pretruth_payloads)
    replicate = wessels_replicate_summary(units)
    batch_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    predictor_rows: list[dict[str, Any]] = []
    for unit in units:
        for batch_id, take in batch_map(unit).items():
            unit_replicate = replicate if unit.study_id == "Wessels" else None
            certificate, boundaries, predictor_details = certificate_for_batch(unit, batch_id, take, unit_replicate)
            batch_rows.append(certificate)
            boundary_rows.extend(boundaries)
            predictor_rows.extend(predictor_details)
    batches = pd.DataFrame(batch_rows).sort_values(["study_id", "batch_id", "unit_id"]).reset_index(drop=True)
    boundaries = pd.DataFrame(boundary_rows).sort_values(["unit_id", "batch_id", "direction", "coverage"]).reset_index(drop=True)
    predictors = pd.DataFrame(predictor_rows).sort_values(["unit_id", "batch_id", "predictor_name"]).reset_index(drop=True)
    if len(batches) != 45 or len(predictors) != 81:
        raise IntegrityFailure(f"Frozen batch/predictor-row count failed: {len(batches)}/{len(predictors)}")

    unit_summary = batches.groupby(["unit_id", "study_id", "lane", "role"], as_index=False).agg(
        n_batches=("batch_id", "size"),
        n_G2a_pass=("G2a_score_nondegenerate_passed", "sum"),
        n_G3a_pass=("G3a_prediction_task_dependence_passed", "sum"),
        n_exact_selection=("registered_selection_exact", "sum"),
        n_tie_aware=("evaluation_status", lambda values: int(np.sum(pd.Series(values).str.contains("TIE_AWARE")))),
        statuses=("evaluation_status", lambda values: ";".join(sorted(set(values)))),
    )

    synthetic = synthetic_tests()
    references = batches.role.eq("real_noncollapsed_reference")
    official_collapse = batches.unit_id.str.contains("E159::Norman_P[34]::official", regex=True)
    raw_collapse = batches.unit_id.str.contains("E159::Norman_P[34]::raw_log_prob", regex=True)
    wessels = batches.study_id.eq("Wessels")
    replogle_k562 = batches.batch_id.str.contains("Replogle_cellline_holdout_1_K562")
    santinha_low = batches.batch_id.str.contains(
        "Santinha_context_holdout_[1245]_", regex=True
    )
    old_status = json.loads(pretruth_payloads[OLD_STATUS.relative_to(ROOT).as_posix()])
    old_manifest_payload = pretruth_payloads[OLD_MANIFEST.relative_to(ROOT).as_posix()]
    if official_collapse.sum() != 2 or raw_collapse.sum() != 2 or wessels.sum() != 3 or references.sum() != 38:
        raise IntegrityFailure("Frozen real-unit regression mask count failed")
    known_tie_mask = replogle_k562 | santinha_low
    if known_tie_mask.sum() != 5:
        raise IntegrityFailure("Frozen known-tie batch count failed")
    checks = {
        "e167_v1_fail_and_manifest_preserved": bool(
            old_status["strict_development_gate_passed"] is False
            and old_status["results_manifest_sha256"] == sha256_bytes(old_manifest_payload)
        ),
        "official_collapse_status_preserved": bool(batches.loc[official_collapse, "evaluation_status"].eq("ABSTAIN_SCORE_SATURATION").all() and (~batches.loc[official_collapse, "G3a_prediction_task_dependence_passed"]).all()),
        "raw_score_cannot_rescue_collapsed_prediction": bool(batches.loc[raw_collapse, "evaluation_status"].eq("ABSTAIN_PREDICTOR_COLLAPSE").all()),
        "wessels_prediction_and_replicate_failure_preserved": bool(batches.loc[wessels, "evaluation_status"].eq("ABSTAIN_PREDICTOR_COLLAPSE").all() and batches.loc[wessels, "reason_codes"].str.contains("G4_UNSTABLE").all()),
        "noncollapsed_reference_prediction_batches_retained": bool(batches.loc[references, "G3a_prediction_task_dependence_passed"].all()),
        "known_cross_cutoff_ties_detected": bool((~batches.loc[known_tie_mask, "registered_selection_exact"]).all()),
        "synthetic_regression_suite": bool(synthetic.passed.all()),
        "no_truth_gate_override": bool(batches.truth_used_for_G2_to_G5.eq(False).all()),
        "all_historical_batches_deployment_unauthorized": bool((~batches.deployment_authorized).all()),
        "all_historical_batches_development_scope": bool(batches.evidence_scope.eq("RETROSPECTIVE_DEVELOPMENT").all()),
        "no_authorized_or_deploy_status_tokens": bool(~batches.evaluation_status.str.contains("AUTHORIZED|DEPLOY", regex=True).any()),
    }
    gate_summary = pd.DataFrame([{"phase": "PRETRUTH", "check": key, "passed": value} for key, value in checks.items()])
    return {
        "units": units,
        "batches": batches,
        "predictors": predictors,
        "boundaries": boundaries,
        "unit_summary": unit_summary,
        "synthetic": synthetic,
        "gate_summary": gate_summary,
        "pretruth_pass": bool(all(checks.values())),
        "replicate": replicate,
    }


def load_postgate_tables(
    units: list[PretruthUnit],
    postgate_payloads: dict[str, bytes],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.read_csv(BytesIO(postgate_payloads[POSTGATE_TRUTH.relative_to(ROOT).as_posix()]), dtype={"unit_id": str, "task_id": str})
    viability = pd.read_csv(BytesIO(postgate_payloads[POSTGATE_VIABILITY.relative_to(ROOT).as_posix()]), dtype={"unit_id": str})
    if set(truth.columns) != {"unit_id", "task_id", "registered_truth_loss"} or set(viability.columns) != {"unit_id", "predictor_viability_fraction_vs_simple"}:
        raise IntegrityFailure("Postgate exact-schema check failed")
    expected = {(unit.unit_id, task) for unit in units for task in unit.task_ids.astype(str)}
    observed = set(zip(truth.unit_id.astype(str), truth.task_id.astype(str)))
    if len(truth) != len(expected) or observed != expected or truth.duplicated(["unit_id", "task_id"]).any():
        raise IntegrityFailure("Postgate truth coverage failed")
    if not np.isfinite(truth.registered_truth_loss.to_numpy(float)).all():
        raise IntegrityFailure("Postgate truth contains non-finite loss")
    if len(viability) != len(units) or viability.unit_id.duplicated().any() or set(viability.unit_id) != {unit.unit_id for unit in units}:
        raise IntegrityFailure("Postgate viability coverage failed")
    finite_viability = viability.loc[np.isfinite(viability.predictor_viability_fraction_vs_simple.to_numpy(float))]
    expected_viability_ids = {
        "E165::Wessels::PRESCRIBE_seed3407", "E165::Wessels::PRESCRIBE_seed3408",
        "E165::Wessels::PRESCRIBE_seed3409", "E87::sciPlex3_to_OpenProblems", "E89::sciPlex3_to_sciPlex4",
    }
    values = finite_viability.predictor_viability_fraction_vs_simple.to_numpy(float)
    if set(finite_viability.unit_id) != expected_viability_ids or not np.all((values >= 0) & (values <= 1)):
        raise IntegrityFailure("Frozen postgate viability identity/range failed")
    return truth, viability


def analyze_postgate(pretruth: dict[str, Any], postgate_payloads: dict[str, bytes]) -> dict[str, Any]:
    units = pretruth["units"]
    batches = pretruth["batches"]
    truth, viability = load_postgate_tables(units, postgate_payloads)
    truth_index = truth.set_index(["unit_id", "task_id"]).registered_truth_loss
    viability_map = viability.set_index("unit_id").predictor_viability_fraction_vs_simple.to_dict()
    rc_rows: list[dict[str, Any]] = []
    aurc_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for unit in units:
        unit_viability = float(viability_map[unit.unit_id])
        if math.isfinite(unit_viability):
            viability_status = "PASS_MAJORITY_TASKS" if unit_viability > 0.5 else "FAIL_NOT_MAJORITY_TASKS"
        else:
            viability_status = "NOT_EVALUATED"
        for batch_id, take in batch_map(unit).items():
            certificate = batches.loc[(batches.unit_id == unit.unit_id) & (batches.batch_id == batch_id)].iloc[0]
            losses = np.asarray([truth_index.loc[(unit.unit_id, str(unit.task_ids[index]))] for index in take], float)
            endpoint_nondegenerate = bool(np.isfinite(losses).all() and np.unique(losses).size >= 2)
            if not endpoint_nondegenerate:
                interpretation = "ENDPOINT_TRUTH_DEGENERATE"
            elif viability_status.startswith("FAIL"):
                interpretation = "UPSTREAM_PREDICTOR_INVALID"
            elif viability_status.startswith("PASS"):
                interpretation = "UPSTREAM_VIABILITY_PASSED"
            else:
                interpretation = "UPSTREAM_VIABILITY_NOT_EVALUATED"
            metric_evaluable = bool(certificate.G2a_score_nondegenerate_passed and endpoint_nondegenerate)
            diagnostic_rows.append({
                "unit_id": unit.unit_id,
                "study_id": unit.study_id,
                "batch_id": batch_id,
                "pretruth_evaluation_status": certificate.evaluation_status,
                "endpoint_truth_nondegenerate": endpoint_nondegenerate,
                "predictor_viability_fraction_vs_simple": unit_viability,
                "postgate_upstream_viability_status": viability_status,
                "postgate_interpretation_status": interpretation,
                "association_metric_evaluable": metric_evaluable,
                "deployment_authorized": False,
            })
            if not metric_evaluable:
                aurc_rows.append({
                    "unit_id": unit.unit_id, "study_id": unit.study_id, "batch_id": batch_id, "n_tasks": len(take),
                    "pretruth_evaluation_status": certificate.evaluation_status,
                    "postgate_interpretation_status": interpretation,
                    "metric_status": "NOT_EVALUATED_PRETRUTH_OR_ENDPOINT_GATE_FAILED",
                    "endpoint_truth_nondegenerate": endpoint_nondegenerate,
                    "predictor_viability_fraction_vs_simple": unit_viability,
                    "candidate_aurc_tie_average": np.nan,
                    "candidate_aurc_best_legal_tie_order": np.nan,
                    "candidate_aurc_worst_legal_tie_order": np.nan,
                    "candidate_aurc_mergesort_input_order": np.nan,
                    "candidate_aurc_partial_identification_width": np.nan,
                    "candidate_max_pointwise_risk_width": np.nan,
                    "magnitude_aurc_tie_average": np.nan,
                    "magnitude_aurc_best_legal_tie_order": np.nan,
                    "magnitude_aurc_worst_legal_tie_order": np.nan,
                    "magnitude_aurc_mergesort_input_order": np.nan,
                    "magnitude_aurc_partial_identification_width": np.nan,
                    "magnitude_max_pointwise_risk_width": np.nan,
                    "delta_tie_average_aurc_candidate_minus_magnitude": np.nan,
                    "postgate_historical_diagnostic_only": True,
                })
                continue
            candidate_curve, candidate_aurc = tie_aware_curve(unit.score[take], losses, SCORE_TOL)
            magnitude_curve, magnitude_aurc = tie_aware_curve(unit.magnitude[take], losses, MAGNITUDE_TOL)
            curve = pd.DataFrame({
                "unit_id": unit.unit_id,
                "batch_id": batch_id,
                "pretruth_evaluation_status": certificate.evaluation_status,
                "postgate_interpretation_status": interpretation,
                "coverage": candidate_curve.coverage,
                "selected_k": candidate_curve.selected_k,
                "candidate_risk_tie_average": candidate_curve.risk_tie_average,
                "candidate_risk_best_legal_tie_order": candidate_curve.risk_best_legal_tie_order,
                "candidate_risk_worst_legal_tie_order": candidate_curve.risk_worst_legal_tie_order,
                "magnitude_risk_tie_average": magnitude_curve.risk_tie_average,
                "magnitude_risk_best_legal_tie_order": magnitude_curve.risk_best_legal_tie_order,
                "magnitude_risk_worst_legal_tie_order": magnitude_curve.risk_worst_legal_tie_order,
                "deployment_authorized": False,
                "postgate_historical_diagnostic_only": True,
            })
            rc_rows.extend(curve.to_dict("records"))
            candidate_prefixed = {f"candidate_{key}": value for key, value in candidate_aurc.items()}
            magnitude_prefixed = {f"magnitude_{key}": value for key, value in magnitude_aurc.items()}
            aurc_rows.append({
                "unit_id": unit.unit_id, "study_id": unit.study_id, "batch_id": batch_id, "n_tasks": len(take),
                "pretruth_evaluation_status": certificate.evaluation_status,
                "postgate_interpretation_status": interpretation,
                "metric_status": "TIE_AWARE_HISTORICAL_DIAGNOSTIC",
                "endpoint_truth_nondegenerate": endpoint_nondegenerate,
                "predictor_viability_fraction_vs_simple": unit_viability,
                **candidate_prefixed,
                **magnitude_prefixed,
                "delta_tie_average_aurc_candidate_minus_magnitude": candidate_aurc["aurc_tie_average"] - magnitude_aurc["aurc_tie_average"],
                "postgate_historical_diagnostic_only": True,
            })
    rc = pd.DataFrame(rc_rows).sort_values(["unit_id", "batch_id", "coverage"]).reset_index(drop=True)
    aurc = pd.DataFrame(aurc_rows).sort_values(["study_id", "batch_id", "unit_id"]).reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(["study_id", "batch_id", "unit_id"]).reset_index(drop=True)
    e87 = diagnostics.unit_id.eq("E87::sciPlex3_to_OpenProblems")
    if e87.sum() != 1:
        raise IntegrityFailure("Frozen E87 postgate batch count failed")
    post_checks = {
        "postgate_truth_all_finite_and_joined": True,
        "endpoint_truth_nondegeneracy_reported": bool(diagnostics.endpoint_truth_nondegenerate.notna().all()),
        "e87_invalid_upstream_predictor_not_authorized": bool(diagnostics.loc[e87, "postgate_interpretation_status"].eq("UPSTREAM_PREDICTOR_INVALID").all()),
        "all_historical_batches_remain_deployment_unauthorized": bool((~diagnostics.deployment_authorized).all()),
        "candidate_and_magnitude_tie_aware_outputs_both_present": bool(
            aurc.loc[aurc.metric_status.eq("TIE_AWARE_HISTORICAL_DIAGNOSTIC"), ["candidate_aurc_tie_average", "magnitude_aurc_tie_average"]].notna().all().all()
        ),
    }
    post_summary = pd.DataFrame([{"phase": "POSTGATE_TRUTH", "check": key, "passed": value} for key, value in post_checks.items()])
    gate_summary = pd.concat([pretruth["gate_summary"], post_summary], ignore_index=True)
    return {
        **pretruth,
        "rc": rc,
        "aurc": aurc,
        "postgate_diagnostics": diagnostics,
        "gate_summary": gate_summary,
        "strict_pass": bool(pretruth["pretruth_pass"] and all(post_checks.values())),
    }


def make_resolution_figure(batches: pd.DataFrame, path: Path) -> None:
    frame = batches.loc[batches.role.eq("real_noncollapsed_reference")].copy()
    colors = np.where(frame.registered_selection_exact, BLUE, ORANGE)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(
        frame.score_unique_fraction,
        frame.score_rank_resolution,
        c=colors,
        s=42,
        linewidth=0.7,
        edgecolor="white",
        alpha=0.92,
    )
    label_mask = (
        frame.score_unique_fraction.lt(0.15)
        | frame.batch_id.str.contains("Replogle_cellline_holdout_[12]", regex=True)
    )
    for row in frame.loc[label_mask].itertuples(index=False):
        label = row.batch_id.replace("Replogle_cellline_holdout_", "Replogle ").replace("Santinha_context_holdout_", "Santinha ")
        ax.annotate(label, (row.score_unique_fraction, row.score_rank_resolution), xytext=(4, 4), textcoords="offset points", fontsize=6.6, color=DARK)
    ax.scatter([], [], c=BLUE, s=42, label="All registered cutoffs exact")
    ax.scatter([], [], c=ORANGE, s=42, label="Tie-aware selection required")
    ax.set_xlabel("Quantized score levels / tasks")
    ax.set_ylabel("Rank resolution")
    ax.set_title("Batch-level score resolution reveals hidden ties", loc="left", fontweight="bold", color=DARK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_aurc_figure(aurc: pd.DataFrame, path: Path) -> None:
    frame = aurc.loc[
        aurc.metric_status.eq("TIE_AWARE_HISTORICAL_DIAGNOSTIC")
        & ~aurc.pretruth_evaluation_status.str.startswith("ABSTAIN")
        & aurc.postgate_interpretation_status.ne("UPSTREAM_PREDICTOR_INVALID")
    ].copy()
    frame = frame.sort_values("candidate_aurc_partial_identification_width", ascending=False).head(16)
    frame = frame.sort_values("candidate_aurc_partial_identification_width")
    labels = [
        value.replace("Replogle_cellline_holdout_", "Replogle ").replace("Santinha_context_holdout_", "Santinha ")
        for value in frame.batch_id
    ]
    left = frame.candidate_aurc_best_legal_tie_order.to_numpy(float)
    right = frame.candidate_aurc_worst_legal_tie_order.to_numpy(float)
    center = frame.candidate_aurc_tie_average.to_numpy(float)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.6, 5.4), facecolor="white")
    ax.set_facecolor("white")
    ax.hlines(y, left, right, color=ORANGE, linewidth=2.2)
    ax.scatter(center, y, color=BLUE, s=28, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("AURC (best legal tie order — worst legal tie order)")
    ax.set_title("Tie-aware AURC intervals among evaluable historical batches", loc="left", fontweight="bold", color=DARK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_flow_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 2.8), facecolor="white")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    items = [
        (0.2, "G2a\nscore varies", BLUE),
        (2.2, "G3a\nprediction varies", BLUE),
        (4.2, "G4 / G5\nstable, non-clone", BLUE),
        (6.2, "G2b\ncutoff ties", ORANGE),
        (8.2, "Evaluation output\nexact or tie-aware", GREEN),
    ]
    for index, (x, label, color) in enumerate(items):
        box = FancyBboxPatch(
            (x, 0.8), 1.55, 1.15,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(x + 0.775, 1.375, label, ha="center", va="center", fontsize=9, color=DARK, fontweight="bold")
        if index < len(items) - 1:
            ax.annotate("", xy=(items[index + 1][0] - 0.08, 1.375), xytext=(x + 1.63, 1.375), arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.4})
    ax.text(0.2, 2.35, "RIAG v2 separates metric evaluability from deployment claims", fontsize=12, fontweight="bold", color=DARK)
    ax.text(6.2, 0.45, "No jitter: report tie-average / bounds, or freeze a secondary rule before truth", fontsize=8, color=ORANGE)
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


def write_pretruth_stage(
    result: dict[str, Any],
    pretruth_hashes: list[dict[str, Any]],
    head: str,
) -> dict[str, Any]:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E167a release is append-only and already exists")
    (STAGING / "reports").mkdir(parents=True)
    (STAGING / "tables").mkdir()
    (STAGING / "figures").mkdir()
    transaction = {"schema": "safeconf_e167a_transaction_v1", "transaction_id": uuid.uuid4().hex, "created_at": now()}
    atomic_json(STAGING / ".E167A_TRANSACTION.json", transaction)
    atomic_csv(STAGING / "tables/E167A_PRETRUTH_INPUT_HASHES.csv", pd.DataFrame(pretruth_hashes))
    atomic_csv(STAGING / "tables/E167A_BATCH_CERTIFICATES.csv", result["batches"])
    atomic_csv(STAGING / "tables/E167A_PREDICTOR_CERTIFICATES.csv", result["predictors"])
    atomic_csv(STAGING / "tables/E167A_COVERAGE_BOUNDARIES.csv", result["boundaries"])
    atomic_csv(STAGING / "tables/E167A_UNIT_SUMMARY.csv", result["unit_summary"])
    atomic_csv(STAGING / "tables/E167A_SYNTHETIC_TESTS.csv", result["synthetic"])
    atomic_csv(STAGING / "tables/E167A_PRETRUTH_GATE_SUMMARY.csv", result["gate_summary"])
    pretruth_files = {
        "pretruth_input_hashes": "tables/E167A_PRETRUTH_INPUT_HASHES.csv",
        "batch_certificates": "tables/E167A_BATCH_CERTIFICATES.csv",
        "predictor_certificates": "tables/E167A_PREDICTOR_CERTIFICATES.csv",
        "coverage_boundaries": "tables/E167A_COVERAGE_BOUNDARIES.csv",
        "unit_summary": "tables/E167A_UNIT_SUMMARY.csv",
        "synthetic_tests": "tables/E167A_SYNTHETIC_TESTS.csv",
        "pretruth_gate_summary": "tables/E167A_PRETRUTH_GATE_SUMMARY.csv",
    }
    snapshot = {
        "schema": "safeconf_e167a_pretruth_gate_snapshot_v1",
        "created_at": now(),
        "git_head": head,
        "transaction_id": transaction["transaction_id"],
        "pretruth_gate_passed": result["pretruth_pass"],
        "n_units": len(result["units"]),
        "n_batches": len(result["batches"]),
        "postgate_truth_files_opened": 0,
        "pretruth_files_sha256": {
            key: {"relative_path": relative, "sha256": sha256_file(STAGING / relative)}
            for key, relative in pretruth_files.items()
        },
    }
    atomic_json(STAGING / "PRETRUTH_GATE_SNAPSHOT.json", snapshot)
    return {
        "transaction": transaction,
        "snapshot_sha256_before_truth": sha256_file(STAGING / "PRETRUTH_GATE_SNAPSHOT.json"),
    }


def validate_pretruth_stage(
    result: dict[str, Any],
    head: str,
    stage_state: dict[str, Any],
) -> None:
    snapshot_path = STAGING / "PRETRUTH_GATE_SNAPSHOT.json"
    snapshot_payload = snapshot_path.read_bytes()
    if sha256_bytes(snapshot_payload) != stage_state["snapshot_sha256_before_truth"]:
        raise IntegrityFailure("Pretruth snapshot changed after truth boundary")
    snapshot = json.loads(snapshot_payload)
    transaction = stage_state["transaction"]
    if (
        snapshot.get("git_head") != head
        or snapshot.get("transaction_id") != transaction["transaction_id"]
        or snapshot.get("n_units") != 19
        or snapshot.get("n_batches") != 45
        or snapshot.get("postgate_truth_files_opened") != 0
        or snapshot.get("pretruth_gate_passed") is not True
    ):
        raise IntegrityFailure("Pretruth snapshot identity/status failed")
    expected_frames = {
        "batch_certificates": result["batches"],
        "predictor_certificates": result["predictors"],
        "coverage_boundaries": result["boundaries"],
        "unit_summary": result["unit_summary"],
        "synthetic_tests": result["synthetic"],
        "pretruth_gate_summary": result["gate_summary"].loc[result["gate_summary"].phase.eq("PRETRUTH")].reset_index(drop=True),
    }
    registered = snapshot.get("pretruth_files_sha256", {})
    if set(registered) != set(expected_frames) | {"pretruth_input_hashes"}:
        raise IntegrityFailure("Pretruth snapshot file registry failed")
    for key, entry in registered.items():
        path = STAGING / entry["relative_path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise IntegrityFailure(f"Snapshotted pretruth file changed: {key}")
        if key in expected_frames and sha256_bytes(dataframe_csv_bytes(expected_frames[key])) != entry["sha256"]:
            raise IntegrityFailure(f"In-memory pretruth frame differs from snapshot: {key}")


def finish_release(
    result: dict[str, Any],
    all_input_hashes: list[dict[str, Any]],
    head: str,
    stage_state: dict[str, Any],
) -> dict[str, Any]:
    if not STAGING.exists() or RELEASE.exists():
        raise IntegrityFailure("E167a pretruth stage missing or release already exists")
    validate_pretruth_stage(result, head, stage_state)
    transaction = stage_state["transaction"]
    atomic_csv(STAGING / "tables/E167A_INPUT_HASHES.csv", pd.DataFrame(all_input_hashes))
    atomic_csv(STAGING / "tables/E167A_TIE_AWARE_RC.csv", result["rc"])
    atomic_csv(STAGING / "tables/E167A_TIE_AWARE_AURC.csv", result["aurc"])
    atomic_csv(STAGING / "tables/E167A_POSTGATE_DIAGNOSTICS.csv", result["postgate_diagnostics"])
    atomic_csv(STAGING / "tables/E167A_GATE_SUMMARY.csv", result["gate_summary"])
    make_resolution_figure(result["batches"], STAGING / "figures/F1_batch_score_resolution.svg")
    make_aurc_figure(result["aurc"], STAGING / "figures/F2_tie_aware_aurc_intervals.svg")
    make_flow_figure(STAGING / "figures/F3_riag_v2_decision_flow.svg")

    batches = result["batches"]
    affected = batches.loc[
        batches.batch_id.str.contains("Replogle|Santinha"),
        [
            "batch_id", "n_tasks", "score_quantized_unique", "G3a_prediction_task_dependence_passed",
            "low_risk_20pct_boundary_status", "high_risk_20pct_boundary_status", "evaluation_status",
        ],
    ]
    valid_widths = result["aurc"].loc[
        result["aurc"].metric_status.eq("TIE_AWARE_HISTORICAL_DIAGNOSTIC")
        & ~result["aurc"].pretruth_evaluation_status.str.startswith("ABSTAIN")
        & result["aurc"].postgate_interpretation_status.ne("UPSTREAM_PREDICTOR_INVALID")
    ]
    widths = valid_widths.sort_values("candidate_aurc_partial_identification_width", ascending=False).head(8)[
        ["batch_id", "pretruth_evaluation_status", "postgate_interpretation_status", "candidate_aurc_tie_average", "candidate_aurc_best_legal_tie_order", "candidate_aurc_worst_legal_tie_order", "candidate_aurc_partial_identification_width"]
    ]
    report = (
        "# E167a｜RIAG v2 批次级、tie-aware 协议修正\n\n"
        f"预设回归测试：`{'PASS' if result['strict_pass'] else 'FAIL'}`。E167 v1 的正式 `FAIL` 保持不变；E167a 仍是历史数据上的方法开发。\n\n"
        "## 修正后的含义\n\n"
        "预测向量是否随任务变化、风险分数是否非退化、具体 cutoff 能否给出唯一集合，现已分开判定。"
        "所有历史状态统一写为 EVALUABLE 或 ABSTAIN，`deployment_authorized` 固定为 false。缺少重复稳定性或上游预测器无效时，不再输出部署授权措辞。\n\n"
        f"正式分析包含 `{len(batches)}` 个历史部署批次；其中 "
        f"`{int(batches.G2a_score_nondegenerate_passed.sum())}` 个通过 G2a，"
        f"`{int(batches.G3a_prediction_task_dependence_passed.sum())}` 个通过 G3a，"
        f"`{int(batches.registered_selection_exact.sum())}` 个在全部登记 cutoff 上集合唯一。\n\n"
        "## Replogle 与 Santinha\n\n" + markdown_table(affected) + "\n\n"
        "## ties 对历史 AURC 的影响\n\n" + markdown_table(widths) + "\n\n"
        "普通 mergesort AURC 会受 CSV 原始行顺序影响。表中的 candidate tie-average、best 和 worst 对任意行排列不变；magnitude 也按相同规则单独计算。"
        "这些区间是门后历史真值审计，未参与 G2–G5 状态判定。\n\n"
        "E87 的上游预测器在历史 truth 上没有一个任务优于 no-change，因此被明确标记为 `UPSTREAM_PREDICTOR_INVALID`；RIAG 的数学可评价性不能覆盖该失败。\n\n"
        "## 下一步\n\n"
        "RIAG v2 必须先原样写入新的外部实验合同，再生成预测和风险分数，最后一次性解封 test truth。"
        "TianKampmann2019 在旧 SafeTrans 归档中已有历史分析，只能作为桥接重分析；不能再写成 untouched independent confirmation。\n"
    )
    atomic_bytes(STAGING / "reports/E167A_REPORT.md", report.encode())
    atomic_bytes(STAGING / "README_先看这个.md", b"# E167a\n\nRead `reports/E167A_REPORT.md` first.\n")

    validate_pretruth_stage(result, head, stage_state)

    manifest_rows = []
    for path in sorted(STAGING.rglob("*")):
        if path.is_symlink():
            raise IntegrityFailure("Symlink found in E167a staging")
        if path.is_file() and path.name not in {"RUN_STATUS.json", "RESULTS_SHA256.csv"}:
            manifest_rows.append({
                "relative_path": path.relative_to(STAGING).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_csv(STAGING / "RESULTS_SHA256.csv", pd.DataFrame(manifest_rows))
    status = {
        "schema": "safeconf_e167a_riag_v2",
        "phase": "complete_postrelease_protocol_correction_not_independent_confirmation",
        "completed_at": now(),
        "git_head_at_formal_run": head,
        "transaction_id": transaction["transaction_id"],
        "raw_expression_files_opened": 0,
        "new_candidate_truth_opened": False,
        "historical_postgate_truth_files_opened_after_snapshot": 2,
        "pretruth_gate_snapshot_sha256": sha256_file(STAGING / "PRETRUTH_GATE_SNAPSHOT.json"),
        "e167_v1_fail_preserved": True,
        "n_historical_units": 19,
        "n_deployment_batches": len(result["batches"]),
        "strict_protocol_regression_passed": result["strict_pass"],
        "deployment_authorized": False,
        "all_gate_checks": {row.check: bool(row.passed) for row in result["gate_summary"].itertuples()},
        "results_manifest_sha256": sha256_file(STAGING / "RESULTS_SHA256.csv"),
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    observed = {path.relative_to(STAGING).as_posix() for path in STAGING.rglob("*") if path.is_file()}
    if observed != ALLOWLIST:
        raise IntegrityFailure(f"E167a allowlist mismatch: {sorted(observed ^ ALLOWLIST)}")
    os.replace(STAGING, RELEASE)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    head = git_head()
    verified_pretruth = verify_pretruth_inputs(head)
    if args.mode == "preflight":
        print(json.dumps({
            "phase": "preflight_passed",
            "git_head": head,
            "n_pretruth_locked_sources_verified": 10,
            "n_postgate_truth_sources_registered_not_opened": 2,
            "e167_v1_fail_preserved": True,
            "raw_expression_files_opened": 0,
        }, ensure_ascii=False, indent=2))
        return
    pretruth = analyze_pretruth(verified_pretruth["payloads"])
    stage_state = write_pretruth_stage(pretruth, verified_pretruth["hashes"], head)
    if not pretruth["pretruth_pass"]:
        raise IntegrityFailure("Pretruth regression failed; postgate truth remains unopened")
    validate_pretruth_stage(pretruth, head, stage_state)
    verified_postgate = verify_postgate_inputs(head, verified_pretruth["frozen_lock"])
    result = analyze_postgate(pretruth, verified_postgate["payloads"])
    revalidate_all_inputs(
        head,
        verified_pretruth["frozen_lock"],
        verified_pretruth["source_lock_sha256"],
    )
    status = finish_release(
        result,
        verified_pretruth["hashes"] + verified_postgate["hashes"],
        head,
        stage_state,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
