#!/usr/bin/env python3
"""Build physically separated historical pretruth and postgate assets for E167a.

This is a one-time development-data conversion. It intentionally reads already
unsealed E167 historical assets, then writes prediction/risk inputs and truth to
different committed files so the E167a formal runner can enforce access order.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scripts import run_e167_risk_identifiability_certificate as e167  # noqa: E402


OUT = ROOT / "docs/实验结果/E167a_riag_resolution_correction_20260716/input_assets"
STAGING = OUT.parent / ".input_assets.staging"
E167_LOCK = ROOT / "docs/实验结果/E167_risk_identifiability_certificate_20260716/SOURCE_LOCK.csv"
E167_RUNNER = ROOT / "tools/scripts/run_e167_risk_identifiability_certificate.py"
E167_CONTRACT = ROOT / "docs/实验结果/E167_risk_identifiability_certificate_20260716/ANALYSIS_CONTRACT.md"


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() or STAGING.exists():
        raise RuntimeError(f"Refusing to overwrite isolated assets or staging: {OUT}")
    STAGING.mkdir(parents=True)

    provenance = pd.read_csv(E167_LOCK, dtype=str)
    if len(provenance) != 30 or set(provenance.columns) != {"path", "sha256"}:
        raise RuntimeError("E167 source lock schema/count failed")
    source_errors = []
    for row in provenance.itertuples(index=False):
        path = ROOT / row.path
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != row.sha256:
            source_errors.append({"path": row.path, "expected": row.sha256, "observed": observed})
    if source_errors:
        raise RuntimeError(f"E167 source hash failure: {source_errors[0]}")

    e153 = e167.load_e153_units()
    wessels, _ = e167.load_wessels_units()
    units = e153 + e167.load_e96_units() + e167.load_e159_units() + wessels + e167.load_chemical_units()
    if len(units) != 19:
        raise RuntimeError(f"Expected 19 units, observed {len(units)}")

    registry_rows = []
    pretruth_rows = []
    truth_rows = []
    viability_rows = []
    mapping_rows = []
    prediction_payload: dict[str, np.ndarray] = {}
    for unit_index, unit in enumerate(units):
        predictor_assets = {}
        for predictor_index, (predictor_name, matrix) in enumerate(sorted(unit.predictors.items())):
            key = f"unit_{unit_index:02d}_predictor_{predictor_index:02d}"
            predictor_assets[predictor_name] = key
            prediction_payload[key] = np.asarray(matrix, dtype=np.float64)
        registry_rows.append({
            "unit_id": unit.unit_id,
            "study_id": unit.study_id,
            "lane": unit.lane,
            "endpoint_id": unit.endpoint_id,
            "perturbation_family": unit.perturbation_family,
            "candidate_name": unit.candidate_name,
            "role": unit.role,
            "score_transform": "identity_as_loaded_by_E167_v1",
            "score_numeric_unit": "provider_output_unit_frozen_no_rescale",
            "score_registered_precision": "1e-6",
            "magnitude_transform": "identity_as_loaded_by_E167_v1",
            "magnitude_numeric_unit": "provider_output_unit_frozen_no_rescale",
            "magnitude_registered_precision": "1e-6",
            "prediction_numeric_unit": "effect_vector_provider_unit_frozen_no_rescale",
            "prediction_registered_precision": "1e-6",
            "predictor_assets_json": json.dumps(predictor_assets, ensure_ascii=False, sort_keys=True),
        })
        for task_index, task_id in enumerate(unit.task_ids.astype(str)):
            pretruth_rows.append({
                "unit_id": unit.unit_id,
                "task_id": task_id,
                "cluster": str(unit.clusters[task_index]),
                "candidate_score_pretruth": float(unit.score[task_index]),
                "magnitude_score_pretruth": float(unit.magnitude[task_index]),
            })
            truth_rows.append({
                "unit_id": unit.unit_id,
                "task_id": task_id,
                "registered_truth_loss": float(unit.loss[task_index]),
            })
            for predictor_name, asset_key in predictor_assets.items():
                mapping_rows.append({
                    "unit_id": unit.unit_id,
                    "predictor_name": predictor_name,
                    "asset_key": asset_key,
                    "row_index": task_index,
                    "task_id": task_id,
                })
        viability_rows.append({
            "unit_id": unit.unit_id,
            "predictor_viability_fraction_vs_simple": unit.predictor_viability_fraction,
        })

    atomic_bytes(STAGING / "E167A_UNIT_REGISTRY.csv", pd.DataFrame(registry_rows).to_csv(index=False).encode())
    atomic_bytes(STAGING / "E167A_PRETRUTH_TASKS.csv", pd.DataFrame(pretruth_rows).to_csv(index=False, float_format="%.17g").encode())
    atomic_bytes(STAGING / "E167A_PREDICTION_TASK_MAP.csv", pd.DataFrame(mapping_rows).to_csv(index=False).encode())
    temporary_npz = STAGING / f".E167A_PRETRUTH_PREDICTIONS.{uuid.uuid4().hex}.tmp.npz"
    np.savez_compressed(temporary_npz, **prediction_payload)
    os.replace(temporary_npz, STAGING / "E167A_PRETRUTH_PREDICTIONS.npz")
    atomic_bytes(STAGING / "E167A_POSTGATE_TRUTH.csv", pd.DataFrame(truth_rows).to_csv(index=False, float_format="%.17g").encode())
    atomic_bytes(STAGING / "E167A_POSTGATE_VIABILITY.csv", pd.DataFrame(viability_rows).to_csv(index=False, float_format="%.17g").encode())

    provenance.insert(0, "source_experiment", "E167_v1_historical_development")
    atomic_bytes(STAGING / "E167A_ASSET_PROVENANCE.csv", provenance.to_csv(index=False).encode())

    rebuilt_tasks = pd.read_csv(STAGING / "E167A_PRETRUTH_TASKS.csv")
    original_scores = np.concatenate([unit.score for unit in units])
    original_magnitude = np.concatenate([unit.magnitude for unit in units])
    score_roundtrip = rebuilt_tasks.candidate_score_pretruth.to_numpy(float)
    magnitude_roundtrip = rebuilt_tasks.magnitude_score_pretruth.to_numpy(float)
    score_label_changes = int(np.sum(np.rint(original_scores / 1e-6) != np.rint(score_roundtrip / 1e-6)))
    magnitude_label_changes = int(np.sum(np.rint(original_magnitude / 1e-6) != np.rint(magnitude_roundtrip / 1e-6)))
    prediction_exact = True
    with np.load(STAGING / "E167A_PRETRUTH_PREDICTIONS.npz") as rebuilt_predictions:
        prediction_exact &= set(rebuilt_predictions.files) == set(prediction_payload)
        for key, expected in prediction_payload.items():
            prediction_exact &= bool(np.array_equal(np.asarray(rebuilt_predictions[key]), expected))

    asset_hashes_before_attestation = {
        path.name: sha256_file(path)
        for path in sorted(STAGING.iterdir())
        if path.is_file()
    }
    attestation = {
        "schema": "safeconf_e167a_isolated_asset_attestation_v1",
        "built_at_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "e167_runner_sha256": sha256_file(E167_RUNNER),
        "e167_contract_sha256": sha256_file(E167_CONTRACT),
        "e167_source_lock_sha256": sha256_file(E167_LOCK),
        "n_e167_sources_verified": len(provenance),
        "source_hash_errors": source_errors,
        "n_units": len(units),
        "n_tasks": len(pretruth_rows),
        "n_prediction_arrays": len(prediction_payload),
        "n_prediction_task_map_rows": len(mapping_rows),
        "max_score_csv_roundtrip_abs_difference": float(np.max(np.abs(original_scores - score_roundtrip))),
        "max_magnitude_csv_roundtrip_abs_difference": float(np.max(np.abs(original_magnitude - magnitude_roundtrip))),
        "score_operational_label_changes": score_label_changes,
        "magnitude_operational_label_changes": magnitude_label_changes,
        "prediction_arrays_exact_roundtrip": prediction_exact,
        "asset_sha256_before_attestation": asset_hashes_before_attestation,
    }
    if score_label_changes or magnitude_label_changes or not prediction_exact:
        raise RuntimeError("Isolated asset equivalence failed")
    atomic_bytes(STAGING / "E167A_ASSET_BUILD_ATTESTATION.json", (json.dumps(attestation, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())

    manifest_rows = []
    for path in sorted(STAGING.iterdir()):
        if path.is_file() and path.name != "E167A_ISOLATED_ASSET_MANIFEST.csv":
            manifest_rows.append({
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_bytes(STAGING / "E167A_ISOLATED_ASSET_MANIFEST.csv", pd.DataFrame(manifest_rows).to_csv(index=False).encode())
    os.replace(STAGING, OUT)
    print(pd.DataFrame(manifest_rows).to_string(index=False))


if __name__ == "__main__":
    main()
