#!/usr/bin/env python3
"""Compute SafeConf features from existing PredictionRecord arrays.

This is a large-dataset recovery path: if the legacy MVP runner has already
written predictions but feature computation is too slow, this script rebuilds
the train-only features needed by protocol v0.2 without re-running predictors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.data.records import validate_prediction_record_contract
from safetrans_confidence.features.schema import build_feature_provenance_table

IDENTITY_COLUMNS = [
    "record_id",
    "task_id",
    "task_key",
    "dataset_name",
    "fold_id",
    "split",
    "context",
    "perturbation",
    "predictor_name",
]
BLIND_REQUIRED_COLUMNS = IDENTITY_COLUMNS + ["predicted_effect_key", "target_control_key"]
BLIND_PRIMARY_FEATURE_COLUMNS = [
    "context_similarity_max",
    "perturbation_support_count",
    "model_disagreement_rmse",
]
EXPECTED_DISAGREEMENT_PREDICTORS = ("V0StrongBaseline", "ContextSimBaseline")
EVALUATION_DIAGNOSTIC_COLUMNS = [
    "context_similarity_mean",
    "perturbation_effect_stability",
    "perturbation_effect_variance",
    "historical_residual_risk",
    "prediction_l2_norm",
    "prediction_abs_mean",
    "fold_train_median_effect_norm",
    "prediction_norm_ratio",
    "prediction_magnitude_deviation",
    "model_disagreement_cosine",
    "ood_nearest_distance",
    "ood_mean_k_distance",
]
VALID_SPLITS = {"train", "val", "test"}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    val = _cosine(a, b)
    return float(1.0 - val) if np.isfinite(val) else float("nan")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    den = np.linalg.norm(x, axis=1, keepdims=True)
    den[den <= 1e-12] = 1.0
    return x / den


def _mean_pairwise_cosine(effects: list[np.ndarray]) -> float:
    if len(effects) < 2:
        return float("nan")
    mat = _normalize_rows(np.stack(effects, axis=0))
    sims = mat @ mat.T
    tri = sims[np.triu_indices(len(effects), k=1)]
    tri = tri[np.isfinite(tri)]
    return float(np.mean(tri)) if len(tri) else float("nan")


def _load_arrays(run_dir: Path) -> tuple[dict, dict, dict]:
    pred = np.load(run_dir / "input" / "predicted_effects.npz")
    true = np.load(run_dir / "input" / "true_effects.npz")
    ctrl = np.load(run_dir / "input" / "target_control_means.npz")
    return pred, true, ctrl


def _load_blind_arrays(run_dir: Path) -> tuple[dict, dict]:
    pred = np.load(run_dir / "input" / "predicted_effects.npz")
    ctrl = np.load(run_dir / "input" / "target_control_means.npz")
    return pred, ctrl


def _validate_contract_or_raise(
    records: pd.DataFrame,
    predicted_effects: object,
    true_effects: object,
    strict: bool,
) -> None:
    issues = validate_prediction_record_contract(
        records,
        strict=strict,
        predicted_effects=predicted_effects,
        true_effects=true_effects,
    )
    if issues:
        mode = "strict" if strict else "legacy-compatible"
        raise ValueError(f"PredictionRecord contract failed in {mode} mode: {'; '.join(issues)}")


def _validate_blind_inputs_or_raise(
    records: pd.DataFrame,
    predicted_effects: object,
    control_arrays: object,
    strict: bool,
) -> None:
    issues: list[str] = []
    missing = sorted(set(BLIND_REQUIRED_COLUMNS).difference(records.columns))
    if missing:
        issues.append("missing_blind_columns=" + ",".join(missing))
    if missing:
        raise ValueError("Blind primary feature contract failed: " + "; ".join(issues))

    required_nonempty = [
        "record_id",
        "task_id",
        "task_key",
        "dataset_name",
        "fold_id",
        "split",
        "context",
        "perturbation",
        "predictor_name",
        "predicted_effect_key",
        "target_control_key",
    ]
    empty = [col for col in required_nonempty if _has_empty_values(records[col])]
    if empty:
        issues.append("empty_blind_values=" + ",".join(empty))
    bad_splits = sorted(set(records["split"].dropna().astype(str)) - VALID_SPLITS)
    if bad_splits:
        issues.append("invalid_split=" + ",".join(bad_splits))
    if records["record_id"].duplicated().any():
        issues.append("duplicate_record_id")
    duplicate_key_cols = ["dataset_name", "fold_id", "split", "task_key", "predictor_name"]
    if records[duplicate_key_cols].duplicated().any():
        issues.append("duplicate_task_predictor_rows")
    if strict and records[["dataset_name", "fold_id", "split", "task_key"]].isna().any().any():
        issues.append("strict_blind_task_identity_has_nan")

    predicted_keys = _array_keys(predicted_effects)
    control_keys = _array_keys(control_arrays)
    missing_predicted = sorted(set(records["predicted_effect_key"].astype(str)) - predicted_keys)
    missing_control = sorted(set(records["target_control_key"].astype(str)) - control_keys)
    if missing_predicted:
        issues.append("missing_predicted_effect_arrays=" + _format_issue_values(missing_predicted))
    if missing_control:
        issues.append("missing_target_control_arrays=" + _format_issue_values(missing_control))

    invalid_shapes: list[str] = []
    for row in records.to_dict("records"):
        record_id = str(row["record_id"])
        pred_key = str(row["predicted_effect_key"])
        ctrl_key = str(row["target_control_key"])
        if pred_key in predicted_keys:
            pred = np.asarray(predicted_effects[pred_key])
            if pred.ndim != 1 or pred.size == 0:
                invalid_shapes.append(f"{record_id}:predicted_effect_key={pred_key}:shape={pred.shape}")
        if ctrl_key in control_keys:
            ctrl = np.asarray(control_arrays[ctrl_key])
            if ctrl.ndim != 1 or ctrl.size == 0:
                invalid_shapes.append(f"{record_id}:target_control_key={ctrl_key}:shape={ctrl.shape}")
    if invalid_shapes:
        issues.append("invalid_blind_array_shape=" + _format_issue_values(invalid_shapes))
    if issues:
        mode = "strict" if strict else "legacy-compatible"
        raise ValueError(f"Blind primary feature contract failed in {mode} mode: {'; '.join(issues)}")


def _historical_residual(task_table: pd.DataFrame, true_arrays: dict) -> tuple[dict[str, float], float]:
    out: dict[str, float] = {}
    all_errs: list[float] = []
    for pert, group in task_table.groupby("perturbation", dropna=False):
        rows = group.to_dict("records")
        errs = []
        for row in rows:
            src = [r for r in rows if r["context"] != row["context"]]
            if not src:
                continue
            pred = np.mean([true_arrays[r["true_effect_key"]] for r in src], axis=0)
            err = _rmse(pred, true_arrays[row["true_effect_key"]])
            errs.append(err)
            all_errs.append(err)
        if errs:
            out[str(pert)] = float(np.median(errs))
    fallback = float(np.median(all_errs)) if all_errs else float("nan")
    return out, fallback


def _disagreement(rec: pd.DataFrame, pred_arrays: dict) -> dict[tuple, tuple[float, float]]:
    out: dict[tuple, tuple[float, float]] = {}
    group_cols = ["dataset_name", "fold_id", "split", "task_id"]
    for key, group in rec.groupby(group_cols, dropna=False):
        by_name = {
            str(row["predictor_name"]): pred_arrays[str(row["predicted_effect_key"])]
            for row in group.to_dict("records")
        }
        if any(predictor not in by_name for predictor in EXPECTED_DISAGREEMENT_PREDICTORS):
            continue
        a = by_name[EXPECTED_DISAGREEMENT_PREDICTORS[0]]
        b = by_name[EXPECTED_DISAGREEMENT_PREDICTORS[1]]
        out[key] = (_rmse(a, b), _cosine_distance(a, b))
    return out


def audit_disagreement_predictor_sets(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["dataset_name", "fold_id", "split", "task_id", "task_key"]
    missing_cols = sorted(set(group_cols + ["predictor_name"]).difference(records.columns))
    if missing_cols:
        return pd.DataFrame(
            [
                {
                    "status": "failed",
                    "missing_columns": ",".join(missing_cols),
                    "expected_predictors": ",".join(EXPECTED_DISAGREEMENT_PREDICTORS),
                }
            ]
        )
    for key, group in records.groupby(group_cols, dropna=False):
        predictors = sorted(group["predictor_name"].dropna().astype(str).unique())
        missing = sorted(set(EXPECTED_DISAGREEMENT_PREDICTORS) - set(predictors))
        rows.append(
            {
                "dataset_name": key[0],
                "fold_id": int(key[1]),
                "split": key[2],
                "task_id": int(key[3]),
                "task_key": key[4],
                "expected_predictors": ",".join(EXPECTED_DISAGREEMENT_PREDICTORS),
                "observed_predictors": ",".join(predictors),
                "missing_predictors": ",".join(missing),
                "status": "ok" if not missing else "missing_expected_predictor",
            }
        )
    return pd.DataFrame(rows)


def _validate_disagreement_predictor_sets_or_raise(records: pd.DataFrame, strict: bool) -> None:
    if not strict:
        return
    audit = audit_disagreement_predictor_sets(records)
    if audit.empty or "status" not in audit.columns:
        raise ValueError("Disagreement predictor-set audit failed: empty audit")
    bad = audit[~audit["status"].eq("ok")]
    if bad.empty:
        return
    examples = []
    for row in bad.head(5).to_dict("records"):
        if row.get("status") == "failed":
            examples.append(f"missing_columns={row.get('missing_columns', '')}")
            continue
        examples.append(
            f"{row.get('dataset_name')}|fold={row.get('fold_id')}|split={row.get('split')}|"
            f"task={row.get('task_key')}|missing={row.get('missing_predictors')}"
        )
    suffix = "" if len(bad) <= 5 else f"; +{len(bad) - 5} more"
    raise ValueError("Missing expected disagreement predictor sets: " + "; ".join(examples) + suffix)


def compute_blind_primary_features_from_records(
    records: pd.DataFrame,
    pred_arrays: object,
    ctrl_arrays: object,
    strict_contract: bool = True,
) -> pd.DataFrame:
    _validate_blind_inputs_or_raise(records, pred_arrays, ctrl_arrays, strict=strict_contract)
    _validate_disagreement_predictor_sets_or_raise(records, strict=strict_contract)
    dis = _disagreement(records, pred_arrays)
    rows: list[dict] = []

    for (dataset, fold_id), fold_records in records.groupby(["dataset_name", "fold_id"], dropna=False):
        task_table = (
            fold_records.sort_values("predictor_name")
            .drop_duplicates("task_id")
            .reset_index(drop=True)
            .copy()
        )
        train_tasks = task_table[task_table["split"].eq("train")].copy()
        if train_tasks.empty:
            raise RuntimeError(f"No train tasks for {dataset} fold {fold_id}")

        train_controls = np.stack(
            [ctrl_arrays[str(row["target_control_key"])] for row in train_tasks.to_dict("records")],
            axis=0,
        )
        train_controls_norm = _normalize_rows(train_controls)
        train_contexts = train_tasks["context"].astype(str).to_numpy()

        by_pert: dict[str, list[dict]] = {}
        for row in train_tasks.to_dict("records"):
            by_pert.setdefault(str(row["perturbation"]), []).append(row)

        task_feature_cache: dict[int, dict] = {}
        for task in task_table.to_dict("records"):
            task_id = int(task["task_id"])
            target_control = ctrl_arrays[str(task["target_control_key"])]
            target_control_norm = target_control.astype(np.float64)
            den = np.linalg.norm(target_control_norm)
            target_control_norm = target_control_norm / den if den > 1e-12 else target_control_norm

            other_ctx = train_contexts != str(task["context"])
            if not other_ctx.any():
                other_ctx = np.ones(len(train_contexts), dtype=bool)
            sims = train_controls_norm[other_ctx] @ target_control_norm
            sims = sims[np.isfinite(sims)]
            same_pert_rows = [
                row for row in by_pert.get(str(task["perturbation"]), [])
                if str(row["context"]) != str(task["context"])
            ]
            task_feature_cache[task_id] = {
                "context_similarity_max": float(np.nanmax(sims)) if len(sims) else float("nan"),
                "perturbation_support_count": int(len({str(row["context"]) for row in same_pert_rows})),
            }

        for rec in fold_records.to_dict("records"):
            key = (dataset, fold_id, rec["split"], int(rec["task_id"]))
            dis_rmse, _dis_cos = dis.get(key, (float("nan"), float("nan")))
            rows.append(
                {
                    "record_id": rec["record_id"],
                    "task_id": int(rec["task_id"]),
                    "task_key": rec["task_key"],
                    "dataset_name": dataset,
                    "fold_id": int(fold_id),
                    "split": rec["split"],
                    "context": rec["context"],
                    "perturbation": rec["perturbation"],
                    "predictor_name": rec["predictor_name"],
                    **task_feature_cache[int(rec["task_id"])],
                    "model_disagreement_rmse": dis_rmse,
                }
            )
    return pd.DataFrame(rows)


def compute_evaluation_diagnostics_from_records(
    records: pd.DataFrame,
    pred_arrays: object,
    true_arrays: object,
    ctrl_arrays: object,
) -> pd.DataFrame:
    missing = sorted({"true_effect_key"}.difference(records.columns))
    if missing:
        raise ValueError("Missing evaluation diagnostic columns: " + ",".join(missing))
    dis = _disagreement(records, pred_arrays)
    rows: list[dict] = []

    for (dataset, fold_id), fold_records in records.groupby(["dataset_name", "fold_id"], dropna=False):
        task_table = (
            fold_records.sort_values("predictor_name")
            .drop_duplicates("task_id")
            .reset_index(drop=True)
            .copy()
        )
        train_tasks = task_table[task_table["split"].eq("train")].copy()
        if train_tasks.empty:
            raise RuntimeError(f"No train tasks for {dataset} fold {fold_id}")

        train_controls = np.stack(
            [ctrl_arrays[str(row["target_control_key"])] for row in train_tasks.to_dict("records")],
            axis=0,
        )
        train_controls_norm = _normalize_rows(train_controls)
        train_effects = {
            int(row["task_id"]): true_arrays[str(row["true_effect_key"])]
            for row in train_tasks.to_dict("records")
        }
        train_effect_norms = [float(np.linalg.norm(v)) for v in train_effects.values()]
        median_effect_norm = float(np.nanmedian(train_effect_norms)) if train_effect_norms else float("nan")
        train_contexts = train_tasks["context"].astype(str).to_numpy()
        historical, historical_fallback = _historical_residual(train_tasks, true_arrays)

        by_pert: dict[str, list[dict]] = {}
        for row in train_tasks.to_dict("records"):
            by_pert.setdefault(str(row["perturbation"]), []).append(row)

        task_feature_cache: dict[int, dict] = {}
        for task in task_table.to_dict("records"):
            task_id = int(task["task_id"])
            target_control = ctrl_arrays[str(task["target_control_key"])]
            target_control_norm = target_control.astype(np.float64)
            den = np.linalg.norm(target_control_norm)
            target_control_norm = target_control_norm / den if den > 1e-12 else target_control_norm

            other_ctx = train_contexts != str(task["context"])
            if not other_ctx.any():
                other_ctx = np.ones(len(train_contexts), dtype=bool)
            sims = train_controls_norm[other_ctx] @ target_control_norm
            sims = sims[np.isfinite(sims)]

            same_pert_rows = [
                row for row in by_pert.get(str(task["perturbation"]), [])
                if str(row["context"]) != str(task["context"])
            ]
            source_effects = [true_arrays[str(row["true_effect_key"])] for row in same_pert_rows]
            if len(source_effects) >= 2:
                effect_stack = np.stack(source_effects, axis=0)
                effect_var = float(np.mean(np.var(effect_stack, axis=0)))
                stability = _mean_pairwise_cosine(source_effects)
            else:
                effect_var = float("nan")
                stability = float("nan")

            task_feature_cache[task_id] = {
                "context_similarity_mean": float(np.nanmean(sims)) if len(sims) else float("nan"),
                "perturbation_effect_stability": stability,
                "perturbation_effect_variance": effect_var,
                "historical_residual_risk": historical.get(str(task["perturbation"]), historical_fallback),
            }

        for rec in fold_records.to_dict("records"):
            pred = pred_arrays[str(rec["predicted_effect_key"])]
            pred_norm = float(np.linalg.norm(pred))
            norm_ratio = pred_norm / (median_effect_norm + 1e-8) if np.isfinite(median_effect_norm) else float("nan")
            key = (dataset, fold_id, rec["split"], int(rec["task_id"]))
            _dis_rmse, dis_cos = dis.get(key, (float("nan"), float("nan")))
            rows.append(
                {
                    "record_id": rec["record_id"],
                    **task_feature_cache[int(rec["task_id"])],
                    "prediction_l2_norm": pred_norm,
                    "prediction_abs_mean": float(np.mean(np.abs(pred))),
                    "fold_train_median_effect_norm": median_effect_norm,
                    "prediction_norm_ratio": float(norm_ratio),
                    "prediction_magnitude_deviation": float(abs(math.log(norm_ratio + 1e-8)))
                    if np.isfinite(norm_ratio)
                    else float("nan"),
                    "model_disagreement_cosine": dis_cos,
                    "ood_nearest_distance": float("nan"),
                    "ood_mean_k_distance": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def compute_features_from_records(
    records: pd.DataFrame,
    pred_arrays: object,
    true_arrays: object,
    ctrl_arrays: object,
    strict_contract: bool = True,
) -> pd.DataFrame:
    _validate_contract_or_raise(records, pred_arrays, true_arrays, strict=strict_contract)
    primary = compute_blind_primary_features_from_records(
        records,
        pred_arrays,
        ctrl_arrays,
        strict_contract=strict_contract,
    )
    diagnostics = compute_evaluation_diagnostics_from_records(records, pred_arrays, true_arrays, ctrl_arrays)
    return primary.merge(diagnostics, on="record_id", how="left")


def compute_blind_primary_features(run_dir: Path, strict_contract: bool = True) -> pd.DataFrame:
    records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    pred_arrays, ctrl_arrays = _load_blind_arrays(run_dir)
    return compute_blind_primary_features_from_records(
        records,
        pred_arrays,
        ctrl_arrays,
        strict_contract=strict_contract,
    )


def compute_features(run_dir: Path, strict_contract: bool = True) -> pd.DataFrame:
    records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    pred_arrays, true_arrays, ctrl_arrays = _load_arrays(run_dir)
    return compute_features_from_records(records, pred_arrays, true_arrays, ctrl_arrays, strict_contract=strict_contract)


def _array_keys(array_store: object) -> set[str]:
    if hasattr(array_store, "keys"):
        return {str(key) for key in array_store.keys()}
    if hasattr(array_store, "files"):
        return {str(key) for key in array_store.files}
    raise TypeError("array stores must expose keys() or files")


def _has_empty_values(values: pd.Series) -> bool:
    return values.isna().any() or values.astype(str).str.strip().eq("").any()


def _format_issue_values(values: list[str], max_values: int = 5) -> str:
    shown = values[:max_values]
    suffix = [] if len(values) <= max_values else [f"+{len(values) - max_values}more"]
    return ",".join(shown + suffix)


def run(run_dir: Path, strict_contract: bool = True) -> dict:
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    features = compute_features(run_dir, strict_contract=strict_contract)
    features.to_csv(run_dir / "tables" / "CONFIDENCE_FEATURES.csv", index=False)
    records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    disagreement_status = audit_disagreement_predictor_sets(records)
    disagreement_status.to_csv(run_dir / "tables" / "DISAGREEMENT_PREDICTOR_SET_STATUS.csv", index=False)

    missing = features.isna().mean(numeric_only=False).sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_rate"]
    missing.to_csv(run_dir / "tables" / "CONFIDENCE_FEATURE_MISSINGNESS.csv", index=False)
    provenance = build_feature_provenance_table(features)
    provenance.to_csv(run_dir / "tables" / "FEATURE_PROVENANCE.csv", index=False)
    desc = features.select_dtypes(include=[np.number]).describe().T.reset_index().rename(columns={"index": "feature"})
    desc.to_csv(run_dir / "tables" / "CONFIDENCE_FEATURE_DESCRIBE.csv", index=False)

    lines = [
        "# Fast feature scoring report",
        "",
        "This report was generated from existing PredictionRecord arrays without re-running predictors.",
        "",
        "The fast path computes protocol v0.2 features and skips expensive OOD nearest-neighbor distance.",
        "",
        f"- feature_rows: {len(features)}",
        f"- datasets: {', '.join(sorted(features['dataset_name'].astype(str).unique()))}",
        "",
        "## Missingness",
        "",
        "```",
        missing.to_string(index=False),
        "```",
    ]
    (run_dir / "reports" / "fast_feature_scoring_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    status = {
        "run_dir": str(run_dir),
        "feature_rows": int(len(features)),
        "status": "ok",
        "contract_mode": "strict" if strict_contract else "legacy-compatible",
        "feature_mode": "legacy_combined_features_and_evaluation_diagnostics",
        "disagreement_predictor_set_status": (
            "ok" if disagreement_status["status"].eq("ok").all() else "has_missing_expected_predictors"
        ),
        "note": "OOD distance intentionally set to NaN in fast path; frozen protocol v0.2 main score does not require it.",
    }
    (run_dir / "FAST_FEATURE_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def run_blind_primary(run_dir: Path, strict_contract: bool = True) -> dict:
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    features = compute_blind_primary_features(run_dir, strict_contract=strict_contract)
    out_path = run_dir / "tables" / "CONFIDENCE_FEATURES_BLIND_PRIMARY.csv"
    features.to_csv(out_path, index=False)
    records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    disagreement_status = audit_disagreement_predictor_sets(records)
    disagreement_status.to_csv(
        run_dir / "tables" / "DISAGREEMENT_PREDICTOR_SET_STATUS_BLIND_PRIMARY.csv",
        index=False,
    )

    missing = features.isna().mean(numeric_only=False).sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_rate"]
    missing.to_csv(run_dir / "tables" / "CONFIDENCE_FEATURE_MISSINGNESS_BLIND_PRIMARY.csv", index=False)
    provenance = build_feature_provenance_table(features)
    provenance.to_csv(run_dir / "tables" / "FEATURE_PROVENANCE_BLIND_PRIMARY.csv", index=False)

    lines = [
        "# Blind primary feature scoring report",
        "",
        "This report was generated without loading held-out true-effect arrays or true-error labels.",
        "",
        "The output contains only the identity columns and the frozen v0.2 primary features.",
        "",
        f"- feature_rows: {len(features)}",
        f"- output: `{out_path}`",
        f"- datasets: {', '.join(sorted(features['dataset_name'].astype(str).unique()))}",
        "",
        "## Missingness",
        "",
        "```",
        missing.to_string(index=False),
        "```",
    ]
    (run_dir / "reports" / "blind_primary_feature_scoring_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    status = {
        "run_dir": str(run_dir),
        "feature_rows": int(len(features)),
        "status": "ok",
        "contract_mode": "strict" if strict_contract else "legacy-compatible",
        "feature_mode": "blind_primary_only",
        "output": str(out_path),
        "disagreement_predictor_set_status": (
            "ok" if disagreement_status["status"].eq("ok").all() else "has_missing_expected_predictors"
        ),
        "note": "This path does not load true_effects.npz and emits only frozen primary scoring features.",
    }
    (run_dir / "FAST_BLIND_PRIMARY_FEATURE_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast SafeConf feature computation from existing arrays.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy-contract",
        action="store_true",
        help="Use non-strict contract mode for legacy audits; formal evidence should omit this flag.",
    )
    parser.add_argument(
        "--blind-primary-only",
        action="store_true",
        help="Compute only blind primary features without loading true_effects.npz.",
    )
    args = parser.parse_args()
    if args.blind_primary_only:
        status = run_blind_primary(args.run_dir, strict_contract=not args.allow_legacy_contract)
    else:
        status = run(args.run_dir, strict_contract=not args.allow_legacy_contract)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
