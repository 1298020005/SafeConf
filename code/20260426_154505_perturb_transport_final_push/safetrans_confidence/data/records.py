from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.scoring.protocol_v0_2 import assign_dataset_family

MERGE_KEYS = [
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

DEFAULT_PREDICTED_EFFECT_FILES = (
    "input/predicted_effects.npz",
    "arrays/predicted_effects.npz",
    "arrays/gears_predicted_effects.npz",
    "arrays/TAHOE_PREDICTED_EFFECTS_SMOKE.npz",
)
DEFAULT_TRUE_EFFECT_FILES = (
    "input/true_effects.npz",
    "arrays/true_effects.npz",
    "arrays/gears_true_effects.npz",
    "arrays/TAHOE_TRUE_EFFECTS_SMOKE.npz",
)


def load_merged_records(
    input_dir: Path,
    validate_contract: bool = False,
    strict_contract: bool = True,
    require_effect_arrays: bool = True,
) -> pd.DataFrame:
    rec = pd.read_csv(input_dir / "tables" / "PREDICTION_RECORDS.csv")
    if validate_contract:
        issues = validate_prediction_record_artifacts(
            input_dir,
            records=rec,
            strict=strict_contract,
            require_effect_arrays=require_effect_arrays,
        )
        if issues:
            raise ValueError("PredictionRecord contract violations: " + ";".join(issues))
    feat = pd.read_csv(input_dir / "tables" / "CONFIDENCE_FEATURES.csv")
    base = rec.merge(feat, on=MERGE_KEYS, how="left")
    base["dataset_family"] = base["dataset_name"].map(assign_dataset_family)
    return base


def find_effect_array_files(run_dir: Path) -> tuple[Path | None, Path | None]:
    run_dir = Path(run_dir)
    return (
        _find_first_existing(run_dir, DEFAULT_PREDICTED_EFFECT_FILES),
        _find_first_existing(run_dir, DEFAULT_TRUE_EFFECT_FILES),
    )


CONTRACT_REQUIRED_COLUMNS = {
    "schema_version",
    "record_id",
    "dataset_name",
    "dataset_group",
    "context",
    "perturbation",
    "predictor_name",
    "fold_id",
    "split",
    "run_type",
    "gene_panel_id",
    "gene_order_hash",
    "effect_definition",
    "normalization_id",
    "error_normalization",
    "predicted_effect_key",
    "true_effect_key",
    "true_error_rmse",
    "true_error_cosine",
}

CONTRACT_FORBIDDEN_FEATURE_COLUMNS = {
    "true_error_rmse",
    "true_error_cosine",
    "true_effect",
    "true_effect_key",
    "true_effect_l2_norm",
    "true_effect_abs_mean",
    "failure_label",
    "score_value",
}

VALID_RUN_TYPES = {"smoke", "formal"}
VALID_SPLITS = {"train", "val", "test"}
VALID_EFFECT_DEFINITIONS = {"mean_diff", "logFC", "other"}
STRICT_PROVENANCE_COLUMNS = ["gene_panel_id", "gene_order_hash", "normalization_id"]
NONEMPTY_CONTRACT_COLUMNS = [
    "schema_version",
    "record_id",
    "task_key",
    "dataset_name",
    "dataset_group",
    "split",
    "context",
    "perturbation",
    "predictor_name",
    "run_type",
    "gene_panel_id",
    "gene_order_hash",
    "effect_definition",
    "normalization_id",
    "error_normalization",
    "predicted_effect_key",
    "true_effect_key",
]


def add_legacy_contract_defaults(records: pd.DataFrame, run_type: str = "formal") -> pd.DataFrame:
    """Add explicit contract columns to legacy PredictionRecord tables.

    This is only a compatibility shim for old outputs. New predictor adapters
    should write these columns directly so cross-predictor comparisons are
    auditable.
    """
    out = records.copy()
    defaults = {
        "schema_version": "safeconf_prediction_record_v1",
        "dataset_group": out.get("dataset_name", pd.Series(["unknown"] * len(out))).map(
            _default_dataset_group
        ),
        "run_type": run_type,
        "gene_panel_id": "legacy_gene_panel_unknown",
        "gene_order_hash": "legacy_gene_order_unknown",
        "effect_definition": "mean_diff",
        "normalization_id": "legacy_normalization_unknown",
        "error_normalization": "raw_rmse",
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    if "true_error_cosine" not in out.columns:
        out["true_error_cosine"] = pd.NA
    return out


def validate_prediction_record_contract(
    records: pd.DataFrame,
    strict: bool = True,
    predicted_effects: object | None = None,
    true_effects: object | None = None,
) -> list[str]:
    """Return contract violations for PredictionRecord tables.

    In strict mode every contract column must exist. Non-strict mode keeps
    legacy audits usable but still checks available invariants.
    """
    issues: list[str] = []
    missing = sorted(CONTRACT_REQUIRED_COLUMNS.difference(records.columns))
    if strict and missing:
        issues.append("missing_contract_columns=" + ",".join(missing))

    basic_missing = sorted(set(MERGE_KEYS).difference(records.columns))
    if basic_missing:
        issues.append("missing_basic_columns=" + ",".join(basic_missing))

    if strict and "record_id" in records.columns and records["record_id"].duplicated().any():
        issues.append("duplicate_record_id")
    elif (
        not strict
        and "record_id" in records.columns
        and "dataset_name" in records.columns
        and records[["dataset_name", "record_id"]].duplicated().any()
    ):
        issues.append("duplicate_record_id_within_dataset")
    if "true_error_rmse" in records.columns:
        err = pd.to_numeric(records["true_error_rmse"], errors="coerce")
        if err.isna().any():
            issues.append("true_error_rmse_has_nan")

    if "run_type" in records.columns:
        bad = sorted(set(records["run_type"].dropna().astype(str)) - VALID_RUN_TYPES)
        if bad:
            issues.append("invalid_run_type=" + ",".join(bad))
    if "split" in records.columns:
        bad = sorted(set(records["split"].dropna().astype(str)) - VALID_SPLITS)
        if bad:
            issues.append("invalid_split=" + ",".join(bad))
    if "effect_definition" in records.columns:
        bad = sorted(set(records["effect_definition"].dropna().astype(str)) - VALID_EFFECT_DEFINITIONS)
        if bad:
            issues.append("invalid_effect_definition=" + ",".join(bad))

    empty_required = [
        col
        for col in NONEMPTY_CONTRACT_COLUMNS
        if col in records.columns and _has_empty_values(records[col])
    ]
    if empty_required:
        issues.append("empty_required_values=" + ",".join(empty_required))

    if strict:
        unknown_provenance = [
            col
            for col in STRICT_PROVENANCE_COLUMNS
            if col in records.columns and records[col].map(_is_unknown_contract_value).any()
        ]
        if unknown_provenance:
            issues.append("unknown_contract_provenance=" + ",".join(unknown_provenance))

    duplicate_key_cols = [
        "dataset_name",
        "fold_id",
        "split",
        "task_key",
        "predictor_name",
    ]
    if (
        set(duplicate_key_cols).issubset(records.columns)
        and records[duplicate_key_cols].duplicated().any()
    ):
        issues.append("duplicate_task_predictor_rows")

    compare_cols = [
        "dataset_name",
        "fold_id",
        "split",
        "task_key",
        "context",
        "perturbation",
        "gene_panel_id",
        "gene_order_hash",
        "effect_definition",
        "normalization_id",
        "true_effect_key",
    ]
    legacy_record_scoped_keys = (
        not strict
        and "schema_version" in records.columns
        and records["schema_version"].astype(str).str.contains("safeconf_prediction_record_v1").all()
    )
    if set(compare_cols).issubset(records.columns) and not legacy_record_scoped_keys:
        group_cols = ["dataset_name", "fold_id", "split", "task_key", "context", "perturbation"]
        for group_key, group in records.groupby(group_cols, dropna=False):
            if group["predictor_name"].nunique(dropna=False) < 2:
                continue
            for col in [
                "gene_panel_id",
                "gene_order_hash",
                "effect_definition",
                "normalization_id",
                "true_effect_key",
            ]:
                if group[col].nunique(dropna=False) > 1:
                    issues.append(f"inconsistent_{col}_for_task={group_key}")
                    break
    elif legacy_record_scoped_keys:
        issues.append("legacy_true_effect_key_record_scoped_check_skipped")
    issues.extend(_validate_effect_array_contract(records, predicted_effects, true_effects))
    return issues


def validate_prediction_record_artifacts(
    run_dir: Path,
    records: pd.DataFrame | None = None,
    strict: bool = True,
    require_effect_arrays: bool = True,
) -> list[str]:
    run_dir = Path(run_dir)
    if records is None:
        records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    issues: list[str] = []
    predicted_path, true_path = find_effect_array_files(run_dir)
    if predicted_path is None:
        issues.append("missing_predicted_effect_array_file")
    if true_path is None:
        issues.append("missing_true_effect_array_file")
    if predicted_path is None or true_path is None:
        if not require_effect_arrays:
            issues = []
        return issues + validate_prediction_record_contract(records, strict=strict)
    with np.load(predicted_path) as predicted_effects, np.load(true_path) as true_effects:
        return validate_prediction_record_contract(
            records,
            strict=strict,
            predicted_effects=predicted_effects,
            true_effects=true_effects,
        )


def assert_no_feature_label_leakage(feature_cols: list[str]) -> None:
    bad = sorted(set(feature_cols) & CONTRACT_FORBIDDEN_FEATURE_COLUMNS)
    if bad:
        raise ValueError(f"label/leakage columns are not allowed as confidence features: {bad}")


def _default_dataset_group(dataset_name: object) -> str:
    name = str(dataset_name)
    if name in {"KaggleCrossCell", "KaggleCrossPatient"}:
        return "kaggle_chem_group"
    if name == "crossPatient":
        return "sparse_cross_patient_group"
    if name in {"Norman", "Adamson", "Dixit", "norman", "adamson", "dixit"}:
        return "gears_crispr_group"
    if name in {"Haber", "Parekh", "Frangieh"}:
        return "gene_public_candidate"
    return "unknown_group"


def _has_empty_values(values: pd.Series) -> bool:
    return values.isna().any() or values.astype(str).str.strip().eq("").any()


def _is_unknown_contract_value(value: object) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return (
        text == ""
        or text in {"unknown", "na", "nan", "none", "null"}
        or "legacy_" in text
        or "unknown" in text
    )


def _validate_effect_array_contract(
    records: pd.DataFrame,
    predicted_effects: object | None,
    true_effects: object | None,
) -> list[str]:
    issues: list[str] = []
    if predicted_effects is None and true_effects is None:
        return issues
    if predicted_effects is None:
        return ["missing_array_store=predicted_effects"]
    if true_effects is None:
        return ["missing_array_store=true_effects"]
    if not {"record_id", "predicted_effect_key", "true_effect_key"}.issubset(records.columns):
        return issues

    predicted_keys = _array_store_keys(predicted_effects)
    true_keys = _array_store_keys(true_effects)
    wanted_predicted = set(records["predicted_effect_key"].dropna().astype(str))
    wanted_true = set(records["true_effect_key"].dropna().astype(str))
    missing_predicted = sorted(wanted_predicted - predicted_keys)
    missing_true = sorted(wanted_true - true_keys)
    if missing_predicted:
        issues.append("missing_predicted_effect_arrays=" + _format_issue_values(missing_predicted))
    if missing_true:
        issues.append("missing_true_effect_arrays=" + _format_issue_values(missing_true))

    invalid_shapes: list[str] = []
    shape_mismatches: list[str] = []
    for row in records.to_dict("records"):
        record_id = str(row["record_id"])
        predicted_key = str(row["predicted_effect_key"])
        true_key = str(row["true_effect_key"])
        if predicted_key not in predicted_keys or true_key not in true_keys:
            continue
        predicted = np.asarray(predicted_effects[predicted_key])
        true = np.asarray(true_effects[true_key])
        predicted_shape = predicted.shape
        true_shape = true.shape
        if predicted.ndim != 1 or predicted.size == 0:
            invalid_shapes.append(
                f"{record_id}:predicted_effect_key={predicted_key}:shape={predicted_shape}"
            )
        if true.ndim != 1 or true.size == 0:
            invalid_shapes.append(f"{record_id}:true_effect_key={true_key}:shape={true_shape}")
        if predicted_shape != true_shape:
            shape_mismatches.append(
                f"{record_id}:predicted_shape={predicted_shape}:true_shape={true_shape}"
            )
    if invalid_shapes:
        issues.append("invalid_effect_array_shape=" + _format_issue_values(invalid_shapes))
    if shape_mismatches:
        issues.append("effect_array_shape_mismatch=" + _format_issue_values(shape_mismatches))
    return issues


def _array_store_keys(array_store: object) -> set[str]:
    if hasattr(array_store, "keys"):
        return {str(key) for key in array_store.keys()}
    if hasattr(array_store, "files"):
        return {str(key) for key in array_store.files}
    raise TypeError("array stores must expose keys() or files")


def _format_issue_values(values: list[str], max_values: int = 5) -> str:
    shown = values[:max_values]
    suffix = [] if len(values) <= max_values else [f"+{len(values) - max_values}more"]
    return ",".join(shown + suffix)


def _find_first_existing(root: Path, rel_paths: tuple[str, ...]) -> Path | None:
    return next((root / rel for rel in rel_paths if (root / rel).exists()), None)
