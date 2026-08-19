from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _find_record_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("PREDICTION_RECORDS.csv")})


def _array_path_for_record_dir(record_dir: Path) -> Path | None:
    candidates = [
        record_dir / "gears_predicted_effects.npz",
        record_dir.parent / "arrays" / "gears_predicted_effects.npz",
        record_dir.parent.parent / "arrays" / "gears_predicted_effects.npz",
    ]
    return next((p for p in candidates if p.exists()), None)


def _load_prediction_arrays(record_dir: Path, records: pd.DataFrame) -> dict[str, np.ndarray]:
    path = _array_path_for_record_dir(record_dir)
    if path is None:
        return {}
    arrays = np.load(path)
    out: dict[str, np.ndarray] = {}
    for _, row in records.iterrows():
        key = str(row.get("predicted_effect_key", ""))
        rid = str(row["record_id"])
        if key in arrays:
            out[rid] = np.asarray(arrays[key], dtype=np.float32).ravel()
    return out


def load_gears_records(input_root: Path) -> pd.DataFrame:
    frames = []
    for record_dir in _find_record_dirs(input_root):
        rec = pd.read_csv(record_dir / "PREDICTION_RECORDS.csv")
        rec["source_record_dir"] = str(record_dir)
        frames.append(rec)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_native_uncertainty_scores(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if "gears_uncertainty_logvar_mean" not in records.columns:
        return pd.DataFrame()
    for _, row in records.iterrows():
        value = pd.to_numeric(pd.Series([row.get("gears_uncertainty_logvar_mean")]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        rows.append(_score_row(row, "gears_native_uncertainty_risk", "risk", float(value), "NATIVE_UNCERTAINTY_OK"))
    return pd.DataFrame(rows)


def build_seed_ensemble_proxy_scores(input_root: Path, records: pd.DataFrame) -> pd.DataFrame:
    if records.empty or "source_record_dir" not in records.columns:
        return pd.DataFrame()
    array_by_record: dict[str, np.ndarray] = {}
    for record_dir_str, sub in records.groupby("source_record_dir", dropna=False):
        record_dir = Path(str(record_dir_str))
        array_by_record.update(_load_prediction_arrays(record_dir, sub))

    rows: list[dict] = []
    task_cols = ["dataset_name", "perturbation"]
    if "task_key" in records.columns:
        task_cols = ["dataset_name", "task_key"]
    for _, group in records.groupby(task_cols, dropna=False):
        available = [(idx, row, array_by_record.get(str(row["record_id"]))) for idx, row in group.iterrows()]
        available = [(idx, row, arr) for idx, row, arr in available if arr is not None]
        if len(available) < 2:
            continue
        stack = np.stack([arr for _, _, arr in available], axis=0)
        mean = stack.mean(axis=0)
        per_seed_risk = np.sqrt(np.mean((stack - mean[None, :]) ** 2, axis=1))
        for (_, row, _arr), value in zip(available, per_seed_risk):
            rows.append(
                _score_row(
                    row,
                    "gears_seed_ensemble_disagreement_risk",
                    "risk",
                    float(value),
                    "PROXY_NOT_NATIVE_UNCERTAINTY",
                )
            )
    return pd.DataFrame(rows)


def _score_row(row: pd.Series, score_name: str, score_type: str, value: float, status: str) -> dict:
    return {
        "record_id": row["record_id"],
        "dataset_name": row.get("dataset_name", ""),
        "dataset_family": row.get("dataset_family", "gears_supplement"),
        "fold_id": int(row.get("fold_id", -1)) if pd.notna(row.get("fold_id", -1)) else -1,
        "split": row.get("split", "test"),
        "context": row.get("context", ""),
        "perturbation": row.get("perturbation", ""),
        "predictor_name": row.get("predictor_name", "GEARS"),
        "score_name": score_name,
        "score_type": score_type,
        "score_value": value,
        "true_error_rmse": row.get("true_error_rmse", np.nan),
        "uncertainty_status": status,
    }


def export_gears_uncertainty(input_root: Path, out_dir: Path) -> dict:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    records = load_gears_records(input_root)
    records.to_csv(out_dir / "tables" / "GEARS_RECORDS_FOR_UNCERTAINTY.csv", index=False)
    native = build_native_uncertainty_scores(records)
    proxy = build_seed_ensemble_proxy_scores(input_root, records)
    scores = pd.concat([native, proxy], ignore_index=True) if not native.empty or not proxy.empty else pd.DataFrame()
    scores.to_csv(out_dir / "tables" / "GEARS_UNCERTAINTY_SCORES.csv", index=False)
    status = {
        "input_root": str(input_root),
        "out_dir": str(out_dir),
        "n_records": int(len(records)),
        "n_native_scores": int(len(native)),
        "n_proxy_scores": int(len(proxy)),
        "has_native_uncertainty": bool(len(native) > 0),
        "has_seed_ensemble_proxy": bool(len(proxy) > 0),
        "status": _status_label(native, proxy),
    }
    (out_dir / "GEARS_UNCERTAINTY_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# GEARS uncertainty export",
        "",
        f"- input_root: `{input_root}`",
        f"- records: {len(records)}",
        f"- native GEARS uncertainty scores: {len(native)}",
        f"- seed ensemble proxy scores: {len(proxy)}",
        "",
        "Naming rules:",
        "",
        "- `gears_native_uncertainty_risk` means native GEARS logvar/uncertainty was available.",
        "- `gears_seed_ensemble_disagreement_risk` is a multi-seed disagreement proxy, not native GEARS uncertainty.",
        "- prediction magnitude is not exported here as uncertainty.",
    ]
    (out_dir / "reports" / "GEARS_UNCERTAINTY_EXPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return status


def _status_label(native: pd.DataFrame, proxy: pd.DataFrame) -> str:
    if len(native) > 0:
        return "NATIVE_UNCERTAINTY_OK"
    if len(proxy) > 0:
        return "PROXY_NOT_NATIVE_UNCERTAINTY"
    return "MISSING_UNCERTAINTY_AND_PROXY"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export native or proxy GEARS uncertainty scores.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_gears_uncertainty(args.input_root, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

