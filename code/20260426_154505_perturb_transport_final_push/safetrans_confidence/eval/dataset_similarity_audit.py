from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.data.records import _default_dataset_group


DEFAULT_RECORD_FILES = ["tables/PREDICTION_RECORDS.csv", "input/PREDICTION_RECORDS.csv"]


def _find_record_file(run_dir: Path) -> Path | None:
    return next((run_dir / rel for rel in DEFAULT_RECORD_FILES if (run_dir / rel).exists()), None)


def load_records(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = _find_record_file(run_dir)
        if path is None:
            continue
        df = pd.read_csv(path)
        df["source_run_dir"] = str(run_dir)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    records = pd.concat(frames, ignore_index=True)
    if "dataset_group" not in records.columns:
        records["dataset_group"] = records["dataset_name"].map(_default_dataset_group)
    return records


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    return float(len(a & b) / denom) if denom else 0.0


def build_dataset_groups(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in records.groupby("dataset_name", dropna=False):
        group = str(g["dataset_group"].dropna().iloc[0]) if "dataset_group" in g and g["dataset_group"].notna().any() else _default_dataset_group(dataset)
        rows.append(
            {
                "dataset_name": dataset,
                "dataset_group": group,
                "n_records": int(len(g)),
                "n_test_records": int((g.get("split", pd.Series(dtype=str)) == "test").sum()),
                "n_contexts": int(g["context"].nunique(dropna=True)) if "context" in g else 0,
                "n_perturbations": int(g["perturbation"].nunique(dropna=True)) if "perturbation" in g else 0,
                "n_predictors": int(g["predictor_name"].nunique(dropna=True)) if "predictor_name" in g else 0,
                "provenance_note": _provenance_note(dataset, group),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_group", "dataset_name"])


def build_pairwise_similarity(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_dataset = {name: g for name, g in records.groupby("dataset_name", dropna=False)}
    for a, b in combinations(sorted(by_dataset), 2):
        ga = by_dataset[a]
        gb = by_dataset[b]
        pa = set(ga["perturbation"].dropna().astype(str)) if "perturbation" in ga else set()
        pb = set(gb["perturbation"].dropna().astype(str)) if "perturbation" in gb else set()
        ca = set(ga["context"].dropna().astype(str)) if "context" in ga else set()
        cb = set(gb["context"].dropna().astype(str)) if "context" in gb else set()
        pra = set(ga["predictor_name"].dropna().astype(str)) if "predictor_name" in ga else set()
        prb = set(gb["predictor_name"].dropna().astype(str)) if "predictor_name" in gb else set()
        group_a = str(ga["dataset_group"].dropna().iloc[0]) if "dataset_group" in ga and ga["dataset_group"].notna().any() else _default_dataset_group(a)
        group_b = str(gb["dataset_group"].dropna().iloc[0]) if "dataset_group" in gb and gb["dataset_group"].notna().any() else _default_dataset_group(b)
        rows.append(
            {
                "dataset_a": a,
                "dataset_b": b,
                "dataset_group_a": group_a,
                "dataset_group_b": group_b,
                "same_dataset_group": bool(group_a == group_b),
                "perturbation_jaccard": _jaccard(pa, pb),
                "context_jaccard": _jaccard(ca, cb),
                "predictor_jaccard": _jaccard(pra, prb),
                "n_shared_perturbations": int(len(pa & pb)),
                "n_shared_contexts": int(len(ca & cb)),
            }
        )
    return pd.DataFrame(rows)


def run_audit(run_dirs: list[Path], out_dir: Path) -> dict:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    records = load_records(run_dirs)
    if records.empty:
        raise RuntimeError("No PredictionRecord tables found for dataset similarity audit.")
    groups = build_dataset_groups(records)
    pairwise = build_pairwise_similarity(records)
    records.to_csv(out_dir / "tables" / "DATASET_AUDIT_RECORDS.csv", index=False)
    groups.to_csv(out_dir / "tables" / "DATASET_SIMILARITY_GROUPS.csv", index=False)
    pairwise.to_csv(out_dir / "tables" / "DATASET_PAIRWISE_SIMILARITY.csv", index=False)
    report = [
        "# Dataset similarity audit",
        "",
        "This audit defines dataset groups before leave-one-dataset-group-out validation.",
        "",
        "## Dataset groups",
        "",
        "```",
        groups.to_string(index=False),
        "```",
        "",
        "## Pairwise similarity preview",
        "",
        "```",
        pairwise.head(40).to_string(index=False) if not pairwise.empty else "No pairs.",
        "```",
    ]
    (out_dir / "reports" / "DATASET_GROUP_AUDIT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    status = {
        "out_dir": str(out_dir),
        "n_run_dirs": int(len(run_dirs)),
        "n_records": int(len(records)),
        "n_datasets": int(groups["dataset_name"].nunique()),
        "n_dataset_groups": int(groups["dataset_group"].nunique()),
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def _provenance_note(dataset: object, group: str) -> str:
    name = str(dataset)
    if group == "kaggle_chem_group":
        return "Kaggle cross-cell/cross-patient chemical generalization family; do not use as independent external pair."
    if group == "sparse_cross_patient_group":
        return "Sparse cross-patient boundary; keep as failure-boundary evidence."
    if group == "gears_crispr_group":
        return "GEARS/CRISPR perturbation family."
    if name in {"Haber", "Parekh", "Frangieh"}:
        return "Public gene perturbation candidate; keep pairwise overlap audit visible."
    return "Unknown provenance; require manual audit before main external claim."


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset similarity and assign dataset groups.")
    parser.add_argument("--run-dir", type=Path, action="append", dest="run_dirs", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.run_dirs, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

