#!/usr/bin/env python3
"""E99: freeze independent multi-context perturbation rectangles.

Rectangle and split selection use only observation labels, per-pair cell counts,
fixed dimensions and hashes.  Expression matrices, effects, model predictions
and errors are never read.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb")
OUT = ROOT / "docs/实验结果/E99_multicontext_external_contract_20260713"
MIN_CELLS = 50
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
INVALID_OR_CONTROL = {"", "control", "ctrl", "nan", "none", "noise", "nt1"}


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    filename: str
    context_column: str
    modality: str
    n_contexts: int


SPECS = (
    DatasetSpec("Lara_exvivo", "LaraAstiasoHuntly2023_exvivo.h5ad", "celltype", "gene_knockout", 5),
    DatasetSpec("Santinha", "SantinhaPlatt2023.h5ad", "cell_types", "gene_knockout", 5),
    DatasetSpec("Cui", "CuiHacohen2023.h5ad", "celltype", "cytokine_stimulus", 6),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def rank_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(map(str, frame.columns))
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def best_rectangle(counts: pd.DataFrame, n_contexts: int) -> tuple[list[str], list[str]]:
    matrix = counts.pivot(index="context", columns="perturbation", values="n_cells").fillna(0).astype(int)
    if "control" not in matrix.columns:
        raise RuntimeError("control label is required for target-context basal profiles")
    best = None
    for indexes in itertools.combinations(range(len(matrix.index)), n_contexts):
        subset = matrix.iloc[list(indexes)]
        shared = subset.columns[subset.ge(MIN_CELLS).all(axis=0)].astype(str).tolist()
        if "control" not in shared:
            continue
        perturbations = sorted(value for value in shared if value.strip().lower() not in INVALID_OR_CONTROL)
        score = (len(perturbations), int(subset[shared].to_numpy().sum()))
        contexts = tuple(map(str, subset.index))
        candidate = (score, tuple(reversed(contexts)), contexts, perturbations)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("no complete rectangle with a valid control column")
    return list(best[2]), list(best[3])


def main() -> None:
    for name in ["manifests", "tables", "reports"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    manifest_rows, rectangle_rows, source_rows = [], [], []

    for spec in SPECS:
        path = DATA / spec.filename
        dataset = ad.read_h5ad(path, backed="r")
        required = {spec.context_column, "perturbation"}
        if not required.issubset(dataset.obs.columns):
            raise RuntimeError(f"{spec.dataset}: missing {sorted(required - set(dataset.obs.columns))}")
        obs = dataset.obs[[spec.context_column, "perturbation"]].copy()
        shape = tuple(map(int, dataset.shape))
        dataset.file.close()
        obs.columns = ["context", "perturbation"]
        obs = obs.astype(str)
        obs = obs[obs["context"].ne("nan")]
        counts = obs.groupby(["context", "perturbation"], observed=True).size().rename("n_cells").reset_index()
        contexts, perturbations = best_rectangle(counts, spec.n_contexts)
        eligible = counts[counts["context"].isin(contexts) & counts["perturbation"].isin(perturbations)].copy()
        expected = len(contexts) * len(perturbations)
        if len(eligible) != expected or int(eligible["n_cells"].min()) < MIN_CELLS:
            raise RuntimeError(f"{spec.dataset}: rectangle invariant failed")
        source_rows.append(
            {
                "dataset": spec.dataset,
                "modality": spec.modality,
                "source_h5ad": str(path),
                "source_sha256": sha256(path),
                "source_n_cells": shape[0],
                "source_n_genes": shape[1],
                "context_column": spec.context_column,
                "n_rectangle_contexts": len(contexts),
                "n_rectangle_perturbations": len(perturbations),
                "rectangle_min_cells": int(eligible["n_cells"].min()),
                "expression_read_during_selection": False,
            }
        )
        for row in eligible.itertuples(index=False):
            rectangle_rows.append(
                {"dataset": spec.dataset, "modality": spec.modality, "context": row.context,
                 "perturbation": row.perturbation, "n_cells": int(row.n_cells)}
            )

        for fold_index, heldout in enumerate(sorted(contexts), start=1):
            fold_id = f"{spec.dataset}_context_holdout_{fold_index}_{heldout.replace(' ', '_')}"
            source_contexts = sorted(value for value in contexts if value != heldout)
            n_new = max(5, int(math.ceil(0.20 * len(perturbations))))
            ordered_perts = sorted(perturbations, key=lambda value: rank_key("E99", fold_id, "perturbation", value))
            new_perts = set(ordered_perts[:n_new])
            seen_perts = set(ordered_perts[n_new:])
            source_seen = [(context, pert) for context in source_contexts for pert in sorted(seen_perts)]
            ordered_pairs = sorted(source_seen, key=lambda pair: rank_key("E99", fold_id, "pair", *pair))
            n_aux = min(30, max(8, int(round(0.10 * len(ordered_pairs)))))
            validation = set(ordered_pairs[:n_aux])
            random_missing = set(ordered_pairs[n_aux : 2 * n_aux])
            base_train = [pair for pair in ordered_pairs if pair not in validation and pair not in random_missing]
            if len(base_train) < 40:
                raise RuntimeError(f"{fold_id}: training rectangle too small after auxiliary splits")

            membership: dict[tuple[str, str], dict[float, bool]] = {}
            for context in source_contexts:
                pairs = sorted(
                    [pair for pair in base_train if pair[0] == context],
                    key=lambda pair: rank_key("E99", fold_id, "fraction", *pair),
                )
                for fraction in FRACTIONS:
                    selected = set(pairs[: max(1, int(round(len(pairs) * fraction)))])
                    for pair in pairs:
                        membership.setdefault(pair, {})[fraction] = pair in selected

            count_map = {(row.context, row.perturbation): int(row.n_cells) for row in eligible.itertuples(index=False)}
            for context in sorted(contexts):
                for perturbation in sorted(perturbations):
                    pair = (context, perturbation)
                    if context == heldout and perturbation in new_perts:
                        split, setting = "test", "context_and_perturbation_unseen"
                    elif context == heldout:
                        split, setting = "test", "context_unseen_row"
                    elif perturbation in new_perts:
                        split, setting = "test", "perturbation_unseen_column"
                    elif pair in validation:
                        split, setting = "val", "source_validation_pair"
                    elif pair in random_missing:
                        split, setting = "test", "random_missing_pair"
                    else:
                        split, setting = "train", "source_train_pair"
                    record = {
                        "dataset": spec.dataset,
                        "modality": spec.modality,
                        "fold_id": fold_id,
                        "heldout_context": heldout,
                        "source_contexts": "+".join(source_contexts),
                        "split": split,
                        "setting": setting,
                        "context": context,
                        "perturbation": perturbation,
                        "n_cells": count_map[pair],
                        "perturbation_seen_in_training": perturbation in seen_perts,
                        "context_seen_in_training": context in source_contexts,
                        "selected_without_expression": True,
                    }
                    for fraction in FRACTIONS:
                        record[f"in_train_fraction_{int(fraction * 100)}"] = bool(
                            split == "train" and membership.get(pair, {}).get(fraction, False)
                        )
                    manifest_rows.append(record)

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["dataset", "fold_id", "split", "setting", "context", "perturbation"]
    )
    rectangle = pd.DataFrame(rectangle_rows)
    sources = pd.DataFrame(source_rows)
    summary = manifest.groupby(["dataset", "modality", "fold_id", "setting", "split"], as_index=False).agg(
        n_tasks=("perturbation", "size"), min_cells=("n_cells", "min"), median_cells=("n_cells", "median")
    )
    fraction_rows = []
    for (dataset_name, fold_id), group in manifest.groupby(["dataset", "fold_id"]):
        expected = int(group["context"].nunique() * group["perturbation"].nunique())
        if len(group) != expected or group.duplicated(["context", "perturbation"]).any():
            raise RuntimeError(f"{fold_id}: partition is not a complete rectangle")
        previous: set[tuple[str, str]] = set()
        for fraction in FRACTIONS:
            selected = set(
                map(tuple, group.loc[group[f"in_train_fraction_{int(fraction * 100)}"], ["context", "perturbation"]].to_numpy())
            )
            if not previous.issubset(selected):
                raise RuntimeError(f"{fold_id}: training fractions are not nested")
            previous = selected
            fraction_rows.append(
                {"dataset": dataset_name, "fold_id": fold_id, "train_fraction": fraction,
                 "n_train_pairs": len(selected), "n_train_contexts": len({x[0] for x in selected}),
                 "n_train_perturbations": len({x[1] for x in selected})}
            )

    manifest.to_csv(OUT / "manifests/E99_TASK_MANIFEST.csv", index=False)
    rectangle.to_csv(OUT / "tables/E99_RECTANGLE_CELL_COUNTS.csv", index=False)
    sources.to_csv(OUT / "tables/E99_SOURCE_AUDIT.csv", index=False)
    summary.to_csv(OUT / "tables/E99_SETTING_SUMMARY.csv", index=False)
    pd.DataFrame(fraction_rows).to_csv(OUT / "tables/E99_TRAIN_FRACTION_SUMMARY.csv", index=False)
    status = {
        "experiment": "E99_multicontext_external_contract",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "min_cells_per_pair": MIN_CELLS,
        "datasets": sources[["dataset", "modality", "n_rectangle_contexts", "n_rectangle_perturbations"]].to_dict("records"),
        "n_folds": int(manifest["fold_id"].nunique()),
        "n_manifest_rows": len(manifest),
        "training_fractions": FRACTIONS,
        "expression_matrix_read_during_selection": False,
        "target_effect_or_error_used_for_selection": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = f"""# E99｜多背景外部矩阵冻结合同

E99 从服务器已有官方 h5ad 中冻结三套独立矩形。矩形选择只使用 context、perturbation 标签和每个 pair 的细胞数，目标是先最大化满足每 pair 至少 {MIN_CELLS} 个细胞的共同扰动数，再按总细胞数破同分。表达矩阵、效应、预测和误差均未读取。

## 数据矩形

{markdown_table(sources[["dataset", "modality", "n_rectangle_contexts", "n_rectangle_perturbations", "rectangle_min_cells"]])}

## Setting 规模

{markdown_table(summary)}

Lara ex vivo 与 Santinha 是两套独立遗传扰动复制；Cui 是细胞因子刺激，用于检验不同扰动类型。每个 context 各做一次整行留出，每折同时包含随机缺失 pair、整列新扰动和双未见任务。E99 只冻结合同，预测器输出与 test truth 尚未生成。
"""
    (OUT / "reports/E99_CONTRACT_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E99 先看这个\n\n先读 `reports/E99_CONTRACT_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(sources.to_string(index=False))


if __name__ == "__main__":
    main()
