#!/usr/bin/env python3
"""E97: freeze Frangieh gene context x perturbation hard-setting contracts.

Only observation labels and cell counts are read.  No expression matrix,
prediction, effect size, or target error participates in task selection.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/perturb_processed.h5ad")
OUT = ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713"
MIN_CELLS = 50
N_NEW_PERTURBATIONS = 30
N_VALIDATION_PAIRS = 30
N_RANDOM_MISSING_PAIRS = 30
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


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


def main() -> None:
    for name in ["tables", "manifests", "reports"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(SOURCE, backed="r")
    required = {"cell_type", "condition"}
    if not required.issubset(adata.obs.columns):
        raise RuntimeError(f"missing obs columns: {sorted(required - set(adata.obs.columns))}")
    obs = adata.obs[["cell_type", "condition"]].copy()
    shape = tuple(map(int, adata.shape))
    adata.file.close()
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["condition"] = obs["condition"].astype(str)
    obs = obs[obs["condition"].str.endswith("+ctrl") & obs["condition"].ne("ctrl")]
    counts = obs.groupby(["cell_type", "condition"], observed=True).size().rename("n_cells").reset_index()
    matrix = counts.pivot(index="cell_type", columns="condition", values="n_cells").fillna(0).astype(int)
    contexts = sorted(matrix.index.astype(str))
    shared = sorted(matrix.columns[matrix.ge(MIN_CELLS).all(axis=0)].astype(str))
    if len(contexts) != 3 or len(shared) < 150:
        raise RuntimeError(f"unexpected Frangieh matrix: contexts={contexts}, shared={len(shared)}")
    eligible = counts[counts.cell_type.isin(contexts) & counts.condition.isin(shared)].copy()

    all_rows = []
    fraction_rows = []
    for fold_index, heldout in enumerate(contexts, start=1):
        fold_id = f"frangieh_context_holdout_{fold_index}_{heldout.replace(' ', '_')}"
        source_contexts = [value for value in contexts if value != heldout]
        ordered_perts = sorted(shared, key=lambda value: rank_key("E97", fold_id, "perturbation", value))
        new_perts = set(ordered_perts[:N_NEW_PERTURBATIONS])
        seen_perts = set(ordered_perts[N_NEW_PERTURBATIONS:])
        source_seen = [(context, pert) for context in source_contexts for pert in sorted(seen_perts)]
        ordered_pairs = sorted(source_seen, key=lambda pair: rank_key("E97", fold_id, "pair", *pair))
        validation = set(ordered_pairs[:N_VALIDATION_PAIRS])
        random_missing = set(ordered_pairs[N_VALIDATION_PAIRS : N_VALIDATION_PAIRS + N_RANDOM_MISSING_PAIRS])
        base_train = [pair for pair in ordered_pairs if pair not in validation and pair not in random_missing]

        # Nested training submatrices are selected within each source context.
        fraction_membership: dict[tuple[str, str], dict[float, bool]] = {}
        for context in source_contexts:
            context_pairs = [pair for pair in base_train if pair[0] == context]
            context_pairs = sorted(context_pairs, key=lambda pair: rank_key("E97", fold_id, "fraction", *pair))
            for fraction in FRACTIONS:
                keep = max(1, int(round(len(context_pairs) * fraction)))
                selected = set(context_pairs[:keep])
                for pair in context_pairs:
                    fraction_membership.setdefault(pair, {})[fraction] = pair in selected

        for row in eligible.itertuples(index=False):
            pair = (row.cell_type, row.condition)
            if row.cell_type == heldout and row.condition in new_perts:
                split, setting = "test", "context_and_perturbation_unseen"
            elif row.cell_type == heldout:
                split, setting = "test", "context_unseen_row"
            elif row.condition in new_perts:
                split, setting = "test", "perturbation_unseen_column"
            elif pair in validation:
                split, setting = "val", "source_validation_pair"
            elif pair in random_missing:
                split, setting = "test", "random_missing_pair"
            else:
                split, setting = "train", "source_train_pair"
            record = {
                "fold_id": fold_id,
                "heldout_context": heldout,
                "source_contexts": "+".join(source_contexts),
                "split": split,
                "setting": setting,
                "context": row.cell_type,
                "perturbation": row.condition,
                "n_cells": int(row.n_cells),
                "perturbation_seen_in_training": row.condition in seen_perts,
                "context_seen_in_training": row.cell_type in source_contexts,
                "selected_without_expression": True,
            }
            for fraction in FRACTIONS:
                record[f"in_train_fraction_{int(fraction * 100)}"] = bool(
                    split == "train" and fraction_membership.get(pair, {}).get(fraction, False)
                )
            all_rows.append(record)
        for fraction in FRACTIONS:
            selected = [pair for pair in base_train if fraction_membership[pair][fraction]]
            fraction_rows.append(
                {
                    "fold_id": fold_id,
                    "heldout_context": heldout,
                    "train_fraction": fraction,
                    "n_train_pairs": len(selected),
                    "n_train_contexts": len(set(pair[0] for pair in selected)),
                    "n_train_perturbations": len(set(pair[1] for pair in selected)),
                }
            )

    manifest = pd.DataFrame(all_rows).sort_values(["fold_id", "split", "setting", "context", "perturbation"])
    fractions = pd.DataFrame(fraction_rows)
    manifest.to_csv(OUT / "manifests/E97_TASK_MANIFEST.csv", index=False)
    fractions.to_csv(OUT / "tables/E97_TRAIN_FRACTION_SUMMARY.csv", index=False)
    eligible.to_csv(OUT / "tables/E97_ELIGIBLE_CELL_COUNTS.csv", index=False)
    summary = manifest.groupby(["fold_id", "setting", "split"], as_index=False).agg(n_tasks=("perturbation", "size"), min_cells=("n_cells", "min"), median_cells=("n_cells", "median"), max_cells=("n_cells", "max"))
    summary.to_csv(OUT / "tables/E97_SETTING_SUMMARY.csv", index=False)

    for fold_id, group in manifest.groupby("fold_id"):
        if len(group) != len(contexts) * len(shared):
            raise RuntimeError(f"{fold_id}: incomplete Cartesian partition")
        if group.duplicated(["context", "perturbation"]).any():
            raise RuntimeError(f"{fold_id}: duplicate task")
        if not group.loc[group.split.eq("test"), "selected_without_expression"].all():
            raise RuntimeError(f"{fold_id}: selection provenance failed")

    status = {
        "experiment": "E97_frangieh_gene_cartesian_contract",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_h5ad": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "source_shape": shape,
        "contexts": contexts,
        "n_contexts": len(contexts),
        "n_shared_perturbations_min_cells_50": len(shared),
        "n_folds": len(contexts),
        "n_manifest_rows": len(manifest),
        "training_fractions": FRACTIONS,
        "expression_matrix_read_during_selection": False,
        "target_effect_or_error_used_for_selection": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = f"""# E97｜Frangieh 遗传扰动三行矩阵冻结合同

Frangieh 原始数据包含 3 个细胞背景和 211 个单基因扰动。按每个“背景×扰动”至少 {MIN_CELLS} 个细胞筛选后，得到完整的 3×{len(shared)} 矩阵。任务只根据标签、细胞数和哈希顺序冻结，没有读取表达矩阵、效应、预测或误差。

每个 fold 留出一个完整背景。30 个扰动同时作为整列未见基因，其余扰动在源背景可见；因此同一合同同时包含整行新背景、整列新扰动、背景与扰动双未见、随机缺失 pair。训练任务再冻结 25%、50%、75%、100% 四个嵌套子矩阵。

## 任务规模

{markdown_table(summary)}

## 训练子矩阵

{markdown_table(fractions)}

E97 只完成实验合同，不把 reference predictor 当作正式双模型结果。后续预测器必须读取 `E97_TASK_MANIFEST.csv`，每个训练比例重新训练，并在预测落盘后才读取 test truth。
"""
    (OUT / "reports/E97_CONTRACT_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E97 先看这个\n\n先读 `reports/E97_CONTRACT_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    print(fractions.to_string(index=False))


if __name__ == "__main__":
    main()
