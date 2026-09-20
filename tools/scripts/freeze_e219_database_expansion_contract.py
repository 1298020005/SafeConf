#!/usr/bin/env python3
"""Freeze three additional genetic-perturbation validation contracts from labels only."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/yyf/datasets/singlecell_perturbation_atlas")
OUT = ROOT / "docs/实验结果/E219_database_expansion_contract_20260920"
MIN_CELLS = 30
NAMESPACE = "E219_DATABASE_EXPANSION_V1"


@dataclass(frozen=True)
class Spec:
    dataset: str
    path: Path
    context_column: str | None
    perturbation_column: str
    control_label: str
    context_identity: str


SPECS = (
    Spec(
        "Parekh",
        DATA_ROOT / "extra_official/cellular_context_generalization/Parekh.h5ad",
        "cell_type",
        "perturbation",
        "CTRL",
        "biological_cell_state",
    ),
    Spec(
        "Papalexi",
        DATA_ROOT / "official_generalization/Papalexi.h5ad",
        "replicate",
        "perturbation",
        "control",
        "technical_replicate",
    ),
    Spec(
        "Adamson",
        DATA_ROOT / "official_generalization/Adamson.h5ad",
        None,
        "perturbation",
        "control",
        "single_context",
    ),
)


class ContractFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, (NAMESPACE, *parts))).encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def normalize_perturbation(values: pd.Series, control_label: str) -> pd.Series:
    values = values.astype(str)
    return values.where(values.ne(control_label), "control")


def load_labels(spec: Spec) -> tuple[pd.DataFrame, tuple[int, int], list[str]]:
    if not spec.path.is_file():
        raise ContractFailure(f"missing {spec.path}")
    data = ad.read_h5ad(spec.path, backed="r")
    required = {spec.perturbation_column}
    if spec.context_column:
        required.add(spec.context_column)
    if not required.issubset(data.obs.columns):
        raise ContractFailure(f"{spec.dataset} missing {sorted(required-set(data.obs.columns))}")
    columns = [spec.perturbation_column] + ([spec.context_column] if spec.context_column else [])
    labels = data.obs[columns].copy()
    shape = tuple(map(int, data.shape))
    obs_columns = list(map(str, data.obs.columns))
    data.file.close()
    labels = labels.astype(str)
    labels["perturbation"] = normalize_perturbation(
        labels[spec.perturbation_column], spec.control_label
    )
    labels["context"] = (
        labels[spec.context_column] if spec.context_column else spec.dataset
    )
    return labels[["context", "perturbation"]], shape, obs_columns


def multi_context_manifest(spec: Spec, counts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    matrix = counts.pivot(index="context", columns="perturbation", values="n_cells").fillna(0).astype(int)
    eligible = matrix.columns[matrix.ge(MIN_CELLS).all(axis=0)].astype(str).tolist()
    if "control" not in eligible:
        raise ContractFailure(f"{spec.dataset} lacks eligible controls")
    perturbations = sorted(value for value in eligible if value != "control")
    contexts = sorted(map(str, matrix.index))
    if len(contexts) < 2 or len(perturbations) < 8:
        raise ContractFailure(f"{spec.dataset} rectangle is too small")
    rows = []
    for heldout in contexts:
        fold_id = f"{spec.dataset}_holdout_{heldout}"
        ordered = sorted(perturbations, key=lambda value: key(spec.dataset, heldout, "column", value))
        n_new = max(2, int(math.ceil(0.20 * len(ordered))))
        new = set(ordered[:n_new])
        seen = set(ordered[n_new:])
        source_pairs = [(c, p) for c in contexts if c != heldout for p in sorted(seen)]
        pair_order = sorted(source_pairs, key=lambda pair: key(spec.dataset, heldout, "pair", *pair))
        n_aux = max(1, int(math.floor(0.10 * len(pair_order))))
        validation = set(pair_order[:n_aux])
        random_missing = set(pair_order[n_aux:2*n_aux])
        for context in contexts:
            for perturbation in perturbations:
                pair = (context, perturbation)
                if context == heldout and perturbation in new:
                    split, setting = "test", "context_and_perturbation_unseen"
                elif context == heldout:
                    split, setting = "test", "context_unseen"
                elif perturbation in new:
                    split, setting = "test", "perturbation_unseen"
                elif pair in validation:
                    split, setting = "val", "source_validation"
                elif pair in random_missing:
                    split, setting = "test", "random_pair"
                else:
                    split, setting = "train", "source_train"
                rows.append({
                    "dataset": spec.dataset,
                    "context_identity": spec.context_identity,
                    "fold_id": fold_id,
                    "heldout_context": heldout,
                    "context": context,
                    "perturbation": perturbation,
                    "split": split,
                    "setting": setting,
                    "n_cells": int(matrix.loc[context, perturbation]),
                    "selected_without_expression": True,
                })
    return pd.DataFrame(rows), {
        "n_contexts": len(contexts),
        "n_perturbations": len(perturbations),
        "n_folds": len(contexts),
        "minimum_pair_cells": int(matrix.loc[contexts, perturbations + ["control"]].min().min()),
    }


def single_context_manifest(spec: Spec, counts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    eligible = counts.loc[counts.n_cells.ge(MIN_CELLS), "perturbation"].astype(str).tolist()
    if "control" not in eligible:
        raise ContractFailure(f"{spec.dataset} lacks eligible controls")
    perturbations = sorted(value for value in eligible if value != "control")
    ordered = sorted(perturbations, key=lambda value: key(spec.dataset, "column", value))
    n_test = max(5, int(math.ceil(0.20 * len(ordered))))
    n_val = max(5, int(math.ceil(0.10 * len(ordered))))
    test, validation = set(ordered[:n_test]), set(ordered[n_test:n_test+n_val])
    count_map = counts.set_index("perturbation").n_cells.to_dict()
    rows = []
    for perturbation in perturbations:
        if perturbation in test:
            split, setting = "test", "perturbation_unseen"
        elif perturbation in validation:
            split, setting = "val", "source_validation"
        else:
            split, setting = "train", "source_train"
        rows.append({
            "dataset": spec.dataset,
            "context_identity": spec.context_identity,
            "fold_id": f"{spec.dataset}_perturbation_holdout",
            "heldout_context": "NA",
            "context": spec.dataset,
            "perturbation": perturbation,
            "split": split,
            "setting": setting,
            "n_cells": int(count_map[perturbation]),
            "selected_without_expression": True,
        })
    return pd.DataFrame(rows), {
        "n_contexts": 1,
        "n_perturbations": len(perturbations),
        "n_folds": 1,
        "minimum_pair_cells": int(min(count_map[p] for p in perturbations + ["control"])),
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise ContractFailure(f"refusing to overwrite {OUT}")
    manifests, audits, count_tables = [], [], []
    for spec in SPECS:
        labels, shape, obs_columns = load_labels(spec)
        counts = labels.groupby(["context", "perturbation"], observed=True).size().rename("n_cells").reset_index()
        if spec.context_column:
            manifest, summary = multi_context_manifest(spec, counts)
        else:
            collapsed = counts.groupby("perturbation", as_index=False).n_cells.sum()
            manifest, summary = single_context_manifest(spec, collapsed)
        if manifest.duplicated(["dataset", "fold_id", "context", "perturbation"]).any():
            raise ContractFailure(f"{spec.dataset} duplicate task identity")
        manifests.append(manifest)
        count_tables.append(counts.assign(dataset=spec.dataset))
        audits.append({
            "dataset": spec.dataset,
            "source_path": str(spec.path),
            "source_sha256": sha256(spec.path),
            "source_cells": shape[0],
            "source_genes": shape[1],
            "context_column": spec.context_column or "__single_context__",
            "perturbation_column": spec.perturbation_column,
            "control_label_original": spec.control_label,
            "context_identity": spec.context_identity,
            "obs_columns": "|".join(obs_columns),
            **summary,
            "expression_values_read_for_selection": False,
            "truth_or_model_error_used_for_selection": False,
        })
    manifest = pd.concat(manifests, ignore_index=True).sort_values(
        ["dataset", "fold_id", "split", "setting", "context", "perturbation"]
    )
    audit = pd.DataFrame(audits)
    counts = pd.concat(count_tables, ignore_index=True)
    summary = manifest.groupby(
        ["dataset", "context_identity", "setting", "split"], as_index=False
    ).agg(n_tasks=("perturbation", "size"), min_cells=("n_cells", "min"))
    atomic_text(OUT / "tables/E219_TASK_MANIFEST.csv", manifest.to_csv(index=False))
    atomic_text(OUT / "tables/E219_SOURCE_AUDIT.csv", audit.to_csv(index=False))
    atomic_text(OUT / "tables/E219_CELL_COUNTS.csv", counts.to_csv(index=False))
    atomic_text(OUT / "tables/E219_SETTING_SUMMARY.csv", summary.to_csv(index=False))
    status = {
        "experiment": "E219_database_expansion_contract",
        "status": "PASS",
        "generated_at": now(),
        "minimum_cells_per_pair": MIN_CELLS,
        "datasets": audit[["dataset", "context_identity", "source_cells", "source_genes", "n_contexts", "n_perturbations", "n_folds", "minimum_pair_cells"]].to_dict("records"),
        "manifest_rows": len(manifest),
        "selection_namespace": NAMESPACE,
        "expression_values_read_for_selection": False,
        "truth_or_model_error_used_for_selection": False,
        "next_gate": "model compatibility smoke before any formal prediction claim",
    }
    atomic_text(OUT / "RUN_STATUS.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = """# E219｜已有数据库扩展合同

## 这一步做了什么

本实验只用细胞标签和每个任务的细胞数冻结新验证任务，没有读取表达值、真实扰动效应、模型预测或误差来选数据。它解决的是“用哪些数据、怎样切分”的问题，本身不是性能结果。

## 已冻结的数据与用途

- Parekh：4,346 个细胞、5,000 个基因、3 个生物细胞状态和 10 个候选扰动；用于检验新背景，但模型兼容性较低，只作为边界测试。
- Papalexi：20,343 个细胞、5,013 个基因、3 个技术重复和 24 个扰动；用于检验重复测量稳定性。技术重复不能写成新的生物学背景。
- Adamson：56,998 个细胞、5,043 个基因和 76 个扰动；用于单一背景下的新扰动留出验证。

合计冻结 382 行任务合同；每个纳入的“背景—扰动”组合至少有 30 个细胞。切分包含训练、验证、新背景、新扰动、背景与扰动同时未见以及随机缺失组合。

## 已通过的兼容性预检

- Adamson：76/76 个扰动同时存在于表达矩阵基因轴和 scGPT 词表，优先进入预测冒烟。
- Papalexi：23/24 个扰动兼容，作为第二优先级。
- Parekh：5/10 个扰动兼容，不作为主要外部确认数据。

兼容性预检只读取基因名和模型词表，没有读取表达值、目标效应或模型误差。

## 下一道门

先在 Adamson 和 Papalexi 各运行一个冻结折的双模型预测冒烟。只有上游预测器达到预先规定的最低能力，才封存完整预测和风险表，再一次性读取测试真值。若上游预测器明显弱于 no-change 基线，该数据只能报告为上游能力失败，不能用来评价 SafeConf。
"""
    atomic_text(OUT / "E219_REPORT.md", report)
    atomic_text(OUT / "ANALYSIS_CONTRACT.md", __doc__.strip() + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
