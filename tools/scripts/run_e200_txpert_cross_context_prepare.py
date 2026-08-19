#!/usr/bin/env python3
"""Audit and freeze the official TxPert K562 cross-context experiment."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E200_txpert_cross_context_k562_20260802"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FREEZE = OUT / "ANALYSIS_FREEZE.md"
STATUS = OUT / "E200_STATUS.json"
SCRIPT = Path(__file__).resolve()
ADAPTER = ROOT / "tools/scripts/txpert_cross_cell_adapter.py"

TXPERT = Path("/home/yyf/archive/external/TxPert")
TXPERT_COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
SCPERTEVAL = Path("/home/yyf/archive/external/scPertEval")
SCPERTEVAL_COMMIT = "8709eb07a0e7d4ecf1c60c977f2018690a749975"
DATA = Path("/home/yyf/data/txpert_official_20260802/cache")
ARCHIVE = DATA / "K562_cross_cell_lines.zip"
DATASET = DATA / "K562_cross_cell_lines/de_adata_test.h5ad"
SPLIT = DATA / "K562_cross_cell_lines/splits/train_test_split.pkl"
SUBGROUP = DATA / "K562_cross_cell_lines/splits/subgroup.pkl"
CHECKPOINT = DATA / "checkpoints/K562_unseen_cell_gat.ckpt"

EXPECTED = {
    ARCHIVE: (1_750_726_245, "c6aef01cb1adb40be9060850a0e660a7a369b35b33bdc3a4b44fb753449183f1"),
    DATASET: (7_767_053_064, "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8"),
    SPLIT: (31_629, "c922dc62ee4263951ec6a45e6e8cfc51e4104d5e1b0704eefd46848acddba402"),
    SUBGROUP: (14_436, "3d0a2f92fdad7809e5e13f4931c0a7eca49e360fd83a25a3b8ca90ec6ebe9e8b"),
    CHECKPOINT: (49_748_832, "22994e5097255f75fc4196bddea8e8227dd4c110694a7794f066e041fbf42763"),
}


class PrepareFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise PrepareFailure(f"missing function {name}: {path}")


def add(gates: list[dict[str, Any]], check: str, observed: Any, expected: Any, passed: bool, detail: str = "") -> None:
    gates.append(
        {
            "check": check,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def category_row(name: str, conditions: set[str], obs: pd.DataFrame) -> dict[str, Any]:
    block = obs.loc[
        obs.cell_line.eq("K562")
        & ~obs.control.astype(bool)
        & obs.condition.isin(conditions)
    ]
    counts = block.groupby("condition", observed=True).size()
    return {
        "category": name,
        "n_tasks": len(counts),
        "n_cells": int(counts.sum()),
        "n_ge30": int(counts.ge(30).sum()),
        "n_10_29": int(counts.between(10, 29).sum()),
        "min_cells": int(counts.min()),
        "median_cells": float(counts.median()),
        "max_cells": int(counts.max()),
    }


def main() -> None:
    generated_outputs = (
        STATUS,
        TABLES / "E200_PREPARE_GATES.csv",
        REPORTS / "E200_PREPARE_REPORT.md",
    )
    if any(path.exists() for path in generated_outputs):
        raise PrepareFailure("E200 prepare output already exists")
    gates: list[dict[str, Any]] = []
    inputs = []
    add(gates, "freeze_exists", FREEZE.is_file(), True, FREEZE.is_file())
    add(gates, "adapter_exists", ADAPTER.is_file(), True, ADAPTER.is_file())
    for label, repo, expected in (
        ("txpert", TXPERT, TXPERT_COMMIT),
        ("scperteval", SCPERTEVAL, SCPERTEVAL_COMMIT),
    ):
        observed = git(repo, "rev-parse", "HEAD")
        add(gates, f"{label}_commit", observed, expected, observed == expected)
        dirty = git(repo, "status", "--porcelain")
        add(gates, f"{label}_clean", dirty or "CLEAN", "CLEAN", not dirty)
    for path, (n_bytes, expected_sha) in EXPECTED.items():
        exists = path.is_file()
        add(gates, f"input_exists:{path.name}", exists, True, exists)
        if not exists:
            continue
        observed_sha = sha256_file(path)
        add(gates, f"input_bytes:{path.name}", path.stat().st_size, n_bytes, path.stat().st_size == n_bytes)
        add(gates, f"input_sha:{path.name}", observed_sha, expected_sha, observed_sha == expected_sha)
        inputs.append(
            {
                "role": "official_input",
                "path": "DATA/" + path.relative_to(Path("/home/yyf/data")).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": observed_sha,
            }
        )
    with zipfile.ZipFile(ARCHIVE) as archive:
        corrupt = archive.testzip()
        members = set(archive.namelist())
    add(gates, "archive_integrity", corrupt or "PASS", "PASS", corrupt is None)
    for suffix in (
        "K562_cross_cell_lines/de_adata_test.h5ad",
        "K562_cross_cell_lines/splits/train_test_split.pkl",
        "K562_cross_cell_lines/splits/subgroup.pkl",
    ):
        add(gates, f"archive_member:{suffix}", suffix in members, True, suffix in members)

    dataset = ad.read_h5ad(DATASET, backed="r")
    obs = dataset.obs.copy()
    shape = dataset.shape
    dataset.file.close()
    add(gates, "dataset_shape", str(shape), "(632488, 3352)", shape == (632_488, 3_352))
    cell_counts = obs.cell_line.value_counts().sort_index()
    context_rows = []
    for context, n_total in cell_counts.items():
        block = obs.loc[obs.cell_line.eq(context)]
        context_rows.append(
            {
                "context": str(context),
                "n_total_cells": int(n_total),
                "n_control_cells": int(block.control.astype(bool).sum()),
                "n_perturbed_cells": int((~block.control.astype(bool)).sum()),
                "n_perturbations": int(block.loc[~block.control.astype(bool), "condition"].nunique()),
                "n_batches": int(block.batch.nunique()),
            }
        )
    n_k562_controls = int(
        obs.loc[obs.cell_line.eq("K562"), "control"].astype(bool).sum()
    )
    add(
        gates,
        "target_K562_controls",
        n_k562_controls,
        10_691,
        n_k562_controls == 10_691,
    )

    split = joblib.load(SPLIT)
    subgroups = joblib.load(SUBGROUP)["test_subgroup"]
    train, val = set(split["train"]), set(split["val"])
    official_cell = set(map(str, subgroups["unseen_cell"]))
    double_ood = set(map(str, subgroups["unseen_cell_pert"]))
    strict = official_cell & train
    val_only = (official_cell & val) - train
    neither = official_cell - train - val
    categories = pd.DataFrame(
        [
            category_row("strict_context_only_train_seen", strict, obs),
            category_row("validation_only", val_only, obs),
            category_row("neither_train_nor_validation", neither, obs),
            category_row("unseen_context_and_perturbation", double_ood, obs),
        ]
    )
    expected_categories = {
        "strict_context_only_train_seen": (580, 80_153, 566, 14),
        "validation_only": (202, 28_023, 199, 3),
        "neither_train_nor_validation": (33, 3_838, 32, 1),
        "unseen_context_and_perturbation": (272, 38_458, 266, 6),
    }
    for row in categories.itertuples(index=False):
        observed = (row.n_tasks, row.n_cells, row.n_ge30, row.n_10_29)
        expected = expected_categories[row.category]
        add(gates, f"category:{row.category}", str(observed), str(expected), observed == expected)
    add(gates, "official_unseen_cell_partition", len(strict | val_only | neither), 815, len(strict | val_only | neither) == 815)
    add(gates, "subgroups_disjoint", len(official_cell & double_ood), 0, not (official_cell & double_ood))

    training_contexts = {"RPE1", "hepg2", "jurkat"}
    train_obs = obs.loc[
        obs.cell_line.isin(training_contexts)
        & ~obs.control.astype(bool)
        & obs.condition.isin(strict)
        & obs.condition.isin(train)
    ]
    train_support = (
        train_obs.groupby("condition", observed=True)
        .agg(
            n_train_cells=("condition", "size"),
            n_train_contexts=("cell_line", "nunique"),
        )
        .reset_index()
    )
    target_support = (
        obs.loc[
            obs.cell_line.eq("K562")
            & ~obs.control.astype(bool)
            & obs.condition.isin(strict)
        ]
        .groupby("condition", observed=True)
        .size()
        .rename("n_target_cells")
        .reset_index()
    )
    task_support = target_support.merge(
        train_support, on="condition", how="left", validate="one_to_one"
    ).sort_values("condition")
    support_distribution = (
        task_support.n_train_contexts.value_counts().sort_index().to_dict()
    )
    add(gates, "strict_actual_train_support", int(task_support.n_train_cells.notna().sum()), 580, task_support.n_train_cells.notna().all())
    add(gates, "strict_support_context_distribution", str(support_distribution), "{1: 75, 2: 162, 3: 343}", support_distribution == {1: 75, 2: 162, 3: 343})

    predictor_source = TXPERT / "gspp/predictor.py"
    predict_step = function_source(predictor_source, "predict_step")
    sample = function_source(predictor_source, "sample_inference")
    forward = function_source(predictor_source, "forward")
    static_rows = pd.DataFrame(
        [
            {"check": "truth_only_saved", "passed": predict_step.count("batch.x") == 1 and "'ground_truths': batch.x" in predict_step},
            {"check": "sample_excludes_truth", "passed": "batch.x" not in sample and "ground_truth" not in sample},
            {"check": "forward_excludes_truth", "passed": "target" not in forward and "ground_truth" not in forward},
        ]
    )
    for row in static_rows.itertuples(index=False):
        add(gates, f"static:{row.check}", row.passed, True, bool(row.passed))
    for role, path in (("prepare_runner", SCRIPT), ("adapter", ADAPTER), ("freeze", FREEZE)):
        inputs.append(
            {
                "role": role,
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    gate_frame = pd.DataFrame(gates)
    write_csv(TABLES / "E200_PREPARE_GATES.csv", gate_frame)
    write_csv(TABLES / "E200_INPUT_HASHES.csv", pd.DataFrame(inputs))
    write_csv(TABLES / "E200_CONTEXT_INVENTORY.csv", pd.DataFrame(context_rows))
    write_csv(TABLES / "E200_SPLIT_AUDIT.csv", categories)
    write_csv(TABLES / "E200_STRICT_TASK_SUPPORT.csv", task_support)
    write_csv(TABLES / "E200_STATIC_LEAKAGE_AUDIT.csv", static_rows)
    report = f"""# E200 prepare 报告

- 生成时间：`{now()}`
- 数据：632,488 个细胞，3,352 个基因，5 个数据标签背景；模型训练背景固定为 RPE1、HepG2、Jurkat，目标为 K562。
- 严格 context-only：580 个任务；主分析 566 个，低细胞数敏感性 14 个。
- 官方 `unseen_cell` 中另有 validation-only 202 个和 train/validation 均未见 33 个，已从主分析剥离。
- 同时未见背景和扰动：272 个，单列压力测试。
- prepare gates：{int(gate_frame.passed.sum())}/{len(gate_frame)}。

公开 checkpoint 只有跨 K562 的单个 GAT。E200 先做单模型与 general baseline 的可复核审计；多模型家族结论保持未回答。
"""
    write_text(REPORTS / "E200_PREPARE_REPORT.md", report)
    passed = bool(gate_frame.passed.all())
    write_json(
        STATUS,
        {
            "experiment": "E200_txpert_cross_context_k562",
            "stage": "PREPARE",
            "status": "PASS" if passed else "FAIL",
            "generated_at": now(),
            "n_strict_context_only_tasks": 580,
            "n_primary_tasks_ge30": 566,
            "n_sensitivity_tasks_10_29": 14,
            "n_validation_only_tasks": 202,
            "n_neither_train_nor_validation_tasks": 33,
            "n_double_ood_tasks": 272,
            "target_context": "K562",
            "training_contexts": ["RPE1", "hepg2", "jurkat"],
            "target_controls_allowed": True,
            "target_perturbations_in_training": False,
            "cross_dataset_transfer_answered": False,
            "multiple_model_families_answered": False,
        },
    )
    if not passed:
        failed = gate_frame.loc[~gate_frame.passed]
        raise PrepareFailure(f"E200 prepare gates failed: {failed.check.tolist()}")
    print(json.dumps({"status": "PASS", "gates": len(gate_frame)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
