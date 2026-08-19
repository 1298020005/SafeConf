#!/usr/bin/env python3
"""Freeze the E182 GSE225807 registered-family certificate before X access."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E182_gse225807_registered_family_20260724"
MANIFESTS = OUT / "manifests"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
SOURCE = Path(
    "/home/yyf/data/safeconf_e182_gse225807/source/"
    "GSE225807_RBP_CRISPRI.h5ad"
)
F0_ATTESTATION = Path(
    "/home/yyf/data/safeconf_e182_gse225807/source/F0_SOURCE_REFORMAT.json"
)
MODEL_SOURCE = (
    ROOT
    / "docs/实验结果/E177_sunshine_external_certificate_20260719/"
    "MODEL_INPUT_LOCK.json"
)
VOCAB = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human/vocab.json"
)

MIN_TARGET_CELLS = 20
MIN_GUIDE_CELLS = 5
REQUIRED_GUIDES = 2
TARGET_COVERAGE = 0.90
SEEDS = (3407, 3408, 3409, 3410, 3411)
SPLIT_SALT = "E182_GSE225807_TARGET_SPLIT_V1_20260724"
SPLIT_ALLOCATION = {
    "Q1_low": {
        "supervised_train": 7,
        "model_validation": 2,
        "conformal_calibration": 5,
        "prospective_evaluation": 5,
    },
    "Q2": {
        "supervised_train": 7,
        "model_validation": 2,
        "conformal_calibration": 5,
        "prospective_evaluation": 5,
    },
    "Q3": {
        "supervised_train": 7,
        "model_validation": 3,
        "conformal_calibration": 4,
        "prospective_evaluation": 5,
    },
    "Q4_high": {
        "supervised_train": 7,
        "model_validation": 2,
        "conformal_calibration": 5,
        "prospective_evaluation": 5,
    },
}
EXPECTED_SPLITS = {
    "supervised_train": 28,
    "model_validation": 9,
    "conformal_calibration": 19,
    "prospective_evaluation": 20,
}


class IntegrityError(RuntimeError):
    """The formal E182 metadata freeze is invalid."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}::{value}".encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def load_vocab() -> set[str]:
    value = json.loads(VOCAB.read_text())
    return set(value if isinstance(value, list) else value.keys())


def assign_splits(eligible: pd.DataFrame) -> pd.DataFrame:
    work = eligible.copy().sort_values("n_target_cells", kind="stable")
    work["cell_count_quartile"] = pd.qcut(
        work["n_target_cells"].rank(method="first"),
        q=4,
        labels=("Q1_low", "Q2", "Q3", "Q4_high"),
    ).astype(str)
    work["selection_hash"] = work["perturbation"].map(stable_hash)
    blocks: list[pd.DataFrame] = []
    for quartile, block in work.groupby(
        "cell_count_quartile", sort=True, observed=True
    ):
        ordered = block.sort_values(["selection_hash", "perturbation"]).copy()
        allocation = SPLIT_ALLOCATION[str(quartile)]
        labels = [
            split
            for split, count in allocation.items()
            for _ in range(count)
        ]
        if len(labels) != len(ordered):
            raise IntegrityError(
                f"E182 quartile allocation changed for {quartile}: "
                f"{len(labels)} != {len(ordered)}"
            )
        ordered["target_split"] = labels
        blocks.append(ordered)
    result = pd.concat(blocks, ignore_index=True)
    observed = result["target_split"].value_counts().to_dict()
    if observed != EXPECTED_SPLITS:
        raise IntegrityError(f"E182 target split changed: {observed}")
    return result.sort_values(["target_split", "selection_hash"]).reset_index(drop=True)


def git_prior_use_audit() -> dict[str, object]:
    patterns = ("GSE225807", "GSM7056649", "GSM7056650")
    matches: dict[str, list[str]] = {}
    for pattern in patterns:
        process = subprocess.run(
            ["git", "grep", "-l", pattern, "HEAD", "--", "docs/实验结果"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode not in (0, 1):
            raise IntegrityError(f"cannot audit prior use of {pattern}")
        matches[pattern] = [
            line for line in process.stdout.splitlines() if line.strip()
        ]
    return {
        "searched_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "patterns": matches,
        "prior_result_or_dataset_mentions": sum(map(len, matches.values())),
    }


def main() -> None:
    if OUT.exists():
        raise IntegrityError(f"append-only E182 output exists: {OUT}")
    if not SOURCE.is_file() or not F0_ATTESTATION.is_file():
        raise FileNotFoundError("E182 F0 source or attestation is missing")
    f0 = json.loads(F0_ATTESTATION.read_text())
    if f0.get("status") != "PASS":
        raise IntegrityError("E182 F0 attestation is not PASS")
    if sha256_file(SOURCE) != f0["output"]["sha256"]:
        raise IntegrityError("E182 F0 H5AD changed before metadata freeze")

    vocab = load_vocab()
    adata = ad.read_h5ad(SOURCE, backed="r")
    try:
        obs = adata.obs.copy()
        var_names = list(map(str, adata.var_names))
        shape = [int(adata.n_obs), int(adata.n_vars)]
        x_backend = type(adata.X).__name__
    finally:
        adata.file.close()
    obs["perturbation"] = obs["perturbation"].astype(str)
    obs["guide_id"] = obs["guide_id"].astype(str)
    obs["source_row_index"] = pd.to_numeric(
        obs["source_row_index"], errors="raise"
    ).astype(int)
    var_set = set(var_names)

    target_obs = obs[
        ~obs["perturbation"].isin(["control", "unassigned", "nan", ""])
    ].copy()
    guide_counts = (
        target_obs.groupby(["perturbation", "guide_id"], observed=True)
        .size()
        .rename("n_guide_cells")
        .reset_index()
    )
    target_counts = (
        target_obs.groupby("perturbation", observed=True)
        .agg(
            n_target_cells=("guide_id", "size"),
            n_guides=("guide_id", "nunique"),
        )
        .reset_index()
    )
    guide_audit = (
        guide_counts.groupby("perturbation", observed=True)
        .agg(
            min_guide_cells=("n_guide_cells", "min"),
            max_guide_cells=("n_guide_cells", "max"),
        )
        .reset_index()
    )
    audit = target_counts.merge(guide_audit, on="perturbation", validate="one_to_one")
    audit["is_single_human_gene_symbol"] = audit["perturbation"].str.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.-]*"
    )
    audit["in_expression_var"] = audit["perturbation"].isin(var_set)
    audit["in_scgpt_vocab"] = audit["perturbation"].isin(vocab)
    audit["eligible_e182_target"] = (
        audit["is_single_human_gene_symbol"]
        & audit["in_expression_var"]
        & audit["in_scgpt_vocab"]
        & audit["n_target_cells"].ge(MIN_TARGET_CELLS)
        & audit["n_guides"].eq(REQUIRED_GUIDES)
        & audit["min_guide_cells"].ge(MIN_GUIDE_CELLS)
    )
    audit["selection_used_expression_matrix_or_truth_effect"] = False
    eligible = assign_splits(audit[audit["eligible_e182_target"]].copy())
    if len(eligible) != sum(EXPECTED_SPLITS.values()):
        raise IntegrityError(f"E182 eligible target count changed: {len(eligible)}")

    tasks = guide_counts[
        guide_counts["perturbation"].isin(eligible["perturbation"])
    ].copy()
    tasks = tasks.merge(
        eligible[
            [
                "perturbation",
                "target_split",
                "cell_count_quartile",
                "selection_hash",
            ]
        ],
        on="perturbation",
        validate="many_to_one",
    )
    tasks["task_id"] = (
        "E182::" + tasks["perturbation"] + "::" + tasks["guide_id"]
    )
    tasks["target_cluster_id"] = tasks["perturbation"]
    tasks["truth_access_phase"] = tasks["target_split"].map(
        {
            "supervised_train": "F2_PRETRUTH_SUPERVISED",
            "model_validation": "F2_PRETRUTH_VALIDATION",
            "conformal_calibration": "F3_CALIBRATION_ONLY",
            "prospective_evaluation": "F4_EVALUATION_ONLY",
        }
    )
    tasks["primary_task_unit"] = "guide_replicate_effect_vs_negative_guide_controls"
    tasks["selection_used_expression_matrix_or_truth_effect"] = False
    if tasks.groupby("perturbation")["guide_id"].nunique().ne(2).any():
        raise IntegrityError("E182 selected target lost a guide replicate")
    if tasks.groupby("perturbation")["target_split"].nunique().max() != 1:
        raise IntegrityError("E182 target identity leaked across splits")

    split_map = eligible.set_index("perturbation")["target_split"].to_dict()
    selected_rows = obs[obs["perturbation"].isin(split_map)].copy()
    selected_rows["target_split"] = selected_rows["perturbation"].map(split_map)
    selected_rows["truth_access_phase"] = selected_rows["target_split"].map(
        {
            "supervised_train": "F2_PRETRUTH_SUPERVISED",
            "model_validation": "F2_PRETRUTH_VALIDATION",
            "conformal_calibration": "F3_CALIBRATION_ONLY",
            "prospective_evaluation": "F4_EVALUATION_ONLY",
        }
    )
    control_rows = obs[obs["perturbation"].eq("control")].copy()
    control_rows["target_split"] = "control"
    control_rows["truth_access_phase"] = "F2_PRETRUTH_CONTROL"
    row_access = pd.concat([control_rows, selected_rows], ignore_index=True)
    row_access = row_access[
        [
            "source_row_index",
            "assignment_barcode",
            "perturbation",
            "guide_id",
            "target_split",
            "truth_access_phase",
        ]
    ].sort_values(["truth_access_phase", "perturbation", "guide_id", "source_row_index"])
    row_access.insert(0, "access_row_id", range(1, len(row_access) + 1))

    prior_use = git_prior_use_audit()
    for directory in (MANIFESTS, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_csv(TABLES / "E182_TARGET_METADATA_AUDIT.csv", audit)
    atomic_csv(MANIFESTS / "E182_SELECTED_TARGETS.csv", eligible)
    atomic_csv(MANIFESTS / "E182_GUIDE_TASK_MANIFEST.csv", tasks)
    atomic_csv(MANIFESTS / "E182_ROW_ACCESS_MANIFEST.csv", row_access)
    atomic_json(OUT / "PRIOR_PROJECT_USE_AUDIT.json", prior_use)

    source_lock = {
        "schema": "safeconf_e182_source_lock_v1",
        "dataset_id": "GSE225807_RBP_CRISPRI",
        "source_path": str(SOURCE),
        "source_bytes": SOURCE.stat().st_size,
        "source_sha256": sha256_file(SOURCE),
        "source_shape": shape,
        "shape": shape,
        "x_backend_type": x_backend,
        "geo_accession": "GSE225807",
        "geo_sample_accessions": ["GSM7056649", "GSM7056650"],
        "organism": "Homo sapiens",
        "cell_line": "K562",
        "perturbation_modality": "CRISPR interference",
        "negative_control_label": "control",
        "n_negative_control_cells": int(obs["perturbation"].eq("control").sum()),
        "f0_attestation_path": str(F0_ATTESTATION),
        "f0_attestation_sha256": sha256_file(F0_ATTESTATION),
        "f0_expression_reformat_disclosure": (
            "The downloaded MatrixMarket values were mechanically transposed into "
            "a row-addressable H5AD before the formal access boundary. No expression "
            "summary, perturbation effect, target split, prediction, or error was computed."
        ),
        "formal_freeze_read_obs_and_var_names_only": True,
        "formal_freeze_read_x_values": False,
        "prior_project_use_audit": prior_use,
    }
    atomic_json(OUT / "SOURCE_LOCK.json", source_lock)

    inherited = json.loads(MODEL_SOURCE.read_text())
    model_lock = {
        "schema": "safeconf_e182_model_input_lock_v1",
        "inherited_frozen_assets_from": str(MODEL_SOURCE.relative_to(ROOT)),
        "scgpt_checkpoint_files": inherited["scgpt_checkpoint_files"],
        "scgpt_vocab": {
            "path": str(VOCAB),
            "sha256": sha256_file(VOCAB),
        },
        "gears_go_prior": inherited["gears_go_prior"],
        "random_seeds": list(SEEDS),
        "gene_panel_size": 512,
        "registered_prediction_family": [
            *(f"scGPT_seed{seed}" for seed in SEEDS),
            *(f"GEARS_seed{seed}" for seed in SEEDS),
        ],
        "centroid": "arithmetic mean of all ten registered effect vectors",
        "learned_or_adaptive_upper_model": None,
        "post_e180_stop_rule": (
            "Only the constant target-cluster split-conformal centroid upper "
            "is primary; no learned upper is fit or selected."
        ),
    }
    atomic_json(OUT / "MODEL_INPUT_LOCK.json", model_lock)

    n_calibration = EXPECTED_SPLITS["conformal_calibration"]
    statistical_lock = {
        "schema": "safeconf_e182_statistical_lock_v1",
        "primary_truth_vector": (
            "guide-level mean log1p-normalized expression minus pooled "
            "negative-guide control mean on a frozen 512-gene panel"
        ),
        "registered_family_rms_error": (
            "sqrt(mean_i RMSE(prediction_i, truth)^2) across ten models"
        ),
        "family_diversity_lower": (
            "sqrt(mean_i RMSE(prediction_i, registered_family_centroid)^2)"
        ),
        "worst_member_lower": "maximum pairwise family distance divided by two",
        "centroid_upper": (
            "constant target-cluster split-conformal quantile of the maximum "
            "centroid RMSE across both guides of each calibration target"
        ),
        "family_rms_upper": "sqrt(centroid_upper^2 + family_diversity_lower^2)",
        "worst_member_upper": "centroid_upper + maximum family radius",
        "calibration_cluster_unit": "target gene containing both guide replicates",
        "evaluation_cluster_unit": "target gene containing both guide replicates",
        "target_coverage": TARGET_COVERAGE,
        "finite_sample_rank_one_based": math.ceil(
            (n_calibration + 1) * TARGET_COVERAGE
        ),
        "target_split_counts": EXPECTED_SPLITS,
        "primary_success_gates": {
            "family_rms_lower_violations": 0,
            "worst_member_lower_violations": 0,
            "hilbert_identity_max_absolute_residual": 1e-10,
            "evaluation_target_simultaneous_coverage_minimum": 0.85,
            "truth_access_or_target_split_violations": 0,
        },
        "evaluation_truth_may_modify_method_model_or_threshold": False,
        "ranking_metrics_are_primary": False,
        "tightness_is_descriptive_not_a_selection_gate": True,
    }
    atomic_json(OUT / "STATISTICAL_ANALYSIS_LOCK.json", statistical_lock)

    plan = f"""# E182 GSE225807 注册模型家族证书：事前方案

## 目的

E182 使用此前未进入项目结果的新公开研究 GSE225807，检验冻结的 10 模型家族证书能否在另一套人源 CRISPRi 数据上复现。研究对象是 K562 细胞中的 RBP 靶向扰动；每个纳入靶基因保留两条 guide，不能把 guide 拆到不同数据划分。

## 纳入与划分

只纳入同时满足以下条件的靶基因：表达轴与 scGPT 词表均存在、恰有两条已分配 guide、每条 guide 至少 {MIN_GUIDE_CELLS} 个细胞、两条合计至少 {MIN_TARGET_CELLS} 个细胞。筛选只读取 barcode、guide、靶基因与表达轴名称，不读取表达值。

冻结 {len(eligible)} 个靶基因：训练 {EXPECTED_SPLITS['supervised_train']} 个、模型验证 {EXPECTED_SPLITS['model_validation']} 个、conformal 校准 {EXPECTED_SPLITS['conformal_calibration']} 个、最终评价 {EXPECTED_SPLITS['prospective_evaluation']} 个。划分在细胞数四分位内按加盐 SHA-256 完成。

## 模型与证书

注册家族为 scGPT 和 GEARS 各 5 个随机种子，共 10 个预测向量。家族多样性给家族 RMS 误差的确定性下界；家族直径的一半给最坏成员误差的确定性下界。19 个校准靶基因的两条 guide 先取质心误差最大值，再按有限样本 90% split conformal 的第 {statistical_lock['finite_sample_rank_one_based']} 顺序统计量确定常数上界。

E180 已否定学习型上界的效率优势，因此 E182 不训练 ExtraTrees、Ridge、分位数回归或其他自适应上界。最终评价真值不能用于改模型、换公式、调覆盖阈值或重划数据。

## 成功标准

两类确定性下界必须零违反；Hilbert 恒等式最大残差不超过 `1e-10`；20 个最终评价靶基因中，至少 17 个靶基因的两条 guide 同时被质心上界覆盖；访问与靶基因泄漏必须为零。证书紧致度只作描述，不作为事后筛选条件。
"""
    atomic_text(OUT / "PREREG_ANALYSIS_PLAN.md", plan)
    readme = """# E182 先看这个

E182 是新公开研究 GSE225807 上的一次性注册验证。当前目录处于 F1 元数据冻结阶段，尚未读取任何正式阶段的表达真值。

实验顺序固定为：F0 无分析重排 → F1 元数据冻结 → F2 训练与预真值预测 → F3 仅打开校准靶点 → 提交并双远端留存 → F4 一次性打开最终评价靶点。
"""
    atomic_text(OUT / "README_先看这个.md", readme)
    report = f"""# E182 元数据冻结报告

- 数据：GSE225807，人源 K562 CRISPRi RBP 扰动。
- 原始矩阵：{shape[0]:,} 个细胞 × {shape[1]:,} 个表达特征。
- guide 可分配细胞：{int(obs['assignment_status'].eq('assigned').sum()):,}。
- 阴性 guide 对照细胞：{int(obs['perturbation'].eq('control').sum()):,}。
- 观察到的非对照靶基因：{len(audit):,}。
- 事前规则纳入靶基因：{len(eligible):,}；guide 任务：{len(tasks):,}。
- 最终评价：{EXPECTED_SPLITS['prospective_evaluation']} 个靶基因、{int(tasks['target_split'].eq('prospective_evaluation').sum())} 条 guide 任务。
- F1 冻结读取表达值：**0**。
- 此前项目结果中 GSE225807/GSM7056649/GSM7056650 命中：{prior_use['prior_result_or_dataset_mentions']}。
"""
    atomic_text(REPORTS / "E182_METADATA_FREEZE_REPORT.md", report)

    artifacts = sorted(
        path for path in OUT.rglob("*") if path.is_file() and path.name != "RUN_STATUS.json"
    )
    status = {
        "schema": "safeconf_e182_metadata_freeze_v1",
        "stage": "F1_METADATA_FREEZE",
        "status": "F1_METADATA_FROZEN",
        "freeze_passed": True,
        "runner": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": sha256_file(RUNNER),
        "source_shape": shape,
        "n_observed_targets": len(audit),
        "n_eligible_targets": len(eligible),
        "target_split_counts": EXPECTED_SPLITS,
        "n_guide_tasks": len(tasks),
        "n_control_cells": int(obs["perturbation"].eq("control").sum()),
        "expression_matrix_x_values_read_during_f1_freeze": 0,
        "x_values_read_during_freeze": 0,
        "prior_project_use_mentions": prior_use["prior_result_or_dataset_mentions"],
        "artifact_sha256": {
            path.relative_to(OUT).as_posix(): sha256_file(path) for path in artifacts
        },
    }
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
