#!/usr/bin/env python3
"""Freeze E180 XuCao2023 guide-replicate confirmation without reading X values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E180_xucao_fresh_guide_certificate_20260723"
MANIFESTS = OUT / "manifests"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
SOURCE = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/XuCao2023.h5ad"
)
E177_MODEL_LOCK = (
    ROOT
    / "docs/实验结果/E177_sunshine_external_certificate_20260719/MODEL_INPUT_LOCK.json"
)
VOCAB = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human/vocab.json"
)

MIN_CELLS_PER_GUIDE = 20
MIN_GUIDES_PER_TARGET = 2
SPLIT_CYCLE = (
    "supervised_train",
    "supervised_train",
    "model_validation",
    "conformal_calibration",
    "prospective_evaluation",
)
SPLIT_SALT = "E180_XUCAO_GUIDE_TARGET_SPLIT_V1_20260723"


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def load_vocab() -> set[str]:
    value = json.loads(VOCAB.read_text())
    return set(value if isinstance(value, list) else value.keys())


def assign_splits(eligible: pd.DataFrame) -> pd.DataFrame:
    work = eligible.copy()
    work["cell_count_quartile"] = pd.qcut(
        work["n_target_cells"],
        q=4,
        labels=("Q1", "Q2", "Q3", "Q4"),
        duplicates="drop",
    ).astype(str)
    work["guide_count_stratum"] = np.where(
        work["n_eligible_guides"] >= 3, "THREE_PLUS_GUIDES", "TWO_GUIDES"
    )
    work["selection_hash"] = work["perturbation"].map(stable_hash)
    assigned: list[pd.DataFrame] = []
    for _, block in work.groupby(
        ["cell_count_quartile", "guide_count_stratum"], sort=True, observed=True
    ):
        block = block.sort_values("selection_hash").copy()
        block["target_split"] = [
            SPLIT_CYCLE[index % len(SPLIT_CYCLE)] for index in range(len(block))
        ]
        assigned.append(block)
    result = pd.concat(assigned, ignore_index=True)
    result = result.sort_values(["target_split", "selection_hash"]).reset_index(drop=True)
    return result


def main() -> None:
    for directory in (MANIFESTS, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(SOURCE)
    vocab_sha = sha256_file(VOCAB)
    vocab = load_vocab()

    adata = ad.read_h5ad(SOURCE, backed="r")
    obs = adata.obs.copy()
    var_names = set(map(str, adata.var_names))
    shape = (int(adata.n_obs), int(adata.n_vars))
    x_backend_type = type(adata.X).__name__

    obs["perturbation"] = obs["perturbation"].astype(str)
    obs["guide_id"] = obs["guide_id"].astype(str)
    target_obs = obs[obs["perturbation"] != "control"].copy()
    guide_counts = (
        target_obs.groupby(["perturbation", "guide_id"], observed=True)
        .size()
        .rename("n_guide_cells")
        .reset_index()
    )
    guide_counts["eligible_guide"] = guide_counts["n_guide_cells"] >= MIN_CELLS_PER_GUIDE

    target_counts = (
        target_obs.groupby("perturbation", observed=True)
        .agg(
            n_target_cells=("guide_id", "size"),
            n_guides=("guide_id", "nunique"),
        )
        .reset_index()
    )
    eligible_guide_counts = (
        guide_counts[guide_counts["eligible_guide"]]
        .groupby("perturbation", observed=True)
        .agg(
            n_eligible_guides=("guide_id", "nunique"),
            min_cells_per_eligible_guide=("n_guide_cells", "min"),
            max_cells_per_eligible_guide=("n_guide_cells", "max"),
        )
        .reset_index()
    )
    audit = target_counts.merge(eligible_guide_counts, on="perturbation", how="left")
    for column in (
        "n_eligible_guides",
        "min_cells_per_eligible_guide",
        "max_cells_per_eligible_guide",
    ):
        audit[column] = audit[column].fillna(0).astype(int)
    audit["is_single_gene_label"] = audit["perturbation"].str.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*"
    )
    audit["in_expression_var"] = audit["perturbation"].isin(var_names)
    audit["in_scgpt_vocab"] = audit["perturbation"].isin(vocab)
    audit["eligible_e180_target"] = (
        audit["is_single_gene_label"]
        & audit["in_expression_var"]
        & audit["in_scgpt_vocab"]
        & (audit["n_eligible_guides"] >= MIN_GUIDES_PER_TARGET)
    )
    audit["selection_used_expression_matrix_or_truth_effect"] = False
    eligible = assign_splits(audit[audit["eligible_e180_target"]].copy())
    if len(eligible) < 100:
        raise RuntimeError(f"too few eligible targets for E180: {len(eligible)}")
    split_counts = eligible["target_split"].value_counts()
    if (split_counts < 20).any():
        raise RuntimeError(f"undersized E180 target split: {split_counts.to_dict()}")

    selected_guides = guide_counts[
        guide_counts["eligible_guide"]
        & guide_counts["perturbation"].isin(eligible["perturbation"])
    ].copy()
    selected_guides = selected_guides.merge(
        eligible[
            [
                "perturbation",
                "target_split",
                "cell_count_quartile",
                "guide_count_stratum",
                "selection_hash",
            ]
        ],
        on="perturbation",
        how="left",
        validate="many_to_one",
    )
    selected_guides["task_id"] = (
        "E180::" + selected_guides["perturbation"] + "::" + selected_guides["guide_id"]
    )
    selected_guides["target_cluster_id"] = selected_guides["perturbation"]
    selected_guides["truth_access_phase"] = selected_guides["target_split"].map(
        {
            "supervised_train": "F2_PRETRUTH_SUPERVISED",
            "model_validation": "F2_PRETRUTH_VALIDATION",
            "conformal_calibration": "F3_CALIBRATION_ONLY",
            "prospective_evaluation": "F4_EVALUATION_ONLY",
        }
    )
    selected_guides["primary_task_unit"] = "guide_replicate_effect_vs_pooled_control"
    selected_guides["cell_cycle_phase_is_primary_context"] = False
    selected_guides["selection_used_expression_matrix_or_truth_effect"] = False

    control_phase = (
        obs[obs["perturbation"] == "control"]["Cell_cycle_phase"]
        .astype(str)
        .value_counts()
        .rename_axis("cell_cycle_phase")
        .reset_index(name="n_control_cells")
    )
    target_phase = (
        target_obs[target_obs["perturbation"].isin(eligible["perturbation"])]
        .groupby(["perturbation", "Cell_cycle_phase"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )

    atomic_csv(TABLES / "E180_TARGET_METADATA_AUDIT.csv", audit)
    atomic_csv(TABLES / "E180_CONTROL_PHASE_COUNTS.csv", control_phase)
    atomic_csv(TABLES / "E180_SELECTED_TARGET_PHASE_COUNTS.csv", target_phase)
    atomic_csv(MANIFESTS / "E180_SELECTED_TARGETS.csv", eligible)
    atomic_csv(MANIFESTS / "E180_GUIDE_TASK_MANIFEST.csv", selected_guides)

    source_lock = {
        "schema": "safeconf_e180_source_lock_v1",
        "dataset_id": "XuCao2023",
        "source_path": str(SOURCE),
        "source_bytes": SOURCE.stat().st_size,
        "source_sha256": source_sha,
        "official_processed_collection": "scPerturb Zenodo record 13350497",
        "collection_doi": "10.5281/zenodo.13350497",
        "shape": list(shape),
        "x_backend_type": x_backend_type,
        "obs_columns": list(map(str, obs.columns)),
        "var_names_read_as_metadata": True,
        "x_values_indexed_or_materialized_during_freeze": False,
        "source_hashing_read_raw_file_bytes_but_did_not_decode_x_values": True,
        "control_label": "control",
        "n_control_cells": int((obs["perturbation"] == "control").sum()),
        "primary_task_unit": "guide-level perturbation effect relative to pooled controls",
        "primary_cluster_unit": "perturbation gene; all eligible guides remain together",
        "cell_cycle_phase_role": (
            "metadata audit and sensitivity analysis only; not treated as a pre-perturbation context"
        ),
    }
    atomic_json(OUT / "SOURCE_LOCK.json", source_lock)

    inherited_model_lock = json.loads(E177_MODEL_LOCK.read_text())
    model_lock = {
        "schema": "safeconf_e180_model_input_lock_v1",
        "inherited_frozen_assets_from": str(E177_MODEL_LOCK.relative_to(ROOT)),
        "scgpt_checkpoint_files": inherited_model_lock["scgpt_checkpoint_files"],
        "scgpt_vocab": {
            "path": str(VOCAB),
            "sha256": vocab_sha,
        },
        "gears_go_prior": inherited_model_lock["gears_go_prior"],
        "random_seeds": [3407, 3408, 3409, 3410, 3411],
        "gene_panel_size": 512,
        "underlying_predictors": ["scGPT", "GEARS"],
        "adaptive_upper_base": {
            "method": "ExtraTreesRegressor",
            "n_estimators": 200,
            "min_samples_leaf": 10,
            "max_features": 0.70,
            "features": [
                "predicted_magnitude",
                "pair_lower_bound",
                "scgpt_magnitude",
                "gears_magnitude",
                "model_cosine",
                "ensemble_mean_abs",
                "ensemble_max_abs",
                "disagreement_mean_abs",
                "sign_agreement",
                "ensemble_abs_p90",
                "ensemble_abs_p99",
                "disagreement_abs_p90",
                "disagreement_abs_p99",
                "ensemble_gene_std",
                "disagreement_gene_std",
                "scgpt_seed_spread",
                "gears_seed_spread",
                "ensemble_seed_spread",
            ],
            "training_truth_role": "model_validation targets only",
            "selection_source": (
                "E179 retrospective nested benchmark; no E180 expression truth used"
            ),
        },
    }
    atomic_json(OUT / "MODEL_INPUT_LOCK.json", model_lock)

    statistical_lock = {
        "schema": "safeconf_e180_statistical_lock_v1",
        "primary_outcome": "pair_mean_rmse_across_scgpt_and_gears",
        "deterministic_lower_bound": "rmse(scgpt_prediction-gears_prediction)/2",
        "lower_bound_success_gate": "zero violations for pair mean and pair max errors",
        "upper_bound_target": "90% target-simultaneous coverage across all eligible guides",
        "conformal_cluster_unit": "perturbation gene",
        "calibration_nonconformity": (
            "maximum over eligible guides of pair_mean_rmse minus frozen pretruth base"
        ),
        "finite_sample_rank": "ceil((n_calibration_targets+1)*0.90)",
        "primary_adaptive_method": "extra_trees_vector",
        "efficiency_reference": "constant split conformal on the identical target split",
        "comparators": [
            "constant",
            "predicted_magnitude",
            "magnitude_plus_pair_lower",
            "extra_trees_vector",
        ],
        "prospective_success_gates": {
            "pair_lower_bound_violations": 0,
            "evaluation_target_simultaneous_coverage_minimum": 0.85,
            "nominal_coverage_reported_separately": 0.90,
            "adaptive_mean_upper_must_not_exceed_constant_mean_upper": True,
            "truth_access_or_split_violation": 0,
        },
        "ranking_status": "secondary_diagnostic_only",
        "evaluation_truth_forbidden_before": (
            "pretruth prediction release and calibration model freeze are committed and on both remotes"
        ),
        "no_posttruth_rescue": [
            "no target replacement",
            "no guide replacement",
            "no feature replacement",
            "no hyperparameter replacement",
            "no seed replacement",
            "no endpoint replacement",
        ],
    }
    atomic_json(OUT / "STATISTICAL_ANALYSIS_LOCK.json", statistical_lock)

    access_log = {
        "schema": "safeconf_e180_hdf5_value_access_log_v1",
        "phase": "F1_METADATA_FREEZE",
        "source": str(SOURCE),
        "obs_and_var_metadata_read": True,
        "x_backend_handle_seen_for_type_audit": True,
        "x_values_indexed_or_materialized": False,
        "calibration_truth_effects_computed": False,
        "evaluation_truth_effects_computed": False,
        "next_allowed_phase": (
            "F2_PRETRUTH_ASSET_BUILD only after freeze code and manifests are committed and pushed"
        ),
    }
    atomic_json(OUT / "HDF5_VALUE_ACCESS_LOG.json", access_log)
    adata.file.close()

    plan = f"""# E180 预注册分析计划：XuCao2023 guide 复现确认

## 研究问题

在一个未进入 E176–E179 正式结果的新研究中，scGPT 与 GEARS 的预测距离能否继续给出零违例的确定性误差下界；E179 冻结的向量特征上界能否在保持靶点级覆盖的同时，比常数 split conformal 更窄。

## 数据与任务

- 官方处理文件：`XuCao2023.h5ad`，{shape[0]:,} 个细胞、{shape[1]:,} 个表达变量。
- 仅纳入基因名同时存在于表达轴和 scGPT 词表、至少有 {MIN_GUIDES_PER_TARGET} 个 guide、且每个 guide 至少 {MIN_CELLS_PER_GUIDE} 个细胞的靶点。
- 主任务是一条 guide 相对 pooled control 的 512 维表达效应。
- 同一基因的全部 guide 始终留在同一分区；评价覆盖事件要求该基因所有 guide 同时被上界覆盖。
- `Cell_cycle_phase` 是扰动后观测标签，不能冒充预先给定的细胞背景；只做敏感性分析。

## 冻结分区

采用只依赖靶点名、细胞数和 guide 数的确定性哈希分层，比例为 40% supervised train、20% model validation、20% conformal calibration、20% prospective evaluation。表达矩阵和真值误差不参与选择。

## 模型和证书

1. scGPT 与 GEARS 使用五个固定随机种子；
2. 确定性证书：`pair_lower = RMSE(p_scGPT-p_GEARS)/2`；
3. 上界比较：常数、预测幅度、`max(预测幅度, pair_lower)`、E179 冻结的 ExtraTrees 向量特征基线；
4. ExtraTrees 只用 model-validation 靶点的预测特征和误差拟合；
5. conformal correction 只用 calibration 靶点，按基因簇最大残差校准；
6. evaluation 真值只允许在预测释放和校准模型同时推送到 GitHub/Gitee 后打开一次。

## 不允许的修改

评价真值打开后，不换靶点、guide、特征、模型、seed、阈值或 endpoint。若主结果失败，保留失败并进入新实验编号。
"""
    atomic_text(OUT / "PREREG_ANALYSIS_PLAN.md", plan)

    split_text = ", ".join(
        f"{name}={count}" for name, count in split_counts.sort_index().items()
    )
    report = f"""# E180 元数据冻结报告

E180 已在不读取表达矩阵数值的条件下完成新研究冻结。

- 数据：XuCao2023，{shape[0]:,} 个细胞，{shape[1]:,} 个表达变量；
- control：{int((obs['perturbation'] == 'control').sum()):,} 个细胞；
- 合格靶点：{len(eligible)} 个；
- 合格 guide 任务：{len(selected_guides)} 个；
- 分区：{split_text}；
- 靶点选择使用表达真值或既往误差：否；
- X 数值读取：0；
- 主分组单位：基因；同一基因的 guide 不跨分区。

下一步必须先提交并推送本冻结文件，再构建 F2 资产。E179 的自适应上界规则已经写入 `MODEL_INPUT_LOCK.json`，后续不得依据 E180 评价结果改动。
"""
    atomic_text(REPORTS / "E180_METADATA_FREEZE_REPORT.md", report)
    atomic_text(
        OUT / "README_先看这个.md",
        "# E180 XuCao2023 新研究确认\n\n"
        "当前状态：**F1 元数据冻结完成，表达真值未读取。**\n\n"
        "- [预注册分析计划](PREREG_ANALYSIS_PLAN.md)\n"
        "- [元数据冻结报告](reports/E180_METADATA_FREEZE_REPORT.md)\n"
        "- [目标清单](manifests/E180_SELECTED_TARGETS.csv)\n"
        "- [guide 任务清单](manifests/E180_GUIDE_TASK_MANIFEST.csv)\n"
        "- [统计锁](STATISTICAL_ANALYSIS_LOCK.json)\n",
    )

    input_paths = [Path(__file__).resolve(), SOURCE, VOCAB, E177_MODEL_LOCK]
    hashes = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "semantic_x_value_access": False,
            }
            for path in input_paths
        ]
    )
    atomic_csv(TABLES / "E180_FREEZE_INPUT_HASHES.csv", hashes)

    status = {
        "experiment": "E180_xucao_fresh_guide_certificate",
        "status": "F1_METADATA_FROZEN",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source_sha256": source_sha,
        "n_cells": shape[0],
        "n_variables": shape[1],
        "n_control_cells": int((obs["perturbation"] == "control").sum()),
        "n_eligible_targets": len(eligible),
        "n_guide_tasks": len(selected_guides),
        "target_split_counts": {key: int(value) for key, value in split_counts.items()},
        "x_values_read_during_freeze": 0,
        "calibration_truth_opened": False,
        "evaluation_truth_opened": False,
    }
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
