#!/usr/bin/env python3
"""E149: expression-blind Replogle K562/RPE1 external replication contract.

This script is deliberately limited to AnnData ``obs``/``var`` metadata and
whole-file byte hashes.  It never indexes or decodes ``X`` and it does not
train a model, form an expression effect, or evaluate a prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb")
SOURCES = {
    "K562": DATA / "ReplogleWeissman2022_K562_essential.h5ad",
    "RPE1": DATA / "ReplogleWeissman2022_rpe1.h5ad",
}
VOCAB = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human/vocab.json"
)
FROZEN_DIRECTION_MODEL = (
    ROOT
    / "docs/实验结果/E135_directional_risk_lodo_20260714/"
    "E135_FROZEN_DIRECTION_MODEL.json"
)
OUT = ROOT / "docs/实验结果/E149_replogle_two_cellline_contract_20260714"
TABLES, MANIFESTS, REPORTS = OUT / "tables", OUT / "manifests", OUT / "reports"

SELECTION_SEED = "E149_REPLOGLE_K562_RPE1_EXTERNAL_20260714_V1"
N_PERTURBATIONS = 128
MIN_CELLS_PER_CONTEXT_PERTURBATION = 100
MIN_BATCHES_PER_CONTEXT_PERTURBATION = 10
PERTURBATION_UNSEEN_FRACTION = 0.20
N_VALIDATION_PAIRS = 16
N_RANDOM_SEEN_TEST_PAIRS = 16
N_CLUSTER_BOOTSTRAP = 3000
CONTROL_LABEL = "control"
EXCLUDED_LABELS = {"", "control", "ctrl", "nan", "none", "non-targeting"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    payload = "|".join([SELECTION_SEED, *map(str, parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_unit_interval(*parts: str) -> float:
    return int(stable_hash(*parts)[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def human_token(gene: str) -> str:
    return str(gene).upper()


def load_metadata():
    """Load only obs/var metadata from backed AnnData objects."""
    metadata = {}
    source_rows = []
    for context, path in SOURCES.items():
        data = ad.read_h5ad(path, backed="r")
        obs = data.obs.copy()
        perturbation = obs["perturbation"].astype(str)
        obs = obs.assign(_perturbation=perturbation)
        counts = perturbation.value_counts()
        grouped = obs.groupby("_perturbation", observed=True)
        metadata[context] = {
            "shape": [int(data.n_obs), int(data.n_vars)],
            "counts": counts.to_dict(),
            "batches": grouped["batch"].nunique().to_dict(),
            "guides": grouped["guide_id"].nunique().to_dict(),
            "transcripts": grouped["transcript"].nunique().to_dict(),
            "gene_ids": grouped["gene_id"].nunique().to_dict(),
            "genes": set(data.var_names.astype(str)),
            "obs_columns": list(map(str, data.obs.columns)),
            "var_columns": list(map(str, data.var.columns)),
        }
        control_mask = perturbation.eq(CONTROL_LABEL)
        source_rows.append(
            {
                "context": context,
                "source_path": str(path),
                "n_cells": int(data.n_obs),
                "n_expression_genes": int(data.n_vars),
                "n_perturbation_labels_including_control": int(counts.size),
                "n_noncontrol_perturbations": int(
                    sum(str(value).lower() not in EXCLUDED_LABELS for value in counts.index)
                ),
                "n_batches": int(obs["batch"].nunique()),
                "n_control_cells": int(control_mask.sum()),
                "n_control_batches": int(obs.loc[control_mask, "batch"].nunique()),
                "control_label": CONTROL_LABEL,
                "cell_line_labels": "+".join(sorted(obs["cell_line"].astype(str).unique())),
                "perturbation_type_labels": "+".join(
                    sorted(obs["perturbation_type"].astype(str).unique())
                ),
                "expression_matrix_X_decoded": False,
            }
        )
        data.file.close()
    return metadata, pd.DataFrame(source_rows)


def eligibility_table(metadata, vocab: set[str]) -> pd.DataFrame:
    contexts = sorted(metadata)
    shared = set(metadata[contexts[0]]["counts"])
    for context in contexts[1:]:
        shared &= set(metadata[context]["counts"])
    shared = sorted(
        perturbation
        for perturbation in shared
        if str(perturbation).lower() not in EXCLUDED_LABELS
    )
    rows = []
    for perturbation in shared:
        row = {
            "perturbation": perturbation,
            "eligibility_hash": stable_hash("eligibility", perturbation),
            "in_all_expression_gene_axes": all(
                perturbation in metadata[context]["genes"] for context in contexts
            ),
            "scgpt_token": human_token(perturbation),
            "in_scgpt_vocabulary": human_token(perturbation) in vocab,
        }
        for context in contexts:
            row[f"n_cells_{context}"] = int(metadata[context]["counts"].get(perturbation, 0))
            row[f"n_batches_{context}"] = int(metadata[context]["batches"].get(perturbation, 0))
            row[f"n_guide_entities_{context}"] = int(metadata[context]["guides"].get(perturbation, 0))
            row[f"n_transcript_groups_{context}"] = int(metadata[context]["transcripts"].get(perturbation, 0))
            row[f"n_gene_ids_{context}"] = int(metadata[context]["gene_ids"].get(perturbation, 0))
        row["minimum_cells_across_contexts"] = min(
            row[f"n_cells_{context}"] for context in contexts
        )
        row["minimum_batches_across_contexts"] = min(
            row[f"n_batches_{context}"] for context in contexts
        )
        row["eligible"] = bool(
            row["in_all_expression_gene_axes"]
            and row["in_scgpt_vocabulary"]
            and row["minimum_cells_across_contexts"] >= MIN_CELLS_PER_CONTEXT_PERTURBATION
            and row["minimum_batches_across_contexts"] >= MIN_BATCHES_PER_CONTEXT_PERTURBATION
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("perturbation").reset_index(drop=True)


def select_perturbations(eligibility: pd.DataFrame) -> pd.DataFrame:
    eligible = eligibility.loc[eligibility["eligible"]].copy()
    eligible["selection_hash"] = eligible["perturbation"].map(
        lambda value: stable_hash("select", value)
    )
    eligible = eligible.sort_values(["selection_hash", "perturbation"]).head(N_PERTURBATIONS)
    if len(eligible) != N_PERTURBATIONS:
        raise RuntimeError(
            f"only {len(eligible)} eligible perturbations; expected {N_PERTURBATIONS}"
        )
    eligible.insert(0, "selection_rank", range(1, len(eligible) + 1))
    eligible["selected_without_expression_values_or_outcomes"] = True
    return eligible.reset_index(drop=True)


def build_manifest(selected: pd.DataFrame) -> pd.DataFrame:
    contexts = sorted(SOURCES)
    perturbations = selected["perturbation"].astype(str).tolist()
    count_lookup = {
        context: selected.set_index("perturbation")[f"n_cells_{context}"].to_dict()
        for context in contexts
    }
    rows = []
    for fold_number, heldout in enumerate(contexts, start=1):
        source = next(context for context in contexts if context != heldout)
        ordered = sorted(
            perturbations,
            key=lambda value: stable_hash("fold", heldout, "perturbation", value),
        )
        n_unseen = int(math.ceil(PERTURBATION_UNSEEN_FRACTION * len(ordered)))
        unseen = set(ordered[:n_unseen])
        seen = set(ordered[n_unseen:])
        source_seen = sorted(
            seen,
            key=lambda value: stable_hash("fold", heldout, "source_seen", value),
        )
        validation = set(source_seen[:N_VALIDATION_PAIRS])
        random_seen_test = set(
            source_seen[
                N_VALIDATION_PAIRS : N_VALIDATION_PAIRS + N_RANDOM_SEEN_TEST_PAIRS
            ]
        )
        fold_id = f"Replogle_cellline_holdout_{fold_number}_{heldout}"
        for context in contexts:
            for perturbation in perturbations:
                if context == source and perturbation in seen:
                    if perturbation in validation:
                        split, setting = "val", "validation_pair"
                    elif perturbation in random_seen_test:
                        split, setting = "test", "random_seen_pair"
                    else:
                        split, setting = "train", "training_pair"
                elif context == source:
                    split, setting = "test", "perturbation_unseen"
                elif perturbation in seen:
                    split, setting = "test", "context_unseen"
                else:
                    split, setting = "test", "context_and_perturbation_unseen"
                train_fraction_u = stable_unit_interval(
                    "train_fraction", fold_id, context, perturbation
                )
                rows.append(
                    {
                        "dataset": "Replogle_two_cellline",
                        "modality": "CRISPRi_gene_knockdown_cellline_shift",
                        "fold_id": fold_id,
                        "heldout_context": heldout,
                        "source_contexts": source,
                        "split": split,
                        "setting": setting,
                        "context": context,
                        "cell_line": context,
                        "perturbation": perturbation,
                        "n_cells": int(count_lookup[context][perturbation]),
                        "perturbation_seen_in_training": perturbation in seen,
                        "context_seen_in_training": context == source,
                        "heldout_context_control_state_available_at_inference": context == heldout,
                        "primary_analysis": split == "test" and context == heldout,
                        "selected_without_expression_values": True,
                        "train_fraction_hash_u": train_fraction_u,
                        "in_train_fraction_25": split == "train" and train_fraction_u < 0.25,
                        "in_train_fraction_50": split == "train" and train_fraction_u < 0.50,
                        "in_train_fraction_75": split == "train" and train_fraction_u < 0.75,
                        "in_train_fraction_100": split == "train",
                    }
                )
    manifest = pd.DataFrame(rows)
    primary = manifest.loc[manifest["primary_analysis"]]
    if primary.groupby(["context", "perturbation"]).size().max() != 1:
        raise RuntimeError("primary context x perturbation tasks are not unique")
    return manifest


def write_contract(
    eligibility: pd.DataFrame,
    selected: pd.DataFrame,
    manifest: pd.DataFrame,
    source_audit: pd.DataFrame,
) -> None:
    primary = manifest.loc[manifest["primary_analysis"]]
    contract = [
        "# E149｜Replogle K562/RPE1 外部复制预注册合同",
        "",
        "## 数据和信息边界",
        "",
        "- 使用 Replogle 等人的 K562 essential 与 RPE1 CRISPRi Perturb-seq 数据（Cell 2022，DOI: 10.1016/j.cell.2022.05.013；PMCID: PMC9380471）。本地输入是 scPerturb 统一整理后的 h5ad，源文件哈希已冻结。",
        "- 两个细胞系来自同一研究，均为公开回顾性数据；它们不是新做的湿实验，也不检验跨研究批次迁移。",
        "- RPE1 原始筛选库包含 common-essential genes 以及依据 K562 现象挑选的部分基因。因此 E149 的结论范围限定为共享的高覆盖 CRISPRi 靶标，不能外推成全基因组随机靶标的细胞系泛化。",
        "- 本次冻结只读取 AnnData 的 `obs`、`var_names`、矩阵形状和文件字节哈希。脚本没有索引或解码 `X`，没有形成表达效应、模型预测或误差。",
        f"- K562 control={int(source_audit.set_index('context').loc['K562','n_control_cells'])} 个；RPE1 control={int(source_audit.set_index('context').loc['RPE1','n_control_cells'])} 个。",
        "",
        "## 固定选择规则",
        "",
        f"1. 扰动标签同时存在于 K562 与 RPE1，排除 control/non-targeting 等标签；",
        f"2. 靶基因同时位于两个表达基因身份轴，并位于固定 scGPT whole-human 词表；",
        f"3. 每个 cell line × perturbation 至少 {MIN_CELLS_PER_CONTEXT_PERTURBATION} 个细胞，且覆盖至少 {MIN_BATCHES_PER_CONTEXT_PERTURBATION} 个 batch；",
        f"4. 对全部 {int(eligibility.eligible.sum())} 个合格扰动按预先固定的 SHA-256 种子排序，取前 {len(selected)} 个。细胞数只作预设门槛，不参与门槛后的优先排序。",
        "",
        "## 外层划分",
        "",
        "- 两个外层 folds 分别留出整个 K562 或 RPE1 的扰动效应。留出细胞系的 control 均值可作为推理时基础状态；留出细胞系的 perturbed expression 不进入训练、验证、校准或风险打分。该任务应称为 control-observed cross-cell-line prediction，而不是完全看不见目标细胞系的 zero-shot。",
        f"- 每折哈希留出 {math.ceil(PERTURBATION_UNSEEN_FRACTION * len(selected))} 个 perturbations；source cell line 内固定 {N_VALIDATION_PAIRS} 个 validation pairs、{N_RANDOM_SEEN_TEST_PAIRS} 个 random seen test pairs，其余 seen pairs 用于训练。",
        f"- 主分析只使用 heldout cell line 的 {len(primary)} 个 fold-specific test rows（每折 {len(primary)//2} 个）；每个 context × perturbation 只出现一次。source cell line 的 random-seen 和 perturbation-unseen rows 仅作次要诊断。",
        "",
        "## 固定模型流程",
        "",
        "- 完整运行后按 E112/E138 流程构建 control-only 512 基因面板，分别训练 scGPT 与 GEARS；训练 epoch、early stopping、优化器、验证校准、预测记录 schema 均不得根据 Replogle 测试结果改动。",
        "- 使用 E135 已冻结的四特征方向风险模型。Replogle 测试真值不得参与风险分数、标准化、阈值或模型系数。",
        "",
        "## 主要终点和通过规则",
        "",
        "- 两个主要终点为两预测器平均 centered Pearson error 与 centered cosine error；每折分别算 Spearman，再对两个 cell-line folds 等权平均。",
        "- Pearson/cosine error 在 fold 内转换为 percentile rank 后取平均，形成复合方向误差 rank。",
        f"- 按 perturbation 整簇重采样 {N_CLUSTER_BOOTSTRAP} 次；同一扰动在两个细胞系的记录一起进入或离开，避免把配对细胞系记录当独立样本。",
        "- 外部复制 gate：两个主要终点的 fold-macro Spearman 均大于 0，并且复合方向误差 rank 的 perturbation-cluster bootstrap 95% CI 下界大于 0。",
        "- 同时完整报告 predicted magnitude、model disagreement、原 pair-risk、冻结方向风险及训练扰动质心简单基线；不得只展示有利分数。",
        "",
        "## 次要分析",
        "",
        "- 分别报告 context_unseen 与 context_and_perturbation_unseen；source 内 perturbation_unseen 只作机制诊断。",
        "- 报告 absolute RMSE、各上游模型误差、两模型平均/最大误差，以及风险分数相对 predicted magnitude 的 bootstrap 差值。",
        "- 若 gate 未通过，不允许在 Replogle 上重调 E135 系数后仍称独立复制；失败结果照常保留。",
    ]
    (OUT / "PREREG_ANALYSIS_PLAN.md").write_text("\n".join(contract) + "\n")
    report = [
        "# E149｜元数据审计与冻结结果",
        "",
        f"- 两个源文件共有 {len(eligibility)} 个非对照扰动标签；满足全部门槛 {int(eligibility.eligible.sum())} 个。",
        f"- SHA-256 固定选择 {len(selected)} 个；跨细胞系最少细胞数 {int(selected.minimum_cells_across_contexts.min())}，最少 batch 数 {int(selected.minimum_batches_across_contexts.min())}。",
        f"- 合同 manifest 共 {len(manifest)} 行，test={int(manifest.split.eq('test').sum())}，主分析 heldout-cell-line test={int(manifest.primary_analysis.sum())}。",
        "- 主分析的 context × perturbation 唯一；后续按 perturbation 聚类 bootstrap。",
        "- 目前只完成元数据冻结。尚未读取表达矩阵、构建资产、训练 scGPT/GEARS 或计算任何终点。",
    ]
    (REPORTS / "E149_CONTRACT_REPORT.md").write_text("\n".join(report) + "\n")
    (OUT / "README_先看这个.md").write_text(
        "# E149 先看这个\n\n"
        "先读 `PREREG_ANALYSIS_PLAN.md`；选择表在 `tables/`，正式划分在 `manifests/E149_TASK_MANIFEST.csv`。当前仅完成 metadata-only freeze。\n"
    )


def main() -> None:
    for directory in [OUT, TABLES, MANIFESTS, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in [*SOURCES.values(), VOCAB, FROZEN_DIRECTION_MODEL] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    metadata, source_audit = load_metadata()
    vocab = set(json.loads(VOCAB.read_text()))
    frozen_model = json.loads(FROZEN_DIRECTION_MODEL.read_text())
    if any("replogle" in str(dataset).lower() for dataset in frozen_model["source_datasets"]):
        raise RuntimeError("Replogle unexpectedly appears in E135 model-development sources")
    eligibility = eligibility_table(metadata, vocab)
    selected = select_perturbations(eligibility)
    manifest = build_manifest(selected)
    split_summary = (
        manifest.groupby(["fold_id", "setting", "split", "primary_analysis"], as_index=False)
        .agg(
            n_tasks=("perturbation", "size"),
            n_unique_perturbations=("perturbation", "nunique"),
            min_cells=("n_cells", "min"),
            median_cells=("n_cells", "median"),
        )
    )

    source_audit.to_csv(TABLES / "E149_SOURCE_METADATA_AUDIT.csv", index=False)
    eligibility.to_csv(TABLES / "E149_ELIGIBILITY_AUDIT.csv", index=False)
    selected.to_csv(TABLES / "E149_SELECTED_PERTURBATIONS.csv", index=False)
    split_summary.to_csv(TABLES / "E149_SPLIT_SUMMARY.csv", index=False)
    manifest.to_csv(MANIFESTS / "E149_TASK_MANIFEST.csv", index=False)
    write_contract(eligibility, selected, manifest, source_audit)

    selection_rules = {
        "selection_seed": SELECTION_SEED,
        "n_perturbations": N_PERTURBATIONS,
        "minimum_cells_per_context_perturbation": MIN_CELLS_PER_CONTEXT_PERTURBATION,
        "minimum_batches_per_context_perturbation": MIN_BATCHES_PER_CONTEXT_PERTURBATION,
        "perturbation_unseen_fraction": PERTURBATION_UNSEEN_FRACTION,
        "n_validation_pairs": N_VALIDATION_PAIRS,
        "n_random_seen_test_pairs": N_RANDOM_SEEN_TEST_PAIRS,
        "primary_analysis": "split == test and context == heldout_context",
        "cluster_bootstrap_unit": "perturbation",
        "cluster_bootstrap_draws": N_CLUSTER_BOOTSTRAP,
    }
    rules_sha256 = hashlib.sha256(
        json.dumps(selection_rules, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifacts = [
        OUT / "PREREG_ANALYSIS_PLAN.md",
        REPORTS / "E149_CONTRACT_REPORT.md",
        TABLES / "E149_SOURCE_METADATA_AUDIT.csv",
        TABLES / "E149_ELIGIBILITY_AUDIT.csv",
        TABLES / "E149_SELECTED_PERTURBATIONS.csv",
        TABLES / "E149_SPLIT_SUMMARY.csv",
        MANIFESTS / "E149_TASK_MANIFEST.csv",
    ]
    source_hashes = {context: sha256(path) for context, path in SOURCES.items()}
    status = {
        "experiment": "E149_replogle_two_cellline_contract",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_before_expression_values_predictions_or_errors",
        "source_study": "Replogle et al. 2022 Perturb-seq",
        "source_article_doi": "10.1016/j.cell.2022.05.013",
        "source_article_pmcid": "PMC9380471",
        "local_asset_provenance": "scPerturb-standardized h5ad",
        "scope_limit": "shared high-coverage CRISPRi targets within one study; not genome-wide random-target or cross-study generalization",
        "sources": {
            context: {
                "path": str(path),
                "sha256": source_hashes[context],
                "shape": metadata[context]["shape"],
            }
            for context, path in SOURCES.items()
        },
        "vocabulary_path": str(VOCAB),
        "vocabulary_sha256": sha256(VOCAB),
        "frozen_direction_model_path": str(FROZEN_DIRECTION_MODEL.relative_to(ROOT)),
        "frozen_direction_model_sha256": sha256(FROZEN_DIRECTION_MODEL),
        "frozen_direction_model_source_datasets": frozen_model["source_datasets"],
        "replogle_absent_from_direction_model_development_sources": True,
        "selection_rules": selection_rules,
        "selection_rules_sha256": rules_sha256,
        "n_shared_noncontrol_perturbations": len(eligibility),
        "n_eligible_perturbations": int(eligibility["eligible"].sum()),
        "n_selected_perturbations": len(selected),
        "minimum_selected_cells_across_contexts": int(selected.minimum_cells_across_contexts.min()),
        "minimum_selected_batches_across_contexts": int(selected.minimum_batches_across_contexts.min()),
        "n_manifest_rows": len(manifest),
        "n_test_rows_all_diagnostics": int(manifest.split.eq("test").sum()),
        "n_primary_heldout_context_test_rows": int(manifest.primary_analysis.sum()),
        "n_primary_unique_context_perturbation_tasks": int(
            manifest.loc[manifest.primary_analysis, ["context", "perturbation"]]
            .drop_duplicates()
            .shape[0]
        ),
        "metadata_fields_used": [
            "obs.perturbation",
            "obs.batch",
            "obs.guide_id",
            "obs.transcript",
            "obs.gene_id",
            "obs.cell_line",
            "obs.perturbation_type",
            "var_names",
            "shape",
        ],
        "expression_matrix_X_indexed_or_decoded": False,
        "expression_values_or_effect_sizes_used_for_selection_or_split": False,
        "prediction_or_error_used_for_selection_or_split": False,
        "model_training_executed": False,
        "artifact_sha256": {
            str(path.relative_to(OUT)): sha256(path) for path in artifacts
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
