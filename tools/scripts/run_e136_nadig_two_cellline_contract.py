#!/usr/bin/env python3
"""E136: freeze an expression-blind two-cell-line Nadig confirmation contract."""

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
    "HepG2": DATA / "NadigOConner2024_hepg2.h5ad",
    "Jurkat": DATA / "NadigOConner2024_jurkat.h5ad",
}
VOCAB = Path("/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human/vocab.json")
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
OUT = ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714"
MANIFESTS, TABLES, REPORTS = OUT / "manifests", OUT / "tables", OUT / "reports"
SEED = "E136_NADIG_20260714"
N_PERTURBATIONS = 96
MIN_CELLS = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(*parts: str) -> str:
    return hashlib.sha256("|".join([SEED, *map(str, parts)]).encode()).hexdigest()


def metadata():
    counts, genes, shapes = {}, {}, {}
    for context, path in SOURCES.items():
        data = ad.read_h5ad(path, backed="r")
        counts[context] = data.obs.perturbation.astype(str).value_counts().to_dict()
        genes[context] = set(data.var_names.astype(str))
        shapes[context] = [int(data.n_obs), int(data.n_vars)]
    return counts, genes, shapes


def main():
    for directory in [OUT, MANIFESTS, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    counts, genes, shapes = metadata()
    vocab = set(json.loads(VOCAB.read_text()))
    contexts = sorted(SOURCES)
    candidates = sorted(
        perturbation for perturbation in set(counts[contexts[0]]) & set(counts[contexts[1]])
        if perturbation not in {"", "control", "ctrl", "nan", "None", "non-targeting"}
        and all(int(counts[context].get(perturbation, 0)) >= MIN_CELLS for context in contexts)
        and all(perturbation in genes[context] for context in contexts)
        and perturbation in vocab
    )
    selected = sorted(candidates, key=lambda value: key("select", value))[:N_PERTURBATIONS]
    if len(selected) != N_PERTURBATIONS:
        raise RuntimeError(f"only {len(selected)} eligible perturbations")
    selection = pd.DataFrame({
        "perturbation": selected,
        "selection_hash": [key("select", value) for value in selected],
        "n_cells_HepG2": [counts["HepG2"][value] for value in selected],
        "n_cells_Jurkat": [counts["Jurkat"][value] for value in selected],
        "in_both_expression_gene_axes": True,
        "in_scgpt_vocabulary": True,
        "expression_values_used_for_selection": False,
    })
    selection.to_csv(TABLES / "E136_SELECTED_PERTURBATIONS.csv", index=False)
    rows = []
    for fold_number, heldout in enumerate(contexts, start=1):
        source = [value for value in contexts if value != heldout][0]
        ordered = sorted(selected, key=lambda value: key("fold", heldout, "perturbation", value))
        n_unseen = int(math.ceil(.20 * len(ordered)))
        unseen, seen = set(ordered[:n_unseen]), set(ordered[n_unseen:])
        source_seen = sorted(seen, key=lambda value: key("fold", heldout, "source_seen", value))
        n_aux = 12
        validation = set(source_seen[:n_aux])
        random_test = set(source_seen[n_aux:2 * n_aux])
        fold_id = f"Nadig_cellline_holdout_{fold_number}_{heldout}"
        for context in contexts:
            for perturbation in selected:
                if context == source and perturbation in seen:
                    split = "val" if perturbation in validation else "test" if perturbation in random_test else "train"
                    setting = "validation_pair" if split == "val" else "random_seen_pair" if split == "test" else "training_pair"
                elif context == source:
                    split, setting = "test", "perturbation_unseen"
                elif perturbation in seen:
                    split, setting = "test", "context_unseen"
                else:
                    split, setting = "test", "context_and_perturbation_unseen"
                rows.append({
                    "dataset": "Nadig_two_cellline",
                    "modality": "CRISPR_gene_knockout_cellline_shift",
                    "fold_id": fold_id,
                    "heldout_context": heldout,
                    "source_contexts": source,
                    "split": split,
                    "setting": setting,
                    "context": context,
                    "cell_line": context,
                    "perturbation": perturbation,
                    "n_cells": int(counts[context][perturbation]),
                    "perturbation_seen_in_training": perturbation in seen,
                    "context_seen_in_training": context == source,
                    "selected_without_expression_values": True,
                    "in_train_fraction_25": split == "train" and int(key(fold_id, context, perturbation, "25")[:8], 16) / 0xFFFFFFFF < .25,
                    "in_train_fraction_50": split == "train" and int(key(fold_id, context, perturbation, "50")[:8], 16) / 0xFFFFFFFF < .50,
                    "in_train_fraction_75": split == "train" and int(key(fold_id, context, perturbation, "75")[:8], 16) / 0xFFFFFFFF < .75,
                    "in_train_fraction_100": split == "train",
                })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFESTS / "E136_TASK_MANIFEST.csv", index=False)
    summary = manifest.groupby(["fold_id", "setting", "split"], as_index=False).agg(n_tasks=("perturbation", "size"), min_cells=("n_cells", "min"), median_cells=("n_cells", "median"))
    summary.to_csv(TABLES / "E136_SPLIT_SUMMARY.csv", index=False)
    model = json.loads(MODEL.read_text())
    status = {
        "experiment": "E136_nadig_two_cellline_contract",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_before_expression_values_predictions_or_errors",
        "sources": {context: {"path": str(path), "sha256": sha256(path), "shape": shapes[context]} for context, path in SOURCES.items()},
        "context_definition": "two biological cell lines: HepG2 and Jurkat",
        "n_contexts": len(contexts),
        "n_eligible_perturbations": len(candidates),
        "n_selected_perturbations": len(selected),
        "minimum_cells_per_selected_pair": int(selection[["n_cells_HepG2", "n_cells_Jurkat"]].min().min()),
        "n_folds": int(manifest.fold_id.nunique()),
        "n_manifest_rows": len(manifest),
        "n_test_tasks": int(manifest.split.eq("test").sum()),
        "expression_values_read_for_selection_or_split": False,
        "target_effect_prediction_or_error_used_for_selection": False,
        "frozen_direction_model_sha256": sha256(MODEL),
        "frozen_direction_model": model,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# E136｜Nadig 双细胞系第七数据确认合同",
        "",
        "HepG2 与 Jurkat 来自同一研究、不同生物细胞系。合同只读取 obs 标签、每个 cell line × perturbation 的细胞数、表达基因身份轴和 scGPT 词表；未读取表达矩阵数值。",
        "",
        f"- 候选共同扰动：{len(candidates)}；哈希固定抽取：{len(selected)}。",
        f"- 每个 pair 至少 {status['minimum_cells_per_selected_pair']} 个细胞。",
        f"- 两个外层 cell-line holdout folds；测试任务共 {status['n_test_tasks']}。",
        "- 每折包含 source 内随机 seen pair、整 cell-line 未见、整 perturbation 未见和二者同时未见。",
        "- E135 方向风险模型的文件哈希已经写入状态文件；后续不得根据 Nadig 结果改系数后仍称确认。",
    ]
    (REPORTS / "E136_CONTRACT_REPORT.md").write_text("\n".join(report) + "\n")
    (OUT / "README_先看这个.md").write_text("# E136 先看这个\n\n先读 `reports/E136_CONTRACT_REPORT.md`，合同在 `manifests/E136_TASK_MANIFEST.csv`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
