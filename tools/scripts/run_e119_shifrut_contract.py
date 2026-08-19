#!/usr/bin/env python3
"""E119: freeze a four-context Shifrut/Marson T-cell CRISPR rectangle."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ShifrutMarson2018.h5ad")
OUT = ROOT / "docs/实验结果/E119_shifrut_four_context_contract_20260714"
MIN_CELLS = 50
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
INVALID = {"", "control", "ctrl", "nan", "none", "noise", "nt1"}


def key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main():
    for d in (OUT / "manifests", OUT / "tables", OUT / "reports"):
        d.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(SOURCE, backed="r")
    obs = a.obs[["sample", "patient", "perturbation_2", "perturbation"]].astype(str).copy()
    shape = tuple(map(int, a.shape))
    a.file.close()
    counts = obs.groupby(["sample", "perturbation"]).size().rename("n_cells").reset_index()
    matrix = counts.pivot(index="sample", columns="perturbation", values="n_cells").fillna(0).astype(int)
    contexts = sorted(matrix.index.astype(str))
    shared = matrix.columns[matrix.ge(MIN_CELLS).all(axis=0)].astype(str).tolist()
    if "control" not in shared:
        raise RuntimeError("shared control is required")
    perturbations = sorted(p for p in shared if p.strip().lower() not in INVALID)
    if len(contexts) != 4 or len(perturbations) < 20:
        raise RuntimeError(f"unexpected rectangle: {len(contexts)} x {len(perturbations)}")
    rows = []
    count_map = {(r.sample, r.perturbation): int(r.n_cells) for r in counts.itertuples(index=False)}
    for fi, heldout in enumerate(contexts, start=1):
        fold = f"Shifrut_context_holdout_{fi}_{heldout}"
        source_contexts = [c for c in contexts if c != heldout]
        n_new = max(5, int(math.ceil(0.20 * len(perturbations))))
        ordered_perts = sorted(perturbations, key=lambda p: key("E119", fold, "perturbation", p))
        new_perts, seen_perts = set(ordered_perts[:n_new]), set(ordered_perts[n_new:])
        source_seen = [(c, p) for c in source_contexts for p in sorted(seen_perts)]
        ordered_pairs = sorted(source_seen, key=lambda x: key("E119", fold, "pair", *x))
        n_aux = min(10, max(8, int(round(0.10 * len(ordered_pairs)))))
        validation, random_missing = set(ordered_pairs[:n_aux]), set(ordered_pairs[n_aux:2 * n_aux])
        base_train = [x for x in ordered_pairs if x not in validation and x not in random_missing]
        membership = {}
        for c in source_contexts:
            cpairs = sorted([x for x in base_train if x[0] == c], key=lambda x: key("E119", fold, "fraction", *x))
            for fraction in FRACTIONS:
                selected = set(cpairs[:max(1, int(round(len(cpairs) * fraction)))])
                for pair in cpairs:
                    membership.setdefault(pair, {})[fraction] = pair in selected
        for c in contexts:
            for p in perturbations:
                pair = (c, p)
                if c == heldout and p in new_perts:
                    split, setting = "test", "context_and_perturbation_unseen"
                elif c == heldout:
                    split, setting = "test", "context_unseen_row"
                elif p in new_perts:
                    split, setting = "test", "perturbation_unseen_column"
                elif pair in validation:
                    split, setting = "val", "source_validation_pair"
                elif pair in random_missing:
                    split, setting = "test", "random_missing_pair"
                else:
                    split, setting = "train", "source_train_pair"
                patient, stimulation = c.split("_", 1)
                row = {"dataset": "Shifrut", "modality": "gene_knockout_plus_TCR_context", "fold_id": fold, "heldout_context": heldout, "source_contexts": "+".join(source_contexts), "split": split, "setting": setting, "context": c, "patient": patient, "tcr_state": stimulation, "perturbation": p, "n_cells": count_map[pair], "perturbation_seen_in_training": p in seen_perts, "context_seen_in_training": c in source_contexts, "selected_without_expression": True}
                for fraction in FRACTIONS:
                    row[f"in_train_fraction_{int(fraction * 100)}"] = bool(split == "train" and membership.get(pair, {}).get(fraction, False))
                rows.append(row)
    manifest = pd.DataFrame(rows).sort_values(["fold_id", "split", "setting", "context", "perturbation"])
    for fold, g in manifest.groupby("fold_id"):
        if len(g) != len(contexts) * len(perturbations) or g.duplicated(["context", "perturbation"]).any():
            raise RuntimeError(f"{fold}: partition invariant failed")
        previous = set()
        for fraction in FRACTIONS:
            selected = set(map(tuple, g.loc[g[f"in_train_fraction_{int(fraction * 100)}"], ["context", "perturbation"]].to_numpy()))
            if not previous.issubset(selected):
                raise RuntimeError(f"{fold}: non-nested fractions")
            previous = selected
    manifest.to_csv(OUT / "manifests/E119_TASK_MANIFEST.csv", index=False)
    summary = manifest.groupby(["fold_id", "setting", "split"], as_index=False).agg(n_tasks=("perturbation", "size"), min_cells=("n_cells", "min"), median_cells=("n_cells", "median"))
    summary.to_csv(OUT / "tables/E119_SETTING_SUMMARY.csv", index=False)
    context = obs[["sample", "patient", "perturbation_2"]].drop_duplicates().sort_values("sample").rename(columns={"sample": "context", "perturbation_2": "tcr_state"})
    context.to_csv(OUT / "tables/E119_CONTEXTS.csv", index=False)
    pd.DataFrame({"perturbation": perturbations}).to_csv(OUT / "tables/E119_PERTURBATIONS.csv", index=False)
    status = {"experiment": "E119_shifrut_four_context_contract", "generated_at": datetime.now().isoformat(timespec="seconds"), "source": str(SOURCE), "source_sha256": sha256(SOURCE), "source_shape": shape, "context_definition": "sample = donor x TCR stimulation", "n_contexts": len(contexts), "n_perturbations": len(perturbations), "n_folds": manifest.fold_id.nunique(), "n_manifest_rows": len(manifest), "minimum_cells_per_pair": int(manifest.n_cells.min()), "expression_read_during_selection": False, "target_effect_or_error_used_for_selection": False}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    lines = ["# E119｜Shifrut–Marson 四背景遗传扰动冻结合同", "", "四个背景由两位健康供体与 TCR 刺激/未刺激组合得到。合同只读取标签和每格细胞数，没有读取表达矩阵、扰动效应、预测或误差。", "", f"- 矩阵：{len(contexts)} contexts × {len(perturbations)} CRISPR targets", f"- 每个 pair 最少细胞：{manifest.n_cells.min()}", f"- 外层 folds：{manifest.fold_id.nunique()}", "", "每折独立包含随机缺失 pair、整行新背景、整列新扰动和行列双未见。测试真值将在 E120 预测与风险冻结后才解封。"]
    (OUT / "reports/E119_CONTRACT_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text("# E119 先看这个\n\n先读 `reports/E119_CONTRACT_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
