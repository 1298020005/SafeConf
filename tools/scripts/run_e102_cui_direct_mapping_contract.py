#!/usr/bin/env python3
"""E102: freeze a Cui cytokine subset with direct scGPT-token mappings."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E99 = ROOT / "docs/实验结果/E99_multicontext_external_contract_20260713"
OUT = ROOT / "docs/实验结果/E102_cui_direct_mapping_contract_20260713"
CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def rank_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def direct_token(label: str, vocab: dict[str, int]) -> tuple[str | None, str | None]:
    upper = label.upper()
    candidates = [
        (upper, "uppercase_exact"),
        (upper.replace("-", ""), "uppercase_remove_hyphen"),
        (re.sub(r"[^A-Z0-9]", "", upper), "uppercase_alphanumeric"),
    ]
    seen = set()
    for token, rule in candidates:
        if token in seen:
            continue
        seen.add(token)
        if token in vocab:
            return token, rule
    return None, None


def main() -> None:
    for name in ["manifests", "tables", "reports"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    rectangle = pd.read_csv(E99 / "tables/E99_RECTANGLE_CELL_COUNTS.csv", keep_default_na=False)
    rectangle = rectangle[rectangle["dataset"].eq("Cui")].copy()
    source = pd.read_csv(E99 / "tables/E99_SOURCE_AUDIT.csv", keep_default_na=False)
    source = source[source["dataset"].eq("Cui")].iloc[0]
    vocab = json.loads((CHECKPOINT / "vocab.json").read_text(encoding="utf-8"))
    mapping_rows = []
    for label in sorted(rectangle["perturbation"].astype(str).unique()):
        token, rule = direct_token(label, vocab)
        mapping_rows.append(
            {"cytokine_label": label, "scgpt_token": token or "", "mapping_rule": rule or "unmapped",
             "scgpt_vocab_id": int(vocab[token]) if token else "", "eligible": token is not None,
             "selection_uses_effect_or_error": False}
        )
    mapping = pd.DataFrame(mapping_rows)
    eligible_perts = sorted(mapping.loc[mapping["eligible"], "cytokine_label"].astype(str))
    excluded = mapping.loc[~mapping["eligible"]].copy()
    contexts = sorted(rectangle["context"].astype(str).unique())
    if len(contexts) != 6 or len(eligible_perts) < 40:
        raise RuntimeError(f"unexpected direct subset: contexts={len(contexts)}, perturbations={len(eligible_perts)}")
    counts = rectangle.set_index(["context", "perturbation"])["n_cells"].to_dict()
    rows, fraction_rows = [], []
    for fold_index, heldout in enumerate(contexts, start=1):
        fold_id = f"Cui_direct_context_holdout_{fold_index}_{heldout.replace(' ', '_')}"
        source_contexts = [value for value in contexts if value != heldout]
        ordered = sorted(eligible_perts, key=lambda value: rank_key("E102", fold_id, "perturbation", value))
        n_new = int(math.ceil(0.20 * len(ordered)))
        new_perts, seen_perts = set(ordered[:n_new]), set(ordered[n_new:])
        source_seen = [(context, pert) for context in source_contexts for pert in sorted(seen_perts)]
        ordered_pairs = sorted(source_seen, key=lambda pair: rank_key("E102", fold_id, "pair", *pair))
        n_aux = min(30, max(8, int(round(0.10 * len(ordered_pairs)))))
        validation = set(ordered_pairs[:n_aux])
        random_missing = set(ordered_pairs[n_aux : 2 * n_aux])
        train = [pair for pair in ordered_pairs if pair not in validation and pair not in random_missing]
        membership: dict[tuple[str, str], dict[float, bool]] = {}
        for context in source_contexts:
            pairs = sorted([pair for pair in train if pair[0] == context], key=lambda pair: rank_key("E102", fold_id, "fraction", *pair))
            for fraction in FRACTIONS:
                selected = set(pairs[: max(1, int(round(len(pairs) * fraction)))])
                for pair in pairs:
                    membership.setdefault(pair, {})[fraction] = pair in selected
        for context in contexts:
            for perturbation in eligible_perts:
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
                    "dataset": "Cui_direct41", "modality": "cytokine_stimulus", "fold_id": fold_id,
                    "heldout_context": heldout, "source_contexts": "+".join(source_contexts), "split": split,
                    "setting": setting, "context": context, "perturbation": perturbation,
                    "scgpt_token": mapping.set_index("cytokine_label").loc[perturbation, "scgpt_token"],
                    "n_cells": int(counts[pair]), "perturbation_seen_in_training": perturbation in seen_perts,
                    "context_seen_in_training": context in source_contexts, "selected_without_expression": True,
                }
                for fraction in FRACTIONS:
                    record[f"in_train_fraction_{int(fraction*100)}"] = bool(
                        split == "train" and membership.get(pair, {}).get(fraction, False)
                    )
                rows.append(record)
        for fraction in FRACTIONS:
            selected = [pair for pair in train if membership[pair][fraction]]
            fraction_rows.append(
                {"fold_id": fold_id, "train_fraction": fraction, "n_train_pairs": len(selected),
                 "n_train_contexts": len({pair[0] for pair in selected}),
                 "n_train_perturbations": len({pair[1] for pair in selected})}
            )
    manifest = pd.DataFrame(rows).sort_values(["fold_id", "split", "setting", "context", "perturbation"])
    for fold_id, group in manifest.groupby("fold_id"):
        if len(group) != len(contexts) * len(eligible_perts) or group.duplicated(["context", "perturbation"]).any():
            raise RuntimeError(f"{fold_id}: rectangle invariant failed")
    summary = manifest.groupby(["fold_id", "setting", "split"], as_index=False).agg(
        n_tasks=("perturbation", "size"), min_cells=("n_cells", "min"), median_cells=("n_cells", "median")
    )
    mapping.to_csv(OUT / "tables/E102_CYTOKINE_MAPPING_AUDIT.csv", index=False)
    excluded.to_csv(OUT / "tables/E102_EXCLUDED_UNMAPPED.csv", index=False)
    manifest.to_csv(OUT / "manifests/E102_TASK_MANIFEST.csv", index=False)
    summary.to_csv(OUT / "tables/E102_SETTING_SUMMARY.csv", index=False)
    pd.DataFrame(fraction_rows).to_csv(OUT / "tables/E102_TRAIN_FRACTION_SUMMARY.csv", index=False)
    status = {
        "experiment": "E102_cui_direct_mapping_contract", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_h5ad": source["source_h5ad"], "source_sha256": source["source_sha256"],
        "n_contexts": len(contexts), "n_original_perturbations": int(rectangle["perturbation"].nunique()),
        "n_direct_mapped_perturbations": len(eligible_perts), "n_excluded_unmapped": len(excluded),
        "n_folds": int(manifest["fold_id"].nunique()), "n_manifest_rows": len(manifest),
        "mapping_uses_manual_aliases": False, "expression_effect_prediction_or_error_used_for_selection": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# E102｜Cui 细胞因子直接映射子集合同

Cui 原矩形有 6 个免疫细胞背景、86 个刺激。E102 只保留经 `uppercase`、去连字符或去非字母数字字符后直接命中 scGPT 词表的 {len(eligible_perts)} 个标签；没有使用手工别名、表达效应、预测或误差。其余 {len(excluded)} 个商品名、复合亚基或别名全部进入排除表，不猜测映射。

新合同按 41 个可执行刺激重新哈希：每折整行新背景 32 tasks、整列新刺激 45 tasks、双未见 9 tasks、随机缺失 16 tasks，另有 16 validation pairs 和 128 train pairs；训练集有 25%/50%/75%/100% 嵌套子矩阵。

- `tables/E102_CYTOKINE_MAPPING_AUDIT.csv`
- `tables/E102_EXCLUDED_UNMAPPED.csv`
- `manifests/E102_TASK_MANIFEST.csv`
"""
    (OUT / "reports/E102_CONTRACT_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "README_先看这个.md").write_text("# E102 先看这个\n\n先读 `reports/E102_CONTRACT_REPORT.md`。\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
