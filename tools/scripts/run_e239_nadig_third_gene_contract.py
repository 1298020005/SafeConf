#!/usr/bin/env python3
"""Freeze a third, gene-disjoint Nadig cohort and stronger context-aware baseline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E239_nadig_third_gene_confirmation_20260923"
E136 = ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714"
E237 = ROOT / "docs/实验结果/E237_nadig_disjoint_gene_confirmation_20260923"
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
COUNT = 128
SEED = "E239_THIRD_GENE_COHORT_20260923"


def module():
    path = ROOT / "tools/scripts/run_e136_nadig_two_cellline_contract.py"
    spec = importlib.util.spec_from_file_location("e136_metadata_for_e239", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def key(*parts: str) -> str:
    return hashlib.sha256("|".join([SEED, *map(str, parts)]).encode()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    e136 = module()
    counts, genes, shapes = e136.metadata()  # obs/var only, no expression values
    vocab = set(json.loads(e136.VOCAB.read_text()))
    contexts = sorted(e136.SOURCES)
    prior_paths = [E136 / "tables/E136_SELECTED_PERTURBATIONS.csv",
                   E237 / "tables/E237_SELECTED_PERTURBATIONS.csv"]
    used = set().union(*(set(pd.read_csv(path).perturbation.astype(str)) for path in prior_paths))
    if len(used) != 224:
        raise RuntimeError("E136/E237 selected genes overlap or changed")
    eligible = sorted(
        gene for gene in set(counts[contexts[0]]) & set(counts[contexts[1]])
        if gene not in {"", "control", "ctrl", "nan", "None", "non-targeting"}
        and all(int(counts[context].get(gene, 0)) >= 50 for context in contexts)
        and all(gene in genes[context] for context in contexts)
        and gene in vocab
    )
    remaining = sorted(set(eligible) - used, key=lambda gene: key("select", gene))
    if len(remaining) < COUNT:
        raise RuntimeError(f"only {len(remaining)} unused eligible genes")
    selected = remaining[:COUNT]
    rows = []
    for number, heldout in enumerate(contexts, start=1):
        source = next(context for context in contexts if context != heldout)
        ordered = sorted(selected, key=lambda gene: key("fold", heldout, gene))
        unseen = set(ordered[:math.ceil(.20 * len(ordered))])
        seen = set(ordered) - unseen
        source_seen = sorted(seen, key=lambda gene: key("source_seen", heldout, gene))
        validation, source_test = set(source_seen[:12]), set(source_seen[12:24])
        fold = f"E239_cellline_holdout_{number}_{heldout}"
        for context in contexts:
            for gene in selected:
                if context == source and gene in seen:
                    split = "val" if gene in validation else "test" if gene in source_test else "train"
                    setting = {"val": "validation_pair", "test": "random_seen_pair", "train": "training_pair"}[split]
                elif context == source:
                    split, setting = "test", "perturbation_unseen"
                elif gene in seen:
                    split, setting = "test", "context_unseen"
                else:
                    split, setting = "test", "context_and_perturbation_unseen"
                rows.append({"dataset": "Nadig_E239_disjoint", "modality": "CRISPRi_gene_knockdown_cellline_shift",
                             "fold_id": fold, "heldout_context": heldout, "source_contexts": source,
                             "split": split, "setting": setting, "context": context, "cell_line": context,
                             "perturbation": gene, "n_cells": int(counts[context][gene]),
                             "perturbation_seen_in_training": gene in seen,
                             "context_seen_in_training": context == source,
                             "selected_without_expression_values": True,
                             "disjoint_from_E136_and_E237": True,
                             "in_train_fraction_100": split == "train"})
    manifest = pd.DataFrame(rows)
    selection = pd.DataFrame({"perturbation": selected,
                              "selection_hash": [key("select", gene) for gene in selected],
                              "n_cells_HepG2": [int(counts["HepG2"][gene]) for gene in selected],
                              "n_cells_Jurkat": [int(counts["Jurkat"][gene]) for gene in selected]})
    if set(selected) & used or manifest.split.eq("test").sum() != 332:
        raise RuntimeError("E239 gene disjointness or test count failed")
    (OUT / "manifests").mkdir(parents=True)
    (OUT / "tables").mkdir()
    selection.to_csv(OUT / "tables/E239_SELECTED_PERTURBATIONS.csv", index=False)
    manifest.to_csv(OUT / "manifests/E239_TASK_MANIFEST.csv", index=False)
    record = {"status": "FROZEN_BEFORE_E239_EXPRESSION_OR_PREDICTION",
              "created_at": datetime.now().isoformat(timespec="seconds"),
              "n_eligible_total": len(eligible), "n_previously_used_genes_excluded": len(used),
              "n_eligible_remaining": len(remaining), "n_selected_disjoint_genes": len(selected),
              "n_test_rows": int(manifest.split.eq("test").sum()), "n_folds": 2,
              "selection_is_post_E139_E237_scenario_followup": True,
              "target_expression_values_read": 0,
              "source_files": {context: {"path": str(path), "shape": shapes[context], "sha256": e136.sha256(path)}
                               for context, path in e136.SOURCES.items()},
              "prior_selection_hashes": {str(path.relative_to(ROOT)): e136.sha256(path) for path in prior_paths},
              "frozen_direction_model_sha256": e136.sha256(MODEL),
              "manifest_sha256": e136.sha256(OUT / "manifests/E239_TASK_MANIFEST.csv")}
    (OUT / "CONTRACT_STATUS.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
