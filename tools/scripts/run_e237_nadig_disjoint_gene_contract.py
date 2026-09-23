#!/usr/bin/env python3
"""Freeze an expression-blind, gene-disjoint Nadig follow-up to E139.

The scenario is selected after seeing E139/E152. The 128 follow-up genes are
chosen only by metadata/QC and a fixed hash; this is same-study confirmation,
not an additional independent biological study.
"""

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
OUT = ROOT / "docs/实验结果/E237_nadig_disjoint_gene_confirmation_20260923"
PRIOR = ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714"
FROZEN_MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
COUNT = 128
MIN_CELLS = 50
SEED = "E237_GENE_DISJOINT_20260923"


def load_e136():
    path = ROOT / "tools/scripts/run_e136_nadig_two_cellline_contract.py"
    spec = importlib.util.spec_from_file_location("e136_metadata_for_e237", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ordering(*parts: str) -> str:
    return hashlib.sha256("|".join([SEED, *map(str, parts)]).encode()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite frozen contract: {OUT}")
    e136 = load_e136()
    counts, genes, shapes = e136.metadata()  # backed h5ad: obs labels and var names only
    vocab = set(json.loads(e136.VOCAB.read_text()))
    contexts = sorted(e136.SOURCES)
    original = pd.read_csv(PRIOR / "tables/E136_SELECTED_PERTURBATIONS.csv")
    original_genes = set(original.perturbation.astype(str))
    candidates = sorted(
        gene for gene in set(counts[contexts[0]]) & set(counts[contexts[1]])
        if gene not in {"", "control", "ctrl", "nan", "None", "non-targeting"}
        and all(int(counts[context].get(gene, 0)) >= MIN_CELLS for context in contexts)
        and all(gene in genes[context] for context in contexts)
        and gene in vocab
    )
    remaining = sorted(set(candidates) - original_genes, key=lambda gene: ordering("select", gene))
    if len(remaining) < COUNT:
        raise RuntimeError(f"only {len(remaining)} disjoint eligible genes; need {COUNT}")
    selected = remaining[:COUNT]
    rows = []
    for number, heldout in enumerate(contexts, start=1):
        source = next(context for context in contexts if context != heldout)
        ordered = sorted(selected, key=lambda gene: ordering("fold", heldout, gene))
        unseen = set(ordered[:math.ceil(.20 * len(ordered))])
        seen = set(ordered) - unseen
        source_seen = sorted(seen, key=lambda gene: ordering("source_seen", heldout, gene))
        validation = set(source_seen[:12])
        source_test = set(source_seen[12:24])
        fold = f"E237_cellline_holdout_{number}_{heldout}"
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
                rows.append({
                    "dataset": "Nadig_E237_disjoint",
                    "modality": "CRISPRi_gene_knockdown_cellline_shift",
                    "fold_id": fold,
                    "heldout_context": heldout,
                    "source_contexts": source,
                    "split": split,
                    "setting": setting,
                    "context": context,
                    "cell_line": context,
                    "perturbation": gene,
                    "n_cells": int(counts[context][gene]),
                    "perturbation_seen_in_training": gene in seen,
                    "context_seen_in_training": context == source,
                    "selected_without_expression_values": True,
                    "disjoint_from_E136": True,
                    "in_train_fraction_100": split == "train",
                })
    manifest = pd.DataFrame(rows)
    selection = pd.DataFrame({
        "perturbation": selected,
        "selection_hash": [ordering("select", gene) for gene in selected],
        "n_cells_HepG2": [int(counts["HepG2"][gene]) for gene in selected],
        "n_cells_Jurkat": [int(counts["Jurkat"][gene]) for gene in selected],
    })
    if set(selection.perturbation) & original_genes or manifest.split.eq("test").sum() < 300:
        raise RuntimeError("disjointness/test-count invariant failed")
    (OUT / "manifests").mkdir(parents=True)
    (OUT / "tables").mkdir()
    selection.to_csv(OUT / "tables/E237_SELECTED_PERTURBATIONS.csv", index=False)
    manifest.to_csv(OUT / "manifests/E237_TASK_MANIFEST.csv", index=False)
    status = {
        "status": "FROZEN_BEFORE_E237_EXPRESSION_OR_PREDICTION",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_is_post_E139_scenario_followup": True,
        "n_original_E136_genes_excluded": len(original_genes),
        "n_eligible_total": len(candidates),
        "n_eligible_remaining": len(remaining),
        "n_selected_disjoint_genes": len(selected),
        "n_test_rows": int(manifest.split.eq("test").sum()),
        "n_folds": int(manifest.fold_id.nunique()),
        "metadata_only_selection": True,
        "target_expression_values_read": 0,
        "source_files": {context: {"path": str(path), "shape": shapes[context], "sha256": e136.sha256(path)}
                         for context, path in e136.SOURCES.items()},
        "prior_selection_sha256": e136.sha256(PRIOR / "tables/E136_SELECTED_PERTURBATIONS.csv"),
        "frozen_direction_model_sha256": e136.sha256(FROZEN_MODEL),
        "manifest_sha256": e136.sha256(OUT / "manifests/E237_TASK_MANIFEST.csv"),
    }
    (OUT / "CONTRACT_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
