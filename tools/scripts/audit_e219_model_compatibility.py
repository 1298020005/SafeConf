#!/usr/bin/env python3
"""Audit E219 gene-axis and scGPT-vocabulary compatibility without truth access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E219 = ROOT / "docs/实验结果/E219_database_expansion_contract_20260920"
MANIFEST = E219 / "tables/E219_TASK_MANIFEST.csv"
AUDIT = E219 / "tables/E219_SOURCE_AUDIT.csv"
VOCAB = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human/vocab.json"
)


class CompatibilityFailure(RuntimeError):
    pass


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not MANIFEST.is_file() or not AUDIT.is_file() or not VOCAB.is_file():
        raise CompatibilityFailure("required E219 contract or scGPT vocabulary is missing")
    manifest = pd.read_csv(MANIFEST)
    sources = pd.read_csv(AUDIT)
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    rows = []
    for source in sources.itertuples(index=False):
        dataset = str(source.dataset)
        data = ad.read_h5ad(Path(source.source_path), backed="r")
        genes = set(map(str, data.var_names))
        data.file.close()
        perturbations = sorted(
            manifest.loc[manifest.dataset.eq(dataset), "perturbation"].astype(str).unique()
        )
        for perturbation in perturbations:
            token = perturbation.upper()
            rows.append({
                "dataset": dataset,
                "perturbation": perturbation,
                "expression_axis_present": perturbation in genes or token in genes,
                "scgpt_token": token,
                "scgpt_vocab_present": token in vocab,
                "basic_dual_adapter_compatible": bool(
                    (perturbation in genes or token in genes) and token in vocab
                ),
            })
    table = pd.DataFrame(rows)
    summary = table.groupby("dataset", as_index=False).agg(
        n_perturbations=("perturbation", "size"),
        expression_axis_compatible=("expression_axis_present", "sum"),
        scgpt_vocab_compatible=("scgpt_vocab_present", "sum"),
        dual_adapter_compatible=("basic_dual_adapter_compatible", "sum"),
    )
    summary["compatibility_fraction"] = (
        summary.dual_adapter_compatible / summary.n_perturbations
    )
    if len(table) != int(sources.n_perturbations.sum()):
        raise CompatibilityFailure("perturbation inventory changed")
    atomic_text(E219 / "tables/E219_MODEL_COMPATIBILITY.csv", table.to_csv(index=False))
    atomic_text(E219 / "tables/E219_MODEL_COMPATIBILITY_SUMMARY.csv", summary.to_csv(index=False))
    status = {
        "experiment": "E219_database_expansion_contract",
        "stage": "MODEL_COMPATIBILITY_PREFLIGHT",
        "status": "PASS",
        "vocab_path": str(VOCAB),
        "vocab_sha256": sha256(VOCAB),
        "expression_values_read": False,
        "target_effect_or_error_used": False,
        "datasets": summary.to_dict("records"),
        "next_gate": "run one-fold prediction smoke on datasets with adequate compatible targets",
    }
    atomic_text(
        E219 / "MODEL_COMPATIBILITY_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
