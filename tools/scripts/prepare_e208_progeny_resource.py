#!/usr/bin/env python3
"""Freeze the official human PROGENy top-500 pathway weights for E208."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyreadr


EXPECTED_COMMIT = "cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f"
EXPECTED_RDA_SHA256 = "3094f1fa1bb5395c1074280402c0e30ef47ba9ec36f4829ca3e539d019255ee5"


class ResourceFailure(RuntimeError):
    """The pinned PROGENy source or extracted resource changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progeny-repo", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    repo = args.progeny_repo.expanduser().absolute()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise ResourceFailure(f"PROGENy source commit changed: {commit}")
    if subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet"], check=False
    ).returncode:
        raise ResourceFailure("PROGENy source has tracked modifications")
    source = repo / "data/model_human_full.rda"
    if sha256_file(source) != EXPECTED_RDA_SHA256:
        raise ResourceFailure("PROGENy human model RDA checksum changed")
    objects = pyreadr.read_r(str(source))
    if set(objects) != {"model_human_full"}:
        raise ResourceFailure(f"unexpected RDA objects: {sorted(objects)}")
    model = objects["model_human_full"].copy()
    required = {"gene", "pathway", "weight", "p.value"}
    if not required.issubset(model.columns):
        raise ResourceFailure("PROGENy model columns changed")
    model["gene"] = model.gene.astype(str)
    model["pathway"] = model.pathway.astype(str)
    selected = (
        model.sort_values(["pathway", "p.value", "gene"], kind="mergesort")
        .groupby("pathway", sort=True, group_keys=False)
        .head(500)
        .copy()
    )
    selected["rank_within_pathway"] = (
        selected.groupby("pathway", sort=True).cumcount() + 1
    )
    selected = selected.rename(columns={"p.value": "p_value"})[
        ["pathway", "gene", "weight", "p_value", "rank_within_pathway"]
    ].reset_index(drop=True)
    counts = selected.groupby("pathway").size()
    if len(counts) != 14 or not counts.eq(500).all() or len(selected) != 7000:
        raise ResourceFailure(f"unexpected PROGENy top-500 shape: {counts.to_dict()}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_csv.with_name(f".{args.output_csv.name}.tmp")
    selected.to_csv(temporary, index=False)
    os.replace(temporary, args.output_csv)
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_PRETRUTH_PROGENY_RESOURCE",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_repository": "https://github.com/saezlab/progeny",
        "source_commit": EXPECTED_COMMIT,
        "source_rda_sha256": EXPECTED_RDA_SHA256,
        "pyreadr_version": pyreadr.__version__,
        "selection": "top 500 per pathway by ascending p.value, gene-symbol tie break",
        "n_pathways": len(counts),
        "n_rows": len(selected),
        "pathway_counts": {str(key): int(value) for key, value in counts.items()},
        "output_csv": str(args.output_csv),
        "output_sha256": sha256_file(args.output_csv),
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(args.status, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
