#!/usr/bin/env python3
"""Run the frozen confidence MVP pipeline for one blind dataset.

This is a thin adapter over the Phase 2.1 runner.  It only changes dataset
selection and column preference; the scoring formula is frozen downstream in
`safetrans_confidence`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "confidence_task"
os.chdir(CONF_DIR)
sys.path.insert(0, str(CONF_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "03_code"))

import build_context_splits as bcs  # noqa: E402
import run_confidence_mvp_v2_1 as mvp  # noqa: E402


def _prepend_unique(items: list[str], value: str | None) -> list[str]:
    out = []
    if value:
        out.append(value)
    for item in items:
        if item not in out:
            out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen-protocol blind dataset.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--h5ad-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-col", required=True)
    parser.add_argument("--perturbation-col", required=True)
    parser.add_argument("--dataset-family", required=True, choices=["gene_main", "chem_robust", "external"])
    parser.add_argument("--filter-col", default=None, help="Optional obs column used to subset the h5ad before task building.")
    parser.add_argument("--filter-value", default=None, help="Required value for --filter-col.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--atlas-root", default=str(mvp.DEFAULT_ATLAS_ROOT))
    parser.add_argument("--n-genes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=5201)
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad_path)
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    resolved_h5ad_path = h5ad_path
    filter_status = {
        "filter_col": args.filter_col,
        "filter_value": args.filter_value,
        "input_h5ad_path": str(h5ad_path),
        "filtered_h5ad_path": "",
        "n_cells_before": None,
        "n_cells_after": None,
        "status": "not_requested",
    }
    if args.filter_col or args.filter_value:
        if not args.filter_col or args.filter_value is None:
            raise ValueError("--filter-col and --filter-value must be provided together")
        adata = sc.read_h5ad(h5ad_path)
        if args.filter_col not in adata.obs.columns:
            raise KeyError(f"Missing filter column {args.filter_col!r} in {h5ad_path}")
        mask = adata.obs[args.filter_col].astype(str).eq(str(args.filter_value))
        if int(mask.sum()) == 0:
            raise ValueError(f"Filter produced zero cells: {args.filter_col} == {args.filter_value}")
        filtered = adata[mask].copy()
        filtered_path = out / "input" / f"{args.dataset_name}__filtered_{args.filter_col}_{args.filter_value}.h5ad"
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.write_h5ad(filtered_path)
        resolved_h5ad_path = filtered_path
        filter_status.update(
            {
                "filtered_h5ad_path": str(filtered_path),
                "n_cells_before": int(adata.n_obs),
                "n_cells_after": int(filtered.n_obs),
                "status": "ok",
            }
        )
        del adata
        del filtered

    mvp.DATASET_NAMES = [args.dataset_name]
    bcs.CONTEXT_CANDIDATES = _prepend_unique(list(bcs.CONTEXT_CANDIDATES), args.context_col)
    bcs.PERT_CANDIDATES = _prepend_unique(list(bcs.PERT_CANDIDATES), args.perturbation_col)

    def resolve_dataset_paths(_atlas_root: Path) -> dict[str, Path]:
        return {args.dataset_name: resolved_h5ad_path}

    mvp.resolve_dataset_paths = resolve_dataset_paths

    (out / "BLIND_DATASET_SPEC.json").write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "dataset_family": args.dataset_family,
                "h5ad_path": str(resolved_h5ad_path),
                "original_h5ad_path": str(h5ad_path),
                "context_col": args.context_col,
                "perturbation_col": args.perturbation_col,
                "filter": filter_status,
                "n_genes": args.n_genes,
                "seed": args.seed,
                "protocol": "v0.2 frozen; no formula tuning in this runner",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    sys.argv = [
        "run_blind_dataset",
        "--project-root",
        args.project_root,
        "--atlas-root",
        args.atlas_root,
        "--out-dir",
        str(out),
        "--n-genes",
        str(args.n_genes),
        "--seed",
        str(args.seed),
    ]
    return int(mvp.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
