#!/usr/bin/env python3
"""Audit the locked E192 predictor for SafeConf public-risk compatibility.

This is qualification only.  It never reads the E192 target truth array and
does not score a method when the prediction/public-effect contract differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

E192 = Path("/home/yyf/data/safeconf_e192_adamson_rpe1")
E192_DOC = Path("/home/yyf/proj/docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729")
PUBLIC_META = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_source_core_20261002_v1/SOURCE_PUBLIC_MEMORY_METADATA.parquet")
GWPS_META = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/gwps_v1/GWPS_PUBLIC_METADATA.parquet")
DEFAULT_OUT = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/e192_contract_audit")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(E192 / "model_assets/GENE_PANEL.csv")
    queries = pd.read_csv(E192 / "model_assets/QUERY_TASKS.csv")
    metrics = pd.read_csv(E192_DOC / "final_evaluation/tables/E192_TASK_METRICS.csv")
    predictions = np.load(E192_DOC / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz")
    scale = pd.read_csv(E192_DOC / "PRETRUTH_INPUT_SCALE_AUDIT.csv")
    build = json.loads((E192 / "model_assets/ASSET_MANIFEST.json").read_text())
    pretruth = json.loads((E192_DOC / "pretruth_release/PRETRUTH_STATUS.json").read_text())
    public = pd.read_parquet(PUBLIC_META)
    gwps = pd.read_parquet(GWPS_META)

    panel_genes = panel.gene_name.astype(str).tolist()
    task_ids = queries.task_id.astype(str).tolist()
    checks = {
        "panel_rows": len(panel), "panel_unique_genes": panel.gene_name.nunique(),
        "query_rows": len(queries), "query_genes": queries.gene.nunique(),
        "task_metrics_rows": len(metrics), "task_id_alignment": set(task_ids) == set(metrics.task_id.astype(str)),
        "prediction_members": sorted(predictions.files),
        "prediction_shapes": {k: list(predictions[k].shape) for k in predictions.files},
        "all_predictions_finite": all(np.isfinite(predictions[k]).all() for k in predictions.files),
        "panel_gene_duplicates": int(panel.gene_name.duplicated().sum()),
        "target_truth_read_in_locked_pretruth": bool(pretruth.get("target_perturbation_x_rows_read", 0)),
        "target_truth_array_audited_only": True,
        "e192_normalization": "per-cell full-library scale to 10000, then log1p",
        "e192_manifest": build,
    }
    if not checks["task_id_alignment"] or len(queries) != 175 or len(panel) != 512:
        raise RuntimeError("E192 locked asset identity changed")

    query_genes = set(queries.gene.astype(str))
    rows = []
    for label, frame, context_rule, contract in [
        ("E201_RPE1_same_context", public[public.context.astype(str).eq("RPE1")], "RPE1", "E201_log1p_matched_batch_delta_v1"),
        ("E201_all_contexts", public, "all", "E201_log1p_matched_batch_delta_v1"),
        ("GWPS_K562_cross_context", gwps[gwps.context.astype(str).eq("K562")], "K562", "GWPS_log1p_4000_effect_v1"),
        ("GWPS_RPE1_same_context", gwps[gwps.context.astype(str).eq("RPE1")], "RPE1", "GWPS_log1p_4000_effect_v1"),
    ]:
        covered = query_genes.intersection(set(frame.perturbation_target.astype(str)))
        for gene in sorted(query_genes):
            rows.append({"asset": label, "context_rule": context_rule, "effect_contract": contract,
                         "gene": gene, "has_history": gene in covered,
                         "n_history_records": int((frame.perturbation_target.astype(str) == gene).sum())})
    coverage = pd.DataFrame(rows)
    coverage.to_csv(out / "E192_PUBLIC_HISTORY_COVERAGE.csv", index=False, lineterminator="\n")

    scale_contracts = sorted(scale.to_dict("records"), key=lambda x: x.get("matrix", ""))
    contract_matches = {
        "E192_prediction_contract": "per-cell full-library scale to 10000 then log1p",
        "E201_public_contract": "E201_log1p_matched_batch_delta_v1 (registered 4000-scale public pipeline)",
        "GWPS_public_contract": "GWPS log1p(4000 * raw retained gene UMI / original UMI_count)",
        "exact_contract_match": False,
        "reason": "E192 prediction/effect scale is 10000 while available SafeConf public effects are registered at 4000; no scale-preserving adapter is frozen for this asset",
    }
    audit = {
        "status": "QUALIFICATION_COMPLETE_CONTRACT_MISMATCH_NO_SCORING",
        "asset_role": "locked predictor and truth-index candidate; not SafeConf score",
        "e192_doc": str(E192_DOC), "panel_sha256": sha(E192 / "model_assets/GENE_PANEL.csv"),
        "pretruth_predictions_sha256": sha(E192_DOC / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz"),
        "query_rows": len(queries), "query_genes": len(query_genes),
        "prediction_members": sorted(predictions.files),
        "pretruth_target_truth_reads": pretruth.get("target_perturbation_x_rows_read", 0),
        "public_history_coverage": {name: int(group.has_history.sum()) for name, group in coverage.groupby("asset")},
        "scale_audit": scale_contracts,
        "contract": contract_matches,
        "target_truth_opened_by_this_script": False,
        "permanent_test_truth_opened": False,
        "next_action": "Either obtain a frozen 10000-scale public-effect source or implement and validate a pre-registered scale adapter on development data; do not score E192 under the current PublicRule contract before that.",
    }
    write_json(out / "E192_INPUT_CONTRACT_AUDIT.json", {"checks": checks, "contract": contract_matches})
    write_json(out / "E192_ASSET_STATUS.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
