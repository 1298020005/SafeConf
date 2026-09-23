#!/usr/bin/env python3
"""Run the predeclared E237 scGPT/GEARS pipeline and freeze deployment scores."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(manifest: Path, out: Path, asset: Path, dataset: str, seed: int):
    path = ROOT / "tools/scripts/run_e112_external_formal_dual_models.py"
    spec = importlib.util.spec_from_file_location("e112_for_e237", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CONTRACT = manifest
    module.OUT = out
    module.SPECS = {dataset: {"source": asset, "context": "context"}}
    # The upstream helper uppercases symbols, but some legitimate scGPT tokens
    # retain their mixed case. Resolve these from metadata/vocabulary alone.
    round_id = dataset.split("_")[1]
    selected = pd.read_csv(out / f"tables/{round_id}_SELECTED_PERTURBATIONS.csv").perturbation.astype(str)
    vocab = set(json.loads(module.SCGPT_VOCAB.read_text()))
    module.ALIASES.update({gene: gene for gene in selected if gene in vocab and gene.upper() not in vocab})
    missing = [gene for gene in selected if module.human_token(gene) not in vocab]
    if missing:
        raise RuntimeError(f"selected perturbation tokens absent after case resolution: {missing}")
    module.SEED = seed
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--round", choices=["E237", "E239"], default="E237")
    args = parser.parse_args()
    round_id = args.round
    out = ROOT / f"docs/实验结果/{round_id}_{'nadig_disjoint_gene_confirmation' if round_id == 'E237' else 'nadig_third_gene_confirmation'}_20260923"
    manifest = out / f"manifests/{round_id}_TASK_MANIFEST.csv"
    asset_path = Path(f"/home/yyf/data/safeconf_{round_id.lower()}_nadig/Nadig_{round_id}_disjoint.h5ad")
    dataset = f"Nadig_{round_id}_disjoint"
    scores_path = out / f"{round_id}_SCORES_FIXED_BEFORE_DIRECTIONAL_EVALUATION.csv"
    contract = json.loads((out / "CONTRACT_STATUS.json").read_text())
    asset = json.loads((out / f"{round_id}_ASSET_STATUS.json").read_text())
    if (sha256(manifest) != contract["manifest_sha256"]
            or sha256(MODEL) != contract["frozen_direction_model_sha256"]
            or sha256(asset_path) != asset["asset_sha256"]):
        raise RuntimeError(f"{round_id} contract/model/asset hash mismatch")
    if scores_path.exists() or (out / dataset / "RUN_STATUS.json").exists():
        raise RuntimeError(f"refusing to overwrite existing {round_id} run")
    source = load_source(manifest, out, asset_path, dataset, 202609237 if round_id == "E237" else 202609239)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = source.run_dataset(dataset, device)
    if result["status"] != "complete" or result["strict_issue_count"] != 0 or result["n_test_tasks"] != contract["n_test_rows"]:
        raise RuntimeError(f"{round_id} upstream prediction incomplete: {result}")
    model = json.loads(MODEL.read_text())
    deploy_cols = ["fold_id", "task_id", "setting", "context", "perturbation",
                   "baseline_predicted_magnitude", "risk_model_disagreement", *model["features_in_order"]]
    risk_path = out / dataset / "TASK_RISK_TABLE.csv"
    scores = pd.read_csv(risk_path, usecols=deploy_cols)
    if len(scores) != contract["n_test_rows"] or scores.duplicated(["fold_id", "task_id"]).any():
        raise RuntimeError(f"{round_id} deployable score row count/key mismatch")
    features = scores[model["features_in_order"]].to_numpy(float)
    scores["directional_risk_frozen"] = float(model["intercept"]) + features @ np.asarray(model["coefficients_in_order"], float)
    if not np.isfinite(scores["directional_risk_frozen"]).all():
        raise RuntimeError("nonfinite directional risk score")
    scores["target_directional_truth_used_for_score"] = False
    scores.to_csv(scores_path, index=False)
    record = {"status": "SCORES_FIXED_BEFORE_DIRECTIONAL_EVALUATION",
              "created_at": datetime.now().isoformat(timespec="seconds"),
              "n_test_rows": len(scores), "score_sha256": sha256(scores_path),
              "risk_table_sha256": sha256(risk_path),
              "prediction_record_sha256": sha256(out / dataset / "PREDICTION_RECORDS.csv"),
              "model_sha256": sha256(MODEL), "manifest_sha256": sha256(manifest),
              "target_directional_truth_used_in_score_formula": False,
              "strict_prior_test_expression_read_blindness": False,
              "upstream_note": "E112 model implementation materializes test target vectors before score release; no pretruth claim."}
    (out / f"{round_id}_SCORE_STATUS.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
