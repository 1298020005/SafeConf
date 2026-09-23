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
OUT = ROOT / "docs/实验结果/E237_nadig_disjoint_gene_confirmation_20260923"
MANIFEST = OUT / "manifests/E237_TASK_MANIFEST.csv"
ASSET = Path("/home/yyf/data/safeconf_e237_nadig/Nadig_E237_disjoint.h5ad")
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
DATASET = "Nadig_E237_disjoint"
SCORES = OUT / "E237_SCORES_FIXED_BEFORE_DIRECTIONAL_EVALUATION.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    path = ROOT / "tools/scripts/run_e112_external_formal_dual_models.py"
    spec = importlib.util.spec_from_file_location("e112_for_e237", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CONTRACT = MANIFEST
    module.OUT = OUT
    module.SPECS = {DATASET: {"source": ASSET, "context": "context"}}
    module.ALIASES.update({"C17orf58": "C17orf58"})
    module.SEED = 202609237
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    contract = json.loads((OUT / "CONTRACT_STATUS.json").read_text())
    asset = json.loads((OUT / "E237_ASSET_STATUS.json").read_text())
    if (sha256(MANIFEST) != contract["manifest_sha256"]
            or sha256(MODEL) != contract["frozen_direction_model_sha256"]
            or sha256(ASSET) != asset["asset_sha256"]):
        raise RuntimeError("E237 contract/model/asset hash mismatch")
    if SCORES.exists() or (OUT / DATASET / "RUN_STATUS.json").exists():
        raise RuntimeError("refusing to overwrite existing E237 run")
    source = load_source()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = source.run_dataset(DATASET, device)
    if result["status"] != "complete" or result["strict_issue_count"] != 0 or result["n_test_tasks"] != contract["n_test_rows"]:
        raise RuntimeError(f"E237 upstream prediction incomplete: {result}")
    model = json.loads(MODEL.read_text())
    deploy_cols = ["fold_id", "task_id", "setting", "context", "perturbation",
                   "baseline_predicted_magnitude", "risk_model_disagreement", *model["features_in_order"]]
    risk_path = OUT / DATASET / "TASK_RISK_TABLE.csv"
    scores = pd.read_csv(risk_path, usecols=deploy_cols)
    if len(scores) != contract["n_test_rows"] or scores.duplicated(["fold_id", "task_id"]).any():
        raise RuntimeError("E237 deployable score row count/key mismatch")
    features = scores[model["features_in_order"]].to_numpy(float)
    scores["directional_risk_frozen"] = float(model["intercept"]) + features @ np.asarray(model["coefficients_in_order"], float)
    if not np.isfinite(scores["directional_risk_frozen"]).all():
        raise RuntimeError("nonfinite directional risk score")
    scores["target_directional_truth_used_for_score"] = False
    scores.to_csv(SCORES, index=False)
    record = {"status": "SCORES_FIXED_BEFORE_DIRECTIONAL_EVALUATION",
              "created_at": datetime.now().isoformat(timespec="seconds"),
              "n_test_rows": len(scores), "score_sha256": sha256(SCORES),
              "risk_table_sha256": sha256(risk_path),
              "prediction_record_sha256": sha256(OUT / DATASET / "PREDICTION_RECORDS.csv"),
              "model_sha256": sha256(MODEL), "manifest_sha256": sha256(MANIFEST),
              "target_directional_truth_used_in_score_formula": False,
              "strict_prior_test_expression_read_blindness": False,
              "upstream_note": "E112 model implementation materializes test target vectors before score release; no pretruth claim."}
    (OUT / "E237_SCORE_STATUS.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
