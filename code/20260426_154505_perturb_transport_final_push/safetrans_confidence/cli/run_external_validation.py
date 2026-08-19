#!/usr/bin/env python3
"""External-dataset validation of the SafeConf reliability model.

Trains the reliability model on the seven internal main datasets only, then
applies it without any refitting to entirely external datasets (different
studies, not in the main table and not sharing a study family with it). Positive
magnitude-controlled signal on the external sets is direct evidence that the
reliability model transfers to genuinely new data, not just to held-out folds of
the training datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.cli.run_safeconf_reliability_model import (
    _partial_rho,
    _risk_axis,
    _train_target_rank,
    load_corrected_base,
)
from safetrans_confidence.eval.selective_prediction import (
    clustered_bootstrap_aurc,
    selective_prediction_summary,
)
from safetrans_confidence.features.normalize import (
    available_transferable_features,
    normalize_features_within_group,
)
from safetrans_confidence.scoring.error_ranker import _make_model

TRAIN_PREDICTORS = ["V0StrongBaseline", "ContextSimBaseline"]


def run(internal_dirs: list[Path], external_dirs: list[Path], out_dir: Path, n_bootstrap: int) -> dict:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    internal, internal_status = load_corrected_base(internal_dirs)
    external, ext_status = load_corrected_base(external_dirs)
    pd.DataFrame(internal_status).to_csv(
        out_dir / "tables" / "INTERNAL_INPUT_STATUS.csv", index=False
    )
    pd.DataFrame(ext_status).to_csv(
        out_dir / "tables" / "EXTERNAL_INPUT_STATUS.csv", index=False
    )
    failed_inputs = [
        row for row in [*internal_status, *ext_status] if row.get("status") != "ok"
    ]
    if failed_inputs:
        status = {
            "out_dir": str(out_dir),
            "status": "failed_input_contract",
            "n_failed_inputs": len(failed_inputs),
            "failed_inputs": failed_inputs,
        }
        (out_dir / "RUN_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "external validation refused partial inputs; see EXTERNAL_INPUT_STATUS.csv"
        )
    internal["origin"] = "internal"
    external["origin"] = "external"

    # Normalize features within each (dataset, fold, predictor) group, separately
    # for internal and external so external normalization uses only its own
    # training distribution (realistic deployment).
    feat_cols = [c for c in available_transferable_features(internal) if c in external.columns]
    internal, norm_cols = normalize_features_within_group(internal, feat_cols)
    external, _ = normalize_features_within_group(external, feat_cols)

    # Train reliability model on internal V0+ContextSim train/val rows only.
    src = internal[
        internal["predictor_name"].isin(TRAIN_PREDICTORS) & internal["split"].isin(["train", "val"])
    ].copy()
    src["_y"] = _train_target_rank(src)
    src = src.dropna(subset=["_y"])
    model = _make_model(5201, model_type="histgbt", n_train_rows=len(src))
    model.fit(src[norm_cols].fillna(0.5).to_numpy(), src["_y"].to_numpy())

    # Apply, without refitting, to external test rows.
    tgt = external[
        external["predictor_name"].isin(TRAIN_PREDICTORS) & (external["split"] == "test")
    ].copy()
    tgt["score_value"] = model.predict(tgt[norm_cols].fillna(0.5).to_numpy())
    tgt["score_type"] = "risk"
    tgt["risk_axis"] = _risk_axis(tgt)
    if "task_key" not in tgt.columns:
        tgt["task_key"] = tgt.get("task_id", tgt["record_id"])

    rows = []
    for ds, g in tgt.groupby("dataset_name"):
        g = g.dropna(subset=["risk_axis", "true_error_rmse"])
        if len(g) < 5:
            rows.append({"dataset_name": ds, "n": int(len(g)), "status": "too_few_rows"})
            continue
        err = g["true_error_rmse"]
        aligned = err.corr(g["risk_axis"], method="spearman")
        partial = _partial_rho(g["risk_axis"], err, g["true_effect_l2_norm"])
        sp = selective_prediction_summary(err.to_numpy(), g["risk_axis"].to_numpy())
        boot = clustered_bootstrap_aurc(
            g, "true_error_rmse", "risk_axis", cluster_col="task_key", n_bootstrap=n_bootstrap
        )
        rows.append(
            {
                "dataset_name": ds, "n": int(len(g)), "status": "ok",
                "aligned_rho": float(aligned) if pd.notna(aligned) else np.nan,
                "partial_rho_control_magnitude": partial,
                "excess_aurc": sp["excess_aurc"],
                "aurc_reduction_vs_random_pct": sp["aurc_reduction_vs_random_pct"],
                "reduction_ci_low": boot.get("aurc_reduction_vs_random_pct_ci_low"),
                "reduction_ci_high": boot.get("aurc_reduction_vs_random_pct_ci_high"),
                "n_task_clusters": boot.get("n_clusters"),
            }
        )
    result = pd.DataFrame(rows).sort_values("dataset_name")
    result.to_csv(out_dir / "tables" / "EXTERNAL_VALIDATION_RESULT.csv", index=False)

    status = {
        "out_dir": str(out_dir),
        "internal_datasets": sorted(internal["dataset_name"].unique().tolist()),
        "external_datasets": sorted(external["dataset_name"].unique().tolist()),
        "n_train_rows": int(len(src)),
        "n_external_test_rows": int(len(tgt)),
        "n_internal_inputs": len(internal_status),
        "n_external_inputs": len(ext_status),
        "status": "ok",
    }
    (out_dir / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def main() -> None:
    p = argparse.ArgumentParser(description="External-dataset validation of SafeConf reliability model.")
    p.add_argument("--internal-dir", type=Path, action="append", dest="internal_dirs", required=True)
    p.add_argument("--external-dir", type=Path, action="append", dest="external_dirs", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    print(json.dumps(run(args.internal_dirs, args.external_dirs, args.out_dir, args.bootstrap), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
