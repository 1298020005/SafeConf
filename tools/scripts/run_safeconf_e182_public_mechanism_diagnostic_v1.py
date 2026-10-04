#!/usr/bin/env python3
"""Bounded post-freeze diagnosis for the E182 public-history transfer failure.

This is deliberately not a new main-method selection run.  E182's registered
calibration split is used to estimate one scalar response-scale correction per
frozen predictor member; the prospective split is then scored once.  The
diagnostic separates (i) an inaccurate public reference from (ii) a risk
learner that cannot use a useful reference.  Raw E182 target expression is
read only here, after the SafeConf configuration was frozen, and all outputs
are labelled exploratory/post-freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parents[2]
E182 = Path("/home/yyf/data/safeconf_e182_gse225807")
E182_DOC = Path("/home/yyf/proj/docs/实验结果/E182_gse225807_registered_family_20260724")
DEFAULT_OUT = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/e182_public_mechanism_diagnostic_v1")


def load_crosscontext_module():
    path = ROOT / "tools/scripts/run_safeconf_e192_crosscontext_v1.py"
    spec = importlib.util.spec_from_file_location("safeconf_e192_crosscontext_diag", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mod = load_crosscontext_module()

    panel = pd.read_csv(E182 / "isolated/F2_pretruth/GENE_PANEL.csv")
    panel_axis_sha256 = hashlib.sha256("\n".join(panel.gene_name.astype(str)).encode("utf-8")).hexdigest()
    pretruth_tasks = pd.read_csv(E182 / "isolated/F2_pretruth/PRETRUTH_TASKS.csv")
    eval_tasks = pd.read_csv(E182_DOC / "final_evaluation/tables/E182_EVALUATION_TASKS.csv")
    cal_tasks = pd.read_csv(E182_DOC / "calibration_release/tables/CALIBRATION_TASK_ERRORS.csv")
    pred = np.load(E182_DOC / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz")
    cal_truth = np.load(E182_DOC / "calibration_release/arrays/CALIBRATION_TRUTH.npz")
    eval_truth = np.load(E182_DOC / "final_evaluation/arrays/E182_EVALUATION_TRUTH.npz")

    target_genes = sorted(set(cal_tasks.perturbation.astype(str)) | set(eval_tasks.perturbation.astype(str)))
    effect_path = out / "E182_K562_PUBLIC_EFFECTS_10000_ALL.npz"
    if effect_path.exists():
        z = np.load(effect_path)
        effects = {k: z[k].astype(float) for k in z.files}
        stream_audit = {"reused_effect_artifact": True, "n_effects": len(effects), "target_truth_read": False}
    else:
        effects, stream_audit = mod.stream_k562_effects(panel, effect_path, target_genes=target_genes)

    # The raw GWPS axis is the common available public panel.  E182's truth
    # and predictions use the registered 512-gene panel; no missing gene is
    # imputed into the comparison.
    with h5py.File(mod.GWPS, "r") as f:
        raw_gene_names = set(mod.h5col(f["var"], "gene_name").astype(str))
    panel_genes = panel.gene_name.astype(str).tolist()
    panel_mask = np.asarray([g in raw_gene_names for g in panel_genes], dtype=bool)
    common_genes = int(panel_mask.sum())
    row_map = {str(t): i for i, t in enumerate(pretruth_tasks.task_id.astype(str))}
    cal_rows = np.asarray([row_map[str(t)] for t in cal_tasks.task_id], dtype=int)
    eval_rows = np.asarray([row_map[str(t)] for t in eval_tasks.task_id], dtype=int)

    cal_scored = cal_tasks.perturbation.astype(str).isin(set(effects)).to_numpy()
    eval_scored = eval_tasks.perturbation.astype(str).isin(set(effects)).to_numpy()
    cal = cal_tasks.loc[cal_scored].reset_index(drop=True)
    ev = eval_tasks.loc[eval_scored].reset_index(drop=True)
    cal_rows = cal_rows[cal_scored]
    eval_rows = eval_rows[eval_scored]
    missing_cal = sorted(set(cal_tasks.perturbation.astype(str)) - set(effects))
    missing_eval = sorted(set(eval_tasks.perturbation.astype(str)) - set(effects))

    # Build the scale diagnostic for each fixed predictor member.  The
    # calibration truth is read only after the main system freeze and is never
    # used to alter the frozen PublicRule result.
    rows = []
    boot = []
    for member in pred.files:
        cal_pred = pred[member][cal_rows][:, panel_mask].astype(float)
        ev_pred = pred[member][eval_rows][:, panel_mask].astype(float)
        cal_y = np.vstack([cal_truth[str(t)][panel_mask] for t in cal.task_id.astype(str)])
        ev_y = np.vstack([eval_truth[str(t)][panel_mask] for t in ev.task_id.astype(str)])
        ev_error_col = f"{member}_rmse"
        if ev_error_col not in ev.columns:
            continue
        ev_truth_error = ev[ev_error_col].to_numpy(float)
        cal_hist = np.vstack([effects[str(g)] for g in cal.perturbation.astype(str)])
        ev_hist = np.vstack([effects[str(g)] for g in ev.perturbation.astype(str)])
        # Fit a single scale on calibration response vectors.  This is a
        # mechanism probe for source/recipient assay scale, not a new default.
        numerator = float(np.sum(cal_hist * (cal_pred - cal_y)))
        denominator = float(np.sum(cal_hist * cal_hist))
        alpha = numerator / denominator if denominator > 1e-15 else float("nan")
        cal_public = np.sqrt(np.mean((cal_pred - cal_hist) ** 2, axis=1))
        cal_scaled = np.sqrt(np.mean((cal_pred - alpha * cal_hist) ** 2, axis=1)) if np.isfinite(alpha) else np.full(len(cal), np.nan)
        ev_public = np.sqrt(np.mean((ev_pred - ev_hist) ** 2, axis=1))
        ev_scaled = np.sqrt(np.mean((ev_pred - alpha * ev_hist) ** 2, axis=1)) if np.isfinite(alpha) else np.full(len(ev), np.nan)
        ev_amp = np.sqrt(np.mean(ev_pred ** 2, axis=1))
        ref_cal = np.sqrt(np.mean((cal_hist - cal_y) ** 2, axis=1))
        cal_truth_error = np.sqrt(np.mean((cal_pred - cal_y) ** 2, axis=1))
        # E182's calibration table has the same registered error columns.
        cal_error = cal[ev_error_col].to_numpy(float) if ev_error_col in cal.columns else cal_truth_error
        for split, frame, truth_error, amp, public, scaled, ref in [
            ("calibration", cal, cal_error, np.sqrt(np.mean(cal_pred ** 2, axis=1)), cal_public, cal_scaled, ref_cal),
            ("prospective", ev, ev_truth_error, ev_amp, ev_public, ev_scaled, np.sqrt(np.mean((ev_hist - ev_y) ** 2, axis=1))),
        ]:
            for method, score in [("Amplitude", amp), ("PublicRule", public), ("ScalarPublicScale", scaled)]:
                if not np.isfinite(score).all():
                    continue
                metrics = mod.task_metric(frame.task_id, truth_error, score)
                rows.append({"model": member, "split": split, "method": method, "alpha": alpha,
                             "public_reference_rmse_mean": float(np.mean(ref)),
                             **metrics})
        if np.isfinite(alpha):
            boot.append({"model": member, "error_column": ev_error_col,
                         "comparison": "ScalarPublicScale_minus_Amplitude",
                         **mod.bootstrap_delta(ev.task_id, ev.perturbation, ev_truth_error, ev_scaled, ev_amp)})

    comparison = pd.DataFrame(rows)
    bootstrap = pd.DataFrame(boot)
    comparison.to_csv(out / "E182_PUBLIC_MECHANISM_DIAGNOSTIC.csv", index=False, lineterminator="\n")
    bootstrap.to_csv(out / "E182_PUBLIC_MECHANISM_DIAGNOSTIC_BOOTSTRAP.csv", index=False, lineterminator="\n")
    write_json(out / "E182_PUBLIC_MECHANISM_DIAGNOSTIC.json", {
        "status": "COMPLETE_POST_FREEZE_MECHANISM_DIAGNOSTIC",
        "study": "GSE225807",
        "calibration_split": "E182 calibration_release; 19 target clusters / 38 tasks before public-history filtering",
        "prospective_split": "E182 final_evaluation; 20 target clusters / 40 tasks before public-history filtering",
        "target_truth_read": True,
        "main_method_selection_changed": False,
        "post_freeze": True,
        "scale_correction": "one scalar alpha per frozen predictor member, fitted by least squares on calibration response vectors only",
        "common_axis_genes": common_genes,
        "panel_axis_sha256": panel_axis_sha256,
        "n_public_effects": len(effects),
        "missing_calibration_genes": missing_cal,
        "missing_prospective_genes": missing_eval,
        "n_calibration_scored_tasks": int(len(cal)),
        "n_prospective_scored_tasks": int(len(ev)),
        "stream_audit": stream_audit,
        "interpretation": "exploratory mechanism diagnosis; not used to replace the frozen PublicRule or to claim prospective confirmation",
        "permanent_test_truth_opened": False,
    })
    print(comparison.to_string(index=False))
    print(bootstrap.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
