#!/usr/bin/env python3
"""Independent-study E182 public-history risk audit.

E182 (GSE225807) is a different K562 study from the Replogle GWPS public
source.  The source effects are rebuilt at E182's registered 1e4 scale from
the existing GWPS raw file.  E182 target truth is not read here; the frozen
task-level error columns from its completed evaluation are used only for the
predefined retrospective score audit.
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
sys.path.insert(0, str(ROOT))
E182 = Path("/home/yyf/data/safeconf_e182_gse225807")
E182_DOC = Path("/home/yyf/proj/docs/实验结果/E182_gse225807_registered_family_20260724")
DEFAULT_OUT = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/e182_independent_public")
EFFECT_FILE_NAME = "E182_K562_PUBLIC_EFFECTS_10000.npz"


def load_crosscontext_module():
    path = ROOT / "tools/scripts/run_safeconf_e192_crosscontext_v1.py"
    spec = importlib.util.spec_from_file_location("safeconf_e192_crosscontext", path)
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
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=DEFAULT_OUT); args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    mod = load_crosscontext_module()
    panel = pd.read_csv(E182 / "isolated/F2_pretruth/GENE_PANEL.csv")
    panel_axis_sha256 = hashlib.sha256("\n".join(panel.gene_name.astype(str)).encode("utf-8")).hexdigest()
    effect_path = out / EFFECT_FILE_NAME
    if effect_path.exists():
        effects = {k: np.load(effect_path)[k] for k in np.load(effect_path).files}
        stream_audit = {"reused_effect_artifact": True, "n_effects": len(effects), "target_truth_read": False}
    else:
        evaluation_targets = pd.read_csv(E182_DOC / "final_evaluation/tables/E182_EVALUATION_TASKS.csv")
        effects, stream_audit = mod.stream_k562_effects(panel, effect_path,
                                                        target_genes=sorted(evaluation_targets.perturbation.astype(str).unique()))
    qorder = E182_DOC / "pretruth_release/tables/QUERY_ORDER.csv"
    pretruth_tasks = pd.read_csv(qorder) if qorder.exists() else pd.read_csv(E182 / "isolated/F2_pretruth/PRETRUTH_TASKS.csv")
    # QUERY_ORDER/PRETRUTH_TASKS contain task IDs and row order for the frozen arrays.
    if "task_id" not in pretruth_tasks.columns:
        raise RuntimeError("E182 pretruth task order lacks task_id")
    eval_tasks = pd.read_csv(E182_DOC / "final_evaluation/tables/E182_EVALUATION_TASKS.csv")
    metrics = eval_tasks.set_index("task_id")
    pred = np.load(E182_DOC / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz")
    row_map = {str(task): i for i, task in enumerate(pretruth_tasks.task_id.astype(str))}
    eval_rows = np.asarray([row_map[str(task)] for task in eval_tasks.task_id], dtype=int)
    panel_genes = panel.gene_name.astype(str).tolist()
    with h5py.File(mod.GWPS, "r") as f:
        raw_gene_names = set(mod.h5col(f["var"], "gene_name").astype(str))
    panel_mask = np.asarray([g in raw_gene_names for g in panel_genes], dtype=bool)
    scored = eval_tasks.perturbation.astype(str).isin(set(effects)).to_numpy()
    scored_tasks = eval_tasks.loc[scored].reset_index(drop=True)
    scored_rows = eval_rows[scored]
    rows, paired = [], []
    for member in pred.files:
        error_col = f"{member}_rmse"
        if error_col not in metrics.columns:
            continue
        vectors = pred[member][scored_rows][:, panel_mask]
        amplitude = np.sqrt(np.mean(vectors ** 2, axis=1))
        public = np.asarray([np.sqrt(np.mean((vectors[i] - effects[scored_tasks.perturbation.iloc[i]]) ** 2)) for i in range(len(scored_tasks))])
        truth = metrics.loc[scored_tasks.task_id.astype(str), error_col].to_numpy(float)
        for method, score in [("Amplitude", amplitude), ("PublicRule_E182_GWPS_K56210000", public)]:
            rows.append({"model": member, "method": method, "error_column": error_col,
                         **mod.task_metric(scored_tasks.task_id, truth, score)})
        paired.append({"model": member, "error_column": error_col,
                       **mod.bootstrap_delta(scored_tasks.task_id, scored_tasks.perturbation, truth, public, amplitude)})
    result = pd.DataFrame(rows); comparison = pd.DataFrame(paired)
    result.to_csv(out / "E182_INDEPENDENT_PUBLIC_SYSTEM_COMPARISON.csv", index=False, lineterminator="\n")
    comparison.to_csv(out / "E182_INDEPENDENT_PUBLIC_PAIRED_BOOTSTRAP.csv", index=False, lineterminator="\n")
    write_json(out / "E182_INDEPENDENT_PUBLIC_AUDIT.json", {
        "status": "COMPLETE_INDEPENDENT_STUDY_SEEN_SCORE_AUDIT",
        "study": "GSE225807", "source_public": "Replogle2022_K562_GWPS",
        "n_evaluation_tasks": len(eval_tasks), "scored_tasks": len(scored_tasks),
        "evaluation_genes": int(eval_tasks.perturbation.nunique()),
        "scored_genes": int(scored_tasks.perturbation.nunique()),
        "missing_public_history_genes": sorted(set(eval_tasks.perturbation.astype(str)) - set(effects)),
        "effect_contract": "both E182 and rebuilt public effects use log1p 1e4 normalized panel effects",
        "panel_axis_sha256": panel_axis_sha256,
        "target_truth_read_by_this_script": False,
        "target_error_source": "pre-existing E182_EVALUATION_TASKS.csv only",
        "independent_study": True,
        "stream_audit": stream_audit,
        "model_members": pred.files,
        "permanent_test_truth_opened": False,
    })
    print(result.to_string(index=False)); print(comparison.to_string(index=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
