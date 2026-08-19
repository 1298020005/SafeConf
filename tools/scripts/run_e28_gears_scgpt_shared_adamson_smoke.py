#!/usr/bin/env python3
"""E28: GEARS–scGPT shared Adamson PredictionRecord smoke.

This is the first strict multi-predictor smoke after E25–E27.

It aligns:
- GEARS formal Adamson seed-1 predictions from E25;
- scGPT whole-human forward predictions through the E27 adapter path;
- one shared Adamson K562/control-derived true effect per perturbation;
- one shared 512-gene panel ordered by the GEARS processed h5ad.

The output is not a formal benchmark.  It is a contract and feasibility smoke
showing that two real model families can be represented under the same
PredictionRecord task/gene/true-effect contract.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code/20260426_154505_perturb_transport_final_push"
TOOLS_ROOT = PROJECT_ROOT / "tools/scripts"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

from safetrans_confidence.data.records import validate_prediction_record_artifacts
from run_e27_scgpt_forward_prediction_record_smoke import (
    _build_model as build_scgpt_model,
    _build_tensors as build_scgpt_tensors,
    _cosine_error,
    _dense,
    _load_checkpoint,
    _safe_corr,
)


E25_DIR = PROJECT_ROOT / "docs/实验结果/E25_gears_strict_prediction_records_20260708"
ADAMSON_H5AD = Path("/home/yyf/data/singlecell_perturbation_atlas/official_generalization/Adamson.h5ad")
GEARS_ADAMSON_PROCESSED = Path("/home/yyf/data/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad")
OUT_DIR = PROJECT_ROOT / "docs/实验结果/E28_gears_scgpt_shared_adamson_smoke_20260708"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT)
        return bool(out.decode("utf-8").strip())
    except Exception:
        return True


def _rel(path: Path) -> str:
    text = str(path)
    for prefix, token in [
        ("/home/yyf/data", "$YYF_DATA"),
        ("/home/yyf/archive", "$YYF_ARCHIVE"),
        ("/home/yyf/proj", "."),
    ]:
        if text.startswith(prefix):
            return token + text[len(prefix) :]
    return text


def _hash_gene_order(genes: list[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()


def _load_gears_gene_order() -> list[str]:
    backed = ad.read_h5ad(GEARS_ADAMSON_PROCESSED, backed="r")
    try:
        return [str(g) for g in backed.var_names]
    finally:
        backed.file.close()


def _select_gears_tasks(n_tasks: int = 3) -> pd.DataFrame:
    rec = pd.read_csv(E25_DIR / "tables/PREDICTION_RECORDS.csv")
    sub = rec[(rec["dataset_name"].eq("adamson")) & (rec["fold_id"].eq(1))].copy()
    sub["perturbation_gene"] = sub["perturbation"].astype(str).str.replace("+ctrl", "", regex=False)
    return sub.sort_values("perturbation_gene").head(n_tasks).reset_index(drop=True)


def _select_adamson_subset(
    checkpoint_vocab: object,
    gears_tasks: pd.DataFrame,
    gears_gene_order: list[str],
    cells_per_pert: int = 8,
    max_genes: int = 512,
) -> dict[str, Any]:
    backed = ad.read_h5ad(ADAMSON_H5AD, backed="r")
    try:
        obs = backed.obs[["perturbation"]].copy()
        var_names = [str(g) for g in backed.var_names]
        var_set = set(var_names)
        labels = obs["perturbation"].astype(str).to_numpy()
        perturbations = ["control"] + gears_tasks["perturbation_gene"].astype(str).tolist()
        missing = [p for p in perturbations if p != "control" and p not in set(labels)]
        if missing:
            raise RuntimeError(f"selected perturbations missing from Adamson h5ad: {missing}")
        indices: list[int] = []
        for perturbation in perturbations:
            idx = np.flatnonzero(labels == perturbation)[:cells_per_pert]
            if len(idx) < cells_per_pert:
                raise RuntimeError(f"not enough cells for {perturbation}: {len(idx)}")
            indices.extend(idx.tolist())
        subset_view = backed[indices, :]
        subset = subset_view.to_memory() if hasattr(subset_view, "to_memory") else subset_view.copy()
    finally:
        if getattr(backed, "file", None) is not None:
            backed.file.close()

    subset.var_names_make_unique()
    if "logNor" in subset.layers:
        expr_all = _dense(subset.layers["logNor"])
        expression_layer = "logNor"
    else:
        expr_all = _dense(subset.X)
        expression_layer = "X"
    expr_all = np.nan_to_num(expr_all, nan=0.0, posinf=0.0, neginf=0.0)
    subset_var = [str(g) for g in subset.var_names]
    subset_var_set = set(subset_var)
    vocab_set = set(checkpoint_vocab.get_stoi().keys())

    candidate_genes = [g for g in gears_gene_order if g in subset_var_set and g in vocab_set]
    forced = gears_tasks["perturbation_gene"].astype(str).tolist()
    selected_set: set[str] = {g for g in forced if g in candidate_genes}
    var_index = {g: i for i, g in enumerate(subset_var)}
    gene_sums = {
        g: float(expr_all[:, var_index[g]].sum())
        for g in candidate_genes
        if g in var_index
    }
    for gene, _score in sorted(gene_sums.items(), key=lambda kv: -kv[1]):
        selected_set.add(gene)
        if len(selected_set) >= max_genes:
            break
    selected_genes = [g for g in gears_gene_order if g in selected_set]
    if len(selected_genes) < max_genes:
        raise RuntimeError(f"selected only {len(selected_genes)} genes; expected {max_genes}")
    selected_genes = selected_genes[:max_genes]
    expr = expr_all[:, [var_index[g] for g in selected_genes]].astype(np.float32)
    selected_subset = subset[:, selected_genes].copy()
    return {
        "subset": selected_subset,
        "expression": expr,
        "expression_layer": expression_layer,
        "selected_perturbations": perturbations,
        "selected_genes": selected_genes,
        "cells_per_perturbation": {
            p: int((selected_subset.obs["perturbation"].astype(str) == p).sum()) for p in perturbations
        },
    }


def _run_scgpt_forward(
    checkpoint: dict[str, Any],
    subset_payload: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    perturbations = subset_payload["subset"].obs["perturbation"].astype(str).tolist()
    tensors = build_scgpt_tensors(
        expression=subset_payload["expression"],
        genes=subset_payload["selected_genes"],
        perturbations=perturbations,
        vocab=checkpoint["vocab"],
        args=checkpoint["args"],
    )
    model, model_info = build_scgpt_model(
        checkpoint["vocab"],
        checkpoint["args"],
        checkpoint["state"],
        subset_payload["selected_perturbations"],
        device,
    )
    batch = {k: v.to(device) for k, v in tensors.items()}
    valid = ~batch["src_key_padding_mask"]
    if "<cls>" in checkpoint["vocab"]:
        valid = valid & batch["gene_ids"].ne(checkpoint["vocab"]["<cls>"])
    full_mask_values = batch["target_values"].clone().masked_fill(
        valid, float(checkpoint["args"].get("mask_value", -1))
    )
    model.eval()
    with torch.no_grad():
        out = model(
            batch["gene_ids"],
            full_mask_values,
            batch["pert_flags"],
            batch["src_key_padding_mask"],
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )
    pred = out["mlm_output"].detach().cpu().numpy()
    target = batch["target_values"].detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    pred_rows = [pred[i][valid_np[i]] for i in range(pred.shape[0])]
    target_rows = [target[i][valid_np[i]] for i in range(target.shape[0])]
    pred_matrix = np.stack(pred_rows).astype(np.float32)
    target_matrix = np.stack(target_rows).astype(np.float32)
    pert_array = np.asarray(perturbations, dtype=object)
    control_mask = pert_array == "control"
    control_pred = pred_matrix[control_mask].mean(axis=0)
    control_true = target_matrix[control_mask].mean(axis=0)
    pred_delta: dict[str, np.ndarray] = {}
    true_delta: dict[str, np.ndarray] = {}
    for perturbation in subset_payload["selected_perturbations"]:
        if perturbation == "control":
            continue
        mask = pert_array == perturbation
        pred_delta[perturbation] = (pred_matrix[mask].mean(axis=0) - control_pred).astype(np.float32)
        true_delta[perturbation] = (target_matrix[mask].mean(axis=0) - control_true).astype(np.float32)
    model_info.update(
        {
            "prediction_matrix_mean": float(pred_matrix.mean()),
            "prediction_matrix_std": float(pred_matrix.std()),
            "target_matrix_mean": float(target_matrix.mean()),
            "target_matrix_std": float(target_matrix.std()),
        }
    )
    return pred_delta, true_delta, model_info


def _subset_gears_predictions(
    gears_tasks: pd.DataFrame,
    selected_genes: list[str],
    gears_gene_order: list[str],
) -> dict[str, np.ndarray]:
    index = [gears_gene_order.index(g) for g in selected_genes]
    with np.load(E25_DIR / "arrays/gears_predicted_effects.npz") as pred_npz:
        out: dict[str, np.ndarray] = {}
        for row in gears_tasks.to_dict("records"):
            gene = str(row["perturbation_gene"])
            arr = np.asarray(pred_npz[str(row["predicted_effect_key"])], dtype=np.float32)
            out[gene] = arr[index].astype(np.float32)
        return out


def _build_records(
    gears_tasks: pd.DataFrame,
    selected_genes: list[str],
    scgpt_pred: dict[str, np.ndarray],
    shared_true: dict[str, np.ndarray],
    gears_pred: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    gene_hash = _hash_gene_order(selected_genes)
    gene_panel_id = f"shared::adamson::gears_scgpt::n_genes_{len(selected_genes)}"
    pred_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for gene in gears_tasks["perturbation_gene"].astype(str).tolist():
        true_key = f"E28::Adamson::{gene}::shared_true"
        true_arrays[true_key] = shared_true[gene].astype(np.float32)
        predictors = [
            ("GEARS_formal_seed1_subset512", gears_pred[gene]),
            ("scGPT_whole_human_forward_subset512", scgpt_pred[gene]),
        ]
        for predictor_name, pred_vec in predictors:
            record_id = f"E28::Adamson::{gene}::{predictor_name}"
            pred_key = record_id + "::pred"
            pred_arrays[pred_key] = pred_vec.astype(np.float32)
            rmse = float(np.sqrt(np.mean((pred_vec - shared_true[gene]) ** 2)))
            cosine = _cosine_error(pred_vec, shared_true[gene])
            pearson = _safe_corr(pred_vec, shared_true[gene], "pearson")
            spearman = _safe_corr(pred_vec, shared_true[gene], "spearman")
            rows.append(
                {
                    "schema_version": "safeconf_prediction_record_v1",
                    "record_id": record_id,
                    "task_id": gene,
                    "task_key": f"Adamson::{gene}",
                    "dataset_name": "Adamson_GEARS_scGPT_shared_smoke",
                    "dataset_group": "adamson_crispr_shared_smoke",
                    "fold_id": 0,
                    "split": "test",
                    "context": "Adamson_shared_panel_smoke",
                    "perturbation": gene,
                    "predictor_name": predictor_name,
                    "run_type": "smoke",
                    "gene_panel_id": gene_panel_id,
                    "gene_order_hash": gene_hash,
                    "effect_definition": "mean_diff",
                    "normalization_id": "adamson_shared_expression_delta_subset_v1",
                    "error_normalization": "raw_rmse",
                    "predicted_effect_key": pred_key,
                    "true_effect_key": true_key,
                    "true_error_rmse": rmse,
                    "true_error_cosine": cosine,
                    "n_cells": 8,
                }
            )
            metrics.append(
                {
                    "perturbation": gene,
                    "predictor_name": predictor_name,
                    "rmse": rmse,
                    "cosine_error": cosine,
                    "pearson": pearson,
                    "spearman": spearman,
                    "pred_l2": float(np.linalg.norm(pred_vec)),
                    "true_l2": float(np.linalg.norm(shared_true[gene])),
                }
            )
    return pd.DataFrame(rows), pred_arrays, true_arrays, pd.DataFrame(metrics)


def _write_report(status: dict[str, Any], metrics: pd.DataFrame, manifest: pd.DataFrame, validation: pd.DataFrame) -> None:
    md = f"""# E28 GEARS–scGPT shared Adamson smoke

生成时间：{_now()}

## 结论

E28 在 Adamson 上构造了一个最小的 GEARS–scGPT 同任务、同 gene panel、同 true effect 的 strict smoke。

- predictors：GEARS_formal_seed1_subset512；scGPT_whole_human_forward_subset512
- perturbations：{', '.join(status['perturbations'])}
- genes：{status['n_genes']}
- PredictionRecords：{status['n_prediction_records']}
- strict issue_count：{status['strict_issue_count']}

边界：这是合同和对齐 smoke，不是正式性能 benchmark。scGPT 使用 forward-only adapter；true effect 来自同一 Adamson 小子集，用于统一合同检查。
"""
    (OUT_DIR / "reports/E28_GEARS_SCGPT_SHARED_ADAMSON_SMOKE_REPORT.md").write_text(md, encoding="utf-8")
    page = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>E28 GEARS scGPT shared Adamson smoke</title>
<style>
body{{margin:0;background:#fff;color:#1f2933;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif;}}
main{{max-width:1120px;margin:0 auto;padding:42px 28px 72px;}}
h1{{font-size:30px;margin:0 0 10px;}} h2{{font-size:21px;margin:34px 0 12px;border-top:1px solid #e5e7eb;padding-top:24px;}}
p{{line-height:1.75;font-size:16px;}} table{{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0 22px;}}
th,td{{border-bottom:1px solid #e5e7eb;text-align:left;padding:9px 10px;vertical-align:top;}} th{{background:#f7f7f7;}}
.note{{border-left:4px solid #4677C8;background:#f8fbff;padding:12px 16px;border-radius:8px;}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0;}} .card{{border:1px solid #e5e7eb;border-radius:14px;padding:16px;background:#fafafa;}}
.k{{font-size:26px;font-weight:760;color:#111827;}} .l{{font-size:13px;color:#66788a;margin-top:4px;}}
</style></head>
<body><main>
<h1>E28 GEARS–scGPT shared Adamson smoke</h1>
<p class="note">同任务、同 512-gene panel、同 true effect key 的双预测器 strict smoke。不是正式性能 benchmark。</p>
<div class="cards">
<div class="card"><div class="k">{status['n_prediction_records']}</div><div class="l">PredictionRecords</div></div>
<div class="card"><div class="k">{status['n_tasks']}</div><div class="l">tasks</div></div>
<div class="card"><div class="k">{status['n_genes']}</div><div class="l">shared genes</div></div>
<div class="card"><div class="k">{status['strict_issue_count']}</div><div class="l">strict issues</div></div>
</div>
<h2>Manifest</h2>{manifest.to_html(index=False, escape=False)}
<h2>Metrics</h2>{metrics.to_html(index=False, escape=False)}
<h2>Validation</h2>{validation.to_html(index=False, escape=False)}
</main></body></html>
"""
    (OUT_DIR / "reports/E28_GEARS_SCGPT_SHARED_ADAMSON_SMOKE.html").write_text(page, encoding="utf-8")


def _write_readme(status: dict[str, Any]) -> None:
    (OUT_DIR / "README_先看这个.md").write_text(
        f"""# E28 GEARS–scGPT shared Adamson smoke

先看结论：E28 已经把 GEARS 和 scGPT 放进同一个 Adamson 512-gene strict smoke 合同中。

- PredictionRecords：{status['n_prediction_records']}
- Tasks：{status['n_tasks']}
- Genes：{status['n_genes']}
- strict issue：{status['strict_issue_count']}

边界：这是同任务/gene panel 对齐 smoke，不是正式 benchmark。下一步才是扩大任务数、固定 split，并把 scGPT adapter 做成正式 predictor 输出。
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for rel in ["tables", "arrays", "reports"]:
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(5228)
    checkpoint = _load_checkpoint()
    gears_tasks = _select_gears_tasks(n_tasks=3)
    gears_gene_order = _load_gears_gene_order()
    subset_payload = _select_adamson_subset(checkpoint["vocab"], gears_tasks, gears_gene_order)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scgpt_pred, shared_true, scgpt_info = _run_scgpt_forward(checkpoint, subset_payload, device)
    gears_pred = _subset_gears_predictions(gears_tasks, subset_payload["selected_genes"], gears_gene_order)
    records, pred_arrays, true_arrays, metrics = _build_records(
        gears_tasks, subset_payload["selected_genes"], scgpt_pred, shared_true, gears_pred
    )
    records.to_csv(OUT_DIR / "tables/PREDICTION_RECORDS.csv", index=False)
    records.to_csv(OUT_DIR / "tables/E28_SHARED_PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(OUT_DIR / "arrays/predicted_effects.npz", **pred_arrays)
    np.savez_compressed(OUT_DIR / "arrays/true_effects.npz", **true_arrays)
    metrics.to_csv(OUT_DIR / "tables/E28_SHARED_PREDICTOR_METRICS.csv", index=False)
    pd.DataFrame({"gene": subset_payload["selected_genes"]}).to_csv(
        OUT_DIR / "tables/E28_SHARED_GENE_PANEL.csv", index=False
    )
    manifest = gears_tasks[
        ["fold_id", "perturbation", "perturbation_gene", "record_id", "predicted_effect_key"]
    ].copy()
    manifest["scgpt_cells"] = manifest["perturbation_gene"].map(
        subset_payload["cells_per_perturbation"]
    )
    manifest["shared_gene_count"] = len(subset_payload["selected_genes"])
    manifest.to_csv(OUT_DIR / "tables/E28_SHARED_TASK_MANIFEST.csv", index=False)
    issues = validate_prediction_record_artifacts(OUT_DIR, records=records, strict=True)
    validation = pd.DataFrame(
        [{"scope": "e28_gears_scgpt_shared_adamson_smoke", "strict": True, "issue_count": len(issues), "issues": "; ".join(issues)}]
    )
    validation.to_csv(OUT_DIR / "tables/E28_SHARED_VALIDATION.csv", index=False)
    status = {
        "status": "ok" if not issues else "has_issues",
        "generated_at": _now(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "out": os.path.relpath(OUT_DIR, PROJECT_ROOT),
        "device": str(device),
        "n_prediction_records": int(len(records)),
        "n_tasks": int(manifest.shape[0]),
        "n_genes": int(len(subset_payload["selected_genes"])),
        "perturbations": manifest["perturbation_gene"].astype(str).tolist(),
        "strict_issue_count": int(len(issues)),
        "strict_issues": issues,
        "cells_per_perturbation": subset_payload["cells_per_perturbation"],
        "scgpt_matched_key_count": scgpt_info["matched_key_count"],
        "scgpt_total_model_key_count": scgpt_info["total_model_key_count"],
        "sources": {
            "e25": os.path.relpath(E25_DIR, PROJECT_ROOT),
            "adamson_h5ad": _rel(ADAMSON_H5AD),
            "gears_processed": _rel(GEARS_ADAMSON_PROCESSED),
        },
    }
    (OUT_DIR / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(status, metrics, manifest, validation)
    _write_readme(status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
