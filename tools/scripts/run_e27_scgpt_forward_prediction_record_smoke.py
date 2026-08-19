#!/usr/bin/env python3
"""E27: scGPT forward-only PredictionRecord smoke.

This script turns the archived scGPT assets from "present but not importable by
default" into a concrete, strict SafeConf smoke package:

- uses the archived scGPT source path through ``sys.path``;
- loads the archived whole-human checkpoint;
- selects a tiny Replogle K562 essential subset with genes covered by the
  checkpoint vocabulary;
- runs a forward-only TransformerGenerator pass;
- exports per-perturbation predicted/true delta vectors as PredictionRecords;
- validates the strict SafeConf contract.

It is intentionally a smoke test.  It does not claim formal scGPT performance.
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
SCGPT_REPO = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/codex_scgpt_attnres_workspace/repo"
)
SCGPT_CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
ADATA_PATH = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ReplogleWeissman2022_K562_essential.h5ad"
)
OUT_DIR = PROJECT_ROOT / "docs/实验结果/E27_scgpt_forward_prediction_record_smoke_20260708"

sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(SCGPT_REPO))

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

from safetrans_confidence.data.records import validate_prediction_record_artifacts
from scgpt.model import TransformerGenerator
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained, set_seed


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
        ("/home/yyf/archive", "$YYF_ARCHIVE"),
        ("/home/yyf/data", "$YYF_DATA"),
        ("/home/yyf/proj", "."),
    ]:
        if text.startswith(prefix):
            return token + text[len(prefix) :]
    return text


def _hash_gene_order(genes: list[str]) -> str:
    payload = "\n".join(genes).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _dense(x) -> np.ndarray:
    if issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _cosine_error(pred: np.ndarray, true: np.ndarray) -> float:
    denom = float(np.linalg.norm(pred) * np.linalg.norm(true) + 1e-8)
    return float(1.0 - np.dot(pred, true) / denom)


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if method == "spearman":
        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    return float(np.corrcoef(x, y)[0, 1])


def _load_checkpoint() -> dict[str, Any]:
    args_path = SCGPT_CHECKPOINT / "args.json"
    vocab_path = SCGPT_CHECKPOINT / "vocab.json"
    model_path = SCGPT_CHECKPOINT / "best_model.pt"
    if not (args_path.exists() and vocab_path.exists() and model_path.exists()):
        raise FileNotFoundError(f"incomplete scGPT checkpoint bundle: {SCGPT_CHECKPOINT}")
    args = json.loads(args_path.read_text(encoding="utf-8"))
    vocab = GeneVocab.from_file(vocab_path)
    vocab.set_default_token("<pad>")
    state = torch.load(model_path, map_location="cpu")
    return {"args": args, "vocab": vocab, "state": state}


def _select_subset(vocab: GeneVocab, cells_per_pert: int = 4, max_perts: int = 4, max_genes: int = 128) -> dict[str, Any]:
    backed = ad.read_h5ad(ADATA_PATH, backed="r")
    try:
        obs = backed.obs[["perturbation", "gene"]].copy()
        var_names = [str(g) for g in backed.var_names]
        var_set = set(var_names)
        counts = obs["perturbation"].astype(str).value_counts()
        selected = ["control"]
        for perturbation, _count in counts.items():
            perturbation = str(perturbation)
            if perturbation in {"control", "non-targeting", "CTRL"}:
                continue
            if perturbation in var_set and perturbation in vocab:
                selected.append(perturbation)
            if len(selected) >= max_perts:
                break
        if len(selected) < 2:
            raise RuntimeError("No scGPT-vocab-compatible perturbations found.")
        labels = obs["perturbation"].astype(str).to_numpy()
        indices: list[int] = []
        for perturbation in selected:
            idx = np.flatnonzero(labels == perturbation)[:cells_per_pert]
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

    gene_names_all = [str(g) for g in subset.var_names]
    gene_sums = np.asarray(expr_all.sum(axis=0)).ravel()
    selected_genes: list[str] = []
    seen: set[str] = set()
    for gene in selected[1:]:
        if gene in gene_names_all and gene in vocab and gene not in seen:
            selected_genes.append(gene)
            seen.add(gene)
    for idx in np.argsort(-gene_sums).tolist():
        gene = gene_names_all[idx]
        if gene in seen or gene not in vocab:
            continue
        selected_genes.append(gene)
        seen.add(gene)
        if len(selected_genes) >= max_genes:
            break
    gene_index = [gene_names_all.index(g) for g in selected_genes]
    expression = expr_all[:, gene_index].astype(np.float32)
    return {
        "subset": subset[:, selected_genes].copy(),
        "expression": expression,
        "expression_layer": expression_layer,
        "selected_perturbations": selected,
        "selected_genes": selected_genes,
        "cells_per_perturbation": {
            p: int((subset.obs["perturbation"].astype(str) == p).sum()) for p in selected
        },
    }


def _build_tensors(expression: np.ndarray, genes: list[str], perturbations: list[str], vocab: GeneVocab, args: dict[str, Any]) -> dict[str, torch.Tensor]:
    gene_ids = np.array(vocab(genes), dtype=int)
    pad_value = float(args.get("pad_value", -2))
    tokenized = tokenize_and_pad_batch(
        expression,
        gene_ids,
        max_len=len(genes) + 1,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    src_key_padding_mask = tokenized["genes"].eq(vocab["<pad>"])
    pert_flags = torch.zeros_like(tokenized["genes"], dtype=torch.long)
    pert_flags[src_key_padding_mask] = 2
    for row_index, perturbation in enumerate(perturbations):
        if perturbation in vocab:
            pert_flags[row_index, tokenized["genes"][row_index] == vocab[perturbation]] = 1
    return {
        "gene_ids": tokenized["genes"].long(),
        "target_values": tokenized["values"].float(),
        "src_key_padding_mask": src_key_padding_mask.bool(),
        "pert_flags": pert_flags.long(),
    }


def _build_model(vocab: GeneVocab, args: dict[str, Any], state: dict[str, torch.Tensor], selected_perturbations: list[str], device: torch.device) -> tuple[TransformerGenerator, dict[str, Any]]:
    kwargs = dict(
        ntoken=len(vocab),
        d_model=int(args["embsize"]),
        nhead=int(args["nheads"]),
        d_hid=int(args["d_hid"]),
        nlayers=int(args["nlayers"]),
        nlayers_cls=int(args.get("n_layers_cls", 1)),
        n_cls=max(2, len(selected_perturbations)),
        vocab=vocab,
        dropout=float(args.get("dropout", 0.2)),
        pad_token=str(args.get("pad_token", "<pad>")),
        pad_value=float(args.get("pad_value", -2)),
        pert_pad_id=2,
        do_mvc=False,
        n_input_bins=0,
        explicit_zero_prob=bool(args.get("explicit_zero_prob", False)),
        use_fast_transformer=False,
        pre_norm=bool(args.get("pre_norm", False)),
    )
    model = TransformerGenerator(**kwargs).to(device)
    model_state = model.state_dict()
    matched_keys = [
        key
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    ]
    load_pretrained(model, state, verbose=False)
    return model, {
        "matched_key_count": int(len(matched_keys)),
        "total_model_key_count": int(len(model_state)),
        "arch": {
            "embsize": int(args["embsize"]),
            "nheads": int(args["nheads"]),
            "d_hid": int(args["d_hid"]),
            "nlayers": int(args["nlayers"]),
            "n_layers_cls": int(args.get("n_layers_cls", 1)),
        },
    }


def _forward_prediction_records(
    model: TransformerGenerator,
    tensors: dict[str, torch.Tensor],
    vocab: GeneVocab,
    genes: list[str],
    perturbations: list[str],
    device: torch.device,
    mask_value: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    batch = {k: v.to(device) for k, v in tensors.items()}
    valid = ~batch["src_key_padding_mask"]
    if "<cls>" in vocab:
        valid = valid & batch["gene_ids"].ne(vocab["<cls>"])
    full_mask_values = batch["target_values"].clone().masked_fill(valid, mask_value)
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

    gene_hash = _hash_gene_order(genes)
    gene_panel_id = f"scgpt::replogle_k562essential::forward_smoke::n_genes_{len(genes)}"
    pred_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for perturbation in sorted(set(pert_array.tolist())):
        if perturbation == "control":
            continue
        mask = pert_array == perturbation
        pred_delta = pred_matrix[mask].mean(axis=0) - control_pred
        true_delta = target_matrix[mask].mean(axis=0) - control_true
        rmse = float(np.sqrt(np.mean((pred_delta - true_delta) ** 2)))
        cosine = _cosine_error(pred_delta, true_delta)
        pearson = _safe_corr(pred_delta, true_delta, "pearson")
        spearman = _safe_corr(pred_delta, true_delta, "spearman")
        record_id = f"scGPT::ReplogleK562Essential::forward_smoke::{perturbation}"
        pred_key = record_id + "::pred"
        true_key = record_id + "::true"
        pred_arrays[pred_key] = pred_delta.astype(np.float32)
        true_arrays[true_key] = true_delta.astype(np.float32)
        records.append(
            {
                "schema_version": "safeconf_prediction_record_v1",
                "record_id": record_id,
                "task_id": perturbation,
                "task_key": f"ReplogleK562Essential::{perturbation}",
                "dataset_name": "ReplogleK562Essential_scGPT_smoke",
                "dataset_group": "scgpt_replogle_crispr_smoke",
                "fold_id": 0,
                "split": "test",
                "context": "K562_forward_smoke",
                "perturbation": perturbation,
                "predictor_name": "scGPT_whole_human_forward_smoke",
                "run_type": "smoke",
                "gene_panel_id": gene_panel_id,
                "gene_order_hash": gene_hash,
                "effect_definition": "mean_diff",
                "normalization_id": "scgpt_fullmask_expression_delta_v1",
                "error_normalization": "raw_rmse",
                "predicted_effect_key": pred_key,
                "true_effect_key": true_key,
                "true_error_rmse": rmse,
                "true_error_cosine": cosine,
                "n_cells": int(mask.sum()),
            }
        )
        metrics.append(
            {
                "perturbation": perturbation,
                "n_cells": int(mask.sum()),
                "delta_rmse": rmse,
                "delta_cosine_error": cosine,
                "delta_pearson": pearson,
                "delta_spearman": spearman,
                "pred_delta_l2": float(np.linalg.norm(pred_delta)),
                "true_delta_l2": float(np.linalg.norm(true_delta)),
            }
        )
    forward_summary = {
        "cell_count": int(pred_matrix.shape[0]),
        "gene_count": int(pred_matrix.shape[1]),
        "prediction_matrix_mean": float(pred_matrix.mean()),
        "prediction_matrix_std": float(pred_matrix.std()),
        "target_matrix_mean": float(target_matrix.mean()),
        "target_matrix_std": float(target_matrix.std()),
    }
    return (
        pd.DataFrame(records),
        pred_arrays,
        true_arrays,
        pd.DataFrame(metrics),
        forward_summary,
    )


def _write_report(status: dict[str, Any], metrics: pd.DataFrame, import_checks: pd.DataFrame, asset_checks: pd.DataFrame) -> None:
    report_md = OUT_DIR / "reports/E27_SCGPT_FORWARD_PREDICTION_RECORD_SMOKE_REPORT.md"
    report_html = OUT_DIR / "reports/E27_SCGPT_FORWARD_PREDICTION_RECORD_SMOKE.html"
    md = f"""# E27 scGPT forward PredictionRecord smoke

生成时间：{_now()}

## 结论

E27 修复了“scGPT 资产存在但默认 import 失败”的定位问题：默认 conda env 的 `scgpt.pth` 指向旧路径；使用归档源码路径后，scGPT 可以 import，并且 whole-human checkpoint 可以完成 forward-only smoke。

- PredictionRecords: {status['n_prediction_records']}
- strict issue_count: {status['strict_issue_count']}
- selected perturbations: {', '.join(status['selected_perturbations'])}
- selected genes: {status['selected_gene_count']}
- checkpoint matched keys: {status['matched_key_count']} / {status['total_model_key_count']}

这只是 scGPT 第二模型 adapter 的 smoke。它不代表正式 scGPT 性能，也不替代和 GEARS 的统一 benchmark 对齐。
"""
    report_md.write_text(md, encoding="utf-8")
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>E27 scGPT forward smoke</title>
  <style>
    body{{margin:0;background:#fff;color:#1f2933;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif;}}
    main{{max-width:1120px;margin:0 auto;padding:42px 28px 72px;}}
    h1{{font-size:30px;margin:0 0 10px;}}
    h2{{font-size:21px;margin:34px 0 12px;border-top:1px solid #e5e7eb;padding-top:24px;}}
    p{{line-height:1.75;font-size:16px;}}
    .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0;}}
    .card{{border:1px solid #e5e7eb;border-radius:14px;padding:16px;background:#fafafa;}}
    .k{{font-size:26px;font-weight:760;color:#111827;}}
    .l{{font-size:13px;color:#66788a;margin-top:4px;}}
    table{{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0 22px;}}
    th,td{{border-bottom:1px solid #e5e7eb;text-align:left;padding:9px 10px;vertical-align:top;}}
    th{{background:#f7f7f7;}}
    .note{{border-left:4px solid #4677C8;background:#f8fbff;padding:12px 16px;border-radius:8px;}}
  </style>
</head>
<body>
<main>
  <h1>E27 scGPT forward PredictionRecord smoke</h1>
  <p class="note">这是 scGPT 第二模型 adapter 的严格合同烟测，不是正式性能结论。</p>
  <div class="cards">
    <div class="card"><div class="k">{status['n_prediction_records']}</div><div class="l">PredictionRecords</div></div>
    <div class="card"><div class="k">{status['strict_issue_count']}</div><div class="l">strict issues</div></div>
    <div class="card"><div class="k">{status['selected_gene_count']}</div><div class="l">genes</div></div>
    <div class="card"><div class="k">{status['matched_key_count']}</div><div class="l">matched checkpoint keys</div></div>
  </div>
  <h2>Import checks</h2>{import_checks.to_html(index=False, escape=False)}
  <h2>Asset checks</h2>{asset_checks.to_html(index=False, escape=False)}
  <h2>Perturbation metrics</h2>{metrics.to_html(index=False, escape=False)}
</main>
</body>
</html>
"""
    report_html.write_text(page, encoding="utf-8")


def _write_readme(status: dict[str, Any]) -> None:
    readme = OUT_DIR / "README_先看这个.md"
    readme.write_text(
        f"""# E27 scGPT forward PredictionRecord smoke

先看结论：E27 已经让 scGPT 从“资产存在但默认 import 失败”推进到“可用归档源码 + whole-human checkpoint 生成 strict PredictionRecord smoke”。

- PredictionRecords：{status['n_prediction_records']}
- strict issue：{status['strict_issue_count']}
- selected perturbations：{', '.join(status['selected_perturbations'])}
- selected genes：{status['selected_gene_count']}

边界：这是 forward-only smoke，不是正式 scGPT 性能实验。下一步要把 scGPT 与 GEARS 放到同一任务/同一 gene panel 上做 adapter 对齐。
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for rel in ["tables", "arrays", "reports"]:
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)

    set_seed(5227)
    import_checks = pd.DataFrame(
        [
            {
                "check": "default_scgpt_pth",
                "status": "stale_path",
                "detail": "/home/yyf/.conda/envs/scgpt_env/lib/python3.9/site-packages/scgpt.pth points to /home/yyf/codex_scgpt_attnres_workspace/repo",
            },
            {"check": "archived_scgpt_repo_exists", "status": SCGPT_REPO.exists(), "detail": _rel(SCGPT_REPO)},
            {"check": "python_import_scgpt_with_repo_path", "status": True, "detail": "sys.path insert archived repo"},
        ]
    )
    checkpoint = _load_checkpoint()
    subset = _select_subset(checkpoint["vocab"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensors = _build_tensors(
        expression=subset["expression"],
        genes=subset["selected_genes"],
        perturbations=subset["subset"].obs["perturbation"].astype(str).tolist(),
        vocab=checkpoint["vocab"],
        args=checkpoint["args"],
    )
    model, model_info = _build_model(
        checkpoint["vocab"],
        checkpoint["args"],
        checkpoint["state"],
        subset["selected_perturbations"],
        device,
    )
    records, pred_arrays, true_arrays, metrics, forward_summary = _forward_prediction_records(
        model,
        tensors,
        checkpoint["vocab"],
        subset["selected_genes"],
        subset["subset"].obs["perturbation"].astype(str).tolist(),
        device,
        mask_value=float(checkpoint["args"].get("mask_value", -1)),
    )
    records.to_csv(OUT_DIR / "tables/PREDICTION_RECORDS.csv", index=False)
    records.to_csv(OUT_DIR / "tables/SCGPT_FORWARD_SMOKE_PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(OUT_DIR / "arrays/predicted_effects.npz", **pred_arrays)
    np.savez_compressed(OUT_DIR / "arrays/true_effects.npz", **true_arrays)
    metrics.to_csv(OUT_DIR / "tables/SCGPT_FORWARD_SMOKE_PERTURBATION_METRICS.csv", index=False)
    pd.DataFrame({"gene": subset["selected_genes"]}).to_csv(
        OUT_DIR / "tables/SCGPT_FORWARD_SMOKE_SELECTED_GENES.csv", index=False
    )
    asset_checks = pd.DataFrame(
        [
            {"asset": "scgpt_repo", "exists": SCGPT_REPO.exists(), "path": _rel(SCGPT_REPO)},
            {"asset": "checkpoint_dir", "exists": SCGPT_CHECKPOINT.exists(), "path": _rel(SCGPT_CHECKPOINT)},
            {"asset": "adata", "exists": ADATA_PATH.exists(), "path": _rel(ADATA_PATH)},
            {"asset": "selected_gene_count", "exists": True, "path": str(len(subset["selected_genes"]))},
        ]
    )
    import_checks.to_csv(OUT_DIR / "tables/SCGPT_FORWARD_SMOKE_IMPORT_CHECKS.csv", index=False)
    asset_checks.to_csv(OUT_DIR / "tables/SCGPT_FORWARD_SMOKE_ASSET_CHECKS.csv", index=False)
    issues = validate_prediction_record_artifacts(OUT_DIR, records=records, strict=True)
    validation = pd.DataFrame(
        [{"scope": "e27_scgpt_forward_smoke", "strict": True, "issue_count": len(issues), "issues": "; ".join(issues)}]
    )
    validation.to_csv(OUT_DIR / "tables/SCGPT_FORWARD_SMOKE_VALIDATION.csv", index=False)
    status = {
        "status": "ok" if not issues else "has_issues",
        "generated_at": _now(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "out": os.path.relpath(OUT_DIR, PROJECT_ROOT),
        "device": str(device),
        "n_prediction_records": int(len(records)),
        "strict_issue_count": int(len(issues)),
        "strict_issues": issues,
        "selected_perturbations": subset["selected_perturbations"],
        "selected_gene_count": int(len(subset["selected_genes"])),
        "cells_per_perturbation": subset["cells_per_perturbation"],
        "matched_key_count": model_info["matched_key_count"],
        "total_model_key_count": model_info["total_model_key_count"],
        "forward_summary": forward_summary,
        "source_repo": _rel(SCGPT_REPO),
        "checkpoint_dir": _rel(SCGPT_CHECKPOINT),
        "adata_path": _rel(ADATA_PATH),
    }
    (OUT_DIR / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(status, metrics, import_checks, asset_checks)
    _write_readme(status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
