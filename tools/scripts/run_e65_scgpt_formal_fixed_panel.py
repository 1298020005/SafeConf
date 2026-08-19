#!/usr/bin/env python3
"""E65: a formal scGPT adapter on the frozen E60 Adamson task panel.

Why this experiment exists
--------------------------
E27--E29 used a whole-human scGPT checkpoint for forward-only contract
smokes.  That is not a perturbation predictor evaluation: it never learned
from the Adamson perturbation training split.  E65 follows the official scGPT
perturbation tutorial's training protocol (GEARS ``PertData`` graphs,
``TransformerGenerator``, MSE fine tuning, validation-model selection), but
fixes the test tasks to E60's 24 held-out genes.

The output is deliberately a *reduced-panel* formal comparison, not a claim
that 512 genes are a full-transcriptome benchmark.  The panel contains every
Adamson perturbation-label gene (the frozen 24 tests come first) plus
training-domain high-expression genes.  It is created before model fitting,
without test expression, test effects, or test errors.  E60 GEARS effects are
sliced to exactly that ordering before any
GEARS--scGPT disagreement is calculated.

Usage
-----
  python tools/scripts/run_e65_scgpt_formal_fixed_panel.py --mode prepare
  python tools/scripts/run_e65_scgpt_formal_fixed_panel.py --mode preflight
  python tools/scripts/run_e65_scgpt_formal_fixed_panel.py --mode full

``preflight`` creates/reuses data, builds the pretrained model and verifies a
real GPU forward/backward batch without producing a performance conclusion.
``full`` fine-tunes then writes strict PredictionRecords and an audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code" / "20260426_154505_perturb_transport_final_push"
SCGPT_REPO = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/repo"
)
SCGPT_CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
GEARS_AUDIT = ROOT / "docs" / "实验结果" / "E60_gears_fixed_panel_formal_20260711"
OUT = ROOT / "docs" / "实验结果" / "E65_scgpt_formal_fixed_panel_20260711"
TABLES, ARRAYS, REPORTS, FIGURES, RAW = (
    OUT / "tables",
    OUT / "arrays",
    OUT / "reports",
    OUT / "figures",
    OUT / "raw_scgpt",
)
SOURCE_H5AD = Path("/home/yyf/data/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad")
DATA_ROOT = Path("/home/yyf/data/scgpt_formal_fixed_panel_20260711")
PROCESSED_DIR = DATA_ROOT / "adamson_e65_fixed512"
PROCESSED_H5AD = PROCESSED_DIR / "perturb_processed.h5ad"
MANIFEST = GEARS_AUDIT / "tables" / "E60_FIXED_TEST_PERTURBATIONS.csv"
N_GENES = 512
PANEL_SEED = 20260765
TRAIN_SEED = 20260765

sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(SCGPT_REPO))

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from scipy.stats import rankdata
from torch import nn
from torch_geometric.loader import DataLoader

from gears import PertData
from safetrans_confidence.data.records import validate_prediction_record_artifacts
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab_id


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def ensure_dirs() -> None:
    for directory in (OUT, TABLES, ARRAYS, REPORTS, FIGURES, RAW):
        directory.mkdir(parents=True, exist_ok=True)


def hash_order(genes: list[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()


def dense_mean(x: Any) -> np.ndarray:
    value = x.toarray() if issparse(x) else x
    return np.asarray(value, dtype=np.float32).mean(axis=0)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def cosine_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a, b) / denom)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    aa = rankdata(np.asarray(a, dtype=float), method="average")
    bb = rankdata(np.asarray(b, dtype=float), method="average")
    aa -= aa.mean(); bb -= bb.mean()
    denominator = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    return float(np.dot(aa, bb) / denominator) if denominator > 1e-12 else float("nan")


def bootstrap_ci(score: np.ndarray, error: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        index = rng.integers(0, len(score), len(score))
        value = spearman(score[index], error[index])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def top20_enrichment(score: np.ndarray, error: np.ndarray) -> tuple[int, float]:
    k = max(1, int(math.ceil(0.2 * len(score))))
    selected = error[np.argsort(-score, kind="stable")[:k]]
    return k, float(selected.mean() / error.mean()) if error.mean() > 0 else float("nan")


def fixed_split(test_conditions: list[str], all_conditions: list[str]) -> dict[str, list[str]]:
    """Match E60's deterministic fixed-test override exactly."""
    test = sorted(set(test_conditions))
    all_nonctrl = sorted(c for c in set(all_conditions) if c != "ctrl")
    missing = sorted(set(test) - set(all_nonctrl))
    if missing:
        raise ValueError(f"fixed test conditions absent from source: {missing}")
    train_pool = [c for c in all_nonctrl if c not in set(test)]
    n_val = max(1, min(len(train_pool) // 10 or 1, len(train_pool) - 1))
    val = train_pool[:n_val]
    train = train_pool[n_val:]
    return {"train": ["ctrl"] + train, "val": val, "test": test}


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    required = {"task_gene", "condition", "selection_uses_effect_or_error"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"E60 manifest missing columns: {sorted(missing)}")
    if bool(manifest["selection_uses_effect_or_error"].astype(bool).any()):
        raise ValueError("E60 manifest records an effect/error-dependent selection; refusing reuse")
    return manifest.sort_values("condition").reset_index(drop=True)


def prepare_panel(force_rebuild: bool = False) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    """Select a deterministic 512-gene panel from only the training domain."""
    ensure_dirs()
    panel_path = TABLES / "E65_GENE_PANEL.csv"
    split_path = TABLES / "E65_FIXED_SPLIT.csv"
    status_path = OUT / "PREPARE_STATUS.json"
    if panel_path.exists() and split_path.exists() and not force_rebuild:
        panel = pd.read_csv(panel_path)
        split_df = pd.read_csv(split_path)
        genes = panel["gene"].astype(str).tolist()
        split = {name: sub["condition"].astype(str).tolist() for name, sub in split_df.groupby("split", sort=False)}
        meta = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if len(genes) != N_GENES:
            raise ValueError(f"frozen E65 panel has {len(genes)} genes, expected {N_GENES}")
        return genes, split, meta
    if not SOURCE_H5AD.exists():
        raise FileNotFoundError(SOURCE_H5AD)
    manifest = load_manifest()
    adata = sc.read_h5ad(SOURCE_H5AD)
    try:
        all_conditions = adata.obs["condition"].astype(str).tolist()
        split = fixed_split(manifest["condition"].astype(str).tolist(), all_conditions)
        genes_all = adata.var_names.astype(str).tolist()
        vocab = GeneVocab.from_file(SCGPT_CHECKPOINT / "vocab.json")
        vocab.set_default_token("<pad>")
        test_genes = manifest["task_gene"].astype(str).tolist()
        # GEARS' graph construction needs the perturbed gene as an input
        # position for every condition, including training and validation
        # conditions.  Therefore all observed Adamson perturbation labels are
        # required input genes.  This reads condition names only; it does not
        # read test-cell expression, effects, errors, or metric values.
        condition_genes = sorted(
            {
                token
                for condition in set(all_conditions)
                for token in str(condition).split("+")
                if token and token != "ctrl"
            }
        )
        missing = [gene for gene in condition_genes if gene not in genes_all or gene not in vocab]
        if missing:
            raise ValueError(f"Adamson perturbation genes cannot enter scGPT panel: {missing}")
        train_mask = adata.obs["condition"].astype(str).isin(split["train"]).to_numpy()
        if not train_mask.any():
            raise ValueError("empty training selection")
        sums = np.asarray(adata.X[train_mask].sum(axis=0)).ravel()
        ranked_indices = np.argsort(-sums, kind="stable")
        selected = list(test_genes)
        seen = set(selected)
        for gene in condition_genes:
            if gene not in seen:
                selected.append(gene)
                seen.add(gene)
        for index in ranked_indices:
            gene = genes_all[int(index)]
            if gene in seen or gene not in vocab:
                continue
            selected.append(gene)
            seen.add(gene)
            if len(selected) == N_GENES:
                break
        if len(selected) != N_GENES:
            raise ValueError(f"only {len(selected)} vocabulary-compatible panel genes available")
        # The forced perturbation genes remain first; all other positions use
        # only training-cell expression totals and a stable tie rule.
        panel = pd.DataFrame(
            {
                "panel_index": np.arange(len(selected)),
                "gene": selected,
                "selection_role": ["fixed_test_perturbation_gene"] * len(test_genes)
                + ["training_or_validation_perturbation_gene"] * (len(condition_genes) - len(test_genes))
                + ["highest_train_expression"] * (len(selected) - len(condition_genes)),
                "train_expression_sum": [float(sums[genes_all.index(gene)]) for gene in selected],
                "in_scgpt_vocab": True,
                "uses_test_expression_effect_or_error": False,
            }
        )
        panel.to_csv(panel_path, index=False)
        split_rows = [{"split": name, "condition": condition} for name, items in split.items() for condition in items]
        pd.DataFrame(split_rows).to_csv(split_path, index=False)
        meta = {
            "experiment": "E65_scGPT_formal_fixed_panel",
            "prepared_at": now(),
            "git_head_before_prepare": git_head(),
            "source_h5ad": str(SOURCE_H5AD),
            "source_shape": [int(adata.n_obs), int(adata.n_vars)],
            "n_panel_genes": len(selected),
            "panel_gene_order_hash": hash_order(selected),
            "panel_selection": "all Adamson perturbation-label genes (24 frozen tests first), then remaining positions ranked by total expression over E60-matched training conditions only",
            "test_truth_used_for_panel": False,
            "test_conditions": split["test"],
            "n_train_conditions": len(split["train"]),
            "n_val_conditions": len(split["val"]),
            "n_test_conditions": len(split["test"]),
            "source_urls": {
                "scgpt_official_perturbation_tutorial": "https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Perturbation.ipynb",
                "gears_repository": "https://github.com/snap-stanford/GEARS",
            },
        }
        status_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return selected, split, meta
    finally:
        del adata


def prepare_pertdata(genes: list[str], force_rebuild: bool = False) -> PertData:
    """Create a GEARS-compatible 512-gene graph dataset, once and resumably."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if force_rebuild and PROCESSED_DIR.exists():
        raise RuntimeError("Refusing to delete existing processed E65 data automatically. Choose a new directory instead.")
    pert_data = PertData(str(DATA_ROOT))
    if PROCESSED_H5AD.exists():
        pert_data.load(data_path=str(PROCESSED_DIR))
        return pert_data
    source = sc.read_h5ad(SOURCE_H5AD)
    try:
        source = source[:, genes].copy()
        source.var["gene_name"] = source.var_names.astype(str)
        if "cell_type" not in source.obs:
            source.obs["cell_type"] = "Adamson"
        pert_data.new_data_process("adamson_e65_fixed512", adata=source)
    finally:
        del source
    return pert_data


def apply_fixed_split(pert_data: PertData, split: dict[str, list[str]]) -> None:
    available = set(pert_data.adata.obs["condition"].astype(str).unique())
    requested = set(split["test"])
    if not requested.issubset(available):
        raise ValueError(f"processed dataset lost test conditions: {sorted(requested - available)}")
    pert_data.split = "single"
    pert_data.seed = TRAIN_SEED
    pert_data.set2conditions = split
    split_map = {condition: name for name, conditions in split.items() for condition in conditions}
    pert_data.adata.obs["split"] = pert_data.adata.obs["condition"].astype(str).map(split_map)
    if pert_data.adata.obs["split"].isna().any():
        missing = sorted(pert_data.adata.loc[pert_data.adata.obs["split"].isna(), "condition"].astype(str).unique())
        raise ValueError(f"conditions absent from fixed split: {missing}")


def subsample_training_graphs(pert_data: PertData, split: dict[str, list[str]], max_cells_per_condition: int) -> dict[str, Any]:
    """Bound fine-tuning cost with a condition-stratified, label-blind sample.

    Test graph lists are never shortened.  This changes only the number of
    source training examples; task identities and all held-out targets remain
    E60's frozen panel.
    """
    rng = np.random.default_rng(TRAIN_SEED)
    summary: dict[str, Any] = {"max_train_or_val_cells_per_condition": max_cells_per_condition, "conditions": {}}
    for name in ("train", "val"):
        for condition in split[name]:
            items = pert_data.dataset_processed[condition]
            original = len(items)
            if original > max_cells_per_condition:
                index = np.sort(rng.choice(original, size=max_cells_per_condition, replace=False))
                pert_data.dataset_processed[condition] = [items[int(i)] for i in index]
            summary["conditions"][condition] = {"split": name, "before": original, "after": len(pert_data.dataset_processed[condition])}
    for condition in split["test"]:
        summary["conditions"][condition] = {"split": "test", "before": len(pert_data.dataset_processed[condition]), "after": len(pert_data.dataset_processed[condition])}
    return summary


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(device: torch.device) -> tuple[TransformerGenerator, np.ndarray, dict[str, Any]]:
    args = json.loads((SCGPT_CHECKPOINT / "args.json").read_text(encoding="utf-8"))
    vocab = GeneVocab.from_file(SCGPT_CHECKPOINT / "vocab.json")
    if "<pad>" not in vocab:
        vocab.append_token("<pad>")
    vocab.set_default_token("<pad>")
    model = TransformerGenerator(
        ntoken=len(vocab),
        d_model=int(args["embsize"]),
        nhead=int(args["nheads"]),
        d_hid=int(args["d_hid"]),
        nlayers=int(args["nlayers"]),
        nlayers_cls=int(args.get("n_layers_cls", 1)),
        n_cls=1,
        vocab=vocab,
        dropout=0.0,
        pad_token="<pad>",
        pad_value=0.0,
        pert_pad_id=0,
        do_mvc=False,
        n_input_bins=0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=bool(args.get("pre_norm", False)),
    ).to(device)
    state = torch.load(SCGPT_CHECKPOINT / "best_model.pt", map_location="cpu")
    model_state = model.state_dict()
    # This is the same pretrained-module policy in the official perturbation
    # tutorial: transfer encoder/value_encoder/transformer_encoder and tune the
    # perturbation output layers on the benchmark's training split.
    prefixes = ("encoder", "value_encoder", "transformer_encoder")
    selected = {
        key: value
        for key, value in state.items()
        if any(key.startswith(prefix) for prefix in prefixes)
        and key in model_state
        and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model_state.update(selected)
    model.load_state_dict(model_state)
    return model, np.asarray([], dtype=np.int64), {
        "checkpoint": str(SCGPT_CHECKPOINT),
        "pretrained_prefixes": list(prefixes),
        "matched_pretrained_parameter_tensors": len(selected),
        "model_parameter_tensors": len(model_state),
        "architecture": {key: args.get(key) for key in ("embsize", "nheads", "d_hid", "nlayers", "n_layers_cls")},
        "vocab_size": len(vocab),
        "vocab": vocab,
    }


def make_gene_ids(genes: list[str], vocab: GeneVocab) -> np.ndarray:
    ids = np.asarray([vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=np.int64)
    if np.any(ids == vocab["<pad>"]):
        raise ValueError("E65 panel unexpectedly contains out-of-vocabulary genes")
    return ids


def model_forward(model: TransformerGenerator, batch_data: Any, gene_ids: np.ndarray, device: torch.device, amp: bool, do_sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Official tutorial's all-gene path, kept deterministic for evaluation."""
    batch_data.to(device)
    batch_size = len(batch_data.y)
    x: torch.Tensor = batch_data.x
    n_genes = len(gene_ids)
    values = x[:, 0].view(batch_size, n_genes)
    pert_flags = x[:, 1].long().view(batch_size, n_genes)
    raw_index = torch.arange(n_genes, device=device, dtype=torch.long)
    mapped = map_raw_id_to_vocab_id(raw_index, gene_ids).repeat(batch_size, 1)
    mask = torch.zeros_like(values, dtype=torch.bool, device=device)
    with torch.cuda.amp.autocast(enabled=amp):
        output = model(
            mapped,
            values,
            pert_flags,
            src_key_padding_mask=mask,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=do_sample,
        )["mlm_output"]
    return output, batch_data.y


def train_one_epoch(model: TransformerGenerator, loader: DataLoader, gene_ids: np.ndarray, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, device: torch.device, amp: bool) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        prediction, target = model_forward(model, batch, gene_ids, device, amp, do_sample=False)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        loss = masked_mse_loss(prediction, target, mask)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_mse(model: TransformerGenerator, loader: DataLoader, gene_ids: np.ndarray, device: torch.device, amp: bool) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            prediction, target = model_forward(model, batch, gene_ids, device, amp, do_sample=False)
            losses.append(float(torch.mean((prediction - target) ** 2).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def collect_test_predictions(model: TransformerGenerator, loader: DataLoader, gene_ids: np.ndarray, device: torch.device, amp: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    model.eval()
    pred: dict[str, list[np.ndarray]] = {}
    truth: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader:
            values, target = model_forward(model, batch, gene_ids, device, amp, do_sample=False)
            for condition, p, t in zip(batch.pert, values.detach().cpu().numpy(), target.detach().cpu().numpy()):
                pred.setdefault(str(condition), []).append(np.asarray(p, dtype=np.float32))
                truth.setdefault(str(condition), []).append(np.asarray(t, dtype=np.float32))
    mean_pred = {condition: np.mean(np.stack(items), axis=0).astype(np.float32) for condition, items in pred.items()}
    mean_truth = {condition: np.mean(np.stack(items), axis=0).astype(np.float32) for condition, items in truth.items()}
    sizes = {condition: len(items) for condition, items in pred.items()}
    return mean_pred, mean_truth, sizes


def e60_subset_vectors(genes: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the GEARS ensemble and shared truth in exactly E65's gene order."""
    records = pd.read_csv(GEARS_AUDIT / "tables" / "PREDICTION_RECORDS.csv")
    e60_genes = sc.read_h5ad(SOURCE_H5AD, backed="r").var_names.astype(str).tolist()
    indexes = [e60_genes.index(gene) for gene in genes]
    with np.load(GEARS_AUDIT / "arrays" / "gears_predicted_effects.npz") as pred_store, np.load(GEARS_AUDIT / "arrays" / "gears_true_effects.npz") as true_store:
        ensemble: dict[str, np.ndarray] = {}
        truth: dict[str, np.ndarray] = {}
        for condition, sub in records.groupby("perturbation", sort=True):
            predictions = [np.asarray(pred_store[key], dtype=np.float32)[indexes] for key in sub["predicted_effect_key"]]
            true_vectors = [np.asarray(true_store[key], dtype=np.float32)[indexes] for key in sub["true_effect_key"]]
            if any(np.max(np.abs(item - true_vectors[0])) > 1e-7 for item in true_vectors[1:]):
                raise ValueError(f"E60 truth differs across seeds for {condition}")
            ensemble[str(condition)] = np.mean(np.stack(predictions), axis=0).astype(np.float32)
            truth[str(condition)] = true_vectors[0].astype(np.float32)
    return ensemble, truth


def write_records(genes: list[str], scgpt_raw: dict[str, np.ndarray], scgpt_truth_raw: dict[str, np.ndarray], sizes: dict[str, int], model_info: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build one shared strict contract for E60 GEARS and newly tuned scGPT."""
    controls = sc.read_h5ad(PROCESSED_H5AD, backed="r")
    try:
        ctrl_mask = controls.obs["condition"].astype(str).eq("ctrl")
        ctrl_mean = dense_mean(controls[ctrl_mask].X)
    finally:
        controls.file.close()
    gears_pred, e60_truth = e60_subset_vectors(genes)
    manifest = load_manifest()
    gene_hash = hash_order(genes)
    gene_panel = f"adamson_e60_fixed24_shared512::{gene_hash.split(':', 1)[1][:12]}"
    pred_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for condition in manifest["condition"].astype(str):
        if condition not in scgpt_raw or condition not in scgpt_truth_raw:
            raise ValueError(f"scGPT test output missing fixed task {condition}")
        # The target effect used by both records is E60's original full-data
        # mean effect sliced to this panel.  ``scgpt_truth_raw`` is audited
        # below only to prove the new graph dataset agrees with that target.
        true_effect = e60_truth[condition]
        truth_check = scgpt_truth_raw[condition] - ctrl_mean
        if not np.allclose(true_effect, truth_check, rtol=1e-5, atol=1e-5):
            difference = float(np.max(np.abs(true_effect - truth_check)))
            raise ValueError(f"E65 true effect mismatch for {condition}; max_abs={difference}")
        scgpt_effect = scgpt_raw[condition] - ctrl_mean
        ge_effect = gears_pred[condition]
        true_key = f"E65::Adamson::{condition}::shared_true"
        true_arrays[true_key] = true_effect.astype(np.float32)
        per_task: dict[str, float] = {}
        # The two prediction mechanisms differ, but their exported vector has
        # the same physical definition: mean expression minus the same Adamson
        # control mean, on the same 512-gene ordering.  The model family is
        # represented by ``predictor_name``; keeping one normalization id is a
        # strict-contract requirement for an honest same-task comparison.
        for name, vector in [
            ("GEARS_3seed_ensemble_E60_subset512", ge_effect),
            ("scGPT_official_tutorial_finetuned_subset512", scgpt_effect),
        ]:
            record_id = f"E65::Adamson::{condition}::{name}"
            pred_key = record_id + "::pred"
            pred_arrays[pred_key] = vector.astype(np.float32)
            err = rmse(vector, true_effect)
            per_task[name] = err
            rows.append(
                {
                    "schema_version": "safeconf_prediction_record_v1",
                    "record_id": record_id,
                    "task_id": condition,
                    "task_key": f"Adamson_E60_fixed24::{condition}",
                    "dataset_name": "Adamson_E60_fixed24_shared512",
                    "dataset_group": "adamson_crispr_fixed_task_shared_predictor",
                    "fold_id": TRAIN_SEED,
                    "split": "test",
                    "context": "Adamson_fixed_unseen_gene_panel512",
                    "perturbation": condition,
                    "predictor_name": name,
                    "run_type": "formal",
                    "gene_panel_id": gene_panel,
                    "gene_order_hash": gene_hash,
                    "effect_definition": "mean_diff",
                    "normalization_id": "adamson_mean_expression_minus_ctrl_shared512_v1",
                    "error_normalization": "raw_rmse",
                    "predicted_effect_key": pred_key,
                    "true_effect_key": true_key,
                    "true_error_rmse": err,
                    "true_error_cosine": cosine_error(vector, true_effect),
                    "n_cells": int(sizes[condition]),
                }
            )
        task_rows.append(
            {
                "perturbation": condition,
                "n_cells": int(sizes[condition]),
                "gears_ensemble_rmse": per_task["GEARS_3seed_ensemble_E60_subset512"],
                "scgpt_finetuned_rmse": per_task["scGPT_official_tutorial_finetuned_subset512"],
                "task_mean_rmse": float(np.mean(list(per_task.values()))),
                "task_max_rmse": float(np.max(list(per_task.values()))),
                "risk_gears_scgpt_disagreement": rmse(ge_effect, scgpt_effect),
                "risk_gears_predicted_magnitude": float(np.linalg.norm(ge_effect)),
                "risk_scgpt_predicted_magnitude": float(np.linalg.norm(scgpt_effect)),
                "true_l2_diagnostic": float(np.linalg.norm(true_effect)),
                "uses_target_truth_in_deployable_risk": False,
            }
        )
    records = pd.DataFrame(rows)
    tasks = pd.DataFrame(task_rows)
    records.to_csv(TABLES / "PREDICTION_RECORDS.csv", index=False)
    tasks.to_csv(TABLES / "E65_TASK_RISK_TABLE.csv", index=False)
    np.savez_compressed(ARRAYS / "predicted_effects.npz", **pred_arrays)
    np.savez_compressed(ARRAYS / "true_effects.npz", **true_arrays)
    issues = validate_prediction_record_artifacts(OUT, records=records, strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(TABLES / "E65_STRICT_CONTRACT_ISSUES.csv", index=False)
    return records, tasks, issues


def write_figure(tasks: pd.DataFrame) -> None:
    width, height, left, top = 1080, 680, 108, 92
    x = tasks["risk_gears_scgpt_disagreement"].to_numpy(float)
    y = tasks["task_mean_rmse"].to_numpy(float)
    def coordinate(values: np.ndarray, low: float, high: float) -> tuple[np.ndarray, float, float]:
        minimum, maximum = float(values.min()), float(values.max())
        delta = max(maximum - minimum, 1e-8)
        minimum -= 0.08 * delta; maximum += 0.08 * delta
        return low + (values - minimum) / (maximum - minimum) * (high - low), minimum, maximum
    sx, _, _ = coordinate(x, left, width - 70)
    sy, _, _ = coordinate(y, height - 88, top)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#23313d}.t{font-size:27px;font-weight:700}.s{font-size:16px;fill:#5d6b76}.a{font-size:15px}.m{font-size:12px;fill:#64727d}</style>',
        '<text class="t" x="54" y="44">E65｜正式微调后的 GEARS–scGPT 分歧</text>',
        '<text class="s" x="54" y="72">Adamson 固定 24 个未见基因；横轴只由两个预测 effect 计算，纵轴只用于事后核验。</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-88}" stroke="#71808b"/>',
        f'<line x1="{left}" y1="{height-88}" x2="{width-70}" y2="{height-88}" stroke="#71808b"/>',
        f'<text class="a" x="{(left+width-70)/2:.1f}" y="{height-34}" text-anchor="middle">GEARS ensemble – scGPT disagreement (RMSE)</text>',
        f'<text class="a" transform="translate(30 {(top+height-88)/2:.1f}) rotate(-90)" text-anchor="middle">two-model mean error (RMSE)</text>',
    ]
    for xx, yy, condition in zip(sx, sy, tasks["perturbation"].astype(str)):
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.8" fill="#427b93" opacity=".86"><title>{condition}</title></circle>')
    parts.append('</svg>')
    (FIGURES / "F1_gears_scgpt_disagreement_vs_mean_error.svg").write_text("\n".join(parts), encoding="utf-8")


def write_report(records: pd.DataFrame, tasks: pd.DataFrame, issues: list[str], run: dict[str, Any], n_boot: int) -> None:
    # The primary task-level target was specified in this script before E65
    # produced predictions: mean RMSE across the two predictors.  The other
    # three targets are reported side-by-side so a positive result cannot hide
    # a predictor-specific failure.
    targets = [
        ("gears_ensemble_rmse", "GEARS ensemble RMSE", False),
        ("scgpt_finetuned_rmse", "fine-tuned scGPT RMSE", False),
        ("task_mean_rmse", "mean RMSE across GEARS and scGPT (pre-specified primary)", True),
        ("task_max_rmse", "max RMSE across GEARS and scGPT", False),
    ]
    scores = [
        ("risk_gears_scgpt_disagreement", True),
        ("risk_gears_predicted_magnitude", True),
        ("risk_scgpt_predicted_magnitude", True),
        ("true_l2_diagnostic", False),
    ]
    rows = []
    for target_column, target_label, primary in targets:
        err = tasks[target_column].to_numpy(float)
        for column, deployable in scores:
            values = tasks[column].to_numpy(float)
            low, high = bootstrap_ci(values, err, PANEL_SEED, n_boot) if deployable else (float("nan"), float("nan"))
            k, enrichment = top20_enrichment(values, err) if deployable else (0, float("nan"))
            rows.append(
                {
                    "score": column,
                    "deployable": deployable,
                    "target": target_column,
                    "target_label": target_label,
                    "is_pre_specified_primary_target": primary,
                    "n_tasks": len(tasks),
                    "spearman": spearman(values, err),
                    "bootstrap_rho_ci95_low": low,
                    "bootstrap_rho_ci95_high": high,
                    "top20_k": k,
                    "top20_error_enrichment": enrichment,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "E65_RISK_ERROR_SUMMARY.csv", index=False)
    write_figure(tasks)
    run.update(
        {
            "finished_at": now(),
            "n_records": len(records),
            "n_tasks": len(tasks),
            "strict_issue_count": len(issues),
            "strict_issues": issues,
            "target_truth_used_in_deployable_scores": False,
            "outputs": [
                "tables/PREDICTION_RECORDS.csv",
                "arrays/predicted_effects.npz",
                "arrays/true_effects.npz",
                "tables/E65_TASK_RISK_TABLE.csv",
                "tables/E65_RISK_ERROR_SUMMARY.csv",
                "reports/E65_REPORT.md",
            ],
        }
    )
    (OUT / "RUN_STATUS.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E65｜scGPT 正式微调：与 E60 同一批固定任务",
        "",
        "## 这次做了什么",
        "",
        "E65 不使用旧的 forward-only scGPT smoke。它按 scGPT 官方 perturbation tutorial 的 GEARS `PertData` + `TransformerGenerator` + MSE fine-tuning 训练路径，把 whole-human checkpoint 的 encoder/value encoder/transformer encoder 迁移到 Adamson 训练条件，再在 E60 冻结的 24 个未见基因上测试。",
        "",
        f"共享面板有 {N_GENES} 个基因：Adamson 的全部扰动标签基因都保留，24 个测试基因位于前部；剩余位置从训练条件的表达量排序得到。GEARS E60 三 seed ensemble 和 scGPT 的预测、真值全都投影到同一顺序。strict PredictionRecord issue_count = {len(issues)}。",
        "",
        "## 风险审计",
        "",
        "主目标在运行前固定为两模型的 `task_mean_rmse`。下面把 GEARS 自身、scGPT 自身、两者均值和两者最大误差一起列出；这些并列结果用于检查分歧到底在筛谁的错误。",
        "",
        "| score | 目标误差 | ρ | bootstrap 95% CI | top20 高误差富集 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        interval = "—" if not row.deployable else f"[{row.bootstrap_rho_ci95_low:.3f}, {row.bootstrap_rho_ci95_high:.3f}]"
        enrichment = "—" if not row.deployable else f"{row.top20_error_enrichment:.3f}"
        label = str(row.target_label)
        if bool(row.is_pre_specified_primary_target):
            label += " **(主目标)**"
        lines.append(f"| {row.score} | {label} | {row.spearman:.3f} | {interval} | {enrichment} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "这仍是一个固定 24-task、512-gene 的正式适配器实验。它可以回答同任务上的模型分歧是否有排序信息，不能代替全转录组、多数据集、多模型家族的总验证。尤其不能把“对 scGPT 或双模型平均误差有信号”偷换成“已可靠筛出 GEARS 自身错误”；两个目标必须分别阅读。真实 effect 从未进入可部署分数；`true_l2_diagnostic` 只用于检查上限。",
        "",
        "## 文件",
        "",
        "- 固定任务与切分：`tables/E65_FIXED_SPLIT.csv`",
        "- 基因面板：`tables/E65_GENE_PANEL.csv`",
        "- 严格记录：`tables/PREDICTION_RECORDS.csv`",
        "- 任务分数：`tables/E65_TASK_RISK_TABLE.csv`",
        "- 汇总：`tables/E65_RISK_ERROR_SUMMARY.csv`",
        "- 图：`figures/F1_gears_scgpt_disagreement_vs_mean_error.svg`",
    ]
    (REPORTS / "E65_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E65｜scGPT 正式微调与 E60 固定任务对齐\n\n"
        "先读 `reports/E65_REPORT.md`。\n\n"
        "本实验淘汰了旧的 scGPT forward-only smoke：scGPT 在 E60 同一批 24 个 held-out genes 上做了官方 tutorial 风格的 MSE fine-tuning，并和 GEARS ensemble 使用同一 512-gene order 与同一 true effect。\n",
        encoding="utf-8",
    )


def repack_existing(n_boot: int) -> dict[str, Any]:
    """Repair/export-only pass for metadata, never retraining or re-scoring.

    It exists because strict validation is meant to catch provenance errors
    before results are relied upon.  E65's two effect vectors already use the
    same control-subtracted definition; an early record writer attached a
    model-specific label to that shared normalization.  This function only
    makes the label truthful and re-runs the validator.
    """
    ensure_dirs()
    record_path = TABLES / "PREDICTION_RECORDS.csv"
    task_path = TABLES / "E65_TASK_RISK_TABLE.csv"
    if not record_path.exists() or not task_path.exists():
        raise FileNotFoundError("E65 repack needs an existing completed package")
    records = pd.read_csv(record_path)
    tasks = pd.read_csv(task_path)
    records["normalization_id"] = "adamson_mean_expression_minus_ctrl_shared512_v1"
    records.to_csv(record_path, index=False)
    issues = validate_prediction_record_artifacts(OUT, records=records, strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(TABLES / "E65_STRICT_CONTRACT_ISSUES.csv", index=False)
    prior = json.loads((OUT / "RUN_STATUS.json").read_text(encoding="utf-8")) if (OUT / "RUN_STATUS.json").exists() else {}
    prior.update({
        "repacked_at": now(),
        "repack_scope": "metadata-only; no model, task, prediction vector, truth vector, error, or ranking value changed",
        "normalization_id": "adamson_mean_expression_minus_ctrl_shared512_v1",
    })
    write_report(records, tasks, issues, prior, n_boot)
    return json.loads((OUT / "RUN_STATUS.json").read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    if args.mode == "repack":
        return repack_existing(args.n_boot)
    seed_everything(TRAIN_SEED)
    genes, split, prep_meta = prepare_panel(force_rebuild=args.rebuild_panel)
    pert_data = prepare_pertdata(genes, force_rebuild=args.rebuild_data)
    apply_fixed_split(pert_data, split)
    sampling = subsample_training_graphs(pert_data, split, args.max_train_cells_per_condition)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.eval_batch_size)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    model, _, model_info = load_model(device)
    vocab = model_info.pop("vocab")
    gene_ids = make_gene_ids(genes, vocab)
    run_status: dict[str, Any] = {
        "experiment": "E65_scGPT_formal_fixed_panel",
        "started_at": now(),
        "git_head_before_run": git_head(),
        "mode": args.mode,
        "device": str(device),
        "amp": amp,
        "n_genes": len(genes),
        "gene_order_hash": hash_order(genes),
        "split": {name: len(items) for name, items in split.items()},
        "sampling": sampling,
        "model": model_info,
        "scgpt_protocol": "official Tutorial_Perturbation structure; full fixed panel replaces official random max_seq_len subsampling; deterministic do_sample=False evaluation",
        "e60_manifest": str(MANIFEST.relative_to(ROOT)),
        "target_truth_used_to_train_or_rank": False,
    }
    # A preflight proves the actual architecture, data layout and backward pass
    # work together before a long run is allowed to start.
    if args.mode == "preflight":
        batch = next(iter(pert_data.dataloader["train_loader"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        model.train()
        prediction, target = model_forward(model, batch, gene_ids, device, amp, do_sample=False)
        loss = masked_mse_loss(prediction, target, torch.ones_like(prediction, dtype=torch.bool))
        optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.zero_grad(set_to_none=True)
        run_status.update({"preflight_status": "ok", "preflight_batch_size": len(batch.y), "preflight_loss": float(loss.detach().cpu()), "preflight_gradient_norm_before_clip": grad_norm, "finished_at": now()})
        (OUT / "PREFLIGHT_STATUS.json").write_text(json.dumps(run_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return run_status
    if args.mode == "prepare":
        run_status.update({"prepare_status": "ok", "finished_at": now()})
        (OUT / "PREPARE_RUN_STATUS.json").write_text(json.dumps(run_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return run_status
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_mse = train_one_epoch(model, pert_data.dataloader["train_loader"], gene_ids, optimizer, scaler, device, amp)
        val_mse = evaluate_mse(model, pert_data.dataloader["val_loader"], gene_ids, device, amp)
        scheduler.step()
        improved = math.isfinite(val_mse) and val_mse < best_val
        if improved:
            best_val = val_mse
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "learning_rate": optimizer.param_groups[0]["lr"], "improved": improved, "seconds": time.time() - started})
        pd.DataFrame(history).to_csv(TABLES / "E65_TRAINING_HISTORY.csv", index=False)
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if stale >= args.early_stop:
            break
    if best_state is None:
        raise RuntimeError("scGPT training did not produce a finite validation model")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "best_val_mse": best_val, "history": history, "model_info": model_info}, RAW / "best_model_state.pt")
    predicted, truth, sizes = collect_test_predictions(model, pert_data.dataloader["test_loader"], gene_ids, device, amp)
    records, tasks, issues = write_records(genes, predicted, truth, sizes, model_info)
    run_status.update({"best_val_mse": best_val, "epochs_completed": len(history), "early_stop": stale >= args.early_stop})
    write_report(records, tasks, issues, run_status, args.n_boot)
    return json.loads((OUT / "RUN_STATUS.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="E65 formal scGPT fixed-panel adapter")
    parser.add_argument("--mode", choices=("prepare", "preflight", "full", "repack"), default="preflight")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-train-cells-per-condition", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stop", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()
    try:
        status = run(args)
    except Exception as exc:
        ensure_dirs()
        status = {"experiment": "E65_scGPT_formal_fixed_panel", "status": "failed", "failed_at": now(), "mode": args.mode, "error": repr(exc), "traceback": traceback.format_exc()}
        (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        raise
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
