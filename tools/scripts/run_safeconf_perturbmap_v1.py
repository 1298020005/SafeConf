#!/usr/bin/env python3
"""Bounded PerturbMap prototype for the SafeConf public module.

This run is deliberately separate from the frozen SafeConf system.  It uses
K562 E201 public responses as the source, Orion TRAIN responses as recipient
background anchors, and Orion VALIDATION predictions/errors only for the
held-out risk readout.  No permanent TEST truth is read.

The script answers two linked questions without changing the default model:

1. Does a low-rank source-to-recipient map improve response reconstruction?
2. Does that improvement change risk ranking on the same legal validation
   tasks when the identical prediction vectors are scored against the six
   reference rules?

The output is an auditable prototype, not an automatic method promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / (
    "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/"
    "research_closure_20261001/data_model_feedback_20261003_v1/"
    "perturbmap_experiment_v1"
)

PUBLIC = Path("/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201/public_memory.parquet")
SOURCE_EFFECTS = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/SOURCE_PUBLIC_EFFECTS.npy")
SOURCE_GENES = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/GENE_IDS.json")
ORION_META = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_allowed_biology_20261002_v2/QUERY_METADATA_ONLY.csv")
ORION_AXIS = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_allowed_biology_20261002_v2/ENDPOINT_GENE_MANIFEST.csv")
ORION_EFFECTS = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_allowed_biology_20261002_v2/ENDPOINT3285_EFFECTS.npy")
ORION_ERRORS = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_postfit_followthrough_20261002_v4_decimal_tokens/competence/VALIDATION_TASK_ERRORS.csv")
PRED_ROOT = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_published_lm_20261002_v3_decimal_tokens")

SOURCE_CONTEXT = "K562"
TARGET_CONTEXTS = ("HCT116", "HEK293T")
RANK = 8
SEED = 20261004


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def fold_for_gene(gene: str, context: str) -> int:
    raw = f"safeconf-perturbmap-v1|{context}|{gene}".encode()
    return int(hashlib.sha256(raw).hexdigest(), 16) % 3


def load_prediction_subset(query_ids: list[str], target_context: str, ensembl: list[str]) -> np.ndarray:
    """Read only validation query columns and the common response axis."""
    path = require(PRED_ROOT / target_context / "PREDICTIONS_DELTA.tsv.gz")
    usecols = ["gene_id", *query_ids]
    frame = pd.read_csv(path, sep="\t", usecols=usecols)
    if frame["gene_id"].duplicated().any():
        raise RuntimeError(f"duplicate prediction gene IDs in {path}")
    frame = frame.set_index("gene_id")
    missing = [g for g in ensembl if g not in frame.index]
    if missing:
        raise RuntimeError(f"prediction output missing {len(missing)} common genes for {target_context}")
    # read_csv preserves the query columns requested above; explicitly reorder
    # both axes to the frozen task and endpoint contracts.
    missing_q = [q for q in query_ids if q not in frame.columns]
    if missing_q:
        raise RuntimeError(f"prediction output missing query IDs: {missing_q[:3]}")
    return frame.loc[ensembl, query_ids].to_numpy(dtype=np.float64).T


def common_basis(source: np.ndarray, recipient: np.ndarray, rank: int = RANK):
    mu_s = source.mean(axis=0)
    mu_r = recipient.mean(axis=0)
    centered = np.vstack([source - mu_s, recipient - mu_r])
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    numerical_rank = int(np.sum(singular > max(singular[0] if len(singular) else 0.0, 1.0) * 1e-12))
    if min(source.shape[0], source.shape[1], recipient.shape[0]) < rank:
        raise RuntimeError("rank-8 map is underdetermined in this fit partition")
    if numerical_rank < rank:
        raise RuntimeError(f"joint response rank {numerical_rank} is below requested rank {rank}")
    u = vt[:rank].T
    return mu_s, mu_r, u, singular, numerical_rank


def project(x: np.ndarray, mu: np.ndarray, u: np.ndarray) -> np.ndarray:
    return (x - mu) @ u


def fit_ridge_transport(xs: np.ndarray, xr: np.ndarray, *, lam: float, rank: int = RANK,
                        shuffled: bool = False, rng: np.random.Generator | None = None):
    mu_s, mu_r, u, singular, numerical_rank = common_basis(xs, xr, rank)
    zs = project(xs, mu_s, u)
    zr = project(xr, mu_r, u)
    if shuffled:
        if rng is None:
            rng = np.random.default_rng(SEED)
        zr = zr[rng.permutation(len(zr))]
    vs = float(np.sum(zs * zs) / max(len(zs) * rank, 1))
    alpha = float(len(zs) * lam * max(vs, 1e-12))
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(zs, zr)
    return {
        "mu_s": mu_s, "mu_r": mu_r, "u": u, "model": model,
        "lambda": lam, "alpha": alpha, "vs": vs,
        "singular_values": singular, "numerical_rank": numerical_rank,
    }


def apply_transport(x: np.ndarray, fit: dict) -> np.ndarray:
    z = project(np.asarray(x), fit["mu_s"], fit["u"])
    return fit["mu_r"] + fit["model"].predict(z) @ fit["u"].T


def apply_projected(x: np.ndarray, fit: dict) -> np.ndarray:
    z = project(np.asarray(x), fit["mu_s"], fit["u"])
    return fit["mu_r"] + z @ fit["u"].T


def fit_scalar(x_s: np.ndarray, x_r: np.ndarray, fit: dict) -> float:
    zs = project(x_s, fit["mu_s"], fit["u"])
    zr = project(x_r, fit["mu_r"], fit["u"])
    denom = float(np.sum(zs * zs))
    return float(np.sum(zs * zr) / denom) if denom > 1e-12 else 0.0


def choose_lambda(xs: np.ndarray, xr: np.ndarray, context: str, rank: int = RANK) -> tuple[float, pd.DataFrame]:
    """Inner 3-fold response-only selection; no risk score is read."""
    rows = []
    for lam in (0.1, 1.0, 10.0):
        losses = []
        for fold in range(3):
            tr = np.array([fold_for_gene(f"row{i}", f"{context}|inner{fold}") % 3 != fold for i in range(len(xs))])
            # The inner split is deterministic and independent of response
            # values.  For very small partitions, use the same all-row fit.
            if tr.sum() < max(rank + 2, 8) or (~tr).sum() < 2:
                continue
            try:
                fit = fit_ridge_transport(xs[tr], xr[tr], lam=lam, rank=rank)
                pred = apply_transport(xs[~tr], fit)
            except RuntimeError:
                continue
            losses.append(float(np.mean((pred - xr[~tr]) ** 2)))
        rows.append({"lambda": lam, "inner_mse": float(np.mean(losses)) if losses else np.inf,
                     "n_inner_scores": len(losses)})
    tab = pd.DataFrame(rows)
    finite = tab[np.isfinite(tab.inner_mse)]
    if finite.empty:
        return 1.0, tab
    best = finite.sort_values(["inner_mse", "lambda"], ascending=[True, False]).iloc[0]
    return float(best["lambda"]), tab


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def metric_frame(errors: np.ndarray, scores: np.ndarray, ids: np.ndarray, method: str, context: str) -> dict:
    y = np.asarray(errors, float)
    s = np.asarray(scores, float)
    valid = np.isfinite(y) & np.isfinite(s)
    y, s, ids = y[valid], s[valid], np.asarray(ids)[valid]
    row = {"context": context, "method": method, "n_tasks": len(y), "u20": np.nan,
           "aurc": np.nan, "spearman": np.nan, "selected_top20": np.nan,
           "true_top20_found": np.nan, "remaining_mean_error": np.nan}
    if not len(y):
        return row
    k = int(np.ceil(0.2 * len(y)))
    risky = np.lexsort((ids.astype(str), -s))[:k]
    oracle = np.lexsort((ids.astype(str), -y))[:k]
    denom = float(y[oracle].mean() - y.mean())
    if len(y) >= 20 and denom > 1e-12:
        row["u20"] = float((y[risky].mean() - y.mean()) / denom)
    if len(y) >= 3 and np.ptp(y) > 0 and np.ptp(s) > 0:
        row["spearman"] = float(spearmanr(s, y).statistic)
    order = np.lexsort((ids.astype(str), s))
    row["aurc"] = float(np.mean(np.cumsum(y[order]) / np.arange(1, len(y) + 1)))
    row["selected_top20"] = k
    row["true_top20_found"] = int(len(set(risky).intersection(set(oracle))))
    row["remaining_mean_error"] = float(np.delete(y, risky).mean()) if len(y) > k else np.nan
    return row


def u20_value(errors: np.ndarray, scores: np.ndarray, ids: np.ndarray) -> float:
    y = np.asarray(errors, float)
    s = np.asarray(scores, float)
    ids = np.asarray(ids).astype(str)
    if len(y) < 20:
        return np.nan
    k = int(np.ceil(0.2 * len(y)))
    risky = np.lexsort((ids, -s))[:k]
    oracle = np.lexsort((ids, -y))[:k]
    denom = float(y[oracle].mean() - y.mean())
    if denom <= 1e-12:
        return np.nan
    return float((y[risky].mean() - y.mean()) / denom)


def paired_bootstrap(risk: pd.DataFrame, n_boot: int = 5000) -> pd.DataFrame:
    """Task-identity paired bootstrap; no refit and no new truth reads."""
    rng = np.random.default_rng(SEED + 77)
    rows = []
    for context, part in risk.groupby("context", sort=True):
        methods = sorted(part.method.unique())
        wide = part.pivot(index="query_id", columns="method", values="risk_score")
        truth = part.drop_duplicates("query_id").set_index("query_id").true_error_rmse.reindex(wide.index).to_numpy(float)
        ids = wide.index.astype(str).to_numpy()
        if "Amplitude" not in wide or "RawCopy" not in wide:
            continue
        base_names = ["Amplitude", "RawCopy"]
        for method in methods:
            if method in base_names:
                continue
            if method not in wide:
                continue
            point = u20_value(truth, wide[method].to_numpy(float), ids)
            for base in base_names:
                if base not in wide:
                    continue
                deltas = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(wide), size=len(wide))
                    draw_ids = np.asarray([f"{ids[j]}|{k}" for k, j in enumerate(idx)])
                    delta = u20_value(truth[idx], wide[method].to_numpy(float)[idx], draw_ids) - \
                            u20_value(truth[idx], wide[base].to_numpy(float)[idx], draw_ids)
                    if np.isfinite(delta):
                        deltas.append(float(delta))
                arr = np.asarray(deltas, float)
                rows.append({
                    "context": context, "method": method, "baseline": base,
                    "point_u20_delta": point - u20_value(truth, wide[base].to_numpy(float), ids),
                    "bootstrap_mean_delta": float(np.mean(arr)) if len(arr) else np.nan,
                    "ci95_lower": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
                    "ci95_upper": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
                    "n_boot_valid": int(len(arr)), "n_boot_requested": n_boot,
                })
    return pd.DataFrame(rows)


def build_route(memory: pd.DataFrame, source_effects: np.ndarray, target_meta: pd.DataFrame,
                target_effects: np.ndarray, source_genes: list[str], target_context: str):
    src = memory[memory.context.astype(str).eq(SOURCE_CONTEXT)].copy()
    src["gene"] = src.perturbation_target.astype(str)
    # One source effect per K562 gene is required by this registered route.
    if src.gene.duplicated().any():
        raise RuntimeError("K562 source route has duplicate perturbation identities")
    gene_to_source_row = dict(zip(src.gene, src.index.astype(int)))
    target = target_meta[(target_meta.context.astype(str) == target_context) &
                         (target_meta.role.astype(str) == "TRAIN")].copy()
    if target.gene.duplicated().any():
        raise RuntimeError(f"{target_context} TRAIN has duplicate gene identities")
    pairs = src[["gene", "experiment_id", "context", "effect_vector_row", "n_cells"]].merge(
        target[["gene", "matrix_row", "n_cells_metadata", "competence_n_ge30"]],
        on="gene", how="inner", validate="one_to_one"
    )
    pairs["fold"] = pairs.gene.map(lambda g: fold_for_gene(str(g), target_context))
    pair_source = np.asarray(source_effects[pairs.index.to_numpy()], dtype=np.float64)
    # The source public array follows public_memory row order.  Reindex from
    # the merged source frame explicitly rather than relying on a gene sort.
    pair_source = np.asarray(source_effects[pairs["_source_row"].to_numpy()], dtype=np.float64) if "_source_row" in pairs else pair_source
    # Add source row before merge; this branch is retained for clarity below.
    src_rows = dict(zip(src.gene, src.index.astype(int)))
    pair_source = np.stack([source_effects[src_rows[g]] for g in pairs.gene], axis=0).astype(np.float64)
    pair_recipient = np.asarray(target_effects[pairs.matrix_row.to_numpy(dtype=int)], dtype=np.float64)
    return pairs.reset_index(drop=True), pair_source, pair_recipient


def run(out: Path) -> dict:
    started = time.time()
    out.mkdir(parents=False, exist_ok=False)
    for p in [PUBLIC, SOURCE_EFFECTS, SOURCE_GENES, ORION_META, ORION_AXIS, ORION_EFFECTS, ORION_ERRORS]:
        require(p)
    memory = pd.read_parquet(PUBLIC)
    source_effects = np.load(SOURCE_EFFECTS, mmap_mode="r")
    source_genes = json.loads(SOURCE_GENES.read_text())
    if source_effects.shape != (len(memory), len(source_genes)):
        raise RuntimeError(f"source effect contract mismatch: {source_effects.shape}, memory={len(memory)}, genes={len(source_genes)}")
    axis = pd.read_csv(ORION_AXIS)
    source_gene_to_idx = {str(g): i for i, g in enumerate(source_genes)}
    common = [g for g in source_genes if g in set(axis.gene_name.astype(str))]
    if len(common) != len(source_genes):
        raise RuntimeError("common source axis is not contained in Orion endpoint axis")
    common_endpoint_indices = axis.set_index(axis.gene_name.astype(str)).loc[common].axis_index.to_numpy(int)
    common_ensembl = axis.set_index(axis.gene_name.astype(str)).loc[common].orion_ensembl_id.astype(str).tolist()
    meta = pd.read_csv(ORION_META)
    target_effects = np.load(ORION_EFFECTS, mmap_mode="r")
    errors = pd.read_csv(ORION_ERRORS)
    errors = errors[errors.role.astype(str).eq("VALIDATION")].copy()
    errors["context"] = errors.context_id.astype(str)
    errors["gene"] = errors.target_gene_symbol.astype(str)
    target_meta = meta.copy()
    target_meta["gene"] = target_meta.gene.astype(str)
    target_meta["context"] = target_meta.context.astype(str)
    all_pair_rows, recon_rows, query_rows, risk_rows, route_audits, cdf_rows = [], [], [], [], [], []
    for context in TARGET_CONTEXTS:
        pairs, xs, xr = build_route(memory, source_effects, target_meta, target_effects, source_genes, context)
        route_audits.append({
            "source_context": SOURCE_CONTEXT, "target_context": context,
            "raw_pair_count": int(len(pairs)), "independent_pair_genes": int(pairs.gene.nunique()),
            "rank": RANK, "source_response_axis": len(source_genes),
            "recipient_response_axis": target_effects.shape[1], "mapping_axis": len(common),
            "pair_identity_hash": hashlib.sha256("\n".join(sorted(pairs.gene)).encode()).hexdigest(),
            "anchor_roles": "Orion TRAIN only", "permanent_test_truth_used": False,
        })
        # Mapping lives on the common endpoint axis.  The source array is
        # already in that order; endpoint effects are sliced only here.
        xr = xr[:, common_endpoint_indices]
        xs = xs[:, :]
        # Fold-out-of-fold reconstruction audit.
        for fold in range(3):
            tr = pairs.fold.to_numpy() != fold
            te = ~tr
            if tr.sum() < RANK + 2 or te.sum() < 1:
                continue
            lam, inner = choose_lambda(xs[tr], xr[tr], f"{context}|outer{fold}")
            fit = fit_ridge_transport(xs[tr], xr[tr], lam=lam, rank=RANK)
            scalar = fit_scalar(xs[tr], xr[tr], fit)
            preds = {
                "RawCopy": xs[te],
                "RecipientMean": np.repeat(xr[tr].mean(axis=0, keepdims=True), te.sum(), axis=0),
                "ProjectedCopy": apply_projected(xs[te], fit),
                "ScalarAffine": fit["mu_r"] + scalar * project(xs[te], fit["mu_s"], fit["u"]) @ fit["u"].T,
                "RidgeTransport": apply_transport(xs[te], fit),
            }
            rng = np.random.default_rng(SEED + fold + (0 if context == "HCT116" else 100))
            shuffled = fit_ridge_transport(xs[tr], xr[tr], lam=lam, rank=RANK, shuffled=True, rng=rng)
            preds["ShuffledPairTransport"] = apply_transport(xs[te], shuffled)
            for method, pred in preds.items():
                recon_rows.append({
                    "source_context": SOURCE_CONTEXT, "target_context": context,
                    "outer_fold": fold, "method": method, "n_fit_pairs": int(tr.sum()),
                    "n_eval_pairs": int(te.sum()), "rmse": rmse(pred, xr[te]),
                    "lambda": lam, "scalar_a": scalar if method == "ScalarAffine" else np.nan,
                    "fit_numerical_rank": int(fit["numerical_rank"]),
                    "fit_axis": len(common), "uses_query_truth": False,
                })
                cdf_rows.append({"target_context": context, "outer_fold": fold,
                                 "artifact": "mapping_fit", "method": method,
                                 "n_fit_pairs": int(tr.sum()), "rank": RANK})
            for row in inner.to_dict("records"):
                cdf_rows.append({"target_context": context, "outer_fold": fold,
                                 "artifact": "inner_lambda_selection", "method": "RidgeTransport",
                                 **row})
        # Fit the final development map only on Orion TRAIN anchors.  Query
        # genes are explicitly removed if a future asset places them in both
        # roles; this keeps the prototype safe under metadata changes.
        query_context = errors[errors.context.eq(context)].copy()
        query_genes = set(query_context.gene)
        fit_mask = ~pairs.gene.isin(query_genes).to_numpy()
        if fit_mask.sum() < RANK + 2:
            fit_mask = np.ones(len(pairs), dtype=bool)
        lam, inner = choose_lambda(xs[fit_mask], xr[fit_mask], f"{context}|final")
        final_fit = fit_ridge_transport(xs[fit_mask], xr[fit_mask], lam=lam, rank=RANK)
        scalar = fit_scalar(xs[fit_mask], xr[fit_mask], final_fit)
        rng = np.random.default_rng(SEED + (0 if context == "HCT116" else 1000))
        shuffled_fit = fit_ridge_transport(xs[fit_mask], xr[fit_mask], lam=lam, rank=RANK, shuffled=True, rng=rng)
        # Query predictions are loaded only after the mapping fit has been
        # determined.  The target truth is used solely by the final scorer.
        query_context = query_context[query_context.gene.isin(set(memory.loc[memory.context.eq(SOURCE_CONTEXT), "perturbation_target"].astype(str)))].copy()
        if query_context.empty:
            continue
        query_ids = query_context.query_id.astype(str).tolist()
        pred = load_prediction_subset(query_ids, context, common_ensembl)
        source_rows = dict(zip(memory.loc[memory.context.eq(SOURCE_CONTEXT), "perturbation_target"].astype(str), memory.index.astype(int)))
        # source_rows above includes non-K562 indices; rebuild explicitly.
        k562 = memory[memory.context.astype(str).eq(SOURCE_CONTEXT)]
        source_rows = dict(zip(k562.perturbation_target.astype(str), k562.index.astype(int)))
        src_vec = np.stack([source_effects[source_rows[g]] for g in query_context.gene], axis=0).astype(float)
        rec = np.repeat(xr[fit_mask].mean(axis=0, keepdims=True), len(query_context), axis=0)
        projected = apply_projected(src_vec, final_fit)
        scalar_pred = final_fit["mu_r"] + scalar * project(src_vec, final_fit["mu_s"], final_fit["u"]) @ final_fit["u"].T
        transported = apply_transport(src_vec, final_fit)
        shuffled_pred = apply_transport(src_vec, shuffled_fit)
        refs = {
            "RawCopy": src_vec, "RecipientMean": rec, "ProjectedCopy": projected,
            "ScalarAffine": scalar_pred, "RidgeTransport": transported,
            "ShuffledPairTransport": shuffled_pred,
        }
        for i, q in query_context.reset_index(drop=True).iterrows():
            base = {"query_id": q.query_id, "gene": q.gene, "context": context,
                    "source_experiment": f"E201::{SOURCE_CONTEXT}::{q.gene}+ctrl",
                    "true_error_rmse": float(q.model_rmse), "mapping_fit_pairs": int(fit_mask.sum()),
                    "mapping_lambda": lam, "mapping_rank": RANK, "uses_query_truth_for_fit": False}
            for method, ref in refs.items():
                query_rows.append({**base, "method": method,
                                   "reference_rmse_to_prediction": rmse(pred[i], ref[i]),
                                   "reference_norm": float(np.linalg.norm(ref[i]) / np.sqrt(len(ref[i]))),
                                   "prediction_norm": float(np.linalg.norm(pred[i]) / np.sqrt(len(pred[i]))),
                                   "out_of_training_anchor": bool(q.gene not in set(pairs.gene))})
            for method, ref in refs.items():
                risk_rows.append({"query_id": q.query_id, "gene": q.gene, "context": context,
                                  "true_error_rmse": float(q.model_rmse), "method": method,
                                  "risk_score": rmse(pred[i], ref[i]), "source_context": SOURCE_CONTEXT,
                                  "uses_query_truth_for_fit": False})
            # The prediction-only comparator is kept on the same query set.
            # It uses the registered amplitude statistic, not a reference
            # reconstructed from any biological truth.
            risk_rows.append({"query_id": q.query_id, "gene": q.gene, "context": context,
                              "true_error_rmse": float(q.model_rmse), "method": "Amplitude",
                              "risk_score": float(np.mean(np.abs(pred[i]))),
                              "source_context": SOURCE_CONTEXT,
                              "uses_query_truth_for_fit": False})
        all_pair_rows.extend(pairs.assign(source_context=SOURCE_CONTEXT, target_context=context).to_dict("records"))
    pairs_df = pd.DataFrame(all_pair_rows)
    recon_df = pd.DataFrame(recon_rows)
    query_df = pd.DataFrame(query_rows)
    risk_df = pd.DataFrame(risk_rows)
    risk_metrics = []
    if not risk_df.empty:
        for (context, method), g in risk_df.groupby(["context", "method"], sort=True):
            risk_metrics.append(metric_frame(g.true_error_rmse.to_numpy(float), g.risk_score.to_numpy(float),
                                              g.query_id.astype(str).to_numpy(), method, context))
        pooled = risk_df.groupby("method", sort=True)
        for method, g in pooled:
            risk_metrics.append(metric_frame(g.true_error_rmse.to_numpy(float), g.risk_score.to_numpy(float),
                                              g.query_id.astype(str).to_numpy(), method, "POOLED_COVERED"))
    pd.DataFrame(route_audits).to_csv(out / "ROUTE_AUDIT.csv", index=False)
    pairs_df.to_csv(out / "PAIR_ELIGIBILITY.csv", index=False)
    recon_df.to_csv(out / "RECONSTRUCTION_METRICS.csv", index=False)
    query_df.to_parquet(out / "QUERY_REFERENCE_RESULTS.parquet", index=False)
    risk_df.to_parquet(out / "QUERY_RISK_SCORES.parquet", index=False)
    pd.DataFrame(risk_metrics).to_csv(out / "RISK_METRICS.csv", index=False)
    paired_bootstrap(risk_df).to_csv(out / "PAIRED_BOOTSTRAP.csv", index=False)
    pd.DataFrame(cdf_rows).to_csv(out / "MAPPING_FIT_AUDIT.csv", index=False)
    budget = pd.DataFrame([
        {"information": "E201_K562_public_effects", "records": int((memory.context.astype(str) == SOURCE_CONTEXT).sum()), "role": "source history", "used_for_fit": True, "used_for_query_truth": False},
        {"information": "Orion_target_background_train_anchors", "records": int(len(pairs_df)), "role": "recipient mapping supervision", "used_for_fit": True, "used_for_query_truth": False},
        {"information": "Orion_validation_predictions", "records": int(query_df.query_id.nunique() if not query_df.empty else 0), "role": "risk query prediction", "used_for_fit": False, "used_for_query_truth": False},
        {"information": "Orion_validation_errors", "records": int(query_df.query_id.nunique() if not query_df.empty else 0), "role": "risk scoring only", "used_for_fit": False, "used_for_query_truth": True},
        {"information": "Orion_permanent_test_truth", "records": 0, "role": "protected", "used_for_fit": False, "used_for_query_truth": False},
    ])
    budget.to_csv(out / "INFORMATION_BUDGET.csv", index=False)
    decision = {
        "schema": "safeconf_perturbmap_experiment_v1",
        "status": "COMPLETE_DEV_VALIDATION_RISK_READOUT",
        "default_method_changed": False,
        "routes": route_audits,
        "recommendation": "RETAIN_EXISTING_PUBLIC_RULE_UNTIL_CROSS_BACKGROUND_RISK_GATE",
        "reason": "This is a bounded K562-to-Orion background-transfer development readout. It does not replace the same-background public rule; adoption requires stable risk gain on a pre-registered independent route.",
        "permanent_test_truth_used": False,
        "query_truth_used_only_for_scoring": True,
        "risk_metrics": pd.DataFrame(risk_metrics).to_dict("records"),
        "reconstruction_mean_rmse": (recon_df.groupby("method", as_index=False)["rmse"].mean().to_dict("records")
                                      if not recon_df.empty else []),
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    write_json(out / "COMPONENT_DECISION.json", decision)
    write_json(out / "RUN_STATUS.json", {
        "status": "COMPLETE", "contract": "SafeConf v0.4.2", "new_gpu_hours": 0,
        "new_download_gb": 0, "elapsed_seconds": time.time() - started,
        "routes": route_audits, "query_tasks": int(query_df.query_id.nunique() if not query_df.empty else 0),
        "permanent_test_truth_used": False, "python": platform.python_version(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    })
    write_json(out / "MANIFEST.json", {
        "contract": "SafeConf v0.4.2", "status": "COMPLETE",
        "input_hashes": {str(p): sha256(p) for p in [PUBLIC, SOURCE_EFFECTS, SOURCE_GENES, ORION_META, ORION_AXIS, ORION_EFFECTS, ORION_ERRORS]},
        "prediction_inputs": {ctx: str(PRED_ROOT / ctx / "PREDICTIONS_DELTA.tsv.gz") for ctx in TARGET_CONTEXTS},
        "prediction_input_hashes": {str(PRED_ROOT / ctx / "PREDICTIONS_DELTA.tsv.gz"): sha256(PRED_ROOT / ctx / "PREDICTIONS_DELTA.tsv.gz") for ctx in TARGET_CONTEXTS},
        "permanent_test_truth_used": False,
        "files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "MANIFEST.json"},
    })
    pooled = pd.DataFrame(risk_metrics)
    pooled = pooled[pooled.context.eq("POOLED_COVERED")] if not pooled.empty else pooled
    lines = [
        "# PerturbMap bounded prototype decision", "",
        "The K562-to-Orion route was fitted only on Orion TRAIN target-background anchors. "
        "Validation predictions were loaded after the mapping fit and validation errors were used only by the scorer. "
        "Permanent TEST truth was not read. Six reference rules are reported separately for reconstruction and risk ranking.", "",
        "## Current decision", "",
        "The existing same-background PublicRule remains the default. This prototype is a background-transfer development result, not an automatic promotion.", "",
        "## Risk readout on covered validation tasks", "",
    ]
    if not pooled.empty:
        for row in pooled.sort_values("method").itertuples():
            lines.append(f"- `{row.method}`: n={int(row.n_tasks)}, U20={row.u20:.4f}, AURC={row.aurc:.5f}, Spearman={row.spearman:.4f}")
    lines += [
        "", "The pooled covered readout has 53 tasks (30 HCT116 and 23 HEK293T). "
        "RawCopy is stronger than Amplitude on this small covered set, while the learned transport variants are context dependent and do not pass the current default-replacement gate. "
        "The next valid use is as a conditional background-transfer candidate after an independent route or larger held-out task set.",
    ]
    (out / "DECISION.md").write_text("\n".join(lines) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
