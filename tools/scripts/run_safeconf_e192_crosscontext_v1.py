#!/usr/bin/env python3
"""Cross-context E192 audit after rebuilding public effects at the same scale.

The target predictor arrays are frozen pretruth artifacts.  Public K562
effects are streamed from the already-local GWPS file with the E192 10000
normalization, so the risk calculation never reads E192 target expression or
the target truth array.  This is a SEEN same-study cross-context audit, not an
independent-study confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

E192 = Path("/home/yyf/data/safeconf_e192_adamson_rpe1")
E192_DOC = Path("/home/yyf/proj/docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729")
GWPS = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ReplogleWeissman2022_K562_gwps.h5ad")
DEFAULT_OUT = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/e192_crosscontext")
BOOT = 5000
SEED = 20260930


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def h5col(group, name):
    x = group[name]
    if isinstance(x, h5py.Group):
        cats = x["categories"].asstr()[:] if x["categories"].dtype.kind in "OS" else x["categories"][:]
        codes = x["codes"][:]
        if (codes < 0).any():
            raise RuntimeError(f"missing categorical values in {name}")
        return cats[codes]
    return x.asstr()[:] if x.dtype.kind in "OS" else x[:]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def stream_k562_effects(panel: pd.DataFrame, output: Path) -> tuple[dict[str, np.ndarray], dict]:
    panel_genes = panel.gene_name.astype(str).tolist()
    with h5py.File(GWPS, "r") as f:
        raw_gene_names = h5col(f["var"], "gene_name").astype(str)
        mapping = {g: i for i, g in enumerate(raw_gene_names)}
        missing = sorted(set(panel_genes) - set(mapping))
        used_panel_genes = [g for g in panel_genes if g in mapping]
        panel_cols = [mapping[g] for g in used_panel_genes]
        sorted_cols = np.asarray(sorted(panel_cols), dtype=int)
        reorder = np.asarray([int(np.flatnonzero(sorted_cols == c)[0]) for c in panel_cols], dtype=int)
        perturbations = h5col(f["obs"], "perturbation")
        batches = h5col(f["obs"], "batch")
        totals = h5col(f["obs"], "UMI_count").astype(float)
        codes, perturbation_levels = pd.factorize(perturbations, sort=False)
        batch_codes, batch_levels = pd.factorize(batches, sort=False)
        control_code = int(np.flatnonzero(perturbation_levels == "control")[0])
        n_gene, n_batch, n_panel = len(perturbation_levels), len(batch_levels), len(used_panel_genes)
        selected_codes = {g: int(np.flatnonzero(perturbation_levels == g)[0]) for g in used_panel_genes if g in set(perturbation_levels)}
        missing_targets = sorted(set(used_panel_genes) - set(selected_codes))
        if missing_targets:
            raise RuntimeError(f"panel genes absent from GWPS observations: {missing_targets}")
        gene_sum = np.zeros((n_gene, n_panel), dtype=np.float64)
        gene_count = np.zeros(n_gene, dtype=np.int64)
        gene_batch_counts = np.zeros((n_gene, n_batch), dtype=np.int64)
        ctrl_sum = np.zeros((n_batch, n_panel), dtype=np.float64)
        ctrl_count = np.zeros(n_batch, dtype=np.int64)
        block = 8192
        for start in range(0, len(codes), block):
            stop = min(start + block, len(codes))
            raw = np.asarray(f["X"][start:stop, sorted_cols], dtype=np.float64)[:, reorder]
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise RuntimeError(f"invalid raw count block {start}:{stop}")
            total = totals[start:stop]
            if not np.isfinite(total).all() or (total <= 0).any():
                raise RuntimeError("invalid UMI totals")
            raw *= (10000.0 / total)[:, None]
            np.log1p(raw, out=raw)
            g = codes[start:stop]
            b = batch_codes[start:stop]
            selected = np.isin(g, np.fromiter(selected_codes.values(), dtype=int))
            for code in np.unique(g[selected]):
                rows = g == code
                gene_sum[code] += raw[rows].sum(axis=0)
                gene_count[code] += int(rows.sum())
                gene_batch_counts[code] += np.bincount(b[rows], minlength=n_batch)
            controls = g == control_code
            for code in np.unique(b[controls]):
                rows = controls & (b == code)
                ctrl_sum[code] += raw[rows].sum(axis=0)
                ctrl_count[code] += int(rows.sum())
    ctrl_mean = ctrl_sum / np.maximum(ctrl_count[:, None], 1)
    effects = {}
    for gene, code in selected_codes.items():
        treated_mean = gene_sum[code] / max(gene_count[code], 1)
        # Match the existing GWPS estimand: treated batch frequencies weight
        # the NTC means rather than replacing them with a global control.
        batch_counts = gene_batch_counts[code].astype(float)
        matched = batch_counts @ ctrl_mean / max(gene_count[code], 1)
        effects[gene] = treated_mean - matched
    audit = {"normalization": "log1p(10000 * raw retained gene UMI / original UMI_count)",
             "effect": "cell-equal treated mean minus NTC mean matched to treated batch frequencies",
             "n_panel_genes": len(panel_genes), "n_effects": len(effects),
             "missing_panel_genes": missing,
             "panel_axis_sha256": sha(E192 / "model_assets/GENE_PANEL.csv"),
             "gwps_sha256": sha(GWPS), "target_truth_read": False,
             "raw_target_perturbation_expression_read": False}
    np.savez_compressed(output, **effects)
    return effects, audit


def task_metric(ids, truth, score):
    ids, truth, score = np.asarray(ids).astype(str), np.asarray(truth, float), np.asarray(score, float)
    k = int(np.ceil(.2 * len(truth)))
    top = np.lexsort((ids, -score))[:k]; oracle = np.lexsort((ids, -truth))[:k]
    den = truth[oracle].mean() - truth.mean()
    found = int(len(set(top).intersection(set(oracle))))
    return {"n_tasks": len(truth), "review_count": k,
            "u20": float((truth[top].mean() - truth.mean()) / den) if den > 1e-12 else np.nan,
            "aurc": float(np.mean(np.cumsum(truth[np.lexsort((ids, score))]) / np.arange(1, len(truth) + 1))),
            "spearman": float(spearmanr(score, truth).statistic) if np.ptp(score) > 0 and np.ptp(truth) > 0 else np.nan,
            "true_top20_found": found, "high_risk_miss_rate": 1 - found / k,
            "remaining_mean_error": float(np.delete(truth, top).mean())}


def bootstrap_delta(ids, genes, truth, public, amplitude):
    ids, genes = np.asarray(ids).astype(str), np.asarray(genes).astype(str)
    # Sort once by task ID so stable score sorts implement the registered tie
    # rule.  Vectorized draws avoid a Python sort loop over 5,000 replicates.
    order = np.argsort(ids, kind="stable")
    ids, genes = ids[order], genes[order]
    truth, public, amplitude = np.asarray(truth, float)[order], np.asarray(public, float)[order], np.asarray(amplitude, float)[order]
    blocks = [np.flatnonzero(genes == g) for g in sorted(set(genes))]
    n = len(truth)
    def one_delta(idx, a, b):
        y, ident = truth[idx], ids[idx]; k = int(np.ceil(.2 * len(idx)))
        top_a = np.argsort(-a[idx], kind="stable")[:k]; top_b = np.argsort(-b[idx], kind="stable")[:k]
        oracle = np.argsort(-y, kind="stable")[:k]
        d = y[oracle].mean() - y.mean()
        return (y[top_a].mean() - y.mean()) / d - (y[top_b].mean() - y.mean()) / d if d > 1e-12 else np.nan
    rng = np.random.default_rng(SEED)
    choices = rng.integers(0, len(blocks), size=(BOOT, len(blocks)))
    draws = np.asarray([one_delta(np.concatenate([blocks[int(i)] for i in row]), public, amplitude)
                        for row in choices])
    return {"point_delta": one_delta(np.arange(n), public, amplitude), "bootstrap_mean_delta": float(np.nanmean(draws)),
            "ci95_lower": float(np.nanquantile(draws, .025)), "ci95_upper": float(np.nanquantile(draws, .975)),
            "bootstrap_replicates": BOOT, "n_genes": len(blocks)}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=DEFAULT_OUT); args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(E192 / "model_assets/GENE_PANEL.csv")
    queries = pd.read_csv(E192 / "model_assets/QUERY_TASKS.csv")
    metrics = pd.read_csv(E192_DOC / "final_evaluation/tables/E192_TASK_METRICS.csv").set_index("task_id")
    pred = np.load(E192_DOC / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz")
    effect_path = out / "E192_K562_PUBLIC_EFFECTS_10000.npz"
    if effect_path.exists():
        loaded = np.load(effect_path)
        effects = {k: loaded[k] for k in loaded.files}
        stream_audit = {"reused_effect_artifact": True, "effect_path": str(effect_path),
                        "n_effects": len(effects), "target_truth_read": False}
    else:
        effects, stream_audit = stream_k562_effects(panel, effect_path)
    gene_to_effect = effects
    rows = []
    comparisons = []
    available = queries.gene.astype(str).isin(set(gene_to_effect)).to_numpy()
    used_queries = queries.loc[available].reset_index(drop=True)
    used_indices = np.flatnonzero(available)
    for member in pred.files:
        vectors = pred[member][used_indices]
        # The same public panel mask is used for both amplitude and distance;
        # tasks whose perturbation has no K562 public record are not ranked.
        panel_mask = np.asarray([g in set(gene_to_effect) for g in panel.gene_name.astype(str)], dtype=bool)
        panel_positions = np.flatnonzero(panel_mask)
        vectors = vectors[:, panel_positions]
        amplitude = np.sqrt(np.mean(vectors ** 2, axis=1))
        public = np.asarray([np.sqrt(np.mean((vectors[i] - gene_to_effect[used_queries.gene.iloc[i]]) ** 2)) for i in range(len(used_queries))])
        error_col = f"{member}_rmse"
        truth = metrics.loc[used_queries.task_id.astype(str), error_col].to_numpy(float)
        for method, score in [("Amplitude", amplitude), ("PublicRule_E192_K56210000", public)]:
            m = task_metric(used_queries.task_id, truth, score)
            rows.append({"model": member, "method": method, "error_column": error_col, **m})
        comparisons.append({"model": member, "error_column": error_col, **bootstrap_delta(used_queries.task_id, used_queries.gene, truth, public, amplitude)})
    result = pd.DataFrame(rows); comparison = pd.DataFrame(comparisons)
    write_csv(out / "E192_CROSSCONTEXT_SYSTEM_COMPARISON.csv", result)
    write_csv(out / "E192_CROSSCONTEXT_PAIRED_BOOTSTRAP.csv", comparison)
    write_json(out / "E192_CROSSCONTEXT_AUDIT.json", {
        "status": "COMPLETE_SEEN_SAME_STUDY_CROSS_CONTEXT",
        "n_tasks": len(queries), "scored_tasks": len(used_queries), "n_genes": queries.gene.nunique(),
        "source_context": "K562", "target_context": "RPE1",
        "effect_contract": "E192-compatible 10000 scale rebuilt from GWPS raw counts",
        "target_truth_read": False, "target_error_source": "pre-existing E192_TASK_METRICS.csv only",
        "independent_study_confirmation": False,
        "stream_audit": stream_audit,
        "model_members": pred.files,
        "unscored_query_genes": sorted(set(queries.gene.astype(str)) - set(gene_to_effect)),
        "permanent_test_truth_opened": False,
    })
    print(result.to_string(index=False)); print(comparison.to_string(index=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
