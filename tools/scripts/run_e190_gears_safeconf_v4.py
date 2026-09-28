#!/usr/bin/env python3
"""Run frozen SafeConf-v4 on the opened E190 GEARS cross-study asset.

This is cross-family *development* evidence.  E190 target truth was opened in
July 2026, so this script must never label its output independent confirmation
or alter FINAL_CANDIDATE_FREEZE.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import run_safeconf_v4_development as core  # noqa: E402


STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
OUT = STAGE / "safeconf_v4_development/e190_gears_crossfamily"
PUBLIC = Path("/home/yyf/proj/docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729")
ASSETS = Path("/home/yyf/data/safeconf_e190_adamson_replogle/model_assets")
SEED = 20260930
N_BOOTSTRAP = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lock(lock_csv: Path, locked_suffix: str, actual: Path) -> str:
    locks = pd.read_csv(lock_csv, keep_default_na=False)
    row = locks.loc[locks.path.astype(str).str.endswith(locked_suffix)]
    if len(row) != 1:
        raise RuntimeError(f"lock entry is not unique: {locked_suffix}")
    expected = str(row.iloc[0].sha256)
    observed = sha256(actual)
    if expected != observed or int(row.iloc[0].bytes) != actual.stat().st_size:
        raise RuntimeError(f"locked E190 asset changed: {actual}")
    return observed


def verify_inputs() -> dict[str, str]:
    checks = {
        "prediction": (
            PUBLIC / "pretruth_release/RELEASE_LOCKS.csv",
            "arrays/PRETRUTH_PREDICTIONS.npz",
            PUBLIC / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz",
        ),
        "truth": (
            PUBLIC / "evaluation_truth/TRUTH_LOCKS.csv",
            "arrays/TARGET_TRUE_EFFECTS.npz",
            PUBLIC / "evaluation_truth/arrays/TARGET_TRUE_EFFECTS.npz",
        ),
        "source_effect": (
            PUBLIC / "MODEL_ASSET_LOCKS.csv",
            "model_assets/SOURCE_GENE_EFFECTS.npz",
            ASSETS / "SOURCE_GENE_EFFECTS.npz",
        ),
        "train_effects": (
            PUBLIC / "MODEL_ASSET_LOCKS.csv",
            "model_assets/TRAIN_EFFECTS.npz",
            ASSETS / "TRAIN_EFFECTS.npz",
        ),
        "validation_effects": (
            PUBLIC / "MODEL_ASSET_LOCKS.csv",
            "model_assets/VALIDATION_EFFECTS.npz",
            ASSETS / "VALIDATION_EFFECTS.npz",
        ),
    }
    return {name: verify_lock(*spec) for name, spec in checks.items()}


def load_npz_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key], np.float64) for key in archive.files}


def build_frame() -> tuple[pd.DataFrame, dict]:
    query = pd.read_csv(ASSETS / "QUERY_TASKS.csv", keep_default_na=False)
    if len(query) != 692 or query.gene.nunique() != 47:
        raise RuntimeError("E190 frozen query manifest changed")

    with np.load(PUBLIC / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        keys = sorted(key for key in archive.files if key.startswith("GEARS_"))
        if len(keys) != 3:
            raise RuntimeError("expected the frozen three-member GEARS family")
        prediction = np.mean([np.asarray(archive[key], np.float64) for key in keys], axis=0)
    source_map = load_npz_map(ASSETS / "SOURCE_GENE_EFFECTS.npz")
    source = np.stack([source_map[gene] for gene in query.gene.astype(str)])

    # Construct every prediction/history feature before loading target truth.
    abs_effect = np.abs(prediction)
    frame = query[["task_id", "gene", "batch", "n_target_cells"]].copy()
    frame["target"] = "K562"
    frame["predicted_magnitude"] = np.sqrt(np.mean(prediction**2, axis=1))
    frame["prediction_abs_mean"] = abs_effect.mean(axis=1)
    frame["prediction_signed_mean"] = prediction.mean(axis=1)
    frame["prediction_std"] = prediction.std(axis=1)
    frame["prediction_abs_q95"] = np.quantile(abs_effect, 0.95, axis=1)
    row_scale = np.maximum(frame.prediction_abs_q95.to_numpy(float), 1e-12)
    frame["prediction_sparsity"] = np.mean(abs_effect <= row_scale[:, None] * 0.01, axis=1)

    source_assignments = pd.read_csv(PUBLIC / "E190_SOURCE_CELL_FOLD_ASSIGNMENTS.csv")
    support = source_assignments.groupby("gene", as_index=False).agg(
        n_source_cells=("source_cell_id", "count"),
        n_source_contexts=("split", lambda _: 1),
        n_source_batches=("perturbation", "nunique"),
    )
    fold_counts = source_assignments.groupby(["gene", "fold"]).size().groupby("gene").min()
    support["min_source_cells"] = support.gene.map(fold_counts).astype(float)
    frame = frame.merge(support, on="gene", how="left", validate="many_to_one")

    numerator = np.sum(prediction * source, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(source, axis=1)
    frame["prediction_source_cosine"] = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
    )
    gap = np.sqrt(np.mean((prediction - source) ** 2, axis=1))
    frame["negative_model_source_gap"] = -gap
    frame["source_transfer_magnitude"] = np.sqrt(np.mean(source**2, axis=1))

    train_meta = pd.read_csv(ASSETS / "TRAIN_TASKS.csv", keep_default_na=False)
    validation_meta = pd.read_csv(ASSETS / "VALIDATION_TASKS.csv", keep_default_na=False)
    effect_map = load_npz_map(ASSETS / "TRAIN_EFFECTS.npz")
    effect_map.update(load_npz_map(ASSETS / "VALIDATION_EFFECTS.npz"))
    source_meta = pd.concat([train_meta, validation_meta], ignore_index=True)
    dispersions: dict[str, float] = {}
    for gene, part in source_meta.groupby("gene", sort=True):
        folds = np.stack([effect_map[task] for task in part.task_id.astype(str)])
        center = folds.mean(axis=0)
        dispersions[str(gene)] = float(np.sqrt(np.mean((folds - center) ** 2)))
    frame["source_delta_dispersion"] = frame.gene.map(dispersions).astype(float)
    frame["conflict_missing"] = 0.0

    feature_columns = sorted(set(sum(core.GROUPS.values(), [])))
    if not np.isfinite(frame[feature_columns].to_numpy(float)).all():
        raise RuntimeError("E190 prediction/history feature table contains non-finite values")

    # Load labels only after the feature table is complete.
    truth_map = load_npz_map(PUBLIC / "evaluation_truth/arrays/TARGET_TRUE_EFFECTS.npz")
    truth = np.stack([truth_map[task] for task in frame.task_id.astype(str)])
    error = np.sqrt(np.mean((prediction - truth) ** 2, axis=1))
    frame["true_error_rmse"] = error
    frame["family_centroid_rmse"] = error  # frozen core runner compatibility
    return frame, {
        "n_tasks": int(len(frame)),
        "n_gene_clusters": int(frame.gene.nunique()),
        "n_target_batches": int(frame.batch.nunique()),
        "upstream": "GEARS_3seed_centroid",
        "output_contract_id": "E190_GEARS_direct_effect_512gene_v1",
        "history_source": "external_Adamson_source_study_frozen_pretruth",
        "quality_fields_available": False,
        "conflict_proxy_coverage": 1.0,
        "sealed_confirmation_opened": False,
    }


def fold_results(predictions: pd.DataFrame) -> pd.DataFrame:
    fold_map = predictions.loc[predictions.method.eq("Ridge_USR"), ["task_id", "fold"]]
    fold_map = fold_map.drop_duplicates("task_id")
    work = predictions.merge(fold_map, on="task_id", suffixes=("", "_oof"), how="left", validate="many_to_one")
    work["eval_fold"] = work.fold_oof.astype(int)
    rows = []
    for (fold, method), group in work.groupby(["eval_fold", "method"], sort=True):
        ids = group.task_id.to_numpy(str)
        risk = group.predicted_risk.to_numpy(float)
        truth = group.true_error_rmse.to_numpy(float)
        rows.append({
            "fold": int(fold), "method": method, "n_tasks": int(len(group)),
            "utility20": core.utility20(ids, risk, truth),
            "spearman": core.rho(risk, truth),
            **core.selective_metrics(ids, risk, truth),
        })
    return pd.DataFrame(rows)


def bootstrap(predictions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("UniversalP_vs_magnitude", "Ridge_U", "Magnitude_raw"),
        ("Support_given_P", "Ridge_US", "Ridge_U"),
        ("V1_vs_magnitude", "Ridge_USR", "Magnitude_raw"),
        ("V1_vs_UniversalP", "Ridge_USR", "Ridge_U"),
        ("Relevance_given_support", "Ridge_USR", "Ridge_US"),
        ("Conflict_given_relevance", "Ridge_USRC", "Ridge_USR"),
        ("Content_given_conflict", "Ridge_USRCH", "Ridge_USRC"),
        ("V2_vs_V1", "V2_nested", "Ridge_USR"),
        ("V1_vs_shuffled_history", "Ridge_USR", "Ridge_USR_history_shuffled"),
    ]
    wide = predictions.pivot(
        index=["task_id", "target", "gene", "true_error_rmse"], columns="method", values="predicted_risk"
    ).reset_index()
    if len(wide) != 692 or wide.isna().any().any():
        raise RuntimeError("E190 OOF prediction matrix is incomplete")
    genes = np.asarray(sorted(wide.gene.unique()))
    gene_rows = [np.flatnonzero(wide.gene.to_numpy() == gene) for gene in genes]
    ids = wide.task_id.to_numpy(str)
    truth = wide.true_error_rmse.to_numpy(float)
    needed = {name for _, a, b in comparisons for name in (a, b)}
    risks = {name: wide[name].to_numpy(float) for name in needed}
    rng = np.random.default_rng(SEED)
    u_draws = {name: [] for name, _, _ in comparisons}
    r_draws = {name: [] for name, _, _ in comparisons}
    for _ in range(N_BOOTSTRAP):
        selected = rng.integers(0, len(genes), len(genes))
        rows = np.concatenate([gene_rows[index] for index in selected])
        metrics = {
            method: (core.utility20(ids[rows], risk[rows], truth[rows]), core.rho(risk[rows], truth[rows]))
            for method, risk in risks.items()
        }
        for name, left, right in comparisons:
            u_draws[name].append(metrics[left][0] - metrics[right][0])
            r_draws[name].append(metrics[left][1] - metrics[right][1])
    point = summary.set_index("method")
    result = []
    for name, left, right in comparisons:
        u = np.asarray(u_draws[name], float)
        r = np.asarray(r_draws[name], float)
        result.append({
            "comparison": name, "method_a": left, "method_b": right,
            "delta_utility20": float(point.loc[left, "utility20"] - point.loc[right, "utility20"]),
            "utility_ci95_lower": float(np.nanquantile(u, 0.025)),
            "utility_ci95_upper": float(np.nanquantile(u, 0.975)),
            "delta_spearman": float(point.loc[left, "spearman"] - point.loc[right, "spearman"]),
            "spearman_ci95_lower": float(np.nanquantile(r, 0.025)),
            "spearman_ci95_upper": float(np.nanquantile(r, 0.975)),
            "bootstrap_replicates": N_BOOTSTRAP,
        })
    return pd.DataFrame(result)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite opened result directory: {OUT}")
    competence = pd.read_csv(STAGE / "UPSTREAM_COMPETENCE_V4.csv")
    gate = competence.loc[
        competence.asset.eq("E190_Adamson_to_Replogle_K562")
        & competence.upstream_model.eq("GEARS_3seed_centroid")
    ]
    if len(gate) != 1 or not bool(gate.iloc[0].passes_2pct_competence_gate):
        raise RuntimeError("E190 GEARS failed or lacks the pre-SafeConf competence gate")

    started = time.time()
    hashes = verify_inputs()
    frame, audit = build_frame()
    predictions, gate_fits = core.evaluate(frame)
    strata, summary, coverage = core.score(predictions)
    folds = fold_results(predictions)
    increments = bootstrap(predictions, summary)
    matched = core.matched_support(frame, predictions)

    OUT.mkdir(parents=True)
    frame.to_csv(OUT / "FEATURE_TABLE.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    predictions.to_csv(OUT / "OOF_PREDICTIONS.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    gate_fits.to_csv(OUT / "V2_GATE_FITS.csv", index=False)
    folds.to_csv(OUT / "FOLD_RESULTS.csv", index=False)
    strata.to_csv(OUT / "STRATUM_RESULTS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    coverage.to_csv(OUT / "RISK_COVERAGE.csv", index=False)
    increments.to_csv(OUT / "INCREMENTAL_RESULTS.csv", index=False)
    matched.to_csv(OUT / "MATCHED_SUPPORT.csv", index=False)

    lookup = summary.set_index("method")
    increment = increments.set_index("comparison")
    report = [
        "# E190 GEARS SafeConf-v4 cross-family development result",
        "",
        "This is an already-opened SEEN asset. It is cross-family development evidence, not independent confirmation.",
        "",
        "| Method | Utility@20 | Spearman | AURC |",
        "|---|---:|---:|---:|",
    ]
    for method in ["Magnitude_raw", "Ridge_U", "Ridge_US", "Ridge_USR", "V2_nested", "HGB_USR", "MLP_USR"]:
        row = lookup.loc[method]
        report.append(f"| {method} | {row.utility20:.6f} | {row.spearman:.6f} | {row.aurc:.6f} |")
    report.extend(["", "## Paired gene-cluster increments", ""])
    for name in ["V1_vs_magnitude", "V1_vs_UniversalP", "Relevance_given_support", "V2_vs_V1", "V1_vs_shuffled_history"]:
        row = increment.loc[name]
        report.append(
            f"- {name}: ΔU20={row.delta_utility20:+.6f} "
            f"[{row.utility_ci95_lower:+.6f}, {row.utility_ci95_upper:+.6f}], "
            f"Δρ={row.delta_spearman:+.6f} "
            f"[{row.spearman_ci95_lower:+.6f}, {row.spearman_ci95_upper:+.6f}]."
        )
    report.extend([
        "",
        "## Interpretation boundary",
        "",
        "- The upstream passed the competence gate before this SafeConf run.",
        "- The target is one K562 cell context; target batches are technical/experimental strata, not new biological contexts.",
        "- Adamson history is external to the Replogle target study and was frozen before target truth.",
        "- Quality is not claimed: one source study/context cannot satisfy the formal Quality gate.",
        "- These results cannot replace the still-missing eligible SEALED confirmation.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    status = {
        "status": "COMPLETE",
        "data_role": "SEEN",
        "independent_confirmation": False,
        "sealed_confirmation_opened": False,
        "final_candidate_changed": False,
        "contract": str(STAGE / "E190_GEARS_SAFECONF_V4_CONTRACT.md"),
        "input_hashes": hashes,
        "audit": audit,
        "elapsed_seconds": time.time() - started,
        "summary_sha256": sha256(OUT / "SUMMARY.csv"),
        "incremental_results_sha256": sha256(OUT / "INCREMENTAL_RESULTS.csv"),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(increments.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
