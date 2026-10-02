#!/usr/bin/env python3
"""One Source DEV biological-magnitude supervision control; no method promotion."""
from __future__ import annotations
import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
from unittest.mock import patch

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import (
    CDF_KEYS, P, PUBLIC, SEEDS, FittedRisk, cluster_weights, fit_risk, ids_hash, metrics, rank_labels, summarize,
)

B = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001")
COMMON = B / "common_gene_axis"
CACHE = COMMON / "risk_cache"
R = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001"
D = R / "source_biology_label_control_v1"
O = B / "source_biology_label_control_v1"
V2 = B / "source_objective_probe_v1/approved_20261002_v2"
ARCHIVE = R / "common_gene_axis/results/MATRIX_TASK_PREDICTIONS.csv.gz"
COUNT = B / "common_public_growth_statistics_20261002_v1/Source_BOOTSTRAP_GENE_COUNTS.npy"
COMPARATOR_REG = R / "source_magnitude_comparator_completion_v1/REGISTRATION.json"
HELPER = ROOT / "tools/scripts/run_safeconf_source_scaling_uncertainty_agent.py"
TRUTH = COMMON / "SOURCE_TRUE_EFFECTS.npy"
TASKS = COMMON / "SOURCE_TASKS.csv"
GENES = COMMON / "GENE_IDS.json"
CORE = tuple(ROOT / name for name in ("tools/safeconf_continual/research.py",
    "tools/safeconf_continual/learners.py", "tools/safeconf_continual/contracts.py"))
PAIRS = (("GAT_to_Exphormer", "TxPert_GAT", "TxPert_Exphormer"),
         ("Exphormer_to_GAT", "TxPert_Exphormer", "TxPert_GAT"))
META = list(dict.fromkeys(["task_id", "gene", "fold"] + CDF_KEYS))
KEYS = ["task_id", "target", "gene", "fold", "upstream"]
METHODS = ["BioNorm_HGB", "OldRank_HGB", "Magnitude", "LearnedWeightedHistoryDistance"]
METRICS = ("utility20", "spearman", "aurc", "high_risk_miss_rate", "error_at_10", "error_at_20", "error_at_50")
CONTEXTS = ["K562", "RPE1", "hepg2", "jurkat"]
COUNT_SHA = "524ac5396accdf0892196cddcd2694a1cbf159eb0b8aff57f9c99a41ebd3908e"
GENE_SHA = "f6a9d3cb6e0c2e6ba3630fb3df066392324ef8f34ebc3999001154079ebdc6c3"
MODEL_SHAS = (
    "37e378db03fe7949a1ffc8ed400b480fb6c6f580be26bccc10f08e3b8f271097",
    "45b24e026f33f3f9d19d21c56603d63b131574d2c6e2ab6f78ff698cf1b0b8a6",
    "f6396d586a4cb244e09e754a0af4b18a7a3c9ded88000bb3abd7b49dd7e11fa2",
    "2d7411bba2a5730c9de0a96fc9e73461b1745daef55d59c3711c7298bbc9eb92",
    "e41b335e92b7c99473971eee7291655bcbe9e6a33e9a1b7277d3d846ca19d64d",
    "a84771208a02dca3d60e4eab598941225716eb166ce7028e8a7c073990b87c80",
    "da130f57069742c7891da137a0f6028cccc8ba3fc0458c27649526b51d79d1a1",
    "4ff7a8865dc9f00cb110307d3d86321d4b581a301a445209873d05461f42f142",
    "32bcfd9a050a2e6c5436d26d4cf3af14a5a247cd238d21105f36d29f51d4247d",
    "935372136d5e4a25b1690624807be0b98a6e7ed12b5b577ae8f22b6128460c94")
OLD_MODELS = tuple(V2 / f"MODEL_{line}_fold{fold}_rank.joblib" for line, _, _ in PAIRS for fold in range(5))
SOURCE_FILES = tuple(CACHE / f"nested_{fold}_{upstream}_Learned.parquet"
    for fold in range(5) for upstream in ("TxPert_GAT", "TxPert_Exphormer"))
PINS = dict(zip(map(str, OLD_MODELS), MODEL_SHAS)) | {
    str(COUNT): COUNT_SHA, str(V2 / "SCORE_FREEZE.json"): "17514b553c28d56030677d2d9cfd60127adebfb73cab41fb37440f9d96f6769c",
    str(V2 / "STATUS.json"): "be2f23c848c97c65ac8ed94fe15ca8aff1733f619ebcca0ce88cc241165bc520",
    str(V2 / "REGISTRATION.json"): "929fea471afd514c1b69bb95ad1d9991b75dcc262537fa59fae1026164f16757",
    str(HELPER): "46e4e48f0b767d4fc363da9884412973beee0407128377a184bc6992c0819c64"}
SCOPE = {"status": "LEAD_AUTHORIZED_INFORMATION_CONTROL", "Source_gene_axis": 2840,
    "directions": [p[0] for p in PAIRS], "gene_folds": list(range(5)), "new_fit_cap": 10,
    "reused_old_rank_models": 10, "seed": SEEDS[0], "features": P + PUBLIC,
    "training_target": "Source biological zero-effect RMSE = RMS(Source_TRUE_EFFECTS)",
    "CDF": "same training-only CDF_KEYS and midrank helper; no baseline CDF refit",
    "control_realized_error_training_labels": 0, "biological_supervision_retained": True,
    "cached_Source_prediction_inputs_retained": True, "query_truth_as_feature": False,
    "prediction_clip": True, "old_clipped_replay_tolerance": 1e-14,
    "paired_contrasts": [["BioNorm_HGB", "OldRank_HGB"], ["OldRank_HGB", "BioNorm_HGB"]],
    "bootstrap": "reuse exact existing 5000x575 Source counts, generation seed20261002; no RNG",
    "new_Public_fits": 0, "new_upstream_calls": 0, "GPU_hours": 0, "download_bytes": 0,
    "external_numeric_access": False, "formal_core_unchanged": True, "method_promotion": False,
    "CPU_threads": 1, "CPU_and_run_wall_cap_seconds": 600}
AUTHORIZATION = ("Root's explicit current task delegates this ONE necessary Source DEV information control "
    "under the user's current lead delegation. It authorizes 10 new control fits and fixed saved-count statistics, "
    "keeps the formal core and outcomes immutable, and is separate from the spent 20-fit objective-probe approval.")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f: f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inputs():
    return SOURCE_FILES + OLD_MODELS + (TRUTH, TASKS, GENES, COUNT, COMPARATOR_REG, HELPER,
        ARCHIVE, V2 / "SCORE_FREEZE.json", V2 / "STATUS.json", V2 / "REGISTRATION.json") + CORE + (Path(__file__).resolve(),)


def prepare():
    if D.exists() or O.exists(): raise RuntimeError("Fresh control report/runtime roots required")
    paths = inputs()
    if any(p.resolve() != p for p in paths): raise PermissionError("Input path substitution refused")
    bindings = {str(p): sha(p) for p in paths}
    if any(bindings[p] != expected for p, expected in PINS.items()): raise RuntimeError("Fixed reuse pin differs")
    old = json.loads((V2 / "REGISTRATION.json").read_text())["input_sha256_before"]
    status = json.loads((V2 / "STATUS.json").read_text())
    if status["status"] != "COMPLETE_DEV_DIAGNOSTIC" or status["all_inputs_unchanged"] is not True:
        raise RuntimeError("Original model diagnostic was not complete with immutable inputs")
    for p in SOURCE_FILES + (ARCHIVE,) + CORE:
        if bindings[str(p)] != old[str(p)]: raise RuntimeError("Old model input/core binding differs")
    registry = json.loads(COMPARATOR_REG.read_text())
    if registry["sorted_gene_order_sha256"] != GENE_SHA: raise RuntimeError("Existing Source gene order differs")
    D.mkdir(); O.mkdir()
    write_json(D / "CONFIG.json", {"scope": SCOPE, "authorization": AUTHORIZATION,
        "script_sha256": bindings[str(Path(__file__).resolve())], "input_sha256_before": bindings,
        "runtime": str(O), "report": str(D), "status": "READY_NOT_RUN"})
    (D / "READINESS.md").write_text("# Source biological label control: ready, not run\n\n" + AUTHORIZATION +
        "\n\nControl/comparator, not a negative null or new main method. SourceCommon2840 only; "
        "10 new fits, 10 pinned old models, exact 13 features/rows/weights/seed and fixed saved gene counts. "
        "No external numeric data or frozen-core change. Truth rows are selected before RMS calculation. "
        "All scores/models freeze before cached errors are parsed. Alignment/replay failure stops without relaxed tolerances.\n")
    print(json.dumps({"status": "READY_NOT_RUN", "script_sha256": bindings[str(Path(__file__).resolve())], "report": str(D)}))


def validate_config():
    config = json.loads((D / "CONFIG.json").read_text())
    if config["scope"] != SCOPE or config["script_sha256"] != sha(Path(__file__)):
        raise PermissionError("Fresh control scope/script binding differs")
    if (O / "STATUS.json").exists() or (O / "SCORE_FREEZE.json").exists():
        raise RuntimeError("A performed/failed/frozen control cannot be rerun")
    if {str(p): sha(p) for p in inputs()} != config["input_sha256_before"]:
        raise RuntimeError("A registered input changed")
    return config


def stat_counter():
    tree = ast.parse(HELPER.read_text())
    chosen = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"weighted_midrank", "counter_metrics"}]
    if len(chosen) != 2: raise RuntimeError("Pinned pure statistics helper differs")
    namespace = {"np": np, "METRICS": METRICS}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(HELPER), "exec"), namespace)
    return namespace["counter_metrics"]


def training_norms(truth, rows):
    selected = np.asarray(truth[rows])  # select allowed training rows BEFORE any RMS
    if selected.dtype != np.float32 or not np.isfinite(selected).all():
        raise RuntimeError("Exact finite float32 Source training effects required")
    return np.sqrt(np.mean(np.square(selected), axis=1))


def feature_frames(tasks):
    result = {}
    for path in SOURCE_FILES:
        frame = pd.read_parquet(path, columns=META + P + PUBLIC, use_threads=False).reset_index(drop=True)
        aligned = tasks.set_index("task_id").reindex(frame.task_id)
        if (len(frame) != 1808 or frame.task_id.duplicated().any() or aligned.isna().any().any()
                or not np.array_equal(frame[["gene", "target", "fold"]].to_numpy(), aligned[["gene", "target", "fold"]].to_numpy())
                or set(frame.output_contract_id) != {"E201_common2840gene_log1p_delta_v1"}
                or not np.isfinite(frame[P + PUBLIC].to_numpy(float)).all()):
            raise RuntimeError("Exact Source feature/task/axis identity failed")
        result[path] = frame
    return result


def source_archive():
    selected = []
    with gzip.open(ARCHIVE, "rt", newline="") as f:
        for row in csv.DictReader(f):
            if row["line"] not in {p[0] for p in PAIRS} or row["seed"] != str(SEEDS[0]) or row["method"] != "Learned_hgb":
                continue  # Source metadata BEFORE risk/error numeric conversion
            selected.append({**{k: row[k] for k in ["line", "task_id", "target", "gene", "upstream"]},
                "fold": int(row["fold"]), "risk": float(row["risk"]), "true_error_rmse": float(row["true_error_rmse"])})
    result = pd.DataFrame(selected)
    if len(result) != 3616 or result.duplicated(["line", "task_id"]).any(): raise RuntimeError("Source archive population differs")
    return result


def evaluate(records, counter):
    counts = np.load(COUNT, mmap_mode="r", allow_pickle=False)
    if counts.shape != (5000, 575) or counts.dtype != np.uint16 or not (counts.sum(axis=1) == 575).all():
        raise RuntimeError("Fixed Source count shape/row totals differ")
    points = np.full((2, 5, 4, 7), np.nan); draws = np.full((2, 5000, 5, 4, 7), np.nan)
    checks = []
    for li, (line, _, _) in enumerate(PAIRS):
        part = records[records.line.eq(line)]
        truth = part[part.method.eq("OldRank_HGB")][KEYS + ["true_error_rmse"]]
        wide = part.pivot(index=KEYS, columns="method", values="risk").reset_index().merge(truth, on=KEYS, validate="one_to_one")
        genes = sorted(wide.gene.astype(str).unique())
        if len(wide) != 1808 or len(genes) != 575 or sha_gene(genes) != GENE_SHA or set(wide.target) != set(CONTEXTS):
            raise RuntimeError("Fixed Source count mapping/population differs")
        gi = wide.gene.map({g: i for i, g in enumerate(genes)}).to_numpy(int)
        for ci, context in enumerate(CONTEXTS):
            ix = np.flatnonzero(wide.target.eq(context)); frame = wide.iloc[ix]
            weights = counts[:, gi[ix]].astype(np.int32)
            for mi, method in enumerate(METHODS):
                score = frame[method].to_numpy(float); measured = metrics(frame, score)
                point = np.array([measured[k] for k in METRICS])
                check = counter(frame.true_error_rmse, frame.task_id, score, np.ones((1, len(frame)), int))[0]
                if not np.allclose(point, check, rtol=0, atol=1e-12, equal_nan=True): raise RuntimeError("Counter/scalar metric validator differs")
                points[li, ci, mi] = point
                draws[li, :, ci, mi] = counter(frame.true_error_rmse, frame.task_id, score, weights,
                    original_valid=np.isfinite(point[0]))
                checks.append({"line": line, "context": context, "method": method,
                    "max_abs_counter_point_difference": float(np.nanmax(np.abs(point - check)))})
        points[li, 4] = np.mean(points[li, :4], axis=0)
        draws[li, :, 4] = np.mean(draws[li, :, :4], axis=1)
        print(json.dumps({"phase": "saved_count_statistics", "line": line}), flush=True)
    np.savez_compressed(O / "ALL_FIXED_SOURCE_METRIC_DRAWS.npz", draws=draws, points=points,
        lines=np.asarray([p[0] for p in PAIRS]), contexts=np.asarray(CONTEXTS + ["macro"]),
        methods=np.asarray(METHODS), metrics=np.asarray(METRICS))
    single, paired = [], []
    for li, (line, _, _) in enumerate(PAIRS):
        for ci, context in enumerate(CONTEXTS + ["macro"]):
            for mi, method in enumerate(METHODS):
                for ki, metric in enumerate(METRICS):
                    v = draws[li, :, ci, mi, ki]; finite = v[np.isfinite(v)]
                    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) >= 2 else (np.nan, np.nan)
                    single.append({"line": line, "context": context, "method": method, "metric": metric,
                        "point": points[li, ci, mi, ki], "ci95_lower": lo, "ci95_upper": hi, "valid_draws": len(finite), "saved_draws": 5000})
            for a, b in ((0, 1), (1, 0)):
                for ki, metric in enumerate(METRICS):
                    delta = draws[li, :, ci, a, ki] - draws[li, :, ci, b, ki]; finite = delta[np.isfinite(delta)]
                    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) >= 2 else (np.nan, np.nan)
                    paired.append({"line": line, "context": context, "method_a": METHODS[a], "method_b": METHODS[b], "metric": metric,
                        "difference_a_minus_b": points[li, ci, a, ki] - points[li, ci, b, ki],
                        "ci95_lower": lo, "ci95_upper": hi, "valid_draws": len(finite), "saved_draws": 5000,
                        "count_generation_seed": 20261002, "new_RNG": 0})
    pd.DataFrame(single).to_csv(D / "METHOD_METRICS.csv", index=False)
    pd.DataFrame(paired).to_csv(D / "PAIRED_METRICS.csv", index=False)
    pd.DataFrame(checks).to_csv(D / "SCALAR_COUNTER_REPRODUCTION.csv", index=False)
    return pd.DataFrame(paired)


def sha_gene(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def timeout(*_): raise TimeoutError("Fixed 600-second control budget exhausted")


def run():
    config = validate_config()  # checks fresh authorized scope before Source numeric reads/fits
    started, cpu = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, timeout); signal.alarm(600)
    signal.signal(signal.SIGXCPU, timeout); resource.setrlimit(resource.RLIMIT_CPU, (590, 600))
    fits, replay, cdf, model_paths, score_frames = [], [], [], [], []
    state, failure = "FAILED", None
    try:
        tasks = pd.read_csv(TASKS, dtype={k: str for k in ["task_id", "gene", "target", "condition"]})
        genes = json.loads(GENES.read_text()); truth = np.load(TRUTH, mmap_mode="r", allow_pickle=False)
        if (len(tasks) != 1808 or tasks.task_id.duplicated().any() or tasks.gene.nunique() != 575
                or set(tasks.target) != set(CONTEXTS) or not tasks.groupby("gene").fold.nunique().eq(1).all()
                or len(genes) != 2840 or len(set(genes)) != 2840 or truth.shape != (1808, 2840) or truth.dtype != np.float32):
            raise RuntimeError("Fixed Source task/order/axis metadata failed")
        write_json(O / "AXIS_TASK_BINDING.json", {"task_order_sha256": sha_gene(tasks.task_id),
            "gene_axis_order_sha256": sha_gene(genes), "sorted_perturbation_gene_sha256": sha_gene(sorted(tasks.gene.unique())),
            "Source_shape": list(truth.shape), "Source_dtype": str(truth.dtype), "Source_task_csv_sha256": sha(TASKS), "Source_gene_ids_sha256": sha(GENES)})
        if sha_gene(sorted(tasks.gene.unique())) != GENE_SHA: raise RuntimeError("Saved Source count gene order mismatch")
        rows_by_task = dict(zip(tasks.task_id, range(len(tasks))))
        frames = feature_frames(tasks)
        with threadpool_limits(limits=1):
            for line, source, query_upstream in PAIRS:
                for fold in range(5):
                    train = frames[CACHE / f"nested_{fold}_{source}_Learned.parquet"]
                    train = train[train.fold.ne(fold)].reset_index(drop=True)
                    query = frames[CACHE / f"nested_{fold}_{query_upstream}_Learned.parquet"]
                    query = query[query.fold.eq(fold)].reset_index(drop=True)
                    if set(train.gene) & set(query.gene): raise RuntimeError("Outer training/query genes overlap")
                    norm = training_norms(truth, np.array([rows_by_task[t] for t in train.task_id]))
                    labelled = train.copy(); labelled["true_error_rmse"] = norm
                    labels, audit = rank_labels(labelled, f"SourceBioNorm/{line}/outer{fold}")
                    if not np.isfinite(labels).all() or len(fits) >= 10: raise RuntimeError("Invalid biological CDF or fit cap")
                    cdf.extend({**r, "line": line, "fold": fold, "label_source": "Source biological zero-effect RMSE",
                        "realized_error_training_labels": 0, "biological_norm_sha256": hashlib.sha256(norm.tobytes()).hexdigest(),
                        "rank_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest()} for r in audit)
                    tick = time.perf_counter(); model = fit_risk(labelled, labels, P + PUBLIC, "hgb", SEEDS[0], weighted=True)
                    scores = {"BioNorm_HGB": model.predict(query, clip=True)}
                    old_path = V2 / f"MODEL_{line}_fold{fold}_rank.joblib"; old = joblib.load(old_path)
                    if old.columns != P + PUBLIC: raise RuntimeError("Pinned original model feature list differs")
                    scores["OldRank_HGB"] = old.predict(query, clip=True)
                    scores["Magnitude"] = query.predicted_magnitude.to_numpy(float)
                    scores["LearnedWeightedHistoryDistance"] = np.sqrt(query.prediction_prior_rmse.to_numpy(float)**2 + query.prior_uncertainty.to_numpy(float)**2)
                    if any(not np.isfinite(v).all() for v in scores.values()): raise RuntimeError("Nonfinite fixed score")
                    fits.append({"line": line, "fold": fold, "new_fits": 1, "reused_old_rank_models": 1,
                        "fit_and_prediction_seconds": time.perf_counter() - tick, "training_rows": len(train),
                        "training_genes": train.gene.nunique(), "biological_norm_training_labels": len(train),
                        "realized_error_training_labels": 0, "retained_Source_prediction_input_rows": len(train),
                        "training_gene_hash": ids_hash(train.gene), "query_gene_hash": ids_hash(query.gene),
                        "weight_sum": float(cluster_weights(train).sum()), "new_upstream_calls": 0})
                    new_path = O / f"MODEL_{line}_fold{fold}_BioNorm.joblib"; joblib.dump(model, new_path, compress=3)
                    copied = O / old_path.name; shutil.copyfile(old_path, copied)
                    if sha(copied) != PINS[str(old_path)]: raise RuntimeError("Copied original model changed")
                    model_paths.extend([new_path, copied])
                    for method, score in scores.items():
                        part = query[KEYS].copy(); part["line"], part["seed"], part["method"], part["risk"] = line, SEEDS[0], method, score
                        score_frames.append(part)
            if len(fits) != 10: raise RuntimeError("Incomplete ten-fit control")
            scores = pd.concat(score_frames, ignore_index=True); score_path = O / "SCORE_ONLY_PREDICTIONS.csv.gz"
            scores.to_csv(score_path, index=False)
            freeze = {"status": "ALL_SCORES_MODELS_FROZEN_BEFORE_REALIZED_ERRORS", "new_fits": 10,
                "reused_old_rank_models": 10, "script_sha256": config["script_sha256"],
                "score_only_sha256": sha(score_path), "model_sha256": {str(p): sha(p) for p in model_paths},
                "cached_realized_errors_parsed_before_freeze": 0, "query_truth_norm_as_feature": False}
            write_json(O / "SCORE_FREEZE.json", freeze); write_json(D / "MODEL_SCORE_HASH_RECEIPT.json", freeze)
            # Only now read Source actual-error columns and archive numeric values.
            archive = source_archive(); truth_records = []
            for line, _, query_upstream in PAIRS:
                for fold in range(5):
                    path = CACHE / f"nested_{fold}_{query_upstream}_Learned.parquet"
                    cached = pd.read_parquet(path, columns=["task_id", "fold", "true_error_rmse"], use_threads=False)
                    cached = cached[cached.fold.eq(fold)]
                    reference = archive[archive.line.eq(line) & archive.fold.eq(fold)]
                    old_score = scores[scores.line.eq(line) & scores.fold.eq(fold) & scores.method.eq("OldRank_HGB")]
                    paired = old_score.merge(reference, on=["line"] + KEYS, suffixes=("_new", "_old"), validate="one_to_one")
                    paired = paired.merge(cached[["task_id", "true_error_rmse"]].rename(columns={"true_error_rmse": "error_cache"}), on="task_id", validate="one_to_one")
                    if len(paired) != len(cached) or len(reference) != len(cached): raise RuntimeError("Exact Source replay population differs")
                    error_cache = paired.error_cache.to_numpy()
                    if error_cache.dtype != np.float32 or not np.array_equal(error_cache.view(np.uint32), paired.true_error_rmse.to_numpy().astype(np.float32).view(np.uint32)):
                        raise RuntimeError("Source archive error codec bits differ")
                    difference = float(np.max(np.abs(np.clip(paired.risk_new, 0, 1) - paired.risk_old)))
                    if difference > 1e-14: raise RuntimeError("Original rank model score replay failed")
                    replay.append({"line": line, "fold": fold, "rows": len(paired), "max_abs_score_difference": difference,
                        "tolerance": 1e-14, "error_codec": "exact restored float32 bits; archive decimal metrics"})
                    truth_records.append(reference[["line"] + KEYS + ["true_error_rmse"]])
            records = scores.merge(pd.concat(truth_records), on=["line"] + KEYS, validate="many_to_one")
            if len(records) != len(scores): raise RuntimeError("Complete score/truth join failed")
            records.to_csv(O / "TASK_PREDICTIONS.csv.gz", index=False)
            contexts, macros = summarize(records)
            contexts.to_csv(D / "CONTEXT_METRICS.csv", index=False); macros.to_csv(D / "LINE_MACRO_METRICS.csv", index=False)
            fold_metrics = [dict(zip(["line", "method", "fold", "target"], key)) | metrics(group, group.risk)
                for key, group in records.groupby(["line", "method", "fold", "target"], sort=True)]
            pd.DataFrame(fold_metrics).to_csv(D / "FOLD_CONTEXT_METRICS.csv", index=False)
            paired_metrics = evaluate(records, stat_counter())
            main = paired_metrics[paired_metrics.context.eq("macro") & paired_metrics.metric.eq("utility20") & paired_metrics.method_a.eq("BioNorm_HGB")]
            text = "# Source biological label control: actual result\n\n" + main.to_markdown(index=False) + "\n\n"
            text += "These are paired Source DEV/SEEN information-control estimates, using the fixed saved gene counts (generation seed20261002), not fresh confirmation. "
            text += "Prediction/Public inputs and biological supervision remain present. The contrast concerns error-specific training labels relative to this biological magnitude proxy. "
            text += "No formal core, external score, new family, root-cause or method-promotion claim follows. All planned contexts, directions and seven metrics are retained.\n"
            (D / "SUMMARY.md").write_text(text)
            state = "COMPLETE_SOURCE_BIOLOGY_LABEL_INFORMATION_CONTROL"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"; raise
    finally:
        signal.alarm(0)
        for name, rows in (("FIT_COSTS", fits), ("BIOLOGICAL_CDF_AUDIT", cdf), ("ORIGINAL_SCORE_REPLAY", replay)):
            pd.DataFrame(rows).to_csv(D / f"{name}.csv", index=False)
        after = {str(p): sha(p) for p in inputs()}; unchanged = after == config["input_sha256_before"]
        receipt = {"status": state if unchanged else "FAILED_INPUT_MUTATION", "failure": failure,
            "new_fits": len(fits), "new_fit_cap": 10, "reused_old_rank_models": len(fits),
            "control_realized_error_training_labels": 0, "biological_norm_training_labels": sum(x["training_rows"] for x in fits),
            "retained_Source_prediction_input_rows": sum(x["training_rows"] for x in fits),
            "new_upstream_calls": 0, "historical_Source_prediction_calls_retained": True,
            "new_RNG_draws": 0, "existing_count_seed": 20261002, "external_numeric_access": 0,
            "input_sha256_after": after, "all_inputs_unchanged": unchanged,
            "elapsed_seconds": time.monotonic() - started, "CPU_seconds": time.process_time() - cpu,
            "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "formal_core_unchanged": True, "method_promotion": False}
        write_json(O / "STATUS.json", receipt); write_json(D / "ACTUAL_STATUS.json", receipt)
        if not unchanged: raise RuntimeError("Original inputs/core changed")
    print(json.dumps({"status": state, "new_fits": len(fits), "report": str(D)}))


def self_check():
    seen = []
    class SyntheticTruth:
        def __getitem__(self, rows):
            if list(rows) != [0, 2]: raise AssertionError("Query truth row was selected")
            seen.extend(rows); return np.array([[3., 4.], [0., 6.]], dtype=np.float32)
    norm = training_norms(SyntheticTruth(), np.array([0, 2]))
    assert np.allclose(norm, [np.sqrt(12.5), np.sqrt(18)]) and seen == [0, 2]
    class ToyPrep:
        def transform(self, x): return x
    class ToyModel:
        def predict(self, x): return np.array([-.25, .5, 1.25])
    toy_model = FittedRisk(ToyPrep(), ToyModel(), ["synthetic"])
    assert SCOPE["prediction_clip"] is True and np.array_equal(
        toy_model.predict(pd.DataFrame({"synthetic": [1., 2., 3.]}), clip=True), [0., .5, 1.])
    counter = stat_counter(); n = 42; ids = np.array([f"toy{i:03}" for i in range(n)])
    errors = (np.arange(n) % 7 + 1) / 10; score = (np.arange(n) % 5) / 4
    weights = np.array([np.ones(n, int), (np.arange(n) % 3)], dtype=int)
    maximum = 0.
    for w, actual in zip(weights, counter(errors, ids, score, weights)):
        ix = np.repeat(np.arange(n), w); frame = pd.DataFrame({"task_id": ids[ix], "true_error_rmse": errors[ix]})
        expected = np.array([metrics(frame, score[ix])[k] for k in METRICS])
        assert np.allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True)
        maximum = max(maximum, float(np.nanmax(np.abs(actual - expected))))
    blocked = False
    with patch(__name__ + ".validate_config", side_effect=PermissionError("synthetic missing fresh scope")), \
            patch(__name__ + ".feature_frames", side_effect=AssertionError("Source access before scope")), \
            patch(__name__ + ".fit_risk", side_effect=AssertionError("Fit before scope")):
        try: run()
        except PermissionError: blocked = True
    assert blocked
    print(json.dumps({"status": "SELF_CHECK_PASS", "training_row_indexing_before_norm": True,
        "seven_count_metrics_match_expanded_scalar": True, "max_abs_counter_difference": maximum,
        "fresh_scope_rejection_before_data_or_fit": True, "actual_fits": 0, "new_RNG_draws": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true"); action.add_argument("--self-check", action="store_true"); action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.self_check: self_check()
    else: run()


if __name__ == "__main__": main()
