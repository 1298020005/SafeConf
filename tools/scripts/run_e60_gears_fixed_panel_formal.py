#!/usr/bin/env python3
"""E60: formal, fixed-task GEARS ensemble audit on Adamson.

This is the direct response to the advisor's question: *what predictor's
error is the risk score compared with?*  We train the real GEARS predictor on
the same 24 held-out Adamson gene perturbations for three random seeds.

Risk inputs are built only from the three GEARS prediction vectors:
  - ensemble disagreement (seed-to-ensemble RMSE),
  - ensemble predicted-effect magnitude.
The true held-out effect is read only after scoring to calculate ensemble RMSE.
Because every held-out gene has zero training support by construction, support
is recorded as a constant setting property and is not put into a rank score.

Usage:
  python tools/scripts/run_e60_gears_fixed_panel_formal.py --mode full

``full`` prepares a deterministic task panel, runs two seeds in parallel on
the two local GPUs, runs the third seed, then packages strict records and the
task-level audit.  Resume is safe: completed seed status JSON files are read
and skipped unless ``--rerun-complete`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from xml.sax.saxutils import escape

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code" / "20260426_154505_perturb_transport_final_push"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402


OUT = ROOT / "docs" / "实验结果" / "E60_gears_fixed_panel_formal_20260711"
TABLES, ARRAYS, REPORTS, FIGURES, RAW = (OUT / "tables", OUT / "arrays", OUT / "reports", OUT / "figures", OUT / "raw_gears")
DATASET = "adamson"
SOURCE_H5AD = Path("/home/yyf/data/singlecell_perturbation_atlas/official_generalization/Adamson.h5ad")
DATASET_H5AD = {
    "adamson": SOURCE_H5AD,
    "norman": Path("/home/yyf/data/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad"),
    "frangieh": Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/perturb_processed.h5ad"),
}
PYTHON = Path("/home/yyf/.conda/envs/scgpt_env/bin/python")
SEEDS = (11, 22, 33)
SEED_TO_DEVICE = {11: "cuda:0", 22: "cuda:1", 33: "cuda:0"}
CONTROL = {"control", "ctrl", "vehicle", "dmso", "nt", "non-targeting"}
MAX_TRAIN_CELLS_PER_CONDITION = 0
CONDITION_SAMPLING_SEED = 20260766
USE_UNCERTAINTY = False


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def ensure_dirs() -> None:
    for path in (TABLES, ARRAYS, REPORTS, FIGURES, RAW):
        path.mkdir(parents=True, exist_ok=True)


def configure(dataset: str, out_name: str | None) -> None:
    """Parameterize the fixed-task GEARS audit without changing E60 defaults.

    The Norman invocation is a pre-specified independent replication of the
    same protocol; it does not reuse Adamson tasks or select by error.
    """
    global DATASET, SOURCE_H5AD, OUT, TABLES, ARRAYS, REPORTS, FIGURES, RAW
    if dataset not in DATASET_H5AD:
        raise ValueError(f"unsupported fixed-task dataset: {dataset}")
    DATASET = dataset
    SOURCE_H5AD = DATASET_H5AD[dataset]
    if out_name is None:
        out_name = "E60_gears_fixed_panel_formal_20260711" if dataset == "adamson" else f"E66_{dataset}_gears_fixed_panel_formal_20260711"
    OUT = ROOT / "docs" / "实验结果" / out_name
    TABLES, ARRAYS, REPORTS, FIGURES, RAW = (OUT / "tables", OUT / "arrays", OUT / "reports", OUT / "figures", OUT / "raw_gears")


def manifest_path() -> Path:
    return TABLES / "E60_FIXED_TEST_PERTURBATIONS.csv"


def make_manifest(n_test: int, selection_seed: int, exclude_manifests: tuple[Path, ...] = ()) -> pd.DataFrame:
    """Select held-out genes using only condition labels and cell counts."""
    if manifest_path().exists():
        existing = pd.read_csv(manifest_path())
        if len(existing) != n_test:
            raise ValueError(f"Existing E60 manifest has {len(existing)} rows, requested {n_test}; use the existing frozen panel.")
        return existing
    if not SOURCE_H5AD.exists():
        raise FileNotFoundError(SOURCE_H5AD)
    adata = ad.read_h5ad(SOURCE_H5AD, backed="r")
    condition_column = "condition" if "condition" in adata.obs else "perturbation"
    if condition_column not in adata.obs:
        raise KeyError(f"{DATASET} data lacks a condition/perturbation column")
    counts = adata.obs[condition_column].astype(str).value_counts()
    excluded: set[str] = set()
    for path in exclude_manifests:
        prior = pd.read_csv(path)
        if "condition" in prior:
            excluded.update(prior["condition"].astype(str))
        elif "task_gene" in prior:
            excluded.update(f"{gene}+ctrl" for gene in prior["task_gene"].astype(str))
        else:
            raise ValueError(f"Exclusion manifest lacks condition/task_gene: {path}")
    eligible = sorted(
        name for name, count in counts.items()
        if name.strip().lower() not in CONTROL
        and int(count) >= 200
        and ("+" not in name or name.endswith("+ctrl"))
        and (name if name.endswith("+ctrl") else f"{name}+ctrl") not in excluded
    )
    if len(eligible) < n_test:
        raise ValueError(f"Only {len(eligible)} {DATASET} perturbations have >=200 cells; need {n_test}.")
    rng = np.random.default_rng(selection_seed)
    selected = sorted(map(str, rng.choice(np.asarray(eligible, dtype=object), size=n_test, replace=False)))
    result = pd.DataFrame({
        "task_gene": [x.replace("+ctrl", "") for x in selected],
        "condition": [x if x.endswith("+ctrl") else f"{x}+ctrl" for x in selected],
        "n_cells_in_source_h5ad": [int(counts.get(x, counts.get(x.replace("+ctrl", ""), 0))) for x in selected],
        "selection_rule": f"uniform random sample without replacement from single-gene conditions with >=200 cells after excluding {len(excluded)} frozen prior tasks; rng_seed={selection_seed}",
        "selection_uses_effect_or_error": False,
    })
    result.to_csv(manifest_path(), index=False)
    return result


def seed_root(seed: int) -> Path:
    return RAW / f"seed_{seed}"


def status_path(seed: int) -> Path:
    return seed_root(seed) / "E60_SEED_STATUS.json"


def run_seed(seed: int, epochs: int, hidden: int, device: str, rerun_complete: bool) -> dict:
    existing = status_path(seed)
    if existing.exists() and not rerun_complete:
        prior = json.loads(existing.read_text(encoding="utf-8"))
        if prior.get("returncode") == 0 and prior.get("gears_status", {}).get("status") == "ok":
            print(f"[E60] seed={seed}: reuse completed output", flush=True)
            return prior
    root = seed_root(seed)
    root.mkdir(parents=True, exist_ok=True)
    log = REPORTS / f"E60_GEARS_SEED{seed}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        str(PYTHON), "-m", "safetrans_confidence.cli.run_gears_prediction_records",
        "--dataset", DATASET, "--seed", str(seed), "--split", "single", "--run-type", "formal",
        "--epochs", str(epochs), "--hidden-size", str(hidden), "--decoder-hidden-size", "16",
            "--num-similar-genes", "10", "--batch-size", "32", "--test-batch-size", "64",
            "--max-cells-per-condition", str(MAX_TRAIN_CELLS_PER_CONDITION), "--condition-sampling-seed", str(CONDITION_SAMPLING_SEED),
            "--device", device, "--out-dir", str(root), "--test-perturbations-file", str(manifest_path()),
        "--fixed-test-deterministic-val",
    ]
    if USE_UNCERTAINTY:
        cmd.append("--uncertainty")
    print(f"[E60] seed={seed}, device={device}, epochs={epochs}: start {now()}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    gears_status_path = root / "GEARS_PREDICTION_RECORD_STATUS.json"
    gears_status: dict = {}
    if gears_status_path.exists():
        try:
            gears_status = json.loads(gears_status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            gears_status = {"status_read_error": repr(exc)}
    row = {
        "seed": seed, "device": device, "epochs": epochs, "hidden_size": hidden, "returncode": proc.returncode,
        "started_and_finished_at": now(), "command": cmd, "log": str(log.relative_to(ROOT)), "gears_status": gears_status,
    }
    existing.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[E60] seed={seed}: finish returncode={proc.returncode} {now()}", flush=True)
    return row


def run_training(epochs: int, hidden: int, rerun_complete: bool) -> list[dict]:
    # The two GPUs are independent; train two seeds concurrently, then the
    # third on cuda:0.  This avoids overlapping two GEARS runs on one device.
    first = [11, 22]
    procs: dict[int, subprocess.Popen] = {}
    info: dict[int, dict] = {}
    for seed in first:
        existing = status_path(seed)
        if existing.exists() and not rerun_complete:
            prior = json.loads(existing.read_text(encoding="utf-8"))
            if prior.get("returncode") == 0 and prior.get("gears_status", {}).get("status") == "ok":
                info[seed] = prior
                continue
        root = seed_root(seed); root.mkdir(parents=True, exist_ok=True)
        log = REPORTS / f"E60_GEARS_SEED{seed}.log"
        env = os.environ.copy(); env["PYTHONPATH"] = str(CODE_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [
            str(PYTHON), "-m", "safetrans_confidence.cli.run_gears_prediction_records",
            "--dataset", DATASET, "--seed", str(seed), "--split", "single", "--run-type", "formal",
            "--epochs", str(epochs), "--hidden-size", str(hidden), "--decoder-hidden-size", "16",
            "--num-similar-genes", "10", "--batch-size", "32", "--test-batch-size", "64",
            "--max-cells-per-condition", str(MAX_TRAIN_CELLS_PER_CONDITION), "--condition-sampling-seed", str(CONDITION_SAMPLING_SEED),
            "--device", SEED_TO_DEVICE[seed], "--out-dir", str(root), "--test-perturbations-file", str(manifest_path()),
            "--fixed-test-deterministic-val",
        ]
        if USE_UNCERTAINTY:
            cmd.append("--uncertainty")
        handle = log.open("w", encoding="utf-8")
        procs[seed] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        info[seed] = {"seed": seed, "device": SEED_TO_DEVICE[seed], "epochs": epochs, "hidden_size": hidden, "command": cmd, "log": str(log.relative_to(ROOT)), "_handle": handle}
        print(f"[E60] seed={seed}, device={SEED_TO_DEVICE[seed]}: parallel start {now()}", flush=True)
    for seed, proc in procs.items():
        ret = proc.wait(); info[seed]["_handle"].close(); info[seed].pop("_handle", None); info[seed]["returncode"] = ret; info[seed]["started_and_finished_at"] = now()
        p = seed_root(seed) / "GEARS_PREDICTION_RECORD_STATUS.json"
        info[seed]["gears_status"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        status_path(seed).write_text(json.dumps(info[seed], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[E60] seed={seed}: parallel finish returncode={ret} {now()}", flush=True)
    info[33] = run_seed(33, epochs, hidden, SEED_TO_DEVICE[33], rerun_complete)
    return [info[s] for s in SEEDS]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator > 1e-12 else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def top20_enrichment(score: np.ndarray, err: np.ndarray) -> tuple[int, float]:
    if len(score) < 5:
        return 0, float("nan")
    k = max(1, int(math.ceil(0.2 * len(score))))
    return k, float(err[np.argsort(-score, kind="stable")[:k]].mean() / err.mean())


def bootstrap_ci(score: np.ndarray, err: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    values = []
    for _ in range(n):
        ix = rng.integers(0, len(score), len(score))
        values.append(spearman(score[ix], err[ix]))
    vals = np.asarray(values, dtype=float); vals = vals[np.isfinite(vals)]
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975))) if len(vals) else (float("nan"), float("nan"))


def package(n_boot: int, expected_n_test: int) -> dict:
    manifests = make_manifest(expected_n_test, selection_seed=20260711)
    all_records, pred, truth = [], {}, {}
    run_statuses = []
    for seed in SEEDS:
        st = status_path(seed)
        if not st.exists():
            raise RuntimeError(f"Missing seed status: {st}")
        row = json.loads(st.read_text(encoding="utf-8")); run_statuses.append(row)
        if row.get("returncode") != 0 or row.get("gears_status", {}).get("status") != "ok":
            raise RuntimeError(f"GEARS seed {seed} did not finish successfully")
        root = seed_root(seed) / DATASET / f"seed_{seed}"
        rec = pd.read_csv(root / "tables" / "PREDICTION_RECORDS.csv")
        all_records.append(rec)
        with np.load(root / "arrays" / "gears_predicted_effects.npz") as f:
            pred.update({k: np.asarray(f[k], dtype=np.float32) for k in f.files})
        with np.load(root / "arrays" / "gears_true_effects.npz") as f:
            truth.update({k: np.asarray(f[k], dtype=np.float32) for k in f.files})
    records = pd.concat(all_records, ignore_index=True)
    records.to_csv(TABLES / "PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(ARRAYS / "gears_predicted_effects.npz", **pred)
    np.savez_compressed(ARRAYS / "gears_true_effects.npz", **truth)
    issues = validate_prediction_record_artifacts(OUT, records=records, strict=True)

    rows = []
    ensemble_by_pert: dict[str, np.ndarray] = {}
    truth_by_pert: dict[str, np.ndarray] = {}
    expected = set(manifests["condition"].astype(str))
    observed = set(records["perturbation"].astype(str))
    for pert, sub in records.groupby("perturbation", sort=True):
        sub = sub.sort_values("fold_id")
        vectors = [pred[k] for k in sub["predicted_effect_key"]]
        true_vectors = [truth[k] for k in sub["true_effect_key"]]
        ensemble = np.mean(np.stack(vectors), axis=0)
        reference_true = true_vectors[0]
        ensemble_by_pert[str(pert)] = ensemble
        truth_by_pert[str(pert)] = reference_true
        true_max = max(float(np.max(np.abs(a - reference_true))) for a in true_vectors)
        individual_rmse = [rmse(v, reference_true) for v in vectors]
        native_logvar = sub["gears_uncertainty_logvar_mean"].to_numpy(float)
        rows.append({
            "perturbation": pert, "n_seeds": int(len(sub)), "all_three_seeds_present": set(sub.fold_id.astype(int)) == set(SEEDS),
            "in_frozen_test_manifest": pert in expected, "n_cells_test": int(sub.n_cells.iloc[0]),
            "error_gears_ensemble_rmse": rmse(ensemble, reference_true),
            "error_seed_mean_rmse": float(np.mean(individual_rmse)), "error_seed_std_rmse": float(np.std(individual_rmse, ddof=0)),
            "risk_ensemble_disagreement": float(np.sqrt(np.mean(np.var(np.stack(vectors), axis=0)))),
            "risk_predicted_magnitude": float(np.linalg.norm(ensemble)),
            "risk_gears_native_logvar": float(np.nanmean(native_logvar)) if np.isfinite(native_logvar).any() else np.nan,
            "true_l2_diagnostic": float(np.linalg.norm(reference_true)), "true_max_abs_diff_across_seeds": true_max,
            "training_support_for_heldout_gene": 0,
        })
    task = pd.DataFrame(rows).sort_values("perturbation").reset_index(drop=True)
    task.to_csv(TABLES / "E60_TASK_RISK_TABLE.csv", index=False)
    extra_conditions = ",".join(sorted(observed - expected))
    split_check = pd.DataFrame({"condition": sorted(expected), "observed_in_all_records": [c in observed for c in sorted(expected)], "extra_conditions": [extra_conditions for _ in sorted(expected)]})
    split_check.to_csv(TABLES / "E60_FIXED_TEST_SPLIT_CHECK.csv", index=False)

    # Panel-level perturbation-specific diagnostic inspired by Systema.  It
    # checks whether each prediction is nearest to its own true perturbation
    # centroid, not just whether its average RMSE looks small.
    specific_rows = []
    for pert in sorted(ensemble_by_pert):
        similarities = {
            candidate: cosine(ensemble_by_pert[pert], truth_vec)
            for candidate, truth_vec in truth_by_pert.items()
        }
        closest = max(similarities, key=similarities.get)
        specific_rows.append({
            "perturbation": pert,
            "own_true_delta_cosine": similarities[pert],
            "nearest_true_centroid": closest,
            "nearest_true_delta_cosine": similarities[closest],
            "centroid_match_correct": closest == pert,
            "n_candidate_true_centroids": len(truth_by_pert),
            "metric_scope": "Systema_style_frozen_panel_diagnostic",
        })
    specific = pd.DataFrame(specific_rows)
    specific.to_csv(TABLES / "E60_PERTURBATION_SPECIFIC_EVAL.csv", index=False)

    rng = np.random.default_rng(20260760)
    score_rows = []
    error = task["error_gears_ensemble_rmse"].to_numpy(float)
    score_names = ["risk_ensemble_disagreement", "risk_predicted_magnitude"]
    if task["risk_gears_native_logvar"].notna().all():
        score_names.append("risk_gears_native_logvar")
    for score_name in score_names:
        value = task[score_name].to_numpy(float)
        low, high = bootstrap_ci(value, error, rng, n_boot)
        k, enrich = top20_enrichment(value, error)
        score_rows.append({"predictor_name": "GEARS_3seed_ensemble", "score_name": score_name, "score_deployable": True, "target_error": "error_gears_ensemble_rmse", "n_tasks": len(task), "spearman": spearman(value, error), "bootstrap_rho_ci95_low": low, "bootstrap_rho_ci95_high": high, "top20_k": k, "top20_error_enrichment": enrich})
    score_rows.append({"predictor_name": "GEARS_3seed_ensemble", "score_name": "true_l2_diagnostic", "score_deployable": False, "target_error": "error_gears_ensemble_rmse", "n_tasks": len(task), "spearman": spearman(task["true_l2_diagnostic"].to_numpy(float), error), "bootstrap_rho_ci95_low": np.nan, "bootstrap_rho_ci95_high": np.nan, "top20_k": 0, "top20_error_enrichment": np.nan})
    scores = pd.DataFrame(score_rows); scores.to_csv(TABLES / "E60_RISK_ERROR_SUMMARY.csv", index=False)
    write_svg(task)
    status = {"experiment": f"{DATASET}_GEARS_fixed_panel_formal", "generated_at": now(), "git_head": git_head(), "dataset": DATASET, "predictor": "GEARS", "heldout_setting": f"{expected_n_test} fixed unseen single-gene perturbations", "seeds": list(SEEDS), "n_task_records": int(len(records)), "n_unique_tasks": int(len(task)), "n_boot": n_boot, "risk_inputs": score_names, "gears_native_uncertainty_enabled": USE_UNCERTAINTY, "truth_used_in_risk": False, "training_support_constant": 0, "condition_graph_sampling": {"max_train_cells_per_condition": MAX_TRAIN_CELLS_PER_CONDITION, "sampling_seed": CONDITION_SAMPLING_SEED, "test_graphs_sampled": False}, "perturbation_specific_metric": "Systema_style_frozen_panel_centroid_accuracy", "perturbation_specific_centroid_accuracy": float(specific["centroid_match_correct"].mean()), "strict_issue_count": len(issues), "strict_issues": issues, "run_statuses": run_statuses}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(status, task, scores)
    return status


def write_svg(task: pd.DataFrame) -> None:
    width, height, left, top = 1100, 720, 110, 90
    x, y = task["risk_ensemble_disagreement"].to_numpy(float), task["error_gears_ensemble_rmse"].to_numpy(float)
    def scale(a: np.ndarray, lo: float, hi: float, pad: float = .08) -> np.ndarray:
        amin, amax = float(a.min()), float(a.max()); spread = max(amax-amin, 1e-9); amin -= spread*pad; amax += spread*pad
        return lo + (a-amin)/(amax-amin)*(hi-lo), amin, amax
    sx, xmin, xmax = scale(x, left, width-72); sy0, ymin, ymax = scale(y, height-85, top); sy = sy0
    points=[]
    for xx,yy,name in zip(sx,sy,task.perturbation.astype(str)):
        points.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6" fill="#167c80" opacity=".83"><title>{escape(name)}</title></circle>')
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="#ffffff"/>','<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#23313d}.t{font-size:27px;font-weight:700}.s{font-size:16px;fill:#5c6a75}.a{font-size:15px}.sm{font-size:12px;fill:#5c6a75}</style>',f'<text class="t" x="55" y="43">{DATASET.title()}｜真实 GEARS 误差与三 seed 分歧</text>',f'<text class="s" x="55" y="70">{DATASET.title()}，冻结未见单基因扰动；每点一个任务。横轴只用预测向量，纵轴只用于事后评估。</text>',f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-85}" stroke="#74818b"/>',f'<line x1="{left}" y1="{height-85}" x2="{width-72}" y2="{height-85}" stroke="#74818b"/>',f'<text class="a" x="{(left+width-72)/2:.1f}" y="{height-35}" text-anchor="middle">三 seed 预测分歧（risk input）</text>',f'<text class="a" transform="translate(30 {(top+height-85)/2:.1f}) rotate(-90)" text-anchor="middle">GEARS ensemble RMSE（评估目标）</text>',f'<text class="sm" x="{left}" y="{height-67}">{xmin:.4f}</text>',f'<text class="sm" x="{width-72}" y="{height-67}" text-anchor="end">{xmax:.4f}</text>',f'<text class="sm" x="{left-8}" y="{top+5}" text-anchor="end">{ymax:.4f}</text>',f'<text class="sm" x="{left-8}" y="{height-85}" text-anchor="end">{ymin:.4f}</text>',*points,'</svg>']
    (FIGURES / "F1_gears_disagreement_vs_error.svg").write_text("\n".join(svg), encoding="utf-8")


def write_report(status: dict, task: pd.DataFrame, scores: pd.DataFrame) -> None:
    def fmt(v: object) -> str:
        return f"{float(v):.3f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    lines=[
        f'# {DATASET.title()}｜GEARS 固定任务三 seed 正式审计','',
        '## 这轮直接回答老师的问题','',
        f"这里比较的是 **GEARS 三 seed ensemble 对 {DATASET.title()} 真实 held-out effect 的 RMSE**。每个任务先由三个独立训练的 GEARS 模型给出预测，再从预测向量得到分歧和预测幅度；读取真实效应只发生在最后计算 RMSE 时。",'',
        f"- 固定未见单基因任务：{status['n_unique_tasks']}", f"- GEARS PredictionRecord：{status['n_task_records']}", f"- 训练 seeds：{', '.join(map(str,status['seeds']))}", f"- strict 合同问题：{status['strict_issue_count']}", '- 训练支持度：所有 held-out gene 都是 0，因此不是本 setting 的可排序特征。','',
        '## 任务级结果','',
        '| task | GEARS ensemble RMSE | seed disagreement | predicted magnitude | seed RMSE SD |', '|---|---:|---:|---:|---:|'
    ]
    for _,r in task.sort_values('error_gears_ensemble_rmse',ascending=False).iterrows():
        lines.append(f"| {r['perturbation']} | {r['error_gears_ensemble_rmse']:.4f} | {r['risk_ensemble_disagreement']:.4f} | {r['risk_predicted_magnitude']:.4f} | {r['error_seed_std_rmse']:.4f} |")
    lines += ['', '## 分数对 GEARS 实际误差的关联','', '| score | 可部署 | ρ | bootstrap 95% CI | top20 高误差富集 |', '|---|---|---:|---:|---:|']
    for _,r in scores.iterrows():
        ci = '—' if not bool(r.score_deployable) else f"[{fmt(r.bootstrap_rho_ci95_low)}, {fmt(r.bootstrap_rho_ci95_high)}]"
        lines.append(f"| {r.score_name} | {'是' if r.score_deployable else '否（oracle）'} | {fmt(r.spearman)} | {ci} | {fmt(r.top20_error_enrichment)} |")
    lines += ['', '## 扰动特异性核查','', f"在 {status['n_unique_tasks']} 个冻结测试任务中，GEARS ensemble 的 panel-level centroid accuracy 为 {status['perturbation_specific_centroid_accuracy']:.3f}。计算方式：每个预测 effect 与所有测试任务的真实 effect centroid 都比较余弦相似度，看最近的 centroid 是否是它自己的扰动。它参考 Systema 的扰动特异性思想，但这里是固定面板诊断，不把它称为完整 Systema 复现。", '', '## 边界','', f"1. 这是一个真正模型输出的固定任务实验，但只覆盖 GEARS 和 {DATASET.title()}；不能代替 GEARS、scGPT、CPA 的统一对照。",'2. 所有测试基因都在训练中完全未见，支持度为常数 0；这个设置用于检验未见扰动与 seed 分歧，不能用来评价 support feature。','3. `true_l2_diagnostic` 只保留用于核查，不能作为部署时的风险分数。','', '## 文件','', '- 任务表：`tables/E60_TASK_RISK_TABLE.csv`','- 分数表：`tables/E60_RISK_ERROR_SUMMARY.csv`','- 扰动特异性表：`tables/E60_PERTURBATION_SPECIFIC_EVAL.csv`','- 图：`figures/F1_gears_disagreement_vs_error.svg`',f'- 原始每 seed 输出：`raw_gears/seed_*/{DATASET}/seed_*/`']
    (REPORTS / "E60_GEARS_FIXED_PANEL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(f'# {DATASET.title()} 固定 GEARS 审计\n\n先读 `reports/E60_GEARS_FIXED_PANEL_REPORT.md`。\n\n这是对“风险分数和哪个预测器误差相关”的直接实验：GEARS 在 {DATASET.title()} 固定未见基因扰动上的三 seed ensemble。\n',encoding='utf-8')


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['prepare','train','package','full'],default='full'); p.add_argument('--dataset',choices=sorted(DATASET_H5AD),default='adamson'); p.add_argument('--out-name',default=None); p.add_argument('--epochs',type=int,default=10); p.add_argument('--hidden-size',type=int,default=48); p.add_argument('--n-test',type=int,default=24); p.add_argument('--n-boot',type=int,default=2000); p.add_argument('--selection-seed',type=int,default=20260711); p.add_argument('--exclude-manifest',type=Path,action='append',default=[]); p.add_argument('--max-train-cells-per-condition',type=int,default=0); p.add_argument('--condition-sampling-seed',type=int,default=20260766); p.add_argument('--only-seed',type=int,choices=SEEDS,default=None); p.add_argument('--device',default=None); p.add_argument('--uncertainty',action='store_true'); p.add_argument('--rerun-complete',action='store_true'); args=p.parse_args()
    global MAX_TRAIN_CELLS_PER_CONDITION, CONDITION_SAMPLING_SEED, USE_UNCERTAINTY
    MAX_TRAIN_CELLS_PER_CONDITION = int(args.max_train_cells_per_condition)
    CONDITION_SAMPLING_SEED = int(args.condition_sampling_seed)
    USE_UNCERTAINTY = bool(args.uncertainty)
    configure(args.dataset,args.out_name); ensure_dirs(); make_manifest(args.n_test,selection_seed=args.selection_seed,exclude_manifests=tuple(args.exclude_manifest))
    if args.mode=='prepare': print(manifest_path()); return
    if args.mode in {'train','full'}:
        if args.only_seed is not None:
            if args.mode == 'full':
                raise ValueError('--only-seed must use --mode train; package after all three seed statuses exist')
            run_seed(args.only_seed,args.epochs,args.hidden_size,args.device or SEED_TO_DEVICE[args.only_seed],args.rerun_complete)
        else:
            run_training(args.epochs,args.hidden_size,args.rerun_complete)
    if args.mode in {'package','full'}:
        status=package(args.n_boot,args.n_test); print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
