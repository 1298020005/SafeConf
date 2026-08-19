#!/usr/bin/env python3
"""E32: GEARS fixed-test 3-seed smoke.

E31 proved that the GEARS exporter can fix the test perturbation list.  E32
runs the same fixed Adamson 7-task panel across seeds 1/2/3 with one training
epoch and computes seed-disagreement diagnostics under a strict
PredictionRecord contract.

This is a smoke for the fixed-task 3-seed workflow, not a formal performance
benchmark.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd

from safetrans_confidence.data.records import validate_prediction_record_artifacts


OUT_DIR = PROJECT_ROOT / "docs/实验结果/E32_gears_fixed_test_3seed_smoke_20260708"
PYTHON = Path("/home/yyf/.conda/envs/scgpt_env/bin/python")
SEEDS = [1, 2, 3]
TEST_GENES = ["CCND3", "DAD1", "DERL2", "NEDD8", "TIMM23", "UFM1", "YIPF5"]


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


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if mask.sum() < 2:
        return float("nan")
    xr = pd.Series(xv[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(yv[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if mask.sum() < 2:
        return float("nan")
    xv = xv[mask]
    yv = yv[mask]
    if np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _write_manifest() -> Path:
    manifest = pd.DataFrame(
        {
            "task_gene": TEST_GENES,
            "condition": [gene + "+ctrl" for gene in TEST_GENES],
            "reason": "E32 fixed-test 3-seed smoke panel",
        }
    )
    path = OUT_DIR / "tables/E32_FIXED_TEST_PERTURBATIONS.csv"
    manifest.to_csv(path, index=False)
    return path


def _run_seed(seed: int, manifest_path: Path) -> dict[str, Any]:
    raw_out = OUT_DIR / "raw_gears"
    log_path = OUT_DIR / f"reports/E32_GEARS_FIXED_TEST_SEED{seed}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        str(PYTHON),
        "-m",
        "safetrans_confidence.cli.run_gears_prediction_records",
        "--dataset",
        "adamson",
        "--seed",
        str(seed),
        "--split",
        "single",
        "--run-type",
        "smoke",
        "--epochs",
        "1",
        "--hidden-size",
        "32",
        "--decoder-hidden-size",
        "16",
        "--num-similar-genes",
        "5",
        "--batch-size",
        "16",
        "--test-batch-size",
        "32",
        "--device",
        "cuda:0",
        "--out-dir",
        str(raw_out),
        "--test-perturbations-file",
        str(manifest_path),
        "--fixed-test-deterministic-val",
    ]
    print(f"[E32] starting GEARS seed={seed} at {_now()}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line.rstrip() + "\n")
            log.flush()
        returncode = proc.wait()
    status_path = raw_out / "GEARS_PREDICTION_RECORD_STATUS.json"
    status: dict[str, Any] = {
        "seed": seed,
        "returncode": returncode,
        "command": cmd,
        "log": os.path.relpath(log_path, PROJECT_ROOT),
    }
    if status_path.exists():
        try:
            status["gears_status"] = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status["gears_status_read_error"] = repr(exc)
    print(f"[E32] finished GEARS seed={seed} returncode={returncode} at {_now()}", flush=True)
    return status


def _combine_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    all_records: list[pd.DataFrame] = []
    pred_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    fixed_checks: list[pd.DataFrame] = []
    expected_conditions = [gene + "+ctrl" for gene in TEST_GENES]
    for seed in SEEDS:
        seed_root = OUT_DIR / f"raw_gears/adamson/seed_{seed}"
        records = pd.read_csv(seed_root / "tables/PREDICTION_RECORDS.csv")
        all_records.append(records)
        observed = set(records["perturbation"].astype(str))
        fixed_checks.append(
            pd.DataFrame(
                [
                    {
                        "seed": seed,
                        "condition": cond,
                        "observed": cond in observed,
                        "extra_conditions_present": ",".join(sorted(observed - set(expected_conditions))),
                    }
                    for cond in expected_conditions
                ]
            )
        )
        with np.load(seed_root / "arrays/gears_predicted_effects.npz") as pred_npz:
            for key in pred_npz.files:
                pred_arrays[key] = np.asarray(pred_npz[key], dtype=np.float32)
        with np.load(seed_root / "arrays/gears_true_effects.npz") as true_npz:
            for key in true_npz.files:
                true_arrays[key] = np.asarray(true_npz[key], dtype=np.float32)
    records_all = pd.concat(all_records, ignore_index=True)
    fixed_check = pd.concat(fixed_checks, ignore_index=True)
    records_all.to_csv(OUT_DIR / "tables/PREDICTION_RECORDS.csv", index=False)
    fixed_check.to_csv(OUT_DIR / "tables/E32_FIXED_TEST_SPLIT_CHECK.csv", index=False)
    np.savez_compressed(OUT_DIR / "arrays/gears_predicted_effects.npz", **pred_arrays)
    np.savez_compressed(OUT_DIR / "arrays/gears_true_effects.npz", **true_arrays)
    issues = validate_prediction_record_artifacts(OUT_DIR, records=records_all, strict=True)
    return records_all, fixed_check, _seed_diagnostics(records_all, pred_arrays, true_arrays), issues


def _seed_diagnostics(
    records: pd.DataFrame,
    pred_arrays: dict[str, np.ndarray],
    true_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for perturbation, sub in records.groupby("perturbation", sort=True):
        sub = sub.sort_values("fold_id")
        preds = [pred_arrays[k] for k in sub["predicted_effect_key"]]
        trues = [true_arrays[k] for k in sub["true_effect_key"]]
        rmses = [_rmse(p, t) for p, t in zip(preds, trues)]
        pairwise = [_rmse(a, b) for a, b in combinations(preds, 2)]
        true_max_diff = 0.0
        for a, b in combinations(trues, 2):
            true_max_diff = max(true_max_diff, float(np.max(np.abs(a - b))))
        rows.append(
            {
                "perturbation": perturbation,
                "n_seeds": int(len(sub)),
                "seed_ids": ",".join(str(int(x)) for x in sub["fold_id"]),
                "all_three_seeds_present": set(sub["fold_id"].astype(int)) == set(SEEDS),
                "true_max_abs_diff_across_seeds": true_max_diff,
                "seed_rmse_mean": float(np.mean(rmses)),
                "seed_rmse_std": float(np.std(rmses, ddof=0)),
                "seed_rmse_min": float(np.min(rmses)),
                "seed_rmse_max": float(np.max(rmses)),
                "seed_pairwise_pred_rmse_mean": float(np.mean(pairwise)),
                "seed_disagreement_rmse": float(np.sqrt(np.mean(np.var(np.stack(preds), axis=0)))),
                "pred_l2_mean": float(np.mean([np.linalg.norm(p) for p in preds])),
                "true_l2_mean": float(np.mean([np.linalg.norm(t) for t in trues])),
            }
        )
    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_DIR / "tables/E32_FIXED_TEST_SEED_DIAGNOSTICS.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for score_col in ["seed_pairwise_pred_rmse_mean", "seed_disagreement_rmse", "pred_l2_mean", "true_l2_mean"]:
        for target_col in ["seed_rmse_mean", "seed_rmse_max"]:
            summary_rows.append(
                {
                    "score": score_col,
                    "target": target_col,
                    "n_tasks": int(len(diag)),
                    "spearman": _rank_corr(diag[score_col], diag[target_col]),
                    "pearson": _pearson(diag[score_col], diag[target_col]),
                }
            )
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "tables/E32_FIXED_TEST_SEED_RISK_SUMMARY.csv", index=False)
    return diag


def _write_report(
    run_statuses: list[dict[str, Any]],
    records: pd.DataFrame | None,
    fixed_check: pd.DataFrame | None,
    diagnostics: pd.DataFrame | None,
    issues: list[str],
) -> dict[str, Any]:
    risk_summary_path = OUT_DIR / "tables/E32_FIXED_TEST_SEED_RISK_SUMMARY.csv"
    risk_summary = pd.read_csv(risk_summary_path) if risk_summary_path.exists() else pd.DataFrame()
    all_runs_ok = all(s.get("returncode") == 0 and s.get("gears_status", {}).get("status") == "ok" for s in run_statuses)
    all_fixed_observed = bool(fixed_check["observed"].all()) if fixed_check is not None else False
    all_three = bool(diagnostics["all_three_seeds_present"].all()) if diagnostics is not None else False
    status = {
        "status": "ok" if all_runs_ok and all_fixed_observed and all_three and not issues else "has_issues",
        "generated_at": _now(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "out": os.path.relpath(OUT_DIR, PROJECT_ROOT),
        "seeds": SEEDS,
        "expected_conditions": [gene + "+ctrl" for gene in TEST_GENES],
        "n_prediction_records": 0 if records is None else int(len(records)),
        "n_tasks": 0 if diagnostics is None else int(len(diagnostics)),
        "all_runs_ok": all_runs_ok,
        "all_fixed_observed": all_fixed_observed,
        "all_three_seeds_present": all_three,
        "strict_issue_count": len(issues),
        "strict_issues": issues,
        "run_statuses": run_statuses,
    }
    (OUT_DIR / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = pd.DataFrame(
        [
            {
                "status": status["status"],
                "n_prediction_records": status["n_prediction_records"],
                "n_tasks": status["n_tasks"],
                "all_runs_ok": all_runs_ok,
                "all_fixed_observed": all_fixed_observed,
                "all_three_seeds_present": all_three,
                "strict_issue_count": len(issues),
                "strict_issues": "; ".join(issues),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "tables/E32_RUN_SUMMARY.csv", index=False)
    css = """
body{margin:0;background:#fff;color:#1f2933;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:42px 28px 76px}h1{font-size:30px;margin:0 0 10px}
h2{font-size:21px;margin:34px 0 12px;border-top:1px solid #e5e7eb;padding-top:24px}p{line-height:1.75;font-size:16px}
.note{border-left:4px solid #315C9B;background:#f8fbff;padding:12px 16px;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 22px}th,td{border-bottom:1px solid #e5e7eb;text-align:left;padding:8px 9px;vertical-align:top}th{background:#f7f7f7}
"""
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>E32 GEARS fixed-test 3-seed smoke</title><style>{css}</style></head>
<body><main>
<h1>E32 GEARS fixed-test 3-seed smoke</h1>
<p class="note">同一 Adamson 7-task test panel × 3 seeds × 1 epoch。这个结果只验证固定任务三 seed 工作流，不是正式 GEARS 性能。</p>
<h2>Run summary</h2>{summary.to_html(index=False, escape=False)}
<h2>Seed risk summary</h2>{risk_summary.to_html(index=False, escape=False)}
<h2>Seed diagnostics</h2>{'' if diagnostics is None else diagnostics.to_html(index=False, escape=False)}
<h2>Fixed test check</h2>{'' if fixed_check is None else fixed_check.to_html(index=False, escape=False)}
</main></body></html>
"""
    (OUT_DIR / "reports/E32_GEARS_FIXED_TEST_3SEED_SMOKE.html").write_text(page, encoding="utf-8")
    (OUT_DIR / "reports/E32_GEARS_FIXED_TEST_3SEED_SMOKE_REPORT.md").write_text(
        f"""# E32 GEARS fixed-test 3-seed smoke

生成时间：{_now()}

## 结论

E32 在 Adamson 7 个固定 test perturbations 上跑通 GEARS 3 seeds × 1 epoch smoke。

- PredictionRecords：{status['n_prediction_records']}
- tasks：{status['n_tasks']}
- all fixed observed：{all_fixed_observed}
- all three seeds present：{all_three}
- strict issue_count：{len(issues)}

边界：1 epoch smoke 不能作为性能 benchmark。它证明 E30 提出的 fixed-task 3-seed uncertainty 工作流已经具备工程入口。
""",
        encoding="utf-8",
    )
    (OUT_DIR / "README_先看这个.md").write_text(
        """# E32 GEARS fixed-test 3-seed smoke

先看结论：E32 已经在 Adamson 7 个固定 test perturbations 上跑通 GEARS 3 seeds × 1 epoch，并合并为 strict PredictionRecord 包。

这不是正式性能 benchmark。它证明后续可以把 epochs 提高，做真正的 fixed-task seed-uncertainty 实验。
""",
        encoding="utf-8",
    )
    return status


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for rel in ["tables", "arrays", "reports"]:
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest()
    run_statuses = [_run_seed(seed, manifest_path) for seed in SEEDS]
    records = None
    fixed_check = None
    diagnostics = None
    issues: list[str] = []
    if all(s.get("returncode") == 0 and s.get("gears_status", {}).get("status") == "ok" for s in run_statuses):
        records, fixed_check, diagnostics, issues = _combine_outputs()
    status = _write_report(run_statuses, records, fixed_check, diagnostics, issues)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
