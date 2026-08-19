#!/usr/bin/env python3
"""E31: GEARS fixed-test split smoke.

E30 showed that the existing E25 GEARS runs cannot support seed-ensemble
uncertainty because the tested perturbations differ across seeds.

E31 validates the engineering fix: the GEARS exporter now accepts a fixed test
perturbation manifest.  This script runs a small 1-epoch Adamson smoke with the
same seven perturbations used by E29 and checks that the exported
PredictionRecords are exactly those fixed test tasks.

The output is a smoke, not a performance benchmark.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(CODE_ROOT))

import pandas as pd

from safetrans_confidence.data.records import validate_prediction_record_artifacts


OUT_DIR = PROJECT_ROOT / "docs/实验结果/E31_gears_fixed_test_split_smoke_20260708"
PYTHON = Path("/home/yyf/.conda/envs/scgpt_env/bin/python")
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


def _write_manifest() -> Path:
    manifest = pd.DataFrame(
        {
            "task_gene": TEST_GENES,
            "condition": [gene + "+ctrl" for gene in TEST_GENES],
            "reason": "E29 shared Adamson fixed-test smoke panel",
        }
    )
    path = OUT_DIR / "tables/E31_FIXED_TEST_PERTURBATIONS.csv"
    manifest.to_csv(path, index=False)
    return path


def _run_gears(manifest_path: Path) -> dict[str, Any]:
    raw_out = OUT_DIR / "raw_gears"
    log_path = OUT_DIR / "reports/E31_GEARS_FIXED_TEST_SPLIT_SMOKE_RUN.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        str(PYTHON),
        "-m",
        "safetrans_confidence.cli.run_gears_prediction_records",
        "--dataset",
        "adamson",
        "--seed",
        "1",
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
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    status_path = raw_out / "GEARS_PREDICTION_RECORD_STATUS.json"
    status: dict[str, Any] = {
        "returncode": proc.returncode,
        "command": cmd,
        "log": os.path.relpath(log_path, PROJECT_ROOT),
    }
    if status_path.exists():
        try:
            status["gears_status"] = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status["gears_status_read_error"] = repr(exc)
    return status


def _promote_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    seed_root = OUT_DIR / "raw_gears/adamson/seed_1"
    src_tables = seed_root / "tables"
    src_arrays = seed_root / "arrays"
    records = pd.read_csv(src_tables / "PREDICTION_RECORDS.csv")
    shutil.copy2(src_tables / "PREDICTION_RECORDS.csv", OUT_DIR / "tables/PREDICTION_RECORDS.csv")
    shutil.copy2(src_arrays / "gears_predicted_effects.npz", OUT_DIR / "arrays/gears_predicted_effects.npz")
    shutil.copy2(src_arrays / "gears_true_effects.npz", OUT_DIR / "arrays/gears_true_effects.npz")
    expected = pd.DataFrame({"condition": [gene + "+ctrl" for gene in TEST_GENES]})
    observed = pd.DataFrame({"condition": sorted(records["perturbation"].astype(str).unique())})
    check = expected.merge(observed.assign(observed=True), on="condition", how="left")
    check["observed"] = check["observed"].fillna(False).astype(bool)
    extra = sorted(set(observed["condition"]) - set(expected["condition"]))
    check["extra_conditions_present"] = ",".join(extra)
    check.to_csv(OUT_DIR / "tables/E31_FIXED_TEST_SPLIT_CHECK.csv", index=False)
    issues = validate_prediction_record_artifacts(OUT_DIR, records=records, strict=True)
    return records, check, issues


def _write_report(status: dict[str, Any], records: pd.DataFrame | None, check: pd.DataFrame | None, issues: list[str]) -> None:
    gears_status = status.get("gears_status", {})
    ok = (
        status.get("returncode") == 0
        and isinstance(gears_status, dict)
        and gears_status.get("status") == "ok"
        and not issues
        and check is not None
        and bool(check["observed"].all())
    )
    summary = pd.DataFrame(
        [
            {
                "status": "ok" if ok else "has_issues",
                "returncode": status.get("returncode"),
                "gears_status": gears_status.get("status") if isinstance(gears_status, dict) else "missing",
                "n_prediction_records": 0 if records is None else int(len(records)),
                "n_expected_conditions": len(TEST_GENES),
                "all_expected_observed": False if check is None else bool(check["observed"].all()),
                "strict_issue_count": len(issues),
                "strict_issues": "; ".join(issues),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "tables/E31_RUN_SUMMARY.csv", index=False)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>E31 GEARS fixed-test split smoke</title>
<style>
body{{margin:0;background:#fff;color:#1f2933;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:42px 28px 76px}}h1{{font-size:30px;margin:0 0 10px}}h2{{font-size:21px;margin:34px 0 12px;border-top:1px solid #e5e7eb;padding-top:24px}}
p{{line-height:1.75;font-size:16px}}.note{{border-left:4px solid #315C9B;background:#f8fbff;padding:12px 16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 22px}}th,td{{border-bottom:1px solid #e5e7eb;text-align:left;padding:8px 9px;vertical-align:top}}th{{background:#f7f7f7}}
</style></head><body><main>
<h1>E31 GEARS fixed-test split smoke</h1>
<p class="note">这个 smoke 只验证 GEARS exporter 能固定 test perturbation 清单；1 epoch 结果不能当性能。</p>
<h2>Run summary</h2>{summary.to_html(index=False, escape=False)}
<h2>Fixed test check</h2>{'' if check is None else check.to_html(index=False, escape=False)}
<h2>PredictionRecords</h2>{'' if records is None else records.to_html(index=False, escape=False)}
</main></body></html>
"""
    (OUT_DIR / "reports/E31_GEARS_FIXED_TEST_SPLIT_SMOKE.html").write_text(page, encoding="utf-8")
    md = f"""# E31 GEARS fixed-test split smoke

生成时间：{_now()}

## 结论

E31 验证 GEARS exporter 的 `--test-perturbations-file` 能把 Adamson test split 固定到 E29 的 7 个任务。

- GEARS returncode：{status.get('returncode')}
- GEARS status：{gears_status.get('status') if isinstance(gears_status, dict) else 'missing'}
- PredictionRecords：{0 if records is None else len(records)}
- strict issue_count：{len(issues)}
- all expected observed：{False if check is None else bool(check['observed'].all())}

边界：这是 1-epoch smoke，不是 GEARS 性能，也不是 seed-uncertainty 正式结果。它只证明 E32/E33 可以在固定任务上做三 seed 重跑。
"""
    (OUT_DIR / "reports/E31_GEARS_FIXED_TEST_SPLIT_SMOKE_REPORT.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "README_先看这个.md").write_text(
        """# E31 GEARS fixed-test split smoke

先看结论：GEARS exporter 已能通过 `--test-perturbations-file` 固定 Adamson test perturbations。E31 使用 E29 的 7 个任务跑 1 epoch smoke，并导出 strict PredictionRecord。

这不是性能 benchmark。它是为后续固定任务三 seed 重跑铺路。
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for rel in ["tables", "arrays", "reports"]:
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest()
    status = _run_gears(manifest_path)
    records = None
    check = None
    issues: list[str] = []
    gears_status = status.get("gears_status", {})
    if status.get("returncode") == 0 and isinstance(gears_status, dict) and gears_status.get("status") == "ok":
        records, check, issues = _promote_and_validate()
    run_status = {
        "status": "ok"
        if (
            status.get("returncode") == 0
            and isinstance(gears_status, dict)
            and gears_status.get("status") == "ok"
            and not issues
            and check is not None
            and bool(check["observed"].all())
        )
        else "has_issues",
        "generated_at": _now(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "out": os.path.relpath(OUT_DIR, PROJECT_ROOT),
        "raw_status": status,
        "strict_issue_count": len(issues),
        "strict_issues": issues,
        "expected_conditions": [gene + "+ctrl" for gene in TEST_GENES],
        "n_prediction_records": 0 if records is None else int(len(records)),
    }
    (OUT_DIR / "RUN_STATUS.json").write_text(
        json.dumps(run_status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(status, records, check, issues)
    print(json.dumps(run_status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
