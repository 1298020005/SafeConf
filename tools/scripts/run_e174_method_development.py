#!/usr/bin/env python3
"""Freeze E174 base error estimators using only already unsealed E173 data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from typing import Any

import numpy as np
import pandas as pd

from e174_conformal_common import (
    MODEL_SPECS,
    add_pair_columns,
    apply_cluster_upper,
    calibrate_cluster_upper,
    fit_ridge,
    identity_split,
    predict_ridge,
    target_key,
)


ROOT = Path(__file__).resolve().parents[2]
E173 = ROOT / "docs/实验结果/E173_falsification_aware_pair_certificate_20260719"
E174 = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719"
OUT = E174 / "method_development"
SCRIPT = Path(__file__).resolve()
COMMON = ROOT / "tools/scripts/e174_conformal_common.py"
DEVELOPMENT_TABLE = E173 / "tables/E173_TASK_OBJECTIVES.csv"
REPEATS = 20
COVERAGE = 0.90
MIN_RELATIVE_MEAN_UPPER_IMPROVEMENT = 0.005
MIN_REPEAT_WIN_FRACTION = 0.75
OUTCOMES = ("ensemble_rmse", "pair_mean_rmse")


class IntegrityFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode().strip()


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{head}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"uncommitted method input: {relative}") from exc
    if committed != path.read_bytes():
        raise IntegrityFailure(f"working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_dual_remote(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityFailure("method development requires a named branch")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git("fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False)
        if result.returncode:
            raise IntegrityFailure(f"cannot verify method code on {remote}")
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", head, remote_head, check=False).returncode:
            raise IntegrityFailure(f"current HEAD absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return branch, remote_heads


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def target_simultaneous_coverage(
    frame: pd.DataFrame, outcome: str, upper: np.ndarray
) -> float:
    work = frame[["panel_id", "perturbed_gene_id", outcome]].copy()
    work["covered"] = frame[outcome].to_numpy(float) <= np.asarray(upper, dtype=float) + 1e-12
    work["target_key"] = target_key(frame).to_numpy()
    return float(work.groupby("target_key").covered.all().mean())


def cross_validate(development: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat in range(REPEATS):
        split = identity_split(development, f"E174_METHOD_DEVELOPMENT_REPEAT_{repeat:02d}")
        work = development.copy()
        work["development_split"] = target_key(work).map(split)
        train = work.loc[work.development_split.eq("train")].copy()
        calibration = work.loc[work.development_split.eq("calibration")].copy()
        evaluation = work.loc[work.development_split.eq("evaluation")].copy()
        if [target_key(part).nunique() for part in (train, calibration, evaluation)] != [600, 200, 200]:
            raise IntegrityFailure("development 60/20/20 target split changed")
        for outcome in OUTCOMES:
            for spec in MODEL_SPECS:
                model = fit_ridge(train, outcome, spec)
                calibration_base = predict_ridge(calibration, model)
                calibration_rule = calibrate_cluster_upper(
                    calibration, calibration_base, outcome, COVERAGE
                )
                evaluation_base = predict_ridge(evaluation, model)
                upper = apply_cluster_upper(evaluation, evaluation_base, calibration_rule)
                lower = (
                    evaluation.pair_lower_bound_rmse.to_numpy(float)
                    if outcome == "pair_mean_rmse"
                    else np.zeros(len(evaluation), dtype=float)
                )
                rows.append(
                    {
                        "repeat": repeat,
                        "outcome": outcome,
                        "model_spec": spec,
                        "n_train_targets": target_key(train).nunique(),
                        "n_calibration_targets": target_key(calibration).nunique(),
                        "n_evaluation_targets": target_key(evaluation).nunique(),
                        "target_simultaneous_coverage": target_simultaneous_coverage(
                            evaluation, outcome, upper
                        ),
                        "mean_upper": float(np.mean(upper)),
                        "median_upper": float(np.median(upper)),
                        "mean_interval_width": float(np.mean(upper - lower)),
                        "calibration_quantile": float(calibration_rule["quantile"]),
                        "mean_absolute_base_error": float(
                            np.mean(np.abs(evaluation[outcome].to_numpy(float) - evaluation_base))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def decide_models(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        block = cv.loc[cv.outcome.eq(outcome)]
        pivot = block.pivot(index="repeat", columns="model_spec", values="mean_upper")
        magnitude = pivot["magnitude"]
        composite = pivot["magnitude_plus_pair_lower"]
        relative = float((magnitude.mean() - composite.mean()) / magnitude.mean())
        wins = float((composite < magnitude).mean())
        passed = relative >= MIN_RELATIVE_MEAN_UPPER_IMPROVEMENT and wins >= MIN_REPEAT_WIN_FRACTION
        rows.append(
            {
                "outcome": outcome,
                "candidate": "magnitude_plus_pair_lower",
                "fallback": "magnitude",
                "candidate_mean_upper": float(composite.mean()),
                "fallback_mean_upper": float(magnitude.mean()),
                "relative_mean_upper_improvement": relative,
                "candidate_repeat_win_fraction": wins,
                "required_relative_improvement": MIN_RELATIVE_MEAN_UPPER_IMPROVEMENT,
                "required_repeat_win_fraction": MIN_REPEAT_WIN_FRACTION,
                "incremental_gate_passed": passed,
                "selected_model_spec": "magnitude_plus_pair_lower" if passed else "magnitude",
            }
        )
    return pd.DataFrame(rows)


def write_manifest() -> str:
    files = sorted(
        path for path in OUT.rglob("*") if path.is_file() and path.name not in {"MANIFEST.sha256", "RUN_STATUS.json"}
    )
    text = "".join(
        f"{sha256_file(path)}  {path.relative_to(OUT).as_posix()}\n" for path in files
    )
    atomic_bytes(OUT / "MANIFEST.sha256", text.encode())
    return sha256_file(OUT / "MANIFEST.sha256")


def main() -> None:
    if OUT.exists():
        raise IntegrityFailure(f"append-only method output exists: {OUT}")
    e174_status = json.loads((E174 / "RUN_STATUS.json").read_text())
    if (
        e174_status.get("status") != "PASS"
        or e174_status.get("all_expression_x_values_read") != 0
        or e174_status.get("e174_calibration_targeting_x_values_read") != 0
        or e174_status.get("e174_evaluation_targeting_x_values_read") != 0
    ):
        raise IntegrityFailure("E174 no-X metadata boundary changed")
    e173_status = json.loads((E173 / "RUN_STATUS.json").read_text())
    if sha256_file(E173 / "MANIFEST.sha256") != e173_status.get("manifest_sha256"):
        raise IntegrityFailure("E173 manifest changed")

    head = git_text("rev-parse", "HEAD")
    branch, remote_heads = verify_dual_remote(head)
    input_paths = [
        SCRIPT,
        COMMON,
        DEVELOPMENT_TABLE,
        E173 / "MANIFEST.sha256",
        E173 / "RUN_STATUS.json",
        E174 / "PREREG_ANALYSIS_PLAN.md",
        E174 / "RUN_STATUS.json",
        E174 / "STATISTICAL_ANALYSIS_LOCK.json",
    ]
    input_hashes = [require_committed(path, head) for path in input_paths]
    development = add_pair_columns(pd.read_csv(DEVELOPMENT_TABLE, keep_default_na=False))
    if len(development) != 3000 or target_key(development).nunique() != 1000:
        raise IntegrityFailure("E173 development population changed")
    if set(development.panel_id.astype(str)) != {"E168", "Q01", "Q02", "Q03", "Q04"}:
        raise IntegrityFailure("E173 development panels changed")

    cv = cross_validate(development)
    gate = decide_models(cv)
    models: dict[str, dict[str, Any]] = {}
    for outcome in OUTCOMES:
        models[outcome] = {
            spec: fit_ridge(development, outcome, spec) for spec in MODEL_SPECS
        }
    selected = gate.set_index("outcome").selected_model_spec.astype(str).to_dict()

    (OUT / "tables").mkdir(parents=True)
    (OUT / "reports").mkdir()
    atomic_csv(OUT / "tables/DEVELOPMENT_CROSSVALIDATION.csv", cv)
    atomic_csv(OUT / "tables/DEVELOPMENT_MODEL_GATE.csv", gate)
    atomic_csv(OUT / "tables/INPUT_HASHES.csv", pd.DataFrame(input_hashes))
    snapshot = {
        "schema": "safeconf_e174_method_gate_snapshot_v1",
        "experiment": "E174_rotated_donor_conformal_certificate",
        "stage": "F1B_PRIOR_DATA_METHOD_FREEZE",
        "status": "PASS",
        "git_head": head,
        "git_branch": branch,
        "remote_heads": remote_heads,
        "development_targets": 1000,
        "development_tasks": 3000,
        "development_source": "E168_plus_E172_via_E173_posttruth_audit",
        "e174_expression_x_values_read": 0,
        "e174_calibration_truth_used": False,
        "e174_evaluation_truth_used": False,
        "coverage_target": COVERAGE,
        "repeated_identity_splits": REPEATS,
        "candidate_increment_gate": {
            "minimum_relative_mean_upper_improvement": MIN_RELATIVE_MEAN_UPPER_IMPROVEMENT,
            "minimum_repeat_win_fraction": MIN_REPEAT_WIN_FRACTION,
        },
        "selected_model_spec": selected,
        "models": models,
        "legacy_fixed_safeconf_increment_claim_retired": True,
        "deployment_authorized": False,
    }
    atomic_json(OUT / "METHOD_GATE_SNAPSHOT.json", snapshot)
    report = f"""# E174 prior-data method development

E174 的 800 个目标 expression X 在本阶段读取数为 **0**。基础误差估计器只使用 E168+E172 已解封的 1,000 个目标、3,000 个任务。

20 次按 panel 与 seen/unseen 分层的 target-level 60/20/20 重采样中，复合候选必须同时达到平均上界至少 {MIN_RELATIVE_MEAN_UPPER_IMPROVEMENT:.1%} 的相对缩短，并在至少 {MIN_REPEAT_WIN_FRACTION:.0%} 的重复中胜过 magnitude，才能取代 magnitude。冻结选择为：ensemble RMSE 使用 `{selected['ensemble_rmse']}`；pair-mean RMSE 使用 `{selected['pair_mean_rmse']}`。

未通过的复合候选不会在 E174 校准或评价真值打开后复活。constant、magnitude 与复合模型都保留用于透明效率对照；最终主输出只使用此处冻结的选择。
"""
    atomic_bytes(OUT / "reports/METHOD_DEVELOPMENT_REPORT.md", report.encode())
    manifest_sha = write_manifest()
    status = {
        "schema": "safeconf_e174_method_development_status_v1",
        "status": "COMPLETE",
        "git_head": head,
        "selected_model_spec": selected,
        "e174_expression_x_values_read": 0,
        "e174_calibration_truth_used": False,
        "e174_evaluation_truth_used": False,
        "manifest_sha256": manifest_sha,
        "deployment_authorized": False,
        "python": sys.version,
        "platform": platform.platform(),
    }
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
