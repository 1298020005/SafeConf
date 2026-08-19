#!/usr/bin/env python3
"""Seal the E201 TxPert MeanBaseline-equivalent task centroids before truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from build_e201_pretruth_task_base import source_evidence


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
FREEZE = OUT / "OFFICIAL_GENERAL_BASELINE_FREEZE.md"
TASK_BUILDER = ROOT / "tools/scripts/build_e201_pretruth_task_base.py"
TASK_STATUS = OUT / "E201_PRETRUTH_TASK_BASE_STATUS.json"
TASK_TABLE = OUT / "tables/E201_PRETRUTH_TASK_BASE.csv"
SOURCE_SUPPORT = OUT / "tables/E201_SOURCE_CONTEXT_SUPPORT.csv"
RISK_STATUS = OUT / "E201_PRETRUTH_RISK_STATUS.json"
RISK_TABLE = OUT / "tables/E201_PRETRUTH_RISK_FEATURES.csv"
PREFLIGHT = OUT / "E201_OFFICIAL_GENERAL_BASELINE_PREFLIGHT.json"
STATUS = OUT / "E201_OFFICIAL_GENERAL_BASELINE_STATUS.json"
SUPPORT_AUDIT = OUT / "tables/E201_OFFICIAL_GENERAL_BASELINE_SUPPORT_AUDIT.csv"
ACCESS_AUDIT = OUT / "tables/E201_OFFICIAL_GENERAL_BASELINE_ACCESS_AUDIT.csv"
REPORT = OUT / "OFFICIAL_GENERAL_BASELINE_REPORT.md"

TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
N_TASKS = 2_008
N_PRIMARY = 1_808
N_GENES = 3_352
EQUIVALENCE_TOL = 5e-6
E200_GENERAL = Path(
    "txpert_official_20260802/e200/predictions/general_baseline/test_predictions.h5ad"
)
E200_CONTROL = Path(
    "txpert_official_20260802/e200/predictions/general_baseline/test_controls.h5ad"
)
E200_EXPECTED = {
    E200_GENERAL: (
        2_025_572_536,
        "0d2200b0762b5aa4f7f29314bbda99032a78a4f959f937c9d14cbd444b437d30",
    ),
    E200_CONTROL: (
        2_025_572_536,
        "faab78f1847c2c59eea4a71d93b58b419edec2dd291e63174ab3bca3938a987a",
    ),
}


class BaselineFailure(RuntimeError):
    """Fail-closed pretruth general-baseline error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--vector-output-dir", type=Path)
    parser.add_argument("--e200-validation-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.data_root is None:
        parser.error("--data-root is required outside --self-test")
    if (
        not args.self_test
        and not args.e200_validation_only
        and args.vector_output_dir is None
    ):
        parser.error("formal mode requires --vector-output-dir")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def tracked_clean(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    commands = (
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{relative}"],
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", relative],
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
            relative,
        ],
    )
    return all(
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
        for command in commands
    )


def remote_tip(remote: str, branch: str) -> str:
    line = git_text("ls-remote", remote, f"refs/heads/{branch}")
    if not line:
        raise BaselineFailure(f"missing remote branch: {remote}/{branch}")
    return line.split()[0]


def verify_git_release(include_risk: bool) -> str:
    required = [
        SCRIPT,
        FREEZE,
        TASK_BUILDER,
        TASK_STATUS,
        TASK_TABLE,
        SOURCE_SUPPORT,
    ]
    if include_risk:
        required.extend([RISK_STATUS, RISK_TABLE, PREFLIGHT])
    if not all(path.is_file() and tracked_clean(path) for path in required):
        raise BaselineFailure(
            "baseline code/freeze/input seal is not tracked and clean"
        )
    branch = git_text("branch", "--show-current")
    head = git_text("rev-parse", "HEAD")
    if not branch:
        raise BaselineFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if remote_tip(remote, branch) != head:
            raise BaselineFailure(f"{remote}/{branch} differs from local HEAD")
    return head


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def verify_file(path: Path, expected_bytes: int, expected_sha: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or sha256_file(path) != expected_sha
    ):
        raise BaselineFailure(f"sealed file changed: {path}")


def load_task_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    status = json.loads(TASK_STATUS.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or int(status.get("n_tasks", -1)) != N_TASKS
        or int(status.get("n_primary_tasks", -1)) != N_PRIMARY
        or int(status.get("target_perturbed_expression_rows_opened", -1)) != 0
        or status.get("target_predictions_opened") is not False
        or status.get("target_outcomes_evaluated") is not False
    ):
        raise BaselineFailure("task-base status failed")
    for record in status.get("tracked_outputs", []):
        path = ROOT / record["path"]
        verify_file(path, int(record["bytes"]), record["sha256"])
    tasks = pd.read_csv(TASK_TABLE, keep_default_na=True)
    support = pd.read_csv(SOURCE_SUPPORT, keep_default_na=False)
    if (
        len(tasks) != N_TASKS
        or tasks.task_id.nunique() != N_TASKS
        or int(tasks.analysis_stratum.eq("primary_ge30").sum()) != N_PRIMARY
        or not np.array_equal(
            tasks.source_mean_delta_row.to_numpy(int), np.arange(N_TASKS)
        )
        or len(support) != 5_238
    ):
        raise BaselineFailure("task/support inventory changed")
    return tasks, support, status


def load_control_centroids(
    data_root: Path,
) -> tuple[np.ndarray, dict, dict[str, Any]]:
    risk = json.loads(RISK_STATUS.read_text(encoding="utf-8"))
    if (
        risk.get("status") != "PASS"
        or int(risk.get("n_tasks", -1)) != N_TASKS
        or int(risk.get("n_primary_tasks", -1)) != N_PRIMARY
        or int(risk.get("target_expression_nonzero_values_seen", -1)) != 0
        or risk.get("target_truth_materialized") is not False
        or risk.get("target_outcomes_evaluated") is not False
    ):
        raise BaselineFailure("pretruth risk seal failed")
    risk_table_record = risk["risk_table"]
    verify_file(
        RISK_TABLE,
        int(risk_table_record["bytes"]),
        risk_table_record["sha256"],
    )
    matches = [
        record
        for record in risk.get("vector_files", [])
        if Path(record["path"]).name == "E201_CONTROL_CENTROIDS.npy"
    ]
    if len(matches) != 1 or not matches[0]["path"].startswith("DATA/"):
        raise BaselineFailure("sealed control centroid record missing")
    record = matches[0]
    path = data_root / record["path"][len("DATA/") :]
    verify_file(path, int(record["bytes"]), record["sha256"])
    controls = np.load(path, mmap_mode="r")
    if controls.shape != (N_TASKS, N_GENES) or str(controls.dtype) != "float32":
        raise BaselineFailure("sealed control centroid shape changed")
    if not np.isfinite(controls).all():
        raise BaselineFailure("non-finite sealed control centroid")
    return controls, risk, record


def compute_weighted_source_deltas(
    tasks: pd.DataFrame,
    frozen_support: pd.DataFrame,
    data_root: Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    vectors = np.empty((N_TASKS, N_GENES), dtype=np.float32)
    support_blocks = []
    access_blocks = []
    audit_rows = []
    cache_root = data_root / "txpert_official_20260802/cache"
    for target in TARGETS:
        target_tasks = tasks.loc[tasks.target.eq(target)]
        conditions = set(target_tasks.condition.astype(str))
        deltas, observed_support, access = source_evidence(
            target,
            cache_root / f"E201_blind_{target}",
            conditions,
        )
        support_blocks.append(observed_support)
        access_blocks.append(access)
        for task in target_tasks.itertuples(index=False):
            block = observed_support.loc[
                observed_support.condition.eq(str(task.condition))
            ].sort_values("source_context")
            context_map = deltas[str(task.condition)]
            if set(block.source_context.astype(str)) != set(context_map):
                raise BaselineFailure(f"source context changed: {task.task_id}")
            weights = block.n_source_perturbed_cells.to_numpy(int)
            members = np.stack(
                [
                    context_map[str(context)]
                    for context in block.source_context.astype(str)
                ]
            )
            if int(weights.sum()) != int(task.n_source_cells):
                raise BaselineFailure(f"source cell count changed: {task.task_id}")
            weighted = np.average(members, axis=0, weights=weights)
            if not np.isfinite(weighted).all():
                raise BaselineFailure(f"non-finite weighted delta: {task.task_id}")
            row = int(task.source_mean_delta_row)
            vectors[row] = weighted.astype(np.float32)
            audit_rows.append(
                {
                    "task_id": task.task_id,
                    "target": target,
                    "condition": task.condition,
                    "n_source_contexts": len(block),
                    "n_source_cells": int(weights.sum()),
                    "weight_sum": int(weights.sum()),
                    "target_perturbed_expression_rows_opened": 0,
                }
            )
    observed = (
        pd.concat(support_blocks, ignore_index=True)
        .sort_values(["target", "condition", "source_context"])
        .reset_index(drop=True)
    )
    expected = frozen_support.sort_values(
        ["target", "condition", "source_context"]
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    except AssertionError as exc:
        raise BaselineFailure("source support differs from frozen task base") from exc
    access = pd.concat(access_blocks, ignore_index=True)
    if (
        len(observed) != 5_238
        or len(access) != 24
        or int(access.target_perturbed_expression_rows.sum()) != 0
        or not np.isfinite(vectors).all()
    ):
        raise BaselineFailure("weighted source-delta access contract failed")
    return vectors, pd.DataFrame(audit_rows), access


def condition_from_label(label: str) -> str:
    prefix = "K562_"
    suffix = "_1+1"
    if not label.startswith(prefix) or not label.endswith(suffix):
        raise BaselineFailure(f"unexpected E200 label: {label}")
    condition = label[len(prefix) : -len(suffix)]
    if not condition.endswith("+ctrl"):
        raise BaselineFailure(f"unexpected E200 condition: {label}")
    return condition


def validate_against_e200(
    tasks: pd.DataFrame,
    weighted_deltas: np.ndarray,
    data_root: Path,
) -> dict[str, Any]:
    paths = {relative: data_root / relative for relative in E200_EXPECTED}
    for relative, (expected_bytes, expected_sha) in E200_EXPECTED.items():
        verify_file(paths[relative], expected_bytes, expected_sha)
    prediction = ad.read_h5ad(paths[E200_GENERAL], backed="r")
    control = ad.read_h5ad(paths[E200_CONTROL], backed="r")
    try:
        if prediction.shape != control.shape or prediction.shape[1] != N_GENES:
            raise BaselineFailure("E200 baseline/control shapes differ")
        for column in (
            "pert_cond_names",
            "cell_types",
            "experimental_batches",
        ):
            if not np.array_equal(
                prediction.obs[column].astype(str).to_numpy(),
                control.obs[column].astype(str).to_numpy(),
            ):
                raise BaselineFailure(f"E200 baseline alignment changed: {column}")
        labels = prediction.obs.pert_cond_names.astype(str)
        conditions = labels.map(condition_from_label)
        groups = conditions.groupby(conditions, sort=False).groups
        k562_tasks = tasks.loc[tasks.target.eq("K562")]
        if len(k562_tasks) != 580:
            raise BaselineFailure("K562 equivalence task inventory changed")
        maximum = 0.0
        sum_squares = 0.0
        n_values = 0
        task_failures = 0
        for task in k562_tasks.itertuples(index=False):
            if str(task.condition) not in groups:
                raise BaselineFailure(f"E200 task missing: {task.task_id}")
            indices = np.asarray(groups[str(task.condition)], dtype=int)
            if len(indices) != int(task.n_target_cells):
                raise BaselineFailure(f"E200 task cell count changed: {task.task_id}")
            official_delta = np.asarray(prediction.X[indices], dtype=np.float64).mean(
                axis=0
            ) - np.asarray(control.X[indices], dtype=np.float64).mean(axis=0)
            frozen_delta = np.asarray(
                weighted_deltas[int(task.source_mean_delta_row)], dtype=np.float64
            )
            difference = official_delta - frozen_delta
            task_maximum = float(np.max(np.abs(difference)))
            maximum = max(maximum, task_maximum)
            sum_squares += float(np.sum(np.square(difference)))
            n_values += difference.size
            task_failures += int(task_maximum > EQUIVALENCE_TOL)
    finally:
        prediction.file.close()
        control.file.close()
    result = {
        "reference": "E200 TxPert public MeanBaseline cell-level output",
        "n_tasks": 580,
        "n_genes": N_GENES,
        "maximum_absolute_delta_residual": maximum,
        "rms_delta_residual": float(np.sqrt(sum_squares / n_values)),
        "tolerance": EQUIVALENCE_TOL,
        "tasks_exceeding_tolerance": task_failures,
        "target_truth_opened": False,
        "passed": bool(maximum <= EQUIVALENCE_TOL and task_failures == 0),
    }
    if not result["passed"]:
        raise BaselineFailure(f"E200 MeanBaseline equivalence failed: {result}")
    return result


def self_test() -> None:
    rng = np.random.default_rng(20260802)
    members = rng.normal(size=(3, 17))
    weights = np.asarray([11, 23, 7])
    direct = np.average(members, axis=0, weights=weights)
    expanded = np.repeat(members, weights, axis=0).mean(axis=0)
    residual = float(np.max(np.abs(direct - expanded)))
    if residual > 1e-12:
        raise BaselineFailure("synthetic weighted-delta identity failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "weighted_delta_identity_max_abs_residual": residual,
                "target_truth_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    data_root = args.data_root.resolve()
    tasks, frozen_support, task_status = load_task_inputs()
    if args.e200_validation_only:
        if PREFLIGHT.exists():
            raise BaselineFailure("official baseline preflight already exists")
        head = verify_git_release(include_risk=False)
        weighted, _, _ = compute_weighted_source_deltas(
            tasks, frozen_support, data_root
        )
        result = validate_against_e200(tasks, weighted, data_root)
        payload = {
            "experiment": "E201_txpert_multitarget_retraining",
            "stage": "OFFICIAL_GENERAL_BASELINE_PREFLIGHT",
            "status": "PASS",
            "generated_at": now(),
            "safeconf_commit": head,
            "task_base_status_sha256": sha256_file(TASK_STATUS),
            "target_perturbed_expression_rows_opened": 0,
            "target_truth_opened": False,
            "e200_official_code_equivalence": result,
        }
        atomic_json(PREFLIGHT, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    vector_dir = args.vector_output_dir.resolve()
    try:
        vector_dir.relative_to(data_root)
    except ValueError as exc:
        raise BaselineFailure("vector output must stay under data root") from exc
    repo_outputs = (STATUS, SUPPORT_AUDIT, ACCESS_AUDIT, REPORT)
    if vector_dir.exists() or any(path.exists() for path in repo_outputs):
        raise BaselineFailure("general-baseline output already exists")
    head = verify_git_release(include_risk=True)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "PASS"
        or preflight.get("target_truth_opened") is not False
        or int(preflight.get("target_perturbed_expression_rows_opened", -1)) != 0
        or preflight.get("task_base_status_sha256") != sha256_file(TASK_STATUS)
        or preflight.get("e200_official_code_equivalence", {}).get("passed") is not True
    ):
        raise BaselineFailure("official general-baseline preflight failed")
    controls, risk_status, control_record = load_control_centroids(data_root)
    weighted, support_audit, access_audit = compute_weighted_source_deltas(
        tasks, frozen_support, data_root
    )
    equivalence = validate_against_e200(tasks, weighted, data_root)
    frozen_equivalence = preflight["e200_official_code_equivalence"]
    for key in ("maximum_absolute_delta_residual", "rms_delta_residual"):
        if not math.isclose(
            float(equivalence[key]),
            float(frozen_equivalence[key]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise BaselineFailure(f"E200 equivalence changed after preflight: {key}")
    centroids = np.asarray(controls, dtype=np.float32) + weighted
    if centroids.shape != (N_TASKS, N_GENES) or not np.isfinite(centroids).all():
        raise BaselineFailure("general-baseline centroid contract failed")

    vector_dir.mkdir(parents=True)
    weighted_path = vector_dir / "E201_OFFICIAL_GENERAL_BASELINE_WEIGHTED_DELTAS.npy"
    centroid_path = vector_dir / "E201_OFFICIAL_GENERAL_BASELINE_CENTROIDS.npy"
    atomic_npy(weighted_path, weighted)
    atomic_npy(centroid_path, centroids.astype(np.float32, copy=False))
    atomic_csv(SUPPORT_AUDIT, support_audit)
    atomic_csv(ACCESS_AUDIT, access_audit)
    report = "\n".join(
        [
            "# E201 TxPert general baseline 预测前封存",
            "",
            f"- 任务：{N_TASKS}（主分析 {N_PRIMARY}）；",
            "- source-context support：5,238；",
            "- target 扰动表达访问：0 行；",
            "- target truth：未打开；",
            "- 计算：按 source 扰动细胞数加权的公开 MeanBaseline 单扰动分支；",
            f"- E200 公开类输出等价性最大绝对残差：{equivalence['maximum_absolute_delta_residual']:.8g}；",
            f"- 等价性 RMS 残差：{equivalence['rms_delta_residual']:.8g}；",
            f"- 固定容差：{EQUIVALENCE_TOL:.1e}，超过容差任务 0。",
            "",
            "该结果是预测前强基线封存，不包含 E201 target error。正式评价会同时比较"
            "四种子 family centroid、该 general baseline、batch-matched control 和"
            "source-transfer baseline。",
            "",
        ]
    )
    atomic_text(REPORT, report)

    def vector_record(path: Path, values: np.ndarray) -> dict[str, Any]:
        return {
            "path": "DATA/" + path.relative_to(data_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }

    status = {
        "experiment": "E201_txpert_multitarget_retraining",
        "stage": "PRETRUTH_OFFICIAL_GENERAL_BASELINE",
        "status": "PASS",
        "generated_at": now(),
        "safeconf_commit": head,
        "task_base_status_sha256": sha256_file(TASK_STATUS),
        "pretruth_risk_status_sha256": sha256_file(RISK_STATUS),
        "official_general_baseline_preflight_sha256": sha256_file(PREFLIGHT),
        "pretruth_risk_table_sha256": risk_status["risk_table"]["sha256"],
        "sealed_control_centroid_sha256": control_record["sha256"],
        "n_tasks": N_TASKS,
        "n_primary_tasks": N_PRIMARY,
        "n_source_support_rows": len(frozen_support),
        "n_source_access_records": len(access_audit),
        "target_perturbed_expression_rows_opened": 0,
        "target_truth_materialized": False,
        "target_outcomes_evaluated": False,
        "e200_official_code_equivalence": equivalence,
        "vector_files": [
            vector_record(weighted_path, weighted),
            vector_record(centroid_path, centroids),
        ],
        "tracked_outputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (SUPPORT_AUDIT, ACCESS_AUDIT, REPORT)
        ],
        "task_base_safeconf_commit": task_status["safeconf_commit"],
    }
    atomic_json(STATUS, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
