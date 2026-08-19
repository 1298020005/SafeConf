#!/usr/bin/env python3
"""Train the frozen E182 family and release predictions without hidden truth."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
BASE = ROOT / "tools/scripts/run_e180_xucao_pretruth.py"
BASE_BUILDER = ROOT / "tools/scripts/build_e180_xucao_pretruth_assets.py"
BUILDER = ROOT / "tools/scripts/build_e182_gse225807_pretruth_assets.py"
E177_RUNNER = ROOT / "tools/scripts/run_e177_sunshine_pretruth.py"
E65_SCRIPT = ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py"
E107_SCRIPT = ROOT / "tools/scripts/run_e107_frangieh_context_gears.py"
GO_FILE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")
OUT = ROOT / "docs/实验结果/E182_gse225807_registered_family_20260724"
RELEASE = OUT / "pretruth_release"
ASSET_ROOT = Path("/home/yyf/data/safeconf_e182_gse225807/isolated/F2_pretruth")
MODEL_LOCK = OUT / "MODEL_INPUT_LOCK.json"
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
STAT_LOCK = OUT / "STATISTICAL_ANALYSIS_LOCK.json"
PLAN = OUT / "PREREG_ANALYSIS_PLAN.md"
TARGETS = OUT / "manifests/E182_SELECTED_TARGETS.csv"
TASKS = OUT / "manifests/E182_GUIDE_TASK_MANIFEST.csv"
SEEDS = (3407, 3408, 3409, 3410, 3411)
N_GENES = 512


class IntegrityError(RuntimeError):
    """The E182 model family or sealed truth interface changed."""


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_base(base: Any) -> None:
    base.RUNNER = RUNNER
    base.OUT = OUT
    base.RELEASE = RELEASE
    base.STAGING = OUT / f".pretruth_release.staging.{os.getpid()}"
    base.ASSET_ROOT = ASSET_ROOT
    base.BUILDER = BUILDER
    base.E177_RUNNER = E177_RUNNER
    base.E65_SCRIPT = E65_SCRIPT
    base.E107_SCRIPT = E107_SCRIPT
    base.GO_FILE = GO_FILE
    base.MODEL_LOCK = MODEL_LOCK
    base.SOURCE_LOCK = SOURCE_LOCK
    base.STAT_LOCK = STAT_LOCK
    base.PLAN = PLAN
    base.TARGETS = TARGETS
    base.TASKS = TASKS
    base.SEEDS = SEEDS
    base.N_GENES = N_GENES


def rms(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=axis))


def assemble_family(
    assets: Any,
    predictions: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    task_order = assets.tasks["task_id"].astype(str).tolist()
    registered_names = [
        *(f"scGPT_seed{seed}" for seed in SEEDS),
        *(f"GEARS_seed{seed}" for seed in SEEDS),
    ]
    if set(predictions) != set(registered_names):
        raise IntegrityError("E182 registered prediction family changed")
    arrays = {
        name: np.stack([predictions[name][task] for task in task_order]).astype(
            np.float32
        )
        for name in registered_names
    }
    family = np.stack([arrays[name] for name in registered_names], axis=0)
    family64 = family.astype(np.float64)
    centroid64 = family64.mean(axis=0)
    centroid = centroid64.astype(np.float32)
    arrays["scGPT_seed_mean"] = np.stack(
        [arrays[f"scGPT_seed{seed}"] for seed in SEEDS]
    ).mean(axis=0).astype(np.float32)
    arrays["GEARS_seed_mean"] = np.stack(
        [arrays[f"GEARS_seed{seed}"] for seed in SEEDS]
    ).mean(axis=0).astype(np.float32)
    arrays["registered_family_centroid"] = centroid

    member_to_centroid = rms(family64 - centroid64[None, :, :], axis=2)
    diversity = np.sqrt(np.mean(np.square(member_to_centroid), axis=0))
    radius = member_to_centroid.max(axis=0)
    diameter = np.zeros(len(task_order), dtype=np.float64)
    for left in range(len(registered_names)):
        for right in range(left + 1, len(registered_names)):
            diameter = np.maximum(
                diameter,
                rms(family[left] - family[right], axis=1),
            )
    architecture_distance = rms(
        arrays["scGPT_seed_mean"] - arrays["GEARS_seed_mean"], axis=1
    )
    geometry = pd.DataFrame(
        {
            "family_diversity_lower": diversity,
            "family_radius": radius,
            "family_diameter": diameter,
            "worst_member_lower": diameter / 2.0,
            "two_architecture_mean_lower": architecture_distance / 2.0,
            "registered_family_centroid_magnitude": rms(centroid, axis=1),
            "scgpt_seed_spread": np.sqrt(
                np.mean(
                    np.var(
                        np.stack(
                            [arrays[f"scGPT_seed{seed}"] for seed in SEEDS]
                        ),
                        axis=0,
                    ),
                    axis=1,
                )
            ),
            "gears_seed_spread": np.sqrt(
                np.mean(
                    np.var(
                        np.stack(
                            [arrays[f"GEARS_seed{seed}"] for seed in SEEDS]
                        ),
                        axis=0,
                    ),
                    axis=1,
                )
            ),
        }
    )
    scores = pd.concat([assets.tasks.reset_index(drop=True), geometry], axis=1)
    scores["constant_centroid_base"] = 0.0
    scores["calibration_or_evaluation_truth_present"] = False
    scores["registered_family_size"] = len(registered_names)
    scores["learned_or_adaptive_upper_fitted"] = False

    validation_rows: list[dict[str, Any]] = []
    for index, row in scores[
        scores["target_split"].eq("model_validation")
    ].iterrows():
        truth = assets.seen_effects[str(row["task_id"])]
        member_errors = rms(
            family64[:, index, :] - truth.astype(np.float64)[None, :], axis=1
        )
        centroid_error = float(
            rms(centroid64[index] - truth.astype(np.float64))
        )
        family_rms_error = float(np.sqrt(np.mean(np.square(member_errors))))
        identity_residual = float(
            family_rms_error**2
            - centroid_error**2
            - float(row["family_diversity_lower"]) ** 2
        )
        validation_rows.append(
            {
                "task_id": row["task_id"],
                "perturbation": row["perturbation"],
                "guide_id": row["guide_id"],
                "centroid_rmse": centroid_error,
                "family_rms_error": family_rms_error,
                "family_diversity_lower": row["family_diversity_lower"],
                "worst_member_error": float(member_errors.max()),
                "worst_member_lower": row["worst_member_lower"],
                "hilbert_identity_residual": identity_residual,
            }
        )
    return scores, arrays, pd.DataFrame(validation_rows)


def run_gates(
    assets: Any,
    scores: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    graph_audit: pd.DataFrame,
    validation: pd.DataFrame,
    e177: Any,
) -> pd.DataFrame:
    hidden_splits = ["conformal_calibration", "prospective_evaluation"]
    hidden = scores["target_split"].isin(hidden_splits)
    registered = [
        *(f"scGPT_seed{seed}" for seed in SEEDS),
        *(f"GEARS_seed{seed}" for seed in SEEDS),
    ]
    checks = {
        "G1_hidden_query_graphs_have_no_y": not (
            graph_audit["graph_role"].eq("query")
            & graph_audit["contains_y"].astype(bool)
            & graph_audit["target_split"].isin(hidden_splits)
        ).any(),
        "G2_hidden_truth_absent_from_interface": not scores.loc[
            hidden, "calibration_or_evaluation_truth_present"
        ].astype(bool).any(),
        "G3_registered_family_complete": all(
            arrays[name].shape == (len(scores), N_GENES) for name in registered
        ),
        "G4_family_geometry_finite": bool(
            np.isfinite(
                scores[
                    [
                        "family_diversity_lower",
                        "family_radius",
                        "family_diameter",
                        "worst_member_lower",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "G5_hidden_predictors_variable": bool(
            e177.predictor_gate(
                arrays["scGPT_seed_mean"][hidden.to_numpy()]
            )["passed"]
            and e177.predictor_gate(
                arrays["GEARS_seed_mean"][hidden.to_numpy()]
            )["passed"]
        ),
        "G6_validation_identity_numeric": bool(
            len(validation) > 0
            and validation["hilbert_identity_residual"].abs().max() <= 1e-10
        ),
        "G7_f2_hidden_x_rows_zero": bool(
            assets.attestation["calibration_target_x_rows_read"] == 0
            and assets.attestation["evaluation_target_x_rows_read"] == 0
        ),
        "G8_no_adaptive_upper_fitted": not scores[
            "learned_or_adaptive_upper_fitted"
        ].astype(bool).any(),
    }
    return pd.DataFrame(
        [{"gate": gate, "passed": bool(passed)} for gate, passed in checks.items()]
    )


def runtime_environment(device: Any) -> dict[str, Any]:
    import torch
    import torch_geometric

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "selected_device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else ""
        ),
    }


def write_release(
    base: Any,
    assets: Any,
    scores: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    validation: pd.DataFrame,
    graph_audit: pd.DataFrame,
    histories: pd.DataFrame,
    model_audit: pd.DataFrame,
    gates: pd.DataFrame,
    audit: tuple[str, str, dict[str, str], list[dict[str, Any]]],
    environment: dict[str, Any],
    wall_seconds: float,
) -> Path:
    head, branch, remotes, hashes = audit
    staging = OUT / f".pretruth_release.staging.{os.getpid()}"
    if RELEASE.exists() or staging.exists():
        raise IntegrityError("E182 pretruth release is append-only")
    try:
        for subdirectory in ("tables", "arrays", "reports"):
            (staging / subdirectory).mkdir(parents=True, exist_ok=False)
        base.atomic_csv(staging / "tables/PRETRUTH_SCORING_INTERFACE.csv", scores)
        base.atomic_npz(staging / "arrays/PRETRUTH_PREDICTIONS.npz", arrays)
        base.atomic_csv(staging / "tables/VALIDATION_CERTIFICATE_AUDIT.csv", validation)
        base.atomic_csv(staging / "tables/QUERY_GRAPH_AUDIT.csv", graph_audit)
        base.atomic_csv(staging / "tables/TRAINING_HISTORY.csv", histories)
        base.atomic_csv(staging / "tables/MODEL_AUDIT.csv", model_audit)
        base.atomic_csv(staging / "tables/PRETRUTH_GATES.csv", gates)
        base.atomic_csv(
            staging / "tables/INPUT_HASHES.csv",
            pd.DataFrame(hashes + assets.input_hashes),
        )
        base.atomic_json(staging / "RUNTIME_ENVIRONMENT.json", environment)
        snapshot = {
            "schema": "safeconf_e182_pretruth_snapshot_v1",
            "experiment": "E182_gse225807_registered_family",
            "status": "PASS" if gates["passed"].all() else "FAIL",
            "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "git_head": head,
            "git_branch": branch,
            "code_freeze_remote_heads": remotes,
            "runner_sha256": base.sha256_file(RUNNER),
            "builder_sha256": base.sha256_file(BUILDER),
            "f2_manifest_sha256": assets.manifest_sha256,
            "n_tasks": len(scores),
            "n_train_tasks": int(
                scores["target_split"].eq("supervised_train").sum()
            ),
            "n_validation_tasks": int(
                scores["target_split"].eq("model_validation").sum()
            ),
            "n_calibration_query_tasks": int(
                scores["target_split"].eq("conformal_calibration").sum()
            ),
            "n_evaluation_query_tasks": int(
                scores["target_split"].eq("prospective_evaluation").sum()
            ),
            "registered_family_size": 10,
            "calibration_target_x_rows_read": 0,
            "evaluation_target_x_rows_read": 0,
            "adaptive_upper_fitted": False,
            "wall_seconds": wall_seconds,
        }
        base.atomic_json(staging / "PRETRUTH_GATE_SNAPSHOT.json", snapshot)
        report = f"""# E182 预真值模型报告

状态：**{snapshot['status']}**。

scGPT 与 GEARS 各 5 个随机种子已经完成训练，并对全部冻结任务生成了 10 个预测向量。监督训练任务 {snapshot['n_train_tasks']} 条，模型验证任务 {snapshot['n_validation_tasks']} 条，校准查询 {snapshot['n_calibration_query_tasks']} 条，最终评价查询 {snapshot['n_evaluation_query_tasks']} 条。

预真值接口只包含预测家族、家族质心和可由预测直接计算的几何量。没有拟合 ExtraTrees 或其他学习型上界；校准与最终评价表达真值仍未读取。
"""
        base.atomic_bytes(
            staging / "reports/E182_PRETRUTH_REPORT.md", report.encode()
        )
        os.replace(staging, RELEASE)
        return RELEASE / "PRETRUTH_GATE_SNAPSHOT.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_formal(device_name: str) -> dict[str, Any]:
    started = time.time()
    base = import_script("e182_pretruth_base", BASE)
    configure_base(base)
    audit = base.formal_input_audit()
    head, branch, remotes, input_hashes = audit
    input_hashes.extend(
        [
            base.require_committed(BASE, head),
            base.require_committed(BASE_BUILDER, head),
        ]
    )
    audit = (head, branch, remotes, input_hashes)
    assets = base.load_assets()
    supervised, query, graph_audit = base.build_graphs(assets)
    e177 = import_script("e182_training_helpers", E177_RUNNER)
    e177.N_GENES = N_GENES
    e177.E65_SCRIPT = E65_SCRIPT
    e177.E107_SCRIPT = E107_SCRIPT
    e177.GO_FILE = GO_FILE

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise IntegrityError("E182 registered CUDA device is unavailable")
    genes = assets.panel["scgpt_token"].astype(str).tolist()
    predictions: dict[str, dict[str, np.ndarray]] = {}
    histories: list[pd.DataFrame] = []
    model_audits: list[dict[str, Any]] = []
    for seed in SEEDS:
        scgpt, history, model_info = e177.train_scgpt(
            seed, supervised, query, genes, device
        )
        predictions[f"scGPT_seed{seed}"] = scgpt
        histories.append(history)
        model_audits.append({"seed": seed, "model": "scGPT", **model_info})
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gears, history, model_info = e177.train_gears(
            seed, supervised, query, genes, assets.coexpression, device
        )
        predictions[f"GEARS_seed{seed}"] = gears
        histories.append(history)
        model_audits.append({"seed": seed, "model": "GEARS", **model_info})
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores, arrays, validation = assemble_family(assets, predictions)
    gates = run_gates(assets, scores, arrays, graph_audit, validation, e177)
    snapshot = write_release(
        base,
        assets,
        scores,
        arrays,
        validation,
        graph_audit,
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(model_audits),
        gates,
        audit,
        runtime_environment(device),
        time.time() - started,
    )
    return {
        "status": "PASS" if gates["passed"].all() else "FAIL",
        "snapshot": str(snapshot.relative_to(ROOT)),
        "snapshot_sha256": base.sha256_file(snapshot),
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--asset-audit-only", action="store_true")
    args = parser.parse_args()
    base = import_script("e182_pretruth_asset_audit", BASE)
    configure_base(base)
    if args.asset_audit_only:
        assets = base.load_assets()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "n_tasks": len(assets.tasks),
                    "n_seen_effects": len(assets.seen_effects),
                    "f2_manifest_sha256": assets.manifest_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run_formal(args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
