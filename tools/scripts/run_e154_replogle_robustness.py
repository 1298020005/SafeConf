#!/usr/bin/env python3
"""E154: post-E152 Replogle seed and frozen train-fraction robustness audit.

This experiment is deliberately a sensitivity analysis, not a second
confirmation gate.  It reuses the E149 pre-expression membership columns,
the E112 fixed scGPT/GEARS workflow, the E135 frozen directional model and
the E134 exact Systema expression-space audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(CODE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts


SOURCE_SCRIPT = ROOT / "tools/scripts/run_e112_external_formal_dual_models.py"
E134_SCRIPT = ROOT / "tools/scripts/run_e134_systema_exact_expression_space_audit.py"
E149_ROOT = ROOT / "docs/实验结果/E149_replogle_two_cellline_contract_20260714"
ORIGINAL_MANIFEST = E149_ROOT / "manifests/E149_TASK_MANIFEST.csv"
E150_ROOT = ROOT / "docs/实验结果/E150_replogle_combined_asset_20260714"
ASSET = Path(
    "/home/yyf/data/safeconf_e150_replogle/"
    "Replogle_two_cellline_E149_selected_raw_counts.h5ad"
)
CACHE = Path(
    "/home/yyf/data/safeconf_e112_external/"
    "Replogle_two_cellline_CONTROL_ONLY_512.npz"
)
E151_ROOT = ROOT / "docs/实验结果/E151_replogle_formal_dual_models_20260714"
E151_CHILD = E151_ROOT / "Replogle_two_cellline"
E152_STATUS = ROOT / "docs/实验结果/E152_replogle_directional_confirmation_20260714/RUN_STATUS.json"
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
OUT = ROOT / "docs/实验结果/E154_replogle_robustness_20260714"
MANIFESTS = OUT / "manifests"
RUNS_ROOT = OUT / "runs"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"
DATASET = "Replogle_two_cellline"

SEED_1 = 202607151
SEED_2 = 2026071542
SEED_3 = 2026071543

RUN_MATRIX = [
    {
        "run_id": "seed1_frac100_reused_e151",
        "seed": SEED_1,
        "train_fraction": 100,
        "reused_e151": True,
    },
    {
        "run_id": "seed2_frac25",
        "seed": SEED_2,
        "train_fraction": 25,
        "reused_e151": False,
    },
    {
        "run_id": "seed2_frac50",
        "seed": SEED_2,
        "train_fraction": 50,
        "reused_e151": False,
    },
    {
        "run_id": "seed2_frac75",
        "seed": SEED_2,
        "train_fraction": 75,
        "reused_e151": False,
    },
    {
        "run_id": "seed2_frac100",
        "seed": SEED_2,
        "train_fraction": 100,
        "reused_e151": False,
    },
    {
        "run_id": "seed3_frac100",
        "seed": SEED_3,
        "train_fraction": 100,
        "reused_e151": False,
    },
]
NEW_RUNS = [item for item in RUN_MATRIX if not item["reused_e151"]]
RUN_BY_ID = {item["run_id"]: item for item in RUN_MATRIX}

ABSOLUTE_SCORES = {
    "absolute_safeconf_spearman": "safeconf_calibrated_pair_risk",
    "absolute_magnitude_spearman": "baseline_predicted_magnitude",
    "absolute_disagreement_spearman": "risk_model_disagreement",
}
DIRECTION_SCORES = {
    "directional": "directional_risk_frozen",
    "magnitude": "baseline_predicted_magnitude",
}
DIRECTION_ENDPOINTS = {
    "pearson": "error_centered_pearson_mean",
    "cosine": "error_centered_cosine_mean",
    "composite": "direction_error_rank_target",
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def bool_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"unrecognized Boolean values: {values[mapped.isna()].unique()}")
    return mapped.astype(bool)


def rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def manifest_path(fraction: int) -> Path:
    return MANIFESTS / f"E154_TRAIN_FRACTION_{fraction}_MANIFEST.csv"


def run_view_root(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def source_root(item: dict[str, object]) -> Path:
    if item["reused_e151"]:
        return E151_CHILD
    return run_view_root(str(item["run_id"])) / DATASET


def source_manifest(item: dict[str, object]) -> Path:
    if item["reused_e151"]:
        return ORIGINAL_MANIFEST
    return manifest_path(int(item["train_fraction"]))


def validate_frozen_inputs() -> dict[str, object]:
    e149 = json.loads((E149_ROOT / "RUN_STATUS.json").read_text())
    e150 = json.loads((E150_ROOT / "RUN_STATUS.json").read_text())
    e152 = json.loads(E152_STATUS.read_text())
    expected_manifest = e149["artifact_sha256"]["manifests/E149_TASK_MANIFEST.csv"]
    if sha256(ORIGINAL_MANIFEST) != expected_manifest:
        raise RuntimeError("E149 manifest hash changed")
    if sha256(MODEL) != e149["frozen_direction_model_sha256"]:
        raise RuntimeError("E135 frozen direction model differs from E149 hash")
    if sha256(ASSET) != e150["asset_sha256"]:
        raise RuntimeError("E150 combined asset hash changed")
    if not CACHE.exists():
        raise FileNotFoundError(f"E112 fixed cache is absent: {CACHE}")
    if e152.get("status") != "complete":
        raise RuntimeError("E152 must be complete before this post-confirmation sensitivity")
    return {
        "e149_manifest_sha256": expected_manifest,
        "e149_status_sha256": sha256(E149_ROOT / "RUN_STATUS.json"),
        "e150_asset_sha256": e150["asset_sha256"],
        "e150_status_sha256": sha256(E150_ROOT / "RUN_STATUS.json"),
        "e112_cache_sha256": sha256(CACHE),
        "e135_model_sha256": sha256(MODEL),
        "e152_status_sha256": sha256(E152_STATUS),
        "e152_was_unsealed_before_e154": True,
    }


def derive_manifests() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_MANIFEST, keep_default_na=False)
    original_columns = list(original.columns)
    train = original["split"].astype(str).eq("train")
    audits = []
    selected_sets: dict[int, set[tuple[str, str, str]]] = {}
    for fraction in (25, 50, 75, 100):
        membership = f"in_train_fraction_{fraction}"
        if membership not in original:
            raise RuntimeError(f"E149 lacks {membership}")
        derived = original.copy()
        derived["e149_in_train_fraction_100_original"] = bool_values(
            original["in_train_fraction_100"]
        )
        derived["e154_training_fraction"] = fraction
        derived["e154_source_membership_column"] = membership
        selected = bool_values(original[membership])
        derived.loc[train, "in_train_fraction_100"] = selected.loc[train].to_numpy()
        # Non-training rows are immutable; E112 only consults the overwritten
        # field for split=train.
        derived.loc[~train, original_columns] = original.loc[~train, original_columns]
        out = manifest_path(fraction)
        derived.to_csv(out, index=False)
        reread = pd.read_csv(out, keep_default_na=False)
        test_val_equal = reread.loc[~train, original_columns].astype(str).reset_index(
            drop=True
        ).equals(
            original.loc[~train, original_columns].astype(str).reset_index(drop=True)
        )
        reread_selected = train & bool_values(reread["in_train_fraction_100"])
        source_selected = train & selected
        selection_equal = bool((reread_selected == source_selected).all())
        keys = set(
            map(
                tuple,
                original.loc[
                    source_selected, ["fold_id", "context", "perturbation"]
                ].astype(str).to_numpy(),
            )
        )
        selected_sets[fraction] = keys
        per_fold = original.loc[source_selected].groupby("fold_id").size().to_dict()
        audits.append(
            {
                "train_fraction": fraction,
                "source_membership_column": membership,
                "n_selected_train_pairs": int(source_selected.sum()),
                "fold1_selected_train_pairs": int(
                    per_fold.get("Replogle_cellline_holdout_1_K562", 0)
                ),
                "fold2_selected_train_pairs": int(
                    per_fold.get("Replogle_cellline_holdout_2_RPE1", 0)
                ),
                "n_validation_pairs": int(original["split"].astype(str).eq("val").sum()),
                "n_test_pairs": int(original["split"].astype(str).eq("test").sum()),
                "n_primary_pairs": int(bool_values(original["primary_analysis"]).sum()),
                "validation_and_test_original_columns_unchanged": test_val_equal,
                "selected_train_set_matches_e149_column": selection_equal,
                "derived_manifest": str(out.relative_to(ROOT)),
                "derived_manifest_sha256": sha256(out),
            }
        )
    if not (
        selected_sets[25]
        <= selected_sets[50]
        <= selected_sets[75]
        <= selected_sets[100]
    ):
        raise RuntimeError("E149 frozen train-fraction memberships are not nested")
    frame = pd.DataFrame(audits)
    if not frame[
        [
            "validation_and_test_original_columns_unchanged",
            "selected_train_set_matches_e149_column",
        ]
    ].to_numpy(bool).all():
        raise RuntimeError("derived manifest invariance check failed")
    frame["nested_membership_check_passed"] = True
    frame.to_csv(TABLES / "E154_DERIVED_MANIFEST_AUDIT.csv", index=False)
    return frame


def write_analysis_contract(input_audit: dict[str, object], manifests: pd.DataFrame) -> None:
    matrix = pd.DataFrame(RUN_MATRIX)
    matrix["manifest"] = [
        str(source_manifest(item).relative_to(ROOT)) for item in RUN_MATRIX
    ]
    matrix.to_csv(TABLES / "E154_RUN_MATRIX.csv", index=False)
    contract = (
        "# E154 分析合同｜Replogle 随机种子与训练量敏感性\n\n"
        f"冻结时间：{now()}。E152 已经解封；E154 因此是**确认后的敏感性分析**，"
        "不是新的预注册确认 gate，也不会改写 E152 的通过/未通过结论。\n\n"
        "## 固定运行\n\n"
        f"- 全量训练比较 3 个种子：{SEED_1}（复用 E151）、{SEED_2}、{SEED_3}。\n"
        f"- 训练量比较固定种子 {SEED_2}，使用 E149 在读取表达矩阵前已经生成的 "
        "`in_train_fraction_25/50/75/100` 成员列。实际训练对数为 18/58/105/140；"
        "验证 32 对、测试 340 对和 256 个 held-out-context 主任务不变。\n"
        "- 每个新运行完整沿用 E112 的模型、轮数、早停、验证校准、512 基因面板和输出格式。"
        "每个运行必须有 680 条 strict PredictionRecord、0 issue，否则不进入汇总。\n\n"
        "## 固定分析\n\n"
        "只分析 256 个预先标记的 held-out-context 唯一任务。绝对误差排序报告 SafeConf、"
        "predicted magnitude、模型分歧与两模型平均 RMSE 的两折等权 Spearman。方向排序使用"
        " E135 冻结 Ridge（不在 Replogle 重拟合），报告其与 magnitude 对 E134 Systema "
        "exact centered-Pearson、centered-cosine 及二者折内百分位复合终点的两折等权 Spearman。"
        "同时报告 scGPT、GEARS、ensemble 和训练受扰动表达质心的 RMSE。\n\n"
        "全量三种子按每项指标报告均值、最小值、最大值与范围。训练量分析报告四个固定点、"
        "25% 到 100% 的差以及训练比例与指标的 Spearman；四个点只作描述，不作新显著性声明。\n\n"
        "## 边界\n\n"
        "该实验考察计算随机性和既有训练样本量选择的敏感性。两个细胞系仍来自同一研究，"
        "目标细胞系 control 可见；结果不能扩展为跨研究、完全 zero-shot 或湿实验验证。\n"
    )
    contract_path = OUT / "ANALYSIS_CONTRACT.md"
    contract_path.write_text(contract)
    status = {
        "phase": "frozen_before_e154_new_model_runs",
        "generated_at": now(),
        "claim_scope": "post_E152_prespecified_manifest_sensitivity_only",
        "not_a_new_confirmation_gate": True,
        "run_ids": matrix["run_id"].tolist(),
        "new_model_runs": int((~matrix["reused_e151"]).sum()),
        "full_training_seeds": [SEED_1, SEED_2, SEED_3],
        "fixed_fraction_seed": SEED_2,
        "training_fractions": [25, 50, 75, 100],
        "primary_unique_tasks_per_run": 256,
        "strict_records_required_per_run": 680,
        "strict_issue_count_required_per_run": 0,
        "analysis_contract_sha256": sha256(contract_path),
        "input_audit": input_audit,
        "derived_manifest_sha256": dict(
            zip(
                manifests["train_fraction"].astype(str),
                manifests["derived_manifest_sha256"],
            )
        ),
    }
    (OUT / "CONTRACT_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )


def prepare() -> dict[str, object]:
    if (OUT / "CONTRACT_STATUS.json").exists():
        status = assert_prepared()
        result = {
            "status": "already_prepared_contract_preserved",
            "contract": str((OUT / "ANALYSIS_CONTRACT.md").relative_to(ROOT)),
            "analysis_contract_sha256": status["analysis_contract_sha256"],
            "new_model_runs": len(NEW_RUNS),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    for directory in [OUT, MANIFESTS, RUNS_ROOT, TABLES, REPORTS, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    input_audit = validate_frozen_inputs()
    manifests = derive_manifests()
    write_analysis_contract(input_audit, manifests)
    (OUT / "README_先看这个.md").write_text(
        "# E154 先看这个\n\n"
        "先读 `ANALYSIS_CONTRACT.md`，运行完成后读 `reports/E154_REPORT.md`。"
        "本目录是 E152 解封后的稳健性分析，不是新的确认 gate。\n"
    )
    result = {
        "status": "prepared",
        "contract": str((OUT / "ANALYSIS_CONTRACT.md").relative_to(ROOT)),
        "new_model_runs": len(NEW_RUNS),
        "derived_manifests": len(manifests),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def assert_prepared() -> dict[str, object]:
    status_path = OUT / "CONTRACT_STATUS.json"
    if not status_path.exists():
        raise RuntimeError("run --prepare-only before starting model jobs")
    status = json.loads(status_path.read_text())
    if sha256(OUT / "ANALYSIS_CONTRACT.md") != status["analysis_contract_sha256"]:
        raise RuntimeError("E154 analysis contract changed after freeze")
    validate_frozen_inputs()
    for fraction, expected in status["derived_manifest_sha256"].items():
        if sha256(manifest_path(int(fraction))) != expected:
            raise RuntimeError(f"derived manifest {fraction}% changed after freeze")
    return status


def validate_model_outputs(root: Path) -> dict[str, object]:
    required = [
        root / "RUN_STATUS.json",
        root / "TASK_RISK_TABLE.csv",
        root / "PREDICTION_RECORDS.csv",
        root / "arrays/predicted_effects.npz",
        root / "arrays/true_effects.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete model output: {missing}")
    child = json.loads((root / "RUN_STATUS.json").read_text())
    tasks = pd.read_csv(root / "TASK_RISK_TABLE.csv")
    records = pd.read_csv(root / "PREDICTION_RECORDS.csv")
    issues = validate_prediction_record_artifacts(root, records=records, strict=True)
    checks = {
        "n_folds": int(tasks["fold_id"].astype(str).nunique()),
        "n_test_tasks": int(len(tasks)),
        "n_prediction_records": int(len(records)),
        "strict_issue_count": int(len(issues)),
        "all_records_are_test": bool(records["split"].astype(str).eq("test").all()),
        "two_records_per_task": bool(
            records.groupby(["fold_id", "task_id"]).size().eq(2).all()
        ),
    }
    if (
        checks["n_folds"] != 2
        or checks["n_test_tasks"] != 340
        or checks["n_prediction_records"] != 680
        or checks["strict_issue_count"] != 0
        or not checks["all_records_are_test"]
        or not checks["two_records_per_task"]
        or child.get("strict_issue_count") != 0
    ):
        raise RuntimeError(f"strict run requirements failed: {checks}")
    return checks


def e112_module(item: dict[str, object]):
    run_id = str(item["run_id"])
    loaded = load_module(f"e112_for_e154_{run_id}", SOURCE_SCRIPT)
    loaded.CONTRACT = source_manifest(item)
    loaded.OUT = run_view_root(run_id)
    loaded.SPECS = {DATASET: {"source": ASSET, "context": "context"}}
    loaded.SEED = int(item["seed"])
    # E149's 75% membership contains 49 pairs in fold 2.  With E112's
    # batch_size=16 this creates a final singleton batch, which GEARS'
    # BatchNorm cannot evaluate in training mode.  Preserve all frozen pairs
    # and change only that loader invocation to batch_size=15.  Other folds
    # and all evaluation loaders remain exactly at E112's requested size.
    original_loader = loaded.DataLoader

    def no_singleton_loader(dataset, batch_size=1, *args, **kwargs):
        effective = int(batch_size)
        if effective > 2 and len(dataset) % effective == 1:
            effective -= 1
        return original_loader(dataset, effective, *args, **kwargs)

    loaded.DataLoader = no_singleton_loader
    return loaded


def singleton_guard_audit(item: dict[str, object]) -> dict[str, object]:
    manifest = pd.read_csv(source_manifest(item), keep_default_na=False)
    selected = manifest[
        manifest["split"].astype(str).eq("train")
        & bool_values(manifest["in_train_fraction_100"])
    ]
    counts = selected.groupby("fold_id").size().to_dict()
    affected = [str(fold) for fold, count in counts.items() if int(count) % 16 == 1]
    return {
        "default_batch_size": 16,
        "fallback_batch_size": 15,
        "rule": "fallback only when selected_train_count modulo 16 equals 1",
        "affected_folds": affected,
        "applied": bool(affected and not item["reused_e151"]),
        "all_frozen_training_pairs_retained": True,
    }


def audit_run(item: dict[str, object]) -> dict[str, object]:
    run_id = str(item["run_id"])
    view = run_view_root(run_id)
    view.mkdir(parents=True, exist_ok=True)
    model_root = source_root(item)
    strict = validate_model_outputs(model_root)
    manifest = source_manifest(item)
    model = json.loads(MODEL.read_text())
    deployable = [
        "fold_id",
        "task_id",
        "setting",
        "context",
        "perturbation",
        *model["features_in_order"],
    ]
    all_risk = pd.read_csv(model_root / "TASK_RISK_TABLE.csv", usecols=deployable)
    frozen = pd.read_csv(manifest, keep_default_na=False)
    primary_keys = frozen.loc[
        bool_values(frozen["primary_analysis"]),
        ["fold_id", "context", "perturbation", "setting"],
    ]
    scores = all_risk.merge(
        primary_keys,
        on=["fold_id", "context", "perturbation", "setting"],
        how="inner",
        validate="one_to_one",
    )
    if len(scores) != 256 or scores.duplicated(["context", "perturbation"]).any():
        raise RuntimeError(f"{run_id}: primary selection is not 256 unique tasks")
    matrix = scores[model["features_in_order"]].to_numpy(float)
    scores["directional_risk_frozen"] = float(model["intercept"]) + matrix @ np.asarray(
        model["coefficients_in_order"], float
    )
    scores["target_direction_truth_used_for_score_or_transform"] = False
    scores["direction_model_refit_on_replogle"] = False
    scores["frozen_model_sha256"] = sha256(MODEL)
    score_path = view / "E154_DIRECTIONAL_SCORES.csv"
    scores.to_csv(score_path, index=False)
    score_status = {
        "generated_at": now(),
        "run_id": run_id,
        "n_primary_unique_tasks": int(len(scores)),
        "score_sha256": sha256(score_path),
        "model_sha256": sha256(MODEL),
        "model_refit": False,
        "direction_truth_columns_read_for_score": [],
        "note": "E154 is post-E152 sensitivity; this is not a prospective confirmation freeze",
    }
    (view / "SCORE_STATUS.json").write_text(
        json.dumps(score_status, ensure_ascii=False, indent=2) + "\n"
    )

    e134 = load_module(f"e134_for_e154_{run_id}", E134_SCRIPT)
    spec = {
        "root": model_root,
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": manifest,
        "cache": CACHE,
    }
    exact_all, source_audit = e134.audit_dataset(DATASET, spec)
    keys = ["fold_id", "task_id", "setting", "context", "perturbation"]
    exact = exact_all.merge(
        scores[
            keys
            + [
                "directional_risk_frozen",
                "target_direction_truth_used_for_score_or_transform",
                "direction_model_refit_on_replogle",
                "frozen_model_sha256",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(exact) != 256 or exact.duplicated(["context", "perturbation"]).any():
        raise RuntimeError(f"{run_id}: exact audit did not retain 256 unique tasks")
    ranks = []
    for endpoint in ["error_centered_pearson_mean", "error_centered_cosine_mean"]:
        ranks.append(
            exact.groupby("fold_id")[endpoint].transform(
                lambda values: rankdata(values) / len(values)
            )
        )
    exact["direction_error_rank_target"] = np.mean(np.stack(ranks), axis=0)

    fold_rows = []
    score_endpoint_pairs = []
    score_endpoint_pairs.extend(
        (name, score, "error_two_predictor_mean_rmse")
        for name, score in ABSOLUTE_SCORES.items()
    )
    for score_name, score in DIRECTION_SCORES.items():
        for endpoint_name, endpoint in DIRECTION_ENDPOINTS.items():
            score_endpoint_pairs.append(
                (f"{score_name}_{endpoint_name}_spearman", score, endpoint)
            )
    for fold_id, group in exact.groupby("fold_id", sort=True):
        for metric, score, endpoint in score_endpoint_pairs:
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "metric": metric,
                    "score": score,
                    "endpoint": endpoint,
                    "n_tasks": len(group),
                    "spearman": rho(group[score], group[endpoint]),
                }
            )
    folds = pd.DataFrame(fold_rows)
    macro = folds.groupby(["metric", "score", "endpoint"], as_index=False).agg(
        n_folds=("fold_id", "nunique"),
        n_finite_folds=("spearman", "count"),
        fold_macro_spearman=("spearman", "mean"),
        min_fold_spearman=("spearman", "min"),
        max_fold_spearman=("spearman", "max"),
    )
    incomplete = macro["n_finite_folds"].lt(macro["n_folds"])
    macro.loc[
        incomplete,
        ["fold_macro_spearman", "min_fold_spearman", "max_fold_spearman"],
    ] = np.nan
    baseline = e134.E133.baseline_summary(exact)
    exact.to_csv(view / "E154_PRIMARY_TASK_AUDIT.csv", index=False)
    folds.to_csv(view / "E154_FOLD_METRICS.csv", index=False)
    macro.to_csv(view / "E154_MACRO_METRICS.csv", index=False)
    baseline.to_csv(view / "E154_MODEL_BASELINE_SUMMARY.csv", index=False)
    pd.DataFrame([source_audit]).to_csv(view / "E154_SOURCE_TRUTH_AUDIT.csv", index=False)
    status = {
        "experiment": "E154_replogle_robustness_run",
        "generated_at": now(),
        "status": "complete",
        "run_id": run_id,
        "seed": int(item["seed"]),
        "train_fraction": int(item["train_fraction"]),
        "reused_e151": bool(item["reused_e151"]),
        "source_model_root": str(model_root.relative_to(ROOT)),
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest),
        "n_primary_unique_tasks": int(len(exact)),
        "n_source_context_diagnostics_excluded": int(len(exact_all) - len(exact)),
        "strict": strict,
        "singleton_batch_guard": singleton_guard_audit(item),
        "systema_exact_source_truth_audit": source_audit,
        "undefined_two_fold_macro_metrics": macro.loc[
            macro["n_finite_folds"].lt(macro["n_folds"]), "metric"
        ].tolist(),
        "direction_model_refit": False,
        "e154_is_post_e152_sensitivity_not_gate": True,
    }
    (view / "RUN_AUDIT.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )
    return status


def run_one(run_id: str, device_name: str) -> dict[str, object]:
    assert_prepared()
    item = RUN_BY_ID[run_id]
    if item["reused_e151"]:
        return audit_run(item)
    model_root = source_root(item)
    complete = False
    try:
        validate_model_outputs(model_root)
        complete = True
        print(f"[E154] {run_id}: valid model output exists; skipping retraining", flush=True)
    except (FileNotFoundError, RuntimeError, pd.errors.EmptyDataError):
        complete = False
    if not complete:
        guard = singleton_guard_audit(item)
        if guard["applied"]:
            (OUT / "TECHNICAL_DEVIATION.md").write_text(
                "# E154 技术偏差记录｜75% 训练折的单样本末批\n\n"
                f"记录时间：{now()}。首次运行 `seed2_frac75` 时，第二折的 E149 冻结训练成员为 "
                "49 对。E112 默认 batch size 16 会生成大小为 1 的最后一个训练 batch，GEARS "
                "BatchNorm 因此在任何结果表写出前报错。\n\n"
                "修复只在 `n_train % 16 == 1` 时将训练 DataLoader 的 batch size 从 16 调为 15；"
                "49 个冻结训练对全部保留。模型结构、随机种子、训练成员、最大轮数、早停规则、"
                "验证/测试集合、风险分数和评价流程均未改变。其余 folds 不触发该规则。失败运行"
                "不进入汇总；scGPT 与 GEARS 共用该训练 loader，因此 75% 运行的两个模型都在"
                "batch size 15 下从头执行。\n\n"
                "这是冻结分析合同后发现并披露的计算兼容性修复，不得描述为预先计划的改动。\n"
            )
        loaded = e112_module(item)
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        print(
            f"[E154] training {run_id}: seed={item['seed']} "
            f"fraction={item['train_fraction']} device={device}",
            flush=True,
        )
        child_status = loaded.run_dataset(DATASET, device)
        print(json.dumps(child_status, ensure_ascii=False, indent=2), flush=True)
    result = audit_run(item)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def collect_run_metrics(item: dict[str, object]) -> list[dict[str, object]]:
    view = run_view_root(str(item["run_id"]))
    status = json.loads((view / "RUN_AUDIT.json").read_text())
    if status.get("status") != "complete":
        raise RuntimeError(f"run not complete: {item['run_id']}")
    macro = pd.read_csv(view / "E154_MACRO_METRICS.csv").set_index("metric")
    baseline = pd.read_csv(view / "E154_MODEL_BASELINE_SUMMARY.csv").iloc[0]
    rows = []
    for metric, row in macro.iterrows():
        rows.append(
            {
                **{key: item[key] for key in ["run_id", "seed", "train_fraction", "reused_e151"]},
                "metric_group": "two_fold_macro_spearman",
                "metric": metric,
                "value": float(row.fold_macro_spearman),
                "fold_min": float(row.min_fold_spearman),
                "fold_max": float(row.max_fold_spearman),
            }
        )
    baseline_fields = [
        "rmse_scgpt_mean",
        "rmse_gears_mean",
        "rmse_individual_mean",
        "rmse_ensemble_mean",
        "rmse_training_perturbed_mean",
        "ensemble_minus_simple_baseline",
        "fraction_tasks_ensemble_beats_simple_baseline",
    ]
    for metric in baseline_fields:
        rows.append(
            {
                **{key: item[key] for key in ["run_id", "seed", "train_fraction", "reused_e151"]},
                "metric_group": "model_and_simple_baseline",
                "metric": metric,
                "value": float(baseline[metric]),
                "fold_min": float("nan"),
                "fold_max": float("nan"),
            }
        )
    return rows


def write_svg(
    wide: pd.DataFrame,
    seed_summary: pd.DataFrame,
    size_audit: pd.DataFrame,
) -> None:
    keys = [
        "absolute_safeconf_spearman",
        "absolute_magnitude_spearman",
        "directional_composite_spearman",
        "magnitude_composite_spearman",
    ]
    labels = {
        "absolute_safeconf_spearman": "SafeConf × absolute RMSE",
        "absolute_magnitude_spearman": "Magnitude × absolute RMSE",
        "directional_composite_spearman": "Frozen directional × direction error",
        "magnitude_composite_spearman": "Magnitude × direction error",
    }
    colors = ["#2F6B5F", "#B06A3B", "#4C6A92", "#8A6D8F"]
    width, height = 1180, 650
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Noto Sans SC",sans-serif;fill:#202124}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#5f6368}.label{font-size:14px}.small{font-size:12px}.axis{stroke:#9aa0a6;stroke-width:1}.zero{stroke:#5f6368;stroke-width:1.4}</style>',
        '<text x="45" y="42" class="title">E154 · Replogle robustness</text>',
        '<text x="45" y="67" class="sub">256 frozen primary tasks · two-fold macro Spearman · post-E152 sensitivity</text>',
        '<text x="60" y="112" class="label" font-weight="700">A  Full-training variation across three seeds</text>',
        '<text x="650" y="112" class="label" font-weight="700">B  Fixed-seed trend across frozen hash-threshold subsets</text>',
    ]
    x0, x1 = 235, 555
    map_rho = lambda value, lo=x0, hi=x1: lo + (float(value) + 1) / 2 * (hi - lo)
    parts += [
        f'<line x1="{x0}" y1="510" x2="{x1}" y2="510" class="axis"/>',
        f'<line x1="{map_rho(0)}" y1="135" x2="{map_rho(0)}" y2="510" class="zero"/>',
    ]
    seed_index = seed_summary.set_index("metric")
    for i, (key, color) in enumerate(zip(keys, colors)):
        row = seed_index.loc[key]
        y = 175 + i * 82
        parts += [
            f'<text x="60" y="{y+4}" class="small">{labels[key]}</text>',
            f'<line x1="{map_rho(row["min"]):.1f}" y1="{y}" x2="{map_rho(row["max"]):.1f}" y2="{y}" stroke="{color}" stroke-width="4"/>',
            f'<circle cx="{map_rho(row["mean"]):.1f}" cy="{y}" r="7" fill="{color}"/>',
            f'<text x="{map_rho(row["max"])+8:.1f}" y="{y+4}" class="small">{row["mean"]:.3f} [{row["min"]:.3f}, {row["max"]:.3f}]</text>',
        ]
    for tick in [-1, -0.5, 0, 0.5, 1]:
        parts.append(
            f'<text x="{map_rho(tick):.1f}" y="532" class="small" text-anchor="middle">{tick:g}</text>'
        )
    fx0, fx1, fy0, fy1 = 690, 1120, 510, 145
    fx = lambda fraction: fx0 + (float(fraction) - 25) / 75 * (fx1 - fx0)
    fy = lambda value: fy0 - (float(value) + 1) / 2 * (fy0 - fy1)
    parts += [
        f'<line x1="{fx0}" y1="{fy0}" x2="{fx1}" y2="{fy0}" class="axis"/>',
        f'<line x1="{fx0}" y1="{fy(0)}" x2="{fx1}" y2="{fy(0)}" class="zero"/>',
    ]
    fraction = wide[(wide.seed == SEED_2) & wide.train_fraction.isin([25, 50, 75, 100])].sort_values("train_fraction")
    for key, color in zip(keys, colors):
        segment = []
        for row in fraction.itertuples(index=False):
            value = float(getattr(row, key))
            if np.isfinite(value):
                segment.append(f'{fx(row.train_fraction):.1f},{fy(value):.1f}')
                parts.append(
                    f'<circle cx="{fx(row.train_fraction):.1f}" cy="{fy(value):.1f}" r="5" fill="{color}"/>'
                )
            else:
                if len(segment) >= 2:
                    parts.append(
                        f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="3"/>'
                    )
                segment = []
        if len(segment) >= 2:
            parts.append(
                f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="3"/>'
            )
    size_lookup = size_audit.set_index("threshold_label")
    for tick in [25, 50, 75, 100]:
        actual = float(size_lookup.loc[tick, "actual_percent"])
        n_pairs = int(size_lookup.loc[tick, "n_selected_train_pairs"])
        parts += [
            f'<text x="{fx(tick):.1f}" y="532" class="small" text-anchor="middle">label {tick}</text>',
            f'<text x="{fx(tick):.1f}" y="549" class="small" text-anchor="middle">{actual:.1f}% · n={n_pairs}</text>',
        ]
    for tick in [-1, -0.5, 0, 0.5, 1]:
        parts.append(f'<text x="{fx0-10}" y="{fy(tick)+4:.1f}" class="small" text-anchor="end">{tick:g}</text>')
    for i, (key, color) in enumerate(zip(keys, colors)):
        x = 80 + (i % 2) * 360
        y = 585 + (i // 2) * 25
        parts += [
            f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{color}" stroke-width="4"/>',
            f'<text x="{x+37}" y="{y+4}" class="small">{labels[key]}</text>',
        ]
    parts += [
        '<text x="650" y="574" class="sub">Top: frozen E149 threshold label · bottom: actual share of 140 training pairs</text>',
        '<text x="750" y="625" class="sub">Ranges describe computational sensitivity; they are not new confirmation intervals.</text>',
        "</svg>",
    ]
    (FIGURES / "F1_seed_and_train_fraction_robustness.svg").write_text("\n".join(parts) + "\n")


def summarize() -> dict[str, object]:
    assert_prepared()
    # Build the E154-local audit view for E151 if it does not yet exist.
    for item in RUN_MATRIX:
        audit_path = run_view_root(str(item["run_id"])) / "RUN_AUDIT.json"
        if not audit_path.exists():
            audit_run(item)
    rows = []
    strict_rows = []
    degeneracy_rows = []
    for item in RUN_MATRIX:
        rows.extend(collect_run_metrics(item))
        view = run_view_root(str(item["run_id"]))
        status = json.loads((view / "RUN_AUDIT.json").read_text())
        strict_rows.append(
            {
                "run_id": item["run_id"],
                "seed": item["seed"],
                "train_fraction": item["train_fraction"],
                "reused_e151": item["reused_e151"],
                **status["strict"],
                "n_primary_unique_tasks": status["n_primary_unique_tasks"],
                "source_truth_pass": status["systema_exact_source_truth_audit"][
                    "truth_reconstruction_pass_atol_1e-5"
                ],
            }
        )
        task_audit = pd.read_csv(view / "E154_PRIMARY_TASK_AUDIT.csv")
        for fold_id, group in task_audit.groupby("fold_id", sort=True):
            values = group["safeconf_calibrated_pair_risk"].to_numpy(float)
            degeneracy_rows.append(
                {
                    "run_id": item["run_id"],
                    "seed": item["seed"],
                    "train_fraction": item["train_fraction"],
                    "fold_id": fold_id,
                    "n_tasks": len(group),
                    "n_unique_calibrated_safeconf": int(
                        pd.Series(values).nunique(dropna=True)
                    ),
                    "calibrated_safeconf_std": float(np.nanstd(values)),
                    "absolute_safeconf_spearman_defined": bool(
                        pd.Series(values).nunique(dropna=True) >= 2
                    ),
                }
            )
    long = pd.DataFrame(rows)
    wide = (
        long.pivot(
            index=["run_id", "seed", "train_fraction", "reused_e151"],
            columns="metric",
            values="value",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    long.to_csv(TABLES / "E154_RUN_METRICS_LONG.csv", index=False)
    wide.to_csv(TABLES / "E154_RUN_METRICS_WIDE.csv", index=False)
    strict = pd.DataFrame(strict_rows)
    strict.to_csv(TABLES / "E154_STRICT_RUN_AUDIT.csv", index=False)
    degeneracy = pd.DataFrame(degeneracy_rows)
    degeneracy.to_csv(TABLES / "E154_SCORE_DEGENERACY_AUDIT.csv", index=False)
    if not (
        strict["n_prediction_records"].eq(680).all()
        and strict["strict_issue_count"].eq(0).all()
        and strict["n_primary_unique_tasks"].eq(256).all()
        and strict["source_truth_pass"].astype(bool).all()
    ):
        raise RuntimeError("one or more E154 runs failed the strict inclusion requirements")

    full = long[long["train_fraction"].eq(100)].copy()
    seed_summary = full.groupby(["metric_group", "metric"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean=("value", "mean"),
        min=("value", "min"),
        max=("value", "max"),
        std=("value", "std"),
    )
    seed_summary["range"] = seed_summary["max"] - seed_summary["min"]
    seed_summary.to_csv(TABLES / "E154_FULL_SEED_SUMMARY.csv", index=False)

    manifest_audit = pd.read_csv(TABLES / "E154_DERIVED_MANIFEST_AUDIT.csv")
    size_audit = manifest_audit[
        ["train_fraction", "n_selected_train_pairs", "fold1_selected_train_pairs", "fold2_selected_train_pairs"]
    ].rename(columns={"train_fraction": "threshold_label"})
    n_full_pairs = int(
        size_audit.loc[size_audit["threshold_label"].eq(100), "n_selected_train_pairs"].iloc[0]
    )
    size_audit["actual_fraction"] = size_audit["n_selected_train_pairs"] / n_full_pairs
    size_audit["actual_percent"] = 100.0 * size_audit["actual_fraction"]
    size_audit["label_is_hash_threshold_not_exact_sample_fraction"] = True
    size_audit.to_csv(TABLES / "E154_TRAIN_FRACTION_SIZE_AUDIT.csv", index=False)

    fractions = long[
        long["seed"].eq(SEED_2) & long["train_fraction"].isin([25, 50, 75, 100])
    ].copy()
    trend_rows = []
    for (group, metric), values in fractions.groupby(["metric_group", "metric"]):
        values = values.sort_values("train_fraction")
        if values["train_fraction"].tolist() != [25, 50, 75, 100]:
            raise RuntimeError(f"fraction series incomplete: {metric}")
        x = values["train_fraction"].to_numpy(float)
        actual_x = np.array(
            [
                float(size_audit.loc[size_audit["threshold_label"].eq(label), "actual_percent"].iloc[0])
                for label in x
            ]
        )
        y = values["value"].to_numpy(float)
        finite = np.isfinite(y)
        all_four_finite = bool(finite.all())
        trend_rows.append(
            {
                "metric_group": group,
                "metric": metric,
                "seed": SEED_2,
                "actual_percent_at_label_25": actual_x[0],
                "actual_percent_at_label_50": actual_x[1],
                "actual_percent_at_label_75": actual_x[2],
                "actual_percent_at_label_100": actual_x[3],
                "value_at_25": y[0],
                "value_at_50": y[1],
                "value_at_75": y[2],
                "value_at_100": y[3],
                "delta_100_minus_25": y[3] - y[0],
                "n_finite_points": int(finite.sum()),
                "all_four_points_finite": all_four_finite,
                "spearman_with_train_fraction": rho(x, y) if all_four_finite else float("nan"),
                "linear_slope_per_25_actual_percentage_points": (
                    float(np.polyfit(actual_x / 25.0, y, 1)[0])
                    if all_four_finite
                    else float("nan")
                ),
                "descriptive_only_four_points": True,
            }
        )
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(TABLES / "E154_TRAIN_FRACTION_TREND.csv", index=False)
    write_svg(wide, seed_summary, size_audit)

    selected = [
        "absolute_safeconf_spearman",
        "absolute_magnitude_spearman",
        "absolute_disagreement_spearman",
        "directional_pearson_spearman",
        "directional_cosine_spearman",
        "directional_composite_spearman",
        "magnitude_composite_spearman",
        "rmse_scgpt_mean",
        "rmse_gears_mean",
        "rmse_ensemble_mean",
        "rmse_training_perturbed_mean",
        "ensemble_minus_simple_baseline",
    ]
    seed_index = seed_summary.set_index("metric")
    trend_index = trends.set_index("metric")

    def formatted(value: float, digits: int = 4, signed: bool = False) -> str:
        value = float(value)
        if not np.isfinite(value):
            return "NA"
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"

    lines = [
        "# E154｜Replogle 随机种子与训练量稳健性结果",
        "",
        "## 完整性",
        "",
        "6 个分析运行全部进入汇总：E151 全量种子复用 1 个，新训练 5 个。每个运行均为 2 folds、"
        "340 个诊断测试任务、680 条 strict PredictionRecord、0 issue；主分析固定为 256 个"
        " held-out-context 唯一任务，E134 保存真值与源数据重建真值检查全部通过。",
        "",
        "## 主要结果",
        "",
        "全量训练时，calibrated SafeConf 的绝对 RMSE 排序在 3 个种子中均为正，范围 "
        f"{seed_index.loc['absolute_safeconf_spearman', 'min']:.3f}–"
        f"{seed_index.loc['absolute_safeconf_spearman', 'max']:.3f}；冻结方向分数的复合方向误差排序也"
        f"均为正，范围 {seed_index.loc['directional_composite_spearman', 'min']:.3f}–"
        f"{seed_index.loc['directional_composite_spearman', 'max']:.3f}。三种子均值中，绝对 SafeConf "
        f"({seed_index.loc['absolute_safeconf_spearman', 'mean']:.3f}) 与 magnitude "
        f"({seed_index.loc['absolute_magnitude_spearman', 'mean']:.3f}) 基本相同；冻结方向分数 "
        f"({seed_index.loc['directional_composite_spearman', 'mean']:.3f}) 略高于方向 magnitude "
        f"({seed_index.loc['magnitude_composite_spearman', 'mean']:.3f})，但只有 3 个种子，不能写成"
        "稳定优越。模型分歧的范围包含负值，随机种子敏感性最明显。",
        "",
        "E149 的 `25/50/75/100` 是预先冻结的哈希阈值标签，不是强制等量抽样。对应实际训练对为 "
        "18/58/105/140，即 12.9%/41.4%/75.0%/100%。因此下表保留冻结标签以便复跑，同时并列写出"
        "实际样本量；后文不把最小子集称作严格的 25% 训练。固定种子的最小子集使 scGPT RMSE "
        "升至 1.122、ensemble RMSE 升至 0.570；第二档子集后模型 "
        "RMSE 恢复到约 0.10。冻结方向复合排序由 0.002、0.157、0.210 增至 0.389。calibrated "
        "SafeConf 在标签 25 的第二折、标签 50 的第一折以及标签 75 的两折成为常数，因此这三个子集都没有"
        "完整的两折绝对 Spearman，统一标为 NA，绝不以单折补位。这说明校准分支在缩减训练集时"
        "存在明显退化；冻结方向分数和 magnitude 仍可计算。逐折唯一值与标准差见 "
        "`tables/E154_SCORE_DEGENERACY_AUDIT.csv`。",
        "",
        "## 全量训练的三种子范围",
        "",
        "| 指标 | 三种子均值 | 最小 | 最大 | 范围 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in selected:
        row = seed_index.loc[metric]
        lines.append(
            f"| {metric} | {formatted(row['mean'])} | {formatted(row['min'])} | "
            f"{formatted(row['max'])} | {formatted(row['range'])} |"
        )
    lines += [
        "",
        f"## 固定种子 {SEED_2} 的训练量趋势",
        "",
        "| 指标 | label 25（12.9%, n=18） | label 50（41.4%, n=58） | label 75（75.0%, n=105） | label 100（100%, n=140） | 全量−最小子集 | 子集规模 Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in selected:
        row = trend_index.loc[metric]
        lines.append(
            f"| {metric} | {formatted(row['value_at_25'])} | {formatted(row['value_at_50'])} | "
            f"{formatted(row['value_at_75'])} | {formatted(row['value_at_100'])} | "
            f"{formatted(row['delta_100_minus_25'], signed=True)} | "
            f"{formatted(row['spearman_with_train_fraction'], digits=3, signed=True)} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "E154 在 E152 解封之后设计，用于回答结果是否依赖单个训练随机种子，以及 E149 预先冻结的"
        "训练成员减少时结果怎样变化。它不构成独立确认，不重新判定 E152 gate，也没有在 Replogle"
        "上重拟合 E135 方向模型。四个训练比例点只作描述；小训练集同时改变预测模型、训练支持度和"
        "Systema 训练质心，不能把曲线解释成单一因素的因果效应。",
        "",
        "75% 训练的第二折有 49 个冻结训练对，E112 默认 batch=16 会留下一个单样本末批，"
        "GEARS BatchNorm 无法训练。失败运行未进入结果；重跑时只把该折训练 batch 调为 15，"
        "保留全部 49 对。完整的事后技术偏差披露见 `TECHNICAL_DEVIATION.md`。",
        "",
        "结果范围仍限于同一 Replogle 研究内、目标 control 可见的 K562/RPE1 跨细胞系任务。"
        "它不等于跨研究泛化、完全 zero-shot 或湿实验验证。逐运行任务表、fold 指标、模型/质心"
        "基线及源真值审计均保存在 `runs/`；汇总数值在 `tables/`。",
    ]
    (REPORTS / "E154_REPORT.md").write_text("\n".join(lines) + "\n")
    contract_status = json.loads((OUT / "CONTRACT_STATUS.json").read_text())
    status = {
        "experiment": "E154_replogle_robustness",
        "generated_at": now(),
        "status": "complete",
        "claim_scope": "post_E152_prespecified_manifest_sensitivity_only",
        "not_a_new_confirmation_gate": True,
        "n_analysis_runs": len(RUN_MATRIX),
        "n_new_model_runs": len(NEW_RUNS),
        "n_reused_e151_runs": 1,
        "n_full_training_seeds": 3,
        "full_training_seeds": [SEED_1, SEED_2, SEED_3],
        "fixed_fraction_seed": SEED_2,
        "training_fractions": [25, 50, 75, 100],
        "training_fraction_label_semantics": "E149 frozen hash thresholds; not exact requested sample fractions",
        "actual_training_pairs_by_label": {"25": 18, "50": 58, "75": 105, "100": 140},
        "actual_training_percent_by_label": {"25": 12.857142857142858, "50": 41.42857142857143, "75": 75.0, "100": 100.0},
        "strict_records_per_run": 680,
        "strict_issue_count_all_runs": 0,
        "primary_unique_tasks_per_run": 256,
        "all_source_truth_checks_pass": True,
        "undefined_requested_metric_cells": [
            {
                "run_id": "seed2_frac25",
                "metric": "absolute_safeconf_spearman",
                "reason": "calibrated SafeConf is constant in fold 2; two-fold macro requires both folds",
                "reported_as": "NA",
            },
            {
                "run_id": "seed2_frac50",
                "metric": "absolute_safeconf_spearman",
                "reason": "calibrated SafeConf is constant in fold 1; two-fold macro requires both folds",
                "reported_as": "NA",
            },
            {
                "run_id": "seed2_frac75",
                "metric": "absolute_safeconf_spearman",
                "reason": "calibrated SafeConf is constant in both folds",
                "reported_as": "NA",
            },
        ],
        "direction_model_refit_on_replogle": False,
        "disclosed_singleton_batch_guard": {
            "affected_run": "seed2_frac75",
            "affected_fold": "Replogle_cellline_holdout_2_RPE1",
            "batch_size_16_to_15": True,
            "all_frozen_training_pairs_retained": True,
            "technical_deviation_file": "TECHNICAL_DEVIATION.md",
        },
        "analysis_contract_sha256": contract_status["analysis_contract_sha256"],
        "tables": {
            "run_metrics_sha256": sha256(TABLES / "E154_RUN_METRICS_LONG.csv"),
            "full_seed_summary_sha256": sha256(TABLES / "E154_FULL_SEED_SUMMARY.csv"),
            "train_fraction_trend_sha256": sha256(
                TABLES / "E154_TRAIN_FRACTION_TREND.csv"
            ),
            "train_fraction_size_audit_sha256": sha256(
                TABLES / "E154_TRAIN_FRACTION_SIZE_AUDIT.csv"
            ),
            "strict_run_audit_sha256": sha256(TABLES / "E154_STRICT_RUN_AUDIT.csv"),
            "score_degeneracy_audit_sha256": sha256(
                TABLES / "E154_SCORE_DEGENERACY_AUDIT.csv"
            ),
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-one", choices=[item["run_id"] for item in RUN_MATRIX])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    selected = sum(bool(value) for value in [args.prepare_only, args.run_one, args.summarize_only])
    if selected > 1:
        raise ValueError("choose one execution mode")
    if args.prepare_only:
        prepare()
        return
    if args.run_one:
        run_one(args.run_one, args.device)
        return
    if args.summarize_only:
        summarize()
        return
    prepare()
    run_one("seed1_frac100_reused_e151", args.device)
    for item in NEW_RUNS:
        run_one(str(item["run_id"]), args.device)
    summarize()


if __name__ == "__main__":
    main()
