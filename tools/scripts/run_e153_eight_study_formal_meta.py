#!/usr/bin/env python3
"""E153: eight-study formal expanded meta-analysis after E152.

The Replogle gate was already unblinded before this aggregation.  E153 therefore
is an expanded post-E152 meta-analysis, not a new preregistered independent gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, t


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
FREEZE_STATUS = OUT / "FREEZE_STATUS.json"
RUN_STATUS = OUT / "RUN_STATUS.json"

E140_SEVEN = ROOT / "docs/实验结果/E146_unique_biological_task_meta_20260714/tables/E146_E140_TASK_INPUT.csv"
NADIG_DIRECTION = ROOT / "docs/实验结果/E146_unique_biological_task_meta_20260714/tables/E146_E139_DIRECTIONAL_INPUT.csv"
E146_STATUS = ROOT / "docs/实验结果/E146_unique_biological_task_meta_20260714/RUN_STATUS.json"
E151 = ROOT / "docs/实验结果/E151_replogle_formal_dual_models_20260714"
E151_DATA = E151 / "Replogle_two_cellline"
E151_PRIMARY = E151_DATA / "PRIMARY_TASK_RISK_TABLE.csv"
E151_STRICT = E151_DATA / "STRICT_ISSUES.csv"
E152 = ROOT / "docs/实验结果/E152_replogle_directional_confirmation_20260714"
E152_SCORE_STATUS = E152 / "SCORE_FREEZE_STATUS.json"
E152_STATUS = E152 / "RUN_STATUS.json"
E152_SCORES = E152 / "tables/E152_DIRECTIONAL_SCORES_BEFORE_TRUTH.csv"
E152_TASKS = E152 / "tables/E152_TASK_AUDIT.csv"
E135_MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
E149_MANIFEST = ROOT / "docs/实验结果/E149_replogle_two_cellline_contract_20260714/manifests/E149_TASK_MANIFEST.csv"

ABSOLUTE_SNAPSHOT = TABLES / "E153_ABSOLUTE_TASK_INPUT.csv"
NADIG_DIRECTION_SNAPSHOT = TABLES / "E153_NADIG_DIRECTION_INPUT.csv"
REPLOGLE_DIRECTION_SNAPSHOT = TABLES / "E153_REPLOGLE_DIRECTION_INPUT.csv"
INPUT_MANIFEST = TABLES / "E153_INPUT_MANIFEST.csv"

EXPECTED_HASHES = {
    E140_SEVEN: "95e5c776f97efbefaaf4d312dbb906ac100701e9682480020e8dd32153188ec9",
    NADIG_DIRECTION: "5c2ec6b5dfa7de56fbb873403a5910faf71c0972f188feb346fa851005517268",
    E151_PRIMARY: "f1c1d9290cf5519ac702e214b2783f37cf6a9bc723f81d4405bb3891aa625118",
    E151 / "RUN_STATUS.json": "15045cfbd2b9a297666cfed2c9baa37f5b8b500bf66f66574e78fd1854dfac3d",
    E151_DATA / "RUN_STATUS.json": "26a9c30d2954d6c8fef3867ef098366fcbaff25810a943abf9d1e16f9d8937b8",
    E151_STRICT: "7096a7005527f58a26495c0046472db127ed6db40ed640899a1e6f3854766ff6",
    E152_SCORE_STATUS: "f5c6e21957c9795654ce2f8efae196e9d8149283dd81e6051ca2d04d1e1d172a",
    E152_STATUS: "15aec1128fe82c748634c496441d3822335839bacb6143e0fca2aeae3fc93025",
    E152_SCORES: "611ac06b5630b33dd3e2f16e62f5a1c5bd903d1cb895ce69ee97e56578011b42",
    E152_TASKS: "c5dd3bb6b61dab9938f88e389e068a3ab856d44f59db76a587b24f4452a0c353",
    E135_MODEL: "77caf3b7b46071ced9577a8bd5289ce4c7bf5899c329ab37e835c41bda07d4b3",
    E149_MANIFEST: "445f665f18d273083d47c8f82d6beb379a6fbc212fdbcd37d9c8b65e16746f0a",
}

TARGET = "error_two_predictor_mean_rmse"
PRIMARY = "safeconf_calibrated_pair_risk"
COMPARATORS = ["risk_model_disagreement", "baseline_predicted_magnitude"]
SCORES = [PRIMARY, *COMPARATORS]
ABS_REQUIRED = [
    "dataset",
    "fold_id",
    "task_id",
    "split",
    "setting",
    "context",
    "perturbation",
    TARGET,
    *SCORES,
]
FOLD_ESTIMAND = "fold_macro_perturbation_cluster"
POOLED_ESTIMAND = "pooled_context_task_median_sensitivity"

DIRECTION_ENDPOINTS = [
    "error_centered_pearson_mean",
    "error_centered_cosine_mean",
    "direction_error_rank_target",
]
DIRECTION_SCORES = [
    "directional_risk_frozen",
    "baseline_predicted_magnitude",
    "risk_model_disagreement",
    "safeconf_calibrated_pair_risk",
]
DIRECTION_COMPARATORS = [
    "baseline_predicted_magnitude",
    "risk_model_disagreement",
    "safeconf_calibrated_pair_risk",
]
N_BOOT = 3000
SEED = 202607153


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path, prefix: bool = False) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    return f"sha256:{value}" if prefix else value


def deterministic_seed(label: str) -> int:
    token = hashlib.sha256(f"{SEED}|{label}".encode()).hexdigest()[:16]
    return int(token, 16) % (2**32 - 1)


def rho(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 4 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    value = np.corrcoef(rankdata(a, method="average"), rankdata(b, method="average"))[0, 1]
    return float(value) if math.isfinite(value) else float("nan")


def fisher_z(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "（无）"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("NA" if not math.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def validate_expected_hashes() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path, expected in EXPECTED_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Frozen source hash changed: {path}: {observed} != {expected}")
        rows.append(
            {
                "role": "frozen_source",
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": f"sha256:{observed}",
            }
        )
    return rows


def validate_e151() -> tuple[pd.DataFrame, dict[str, object]]:
    top = json.loads((E151 / "RUN_STATUS.json").read_text())
    nested = json.loads((E151_DATA / "RUN_STATUS.json").read_text())
    strict = pd.read_csv(E151_STRICT)
    if top.get("status") != "complete" or nested.get("status") != "complete":
        raise RuntimeError("E151 is not complete")
    if top.get("strict_issue_count") != 0 or nested.get("strict_issue_count") != 0 or len(strict) != 0:
        raise RuntimeError("E151 strict PredictionRecord validation is not clean")
    if top.get("n_primary_unique_heldout_context_tasks") != 256:
        raise RuntimeError("E151 top-level primary task count is not 256")
    tasks = pd.read_csv(E151_PRIMARY)
    missing = set(ABS_REQUIRED).difference(tasks.columns)
    if missing:
        raise RuntimeError(f"E151 primary table lacks columns: {sorted(missing)}")
    if len(tasks) != 256 or tasks.task_id.nunique() != 256:
        raise RuntimeError("E151 primary table is not 256 unique tasks")
    if tasks.dataset.nunique() != 1 or tasks.dataset.iloc[0] != "Replogle_two_cellline":
        raise RuntimeError("Unexpected E151 dataset identity")
    if tasks.fold_id.nunique() != 2 or tasks.context.nunique() != 2 or tasks.perturbation.nunique() != 128:
        raise RuntimeError("E151 primary fold/context/perturbation dimensions differ from contract")
    if tasks.duplicated(["dataset", "context", "perturbation"]).any():
        raise RuntimeError("E151 primary has duplicate context-perturbation tasks")
    context_counts = tasks.groupby("perturbation").context.nunique()
    if not context_counts.eq(2).all():
        raise RuntimeError("Not every E151 perturbation has both K562 and RPE1 primary tasks")
    if "test_truth_used_for_score_or_threshold" in tasks and tasks["test_truth_used_for_score_or_threshold"].fillna(False).astype(bool).any():
        raise RuntimeError("E151 reports target truth use in a deployable score")
    return tasks, {
        "top_status_complete": True,
        "nested_status_complete": True,
        "strict_issue_count": 0,
        "n_primary_tasks": len(tasks),
        "n_folds": int(tasks.fold_id.nunique()),
        "n_perturbations": int(tasks.perturbation.nunique()),
    }


def validate_e152(e151: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    score_status = json.loads(E152_SCORE_STATUS.read_text())
    status = json.loads(E152_STATUS.read_text())
    if score_status.get("phase") != "scores_frozen_before_replogle_directional_truth_audit":
        raise RuntimeError("E152 score-freeze phase is invalid")
    if status.get("status") != "complete" or status.get("n_primary_unique_tasks") != 256:
        raise RuntimeError("E152 completion/task contract failed")
    if not status.get("all_source_truth_checks_pass") or status.get("model_refit_on_replogle"):
        raise RuntimeError("E152 source audit failed or model was refit")
    if score_status.get("score_file_sha256") != sha256_file(E152_SCORES):
        raise RuntimeError("E152 frozen score hash differs from SCORE_FREEZE_STATUS")
    if status.get("score_file_sha256") != score_status.get("score_file_sha256"):
        raise RuntimeError("E152 RUN_STATUS score hash differs from freeze status")
    if score_status.get("model_file_sha256") != sha256_file(E135_MODEL):
        raise RuntimeError("E152 frozen model hash differs from E135 model")
    if score_status.get("e151_status_sha256") != sha256_file(E151 / "RUN_STATUS.json"):
        raise RuntimeError("E151 status differs from the file seen during E152 score freeze")
    scores = pd.read_csv(E152_SCORES)
    tasks = pd.read_csv(E152_TASKS)
    keys = ["fold_id", "task_id", "setting", "context", "perturbation"]
    if len(scores) != 256 or len(tasks) != 256:
        raise RuntimeError("E152 score/task files do not contain 256 rows")
    if scores.duplicated(keys).any() or tasks.duplicated(keys).any():
        raise RuntimeError("E152 score/task keys are not one-to-one")
    if tasks.duplicated(["dataset", "context", "perturbation"]).any():
        raise RuntimeError("E152 does not contain 256 unique primary context tasks")
    score_alignment = tasks[keys + ["directional_risk_frozen", "frozen_model_sha256", "target_truth_used_for_score_or_transform"]].merge(
        scores[keys + ["directional_risk_frozen", "frozen_model_sha256", "target_truth_used_for_score_or_transform"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_task", "_freeze"),
    )
    if not score_alignment["_merge"].eq("both").all():
        raise RuntimeError("E152 frozen scores do not exactly align with audited tasks")
    if not np.allclose(
        score_alignment["directional_risk_frozen_task"],
        score_alignment["directional_risk_frozen_freeze"],
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("E152 directional risk changed after score freeze")
    if not score_alignment["frozen_model_sha256_task"].eq(score_status["model_file_sha256"]).all():
        raise RuntimeError("E152 task rows do not carry the frozen E135 model hash")
    if score_alignment[["target_truth_used_for_score_or_transform_task", "target_truth_used_for_score_or_transform_freeze"]].fillna(False).astype(bool).any().any():
        raise RuntimeError("E152 target truth was used for score construction")

    absolute_columns = [TARGET, *SCORES]
    absolute_alignment = e151[keys + absolute_columns].merge(
        tasks[keys + absolute_columns],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_e151", "_e152"),
    )
    if not absolute_alignment["_merge"].eq("both").all():
        raise RuntimeError("E151 primary and E152 audited task keys differ")
    max_absolute_difference = 0.0
    for column in absolute_columns:
        difference = float(
            np.max(
                np.abs(
                    absolute_alignment[f"{column}_e151"].to_numpy(float)
                    - absolute_alignment[f"{column}_e152"].to_numpy(float)
                )
            )
        )
        max_absolute_difference = max(max_absolute_difference, difference)
    if max_absolute_difference > 1e-12:
        raise RuntimeError(f"E151/E152 absolute values differ: {max_absolute_difference}")
    return tasks, {
        "score_freeze_hash_verified": True,
        "score_file_sha256": f"sha256:{sha256_file(E152_SCORES)}",
        "model_hash_verified": True,
        "model_file_sha256": f"sha256:{sha256_file(E135_MODEL)}",
        "task_audit_sha256": f"sha256:{sha256_file(E152_TASKS)}",
        "e151_status_hash_verified": True,
        "n_primary_unique_tasks": len(tasks),
        "all_source_truth_checks_pass": True,
        "model_refit": False,
        "max_abs_e151_e152_absolute_difference": max_absolute_difference,
        "preregistered_directional_gate_passed": bool(status.get("preregistered_directional_gate_passed")),
    }


def validate_nadig_direction() -> tuple[pd.DataFrame, dict[str, object]]:
    tasks = pd.read_csv(NADIG_DIRECTION)
    required = {
        "dataset",
        "fold_id",
        "context",
        "perturbation",
        *DIRECTION_ENDPOINTS,
        *DIRECTION_SCORES,
    }
    missing = required.difference(tasks.columns)
    if missing:
        raise RuntimeError(f"Nadig direction input lacks columns: {sorted(missing)}")
    if len(tasks) != 256 or tasks.fold_id.nunique() != 2 or tasks.perturbation.nunique() != 96:
        raise RuntimeError("Nadig directional dimensions differ from E146 audit")
    if tasks.dataset.nunique() != 1 or tasks.dataset.iloc[0] != "Nadig_two_cellline":
        raise RuntimeError("Unexpected Nadig directional dataset identity")
    return tasks, {
        "n_rows": len(tasks),
        "n_folds": int(tasks.fold_id.nunique()),
        "n_perturbations": int(tasks.perturbation.nunique()),
    }


def freeze() -> None:
    if not CONTRACT.exists():
        raise RuntimeError("ANALYSIS_CONTRACT.md must exist before E153 freeze")
    for directory in [OUT, TABLES, REPORTS, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_rows = validate_expected_hashes()
    e151, e151_audit = validate_e151()
    replogle_direction, e152_audit = validate_e152(e151)
    nadig_direction, nadig_audit = validate_nadig_direction()
    seven = pd.read_csv(E140_SEVEN)
    missing = set(ABS_REQUIRED).difference(seven.columns)
    if missing or len(seven) != 3209 or seven.dataset.nunique() != 7:
        raise RuntimeError(f"E140 seven-study snapshot contract failed; missing={sorted(missing)}")
    absolute = pd.concat(
        [seven[ABS_REQUIRED], e151[ABS_REQUIRED]], ignore_index=True, sort=False
    ).sort_values(["dataset", "fold_id", "context", "perturbation", "task_id"], kind="stable")
    absolute.to_csv(ABSOLUTE_SNAPSHOT, index=False, lineterminator="\n")
    nadig_direction.sort_values(["fold_id", "context", "perturbation"], kind="stable").to_csv(
        NADIG_DIRECTION_SNAPSHOT, index=False, lineterminator="\n"
    )
    replogle_direction.sort_values(["fold_id", "context", "perturbation"], kind="stable").to_csv(
        REPLOGLE_DIRECTION_SNAPSHOT, index=False, lineterminator="\n"
    )
    for role, path in [
        ("frozen_absolute_snapshot", ABSOLUTE_SNAPSHOT),
        ("frozen_nadig_direction_snapshot", NADIG_DIRECTION_SNAPSHOT),
        ("frozen_replogle_direction_snapshot", REPLOGLE_DIRECTION_SNAPSHOT),
    ]:
        manifest_rows.append(
            {
                "role": role,
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path, prefix=True),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(INPUT_MANIFEST, index=False, lineterminator="\n")
    status = {
        "experiment": "E153_eight_study_formal_meta",
        "phase": "frozen_before_e153_statistics",
        "generated_at": now(),
        "analysis_nature": "post_E152_expanded_meta_not_preregistered_independent_gate",
        "contract_sha256": sha256_file(CONTRACT, prefix=True),
        "analysis_script_sha256": sha256_file(Path(__file__), prefix=True),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST, prefix=True),
        "absolute_snapshot_sha256": sha256_file(ABSOLUTE_SNAPSHOT, prefix=True),
        "nadig_direction_snapshot_sha256": sha256_file(NADIG_DIRECTION_SNAPSHOT, prefix=True),
        "replogle_direction_snapshot_sha256": sha256_file(REPLOGLE_DIRECTION_SNAPSHOT, prefix=True),
        "n_absolute_rows": len(absolute),
        "n_absolute_studies": int(absolute.dataset.nunique()),
        "n_absolute_folds": int(absolute[["dataset", "fold_id"]].drop_duplicates().shape[0]),
        "n_absolute_context_tasks": int(absolute[["dataset", "context", "perturbation"]].drop_duplicates().shape[0]),
        "n_absolute_perturbation_clusters": int(absolute[["dataset", "perturbation"]].drop_duplicates().shape[0]),
        "e151_validation": e151_audit,
        "e152_validation": e152_audit,
        "nadig_direction_validation": nadig_audit,
        "n_bootstrap": N_BOOT,
    }
    FREEZE_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def verify_frozen() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, object]]:
    if not FREEZE_STATUS.exists():
        raise RuntimeError("Run --phase freeze before --phase analyze")
    status = json.loads(FREEZE_STATUS.read_text())
    if status.get("phase") != "frozen_before_e153_statistics":
        raise RuntimeError(f"Unexpected E153 freeze phase: {status.get('phase')}")
    checks = {
        CONTRACT: status["contract_sha256"],
        Path(__file__): status["analysis_script_sha256"],
        INPUT_MANIFEST: status["input_manifest_sha256"],
        ABSOLUTE_SNAPSHOT: status["absolute_snapshot_sha256"],
        NADIG_DIRECTION_SNAPSHOT: status["nadig_direction_snapshot_sha256"],
        REPLOGLE_DIRECTION_SNAPSHOT: status["replogle_direction_snapshot_sha256"],
    }
    for path, expected in checks.items():
        if sha256_file(path, prefix=True) != expected:
            raise RuntimeError(f"Frozen E153 artifact changed: {path}")
    absolute = pd.read_csv(ABSOLUTE_SNAPSHOT)
    directions = {
        "Nadig_two_cellline": pd.read_csv(NADIG_DIRECTION_SNAPSHOT),
        "Replogle_two_cellline": pd.read_csv(REPLOGLE_DIRECTION_SNAPSHOT),
    }
    if len(absolute) != status["n_absolute_rows"] or absolute.dataset.nunique() != 8:
        raise RuntimeError("Frozen E153 absolute dimensions changed")
    for frame in [absolute, *directions.values()]:
        for column in ["dataset", "fold_id", "task_id", "context", "perturbation"]:
            if column in frame:
                frame[column] = frame[column].astype(str)
    return absolute, directions, status


def fold_macro(frame: pd.DataFrame, score: str, target: str = TARGET) -> float:
    values = [rho(group[score], group[target]) for _, group in frame.groupby("fold_id", sort=True)]
    finite = np.asarray(values, float)
    return float(np.nanmean(finite)) if np.isfinite(finite).any() else float("nan")


def deterministic_dedup(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["dataset", "context", "perturbation"]
    for key, group in data.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        row.update(
            {
                "n_outer_fold_occurrences": len(group),
                "n_distinct_folds": int(group.fold_id.nunique()),
                "source_fold_ids": ";".join(sorted(group.fold_id.astype(str).unique())),
                TARGET: float(group[TARGET].median()),
            }
        )
        for score in SCORES:
            row[score] = float(group[score].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def absolute_cluster_draws(
    frame: pd.DataFrame,
    dataset: str,
    estimand: str,
    fold_mode: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    group = frame[frame.dataset.eq(dataset)].reset_index(drop=True)
    perturbations = sorted(group.perturbation.astype(str).unique())
    pindex = {value: index for index, value in enumerate(perturbations)}
    if fold_mode:
        fold_arrays = []
        for _, fold in group.groupby("fold_id", sort=True):
            fold_arrays.append(
                {
                    "cluster": np.asarray([pindex[str(value)] for value in fold.perturbation], int),
                    "target": fold[TARGET].to_numpy(float),
                    "scores": {score: fold[score].to_numpy(float) for score in SCORES},
                }
            )
        observed = {score: fold_macro(group, score) for score in SCORES}
    else:
        fold_arrays = [
            {
                "cluster": np.asarray([pindex[str(value)] for value in group.perturbation], int),
                "target": group[TARGET].to_numpy(float),
                "scores": {score: group[score].to_numpy(float) for score in SCORES},
            }
        ]
        observed = {score: rho(group[score], group[TARGET]) for score in SCORES}
    rng = np.random.default_rng(deterministic_seed(f"absolute|{estimand}|{dataset}"))
    rows = []
    for draw in range(N_BOOT):
        counts = rng.multinomial(
            len(perturbations), np.full(len(perturbations), 1 / len(perturbations))
        )
        values: dict[str, float] = {}
        for score in SCORES:
            fold_values = []
            for arrays in fold_arrays:
                indices = np.repeat(np.arange(len(arrays["cluster"])), counts[arrays["cluster"]])
                fold_values.append(rho(arrays["scores"][score][indices], arrays["target"][indices]))
            values[score] = float(np.nanmean(fold_values))
        rows.append(
            {
                "dataset": dataset,
                "estimand": estimand,
                "draw": draw,
                **{f"rho__{score}": values[score] for score in SCORES},
                "delta_rho__safe_minus_disagreement": values[PRIMARY] - values[COMPARATORS[0]],
                "delta_rho__safe_minus_magnitude": values[PRIMARY] - values[COMPARATORS[1]],
                "delta_z__safe_minus_disagreement": fisher_z(values[PRIMARY]) - fisher_z(values[COMPARATORS[0]]),
                "delta_z__safe_minus_magnitude": fisher_z(values[PRIMARY]) - fisher_z(values[COMPARATORS[1]]),
            }
        )
    return pd.DataFrame(rows), observed


def bootstrap_summary(
    draws: pd.DataFrame,
    observed_by_key: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    metric_columns = [
        column for column in draws.columns if column.startswith("rho__") or column.startswith("delta_")
    ]
    rows = []
    for (dataset, estimand), group in draws.groupby(["dataset", "estimand"], sort=True):
        observed_scores = observed_by_key[(dataset, estimand)]
        observed_metrics = {f"rho__{score}": value for score, value in observed_scores.items()}
        observed_metrics.update(
            {
                "delta_rho__safe_minus_disagreement": observed_scores[PRIMARY] - observed_scores[COMPARATORS[0]],
                "delta_rho__safe_minus_magnitude": observed_scores[PRIMARY] - observed_scores[COMPARATORS[1]],
                "delta_z__safe_minus_disagreement": fisher_z(observed_scores[PRIMARY]) - fisher_z(observed_scores[COMPARATORS[0]]),
                "delta_z__safe_minus_magnitude": fisher_z(observed_scores[PRIMARY]) - fisher_z(observed_scores[COMPARATORS[1]]),
            }
        )
        for metric in metric_columns:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) != N_BOOT:
                raise RuntimeError(f"{dataset}/{estimand}/{metric}: only {len(values)} valid draws")
            rows.append(
                {
                    "dataset": dataset,
                    "estimand": estimand,
                    "metric": metric,
                    "observed": observed_metrics[metric],
                    "n_valid_draws": len(values),
                    "bootstrap_median": float(np.median(values)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "p_gt_zero": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(rows)


def random_effects(yi: np.ndarray, sei: np.ndarray) -> dict[str, float]:
    y, se = np.asarray(yi, float), np.asarray(sei, float)
    keep = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[keep], se[keep]
    k = len(y)
    if k < 3:
        return {key: float("nan") for key in [
            "k", "pooled_z", "se_mkh", "ci95_low_z", "ci95_high_z", "tau2_reml",
            "Q", "I2_percent", "prediction_low_z", "prediction_high_z",
        ]}
    variance = se**2

    def objective(tau2: float) -> float:
        weights = 1.0 / (variance + tau2)
        mean = float(np.sum(weights * y) / np.sum(weights))
        return 0.5 * float(
            np.sum(np.log(variance + tau2))
            + np.log(np.sum(weights))
            + np.sum(weights * (y - mean) ** 2)
        )

    upper = max(1.0, float(np.var(y, ddof=1) * 20))
    fit = minimize_scalar(
        objective, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-12}
    )
    tau2 = min([(0.0, objective(0.0)), (float(fit.x), float(fit.fun))], key=lambda item: item[1])[0]
    weights = 1.0 / (variance + tau2)
    mean = float(np.sum(weights * y) / np.sum(weights))
    q_hk = float(np.sum(weights * (y - mean) ** 2) / (k - 1))
    se_mkh = float(np.sqrt(max(1.0, q_hk) / np.sum(weights)))
    critical_mean = float(t.ppf(0.975, k - 1))
    fixed_weights = 1.0 / variance
    fixed_mean = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
    Q = float(np.sum(fixed_weights * (y - fixed_mean) ** 2))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    prediction_se = float(np.sqrt(tau2 + se_mkh**2))
    critical_prediction = float(t.ppf(0.975, k - 2))
    return {
        "k": k,
        "pooled_z": mean,
        "se_mkh": se_mkh,
        "ci95_low_z": mean - critical_mean * se_mkh,
        "ci95_high_z": mean + critical_mean * se_mkh,
        "tau2_reml": tau2,
        "Q": Q,
        "I2_percent": I2,
        "prediction_low_z": mean - critical_prediction * prediction_se,
        "prediction_high_z": mean + critical_prediction * prediction_se,
    }


def build_study_effects(
    absolute: pd.DataFrame,
    dedup: pd.DataFrame,
    draws: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset in sorted(absolute.dataset.unique()):
        specifications = [
            (FOLD_ESTIMAND, absolute[absolute.dataset.eq(dataset)], True),
            (POOLED_ESTIMAND, dedup[dedup.dataset.eq(dataset)], False),
        ]
        for estimand, group, fold_mode in specifications:
            correlations = {
                score: fold_macro(group, score) if fold_mode else rho(group[score], group[TARGET])
                for score in SCORES
            }
            study_draws = draws[
                draws.dataset.eq(dataset) & draws.estimand.eq(estimand)
            ]
            n_clusters = int(group.perturbation.nunique())
            for score in SCORES:
                values = study_draws[f"rho__{score}"].map(fisher_z).to_numpy(float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "dataset": dataset,
                        "estimand": estimand,
                        "analysis": "score_association",
                        "effect": score,
                        "n_independent_perturbation_clusters": n_clusters,
                        "rho_safeconf": correlations[PRIMARY],
                        "rho_comparator": correlations[score],
                        "delta_rho_safe_minus_comparator": correlations[PRIMARY] - correlations[score],
                        "yi_fisher_z": fisher_z(correlations[score]),
                        "sei": float(np.std(values, ddof=1)),
                        "sei_source": "perturbation_cluster_bootstrap_fisher_z",
                    }
                )
            for comparator, suffix in [
                (COMPARATORS[0], "disagreement"),
                (COMPARATORS[1], "magnitude"),
            ]:
                values = study_draws[f"delta_z__safe_minus_{suffix}"].to_numpy(float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "dataset": dataset,
                        "estimand": estimand,
                        "analysis": "safeconf_minus_comparator",
                        "effect": f"safeconf_minus_{suffix}",
                        "n_independent_perturbation_clusters": n_clusters,
                        "rho_safeconf": correlations[PRIMARY],
                        "rho_comparator": correlations[comparator],
                        "delta_rho_safe_minus_comparator": correlations[PRIMARY] - correlations[comparator],
                        "yi_fisher_z": fisher_z(correlations[PRIMARY]) - fisher_z(correlations[comparator]),
                        "sei": float(np.std(values, ddof=1)),
                        "sei_source": "paired_perturbation_cluster_bootstrap_delta_fisher_z",
                    }
                )
    return pd.DataFrame(rows)


def meta_table(studies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (estimand, analysis, effect), group in studies.groupby(
        ["estimand", "analysis", "effect"], sort=True
    ):
        result = random_effects(group.yi_fisher_z, group.sei)
        row = {"estimand": estimand, "analysis": analysis, "effect": effect, **result}
        for name in [
            "pooled_z", "ci95_low_z", "ci95_high_z", "prediction_low_z", "prediction_high_z"
        ]:
            row[name.replace("_z", "_rho_equivalent")] = float(np.tanh(row[name]))
        row["backtransform_interpretation"] = (
            "pooled Spearman"
            if analysis == "score_association"
            else "rho-equivalent Fisher-z difference; not raw delta-rho"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def lodo_table(studies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = sorted(studies.dataset.unique())
    for (estimand, analysis, effect), group in studies.groupby(
        ["estimand", "analysis", "effect"], sort=True
    ):
        for removed in datasets:
            keep = group[~group.dataset.eq(removed)]
            result = random_effects(keep.yi_fisher_z, keep.sei)
            rows.append(
                {
                    "estimand": estimand,
                    "analysis": analysis,
                    "effect": effect,
                    "removed_dataset": removed,
                    **result,
                    "pooled_rho_equivalent": float(np.tanh(result["pooled_z"])),
                    "prediction_low_rho_equivalent": float(np.tanh(result["prediction_low_z"])),
                    "prediction_high_rho_equivalent": float(np.tanh(result["prediction_high_z"])),
                }
            )
    return pd.DataFrame(rows)


def direction_fold_macro(frame: pd.DataFrame, score: str, endpoint: str) -> float:
    values = [rho(group[score], group[endpoint]) for _, group in frame.groupby("fold_id", sort=True)]
    return float(np.nanmean(values))


def direction_cluster_draws(
    frame: pd.DataFrame,
    dataset: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    group = frame.reset_index(drop=True)
    perturbations = sorted(group.perturbation.astype(str).unique())
    pindex = {value: index for index, value in enumerate(perturbations)}
    folds = []
    for _, fold in group.groupby("fold_id", sort=True):
        folds.append(
            {
                "cluster": np.asarray([pindex[str(value)] for value in fold.perturbation], int),
                "scores": {score: fold[score].to_numpy(float) for score in DIRECTION_SCORES},
                "endpoints": {endpoint: fold[endpoint].to_numpy(float) for endpoint in DIRECTION_ENDPOINTS},
            }
        )
    observed = {
        (score, endpoint): direction_fold_macro(group, score, endpoint)
        for score in DIRECTION_SCORES
        for endpoint in DIRECTION_ENDPOINTS
    }
    rng = np.random.default_rng(deterministic_seed(f"direction|{dataset}"))
    rows = []
    for draw in range(N_BOOT):
        counts = rng.multinomial(
            len(perturbations), np.full(len(perturbations), 1 / len(perturbations))
        )
        values: dict[tuple[str, str], float] = {}
        for score in DIRECTION_SCORES:
            for endpoint in DIRECTION_ENDPOINTS:
                fold_values = []
                for fold in folds:
                    indices = np.repeat(
                        np.arange(len(fold["cluster"])), counts[fold["cluster"]]
                    )
                    fold_values.append(
                        rho(fold["scores"][score][indices], fold["endpoints"][endpoint][indices])
                    )
                values[(score, endpoint)] = float(np.nanmean(fold_values))
        row: dict[str, object] = {"dataset": dataset, "draw": draw}
        for (score, endpoint), value in values.items():
            row[f"rho__{score}__{endpoint}"] = value
        for comparator in DIRECTION_COMPARATORS:
            for endpoint in DIRECTION_ENDPOINTS:
                row[f"delta_rho__directional_minus_{comparator}__{endpoint}"] = (
                    values[("directional_risk_frozen", endpoint)] - values[(comparator, endpoint)]
                )
        rows.append(row)
    return pd.DataFrame(rows), observed


def direction_results(
    directions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    study_rows = []
    all_draws = []
    observed_by_study: dict[str, dict[tuple[str, str], float]] = {}
    for dataset, frame in directions.items():
        for fold_id, group in frame.groupby("fold_id", sort=True):
            for score in DIRECTION_SCORES:
                for endpoint in DIRECTION_ENDPOINTS:
                    fold_rows.append(
                        {
                            "dataset": dataset,
                            "fold_id": fold_id,
                            "score": score,
                            "endpoint": endpoint,
                            "n_tasks": len(group),
                            "spearman": rho(group[score], group[endpoint]),
                        }
                    )
        draws, observed = direction_cluster_draws(frame, dataset)
        observed_by_study[dataset] = observed
        all_draws.append(draws)
        for score in DIRECTION_SCORES:
            for endpoint in DIRECTION_ENDPOINTS:
                values = draws[f"rho__{score}__{endpoint}"].to_numpy(float)
                study_rows.append(
                    {
                        "dataset": dataset,
                        "analysis": "score_association",
                        "effect": score,
                        "endpoint": endpoint,
                        "n_folds": int(frame.fold_id.nunique()),
                        "n_independent_perturbation_clusters": int(frame.perturbation.nunique()),
                        "fold_macro_spearman": observed[(score, endpoint)],
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                        "p_gt_zero": float(np.mean(values > 0)),
                    }
                )
        for comparator in DIRECTION_COMPARATORS:
            for endpoint in DIRECTION_ENDPOINTS:
                column = f"delta_rho__directional_minus_{comparator}__{endpoint}"
                values = draws[column].to_numpy(float)
                observed_delta = (
                    observed[("directional_risk_frozen", endpoint)]
                    - observed[(comparator, endpoint)]
                )
                study_rows.append(
                    {
                        "dataset": dataset,
                        "analysis": "directional_minus_comparator",
                        "effect": f"directional_minus_{comparator}",
                        "endpoint": endpoint,
                        "n_folds": int(frame.fold_id.nunique()),
                        "n_independent_perturbation_clusters": int(frame.perturbation.nunique()),
                        "fold_macro_spearman": observed_delta,
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                        "p_gt_zero": float(np.mean(values > 0)),
                    }
                )
    draws = pd.concat(all_draws, ignore_index=True)
    study = pd.DataFrame(study_rows)
    fixed_rows = []
    study_names = sorted(directions)
    for analysis, effects in [
        ("score_association", DIRECTION_SCORES),
        ("directional_minus_comparator", [f"directional_minus_{value}" for value in DIRECTION_COMPARATORS]),
    ]:
        for effect in effects:
            for endpoint in DIRECTION_ENDPOINTS:
                selected = study[
                    study.analysis.eq(analysis)
                    & study.effect.eq(effect)
                    & study.endpoint.eq(endpoint)
                ].set_index("dataset")
                observed_values = [float(selected.loc[name, "fold_macro_spearman"]) for name in study_names]
                if analysis == "score_association":
                    columns = [f"rho__{effect}__{endpoint}"] * 2
                else:
                    comparator = effect.replace("directional_minus_", "")
                    columns = [f"delta_rho__directional_minus_{comparator}__{endpoint}"] * 2
                draw_values = []
                for name, column in zip(study_names, columns):
                    draw_values.append(
                        draws[draws.dataset.eq(name)].sort_values("draw")[column].to_numpy(float)
                    )
                combined = np.mean(np.vstack(draw_values), axis=0)
                fixed_rows.append(
                    {
                        "analysis": analysis,
                        "effect": effect,
                        "endpoint": endpoint,
                        "k_studies": 2,
                        "equal_study_mean_spearman": float(np.mean(observed_values)),
                        "minimum_study_spearman": float(np.min(observed_values)),
                        "maximum_study_spearman": float(np.max(observed_values)),
                        "fixed_two_study_bootstrap_ci95_low": float(np.quantile(combined, 0.025)),
                        "fixed_two_study_bootstrap_ci95_high": float(np.quantile(combined, 0.975)),
                        "inference_scope": "fixed_two_studies_descriptive_not_random_effects_not_population_PI",
                    }
                )
    return pd.DataFrame(fold_rows), study, draws, pd.DataFrame(fixed_rows)


def make_figures(
    boot_summary: pd.DataFrame,
    directional_study: pd.DataFrame,
    directional_fixed: pd.DataFrame,
) -> None:
    main = boot_summary[
        boot_summary.estimand.eq(FOLD_ESTIMAND)
        & boot_summary.metric.isin(
            ["delta_rho__safe_minus_disagreement", "delta_rho__safe_minus_magnitude"]
        )
    ].copy()
    order = sorted(main.dataset.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0), sharey=True)
    for axis, metric, title in zip(
        axes,
        ["delta_rho__safe_minus_disagreement", "delta_rho__safe_minus_magnitude"],
        ["SafeConf − disagreement", "SafeConf − magnitude"],
    ):
        subset = main[main.metric.eq(metric)].set_index("dataset").loc[order]
        y = np.arange(len(order))
        x = subset.observed.to_numpy(float)
        low = x - subset.ci95_low.to_numpy(float)
        high = subset.ci95_high.to_numpy(float) - x
        axis.errorbar(x, y, xerr=np.vstack([low, high]), fmt="o", color="#315F78", ecolor="#7195A7", capsize=3)
        axis.axvline(0, color="#777777", ls="--", lw=0.9)
        axis.set_title(title)
        axis.set_xlabel("Δ Spearman ρ")
        axis.set_yticks(y, order)
        axis.grid(axis="x", color="#E6E6E6", lw=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(FIGURES / "F1_absolute_eight_study_cluster_bootstrap.svg", facecolor="white")
    plt.close(fig)

    endpoint = "direction_error_rank_target"
    ds = directional_study[
        directional_study.analysis.eq("score_association")
        & directional_study.effect.eq("directional_risk_frozen")
        & directional_study.endpoint.eq(endpoint)
    ].copy()
    fixed = directional_fixed[
        directional_fixed.analysis.eq("score_association")
        & directional_fixed.effect.eq("directional_risk_frozen")
        & directional_fixed.endpoint.eq(endpoint)
    ].iloc[0]
    labels = list(ds.dataset) + ["fixed two-study mean"]
    x = list(ds.fold_macro_spearman) + [fixed.equal_study_mean_spearman]
    lo = list(ds.ci95_low) + [fixed.fixed_two_study_bootstrap_ci95_low]
    hi = list(ds.ci95_high) + [fixed.fixed_two_study_bootstrap_ci95_high]
    y = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(8.2, 3.6))
    axis.errorbar(x, y, xerr=np.vstack([np.asarray(x) - lo, np.asarray(hi) - x]), fmt="o", color="#A24A34", ecolor="#C98A78", capsize=3)
    axis.axvline(0, color="#777777", ls="--", lw=0.9)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Spearman ρ with directional error rank")
    axis.set_title("Directional-SafeConf: two-study descriptive summary")
    axis.grid(axis="x", color="#E6E6E6", lw=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(FIGURES / "F2_directional_two_study_descriptive.svg", facecolor="white")
    plt.close(fig)


def write_report(
    absolute: pd.DataFrame,
    multiplicity: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    boot_summary: pd.DataFrame,
    meta: pd.DataFrame,
    lodo: pd.DataFrame,
    directional_study: pd.DataFrame,
    directional_fixed: pd.DataFrame,
    freeze_status: dict[str, object],
) -> None:
    fold_macro = fold_correlations.groupby(["dataset", "score"], as_index=False).spearman.mean()
    fold_pivot = fold_macro.pivot(index="dataset", columns="score", values="spearman").reset_index()
    fold_pivot["safe_minus_disagreement"] = fold_pivot[PRIMARY] - fold_pivot[COMPARATORS[0]]
    fold_pivot["safe_minus_magnitude"] = fold_pivot[PRIMARY] - fold_pivot[COMPARATORS[1]]
    fold_display = fold_pivot[
        ["dataset", PRIMARY, *COMPARATORS, "safe_minus_disagreement", "safe_minus_magnitude"]
    ].round(4)
    main_boot = boot_summary[
        boot_summary.estimand.eq(FOLD_ESTIMAND)
        & boot_summary.metric.isin(
            ["delta_rho__safe_minus_disagreement", "delta_rho__safe_minus_magnitude"]
        )
    ][["dataset", "metric", "observed", "ci95_low", "ci95_high", "p_gt_zero"]].round(4)
    main_meta = meta[
        meta.estimand.eq(FOLD_ESTIMAND)
        & meta.analysis.eq("safeconf_minus_comparator")
    ][
        ["effect", "k", "pooled_z", "ci95_low_z", "ci95_high_z", "tau2_reml", "I2_percent", "prediction_low_z", "prediction_high_z"]
    ].round(4)
    association_meta = meta[
        meta.estimand.eq(FOLD_ESTIMAND) & meta.analysis.eq("score_association")
    ][
        ["effect", "pooled_rho_equivalent", "ci95_low_rho_equivalent", "ci95_high_rho_equivalent", "prediction_low_rho_equivalent", "prediction_high_rho_equivalent", "tau2_reml", "I2_percent"]
    ].round(4)
    sensitivity_meta = meta[
        meta.estimand.eq(POOLED_ESTIMAND)
        & meta.analysis.eq("safeconf_minus_comparator")
    ][
        ["effect", "k", "pooled_z", "ci95_low_z", "ci95_high_z", "tau2_reml", "I2_percent", "prediction_low_z", "prediction_high_z"]
    ].round(4)
    main_lodo = lodo[
        lodo.estimand.eq(FOLD_ESTIMAND)
        & lodo.analysis.eq("safeconf_minus_comparator")
    ][
        ["effect", "removed_dataset", "pooled_z", "ci95_low_z", "ci95_high_z", "prediction_low_z", "prediction_high_z"]
    ].round(4)
    directional_main = directional_study[
        directional_study.endpoint.eq("direction_error_rank_target")
        & (
            (directional_study.analysis.eq("score_association") & directional_study.effect.isin(["directional_risk_frozen", "baseline_predicted_magnitude"]))
            | (
                directional_study.analysis.eq("directional_minus_comparator")
                & directional_study.effect.eq("directional_minus_baseline_predicted_magnitude")
            )
        )
    ][
        ["dataset", "analysis", "effect", "n_independent_perturbation_clusters", "fold_macro_spearman", "ci95_low", "ci95_high"]
    ].round(4)
    directional_two = directional_fixed[
        directional_fixed.endpoint.eq("direction_error_rank_target")
        & (
            (directional_fixed.analysis.eq("score_association") & directional_fixed.effect.isin(["directional_risk_frozen", "baseline_predicted_magnitude"]))
            | (
                directional_fixed.analysis.eq("directional_minus_comparator")
                & directional_fixed.effect.eq("directional_minus_baseline_predicted_magnitude")
            )
        )
    ][
        ["analysis", "effect", "k_studies", "equal_study_mean_spearman", "minimum_study_spearman", "maximum_study_spearman", "fixed_two_study_bootstrap_ci95_low", "fixed_two_study_bootstrap_ci95_high"]
    ].round(4)

    full_main = meta[
        meta.estimand.eq(FOLD_ESTIMAND)
        & meta.analysis.eq("safeconf_minus_comparator")
    ].set_index("effect")
    sign_changes = []
    for effect, row in full_main.iterrows():
        subset = lodo[
            lodo.estimand.eq(FOLD_ESTIMAND)
            & lodo.analysis.eq("safeconf_minus_comparator")
            & lodo.effect.eq(effect)
        ]
        changed = subset[np.sign(subset.pooled_z) != np.sign(row.pooled_z)]
        sign_changes.extend(
            [f"{effect}: 删除 {item.removed_dataset} 后 pooled z={item.pooled_z:+.4f}" for item in changed.itertuples(index=False)]
        )
    sign_text = "；".join(sign_changes) if sign_changes else "主 estimand 的 LODO 未发生合并效应符号反转"

    report = f"""# E153｜八研究正式扩展元分析

## 结论范围

E153 在 E152 已经解封之后，把 Replogle 的256个正式主任务加入 E140 七研究。它是 post-E152 expanded meta-analysis，不是新的预注册独立 gate。Replogle 自身的预注册方向 gate 仍由 E152 单独承担。

输入共 {len(absolute)} 行、{absolute.dataset.nunique()} 个研究、{absolute[['dataset','fold_id']].drop_duplicates().shape[0]} 个fold、{absolute[['dataset','context','perturbation']].drop_duplicates().shape[0]} 个context-task、{absolute[['dataset','perturbation']].drop_duplicates().shape[0]} 个perturbation簇。E151 strict issues=0；E152 score-freeze、模型和任务哈希全部通过；Replogle恰有128个扰动×2个held-out contexts=256个唯一主任务。

## 数据与重复结构

{markdown_table(multiplicity.round(4))}

## Absolute-RMSE：研究内fold-macro

{markdown_table(fold_display)}

正值表示风险分数随误差升高。SafeConf减比较器为正表示SafeConf的错误排序更强。

## 每研究perturbation-cluster区间

{markdown_table(main_boot)}

## 八研究随机效应：主estimand

{markdown_table(main_meta)}

差值位于 Fisher-z 尺度，不能当成原始 Δrho。均值区间使用modified Knapp–Hartung；prediction interval用于表达未来研究可能落入的范围。

### Absolute分数本身

{markdown_table(association_meta)}

## LODO

{markdown_table(main_lodo)}

{sign_text}。LODO只能说明现有八研究中单个研究的影响，不能增加研究数量。

## Pooled-median sensitivity

{markdown_table(sensitivity_meta)}

该敏感性先把同一context-task跨fold取中位数，再做研究内pooled Spearman；它改变了E140的fold-macro estimand。bootstrap仍以perturbation为簇同步全部context，结果不能覆盖上面的主分析。

## Directional-SafeConf：Nadig与Replogle

### 分研究fold-macro

{markdown_table(directional_main)}

### 固定两研究描述性合并

{markdown_table(directional_two)}

这里的区间只条件于Nadig和Replogle这两个已观察研究。k=2不进行REML、Knapp–Hartung、I²或prediction interval，也不声称已经获得跨研究稳定保证。Replogle两个细胞系来自同一研究且目标control可见，其证据范围仍是control-observed跨细胞系复制。

## 审计边界

- E153没有重新训练、重新打分、换任务或换端点。
- E152 frozen score SHA-256：`{freeze_status['e152_validation']['score_file_sha256']}`；E135 frozen model SHA-256：`{freeze_status['e152_validation']['model_file_sha256']}`；两者均验证通过。E151 strict issue count：`{freeze_status['e151_validation']['strict_issue_count']}`。
- 八研究平均效应、异质性和LODO不能保证未来研究、期刊录用或湿实验机制验证。
"""
    (REPORTS / "E153_REPORT.md").write_text(report)


def analyze() -> None:
    absolute, directions, frozen = verify_frozen()
    dedup = deterministic_dedup(absolute)
    per_cluster = absolute.groupby(["dataset", "context", "perturbation"], as_index=False).size()
    multiplicity = absolute.groupby("dataset", as_index=False).agg(
        n_rows=("task_id", "size"),
        n_folds=("fold_id", "nunique"),
        n_contexts=("context", "nunique"),
        n_unique_perturbations=("perturbation", "nunique"),
    )
    context_stats = per_cluster.groupby("dataset", as_index=False).agg(
        n_unique_context_tasks=("perturbation", "size"),
        mean_outer_fold_occurrences=("size", "mean"),
        median_outer_fold_occurrences=("size", "median"),
        max_outer_fold_occurrences=("size", "max"),
    )
    multiplicity = multiplicity.merge(context_stats, on="dataset", validate="one_to_one")
    multiplicity["row_to_unique_context_task_ratio"] = (
        multiplicity.n_rows / multiplicity.n_unique_context_tasks
    )

    fold_rows = []
    for (dataset, fold_id), group in absolute.groupby(["dataset", "fold_id"], sort=True):
        for score in SCORES:
            fold_rows.append(
                {
                    "dataset": dataset,
                    "fold_id": fold_id,
                    "score": score,
                    "target": TARGET,
                    "n_tasks": len(group),
                    "spearman": rho(group[score], group[TARGET]),
                }
            )
    fold_correlations = pd.DataFrame(fold_rows)

    draw_frames = []
    observed_by_key: dict[tuple[str, str], dict[str, float]] = {}
    for dataset in sorted(absolute.dataset.unique()):
        main_draws, main_observed = absolute_cluster_draws(
            absolute, dataset, FOLD_ESTIMAND, True
        )
        sensitivity_draws, sensitivity_observed = absolute_cluster_draws(
            dedup, dataset, POOLED_ESTIMAND, False
        )
        draw_frames.extend([main_draws, sensitivity_draws])
        observed_by_key[(dataset, FOLD_ESTIMAND)] = main_observed
        observed_by_key[(dataset, POOLED_ESTIMAND)] = sensitivity_observed
    draws = pd.concat(draw_frames, ignore_index=True)
    boot_summary = bootstrap_summary(draws, observed_by_key)
    studies = build_study_effects(absolute, dedup, draws)
    meta = meta_table(studies)
    lodo = lodo_table(studies)

    directional_folds, directional_study, directional_draws, directional_fixed = direction_results(directions)

    multiplicity.to_csv(TABLES / "E153_CLUSTER_MULTIPLICITY.csv", index=False)
    dedup.to_csv(TABLES / "E153_POOLED_MEDIAN_TASK_TABLE.csv", index=False)
    fold_correlations.to_csv(TABLES / "E153_ABSOLUTE_FOLD_CORRELATIONS.csv", index=False)
    draws.to_csv(TABLES / "E153_ABSOLUTE_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    boot_summary.to_csv(TABLES / "E153_ABSOLUTE_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    studies.to_csv(TABLES / "E153_ABSOLUTE_STUDY_EFFECTS.csv", index=False)
    meta.to_csv(TABLES / "E153_ABSOLUTE_RANDOM_EFFECTS_META.csv", index=False)
    lodo.to_csv(TABLES / "E153_ABSOLUTE_LODO.csv", index=False)
    directional_folds.to_csv(TABLES / "E153_DIRECTIONAL_FOLD_CORRELATIONS.csv", index=False)
    directional_study.to_csv(TABLES / "E153_DIRECTIONAL_STUDY_RESULTS.csv", index=False)
    directional_draws.to_csv(TABLES / "E153_DIRECTIONAL_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    directional_fixed.to_csv(TABLES / "E153_DIRECTIONAL_TWO_STUDY_DESCRIPTIVE.csv", index=False)

    make_figures(boot_summary, directional_study, directional_fixed)
    write_report(
        absolute,
        multiplicity,
        fold_correlations,
        boot_summary,
        meta,
        lodo,
        directional_study,
        directional_fixed,
        frozen,
    )
    (OUT / "README_先看这个.md").write_text(
        "# E153 先看这个\n\n"
        "先读 `ANALYSIS_CONTRACT.md`，再读 `reports/E153_REPORT.md`。\n"
        "E153 是 post-E152 八研究扩展元分析，不是新的预注册独立 gate。\n"
    )

    primary_meta = meta[
        meta.estimand.eq(FOLD_ESTIMAND)
        & meta.analysis.eq("safeconf_minus_comparator")
    ].set_index("effect")
    direction_primary = directional_fixed[
        directional_fixed.analysis.eq("score_association")
        & directional_fixed.effect.eq("directional_risk_frozen")
        & directional_fixed.endpoint.eq("direction_error_rank_target")
    ].iloc[0]
    status = {
        "experiment": "E153_eight_study_formal_meta",
        "phase": "complete_post_E152_expanded_meta",
        "generated_at": now(),
        "analysis_nature": "post_E152_expanded_meta_not_preregistered_independent_gate",
        "freeze_status_sha256": sha256_file(FREEZE_STATUS, prefix=True),
        "n_absolute_rows": len(absolute),
        "n_studies": int(absolute.dataset.nunique()),
        "n_folds": int(absolute[["dataset", "fold_id"]].drop_duplicates().shape[0]),
        "n_unique_context_tasks": len(dedup),
        "n_unique_perturbation_clusters": int(absolute[["dataset", "perturbation"]].drop_duplicates().shape[0]),
        "n_bootstrap_per_study_per_estimand": N_BOOT,
        "primary_estimand": FOLD_ESTIMAND,
        "sensitivity_estimand": POOLED_ESTIMAND,
        "random_effects_method": "Fisher z + perturbation-cluster SE + REML tau2 + modified Knapp-Hartung + t prediction interval",
        "primary_meta_safeconf_minus_disagreement": {
            key: float(primary_meta.loc["safeconf_minus_disagreement", key])
            for key in ["pooled_z", "ci95_low_z", "ci95_high_z", "tau2_reml", "I2_percent", "prediction_low_z", "prediction_high_z"]
        },
        "primary_meta_safeconf_minus_magnitude": {
            key: float(primary_meta.loc["safeconf_minus_magnitude", key])
            for key in ["pooled_z", "ci95_low_z", "ci95_high_z", "tau2_reml", "I2_percent", "prediction_low_z", "prediction_high_z"]
        },
        "directional_two_study_scope": "fixed Nadig and Replogle descriptive combination; k=2; no random effects or population prediction interval",
        "directional_two_study_equal_mean": float(direction_primary.equal_study_mean_spearman),
        "directional_two_study_fixed_bootstrap_ci95": [
            float(direction_primary.fixed_two_study_bootstrap_ci95_low),
            float(direction_primary.fixed_two_study_bootstrap_ci95_high),
        ],
        "e151_strict_issue_count": int(frozen["e151_validation"]["strict_issue_count"]),
        "e152_score_freeze_hash_verified": bool(frozen["e152_validation"]["score_freeze_hash_verified"]),
        "e152_model_hash_verified": bool(frozen["e152_validation"]["model_hash_verified"]),
        "e152_preregistered_directional_gate_passed": bool(frozen["e152_validation"]["preregistered_directional_gate_passed"]),
        "truth_used_to_change_scores_tasks_or_endpoints": False,
        "confirmatory_gate_claim_allowed_for_E153": False,
    }
    RUN_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print("\nPrimary absolute meta")
    print(primary_meta[["pooled_z", "ci95_low_z", "ci95_high_z", "prediction_low_z", "prediction_high_z"]].to_string())
    print("\nDirectional fixed two-study primary")
    print(direction_primary.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["freeze", "analyze", "all"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase in {"freeze", "all"}:
        freeze()
    if args.phase in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
