#!/usr/bin/env python3
"""Seal E200 cross-context risk features before evaluating target outcomes."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E200_txpert_cross_context_k562_20260802"
FREEZE = OUT / "PRETRUTH_FEATURE_FREEZE.md"
SUPPORT = OUT / "tables/E200_STRICT_TASK_SUPPORT.csv"
SEAL = OUT / "E200_PREDICTION_SEAL_STATUS.json"
RELEASE = OUT / "pretruth_release"
TABLES = RELEASE / "tables"
REPORTS = RELEASE / "reports"

DATA = Path("/home/yyf/data/txpert_official_20260802")
CACHE = DATA / "cache"
PRED = DATA / "e200/predictions"
REFERENCE = CACHE / "K562_cross_cell_lines/de_adata_test.h5ad"
SPLIT = CACHE / "K562_cross_cell_lines/splits/train_test_split.pkl"
SUBGROUP = CACHE / "K562_cross_cell_lines/splits/subgroup.pkl"

GAT_PRED = PRED / "gat/test_predictions.h5ad"
GAT_CONTROL = PRED / "gat/test_controls.h5ad"
BASELINE_PRED = PRED / "general_baseline/test_predictions.h5ad"

EXPECTED = {
    GAT_PRED: (
        2_235_398_016,
        "7647d4c2665ee4c546ea32e429c49d40700b90ca27104515dccb4084f41ec09f",
    ),
    GAT_CONTROL: (
        2_235_398_016,
        "6737ebaf794776a59ef6110cd1ae131c086119152901d62bdb4b462bae8bbed8",
    ),
    BASELINE_PRED: (
        2_025_572_536,
        "0d2200b0762b5aa4f7f29314bbda99032a78a4f959f937c9d14cbd444b437d30",
    ),
    REFERENCE: (
        7_767_053_064,
        "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8",
    ),
    SPLIT: (
        31_629,
        "c922dc62ee4263951ec6a45e6e8cfc51e4104d5e1b0704eefd46848acddba402",
    ),
    SUBGROUP: (
        14_436,
        "3d0a2f92fdad7809e5e13f4931c0a7eca49e360fd83a25a3b8ca90ec6ebe9e8b",
    ),
    SUPPORT: (
        11_862,
        "0d2ee1a379aa98a81d927692919f09faa04b18477de740eb1e3266abfce1312e",
    ),
    SEAL: (
        642,
        "ccb01511b5a5685dc4c44ea46ff9df6efb573038bd56e75cc8bbb71b4743d486",
    ),
}

TRAIN_CONTEXTS = ("RPE1", "hepg2", "jurkat")
N_CELLS = 150_472
N_GENES = 3_352
N_TASKS = 580
OBS_HASH = "33cc9fbfc6ea04da16e1e6d82368e5913242f3a88dc8de34e6f781d0d968521c"
VAR_HASH = "d67c176fda6515159421fea6fbaca860240cb6980ccc51745bff619dfec489ca"
PERT_HASH = "77770556a104100464288fa0fdb9caa95fb6cb3be90a4b5ef04348c601346f62"
BATCH_HASH = "5691048375390ffc72cefc3334b73529c28d0230a570fc09d3663bff52931519"
CELL_HASH = "9be081b0a2a81c0328f9398fdff81398f87c657bcd9f29dd475386a756b1d267"


class PretruthFailure(RuntimeError):
    """Fail-closed E200 pretruth release error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_hash(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def remote_tip(remote: str, branch: str) -> str:
    line = git_output("ls-remote", remote, f"refs/heads/{branch}")
    if not line:
        raise PretruthFailure(f"missing remote branch: {remote}/{branch}")
    return line.split()[0]


def tracked_clean(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{relative}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    clean = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", relative],
        check=False,
    ).returncode == 0
    staged = subprocess.run(
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
        check=False,
    ).returncode == 0
    return tracked and clean and staged


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    for root, prefix in (
        (ROOT.resolve(), ""),
        (Path("/home/yyf/data").resolve(), "DATA/"),
    ):
        try:
            return prefix + resolved.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.name


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def verify_git_release() -> str:
    if not tracked_clean(SCRIPT) or not tracked_clean(FREEZE):
        raise PretruthFailure("pretruth runner/freeze is not tracked and clean")
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    if not branch:
        raise PretruthFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if remote_tip(remote, branch) != head:
            raise PretruthFailure(f"{remote}/{branch} does not match local HEAD")
    return head


def verify_inputs() -> pd.DataFrame:
    rows = []
    for path, (expected_bytes, expected_sha) in EXPECTED.items():
        if not path.is_file():
            raise PretruthFailure(f"missing sealed input: {path}")
        observed_bytes = path.stat().st_size
        observed_sha = sha256_file(path)
        if observed_bytes != expected_bytes or observed_sha != expected_sha:
            raise PretruthFailure(f"sealed input mismatch: {path}")
        rows.append(
            {
                "path": logical_path(path),
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)


def verify_source_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    forbidden_filename = "test_" + "ground" + "_truth.h5ad"
    if forbidden_filename in string_literals or forbidden_filename in source:
        raise PretruthFailure("target outcome file entered pretruth source")
    reference_x_accesses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "X"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "dataset"
    ]
    if len(reference_x_accesses) != 1:
        raise PretruthFailure("reference X access must have one guarded call site")


def verify_prediction(handle: ad.AnnData, label: str) -> None:
    if handle.shape != (N_CELLS, N_GENES):
        raise PretruthFailure(f"{label} shape changed: {handle.shape}")
    checks = {
        "obs": (order_hash(handle.obs_names), OBS_HASH),
        "var": (order_hash(handle.var_names), VAR_HASH),
        "pert": (order_hash(handle.obs.pert_cond_names.astype(str)), PERT_HASH),
        "batch": (
            order_hash(handle.obs.experimental_batches.astype(str)),
            BATCH_HASH,
        ),
        "cell": (order_hash(handle.obs.cell_types.astype(str)), CELL_HASH),
    }
    failed = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if failed:
        raise PretruthFailure(f"{label} alignment changed: {failed}")


def condition_from_label(label: str) -> str:
    prefix, suffix = "K562_", "_1+1"
    if not label.startswith(prefix) or not label.endswith(suffix):
        raise PretruthFailure(f"unexpected prediction label: {label}")
    condition = label[len(prefix) : -len(suffix)]
    parts = condition.split("+")
    if len(parts) != 2 or parts[1] != "ctrl":
        raise PretruthFailure(f"not a single-gene condition: {label}")
    return condition


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(delta))))


def centroid(handle: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(handle.X[indices], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise PretruthFailure("non-finite or malformed prediction block")
    return values.mean(axis=0)


def guarded_reference_rows(
    dataset: ad.AnnData,
    obs: pd.DataFrame,
    indices: np.ndarray,
    row_kind: str,
    strict: set[str],
    access_rows: list[dict[str, Any]],
):
    selected = obs.iloc[indices]
    if len(selected) == 0:
        raise PretruthFailure(f"empty reference access: {row_kind}")
    if not selected.cell_line.astype(str).isin(TRAIN_CONTEXTS).all():
        raise PretruthFailure(f"target context entered reference X: {row_kind}")
    controls = selected.control.astype(bool)
    if row_kind == "training_control":
        if not controls.all():
            raise PretruthFailure("perturbed row entered training-control access")
    elif row_kind == "training_perturbation":
        if controls.any() or not selected.condition.astype(str).isin(strict).all():
            raise PretruthFailure("invalid row entered training-perturbation access")
    else:
        raise PretruthFailure(f"unknown reference row kind: {row_kind}")
    contexts = sorted(selected.cell_line.astype(str).unique())
    access_rows.append(
        {
            "source": logical_path(REFERENCE),
            "row_kind": row_kind,
            "contexts": ";".join(contexts),
            "n_rows": len(indices),
            "target_K562_perturbation_rows": 0,
            "status": "PASS",
        }
    )
    return dataset.X[indices]


def row_mean(matrix, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        raise PretruthFailure("cannot average empty matrix block")
    return np.asarray(matrix[indices].mean(axis=0), dtype=np.float64).ravel()


def build_training_deltas(
    strict: set[str], access_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    dataset = ad.read_h5ad(REFERENCE, backed="r")
    obs = dataset.obs.copy()
    deltas: dict[str, dict[str, np.ndarray]] = {condition: {} for condition in strict}
    support_rows = []
    try:
        for context in TRAIN_CONTEXTS:
            is_context = obs.cell_line.astype(str).eq(context)
            control_global = np.flatnonzero(
                (is_context & obs.control.astype(bool)).to_numpy()
            )
            perturb_global = np.flatnonzero(
                (
                    is_context
                    & ~obs.control.astype(bool)
                    & obs.condition.astype(str).isin(strict)
                ).to_numpy()
            )
            control_x = guarded_reference_rows(
                dataset,
                obs,
                control_global,
                "training_control",
                strict,
                access_rows,
            )
            perturb_x = guarded_reference_rows(
                dataset,
                obs,
                perturb_global,
                "training_perturbation",
                strict,
                access_rows,
            )
            control_obs = obs.iloc[control_global].reset_index(drop=True)
            perturb_obs = obs.iloc[perturb_global].reset_index(drop=True)
            control_means: dict[str, np.ndarray] = {}
            control_batches = control_obs.batch.astype(str).to_numpy()
            for batch in sorted(pd.unique(control_batches)):
                local = np.flatnonzero(control_batches == batch)
                control_means[batch] = row_mean(control_x, local)
            conditions = perturb_obs.condition.astype(str).to_numpy()
            batches = perturb_obs.batch.astype(str).to_numpy()
            for condition in sorted(pd.unique(conditions)):
                local = np.flatnonzero(conditions == condition)
                condition_batches, batch_counts = np.unique(
                    batches[local], return_counts=True
                )
                missing = [
                    batch for batch in condition_batches if batch not in control_means
                ]
                if missing:
                    raise PretruthFailure(
                        f"missing matched controls: {context}/{condition}/{missing}"
                    )
                perturb_mean = row_mean(perturb_x, local)
                matched_control = sum(
                    control_means[batch] * int(count)
                    for batch, count in zip(condition_batches, batch_counts)
                ) / len(local)
                delta = perturb_mean - matched_control
                if not np.isfinite(delta).all():
                    raise PretruthFailure(
                        f"non-finite training delta: {context}/{condition}"
                    )
                deltas[condition][context] = delta
                support_rows.append(
                    {
                        "task_id": condition,
                        "training_context": context,
                        "n_perturbed_cells": len(local),
                        "n_matched_batches": len(condition_batches),
                        "n_available_control_cells": len(control_obs),
                    }
                )
    finally:
        dataset.file.close()
    if any(not context_map for context_map in deltas.values()):
        missing = sorted(k for k, value in deltas.items() if not value)
        raise PretruthFailure(f"strict tasks without training delta: {missing[:10]}")
    return deltas, pd.DataFrame(support_rows)


def dispersion(context_deltas: dict[str, np.ndarray]) -> float:
    members = np.stack(list(context_deltas.values()), axis=0)
    center = members.mean(axis=0)
    return float(np.sqrt(np.mean(np.square(members - center[None, :]))))


def zscore(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    scale = float(numeric.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise PretruthFailure(f"non-positive z-score scale: {values.name}")
    return (numeric - float(numeric.mean())) / scale


def build_features(
    access_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = pd.read_csv(SUPPORT, keep_default_na=False)
    if len(support) != N_TASKS or support.condition.nunique() != N_TASKS:
        raise PretruthFailure("strict support table cardinality changed")
    strict = set(support.condition.astype(str))
    split = joblib.load(SPLIT)
    subgroup = joblib.load(SUBGROUP)["test_subgroup"]
    expected_strict = set(map(str, subgroup["unseen_cell"])) & set(split["train"])
    if strict != expected_strict:
        raise PretruthFailure("strict support table differs from official split")
    deltas, delta_support = build_training_deltas(strict, access_rows)

    handles = {
        "gat": ad.read_h5ad(GAT_PRED, backed="r"),
        "control": ad.read_h5ad(GAT_CONTROL, backed="r"),
        "baseline": ad.read_h5ad(BASELINE_PRED, backed="r"),
    }
    rows = []
    try:
        for key, handle in handles.items():
            verify_prediction(handle, key)
        labels = handles["gat"].obs.pert_cond_names.astype(str).to_numpy()
        condition_to_label = {}
        for label in pd.unique(labels):
            condition = condition_from_label(str(label))
            if condition in condition_to_label:
                raise PretruthFailure(f"duplicate condition label: {condition}")
            condition_to_label[condition] = str(label)
        if not strict.issubset(condition_to_label):
            raise PretruthFailure("strict tasks missing from predictions")

        support_index = support.set_index("condition")
        for condition in sorted(strict):
            label = condition_to_label[condition]
            indices = np.flatnonzero(labels == label)
            gat = centroid(handles["gat"], indices)
            control = centroid(handles["control"], indices)
            baseline = centroid(handles["baseline"], indices)
            context_map = deltas[condition]
            observed_dispersion = (
                dispersion(context_map) if len(context_map) >= 2 else np.nan
            )
            frozen_support = support_index.loc[condition]
            rows.append(
                {
                    "task_id": condition,
                    "condition_label": label,
                    "gene": condition.split("+")[0],
                    "cell_line": "K562",
                    "n_prediction_cells": len(indices),
                    "n_train_cells": int(
                        delta_support.loc[
                            delta_support.task_id.eq(condition),
                            "n_perturbed_cells",
                        ].sum()
                    ),
                    "n_train_contexts": len(context_map),
                    "model_baseline_gap": rmse(gat, baseline),
                    "training_delta_dispersion_observed": observed_dispersion,
                    "training_delta_dispersion_imputed": len(context_map) == 1,
                    "negative_log_train_cells": -math.log1p(
                        int(frozen_support.n_train_cells)
                    ),
                    "support_context_deficit": 3 - len(context_map),
                    "predicted_magnitude": rmse(gat, control),
                }
            )
    finally:
        for handle in handles.values():
            handle.file.close()
    frame = pd.DataFrame(rows)
    observed = frame.training_delta_dispersion_observed.dropna()
    if len(observed) != 505:
        raise PretruthFailure(
            f"unexpected observed dispersion count: {len(observed)}"
        )
    imputation = float(observed.median())
    frame["training_delta_dispersion"] = frame[
        "training_delta_dispersion_observed"
    ].fillna(imputation)
    components = {
        "z_model_baseline_gap": "model_baseline_gap",
        "z_training_delta_dispersion": "training_delta_dispersion",
        "z_negative_log_train_cells": "negative_log_train_cells",
        "z_support_context_deficit": "support_context_deficit",
    }
    for z_name, raw_name in components.items():
        frame[z_name] = zscore(frame[raw_name])
    frame["transfer_risk"] = frame[list(components)].mean(axis=1)
    frame["dispersion_imputation_value"] = imputation
    frame["analysis_stratum"] = np.where(
        frame.n_prediction_cells.ge(30), "primary_ge30", "sensitivity_10_29"
    )
    return frame, delta_support


def build_gates(
    frame: pd.DataFrame,
    delta_support: pd.DataFrame,
    access: pd.DataFrame,
) -> pd.DataFrame:
    support = pd.read_csv(SUPPORT, keep_default_na=False).set_index("condition")
    observed = frame.set_index("task_id")
    components = [
        "z_model_baseline_gap",
        "z_training_delta_dispersion",
        "z_negative_log_train_cells",
        "z_support_context_deficit",
    ]
    identity = np.max(
        np.abs(
            observed.transfer_risk.to_numpy()
            - observed[components].mean(axis=1).to_numpy()
        )
    )
    train_counts = delta_support.groupby("task_id").n_perturbed_cells.sum()
    rows = [
        ("n_tasks", len(frame), N_TASKS, len(frame) == N_TASKS),
        (
            "task_keys_unique",
            frame.task_id.nunique(),
            N_TASKS,
            frame.task_id.nunique() == N_TASKS,
        ),
        (
            "primary_tasks_ge30",
            int(frame.n_prediction_cells.ge(30).sum()),
            566,
            int(frame.n_prediction_cells.ge(30).sum()) == 566,
        ),
        (
            "sensitivity_tasks_10_29",
            int(frame.n_prediction_cells.between(10, 29).sum()),
            14,
            int(frame.n_prediction_cells.between(10, 29).sum()) == 14,
        ),
        (
            "imputed_single_context_tasks",
            int(frame.training_delta_dispersion_imputed.sum()),
            75,
            int(frame.training_delta_dispersion_imputed.sum()) == 75,
        ),
        (
            "target_prediction_counts_match_prepare",
            int(
                (
                    observed.n_prediction_cells.astype(int).to_numpy()
                    == support.loc[observed.index].n_target_cells.astype(int).to_numpy()
                ).sum()
            ),
            N_TASKS,
            np.array_equal(
                observed.n_prediction_cells.astype(int).to_numpy(),
                support.loc[observed.index].n_target_cells.astype(int).to_numpy(),
            ),
        ),
        (
            "train_cell_counts_match_prepare",
            int(
                (
                    train_counts.loc[observed.index].astype(int).to_numpy()
                    == support.loc[observed.index].n_train_cells.astype(int).to_numpy()
                ).sum()
            ),
            N_TASKS,
            np.array_equal(
                train_counts.loc[observed.index].astype(int).to_numpy(),
                support.loc[observed.index].n_train_cells.astype(int).to_numpy(),
            ),
        ),
        (
            "train_context_counts_match_prepare",
            int(
                (
                    observed.n_train_contexts.astype(int).to_numpy()
                    == support.loc[observed.index].n_train_contexts.astype(int).to_numpy()
                ).sum()
            ),
            N_TASKS,
            np.array_equal(
                observed.n_train_contexts.astype(int).to_numpy(),
                support.loc[observed.index].n_train_contexts.astype(int).to_numpy(),
            ),
        ),
        (
            "training_context_task_rows",
            len(delta_support),
            1_428,
            len(delta_support) == 1_428,
        ),
        (
            "guarded_reference_accesses",
            len(access),
            6,
            len(access) == 6,
        ),
        (
            "target_K562_perturbation_rows_opened",
            int(access.target_K562_perturbation_rows.sum()),
            0,
            int(access.target_K562_perturbation_rows.sum()) == 0,
        ),
        (
            "risk_values_finite",
            bool(
                np.isfinite(
                    frame[
                        [
                            "model_baseline_gap",
                            "training_delta_dispersion",
                            "negative_log_train_cells",
                            "support_context_deficit",
                            "predicted_magnitude",
                            "transfer_risk",
                        ]
                    ].to_numpy(float)
                ).all()
            ),
            True,
            bool(
                np.isfinite(
                    frame[
                        [
                            "model_baseline_gap",
                            "training_delta_dispersion",
                            "negative_log_train_cells",
                            "support_context_deficit",
                            "predicted_magnitude",
                            "transfer_risk",
                        ]
                    ].to_numpy(float)
                ).all()
            ),
        ),
        (
            "transfer_risk_identity_max_abs_residual",
            float(identity),
            "<1e-12",
            bool(identity < 1e-12),
        ),
    ]
    for component in components:
        mean = float(frame[component].mean())
        std = float(frame[component].std(ddof=0))
        rows.extend(
            [
                (
                    f"{component}_mean_abs",
                    abs(mean),
                    "<1e-12",
                    abs(mean) < 1e-12,
                ),
                (
                    f"{component}_std_abs_delta",
                    abs(std - 1.0),
                    "<1e-12",
                    abs(std - 1.0) < 1e-12,
                ),
            ]
        )
    return pd.DataFrame(rows, columns=["check", "observed", "expected", "passed"])


def report(frame: pd.DataFrame, gates: pd.DataFrame) -> str:
    imputation = float(frame.dispersion_imputation_value.iloc[0])
    return "\n".join(
        [
            "# E200 结果打开前特征封存",
            "",
            f"- 任务：{len(frame)} 个严格 context-only；主分析 "
            f"{int(frame.n_prediction_cells.ge(30).sum())} 个，敏感性 "
            f"{int(frame.n_prediction_cells.between(10, 29).sum())} 个。",
            f"- 训练背景支持：1/2/3 个背景的任务分别为 "
            f"{int(frame.n_train_contexts.eq(1).sum())}/"
            f"{int(frame.n_train_contexts.eq(2).sum())}/"
            f"{int(frame.n_train_contexts.eq(3).sum())}。",
            f"- 单背景离散度填补中位数：{imputation:.8g}，"
            f"共 {int(frame.training_delta_dispersion_imputed.sum())} 个任务。",
            f"- 封存门槛：{int(gates.passed.sum())}/{len(gates)} PASS。",
            "- K562 扰动表达读取行数：0；尚未计算任何目标误差或路由效用。",
            "",
        ]
    )


def main() -> None:
    if RELEASE.exists():
        raise PretruthFailure("pretruth release is append-only and already exists")
    verify_source_boundary()
    head = verify_git_release()
    input_hashes = verify_inputs()
    access_rows: list[dict[str, Any]] = []
    frame, delta_support = build_features(access_rows)
    access = pd.DataFrame(access_rows)
    gates = build_gates(frame, delta_support, access)
    if not gates.passed.astype(bool).all():
        failed = gates.loc[~gates.passed.astype(bool), "check"].tolist()
        raise PretruthFailure(f"pretruth gates failed: {failed}")
    atomic_csv(TABLES / "E200_PRETRUTH_INPUT_HASHES.csv", input_hashes)
    atomic_csv(TABLES / "E200_PRETRUTH_FEATURES.csv", frame)
    atomic_csv(TABLES / "E200_TRAINING_DELTA_SUPPORT.csv", delta_support)
    atomic_csv(TABLES / "E200_PRETRUTH_ACCESS_AUDIT.csv", access)
    atomic_csv(TABLES / "E200_PRETRUTH_GATES.csv", gates)
    atomic_text(REPORTS / "E200_PRETRUTH_REPORT.md", report(frame, gates))
    status = {
        "experiment": "E200_txpert_cross_context_k562",
        "stage": "PRETRUTH_FEATURE_RELEASE",
        "generated_at": now(),
        "status": "PASS",
        "git_head": head,
        "n_tasks": len(frame),
        "n_primary_tasks_ge30": int(frame.n_prediction_cells.ge(30).sum()),
        "n_sensitivity_tasks_10_29": int(
            frame.n_prediction_cells.between(10, 29).sum()
        ),
        "n_single_context_dispersion_imputed": int(
            frame.training_delta_dispersion_imputed.sum()
        ),
        "target_K562_perturbation_expression_rows_opened": int(
            access.target_K562_perturbation_rows.sum()
        ),
        "target_outcomes_evaluated": False,
        "gates_passed": int(gates.passed.sum()),
        "gates_total": len(gates),
    }
    atomic_json(RELEASE / "E200_PRETRUTH_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
