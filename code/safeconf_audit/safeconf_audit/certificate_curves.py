#!/usr/bin/env python3
"""Recompute SafeConf-Cert operating curves from the released E183 table.

The command deliberately reads only the committed E181/E182/E183 CSV/JSON
release.  It does not open raw expression matrices, model checkpoints, or any
E205/E208 truth artifact.

For an application error tolerance ``tau`` and a deterministic lower bound
``L``, ``L > tau`` certifies that the corresponding observed error ``R`` is
also above ``tau``.  A small lower bound is *unknown*, not evidence of safety.
The calibrated upper bound is audited separately because its coverage is
conditional on the stated conformal assumptions, not a pointwise guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


E181_REL = Path(
    "docs/实验结果/E181_registered_family_hilbert_certificate_20260724"
)
E182_REL = Path(
    "docs/实验结果/E182_gse225807_registered_family_20260724/final_evaluation"
)
E183_REL = Path(
    "docs/实验结果/E183_all_study_family_synthesis_20260724"
)
DEFAULT_TAU = (
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,
    0.060,
    0.070,
    0.080,
    0.100,
    0.150,
    0.200,
    0.300,
)
NUMERIC_TOLERANCE = 1e-10


class CertificateIntegrityError(RuntimeError):
    """The released certificate table violates its recorded contract."""


@dataclass(frozen=True)
class Objective:
    name: str
    error: str
    lower: str
    upper: str
    stored_tightness: str
    stored_width: str


OBJECTIVES = (
    Objective(
        "family_rms",
        "family_rms_error",
        "diversity_lower",
        "family_upper",
        "family_lower_tightness",
        "family_interval_width",
    ),
    Objective(
        "worst_member",
        "worst_member_error",
        "diameter_lower",
        "worst_upper",
        "worst_lower_tightness",
        "worst_interval_width",
    ),
)


def default_repo() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_tau_grid(value: str | None) -> tuple[float, ...]:
    """Parse a comma-separated, strictly increasing non-negative tau grid."""
    if value is None:
        return DEFAULT_TAU
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tau values must be numeric") from exc
    if not values:
        raise argparse.ArgumentTypeError("tau grid must contain at least one value")
    if not all(np.isfinite(values)) or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("tau values must be finite and non-negative")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("tau values must be unique and strictly increasing")
    return values


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    unknown = set(normalized.unique()) - allowed
    if unknown:
        raise CertificateIntegrityError(f"unrecognized boolean values: {sorted(unknown)}")
    return normalized.isin({"true", "1", "yes"})


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise CertificateIntegrityError(f"{column} contains missing/non-finite values")


def validate_task_table(tasks: pd.DataFrame, tol: float = NUMERIC_TOLERANCE) -> dict:
    """Validate the numerical certificate identities used by the curve audit."""
    required = {
        "study",
        "task_id",
        "target_cluster",
        "family",
        "n_members",
        "family_identity_abs_residual",
        "family_lower_violation",
        "worst_lower_violation",
        "family_upper_covered",
        "worst_upper_covered",
    }
    for objective in OBJECTIVES:
        required.update(
            {
                objective.error,
                objective.lower,
                objective.upper,
                objective.stored_tightness,
                objective.stored_width,
            }
        )
    missing = sorted(required - set(tasks.columns))
    if missing:
        raise CertificateIntegrityError(f"missing task columns: {missing}")
    if tasks.empty or tasks["task_id"].duplicated().any():
        raise CertificateIntegrityError("task table is empty or task_id is not unique")
    if not tasks["family"].eq("frozen_10_seed_family").all():
        raise CertificateIntegrityError("E183 must contain only frozen_10_seed_family")
    if not pd.to_numeric(tasks["n_members"]).eq(10).all():
        raise CertificateIntegrityError("registered family size changed")

    numeric = ["family_identity_abs_residual"]
    for objective in OBJECTIVES:
        numeric.extend(
            [
                objective.error,
                objective.lower,
                objective.upper,
                objective.stored_tightness,
                objective.stored_width,
            ]
        )
    _finite_numeric(tasks, numeric)

    checks: dict[str, object] = {}
    for objective in OBJECTIVES:
        error = tasks[objective.error].to_numpy(float)
        lower = tasks[objective.lower].to_numpy(float)
        upper = tasks[objective.upper].to_numpy(float)
        if (lower < -tol).any() or (upper < -tol).any():
            raise CertificateIntegrityError(f"{objective.name}: negative bound")
        direct_lower_violation = lower > error + tol
        stored_lower_violation = _as_bool(
            tasks[
                "family_lower_violation"
                if objective.name == "family_rms"
                else "worst_lower_violation"
            ]
        ).to_numpy(bool)
        if not np.array_equal(direct_lower_violation, stored_lower_violation):
            raise CertificateIntegrityError(
                f"{objective.name}: stored lower violation flags disagree with values"
            )
        direct_upper_covered = error <= upper + tol
        stored_upper_covered = _as_bool(
            tasks[
                "family_upper_covered"
                if objective.name == "family_rms"
                else "worst_upper_covered"
            ]
        ).to_numpy(bool)
        if not np.array_equal(direct_upper_covered, stored_upper_covered):
            raise CertificateIntegrityError(
                f"{objective.name}: stored upper coverage flags disagree with values"
            )
        recomputed_tightness = np.divide(
            lower,
            error,
            out=np.full_like(lower, np.nan),
            where=error > tol,
        )
        stored_tightness = tasks[objective.stored_tightness].to_numpy(float)
        if not np.allclose(
            recomputed_tightness,
            stored_tightness,
            atol=tol,
            rtol=tol,
            equal_nan=True,
        ):
            raise CertificateIntegrityError(
                f"{objective.name}: stored lower tightness is not lower/error"
            )
        width = upper - lower
        if (width < -tol).any():
            raise CertificateIntegrityError(
                f"{objective.name}: upper bound is below lower bound"
            )
        if not np.allclose(
            width,
            tasks[objective.stored_width].to_numpy(float),
            atol=tol,
            rtol=tol,
        ):
            raise CertificateIntegrityError(
                f"{objective.name}: stored interval width is not upper-lower"
            )
        checks[f"{objective.name}_lower_violations"] = int(direct_lower_violation.sum())
        checks[f"{objective.name}_upper_task_coverage"] = float(direct_upper_covered.mean())

    max_residual = float(tasks["family_identity_abs_residual"].abs().max())
    if max_residual > tol:
        raise CertificateIntegrityError(
            f"Hilbert identity residual {max_residual:.3e} exceeds {tol:.1e}"
        )
    checks["max_hilbert_identity_abs_residual"] = max_residual
    return checks


def load_released_tasks(repo: Path) -> tuple[pd.DataFrame, dict]:
    """Load E183 and verify its lineage to the released E181/E182 tables."""
    e181_path = repo / E181_REL / "tables/E181_TASK_CERTIFICATES.csv"
    e182_path = repo / E182_REL / "tables/E182_EVALUATION_TASKS.csv"
    e182_status_path = repo / E182_REL / "E182_FINAL_SUMMARY.json"
    e183_path = repo / E183_REL / "tables/E183_COMBINED_TASK_CERTIFICATES.csv"
    e183_targets_path = repo / E183_REL / "tables/E183_TARGET_CERTIFICATES.csv"
    e183_status_path = repo / E183_REL / "RUN_STATUS.json"
    for path in (
        e181_path,
        e182_path,
        e182_status_path,
        e183_path,
        e183_targets_path,
        e183_status_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    e181 = pd.read_csv(e181_path, keep_default_na=False)
    e181 = e181.loc[e181["family"].eq("frozen_10_seed_family")].copy()
    e182 = pd.read_csv(e182_path, keep_default_na=False)
    e182_status = json.loads(e182_status_path.read_text(encoding="utf-8"))
    tasks = pd.read_csv(e183_path, keep_default_na=False)
    reported_targets = pd.read_csv(e183_targets_path, keep_default_na=False)
    e183_status = json.loads(e183_status_path.read_text(encoding="utf-8"))
    if len(e181) != 2393 or len(e182) != 40 or len(tasks) != 2433:
        raise CertificateIntegrityError(
            f"release population changed: E181={len(e181)}, E182={len(e182)}, E183={len(tasks)}"
        )
    if e182_status.get("status") != "FAIL" or int(e182_status.get("covered_targets", -1)) != 16:
        raise CertificateIntegrityError("E182 registered FAIL state changed")

    prior = tasks.loc[~tasks["study"].eq("E182_GSE225807")].sort_values("task_id")
    source_prior = e181.sort_values("task_id")
    if prior["task_id"].tolist() != source_prior["task_id"].tolist():
        raise CertificateIntegrityError("E181-to-E183 task keys changed")
    comparison_columns = [
        "family_rms_error",
        "centroid_rmse",
        "diversity_lower",
        "worst_member_error",
        "diameter_lower",
        "family_upper",
        "worst_upper",
    ]
    if not np.allclose(
        prior[comparison_columns].to_numpy(float),
        source_prior[comparison_columns].to_numpy(float),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise CertificateIntegrityError("E181-to-E183 numeric mapping changed")

    new = tasks.loc[tasks["study"].eq("E182_GSE225807")].sort_values("task_id")
    source_new = e182.sort_values("task_id")
    if new["task_id"].tolist() != source_new["task_id"].tolist():
        raise CertificateIntegrityError("E182-to-E183 task keys changed")
    pairs = (
        ("family_rms_error", "family_rms_error"),
        ("centroid_rmse", "centroid_rmse"),
        ("diversity_lower", "family_diversity_lower"),
        ("worst_member_error", "worst_member_error"),
        ("diameter_lower", "worst_member_lower"),
        ("family_upper", "family_rms_upper"),
        ("worst_upper", "worst_member_upper"),
    )
    for mapped, source in pairs:
        if not np.allclose(
            new[mapped].to_numpy(float),
            source_new[source].to_numpy(float),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise CertificateIntegrityError(f"E182-to-E183 mapping changed for {mapped}")

    checks = validate_task_table(tasks)
    target_keys = ["study", "target_cluster"]
    target_coverage = (
        tasks.assign(
            _family_covered=(
                tasks["family_rms_error"].astype(float)
                <= tasks["family_upper"].astype(float) + NUMERIC_TOLERANCE
            ),
            _worst_covered=(
                tasks["worst_member_error"].astype(float)
                <= tasks["worst_upper"].astype(float) + NUMERIC_TOLERANCE
            ),
        )
        .groupby(target_keys, observed=True)
        .agg(
            family_upper_simultaneous_covered=("_family_covered", "all"),
            worst_upper_simultaneous_covered=("_worst_covered", "all"),
        )
        .reset_index()
        .sort_values(target_keys)
        .reset_index(drop=True)
    )
    reported_targets = reported_targets.sort_values(target_keys).reset_index(drop=True)
    if target_coverage[target_keys].to_records(index=False).tolist() != reported_targets[
        target_keys
    ].to_records(index=False).tolist():
        raise CertificateIntegrityError("E183 target-cluster keys changed")
    for column in (
        "family_upper_simultaneous_covered",
        "worst_upper_simultaneous_covered",
    ):
        if not np.array_equal(
            target_coverage[column].to_numpy(bool),
            _as_bool(reported_targets[column]).to_numpy(bool),
        ):
            raise CertificateIntegrityError(f"E183 target coverage changed for {column}")
    family_targets_covered = int(
        target_coverage["family_upper_simultaneous_covered"].sum()
    )
    worst_targets_covered = int(
        target_coverage["worst_upper_simultaneous_covered"].sum()
    )
    if (
        family_targets_covered != int(e183_status["family_upper_targets_covered"])
        or worst_targets_covered != int(e183_status["worst_upper_targets_covered"])
    ):
        raise CertificateIntegrityError("E183 RUN_STATUS target coverage changed")
    checks.update(
        {
            "e181_registered_family_tasks": len(e181),
            "e182_evaluation_tasks": len(e182),
            "e182_preregistered_status": "FAIL",
            "e183_tasks": len(tasks),
            "e183_target_clusters": int(
                tasks.groupby(["study", "target_cluster"], observed=True).ngroups
            ),
            "e183_family_upper_targets_covered": family_targets_covered,
            "e183_worst_upper_targets_covered": worst_targets_covered,
        }
    )
    return tasks, checks


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _curve_row(
    values: pd.DataFrame,
    *,
    tau: float,
    objective: str,
    study: str,
    unit: str,
) -> dict[str, object]:
    true_high = values["error"].to_numpy(float) > tau
    certified_high = values["lower"].to_numpy(float) > tau
    false_certificate = certified_high & ~true_high
    true_positive = certified_high & true_high
    upper_covered = (
        values["upper_covered"].to_numpy(bool)
        if "upper_covered" in values
        else values["error"].to_numpy(float)
        <= (values["upper"].to_numpy(float) + NUMERIC_TOLERANCE)
    )
    n = len(values)
    n_true = int(true_high.sum())
    n_certified = int(certified_high.sum())
    n_tp = int(true_positive.sum())
    return {
        "unit": unit,
        "study": study,
        "objective": objective,
        "tau": tau,
        "n_units": n,
        "n_true_high": n_true,
        "n_certified_high": n_certified,
        "n_certified_true_high": n_tp,
        "n_false_certificates": int(false_certificate.sum()),
        "certified_high_coverage": _safe_ratio(n_certified, n),
        "certified_high_recall": _safe_ratio(n_tp, n_true),
        "certified_high_precision": _safe_ratio(n_tp, n_certified),
        "unknown_fraction": _safe_ratio(n - n_certified, n),
        "upper_empirical_coverage": float(upper_covered.mean()),
        "guarantee_scope": (
            "high certificate is deterministic; upper coverage is conformal/empirical"
        ),
    }


def _objective_values(tasks: pd.DataFrame, objective: Objective) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study": tasks["study"].astype(str),
            "target_cluster": tasks["target_cluster"].astype(str),
            "error": tasks[objective.error].astype(float),
            "lower": tasks[objective.lower].astype(float),
            "upper": tasks[objective.upper].astype(float),
            "upper_covered": (
                tasks[objective.error].astype(float)
                <= tasks[objective.upper].astype(float) + NUMERIC_TOLERANCE
            ),
        }
    )


def compute_operating_curves(
    tasks: pd.DataFrame, tau_grid: Sequence[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute task- and target-cluster high-error certificate curves."""
    task_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    studies = list(dict.fromkeys(tasks["study"].astype(str).tolist()))
    for objective in OBJECTIVES:
        values = _objective_values(tasks, objective)
        for study in [*studies, "POOLED_DESCRIPTIVE"]:
            block = values if study == "POOLED_DESCRIPTIVE" else values.loc[values.study.eq(study)]
            # A target is high if any task in it is high.  The max lower bound
            # preserves the same deterministic implication at target level.
            target = (
                block.groupby(["study", "target_cluster"], observed=True)
                .agg(
                    error=("error", "max"),
                    lower=("lower", "max"),
                    upper=("upper", "max"),
                    # Target-simultaneous coverage requires every technical
                    # task in the target cluster to be covered.  Comparing
                    # max(error) with max(upper) would be incorrect.
                    upper_covered=("upper_covered", "all"),
                )
                .reset_index()
            )
            for tau in tau_grid:
                task_rows.append(
                    _curve_row(
                        block,
                        tau=float(tau),
                        objective=objective.name,
                        study=study,
                        unit="task",
                    )
                )
                target_rows.append(
                    _curve_row(
                        target,
                        tau=float(tau),
                        objective=objective.name,
                        study=study,
                        unit="target_cluster",
                    )
                )
    return pd.DataFrame(task_rows), pd.DataFrame(target_rows)


def compute_geometry_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    """Recompute lower tightness and two-sided interval widths by study."""
    rows: list[dict[str, object]] = []
    studies = list(dict.fromkeys(tasks["study"].astype(str).tolist()))
    for objective in OBJECTIVES:
        for study in [*studies, "POOLED_DESCRIPTIVE"]:
            block = tasks if study == "POOLED_DESCRIPTIVE" else tasks.loc[tasks.study.eq(study)]
            error = block[objective.error].to_numpy(float)
            lower = block[objective.lower].to_numpy(float)
            upper = block[objective.upper].to_numpy(float)
            tightness = np.divide(
                lower,
                error,
                out=np.full_like(lower, np.nan),
                where=error > NUMERIC_TOLERANCE,
            )
            width = upper - lower
            rows.append(
                {
                    "study": study,
                    "objective": objective.name,
                    "n_tasks": len(block),
                    "lower_violations": int((lower > error + NUMERIC_TOLERANCE).sum()),
                    "upper_tasks_covered": int(
                        (error <= upper + NUMERIC_TOLERANCE).sum()
                    ),
                    "upper_task_coverage": float(
                        (error <= upper + NUMERIC_TOLERANCE).mean()
                    ),
                    "lower_tightness_mean": float(np.nanmean(tightness)),
                    "lower_tightness_q10": float(np.nanquantile(tightness, 0.10)),
                    "lower_tightness_median": float(np.nanmedian(tightness)),
                    "lower_tightness_q90": float(np.nanquantile(tightness, 0.90)),
                    "interval_width_mean": float(np.mean(width)),
                    "interval_width_median": float(np.median(width)),
                    "interval_width_q90": float(np.quantile(width, 0.90)),
                    "lower_slack_median": float(np.median(error - lower)),
                    "upper_slack_median": float(np.median(upper - error)),
                }
            )
    return pd.DataFrame(rows)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, float_format="%.17g"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_figures(task_curves: pd.DataFrame, geometry: pd.DataFrame, out: Path) -> None:
    """Write small white-background diagnostic figures from recomputed tables."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 8.5,
        }
    )
    colors = {
        "E176_primary_CD4": "#3A6EA5",
        "E177_Sunshine": "#2A8C82",
        "E180_XuCao": "#D97732",
        "E182_GSE225807": "#B84A4A",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharex=True, sharey=True)
    for ax, objective in zip(axes, ("family_rms", "worst_member")):
        data = task_curves.loc[
            task_curves.objective.eq(objective)
            & ~task_curves.study.eq("POOLED_DESCRIPTIVE")
        ]
        for study, block in data.groupby("study", sort=False, observed=True):
            ax.plot(
                block["certified_high_coverage"],
                block["certified_high_recall"],
                marker="o",
                ms=2.8,
                lw=1.2,
                color=colors.get(study, "#4D4D4D"),
                label=study.replace("_", " "),
            )
        ax.set_title(objective.replace("_", " "))
        ax.set_xlabel("Certified-high task coverage")
        ax.grid(color="#E9EEF3", lw=0.7)
    axes[0].set_ylabel("Recall among observed high-error tasks")
    axes[1].legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(out / f"F1_CERTIFIED_HIGH_OPERATING_CURVES.{suffix}", **kwargs)
    plt.close(fig)

    main = geometry.loc[~geometry.study.eq("POOLED_DESCRIPTIVE")]
    x = np.arange(len(main) // 2)
    studies = main.loc[main.objective.eq("family_rms"), "study"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, column, label in (
        (axes[0], "lower_tightness_median", "Median lower / observed error"),
        (axes[1], "interval_width_median", "Median upper - lower"),
    ):
        family = main.loc[main.objective.eq("family_rms"), column].to_numpy(float)
        worst = main.loc[main.objective.eq("worst_member"), column].to_numpy(float)
        ax.bar(x - 0.18, family, width=0.36, color="#3A6EA5", label="family RMS")
        ax.bar(x + 0.18, worst, width=0.36, color="#2A8C82", label="worst member")
        ax.set_xticks(x, [item.replace("_", " ") for item in studies], rotation=22, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#E9EEF3", lw=0.7)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(out / f"F2_CERTIFICATE_GEOMETRY.{suffix}", **kwargs)
    plt.close(fig)


def run(
    repo: Path,
    output_dir: Path,
    tau_grid: Sequence[float] = DEFAULT_TAU,
    *,
    figures: bool = True,
) -> dict:
    tasks, integrity = load_released_tasks(repo)
    task_curves, target_curves = compute_operating_curves(tasks, tau_grid)
    geometry = compute_geometry_summary(tasks)
    if int(task_curves["n_false_certificates"].sum()) != 0:
        raise CertificateIntegrityError("deterministic high-error certificate produced a false flag")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "CERTIFIED_HIGH_TASK_CURVES.csv", task_curves)
    _atomic_csv(output_dir / "CERTIFIED_HIGH_TARGET_CURVES.csv", target_curves)
    _atomic_csv(output_dir / "CERTIFICATE_GEOMETRY_SUMMARY.csv", geometry)
    if figures:
        make_figures(task_curves, geometry, output_dir)

    input_paths = [
        repo / E181_REL / "tables/E181_TASK_CERTIFICATES.csv",
        repo / E182_REL / "tables/E182_EVALUATION_TASKS.csv",
        repo / E182_REL / "E182_FINAL_SUMMARY.json",
        repo / E183_REL / "tables/E183_COMBINED_TASK_CERTIFICATES.csv",
        repo / E183_REL / "tables/E183_TARGET_CERTIFICATES.csv",
        repo / E183_REL / "RUN_STATUS.json",
    ]
    code_path = Path(__file__).resolve()
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    status = {
        "schema": "safeconf_certificate_operating_curves_v1",
        "status": "PASS",
        "analysis_type": "retrospective_recomputation_from_released_e183_table",
        "git_head_at_run": git_head,
        "code_sha256": _sha256(code_path),
        "input_sha256": {
            path.relative_to(repo).as_posix(): _sha256(path) for path in input_paths
        },
        "tau_grid": [float(item) for item in tau_grid],
        "integrity": integrity,
        "outputs": {
            "task_curve_rows": len(task_curves),
            "target_curve_rows": len(target_curves),
            "geometry_rows": len(geometry),
        },
        "provenance": {
            "read": [
                str(E181_REL / "tables/E181_TASK_CERTIFICATES.csv"),
                str(E182_REL / "tables/E182_EVALUATION_TASKS.csv"),
                str(E182_REL / "E182_FINAL_SUMMARY.json"),
                str(E183_REL / "tables/E183_COMBINED_TASK_CERTIFICATES.csv"),
                str(E183_REL / "tables/E183_TARGET_CERTIFICATES.csv"),
                str(E183_REL / "RUN_STATUS.json"),
            ],
            "raw_expression_matrices_read": 0,
            "truth_array_files_read": 0,
            "e205_truth_read": 0,
            "e208_truth_read": 0,
        },
        "interpretation_limits": [
            "E181 and E183 are retrospective; E183 was produced after E182 truth was opened.",
            "E182 remains a preregistered FAIL (16/20 target clusters); this audit does not reclassify it.",
            "L > tau is a deterministic high-error certificate; L <= tau is UNKNOWN, not safe.",
            "Upper-bound coverage is marginal/conditional on the conformal contract, not a pointwise guarantee after arbitrary selection.",
            "The default tau grid is a retrospective diagnostic grid and must be frozen or externally justified before a new prospective study.",
            "Pooled absolute-tau rows are descriptive because error normalization and biology differ by study.",
        ],
    }
    _atomic_text(
        output_dir / "STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    readme = """# SafeConf-Cert threshold operating curves

This directory is rebuilt only from the released E181/E182/E183 tables.

- `CERTIFIED_HIGH_TASK_CURVES.csv`: for each absolute error tolerance `tau`,
  the fraction certified high and recall among observed high-error tasks.
- `CERTIFIED_HIGH_TARGET_CURVES.csv`: the same calculation after grouping all
  technical tasks for one target; maxima preserve the lower-bound implication.
- `CERTIFICATE_GEOMETRY_SUMMARY.csv`: recomputed lower-bound tightness,
  interval width, and empirical upper coverage.
- `STATUS.json`: input lineage, numerical gates, and interpretation limits.

`certified_high_coverage` is the fraction of all units with `lower > tau`; it
is an issuance rate, not conformal coverage.  `lower <= tau` means `UNKNOWN`,
not safe.  Pooled rows and the default tau grid are retrospective diagnostics.
"""
    _atomic_text(output_dir / "README.md", readme)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=default_repo())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime/certificate_operating_curves"),
    )
    parser.add_argument(
        "--tau-grid",
        help="comma-separated absolute RMSE tolerances; default is a fixed diagnostic grid",
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = repo / output
    tau_grid = parse_tau_grid(args.tau_grid)
    status = run(repo, output, tau_grid, figures=not args.no_figures)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
