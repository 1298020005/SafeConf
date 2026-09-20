#!/usr/bin/env python3
"""Post-release E216 diagnostic on the fixed E211 tolerance grid.

This script never changes the registered E216 ranking result.  It recomputes
the Hilbert identity from the sealed float32 member tensors in long-double
arithmetic and evaluates the absolute tolerance grid published by E211 five
days before E216 truth release.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


FIXED_E211_TAU = (
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


class DiagnosticFailure(RuntimeError):
    """The sealed tensor or released task table changed."""


def recompute_identity(
    members: np.ndarray, weights: np.ndarray, truth: np.ndarray
) -> dict[str, np.ndarray | float]:
    values = np.asarray(members, dtype=np.longdouble)
    weight = np.asarray(weights, dtype=np.longdouble)
    target = np.asarray(truth, dtype=np.longdouble)
    if values.ndim != 3 or target.shape != values.shape[1:]:
        raise DiagnosticFailure("member and truth tensors are not aligned")
    if weight.shape != (values.shape[0],) or not np.isclose(float(weight.sum()), 1.0):
        raise DiagnosticFailure("registered member weights changed")
    centroid = np.einsum("m,mtg->tg", weight, values)
    lower_squared = np.einsum(
        "m,mtg->t", weight, np.square(values - centroid[None, :, :])
    ) / values.shape[2]
    rms_squared = np.einsum(
        "m,mtg->t", weight, np.square(values - target[None, :, :])
    ) / values.shape[2]
    centroid_squared = np.mean(
        np.square(centroid - target), axis=1, dtype=np.longdouble
    )
    residual = rms_squared - centroid_squared - lower_squared
    return {
        "lower_bound": np.sqrt(lower_squared),
        "family_rms_error": np.sqrt(rms_squared),
        "max_abs_identity_residual": float(np.max(np.abs(residual))),
    }


def certificate_curve(lower: np.ndarray, error: np.ndarray) -> pd.DataFrame:
    lower = np.asarray(lower, dtype=float)
    error = np.asarray(error, dtype=float)
    rows = []
    for tau in FIXED_E211_TAU:
        certified = lower > tau
        true_high = error > tau
        certified_true = certified & true_high
        rows.append(
            {
                "tau": tau,
                "n_tasks": len(lower),
                "n_true_high": int(true_high.sum()),
                "n_certified_high": int(certified.sum()),
                "n_certified_true_high": int(certified_true.sum()),
                "n_false_certificates": int((certified & ~true_high).sum()),
                "certified_high_coverage": float(certified.mean()),
                "certified_high_recall": (
                    float(certified_true.sum() / true_high.sum())
                    if true_high.any()
                    else np.nan
                ),
                "certified_high_precision": (
                    float(certified_true.sum() / certified.sum())
                    if certified.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--task-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise DiagnosticFailure(f"refusing to overwrite: {args.output_dir}")

    family = np.load(args.family.resolve(), allow_pickle=False)
    truth = np.load(args.truth.resolve(), allow_pickle=False)
    released = pd.read_csv(args.task_results.resolve())
    identity = recompute_identity(
        family["member_predictions"],
        family["member_weight"],
        truth["truth_centroids"],
    )
    lower = np.asarray(identity["lower_bound"], dtype=float)
    error = np.asarray(identity["family_rms_error"], dtype=float)
    if len(released) != 224 or not np.allclose(
        lower,
        released.architecture_balanced_lower_bound.to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise DiagnosticFailure("released lower bound differs from sealed tensor")
    if not np.allclose(
        error,
        released.registered_family_rms_error.to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise DiagnosticFailure("released family error differs from sealed tensor")

    curve = certificate_curve(lower, error)
    status = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "POST_RELEASE_FIXED_E211_TAU_NUMERICAL_DIAGNOSTIC",
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "e216_formal_primary_gate_changed": False,
        "tau_grid_source": "E211 fixed absolute grid committed 2026-09-15",
        "registered_tasks": 224,
        "max_abs_longdouble_identity_residual": identity[
            "max_abs_identity_residual"
        ],
        "lower_bound_violations": int((lower > error).sum()),
        "median_lower_bound_tightness": float(np.median(lower / error)),
        "all_fixed_tau_false_certificates": int(curve.n_false_certificates.sum()),
    }
    if (
        status["max_abs_longdouble_identity_residual"] > 1e-15
        or status["lower_bound_violations"] != 0
        or status["all_fixed_tau_false_certificates"] != 0
    ):
        raise DiagnosticFailure("fixed-grid numerical certificate diagnostic failed")
    args.output_dir.mkdir(parents=True)
    curve.to_csv(args.output_dir / "E216_E211_FIXED_TAU_CURVES.csv", index=False)
    (args.output_dir / "E216_FIXED_TAU_DIAGNOSTIC_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
