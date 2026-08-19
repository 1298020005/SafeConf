#!/usr/bin/env python3
"""Evaluate fixed SafeConf and magnitude rank blends on Tahoe triage records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.cli.run_tahoe_triage_audit import (
    TOP_FRACTIONS,
    top_fraction_enrichment,
)

PRIMARY_SCORE = "combined_equal"
SCORE_NAMES = (
    "safeconf_full",
    "predicted_magnitude",
    PRIMARY_SCORE,
    "combined_magnitude75",
    "combined_safeconf75",
)


def add_fixed_rank_blends(score_table: pd.DataFrame) -> pd.DataFrame:
    """Add fixed prediction-side rank blends without consulting true errors."""
    required = {"fold_id", "predictor_name", "safeconf_full", "predicted_magnitude"}
    missing = sorted(required.difference(score_table.columns))
    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    out = score_table.copy().reset_index(drop=True)
    group_cols = ["fold_id", "predictor_name"]
    for source, target in (
        ("safeconf_full", "safeconf_rank"),
        ("predicted_magnitude", "magnitude_rank"),
    ):
        numeric = pd.to_numeric(out[source], errors="coerce")
        out[target] = numeric.groupby(
            [out[column] for column in group_cols], dropna=False
        ).rank(method="average", pct=True)

    safeconf = out["safeconf_rank"]
    magnitude = out["magnitude_rank"]
    out[PRIMARY_SCORE] = 0.50 * safeconf + 0.50 * magnitude
    out["combined_magnitude75"] = 0.25 * safeconf + 0.75 * magnitude
    out["combined_safeconf75"] = 0.75 * safeconf + 0.25 * magnitude
    return out


def point_summary(score_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    errors = pd.to_numeric(score_table["true_error_rmse"], errors="coerce")
    for score_name in SCORE_NAMES:
        risk = pd.to_numeric(score_table[score_name], errors="coerce")
        aligned = risk.corr(errors, method="spearman")
        for fraction in TOP_FRACTIONS:
            metric = top_fraction_enrichment(
                errors.to_numpy(dtype=float), risk.to_numpy(dtype=float), fraction
            )
            rows.append(
                {
                    "score_name": score_name,
                    "top_fraction": fraction,
                    "n_records": metric["n"],
                    "aligned_rho": aligned,
                    "precision": metric["precision"],
                    "random_expected_precision": metric["random_expected"],
                    "enrichment": metric["enrichment"],
                }
            )
    return pd.DataFrame(rows)


def bootstrap_top10(
    score_table: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 5201,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [
        group.index.to_numpy(dtype=int)
        for _, group in score_table.groupby("task_key", sort=False)
    ]
    if len(groups) < 10:
        return pd.DataFrame(), pd.DataFrame()

    errors = pd.to_numeric(score_table["true_error_rmse"], errors="coerce").to_numpy(dtype=float)
    score_arrays = {
        name: pd.to_numeric(score_table[name], errors="coerce").to_numpy(dtype=float)
        for name in SCORE_NAMES
    }
    rng = np.random.default_rng(seed)
    draws: list[dict] = []
    for index in range(n_bootstrap):
        picks = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[int(pick)] for pick in picks])
        enrichments = {
            name: top_fraction_enrichment(errors[rows], values[rows], 0.10)["enrichment"]
            for name, values in score_arrays.items()
        }
        draws.append(
            {
                "bootstrap_index": index,
                **{f"{name}_enrichment": value for name, value in enrichments.items()},
                "combined_equal_minus_magnitude": (
                    enrichments[PRIMARY_SCORE] - enrichments["predicted_magnitude"]
                ),
                "combined_equal_minus_safeconf": (
                    enrichments[PRIMARY_SCORE] - enrichments["safeconf_full"]
                ),
            }
        )

    draw_df = pd.DataFrame(draws)
    summary: dict[str, float | int] = {
        "n_bootstrap": n_bootstrap,
        "n_task_clusters": len(groups),
    }
    for column in draw_df.columns:
        if column == "bootstrap_index":
            continue
        values = pd.to_numeric(draw_df[column], errors="coerce").dropna()
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_ci_low"] = float(values.quantile(0.025))
        summary[f"{column}_ci_high"] = float(values.quantile(0.975))
    return pd.DataFrame([summary]), draw_df


def decide_gate(bootstrap_summary: pd.DataFrame) -> str:
    row = bootstrap_summary.iloc[0]
    combined_low = float(row["combined_equal_enrichment_ci_low"])
    difference_low = float(row["combined_equal_minus_magnitude_ci_low"])
    if combined_low <= 1:
        return "FAIL"
    if difference_low > 0:
        return "PASS_ADDS_VALUE"
    return "PASS_USEFUL_NOT_BETTER"


def run(score_csv: Path, out_dir: Path, n_bootstrap: int = 1000, seed: int = 5201) -> dict:
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    scores = add_fixed_rank_blends(pd.read_csv(score_csv))
    point = point_summary(scores)
    bootstrap, draws = bootstrap_top10(scores, n_bootstrap=n_bootstrap, seed=seed)
    gate = decide_gate(bootstrap)

    scores.to_csv(tables / "TAHOE_D5_SCORE_AUDIT.csv", index=False)
    point.to_csv(tables / "TAHOE_D5_POINT_SUMMARY.csv", index=False)
    bootstrap.to_csv(tables / "TAHOE_D5_TOP10_TASK_CLUSTER_CI.csv", index=False)
    draws.to_csv(tables / "TAHOE_D5_BOOTSTRAP_DRAWS.csv", index=False)

    top10 = point[np.isclose(point["top_fraction"].astype(float), 0.10)]
    enrichment = top10.set_index("score_name")["enrichment"].to_dict()
    ci = bootstrap.iloc[0]
    status = {
        "status": "ok",
        "gate": gate,
        "score_csv": str(score_csv),
        "out_dir": str(out_dir),
        "n_records": int(len(scores)),
        "n_task_clusters": int(scores["task_key"].nunique()),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "primary_score": PRIMARY_SCORE,
        "safeconf_top10_enrichment": float(enrichment["safeconf_full"]),
        "magnitude_top10_enrichment": float(enrichment["predicted_magnitude"]),
        "combined_equal_top10_enrichment": float(enrichment[PRIMARY_SCORE]),
        "combined_equal_top10_ci_low": float(ci["combined_equal_enrichment_ci_low"]),
        "combined_equal_minus_magnitude_ci_low": float(
            ci["combined_equal_minus_magnitude_ci_low"]
        ),
        "combined_equal_minus_magnitude_ci_high": float(
            ci["combined_equal_minus_magnitude_ci_high"]
        ),
    }
    (out_dir / "TAHOE_D5_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Tahoe D5 fixed-combination triage

- Gate: `{gate}`
- SafeConf top-10 enrichment: {status['safeconf_top10_enrichment']:.3f}
- Magnitude top-10 enrichment: {status['magnitude_top10_enrichment']:.3f}
- Equal-rank combination top-10 enrichment: {status['combined_equal_top10_enrichment']:.3f}
- Equal-rank combination CI lower: {status['combined_equal_top10_ci_low']:.3f}
- Combination minus magnitude CI: [{status['combined_equal_minus_magnitude_ci_low']:.3f}, {status['combined_equal_minus_magnitude_ci_high']:.3f}]
- Test task clusters: {status['n_task_clusters']}
"""
    (reports / "TAHOE_D5_REPORT.md").write_text(report, encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed Tahoe triage score combinations.")
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5201)
    args = parser.parse_args()
    print(json.dumps(run(args.score_csv, args.out_dir, args.bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()
