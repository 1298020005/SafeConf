#!/usr/bin/env python3
"""Audit high-error prediction triage on Tahoe SafeConf records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.cli.run_tahoe_pseudobulk_smoke import robust_z

TOP_FRACTIONS = (0.05, 0.10, 0.20)
PRIMARY_SCORE = "safeconf_full"
MAGNITUDE_SCORE = "predicted_magnitude"


def _deterministic_random(record_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}::{record_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def _predicted_magnitudes(records: pd.DataFrame, npz_path: Path) -> pd.Series:
    arrays = np.load(npz_path)
    values: dict[str, float] = {}
    for key in records["predicted_effect_key"].astype(str).drop_duplicates():
        array = np.asarray(arrays[key], dtype=float)
        values[key] = float(np.linalg.norm(array) / math.sqrt(len(array)))
    return records["predicted_effect_key"].astype(str).map(values)


def build_score_table(run_dir: Path, seed: int = 5201) -> pd.DataFrame:
    tables = run_dir / "tables"
    records = pd.read_csv(tables / "TAHOE_PREDICTION_RECORDS_SMOKE.csv")
    scores = pd.read_csv(tables / "TAHOE_PROTOCOL_SCORES_SMOKE.csv")
    base = records.merge(
        scores[["record_id", "score_value", "risk_axis_value"]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    base["predicted_magnitude"] = _predicted_magnitudes(
        base[base["split"].astype(str).eq("test")],
        run_dir / "arrays" / "TAHOE_PREDICTED_EFFECTS_SMOKE.npz",
    )

    frames: list[pd.DataFrame] = []
    for (fold, predictor), group in base.groupby(["fold_id", "predictor_name"], dropna=False):
        reference = group[group["split"].astype(str).isin(["train", "val"])]
        scored = group[group["split"].astype(str).eq("test")].copy()
        if scored.empty:
            continue
        support = np.log1p(pd.to_numeric(scored["perturbation_support_count"], errors="coerce"))
        support_ref = np.log1p(pd.to_numeric(reference["perturbation_support_count"], errors="coerce"))
        disagreement = pd.to_numeric(scored["model_disagreement_rmse"], errors="coerce")
        disagreement_ref = pd.to_numeric(reference["model_disagreement_rmse"], errors="coerce")
        scored["support_only"] = -robust_z(support, support_ref)
        scored["disagreement_only"] = robust_z(disagreement, disagreement_ref)
        frames.append(scored)
    test = pd.concat(frames, ignore_index=True)
    test["safeconf_full"] = pd.to_numeric(test["risk_axis_value"], errors="coerce")
    test["random"] = test["record_id"].astype(str).map(lambda value: _deterministic_random(value, seed))
    test["oracle_error"] = pd.to_numeric(test["true_error_rmse"], errors="coerce")
    return test


def top_fraction_enrichment(errors: np.ndarray, risk: np.ndarray, fraction: float) -> dict:
    mask = np.isfinite(errors) & np.isfinite(risk)
    errors = np.asarray(errors[mask], dtype=float)
    risk = np.asarray(risk[mask], dtype=float)
    n = len(errors)
    if n < 10:
        return {"n": n, "precision": np.nan, "random_expected": np.nan, "enrichment": np.nan}
    n_top = max(1, int(math.ceil(fraction * n)))
    high_error = np.zeros(n, dtype=bool)
    high_error[np.argsort(-errors, kind="stable")[:n_top]] = True
    selected = np.argsort(-risk, kind="stable")[:n_top]
    precision = float(high_error[selected].mean())
    random_expected = float(high_error.mean())
    return {
        "n": n,
        "precision": precision,
        "random_expected": random_expected,
        "enrichment": precision / random_expected if random_expected > 0 else np.nan,
    }


def point_summary(score_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_specs = [("overall", "overall", score_table)]
    group_specs.extend(
        ("predictor", str(predictor), group)
        for predictor, group in score_table.groupby("predictor_name", dropna=False)
    )
    score_names = [
        "safeconf_full",
        "predicted_magnitude",
        "support_only",
        "disagreement_only",
        "random",
        "oracle_error",
    ]
    for level, group_id, group in group_specs:
        errors = pd.to_numeric(group["true_error_rmse"], errors="coerce").to_numpy(dtype=float)
        for score_name in score_names:
            risk = pd.to_numeric(group[score_name], errors="coerce").to_numpy(dtype=float)
            aligned = pd.Series(risk).corr(pd.Series(errors), method="spearman")
            for fraction in TOP_FRACTIONS:
                metric = top_fraction_enrichment(errors, risk, fraction)
                rows.append(
                    {
                        "level": level,
                        "group_id": group_id,
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
    groups = [group.index.to_numpy(dtype=int) for _, group in score_table.groupby("task_key", sort=False)]
    if len(groups) < 10:
        return pd.DataFrame(), pd.DataFrame()
    errors_all = pd.to_numeric(score_table["true_error_rmse"], errors="coerce").to_numpy(dtype=float)
    full_all = pd.to_numeric(score_table[PRIMARY_SCORE], errors="coerce").to_numpy(dtype=float)
    magnitude_all = pd.to_numeric(score_table[MAGNITUDE_SCORE], errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws: list[dict] = []
    for index in range(n_bootstrap):
        picks = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[int(pick)] for pick in picks])
        full = top_fraction_enrichment(errors_all[rows], full_all[rows], 0.10)["enrichment"]
        magnitude = top_fraction_enrichment(errors_all[rows], magnitude_all[rows], 0.10)["enrichment"]
        draws.append(
            {
                "bootstrap_index": index,
                "safeconf_top10_enrichment": full,
                "magnitude_top10_enrichment": magnitude,
                "safeconf_minus_magnitude": full - magnitude,
            }
        )
    draw_df = pd.DataFrame(draws)
    summary: dict[str, float | int] = {
        "n_bootstrap": n_bootstrap,
        "n_task_clusters": len(groups),
    }
    for column in ["safeconf_top10_enrichment", "magnitude_top10_enrichment", "safeconf_minus_magnitude"]:
        values = pd.to_numeric(draw_df[column], errors="coerce").dropna()
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_ci_low"] = float(values.quantile(0.025))
        summary[f"{column}_ci_high"] = float(values.quantile(0.975))
    return pd.DataFrame([summary]), draw_df


def run(run_dir: Path, out_dir: Path, n_bootstrap: int = 1000, seed: int = 5201) -> dict:
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    scores = build_score_table(run_dir, seed=seed).reset_index(drop=True)
    point = point_summary(scores)
    bootstrap, draws = bootstrap_top10(scores, n_bootstrap=n_bootstrap, seed=seed)
    scores.to_csv(tables / "TAHOE_D3_SCORE_AUDIT.csv", index=False)
    point.to_csv(tables / "TAHOE_D3_TRIAGE_POINT_SUMMARY.csv", index=False)
    bootstrap.to_csv(tables / "TAHOE_D3_TOP10_TASK_CLUSTER_CI.csv", index=False)
    draws.to_csv(tables / "TAHOE_D3_TOP10_BOOTSTRAP_DRAWS.csv", index=False)

    overall = point[
        point["level"].astype(str).eq("overall")
        & np.isclose(point["top_fraction"].astype(float), 0.10)
    ]
    lookup = overall.set_index("score_name")["enrichment"].to_dict()
    ci = bootstrap.iloc[0]
    full_ci_low = float(ci["safeconf_top10_enrichment_ci_low"])
    difference_low = float(ci["safeconf_minus_magnitude_ci_low"])
    difference_high = float(ci["safeconf_minus_magnitude_ci_high"])
    if full_ci_low <= 1:
        gate = "FAIL"
    elif difference_low > 0:
        gate = "PASS_SAFECONF_STRONGER"
    elif difference_high < 0:
        gate = "PASS_MAGNITUDE_STRONGER"
    else:
        gate = "PASS_COMPARABLE"
    status = {
        "status": "ok",
        "gate": gate,
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_records": int(len(scores)),
        "n_task_clusters": int(scores["task_key"].nunique()),
        "n_bootstrap": int(n_bootstrap),
        "safeconf_top10_enrichment": float(lookup.get(PRIMARY_SCORE, np.nan)),
        "magnitude_top10_enrichment": float(lookup.get(MAGNITUDE_SCORE, np.nan)),
        "random_top10_enrichment": float(lookup.get("random", np.nan)),
        "safeconf_top10_ci_low": full_ci_low,
        "safeconf_minus_magnitude_ci_low": difference_low,
        "safeconf_minus_magnitude_ci_high": difference_high,
    }
    (out_dir / "TAHOE_D3_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report = f"""# Tahoe D3 prediction-triage audit

- Gate: `{gate}`
- SafeConf top-10 enrichment: {status['safeconf_top10_enrichment']:.3f}
- Predicted-magnitude top-10 enrichment: {status['magnitude_top10_enrichment']:.3f}
- Random top-10 enrichment: {status['random_top10_enrichment']:.3f}
- SafeConf top-10 task-cluster CI lower: {full_ci_low:.3f}
- SafeConf minus magnitude CI: [{difference_low:.3f}, {difference_high:.3f}]
- Test task clusters: {status['n_task_clusters']}
"""
    (reports / "TAHOE_D3_TRIAGE_REPORT.md").write_text(report, encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Tahoe high-error prediction triage.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5201)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.run_dir, args.out_dir, n_bootstrap=args.bootstrap, seed=args.seed),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
