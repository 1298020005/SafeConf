from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.data.records import (
    add_legacy_contract_defaults,
    assert_no_feature_label_leakage,
    load_merged_records,
    validate_prediction_record_contract,
)
from safetrans_confidence.eval.signal_validity_audit import (
    _assign_normalized_rmse,
    _eval_group,
    _load_true_effect_magnitudes,
    _risk_axis,
    load_run,
)
from safetrans_confidence.features.schema import available_feature_columns
from safetrans_confidence.scoring.error_ranker import (
    LINEAR_ERROR_RANKER_FEATURES,
    _make_model,
    _prepare_xy,
)
from safetrans_confidence.scoring.protocol_v0_2 import build_protocol_v0_2_scores


DEFAULT_RUN_DIRS = [
    Path("outputs/confidence_task_mvp_v2_1"),
    Path("outputs/confidence_task_Frangieh_blind"),
    Path("outputs/confidence_task_kcp_blind_probe"),
    Path("outputs/confidence_task_crossPatient_blind"),
]

TRUE_EFFECT_FILES = ["input/true_effects.npz", "arrays/true_effects.npz"]
LODO_EXISTING_SCORE_NAMES = {
    "model_disagreement_risk",
}
LODO_FOCUS_SCORE_NAMES = {
    "protocol_v0_2_family_confidence",
    "lodo_error_ranker_risk",
    "safeconf_linear_ranker_risk",
    "model_disagreement_risk",
}


def _find_true_effect_npz(run_dir: Path) -> Path | None:
    return next((run_dir / rel for rel in TRUE_EFFECT_FILES if (run_dir / rel).exists()), None)


def _load_group_map(group_table: Path | None) -> dict[str, str]:
    if group_table is None or not group_table.exists():
        return {}
    groups = pd.read_csv(group_table)
    if not {"dataset_name", "dataset_group"}.issubset(groups.columns):
        return {}
    return dict(zip(groups["dataset_name"].astype(str), groups["dataset_group"].astype(str)))


def load_base_tables(run_dirs: list[Path], group_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bases = []
    records_for_contract = []
    for run_dir in run_dirs:
        base = load_merged_records(run_dir)
        base["source_run_dir"] = str(run_dir)
        base = add_legacy_contract_defaults(base)
        if group_map:
            base["dataset_group"] = base["dataset_name"].astype(str).map(group_map).fillna(
                base["dataset_group"]
            )
        bases.append(base)

        record_path = run_dir / "tables" / "PREDICTION_RECORDS.csv"
        rec = pd.read_csv(record_path)
        rec["source_run_dir"] = str(run_dir)
        rec = add_legacy_contract_defaults(rec)
        if group_map:
            rec["dataset_group"] = rec["dataset_name"].astype(str).map(group_map).fillna(
                rec["dataset_group"]
            )
        records_for_contract.append(rec)

    merged_base = pd.concat(bases, ignore_index=True) if bases else pd.DataFrame()
    contract_records = (
        pd.concat(records_for_contract, ignore_index=True) if records_for_contract else pd.DataFrame()
    )
    return merged_base, contract_records


def _extract_linear_coefficients(
    model,
    feature_cols: list[str],
    meta: dict,
) -> list[dict]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return []
    alpha = getattr(estimator, "alpha_", getattr(estimator, "alpha", np.nan))
    l1_ratio = getattr(estimator, "l1_ratio_", getattr(estimator, "l1_ratio", np.nan))
    intercept = getattr(estimator, "intercept_", np.nan)
    rows = []
    for feature_name, value in zip(feature_cols, np.asarray(coef, dtype=float).ravel()):
        rows.append(
            {
                **meta,
                "feature_name": feature_name,
                "coefficient": float(value),
                "abs_coefficient": float(abs(value)),
                "intercept": float(intercept) if np.isfinite(intercept) else np.nan,
                "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
                "l1_ratio": float(l1_ratio) if np.isfinite(l1_ratio) else np.nan,
            }
        )
    return rows


def build_lodo_error_ranker_scores(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train one shallow error-ranker per held-out dataset group and predictor.

    The held-out dataset group is never used for fitting.  This is the key
    difference from ordinary fold-local learned scores.
    """
    full_feature_cols = available_feature_columns(base)
    linear_feature_cols = [c for c in LINEAR_ERROR_RANKER_FEATURES if c in base.columns]
    assert_no_feature_label_leakage(full_feature_cols)
    assert_no_feature_label_leakage(linear_feature_cols)
    rows: list[dict] = []
    status_rows: list[dict] = []
    coef_rows: list[dict] = []
    if not full_feature_cols:
        return (
            pd.DataFrame(),
            pd.DataFrame([{"status": "skipped_no_feature_columns", "n_features": 0}]),
            pd.DataFrame(),
        )

    model_specs = [
        {
            "score_name": "lodo_error_ranker_risk",
            "model_type": "histgbt",
            "feature_cols": full_feature_cols,
            "feature_set": "all_available_features",
        },
        {
            "score_name": "safeconf_linear_ranker_risk",
            "model_type": "elasticnet",
            "feature_cols": linear_feature_cols,
            "feature_set": "deduplicated_linear_features",
        },
    ]

    for heldout_group in sorted(base["dataset_group"].dropna().astype(str).unique()):
        heldout = base[(base["dataset_group"].astype(str) == heldout_group) & base["split"].eq("test")]
        train_pool = base[
            (base["dataset_group"].astype(str) != heldout_group)
            & base["split"].isin(["train", "val"])
        ].copy()
        for spec in model_specs:
            feature_cols = spec["feature_cols"]
            for predictor, target in heldout.groupby("predictor_name", dropna=False):
                train = train_pool[train_pool["predictor_name"].astype(str) == str(predictor)].copy()
                status = {
                    "heldout_dataset_group": heldout_group,
                    "predictor_name": predictor,
                    "n_train_pool": int(len(train)),
                    "n_test": int(len(target)),
                    "n_features": int(len(feature_cols)),
                    "feature_columns": ",".join(feature_cols),
                    "feature_set": spec["feature_set"],
                    "model_type": spec["model_type"],
                    "score_name": spec["score_name"],
                }
                y = pd.to_numeric(train.get("true_error_rmse", pd.Series(dtype=float)), errors="coerce")
                valid = y.notna()
                train = train.loc[valid].copy()
                y = y.loc[valid]
                if len(feature_cols) == 0 or len(target) == 0 or len(train) < 12 or float(y.std(ddof=0)) <= 1e-12:
                    status["status"] = "skipped_too_few_or_degenerate"
                    status_rows.append(status)
                    continue
                train_x, target_x = _prepare_xy(train, target, feature_cols)
                model = _make_model(5201, model_type=spec["model_type"], n_train_rows=len(train))
                model.fit(train_x, y)
                pred = model.predict(target_x)
                for (_, row), value in zip(target.iterrows(), pred):
                    rows.append(
                        {
                            "record_id": row["record_id"],
                            "dataset_name": row["dataset_name"],
                            "dataset_family": row.get("dataset_family", ""),
                            "dataset_group": row["dataset_group"],
                            "heldout_dataset_group": heldout_group,
                            "fold_id": int(row["fold_id"]),
                            "split": row["split"],
                            "context": row["context"],
                            "perturbation": row["perturbation"],
                            "predictor_name": row["predictor_name"],
                            "score_name": spec["score_name"],
                            "score_type": "risk",
                            "score_value": float(value),
                            "true_error_rmse": float(row["true_error_rmse"]),
                        }
                    )
                coef_rows.extend(
                    _extract_linear_coefficients(
                        model,
                        feature_cols,
                        {
                            "heldout_dataset_group": heldout_group,
                            "predictor_name": predictor,
                            "score_name": spec["score_name"],
                            "model_type": spec["model_type"],
                            "feature_set": spec["feature_set"],
                            "n_train_pool": int(len(train)),
                            "n_test": int(len(target)),
                        },
                    )
                )
                status["status"] = "ok"
                status_rows.append(status)
    return pd.DataFrame(rows), pd.DataFrame(status_rows), pd.DataFrame(coef_rows)


def _load_existing_scores_with_magnitude(run_dirs: list[Path], group_map: dict[str, str]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        scores, status = load_run(run_dir)
        if status.get("status") != "ok" or scores.empty:
            continue
        scores = scores.copy()
        scores = scores[scores["score_name"].astype(str).isin(LODO_EXISTING_SCORE_NAMES)].copy()
        if scores.empty:
            continue
        if group_map:
            scores["dataset_group"] = scores["dataset_name"].astype(str).map(group_map)
        if "dataset_group" not in scores.columns or scores["dataset_group"].isna().any():
            from safetrans_confidence.data.records import _default_dataset_group

            scores["dataset_group"] = scores.get("dataset_group", pd.Series(index=scores.index)).fillna(
                scores["dataset_name"].map(_default_dataset_group)
            )
        frames.append(scores)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _records_with_magnitude(base: pd.DataFrame, run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        sub = base[base["source_run_dir"].astype(str) == str(run_dir)].copy()
        if sub.empty:
            continue
        npz = _find_true_effect_npz(run_dir)
        if npz is None:
            continue
        mags = _load_true_effect_magnitudes(sub.drop_duplicates("record_id"), npz)
        rows.append(sub.merge(mags, on="record_id", how="left"))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        if "run_dir" not in out.columns and "source_run_dir" in out.columns:
            out["run_dir"] = out["source_run_dir"]
        out = _assign_normalized_rmse(out)
    return out


def _score_metadata_with_magnitude(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "dataset_name",
        "record_id",
        "task_id",
        "task_key",
        "dataset_group",
        "true_effect_key",
        "true_effect_l2_norm",
        "true_effect_abs_mean",
        "true_effect_std",
        "true_effect_n_genes",
        "effect_scale_rmse",
        "normalization_eps",
        "normalized_rmse",
    ]
    available = [c for c in keep_cols if c in meta.columns]
    merge_keys = ["dataset_name", "record_id"] if "dataset_name" in scores.columns else ["record_id"]
    meta_unique = meta[available].drop_duplicates(merge_keys)
    merged = scores.merge(meta_unique, on=merge_keys, how="left")
    if "dataset_group_x" in merged.columns or "dataset_group_y" in merged.columns:
        left = merged["dataset_group_x"] if "dataset_group_x" in merged.columns else pd.Series(index=merged.index, dtype=object)
        right = merged["dataset_group_y"] if "dataset_group_y" in merged.columns else pd.Series(index=merged.index, dtype=object)
        merged["dataset_group"] = left.fillna(right)
        merged = merged.drop(
            columns=[c for c in ["dataset_group_x", "dataset_group_y"] if c in merged.columns]
        )
    merged["risk_axis"] = _risk_axis(merged)
    return merged


def _eval_scores(scores: pd.DataFrame, level_cols: list[str]) -> pd.DataFrame:
    test = scores[scores["split"].eq("test")].dropna(
        subset=["score_value", "true_error_rmse", "true_effect_l2_norm"]
    )
    rows = []
    for key, group in test.groupby(level_cols, dropna=False):
        meta = dict(zip(level_cols, key if isinstance(key, tuple) else (key,)))
        row = {**meta, **_eval_group(group)}
        row["score_type"] = str(group["score_type"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _risk_coverage(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = scores[scores["split"].eq("test")].dropna(subset=["score_value", "true_error_rmse"])
    group_cols = ["dataset_group", "dataset_name", "predictor_name", "score_name"]
    for key, group in test.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        score_type = str(group["score_type"].iloc[0])
        ordered = group.sort_values("score_value", ascending=(score_type == "risk"))
        for coverage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            n_keep = max(1, int(np.ceil(len(ordered) * coverage)))
            kept = ordered.head(n_keep)
            rows.append(
                {
                    **meta,
                    "score_type": score_type,
                    "coverage": float(coverage),
                    "n_total": int(len(ordered)),
                    "n_kept": int(len(kept)),
                    "mean_rmse": float(kept["true_error_rmse"].mean()),
                    "median_rmse": float(kept["true_error_rmse"].median()),
                }
            )
    return pd.DataFrame(rows)


def _decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    focus = summary[
        summary["score_name"].isin(
            LODO_FOCUS_SCORE_NAMES
        )
    ].copy()
    if focus.empty:
        return focus
    cols = [
        "dataset_group",
        "dataset_name",
        "predictor_name",
        "score_name",
        "n",
        "raw_spearman",
        "normalized_rmse_spearman",
        "partial_rho_control_magnitude",
        "within_perturbation_weighted_rho",
        "within_context_weighted_rho",
    ]
    return focus[[c for c in cols if c in focus.columns]].sort_values(
        ["dataset_group", "dataset_name", "predictor_name", "score_name"]
    )


def _bootstrap_ci_table(
    scores: pd.DataFrame,
    level_cols: list[str],
    n_bootstrap: int = 200,
    seed: int = 5201,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    test = scores[scores["split"].eq("test")].dropna(
        subset=["score_value", "true_error_rmse", "true_effect_l2_norm"]
    )
    test = test[test["score_name"].astype(str).isin(LODO_FOCUS_SCORE_NAMES)].copy()
    rows = []
    for key, group in test.groupby(level_cols, dropna=False):
        meta = dict(zip(level_cols, key if isinstance(key, tuple) else (key,)))
        if len(group) < 8:
            rows.append({**meta, "n": int(len(group)), "status": "skipped_too_few_rows"})
            continue
        raw_values = []
        partial_values = []
        idx = group.index.to_numpy()
        for _ in range(n_bootstrap):
            sample_idx = rng.choice(idx, size=len(idx), replace=True)
            sample = group.loc[sample_idx].copy().reset_index(drop=True)
            metrics = _eval_group(sample)
            raw = metrics.get("raw_spearman", np.nan)
            partial = metrics.get("partial_rho_control_magnitude", np.nan)
            if np.isfinite(raw):
                raw_values.append(float(raw))
            if np.isfinite(partial):
                partial_values.append(float(partial))
        row = {**meta, "n": int(len(group)), "n_bootstrap": int(n_bootstrap), "status": "ok"}
        if len(raw_values) >= 20:
            row["raw_spearman_ci_low"], row["raw_spearman_ci_high"] = np.quantile(
                raw_values, [0.025, 0.975]
            )
        else:
            row["raw_spearman_ci_low"], row["raw_spearman_ci_high"] = np.nan, np.nan
        if len(partial_values) >= 20:
            row["partial_rho_ci_low"], row["partial_rho_ci_high"] = np.quantile(
                partial_values, [0.025, 0.975]
            )
        else:
            row["partial_rho_ci_low"], row["partial_rho_ci_high"] = np.nan, np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_lodo(
    run_dirs: list[Path],
    out_dir: Path,
    group_table: Path | None = None,
    gears_uncertainty_scores: Path | None = None,
    n_bootstrap: int = 200,
) -> dict:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    group_map = _load_group_map(group_table)
    base, contract_records = load_base_tables(run_dirs, group_map)
    if base.empty:
        raise RuntimeError("No benchmark base tables were loaded.")

    contract_issues = validate_prediction_record_contract(contract_records, strict=False)
    base_with_magnitude = _records_with_magnitude(base, run_dirs)
    if base_with_magnitude.empty:
        raise RuntimeError("Could not attach true-effect magnitudes for LODO audit.")

    protocol_scores, protocol_formula = build_protocol_v0_2_scores(base)
    protocol_scores = protocol_scores[
        protocol_scores["score_name"].astype(str).isin(
            {"protocol_v0_2_family_confidence", "model_disagreement_risk"}
        )
    ].copy()
    existing_scores = pd.DataFrame()
    lodo_scores, lodo_status, linear_coefficients = build_lodo_error_ranker_scores(base)

    all_score_frames = []
    if not existing_scores.empty:
        all_score_frames.append(existing_scores)
    if not protocol_scores.empty:
        all_score_frames.append(_score_metadata_with_magnitude(protocol_scores, base_with_magnitude))
    if not lodo_scores.empty:
        all_score_frames.append(_score_metadata_with_magnitude(lodo_scores, base_with_magnitude))
    if gears_uncertainty_scores and gears_uncertainty_scores.exists():
        gears = pd.read_csv(gears_uncertainty_scores)
        gears["dataset_group"] = gears.get("dataset_name", pd.Series(dtype=str)).astype(str).map(
            group_map
        )
        all_score_frames.append(gears)

    if not all_score_frames:
        raise RuntimeError("No scores available for LODO evaluation.")

    all_scores = pd.concat(all_score_frames, ignore_index=True, sort=False)
    if "dataset_group" not in all_scores.columns or all_scores["dataset_group"].isna().any():
        from safetrans_confidence.data.records import _default_dataset_group

        all_scores["dataset_group"] = all_scores.get(
            "dataset_group", pd.Series(index=all_scores.index)
        ).fillna(all_scores["dataset_name"].map(_default_dataset_group))
    all_scores["risk_axis"] = _risk_axis(all_scores)

    base.to_csv(out_dir / "tables" / "LODO_BASE_RECORDS_WITH_FEATURES.csv", index=False)
    base_with_magnitude.to_csv(out_dir / "tables" / "LODO_RECORDS_WITH_EFFECT_MAGNITUDE.csv", index=False)
    protocol_scores.to_csv(out_dir / "tables" / "PROTOCOL_V0_2_RECOMPUTED_SCORES.csv", index=False)
    protocol_formula.to_csv(out_dir / "tables" / "PROTOCOL_V0_2_FORMULA_AUDIT.csv", index=False)
    lodo_scores.to_csv(out_dir / "tables" / "LODO_ERROR_RANKER_SCORES.csv", index=False)
    lodo_status.to_csv(out_dir / "tables" / "LODO_ERROR_RANKER_STATUS.csv", index=False)
    linear_coefficients.to_csv(out_dir / "tables" / "SAFECONF_LINEAR_RANKER_COEFFICIENTS.csv", index=False)
    all_scores.to_csv(out_dir / "tables" / "LODO_SCORES.csv", index=False)

    level_cols = ["dataset_group", "dataset_name", "predictor_name", "score_name"]
    summary = _eval_scores(all_scores, level_cols)
    summary.to_csv(out_dir / "tables" / "LODO_BASELINE_LADDER.csv", index=False)
    ci = _bootstrap_ci_table(all_scores, level_cols, n_bootstrap=n_bootstrap)
    ci.to_csv(out_dir / "tables" / "LODO_BOOTSTRAP_CI.csv", index=False)
    main = _decision_table(summary)
    if not main.empty and not ci.empty:
        merge_cols = ["dataset_group", "dataset_name", "predictor_name", "score_name"]
        main = main.merge(
            ci[
                [
                    *merge_cols,
                    "raw_spearman_ci_low",
                    "raw_spearman_ci_high",
                    "partial_rho_ci_low",
                    "partial_rho_ci_high",
                    "status",
                ]
            ],
            on=merge_cols,
            how="left",
            suffixes=("", "_bootstrap"),
        )
    main.to_csv(out_dir / "tables" / "LODO_MAIN_TABLE.csv", index=False)
    _risk_coverage(all_scores).to_csv(out_dir / "tables" / "LODO_RISK_COVERAGE.csv", index=False)

    report_lines = [
        "# Group-LODO audit",
        "",
        "This run leaves out entire dataset groups before training `lodo_error_ranker_risk`.",
        "",
        "Important interpretation:",
        "",
        "- `lodo_error_ranker_risk` is a shallow error scorer, not a perturbation predictor.",
        "- Held-out group records are never used for ErrorRanker fitting.",
        "- Main evidence is `partial_rho_control_magnitude` and within-stratum rho, not raw rho alone.",
        "",
        "## Contract issues",
        "",
        "```",
        "\n".join(contract_issues) if contract_issues else "No blocking contract issues in non-strict legacy mode.",
        "```",
        "",
        "## LODO ErrorRanker status",
        "",
        "```",
        lodo_status.to_string(index=False) if not lodo_status.empty else "No LODO ErrorRanker scores.",
        "```",
        "",
        "## Main table preview",
        "",
        "```",
        main.to_string(index=False) if not main.empty else "No main rows.",
        "```",
    ]
    (out_dir / "reports" / "LODO_DECISION.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    status = {
        "out_dir": str(out_dir),
        "n_run_dirs": int(len(run_dirs)),
        "n_records": int(len(base)),
        "n_dataset_groups": int(base["dataset_group"].nunique()),
        "n_scores": int(len(all_scores)),
        "n_lodo_error_ranker_scores": int(len(lodo_scores)),
        "n_linear_ranker_coefficients": int(len(linear_coefficients)),
        "n_bootstrap": int(n_bootstrap),
        "contract_issues": contract_issues,
        "status": "ok",
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leave-one-dataset-group-out confidence scoring audit."
    )
    parser.add_argument("--run-dir", type=Path, action="append", dest="run_dirs")
    parser.add_argument("--group-table", type=Path, default=None)
    parser.add_argument("--gears-uncertainty-scores", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = args.run_dirs or DEFAULT_RUN_DIRS
    print(
        json.dumps(
            run_lodo(
                run_dirs,
                args.out_dir,
                args.group_table,
                args.gears_uncertainty_scores,
                args.n_bootstrap,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
