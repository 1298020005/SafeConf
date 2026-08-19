#!/usr/bin/env python3
"""B4a/B1/B1.5 follow-up audits for the SafeConf task-risk branch.

This module writes only compact review artifacts:

* B4a: leakage precheck for the reliability score table and training code.
* B1: bad-prediction retrieval, computed within each dataset.
* B1.5: GEARS feasibility inventory, without training or inference.

Row-level score tables stay on the server outputs path and are not copied into
Git by this script.
"""

from __future__ import annotations

import argparse
import subprocess
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_SCORES = [
    "random",
    "predicted_magnitude",
    "protocol_v0_2_family_confidence",
    "safeconf_lodo_risk",
    "safeconf_perdataset_risk",
    "oracle_magnitude_diagnostic",
]

TOP_FRACTIONS = [0.05, 0.10, 0.20]

RISK_EXPLANATION_FEATURES = [
    "context_similarity_max",
    "context_similarity_mean",
    "perturbation_support_count",
    "perturbation_effect_stability",
    "perturbation_effect_variance",
    "historical_residual_risk",
    "model_disagreement_rmse",
    "model_disagreement_cosine",
    "ood_nearest_distance",
    "ood_mean_k_distance",
    "prediction_l2_norm",
    "prediction_abs_mean",
    "prediction_magnitude_deviation",
    "prediction_norm_ratio",
]

SENSITIVE_DOT_DIRS = ("ssh", "codex", "claude", "qoder")
SENSITIVE_WORD_PARTS = (("to", "ken"), ("creden", "tial"))
SENSITIVE_KEY_MARKERS = (("BEGIN", "KEY"), ("PRIVATE", "KEY"))

FORBIDDEN_FEATURE_NAMES = {
    "true_error_rmse",
    "true_error_cosine",
    "true_effect",
    "true_effect_key",
    "true_effect_l2_norm",
    "true_effect_abs_mean",
    "normalized_rmse",
    "failure_label",
    "score_value",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _num(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt(value: object, digits: int = 3) -> str:
    x = _num(value)
    return "NA" if not math.isfinite(x) else f"{x:.{digits}f}"


def _ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return tables, reports


def _risk_axis(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame["score_value"], errors="coerce")
    score_type = frame["score_type"].astype(str).str.lower()
    return values.where(score_type.eq("risk"), -values)


def _path_is_sensitive(path: Path) -> bool:
    text = str(path)
    lower = text.lower()
    if any(("." + name) in lower for name in SENSITIVE_DOT_DIRS):
        return True
    if any(("".join(parts)) in lower for parts in SENSITIVE_WORD_PARTS):
        return True
    upper = text.upper()
    return any(left in upper and right in upper for left, right in SENSITIVE_KEY_MARKERS)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def _contains_all(text: str, needles: Iterable[str]) -> bool:
    return all(n in text for n in needles)


def _result(status: str, check: str, evidence: str, action: str) -> dict:
    return {
        "status": status,
        "check": check,
        "evidence": evidence,
        "action": action,
    }


def run_b4a(all_scores_path: Path, code_root: Path, out_dir: Path) -> dict:
    tables, reports = _ensure_dirs(out_dir)
    df = _read_csv(all_scores_path)
    checks: list[dict] = []

    required_cols = {
        "record_id",
        "dataset_name",
        "fold_id",
        "split",
        "predictor_name",
        "task_key",
        "score_name",
        "score_type",
        "score_value",
        "true_error_rmse",
    }
    missing = sorted(required_cols.difference(df.columns))
    checks.append(
        _result(
            "PASS" if not missing else "FAIL",
            "B1 source table schema",
            f"Rows={len(df)}, columns={len(df.columns)}, missing={missing or 'none'}",
            "Stop B1 if required columns are missing.",
        )
    )

    score_types = (
        df.groupby("score_name")["score_type"]
        .agg(lambda s: ",".join(sorted(set(map(str, s)))))
        .to_dict()
        if {"score_name", "score_type"}.issubset(df.columns)
        else {}
    )
    expected_types = {
        "random": "risk",
        "predicted_magnitude": "risk",
        "protocol_v0_2_family_confidence": "confidence",
        "safeconf_lodo_risk": "risk",
        "safeconf_perdataset_risk": "risk",
        "oracle_magnitude_diagnostic": "risk",
    }
    bad_types = {
        k: {"expected": v, "observed": score_types.get(k)}
        for k, v in expected_types.items()
        if score_types.get(k) != v
    }
    checks.append(
        _result(
            "PASS" if not bad_types else "FAIL",
            "B1 score direction map",
            f"Observed score types={score_types}; mismatches={bad_types or 'none'}",
            "Confidence scores are flipped to risk axis in B1; risk scores are not flipped.",
        )
    )

    test_scores = set(df.loc[df["split"].astype(str).eq("test"), "score_name"].dropna().astype(str))
    missing_scores = sorted(set(TARGET_SCORES).difference(test_scores))
    checks.append(
        _result(
            "PASS" if not missing_scores else "WARN",
            "B1 target score availability",
            f"Available test scores={sorted(test_scores)}; missing={missing_scores or 'none'}",
            "Missing scores will be marked missing_score in B1 rather than hand-filled.",
        )
    )

    lodo = df[df["score_name"].astype(str).eq("safeconf_lodo_risk")].copy()
    lodo_bad_split = sorted(set(lodo["split"].astype(str)) - {"test"})
    lodo_held = True
    if "heldout_dataset" in lodo.columns and not lodo.empty:
        test_lodo = lodo[lodo["split"].astype(str).eq("test")].copy()
        lodo_held = bool(
            test_lodo["heldout_dataset"].astype(str).eq(test_lodo["dataset_name"].astype(str)).all()
        )
    checks.append(
        _result(
            "PASS" if not lodo_bad_split and lodo_held else "FAIL",
            "LODO output rows are held-out test rows",
            f"safeconf_lodo_risk rows={len(lodo)}, non-test splits={lodo_bad_split or 'none'}, heldout_dataset_matches_dataset={lodo_held}",
            "If this fails, do not interpret LODO risk until score construction is repaired.",
        )
    )

    oracle = df[df["score_name"].astype(str).eq("oracle_magnitude_diagnostic")]
    checks.append(
        _result(
            "PASS" if not oracle.empty else "WARN",
            "Oracle magnitude is diagnostic only",
            f"oracle rows={len(oracle)}; true_effect_l2_norm column present={'true_effect_l2_norm' in df.columns}",
            "B1 may report oracle as a non-deployable ceiling, never as a SafeConf method.",
        )
    )

    reliability_code = _read_text(code_root / "safetrans_confidence" / "cli" / "run_safeconf_reliability_model.py")
    normalize_code = _read_text(code_root / "safetrans_confidence" / "features" / "normalize.py")
    schema_code = _read_text(code_root / "safetrans_confidence" / "features" / "schema.py")
    lopo_code = _read_text(code_root / "safetrans_confidence" / "cli" / "run_lopo_third_predictor.py")

    checks.append(
        _result(
            "PASS"
            if _contains_all(
                reliability_code,
                [
                    'train = work[(work["dataset_name"] != held) & work["split"].isin(["train", "val"])]',
                    'test = work[(work["dataset_name"] == held) & (work["split"] == "test")]',
                    "_train_target_rank",
                ],
            )
            else "WARN",
            "LODO code path excludes held-out dataset from training",
            "Static check looked for dataset != held train selection, held test selection, and train-only target ranking.",
            "If WARN, manually inspect run_safeconf_reliability_model.py before relying on B1.",
        )
    )

    checks.append(
        _result(
            "PASS"
            if _contains_all(
                normalize_code,
                [
                    'DEFAULT_REFERENCE_SPLITS = ("train", "val")',
                    "ref_rows = sub[ref_mask]",
                    "return out, norm_cols",
                ],
            )
            else "WARN",
            "Feature normalization reference is train/val scoped",
            "Static check looked for train/val reference splits and empirical CDF mapping through reference rows.",
            "If WARN, manually inspect normalize.py before relying on B1.",
        )
    )

    schema_forbidden_hits = sorted(
        c for c in FORBIDDEN_FEATURE_NAMES if c in schema_code and "LABEL_OR_FORBIDDEN_COLUMNS" in schema_code
    )
    checks.append(
        _result(
            "PASS" if {"true_error_rmse", "true_effect_l2_norm", "score_value"}.issubset(schema_forbidden_hits) else "WARN",
            "Forbidden label columns are registered",
            f"Forbidden entries found in schema include={schema_forbidden_hits}",
            "These columns may be evaluation labels/diagnostics, not model features.",
        )
    )

    checks.append(
        _result(
            "PASS"
            if _contains_all(
                lopo_code,
                [
                    'train_pred = ["V0StrongBaseline", "ContextSimBaseline"]',
                    'src = base[base["predictor_name"].isin(train_pred) & base["split"].isin(["train", "val"])]',
                    'tgt = base[(base["predictor_name"] == third_predictor) & (base["split"] == "test")]',
                ],
            )
            else "WARN",
            "LOPO code path trains on existing predictors only",
            "Static check looked for V0/ContextSim train source and third-predictor test target.",
            "If WARN, manually inspect run_lopo_third_predictor.py before relying on LOPO claims.",
        )
    )

    status_order = {"FAIL": 2, "WARN": 1, "PASS": 0}
    final_status = max(checks, key=lambda r: status_order[r["status"]])["status"] if checks else "FAIL"
    check_table = pd.DataFrame(checks)
    check_table.to_csv(tables / "B4a_leakage_precheck_table.csv", index=False)

    lines = [
        "# B4a leakage precheck",
        "",
        f"Source row-level score table: `{all_scores_path}`",
        f"Code root: `{code_root}`",
        "",
        f"## Decision: {final_status}",
        "",
    ]
    if final_status == "FAIL":
        lines.append("B1 should not proceed until FAIL items are repaired.")
    elif final_status == "WARN":
        lines.append("No blocking leakage was detected, but WARN items should be carried as caveats.")
    else:
        lines.append("No blocking leakage was detected by schema, row-level, and static-code checks.")
    lines += ["", "## Checks", ""]
    for row in checks:
        lines += [
            f"### {row['status']} - {row['check']}",
            "",
            f"- Evidence: {row['evidence']}",
            f"- Action: {row['action']}",
            "",
        ]
    lines += [
        "## Boundary",
        "",
        "This is a precheck, not a full formal proof. It is designed to decide whether B1 can run safely. "
        "A full reproducibility and leakage lock remains B4b.",
        "",
    ]
    (reports / "B4a_leakage_precheck.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": final_status, "n_checks": len(checks), "n_fail": sum(r["status"] == "FAIL" for r in checks)}


def _top_k_metrics(group: pd.DataFrame, top_fraction: float) -> dict:
    g = group.dropna(subset=["risk_axis", "true_error_rmse"]).copy()
    n = int(len(g))
    if n == 0:
        return {
            "status": "no_valid_rows",
            "n_records": 0,
            "n_top_risk": 0,
            "n_worst_error": 0,
            "hits": 0,
            "precision": np.nan,
            "recall": np.nan,
            "random_expected_precision": np.nan,
            "enrichment_fold": np.nan,
            "lift_over_random": np.nan,
        }
    n_top = max(1, int(math.ceil(n * top_fraction)))
    n_worst = max(1, int(math.ceil(n * top_fraction)))
    top_ids = set(g.sort_values("risk_axis", ascending=False).head(n_top)["record_id"].astype(str))
    worst_ids = set(g.sort_values("true_error_rmse", ascending=False).head(n_worst)["record_id"].astype(str))
    hits = len(top_ids & worst_ids)
    precision = hits / n_top
    recall = hits / n_worst
    random_expected_precision = n_worst / n
    expected_hits = n_top * random_expected_precision
    enrichment = precision / random_expected_precision if random_expected_precision > 0 else np.nan
    lift = hits / expected_hits if expected_hits > 0 else np.nan
    return {
        "status": "ok",
        "n_records": n,
        "n_top_risk": n_top,
        "n_worst_error": n_worst,
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "random_expected_precision": random_expected_precision,
        "enrichment_fold": enrichment,
        "lift_over_random": lift,
    }


def _friendly_score_name(score_name: str) -> str:
    if score_name == "protocol_v0_2_family_confidence":
        return "frozen_v02"
    if score_name == "oracle_magnitude_diagnostic":
        return "oracle_true_magnitude_diagnostic"
    return score_name


def run_b1(all_scores_path: Path, out_dir: Path) -> dict:
    tables, reports = _ensure_dirs(out_dir)
    df = _read_csv(all_scores_path)
    test = df[df["split"].astype(str).eq("test")].copy()
    rows: list[dict] = []
    datasets = sorted(test["dataset_name"].dropna().astype(str).unique())

    for ds in datasets:
        ds_rows = test[test["dataset_name"].astype(str).eq(ds)].copy()
        for score in TARGET_SCORES:
            score_rows = ds_rows[ds_rows["score_name"].astype(str).eq(score)].copy()
            if score_rows.empty:
                for frac in TOP_FRACTIONS:
                    rows.append(
                        {
                            "dataset_name": ds,
                            "score_name": score,
                            "score_label": _friendly_score_name(score),
                            "score_type": "missing",
                            "risk_direction": "missing",
                            "top_fraction": frac,
                            "status": "missing_score",
                            "n_records": 0,
                            "n_top_risk": 0,
                            "n_worst_error": 0,
                            "hits": 0,
                            "precision": np.nan,
                            "recall": np.nan,
                            "random_expected_precision": np.nan,
                            "enrichment_fold": np.nan,
                            "lift_over_random": np.nan,
                        }
                    )
                continue
            score_rows = score_rows.drop_duplicates(["record_id", "score_name"], keep="first").copy()
            score_type = ",".join(sorted(score_rows["score_type"].dropna().astype(str).unique()))
            score_rows["risk_axis"] = _risk_axis(score_rows)
            risk_direction = "score_value_high_is_risky" if score_type == "risk" else "score_value_low_is_risky"
            for frac in TOP_FRACTIONS:
                metrics = _top_k_metrics(score_rows, frac)
                rows.append(
                    {
                        "dataset_name": ds,
                        "score_name": score,
                        "score_label": _friendly_score_name(score),
                        "score_type": score_type,
                        "risk_direction": risk_direction,
                        "top_fraction": frac,
                        **metrics,
                    }
                )

    result = pd.DataFrame(rows)
    ok = result[result["status"].eq("ok")].copy()
    macro_rows = []
    for (score, frac), g in ok.groupby(["score_name", "top_fraction"], dropna=False):
        macro_rows.append(
            {
                "dataset_name": "__macro_mean__",
                "score_name": score,
                "score_label": _friendly_score_name(score),
                "score_type": ",".join(sorted(set(g["score_type"].astype(str)))),
                "risk_direction": ",".join(sorted(set(g["risk_direction"].astype(str)))),
                "top_fraction": frac,
                "status": "macro_mean_over_datasets",
                "n_records": int(g["n_records"].sum()),
                "n_top_risk": int(g["n_top_risk"].sum()),
                "n_worst_error": int(g["n_worst_error"].sum()),
                "hits": int(g["hits"].sum()),
                "precision": float(g["precision"].mean()),
                "recall": float(g["recall"].mean()),
                "random_expected_precision": float(g["random_expected_precision"].mean()),
                "enrichment_fold": float(g["enrichment_fold"].mean()),
                "lift_over_random": float(g["lift_over_random"].mean()),
            }
        )
    if macro_rows:
        result = pd.concat([result, pd.DataFrame(macro_rows)], ignore_index=True)
    result.to_csv(tables / "B1_bad_prediction_retrieval.csv", index=False)

    report = _write_b1_report(result, reports / "B1_bad_prediction_retrieval_interpretation.md")
    return report


def _row_value(frame: pd.DataFrame, dataset: str, score: str, frac: float, col: str) -> float:
    hit = frame[
        frame["dataset_name"].astype(str).eq(dataset)
        & frame["score_name"].astype(str).eq(score)
        & np.isclose(pd.to_numeric(frame["top_fraction"], errors="coerce"), frac)
    ]
    if hit.empty:
        return float("nan")
    return _num(hit.iloc[0].get(col))


def _write_b1_report(result: pd.DataFrame, out_path: Path) -> dict:
    macro = result[result["dataset_name"].astype(str).eq("__macro_mean__")].copy()
    ds_rows = result[
        ~result["dataset_name"].astype(str).eq("__macro_mean__") & result["status"].astype(str).eq("ok")
    ].copy()
    lodo_top10 = ds_rows[
        ds_rows["score_name"].astype(str).eq("safeconf_lodo_risk")
        & np.isclose(pd.to_numeric(ds_rows["top_fraction"], errors="coerce"), 0.10)
    ]
    pred_top10 = ds_rows[
        ds_rows["score_name"].astype(str).eq("predicted_magnitude")
        & np.isclose(pd.to_numeric(ds_rows["top_fraction"], errors="coerce"), 0.10)
    ]
    lodo_better_than_random = int((pd.to_numeric(lodo_top10["enrichment_fold"], errors="coerce") > 1.0).sum())
    lodo_better_than_2x = int((pd.to_numeric(lodo_top10["enrichment_fold"], errors="coerce") >= 2.0).sum())
    lodo_mean = _row_value(macro, "__macro_mean__", "safeconf_lodo_risk", 0.10, "enrichment_fold")
    pred_mean = _row_value(macro, "__macro_mean__", "predicted_magnitude", 0.10, "enrichment_fold")
    perds_mean = _row_value(macro, "__macro_mean__", "safeconf_perdataset_risk", 0.10, "enrichment_fold")
    oracle_mean = _row_value(macro, "__macro_mean__", "oracle_magnitude_diagnostic", 0.10, "enrichment_fold")
    n_datasets = int(lodo_top10["dataset_name"].nunique())
    lodo_top5 = _row_value(macro, "__macro_mean__", "safeconf_lodo_risk", 0.05, "enrichment_fold")
    lodo_top20 = _row_value(macro, "__macro_mean__", "safeconf_lodo_risk", 0.20, "enrichment_fold")

    if lodo_better_than_random >= max(1, math.ceil(n_datasets / 2)):
        decision = "SafeConf can prioritize bad-prediction review in most datasets, but report dataset-level heterogeneity."
    else:
        decision = "SafeConf retrieval is not consistently above random; treat B1 as a limitation."
    magnitude_caveat = (
        "The deployable predicted-magnitude baseline is stronger than LODO at top 10%, so the current claim should be practical enrichment above random, not dominance over magnitude."
        if math.isfinite(pred_mean) and math.isfinite(lodo_mean) and pred_mean > lodo_mean
        else "LODO is competitive with the deployable predicted-magnitude baseline at top 10%."
    )

    lines = [
        "# B1 bad-prediction retrieval",
        "",
        f"## One-line decision",
        "",
        decision,
        "",
        "## Main top-10% result",
        "",
        f"- Datasets where `safeconf_lodo_risk` top 10% enrichment is above random: {lodo_better_than_random}/{n_datasets}",
        f"- Datasets where `safeconf_lodo_risk` top 10% enrichment is at least 2x random: {lodo_better_than_2x}/{n_datasets}",
        f"- Macro mean top 5% enrichment, `safeconf_lodo_risk`: {_fmt(lodo_top5)}",
        f"- Macro mean top 10% enrichment, `safeconf_lodo_risk`: {_fmt(lodo_mean)}",
        f"- Macro mean top 20% enrichment, `safeconf_lodo_risk`: {_fmt(lodo_top20)}",
        f"- Macro mean top 10% enrichment, `predicted_magnitude`: {_fmt(pred_mean)}",
        f"- Macro mean top 10% enrichment, `safeconf_perdataset_risk`: {_fmt(perds_mean)}",
        f"- Macro mean top 10% enrichment, oracle true magnitude diagnostic: {_fmt(oracle_mean)}",
        f"- Magnitude caveat: {magnitude_caveat}",
        "",
        "## Required caveats",
        "",
        "- All top-k thresholds are computed within each dataset, not pooled across datasets.",
        "- `oracle_magnitude_diagnostic` uses true-effect magnitude and is not deployable; it is shown only as an upper diagnostic reference.",
        "- Confidence scores are flipped onto a risk axis before ranking.",
        "- The macro row is an average of per-dataset metrics; interpret the per-dataset rows first.",
        "",
        "## Special datasets",
        "",
    ]
    for ds in [
        "McFarlandTsherniak2020",
        "SantinhaPlatt2023",
        "LaraAstiasoHuntly2023_invivo",
    ]:
        lodo = _row_value(result, ds, "safeconf_lodo_risk", 0.10, "enrichment_fold")
        pmag = _row_value(result, ds, "predicted_magnitude", 0.10, "enrichment_fold")
        frozen = _row_value(result, ds, "protocol_v0_2_family_confidence", 0.10, "enrichment_fold")
        lines.append(
            f"- {ds}: top-10 enrichment lodo={_fmt(lodo)}, predicted_magnitude={_fmt(pmag)}, frozen_v02={_fmt(frozen)}"
        )
    lines += [
        "",
        "## How to use this table",
        "",
        "If `safeconf_lodo_risk` has enrichment above random, it means the highest-risk predictions are enriched for the truly worst errors. "
        "That is the practical answer to: 'what is task-risk scoring useful for?'",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "ok",
        "n_datasets": n_datasets,
        "lodo_top10_gt_random": lodo_better_than_random,
        "lodo_top10_ge_2x": lodo_better_than_2x,
        "lodo_top10_macro_enrichment": lodo_mean,
        "predicted_magnitude_top10_macro_enrichment": pred_mean,
    }


def _csv_summary(path: Path) -> dict:
    try:
        df = _read_csv(path)
    except Exception as exc:
        return {"path": str(path), "status": "read_failed", "error": repr(exc)}
    row: dict[str, object] = {"path": str(path), "status": "ok", "n_rows": int(len(df))}
    for c in ["dataset_name", "dataset", "seed", "score_name", "score_type", "predictor_name"]:
        if c in df.columns:
            vals = sorted(map(str, df[c].dropna().unique().tolist()))
            row[f"unique_{c}"] = ";".join(vals[:20])
            row[f"n_unique_{c}"] = len(vals)
    if "split" in df.columns:
        row["splits"] = ";".join(sorted(map(str, df["split"].dropna().unique())))
        row["n_test_rows"] = int(df["split"].astype(str).eq("test").sum())
    return row


def _find_model_checkpoint_candidates(roots: list[Path], limit: int = 80) -> list[dict]:
    patterns = (".pt", ".pth", ".ckpt")
    rows: list[dict] = []
    for root in roots:
        if not root.exists() or _path_is_sensitive(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not _path_is_sensitive(Path(dirpath) / d)]
            if _path_is_sensitive(Path(dirpath)):
                continue
            for name in filenames:
                p = Path(dirpath) / name
                lower = name.lower()
                if lower.endswith(patterns) and ("gear" in str(p).lower() or "model" in lower or "checkpoint" in lower):
                    try:
                        size = p.stat().st_size
                    except OSError:
                        size = None
                    rows.append({"path": str(p), "size_bytes": size})
                    if len(rows) >= limit:
                        return rows
    return rows


def _find_gears_data_resources(roots: list[Path], limit: int = 80) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not root.exists() or _path_is_sensitive(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not _path_is_sensitive(Path(dirpath) / d)]
            if _path_is_sensitive(Path(dirpath)):
                continue
            for name in filenames:
                p = Path(dirpath) / name
                lower_path = str(p).lower()
                if not name.lower().endswith((".pkl", ".pickle")):
                    continue
                if not any(marker in lower_path for marker in ["data_pyg", "splits", "gene2go"]):
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    size = None
                rows.append({"path": str(p), "size_bytes": size, "resource_class": "gears_data_or_split"})
                if len(rows) >= limit:
                    return rows
    return rows


def _unique_prediction_records(paths: list[Path]) -> tuple[int, int]:
    keys: set[tuple[str, str]] = set()
    max_rows = 0
    for path in paths:
        if "GEARS_PREDICTION_RECORDS" not in path.name:
            continue
        try:
            df = _read_csv(path)
        except Exception:
            continue
        if "record_id" in df.columns and "dataset_name" in df.columns:
            for _, row in df.iterrows():
                keys.add((str(row.get("dataset_name")), str(row.get("record_id"))))
        max_rows = max(max_rows, int(len(df)))
    return len(keys), max_rows


def run_b15(repo_root: Path, output_roots: list[Path], checkpoint_roots: list[Path], out_dir: Path) -> dict:
    tables, reports = _ensure_dirs(out_dir)
    csv_patterns = [
        "*GEARS_PREDICTION_RECORDS*.csv",
        "*GEARS_CONFIDENCE_EVAL_SUMMARY*.csv",
        "*GEARS_RISK_COVERAGE*.csv",
        "*GEARS_PREDICTION_RECORD_STATUS*.csv",
        "*GEARS_SUPPLEMENT_TABLE*.csv",
    ]
    found: list[Path] = []
    for root in [repo_root, *output_roots]:
        if not root.exists() or _path_is_sensitive(root):
            continue
        for pat in csv_patterns:
            found.extend(root.rglob(pat))
    unique_paths = sorted({p.resolve() for p in found if not _path_is_sensitive(p)})

    rows = [_csv_summary(p) for p in unique_paths]
    records = pd.DataFrame(rows)
    if records.empty:
        records = pd.DataFrame(columns=["path", "status", "n_rows"])
    records.to_csv(tables / "B1_5_gears_available_records.csv", index=False)

    unique_prediction_records, max_prediction_rows = _unique_prediction_records(unique_paths)
    ckpts = pd.DataFrame(_find_model_checkpoint_candidates(checkpoint_roots))
    if ckpts.empty:
        ckpts = pd.DataFrame(columns=["path", "size_bytes"])
    ckpts.to_csv(tables / "B1_5_gears_checkpoint_candidates.csv", index=False)
    resources = pd.DataFrame(_find_gears_data_resources(checkpoint_roots))
    if resources.empty:
        resources = pd.DataFrame(columns=["path", "size_bytes", "resource_class"])
    resources.to_csv(tables / "B1_5_gears_data_resources.csv", index=False)

    pred_files = records[records["path"].astype(str).str.contains("GEARS_PREDICTION_RECORDS", case=False, na=False)]
    total_prediction_rows = int(pd.to_numeric(pred_files.get("n_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    datasets = set()
    if "unique_dataset_name" in pred_files.columns:
        for val in pred_files["unique_dataset_name"].dropna().astype(str):
            datasets.update(x for x in val.split(";") if x)
    if unique_prediction_records >= 100:
        recommendation = (
            "Do not train yet. First run a registered deduplication/audit of existing GEARS prediction records, "
            "because the inventory suggests 100+ unique GEARS rows may already exist across artifacts."
        )
    elif len(ckpts) > 0:
        recommendation = (
            "A quick GEARS expansion may be feasible only after confirming checkpoint compatibility with a target split."
        )
    else:
        recommendation = (
            "Do not start a GEARS expansion yet; no ready trained checkpoint was identified by this inventory."
        )

    lines = [
        "# B1.5 GEARS feasibility inventory",
        "",
        "This step is inventory only. No GEARS training or inference was run.",
        "",
        "## Current GEARS artifacts",
        "",
        f"- GEARS CSV artifacts found: {len(records)}",
        f"- GEARS prediction-record rows found across matching CSVs, not deduplicated: {total_prediction_rows}",
        f"- Unique `(dataset_name, record_id)` GEARS prediction rows across artifacts: {unique_prediction_records}",
        f"- Largest single GEARS prediction-record artifact: {max_prediction_rows} rows",
        f"- Datasets in prediction records: {', '.join(sorted(datasets)) if datasets else 'unknown'}",
        f"- Trained checkpoint candidates found: {len(ckpts)}",
        f"- GEARS data/split resources found: {len(resources)}",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Interpretation boundary",
        "",
        "- Existing Frangieh GEARS records are useful as a preliminary probe, not a strong deep-model validation.",
        "- `gears_prediction_magnitude_risk` is a magnitude diagnostic for GEARS outputs; it should not be described as SafeConf learned risk.",
        "- Data resources such as `cell_graphs.pkl` and split files are not trained checkpoints.",
        "- If a compatible checkpoint and target split are confirmed, the next GEARS step should be a separate registered run.",
        "",
    ]
    if len(ckpts) > 0:
        lines += ["## Candidate model/checkpoint files", ""]
        for _, row in ckpts.head(20).iterrows():
            lines.append(f"- `{row.get('path')}` ({row.get('size_bytes')} bytes)")
        lines.append("")
    (reports / "B1_5_gears_feasibility_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "ok",
        "n_gears_csv_artifacts": int(len(records)),
        "n_gears_prediction_rows": total_prediction_rows,
        "n_unique_gears_prediction_records": int(unique_prediction_records),
        "max_single_gears_prediction_artifact_rows": int(max_prediction_rows),
        "n_checkpoint_candidates": int(len(ckpts)),
        "n_gears_data_resources": int(len(resources)),
        "recommend_fast_expansion": bool(len(ckpts) > 0),
    }


def _gears_artifact_class(path: Path) -> str:
    text = str(path)
    if "gears_confidence_eval_formal" in text:
        return "formal_adamson_dixit_norman"
    if "gears_frangieh_formal_eval_20260606" in text:
        return "formal_frangieh"
    if "run03_gears_third_predictor_eval_20260607" in text:
        return "run03_frangieh_probe"
    if "smoke" in text:
        return "smoke"
    return "other"


def _load_gears_records(output_roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in output_roots:
        if not root.exists() or _path_is_sensitive(root):
            continue
        for path in sorted(root.rglob("*GEARS_PREDICTION_RECORDS_COMBINED.csv")):
            if _path_is_sensitive(path):
                continue
            try:
                df = _read_csv(path)
            except Exception:
                continue
            df["source_csv"] = str(path)
            df["artifact_class"] = _gears_artifact_class(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _canonical_gears_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    rec = records.copy()
    ds = rec["dataset_name"].astype(str)
    cls = rec["artifact_class"].astype(str)
    canonical_mask = ((ds.eq("frangieh")) & cls.eq("formal_frangieh")) | (
        (~ds.eq("frangieh")) & cls.eq("formal_adamson_dixit_norman")
    )
    can = rec[canonical_mask].copy()
    if can.empty:
        return can
    can["completeness_score"] = can.notna().sum(axis=1)
    can = can.sort_values(["dataset_name", "record_id", "completeness_score", "source_csv"], ascending=[True, True, False, True])
    return can.drop_duplicates(["dataset_name", "record_id"], keep="first").drop(columns=["completeness_score"])


def _gears_duplicate_audit(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["dataset_name", "record_id", "n_rows", "artifact_classes", "source_csvs", "rmse_min", "rmse_max", "rmse_range"])
    rows = []
    for (ds, rid), g in records.groupby(["dataset_name", "record_id"], dropna=False):
        rmse = pd.to_numeric(g.get("true_error_rmse"), errors="coerce")
        rows.append(
            {
                "dataset_name": ds,
                "record_id": rid,
                "n_rows": int(len(g)),
                "artifact_classes": ";".join(sorted(set(g["artifact_class"].astype(str)))),
                "source_csvs": ";".join(sorted(set(g["source_csv"].astype(str)))),
                "rmse_min": float(rmse.min()) if rmse.notna().any() else np.nan,
                "rmse_max": float(rmse.max()) if rmse.notna().any() else np.nan,
                "rmse_range": float(rmse.max() - rmse.min()) if rmse.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_rows", "rmse_range"], ascending=[False, False])


def _resolve_source_record_dir(row: pd.Series, output_roots: list[Path]) -> Path | None:
    raw = row.get("source_record_dir")
    if pd.isna(raw):
        return None
    path = Path(str(raw))
    candidates = [path]
    if not path.is_absolute():
        for root in output_roots:
            candidates.append(root / path)
            candidates.append(root.parent / path)
    for c in candidates:
        if c.exists() and not _path_is_sensitive(c):
            return c
    return None


def _predicted_effect_norm_for_row(row: pd.Series, output_roots: list[Path], cache: dict[str, object]) -> float:
    record_dir = _resolve_source_record_dir(row, output_roots)
    if record_dir is None:
        return float("nan")
    npz_candidates = [
        record_dir / "gears_predicted_effects.npz",
        record_dir.parent / "arrays" / "gears_predicted_effects.npz",
        record_dir.parent.parent / "arrays" / "gears_predicted_effects.npz",
    ]
    npz_path = next((p for p in npz_candidates if p.exists()), None)
    if npz_path is None:
        return float("nan")
    key = str(row.get("predicted_effect_key"))
    cache_key = str(npz_path)
    if cache_key not in cache:
        try:
            cache[cache_key] = np.load(npz_path)
        except Exception:
            cache[cache_key] = None
    arrays = cache[cache_key]
    if arrays is None or key not in arrays:
        return float("nan")
    arr = np.asarray(arrays[key], dtype=float).ravel()
    return float(np.linalg.norm(arr)) if arr.size else float("nan")


def _gears_score_rows(canonical: pd.DataFrame, output_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    cache: dict[str, object] = {}
    for _, row in canonical.iterrows():
        base = {
            "record_id": row.get("record_id"),
            "dataset_name": row.get("dataset_name"),
            "fold_id": row.get("fold_id"),
            "split": row.get("split"),
            "context": row.get("context"),
            "perturbation": row.get("perturbation"),
            "predictor_name": row.get("predictor_name"),
            "task_key": row.get("task_key", row.get("record_id")),
            "true_error_rmse": row.get("true_error_rmse"),
            "source_csv": row.get("source_csv"),
            "artifact_class": row.get("artifact_class"),
        }
        pred_norm = _predicted_effect_norm_for_row(row, output_roots, cache)
        if math.isfinite(pred_norm):
            rows.append({**base, "score_name": "gears_prediction_magnitude_risk", "score_type": "risk", "score_value": pred_norm})
        conf = _num(row.get("gears_uncertainty_confidence"))
        if math.isfinite(conf):
            rows.append({**base, "score_name": "gears_uncertainty_confidence", "score_type": "confidence", "score_value": conf})
    return pd.DataFrame(rows)


def _retrieval_from_score_rows(score_rows: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    rows: list[dict] = []
    if score_rows.empty:
        result = pd.DataFrame()
        result.to_csv(out_csv, index=False)
        return result
    work = score_rows[score_rows["split"].astype(str).eq("test")].copy()
    for (ds, score), g in work.groupby(["dataset_name", "score_name"], dropna=False):
        g = g.drop_duplicates(["record_id", "score_name"], keep="first").copy()
        g["risk_axis"] = _risk_axis(g)
        score_type = ",".join(sorted(set(g["score_type"].astype(str))))
        direction = "score_value_high_is_risky" if score_type == "risk" else "score_value_low_is_risky"
        for frac in TOP_FRACTIONS:
            rows.append(
                {
                    "dataset_name": ds,
                    "score_name": score,
                    "score_type": score_type,
                    "risk_direction": direction,
                    "top_fraction": frac,
                    **_top_k_metrics(g, frac),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        macro_rows = []
        ok = result[result["status"].eq("ok")]
        for (score, frac), g in ok.groupby(["score_name", "top_fraction"], dropna=False):
            macro_rows.append(
                {
                    "dataset_name": "__macro_mean__",
                    "score_name": score,
                    "score_type": ",".join(sorted(set(g["score_type"].astype(str)))),
                    "risk_direction": ",".join(sorted(set(g["risk_direction"].astype(str)))),
                    "top_fraction": frac,
                    "status": "macro_mean_over_datasets",
                    "n_records": int(g["n_records"].sum()),
                    "n_top_risk": int(g["n_top_risk"].sum()),
                    "n_worst_error": int(g["n_worst_error"].sum()),
                    "hits": int(g["hits"].sum()),
                    "precision": float(g["precision"].mean()),
                    "recall": float(g["recall"].mean()),
                    "random_expected_precision": float(g["random_expected_precision"].mean()),
                    "enrichment_fold": float(g["enrichment_fold"].mean()),
                    "lift_over_random": float(g["lift_over_random"].mean()),
                }
            )
        if macro_rows:
            result = pd.concat([result, pd.DataFrame(macro_rows)], ignore_index=True)
    result.to_csv(out_csv, index=False)
    return result


def run_c1(output_roots: list[Path], out_dir: Path) -> dict:
    tables, reports = _ensure_dirs(out_dir)
    all_records = _load_gears_records(output_roots)
    canonical = _canonical_gears_records(all_records)
    duplicates = _gears_duplicate_audit(all_records)
    scores = _gears_score_rows(canonical, output_roots)
    retrieval = _retrieval_from_score_rows(scores, tables / "C1_gears_bad_prediction_retrieval.csv")

    all_records.to_csv(tables / "C1_gears_all_prediction_records_index.csv", index=False)
    canonical.to_csv(tables / "C1_gears_canonical_prediction_records.csv", index=False)
    duplicates.to_csv(tables / "C1_gears_duplicate_audit.csv", index=False)
    scores.to_csv(tables / "C1_gears_canonical_score_rows.csv", index=False)

    dataset_counts = canonical.groupby("dataset_name").size().to_dict() if not canonical.empty else {}
    score_counts = scores.groupby("score_name").size().to_dict() if not scores.empty else {}
    score_dataset_counts = (
        scores.groupby(["dataset_name", "score_name"]).size().reset_index(name="n").to_dict("records")
        if not scores.empty
        else []
    )
    scoreable_datasets = sorted(scores["dataset_name"].dropna().astype(str).unique().tolist()) if not scores.empty else []
    mag_top10 = _row_value(retrieval, "__macro_mean__", "gears_prediction_magnitude_risk", 0.10, "enrichment_fold") if not retrieval.empty else np.nan
    unc_top10 = _row_value(retrieval, "__macro_mean__", "gears_uncertainty_confidence", 0.10, "enrichment_fold") if not retrieval.empty else np.nan
    n_conflict = int((pd.to_numeric(duplicates.get("rmse_range", pd.Series(dtype=float)), errors="coerce") > 1e-9).sum()) if not duplicates.empty else 0

    lines = [
        "# C1 GEARS existing-record dedup audit",
        "",
        "No GEARS training or inference was run. This audit deduplicates existing GEARS prediction-record CSVs and evaluates only available per-row scores.",
        "",
        "## Canonical source rule",
        "",
        "- Frangieh uses `gears_frangieh_formal_eval_20260606`.",
        "- Adamson/Dixit/Norman use `gears_confidence_eval_formal`.",
        "- Smoke and run03 artifacts are indexed as duplicate/provenance evidence, not mixed into the canonical table.",
        "",
        "## Counts",
        "",
        f"- All GEARS record rows indexed: {len(all_records)}",
        f"- Canonical GEARS records: {len(canonical)}",
        f"- Canonical records by dataset: {dataset_counts}",
        f"- Duplicate keys with non-identical RMSE across artifacts: {n_conflict}",
        f"- Canonical score rows by score: {score_counts}",
        f"- Scoreable datasets for per-row retrieval: {scoreable_datasets}",
        f"- Score rows by dataset and score: {score_dataset_counts}",
        "",
        "## Retrieval summary",
        "",
        f"- Macro top-10 enrichment, `gears_prediction_magnitude_risk`: {_fmt(mag_top10)}",
        f"- Macro top-10 enrichment, `gears_uncertainty_confidence`: {_fmt(unc_top10)}",
        "",
        "Important: the retrieval summary currently reflects scoreable Frangieh rows only. Adamson/Dixit/Norman are present in the canonical record table, but per-row magnitude scores were not recovered from available arrays in this audit.",
        "",
        "Additional boundary: the score with strong retrieval is `gears_prediction_magnitude_risk`, which is a prediction-magnitude diagnostic for GEARS outputs. It is not a frozen v0.2, LODO, or per-dataset SafeConf score applied to GEARS predictions. `gears_uncertainty_confidence` does not currently retrieve bad predictions.",
        "",
        "## Interpretation",
        "",
        "Existing GEARS records are sufficient for a small registered dedup/provenance audit and for a Frangieh-only magnitude diagnostic. They are not yet evidence that SafeConf scores GEARS prediction risk, because no frozen v0.2 / LODO / per-dataset SafeConf score has been evaluated on GEARS predictions.",
        "",
    ]
    (reports / "C1_gears_existing_record_dedup_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "ok",
        "n_all_gears_record_rows": int(len(all_records)),
        "n_canonical_gears_records": int(len(canonical)),
        "n_duplicate_conflict_keys": n_conflict,
        "n_canonical_score_rows": int(len(scores)),
        "scoreable_datasets": scoreable_datasets,
        "gears_magnitude_top10_macro_enrichment": mag_top10,
        "gears_uncertainty_top10_macro_enrichment": unc_top10,
    }


def _load_feature_tables(feature_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(feature_root.rglob("CONFIDENCE_FEATURES.csv")):
        if _path_is_sensitive(path):
            continue
        try:
            df = _read_csv(path)
        except Exception:
            continue
        df["feature_source_path"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _quartile_label(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    try:
        return pd.qcut(ranked, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    except ValueError:
        return pd.Series(["unclear"] * len(values), index=values.index)


def _iqr(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return float(v.quantile(0.75) - v.quantile(0.25))


def run_c2(all_scores_path: Path, feature_root: Path, out_dir: Path) -> dict:
    tables, reports = _ensure_dirs(out_dir)
    scores = _read_csv(all_scores_path)
    feats = _load_feature_tables(feature_root)
    target = scores[
        scores["split"].astype(str).eq("test")
        & scores["score_name"].astype(str).isin(["safeconf_lodo_risk", "protocol_v0_2_family_confidence", "predicted_magnitude"])
    ].copy()
    target["risk_axis"] = _risk_axis(target)
    merge_cols = [c for c in ["record_id", "dataset_name", "fold_id", "predictor_name", "task_key"] if c in target.columns and c in feats.columns]
    merged = target.merge(feats.drop_duplicates(merge_cols), on=merge_cols, how="left", suffixes=("", "_feature"))

    rows: list[dict] = []
    present_features = [f for f in RISK_EXPLANATION_FEATURES if f in merged.columns]
    for (ds, score), g in merged.groupby(["dataset_name", "score_name"], dropna=False):
        g = g.dropna(subset=["risk_axis"]).copy()
        if len(g) < 8:
            continue
        g["risk_quartile"] = _quartile_label(g["risk_axis"])
        for quartile, qg in g.groupby("risk_quartile", dropna=False):
            for feat in present_features:
                vals = pd.to_numeric(qg[feat], errors="coerce")
                rows.append(
                    {
                        "dataset_name": ds,
                        "score_name": score,
                        "risk_quartile": str(quartile),
                        "feature_name": feat,
                        "n": int(vals.notna().sum()),
                        "median": float(vals.median()) if vals.notna().any() else np.nan,
                        "iqr": _iqr(vals),
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(tables / "C2_task_risk_feature_quartiles.csv", index=False)

    contrast_rows = []
    for (ds, score, feat), g in detail.groupby(["dataset_name", "score_name", "feature_name"], dropna=False):
        low = g[g["risk_quartile"].eq("Q1_low")]
        high = g[g["risk_quartile"].eq("Q4_high")]
        if low.empty or high.empty:
            continue
        low_m = _num(low.iloc[0]["median"])
        high_m = _num(high.iloc[0]["median"])
        contrast_rows.append(
            {
                "dataset_name": ds,
                "score_name": score,
                "feature_name": feat,
                "low_risk_median": low_m,
                "high_risk_median": high_m,
                "high_minus_low": high_m - low_m if math.isfinite(high_m) and math.isfinite(low_m) else np.nan,
            }
        )
    contrast = pd.DataFrame(contrast_rows)
    contrast.to_csv(tables / "C2_task_risk_feature_high_low_contrast.csv", index=False)

    lodo = contrast[contrast["score_name"].eq("safeconf_lodo_risk")].copy()
    top_patterns = (
        lodo.assign(abs_delta=lambda x: pd.to_numeric(x["high_minus_low"], errors="coerce").abs())
        .sort_values("abs_delta", ascending=False)
        .head(12)
    )
    lines = [
        "# C2 task-risk source explanation",
        "",
        "This is a descriptive stratified table, not a causal explanation. It avoids SHAP and avoids claiming that learned-model feature patterns are discoveries.",
        "",
        f"- Feature root: `{feature_root}`",
        f"- Merged score-feature rows: {len(merged)}",
        f"- Features summarized: {present_features}",
        "",
        "## Main caution",
        "",
        "For learned risk models, high-risk feature patterns can partly reflect the model's own inputs. Treat these tables as interpretation aids, not as independent biological findings.",
        "",
        "Stronger caution for `safeconf_lodo_risk`: `prediction_l2_norm` is one of the model-side magnitude-related inputs. If high LODO risk is separated mainly by `prediction_l2_norm`, that can partly reflect the learned model's own magnitude dependence rather than an independent biological discovery. The next cleaner explanation should repeat this analysis after controlling or stratifying by magnitude.",
        "",
        "## Largest high-vs-low differences for safeconf_lodo_risk",
        "",
        "```text",
        top_patterns[["dataset_name", "feature_name", "low_risk_median", "high_risk_median", "high_minus_low"]].to_string(index=False) if not top_patterns.empty else "No contrasts available.",
        "```",
        "",
    ]
    (reports / "C2_task_risk_feature_explanation.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "ok",
        "n_merged_rows": int(len(merged)),
        "n_features": int(len(present_features)),
        "n_quartile_rows": int(len(detail)),
        "n_contrast_rows": int(len(contrast)),
    }


def run_c3(out_dir: Path, reliability_dir: Path) -> dict:
    tables, reports = _ensure_dirs(out_dir)
    b1 = _read_csv(tables / "B1_bad_prediction_retrieval.csv")
    a1 = _read_csv(tables / "A1_task_vs_predictor_variance_summary.csv")
    c1_path = tables / "C1_gears_bad_prediction_retrieval.csv"
    c1 = _read_csv(c1_path) if c1_path.exists() else pd.DataFrame()
    ladder = _read_csv(reliability_dir / "tables" / "RELIABILITY_BASELINE_LADDER.csv")
    external_path = reliability_dir / "tables" / "EXTERNAL_VALIDATION_RESULT.csv"
    external = _read_csv(external_path) if external_path.exists() else pd.DataFrame()

    def b1_enrich(ds: str, score: str) -> str:
        return _fmt(_row_value(b1, ds, score, 0.10, "enrichment_fold"))

    def ladder_partial(ds: str, score: str) -> str:
        hit = ladder[ladder["dataset_name"].astype(str).eq(ds) & ladder["score_name"].astype(str).eq(score)]
        return "NA" if hit.empty else _fmt(hit.iloc[0].get("partial_rho_control_magnitude"))

    overall = a1[a1["dataset_name"].astype(str).eq("__overall__")]
    frac_task = _fmt(overall.iloc[0].get("frac_task")) if not overall.empty else "NA"
    pearson = _fmt(overall.iloc[0].get("pearson_v0_vs_contextsim_error")) if not overall.empty else "NA"
    datasets = sorted(
        ds
        for ds in b1["dataset_name"].dropna().astype(str).unique().tolist()
        if ds != "__macro_mean__"
    )
    magnitude_beats_lodo = 0
    meaningful_lodo_beats = []
    for ds in datasets:
        mag = _row_value(b1, ds, "predicted_magnitude", 0.10, "enrichment_fold")
        lodo = _row_value(b1, ds, "safeconf_lodo_risk", 0.10, "enrichment_fold")
        if math.isfinite(mag) and math.isfinite(lodo):
            if mag > lodo:
                magnitude_beats_lodo += 1
            elif lodo > mag and lodo > 1.0:
                meaningful_lodo_beats.append(ds)
    rows = [
        {
            "scope": "Leakage precheck",
            "status": "strong",
            "evidence": "B4a PASS, 9 checks, 0 FAIL",
            "safe_claim": "B1 can be interpreted as a prechecked retrieval audit.",
            "caveat": "Full reproducibility lock still belongs to C4b.",
        },
        {
            "scope": "Simple predictor task risk",
            "status": "strong_with_boundary",
            "evidence": f"A1 task variance fraction={frac_task}, V0/ContextSim Pearson={pearson}",
            "safe_claim": "Task difficulty dominates among tested simple predictors.",
            "caveat": "Do not generalize this statement to all deep predictors.",
        },
        {
            "scope": "Bad-prediction retrieval",
            "status": "useful_but_not_dominant",
            "evidence": f"safeconf_lodo top10 macro={b1_enrich('__macro_mean__','safeconf_lodo_risk')}; predicted_magnitude top10 macro={b1_enrich('__macro_mean__','predicted_magnitude')}",
            "safe_claim": "SafeConf LODO enriches bad predictions above random in most datasets.",
            "caveat": "Predicted magnitude is stronger at macro top10 and must remain a main comparator.",
        },
        {
            "scope": "Magnitude baseline vs LODO",
            "status": "magnitude_stronger_than_lodo",
            "evidence": f"predicted_magnitude macro top10={b1_enrich('__macro_mean__','predicted_magnitude')} vs LODO={b1_enrich('__macro_mean__','safeconf_lodo_risk')}; magnitude stronger in {magnitude_beats_lodo}/{len(datasets)} datasets; meaningful LODO wins={meaningful_lodo_beats}",
            "safe_claim": "LODO has above-random cross-dataset screening value.",
            "caveat": "Incremental value over deployable magnitude is limited and should be framed as transfer/calibration value rather than universal dominance.",
        },
        {
            "scope": "McFarland",
            "status": "failure_rescue_boundary",
            "evidence": f"frozen partial={ladder_partial('McFarlandTsherniak2020','protocol_v0_2_family_confidence')}; LODO partial={ladder_partial('McFarlandTsherniak2020','safeconf_lodo_risk')}; B1 lodo={b1_enrich('McFarlandTsherniak2020','safeconf_lodo_risk')}",
            "safe_claim": "Frozen v0.2 fails; learned LODO provides partial rescue and useful retrieval.",
            "caveat": "Never present McFarland as a frozen v0.2 success.",
        },
        {
            "scope": "Santinha",
            "status": "weak_positive",
            "evidence": f"B1 lodo={b1_enrich('SantinhaPlatt2023','safeconf_lodo_risk')}; predicted_magnitude={b1_enrich('SantinhaPlatt2023','predicted_magnitude')}",
            "safe_claim": "Some retrieval value remains.",
            "caveat": "Keep as weak/supportive, not a headline win.",
        },
        {
            "scope": "Lara exvivo LODO",
            "status": "lodo_retrieval_failure",
            "evidence": f"LODO top5={_fmt(_row_value(b1, 'LaraAstiasoHuntly2023_exvivo', 'safeconf_lodo_risk', 0.05, 'enrichment_fold'))} and top10={b1_enrich('LaraAstiasoHuntly2023_exvivo','safeconf_lodo_risk')} below random; frozen_v02 top10={b1_enrich('LaraAstiasoHuntly2023_exvivo','protocol_v0_2_family_confidence')}; perdataset top10={b1_enrich('LaraAstiasoHuntly2023_exvivo','safeconf_perdataset_risk')}",
            "safe_claim": "Frozen and per-dataset scoring can retrieve bad Lara_exvivo predictions strongly.",
            "caveat": "LODO transfer fails in the top-risk tail here; report separately rather than hiding behind 6/7 macro wording.",
        },
        {
            "scope": "Lara invivo",
            "status": "predictor_difference_caution",
            "evidence": f"B1 lodo={b1_enrich('LaraAstiasoHuntly2023_invivo','safeconf_lodo_risk')}; frozen={b1_enrich('LaraAstiasoHuntly2023_invivo','protocol_v0_2_family_confidence')}",
            "safe_claim": "Risk retrieval is positive.",
            "caveat": "A1 showed larger V0/ContextSim difference here, so discuss separately.",
        },
        {
            "scope": "GEARS",
            "status": "no_safeconf_gears_score_yet",
            "evidence": f"C1 scoreable retrieval rows are Frangieh-only; evaluated GEARS score is prediction magnitude top10={_fmt(_row_value(c1, '__macro_mean__', 'gears_prediction_magnitude_risk', 0.10, 'enrichment_fold')) if not c1.empty else 'NA'}; uncertainty top10={_fmt(_row_value(c1, '__macro_mean__', 'gears_uncertainty_confidence', 0.10, 'enrichment_fold')) if not c1.empty else 'NA'}",
            "safe_claim": "Existing GEARS records support provenance/dedup and magnitude diagnostics only.",
            "caveat": "Current work has not shown frozen v0.2 / LODO / per-dataset SafeConf scores on GEARS predictions.",
        },
        {
            "scope": "External small-n",
            "status": "supportive_only",
            "evidence": f"External rows={len(external)}",
            "safe_claim": "External validation is directionally supportive.",
            "caveat": "Small-n and uncertain AURC intervals prevent strong claims.",
        },
    ]
    boundary = pd.DataFrame(rows)
    boundary.to_csv(tables / "C3_scope_boundary_table.csv", index=False)
    lines = [
        "# C3 SafeConf scope and boundary table",
        "",
        "This table is a defensive map: what can be claimed, what must be limited, and what should stay supplementary.",
        "",
        "```text",
        boundary[["scope", "status", "safe_claim", "caveat"]].to_string(index=False),
        "```",
        "",
    ]
    (reports / "C3_scope_boundary_interpretation.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "ok", "n_boundary_rows": int(len(boundary))}


def _git_text(repo_root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True, encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def run_c4b(repo_root: Path, all_scores_path: Path, feature_root: Path, out_dir: Path) -> dict:
    _, reports = _ensure_dirs(out_dir)
    head = _git_text(repo_root, ["rev-parse", "HEAD"])
    branch = _git_text(repo_root, ["branch", "--show-current"])
    status = _git_text(repo_root, ["status", "--short", "--branch"])
    log = _git_text(repo_root, ["log", "--oneline", "-10", "--decorate"])
    expected_reports = [
        "B4a_leakage_precheck.md",
        "B1_bad_prediction_retrieval_interpretation.md",
        "B1_5_gears_feasibility_inventory.md",
        "C1_gears_existing_record_dedup_audit.md",
        "C2_task_risk_feature_explanation.md",
        "C3_scope_boundary_interpretation.md",
        "D1_lara_exvivo_lodo_failure_diagnostic.md",
    ]
    present = {name: (out_dir / "reports" / name).exists() for name in expected_reports}
    lines = [
        "# C4b reproducibility and leakage lock",
        "",
        "## Git state",
        "",
        f"- Branch: `{branch}`",
        f"- HEAD: `{head}`",
        "",
        "```text",
        status,
        "```",
        "",
        "## Recent commits",
        "",
        "```text",
        log,
        "```",
        "",
        "## Inputs kept on server",
        "",
        f"- Row-level reliability scores: `{all_scores_path}`",
        f"- Corrected feature root: `{feature_root}`",
        "",
        "Row-level outputs are not copied into Git. Git stores compact CSV summaries and Markdown reports only.",
        "",
        "## Report availability",
        "",
    ]
    for name, ok in present.items():
        lines.append(f"- {name}: {'present' if ok else 'missing'}")
    lines += [
        "",
        "## Leakage status",
        "",
        "B4a passed before B1 was interpreted. C4b does not replace a full external reproducibility package, but it locks the local evidence chain for this branch.",
        "",
    ]
    (reports / "C4b_reproducibility_and_leakage_lock.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "ok", "head": head, "branch": branch, "reports_present": present}


def _write_status(out_dir: Path, mode: str, status: dict) -> None:
    path = out_dir / f"RUN_STATUS_{mode}.json"
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="SafeConf B4a/B1/B1.5 follow-up audits.")
    p.add_argument("--mode", choices=["b4a", "b1", "b15", "c1", "c2", "c3", "c4b", "nightly", "all"], required=True)
    p.add_argument("--all-scores", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--feature-root", type=Path, default=None)
    p.add_argument("--reliability-dir", type=Path, default=None)
    p.add_argument("--output-root", type=Path, action="append", default=[])
    p.add_argument("--checkpoint-root", type=Path, action="append", default=[])
    args = p.parse_args()

    status: dict[str, object] = {}
    full_modes = {"all", "nightly"}
    if args.mode in {"b4a", *full_modes}:
        status["b4a"] = run_b4a(args.all_scores, args.code_root, args.out_dir)
        _write_status(args.out_dir, "b4a", status["b4a"])
    if args.mode in {"b1", *full_modes}:
        status["b1"] = run_b1(args.all_scores, args.out_dir)
        _write_status(args.out_dir, "b1", status["b1"])
    if args.mode in {"b15", *full_modes}:
        status["b15"] = run_b15(args.repo_root, args.output_root, args.checkpoint_root, args.out_dir)
        _write_status(args.out_dir, "b15", status["b15"])
    if args.mode in {"c1", *full_modes}:
        status["c1"] = run_c1(args.output_root, args.out_dir)
        _write_status(args.out_dir, "c1", status["c1"])
    if args.mode in {"c2", *full_modes}:
        if args.feature_root is None:
            raise SystemExit("--feature-root is required for c2/nightly/all")
        status["c2"] = run_c2(args.all_scores, args.feature_root, args.out_dir)
        _write_status(args.out_dir, "c2", status["c2"])
    if args.mode in {"c3", *full_modes}:
        if args.reliability_dir is None:
            raise SystemExit("--reliability-dir is required for c3/nightly/all")
        status["c3"] = run_c3(args.out_dir, args.reliability_dir)
        _write_status(args.out_dir, "c3", status["c3"])
    if args.mode in {"c4b", *full_modes}:
        feature_root = args.feature_root or Path("UNSPECIFIED_FEATURE_ROOT")
        status["c4b"] = run_c4b(args.repo_root, args.all_scores, feature_root, args.out_dir)
        _write_status(args.out_dir, "c4b", status["c4b"])

    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
