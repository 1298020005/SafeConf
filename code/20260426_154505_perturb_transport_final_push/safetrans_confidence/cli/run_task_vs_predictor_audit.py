#!/usr/bin/env python3
"""A0/A1 audit for SafeConf task risk versus predictor-specific error.

A0 summarizes the corrected formal evidence that is already exported to Git.
A1 answers a narrower question raised after the teacher discussion:

    On the same task, do V0StrongBaseline and ContextSimBaseline tend to fail
    together, or is the error mostly predictor-specific?

The script intentionally writes only compact summary artifacts. Row-level
PredictionRecord outputs remain in server outputs and are not committed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


V0_NAME = "V0StrongBaseline"
CONTEXTSIM_NAME = "ContextSimBaseline"

A1_PAIRED_COLUMNS = [
    "dataset_name",
    "fold_id",
    "task_key",
    "context",
    "perturbation",
    "v0_error_rmse",
    "contextsim_error_rmse",
    "error_diff_v0_minus_contextsim",
    "abs_error_diff",
]

A1_SUMMARY_COLUMNS = [
    "dataset_name",
    "n_paired_tasks",
    "spearman_v0_vs_contextsim_error",
    "pearson_v0_vs_contextsim_error",
    "mean_v0_error",
    "mean_contextsim_error",
    "mean_error_diff_v0_minus_contextsim",
    "median_abs_error_diff",
    "p90_abs_error_diff",
    "cohens_d_paired_diff",
    "ss_total",
    "ss_task",
    "ss_predictor",
    "ss_residual",
    "frac_task",
    "frac_predictor",
    "frac_residual",
    "interpretation_flag",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt(value: object, digits: int = 3) -> str:
    x = _num(value)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _evidence_label(score_name: str, row: pd.Series) -> str:
    partial = _num(row.get("partial_rho_control_magnitude"))
    reduction_low = _num(row.get("reduction_ci_low"))
    reduction_high = _num(row.get("reduction_ci_high"))
    if score_name == "protocol_v0_2_family_confidence":
        if partial < 0:
            return "failure_boundary"
        if partial < 0.25:
            return "weak_positive"
        return "core_positive"
    if score_name in {"safeconf_lodo_risk", "safeconf_perdataset_risk"}:
        if partial > 0 and reduction_low > 0:
            return "strong_transfer_positive"
        if partial > 0:
            return "positive_but_auc_ci_uncertain"
        return "not_supported"
    if partial > 0 and reduction_low > 0:
        return "supportive_positive"
    if partial > 0:
        return "supportive_but_uncertain"
    return "not_supported"


def build_a0_evidence_map(formal_dir: Path, reliability_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    formal = _read_csv(formal_dir / "tables" / "FORMAL_MAIN_TABLE.csv")
    ladder = _read_csv(reliability_dir / "tables" / "RELIABILITY_BASELINE_LADDER.csv")
    lopo_pert = _read_csv(reliability_dir / "tables" / "LOPO_PERTMEAN_RESULT.csv")
    lopo_knn = _read_csv(reliability_dir / "tables" / "LOPO_CONTROLKNN_RESULT.csv")
    external = _read_csv(reliability_dir / "tables" / "EXTERNAL_VALIDATION_RESULT.csv")

    for _, r in formal.iterrows():
        rows.append(
            {
                "evidence_family": "corrected_7main_frozen_v0_2",
                "dataset_name": r.get("dataset_name"),
                "score_or_probe": r.get("score_name", "protocol_v0_2_family_confidence"),
                "n": r.get("n"),
                "aligned_rho": r.get("aligned_rho"),
                "partial_rho_control_magnitude": r.get("partial_rho_control_magnitude"),
                "aurc_reduction_vs_random_pct": r.get("rc80_improve_pct", r.get("aurc_reduction_vs_random_pct")),
                "evidence_label": _evidence_label("protocol_v0_2_family_confidence", r),
                "interpretation": "Frozen v0.2 baseline; interpretable, but not the learned transfer layer.",
            }
        )

    for score in ["predicted_magnitude", "protocol_v0_2_family_confidence", "safeconf_lodo_risk"]:
        sub = ladder[ladder.get("score_name", pd.Series(dtype=str)).astype(str).eq(score)]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "evidence_family": "corrected_reliability_ladder",
                    "dataset_name": r.get("dataset_name"),
                    "score_or_probe": score,
                    "n": r.get("n"),
                    "aligned_rho": r.get("aligned_rho"),
                    "partial_rho_control_magnitude": r.get("partial_rho_control_magnitude"),
                    "aurc_reduction_vs_random_pct": r.get("aurc_reduction_vs_random_pct"),
                    "evidence_label": _evidence_label(score, r),
                    "interpretation": (
                        "LODO is the main learned transfer claim."
                        if score == "safeconf_lodo_risk"
                        else "Baseline comparator that should be reported honestly."
                    ),
                }
            )

    for name, frame in [("PertMeanPredictor", lopo_pert), ("ControlKNNPredictor", lopo_knn)]:
        for _, r in frame.iterrows():
            rows.append(
                {
                    "evidence_family": "lopo_unseen_predictor",
                    "dataset_name": r.get("dataset_name"),
                    "score_or_probe": name,
                    "n": r.get("n"),
                    "aligned_rho": r.get("aligned_rho"),
                    "partial_rho_control_magnitude": r.get("partial_rho_control_magnitude"),
                    "aurc_reduction_vs_random_pct": r.get("aurc_reduction_vs_random_pct"),
                    "evidence_label": _evidence_label("safeconf_lodo_risk", r),
                    "interpretation": "Unseen-predictor probe; supports predictor transfer, not a claim of full model independence.",
                }
            )

    for _, r in external.iterrows():
        rows.append(
            {
                "evidence_family": "external_validation",
                "dataset_name": r.get("dataset_name"),
                "score_or_probe": "safeconf_external_risk",
                "n": r.get("n"),
                "aligned_rho": r.get("aligned_rho"),
                "partial_rho_control_magnitude": r.get("partial_rho_control_magnitude"),
                "aurc_reduction_vs_random_pct": r.get("aurc_reduction_vs_random_pct"),
                "evidence_label": _evidence_label("external", r),
                "interpretation": "External-study support; small-n datasets must be described cautiously.",
            }
        )

    return pd.DataFrame(rows)


def _score_summary(evidence: pd.DataFrame, family: str, score: str | None = None) -> pd.DataFrame:
    sub = evidence[evidence["evidence_family"].eq(family)].copy()
    if score is not None:
        sub = sub[sub["score_or_probe"].eq(score)]
    return sub


def write_a0_review(evidence: pd.DataFrame, out_path: Path) -> None:
    frozen = _score_summary(evidence, "corrected_7main_frozen_v0_2")
    lodo = _score_summary(evidence, "corrected_reliability_ladder", "safeconf_lodo_risk")
    lopo = _score_summary(evidence, "lopo_unseen_predictor")
    external = _score_summary(evidence, "external_validation")

    def _line_for(frame: pd.DataFrame, dataset: str, col: str) -> str:
        hit = frame[frame["dataset_name"].astype(str).eq(dataset)]
        if hit.empty:
            return "NA"
        return _fmt(hit.iloc[0].get(col))

    mcf_frozen = _line_for(frozen, "McFarlandTsherniak2020", "partial_rho_control_magnitude")
    mcf_lodo = _line_for(lodo, "McFarlandTsherniak2020", "partial_rho_control_magnitude")
    lodo_pos = int((_score_summary(evidence, "corrected_reliability_ladder", "safeconf_lodo_risk")[
        "partial_rho_control_magnitude"
    ].astype(float) > 0).sum()) if not lodo.empty else 0
    lodo_total = int(len(lodo))
    lopo_pos = int((lopo["partial_rho_control_magnitude"].astype(float) > 0).sum()) if not lopo.empty else 0
    lopo_total = int(len(lopo))
    external_pos = int((external["partial_rho_control_magnitude"].astype(float) > 0).sum()) if not external.empty else 0
    external_total = int(len(external))

    text = f"""# A0 existing evidence review

Date: 2026-06-11

## What is already answered

- Corrected seven-main frozen v0.2 remains the interpretable baseline. It is useful, but it is not the strongest current method layer.
- McFarland is still a frozen-v0.2 failure boundary: frozen partial rho = {mcf_frozen}.
- The learned LODO reliability layer rescues McFarland on the corrected run: LODO partial rho = {mcf_lodo}.
- LODO is positive in {lodo_pos}/{lodo_total} corrected main datasets. This is the cleanest dataset-transfer evidence.
- LOPO unseen-predictor probes are positive in {lopo_pos}/{lopo_total} dataset-by-predictor checks. This supports predictor transfer, while still sharing task-level features with the training predictors.
- External validation has positive partial rho in {external_pos}/{external_total} studies, but several external AURC intervals are uncertain; write it as supportive evidence, not as four conclusive wins.

## What A0 does not answer

A0 does not directly say whether V0StrongBaseline and ContextSimBaseline fail on the same tasks. That requires A1: a paired same-task error audit.

## Working interpretation before A1

SafeConf should currently be described as an external task-risk / prediction-risk protocol with a learned reliability layer. Avoid claiming that frozen v0.2 alone is a complete model-specific reliability evaluator.
"""
    out_path.write_text(text, encoding="utf-8")


def _load_a1_source(all_scores: Path | None, run_dirs: list[Path]) -> tuple[pd.DataFrame, str]:
    if all_scores is not None and all_scores.exists():
        src = pd.read_csv(all_scores)
        if "score_name" in src.columns:
            preferred = src[src["score_name"].astype(str).eq("predicted_magnitude")].copy()
            if not preferred.empty:
                return preferred, str(all_scores)
        return src, str(all_scores)
    if run_dirs:
        from safetrans_confidence.cli.run_safeconf_reliability_model import load_corrected_base

        base, _ = load_corrected_base(run_dirs)
        return base, "run_dirs"
    return pd.DataFrame(), "missing"


def _task_id_columns(frame: pd.DataFrame) -> list[str]:
    if "task_key" in frame.columns and frame["task_key"].notna().any():
        return ["dataset_name", "fold_id", "task_key"]
    cols = ["dataset_name", "fold_id", "context", "perturbation"]
    return [c for c in cols if c in frame.columns]


def build_paired_predictor_errors(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame()
    test = source[source["split"].astype(str).eq("test")].copy()
    needed = {"dataset_name", "fold_id", "predictor_name", "true_error_rmse"}
    missing = needed.difference(test.columns)
    if missing:
        raise ValueError("A1 source is missing required columns: " + ",".join(sorted(missing)))
    test = test[test["predictor_name"].isin([V0_NAME, CONTEXTSIM_NAME])].copy()
    id_cols = _task_id_columns(test)
    keep_cols = id_cols + ["predictor_name", "true_error_rmse"]
    for c in ["context", "perturbation"]:
        if c in test.columns and c not in keep_cols:
            keep_cols.append(c)
    slim = test[keep_cols].dropna(subset=["true_error_rmse"]).copy()
    value_cols = id_cols + ["predictor_name"]
    slim = slim.drop_duplicates(value_cols, keep="first")
    wide = slim.pivot_table(
        index=id_cols,
        columns="predictor_name",
        values="true_error_rmse",
        aggfunc="first",
    ).reset_index()
    if V0_NAME not in wide.columns or CONTEXTSIM_NAME not in wide.columns:
        return pd.DataFrame()
    wide = wide.dropna(subset=[V0_NAME, CONTEXTSIM_NAME]).copy()
    meta_cols = [c for c in ["context", "perturbation"] if c in slim.columns and c not in id_cols]
    if meta_cols:
        meta = slim[id_cols + meta_cols].drop_duplicates(id_cols, keep="first")
        wide = wide.merge(meta, on=id_cols, how="left")
    wide = wide.rename(
        columns={
            V0_NAME: "v0_error_rmse",
            CONTEXTSIM_NAME: "contextsim_error_rmse",
        }
    )
    wide["error_diff_v0_minus_contextsim"] = wide["v0_error_rmse"] - wide["contextsim_error_rmse"]
    wide["abs_error_diff"] = wide["error_diff_v0_minus_contextsim"].abs()
    return wide


def _variance_decomposition(v0: np.ndarray, cs: np.ndarray) -> dict:
    y = np.column_stack([v0, cs]).astype(float)
    y = y[np.isfinite(y).all(axis=1)]
    if y.shape[0] < 2:
        return {
            "ss_total": np.nan,
            "ss_task": np.nan,
            "ss_predictor": np.nan,
            "ss_residual": np.nan,
            "frac_task": np.nan,
            "frac_predictor": np.nan,
            "frac_residual": np.nan,
        }
    task_means = y.mean(axis=1)
    predictor_means = y.mean(axis=0)
    grand = float(y.mean())
    p = y.shape[1]
    t = y.shape[0]
    ss_total = float(((y - grand) ** 2).sum())
    ss_task = float(p * ((task_means - grand) ** 2).sum())
    ss_predictor = float(t * ((predictor_means - grand) ** 2).sum())
    ss_residual = float(((y - task_means[:, None] - predictor_means[None, :] + grand) ** 2).sum())
    denom = ss_total if ss_total > 0 else np.nan
    return {
        "ss_total": ss_total,
        "ss_task": ss_task,
        "ss_predictor": ss_predictor,
        "ss_residual": ss_residual,
        "frac_task": ss_task / denom,
        "frac_predictor": ss_predictor / denom,
        "frac_residual": ss_residual / denom,
    }


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return np.nan
    x = a[mask].astype(float)
    y = b[mask].astype(float)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return np.nan
    x_rank = pd.Series(a[mask]).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(b[mask]).rank(method="average").to_numpy(dtype=float)
    return _pearson_corr(x_rank, y_rank)


def _paired_summary(name: str, frame: pd.DataFrame) -> dict:
    v0 = pd.to_numeric(frame["v0_error_rmse"], errors="coerce").to_numpy(dtype=float)
    cs = pd.to_numeric(frame["contextsim_error_rmse"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v0) & np.isfinite(cs)
    v0, cs = v0[mask], cs[mask]
    diff = v0 - cs
    abs_diff = np.abs(diff)
    if len(diff) == 0:
        return {"dataset_name": name, "n_paired_tasks": 0}
    diff_std = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    var = _variance_decomposition(v0, cs)
    row = {
        "dataset_name": name,
        "n_paired_tasks": int(len(diff)),
        "spearman_v0_vs_contextsim_error": _spearman_corr(v0, cs),
        "pearson_v0_vs_contextsim_error": _pearson_corr(v0, cs),
        "mean_v0_error": float(np.mean(v0)),
        "mean_contextsim_error": float(np.mean(cs)),
        "mean_error_diff_v0_minus_contextsim": float(np.mean(diff)),
        "median_abs_error_diff": float(np.median(abs_diff)),
        "p90_abs_error_diff": float(np.quantile(abs_diff, 0.90)),
        "cohens_d_paired_diff": float(np.mean(diff) / diff_std) if diff_std and np.isfinite(diff_std) else np.nan,
    }
    row.update(var)
    row["interpretation_flag"] = _interpret_a1_row(row)
    return row


def _interpret_a1_row(row: dict) -> str:
    frac_task = _num(row.get("frac_task"))
    frac_predictor = _num(row.get("frac_predictor"))
    spearman = _num(row.get("spearman_v0_vs_contextsim_error"))
    d = abs(_num(row.get("cohens_d_paired_diff")))
    if frac_task >= 0.80 and spearman > 0.50:
        return "task_difficulty_dominant"
    if frac_predictor >= 0.10 or d >= 0.50:
        return "predictor_difference_nontrivial"
    if spearman > 0:
        return "mixed_but_positive_task_signal"
    return "dataset_specific_or_unclear"


def build_a1_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rows = []
    for ds, sub in paired.groupby("dataset_name", dropna=False):
        rows.append(_paired_summary(str(ds), sub))
    rows.append(_paired_summary("__overall__", paired))
    return pd.DataFrame(rows)


def write_a1_interpretation(summary: pd.DataFrame, out_path: Path, source_name: str) -> None:
    if summary.empty:
        text = f"""# A1 task-versus-predictor audit

Status: not run

Reason: no row-level A1 source was available.

Expected source on server:

`outputs/safeconf_reliability_model_corrected_20260610/tables/RELIABILITY_ALL_SCORES.csv`

If that file is missing, regenerate it with `run_safeconf_reliability_model.py`
from the corrected seven-main run directories, then re-run this audit.
"""
        out_path.write_text(text, encoding="utf-8")
        return

    overall = summary[summary["dataset_name"].eq("__overall__")]
    overall_row = overall.iloc[0] if not overall.empty else summary.iloc[-1]
    n_task_dom = int(summary[~summary["dataset_name"].eq("__overall__")]["interpretation_flag"].eq("task_difficulty_dominant").sum())
    n_pred = int(summary[~summary["dataset_name"].eq("__overall__")]["interpretation_flag"].eq("predictor_difference_nontrivial").sum())
    total = int(len(summary[~summary["dataset_name"].eq("__overall__")]))
    if n_task_dom >= max(1, math.ceil(total / 2)):
        decision = "Task difficulty is the leading narrative."
    elif n_pred >= 3:
        decision = "Predictor-specific differences are large enough to trigger A2."
    else:
        decision = "The result is mixed; keep task-risk framing but avoid overclaiming model-specific reliability."

    text = f"""# A1 task-versus-predictor audit

Source: `{source_name}`

## One-line decision

{decision}

## Overall summary

- Paired test tasks: {int(overall_row.get("n_paired_tasks", 0))}
- Spearman(V0 error, ContextSim error): {_fmt(overall_row.get("spearman_v0_vs_contextsim_error"))}
- Pearson(V0 error, ContextSim error): {_fmt(overall_row.get("pearson_v0_vs_contextsim_error"))}
- Task variance fraction: {_fmt(overall_row.get("frac_task"))}
- Predictor variance fraction: {_fmt(overall_row.get("frac_predictor"))}
- Paired Cohen's d: {_fmt(overall_row.get("cohens_d_paired_diff"))}

## Dataset-level count

- Task-difficulty dominant: {n_task_dom}/{total}
- Predictor-difference nontrivial: {n_pred}/{total}

## Interpretation rule

- If task variance dominates in most datasets and predictor errors are strongly correlated, SafeConf should be framed as predictor-agnostic task-risk scoring.
- If predictor variance or paired Cohen's d is large in several datasets, start A2 as a minimal model-aware extension.
"""
    out_path.write_text(text, encoding="utf-8")


def run(
    formal_dir: Path,
    reliability_dir: Path,
    out_dir: Path,
    all_scores: Path | None,
    run_dirs: list[Path],
) -> dict:
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    evidence = build_a0_evidence_map(formal_dir, reliability_dir)
    evidence.to_csv(tables / "A0_existing_evidence_map.csv", index=False)
    write_a0_review(evidence, reports / "A0_existing_evidence_review.md")

    source, source_name = _load_a1_source(all_scores, run_dirs)
    paired = build_paired_predictor_errors(source) if not source.empty else pd.DataFrame()
    summary = build_a1_summary(paired)
    if not paired.empty:
        paired.to_csv(tables / "A1_paired_predictor_error_table.csv", index=False)
    else:
        pd.DataFrame(columns=A1_PAIRED_COLUMNS).to_csv(
            tables / "A1_paired_predictor_error_table.csv", index=False
        )
    if not summary.empty:
        summary.to_csv(tables / "A1_task_vs_predictor_variance_summary.csv", index=False)
    else:
        pd.DataFrame(columns=A1_SUMMARY_COLUMNS).to_csv(
            tables / "A1_task_vs_predictor_variance_summary.csv", index=False
        )
    write_a1_interpretation(summary, reports / "A1_task_risk_interpretation.md", source_name)

    status = {
        "formal_dir": str(formal_dir),
        "reliability_dir": str(reliability_dir),
        "out_dir": str(out_dir),
        "a0_rows": int(len(evidence)),
        "a1_source": source_name,
        "a1_paired_rows": int(len(paired)),
        "a1_summary_rows": int(len(summary)),
        "status": "ok" if len(evidence) else "missing_a0_inputs",
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return status


def main() -> None:
    p = argparse.ArgumentParser(description="SafeConf A0/A1 task-risk audit.")
    p.add_argument("--formal-dir", type=Path, required=True)
    p.add_argument("--reliability-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--all-scores", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, action="append", dest="run_dirs", default=[])
    args = p.parse_args()
    print(
        json.dumps(
            run(args.formal_dir, args.reliability_dir, args.out_dir, args.all_scores, args.run_dirs),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
