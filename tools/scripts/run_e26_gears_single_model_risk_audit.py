#!/usr/bin/env python3
"""E26: single-model risk audit on the strict GEARS evidence package.

E25 made the GEARS formal outputs strict-contract compatible.  E26 asks a
narrower and honest question: with only one real predictor family (GEARS), which
single-model signals are associated with true GEARS error?

This is not a GEARS/scGPT/CPA multi-model validation.  Disagreement cannot be
computed with one predictor.  Native GEARS uncertainty is also absent in the E25
formal records, so it is reported as unavailable.
"""

from __future__ import annotations

import html
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(CODE_ROOT))

from safetrans_confidence.data.records import validate_prediction_record_artifacts
from safetrans_confidence.eval.metrics import evaluate_scores, raw_spearman


E25_DIR = PROJECT_ROOT / "docs/实验结果/E25_gears_strict_prediction_records_20260708"
OUT_DIR = PROJECT_ROOT / "docs/实验结果/E26_gears_single_model_risk_audit_20260708"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_head() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    import subprocess

    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT)
        return bool(out.decode("utf-8").strip())
    except Exception:
        return True


def _fmt_float(x: object, digits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _load_e25() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    records = pd.read_csv(E25_DIR / "tables/PREDICTION_RECORDS.csv")
    issues = validate_prediction_record_artifacts(E25_DIR, records=records, strict=True)
    with np.load(E25_DIR / "arrays/gears_predicted_effects.npz") as pred_npz:
        pred = {str(k): np.asarray(pred_npz[k], dtype=np.float32) for k in pred_npz.files}
    with np.load(E25_DIR / "arrays/gears_true_effects.npz") as true_npz:
        true = {str(k): np.asarray(true_npz[k], dtype=np.float32) for k in true_npz.files}
    return records, pred, true, issues


def _add_vector_diagnostics(records: pd.DataFrame, pred: dict[str, np.ndarray], true: dict[str, np.ndarray]) -> pd.DataFrame:
    out = records.copy()
    pred_l2 = []
    pred_abs_mean = []
    pred_nonzero_frac = []
    true_l2 = []
    true_abs_mean = []
    error_l2_over_true_l2 = []
    for row in out.to_dict("records"):
        p = pred[str(row["predicted_effect_key"])]
        t = true[str(row["true_effect_key"])]
        pred_l2.append(float(np.linalg.norm(p)))
        pred_abs_mean.append(float(np.mean(np.abs(p))))
        pred_nonzero_frac.append(float(np.mean(np.abs(p) > 1e-8)))
        t_l2 = float(np.linalg.norm(t))
        true_l2.append(t_l2)
        true_abs_mean.append(float(np.mean(np.abs(t))))
        error_l2_over_true_l2.append(float(row["true_error_rmse"]) / (t_l2 / math.sqrt(len(t)) + 1e-8))
    out["predicted_effect_l2"] = pred_l2
    out["predicted_effect_abs_mean"] = pred_abs_mean
    out["predicted_effect_nonzero_frac"] = pred_nonzero_frac
    out["true_effect_l2_diagnostic"] = true_l2
    out["true_effect_abs_mean_diagnostic"] = true_abs_mean
    out["rmse_over_true_effect_rms_diagnostic"] = error_l2_over_true_l2
    out["log1p_n_cells"] = np.log1p(pd.to_numeric(out["n_cells"], errors="coerce"))
    out["low_support_risk"] = -out["log1p_n_cells"]
    return out


def _build_score_table(records: pd.DataFrame) -> pd.DataFrame:
    score_specs = [
        ("gears_predicted_effect_l2_risk", "risk", "predicted_effect_l2", True, "GEARS predicted effect L2 norm"),
        (
            "gears_predicted_effect_abs_mean_risk",
            "risk",
            "predicted_effect_abs_mean",
            True,
            "Mean absolute predicted effect",
        ),
        ("gears_low_support_risk", "risk", "low_support_risk", True, "-log1p(n_cells)"),
        ("gears_cell_support_confidence", "confidence", "log1p_n_cells", True, "log1p(n_cells)"),
        (
            "true_effect_l2_diagnostic",
            "risk",
            "true_effect_l2_diagnostic",
            False,
            "True effect L2, diagnostic only",
        ),
        (
            "true_effect_abs_mean_diagnostic",
            "risk",
            "true_effect_abs_mean_diagnostic",
            False,
            "True effect absolute mean, diagnostic only",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for _, row in records.iterrows():
        base = {
            "record_id": row["record_id"],
            "dataset_name": row["dataset_name"],
            "dataset_family": "gears_crispr_group",
            "fold_id": int(row["fold_id"]),
            "split": row["split"],
            "context": row["context"],
            "perturbation": row["perturbation"],
            "predictor_name": row["predictor_name"],
            "true_error_rmse": float(row["true_error_rmse"]),
        }
        for score_name, score_type, col, deployable, description in score_specs:
            value = row.get(col)
            if pd.notna(value):
                rows.append(
                    {
                        **base,
                        "score_name": score_name,
                        "score_type": score_type,
                        "score_value": float(value),
                        "deployable": bool(deployable),
                        "description": description,
                    }
                )
        if "gears_uncertainty_confidence" in records.columns and pd.notna(row.get("gears_uncertainty_confidence")):
            rows.append(
                {
                    **base,
                    "score_name": "gears_native_uncertainty_confidence",
                    "score_type": "confidence",
                    "score_value": float(row["gears_uncertainty_confidence"]),
                    "deployable": True,
                    "description": "Native GEARS uncertainty confidence",
                }
            )
    return pd.DataFrame(rows)


def _residualize(y: pd.Series, covariate: pd.Series) -> pd.Series:
    y_rank = pd.to_numeric(y, errors="coerce").rank()
    x_rank = pd.to_numeric(covariate, errors="coerce").rank()
    mask = y_rank.notna() & x_rank.notna()
    resid = pd.Series(np.nan, index=y.index, dtype=float)
    if int(mask.sum()) < 4:
        return resid
    x = np.column_stack([np.ones(int(mask.sum())), x_rank[mask].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(x, y_rank[mask].to_numpy(dtype=float), rcond=None)
    resid.loc[mask] = y_rank[mask].to_numpy(dtype=float) - x @ beta
    return resid


def _partial_spearman_table(records: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    merged = scores.merge(
        records[["record_id", "true_effect_l2_diagnostic", "true_effect_abs_mean_diagnostic"]],
        on="record_id",
        how="left",
    )
    for score_name, g in merged.groupby("score_name"):
        for scope, sub in [("overall", g)] + [
            (str(ds), ds_g) for ds, ds_g in g.groupby("dataset_name")
        ]:
            if len(sub) < 4:
                rows.append(
                    {
                        "scope": scope,
                        "score_name": score_name,
                        "n": int(len(sub)),
                        "partial_spearman_control_true_l2": np.nan,
                        "partial_spearman_control_true_abs_mean": np.nan,
                    }
                )
                continue
            err_resid_l2 = _residualize(sub["true_error_rmse"], sub["true_effect_l2_diagnostic"])
            score_resid_l2 = _residualize(sub["score_value"], sub["true_effect_l2_diagnostic"])
            err_resid_abs = _residualize(sub["true_error_rmse"], sub["true_effect_abs_mean_diagnostic"])
            score_resid_abs = _residualize(sub["score_value"], sub["true_effect_abs_mean_diagnostic"])
            rows.append(
                {
                    "scope": scope,
                    "score_name": score_name,
                    "n": int(len(sub)),
                    "partial_spearman_control_true_l2": raw_spearman(score_resid_l2, err_resid_l2),
                    "partial_spearman_control_true_abs_mean": raw_spearman(score_resid_abs, err_resid_abs),
                }
            )
    return pd.DataFrame(rows)


def _top_error_enrichment(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    deploy = scores[scores["deployable"].astype(bool)].copy()
    for score_name, g in deploy.groupby("score_name"):
        for scope, sub in [("overall", g)] + [(str(ds), ds_g) for ds, ds_g in g.groupby("dataset_name")]:
            sub = sub.dropna(subset=["score_value", "true_error_rmse"]).copy()
            if len(sub) < 5:
                continue
            score_type = str(sub["score_type"].iloc[0])
            high_error_cut = sub["true_error_rmse"].quantile(0.8)
            base_rate = float((sub["true_error_rmse"] >= high_error_cut).mean())
            k = max(1, math.ceil(0.2 * len(sub)))
            if score_type == "risk":
                picked = sub.sort_values("score_value", ascending=False).head(k)
            else:
                picked = sub.sort_values("score_value", ascending=True).head(k)
            hit_rate = float((picked["true_error_rmse"] >= high_error_cut).mean())
            rows.append(
                {
                    "scope": scope,
                    "score_name": score_name,
                    "score_type": score_type,
                    "n": int(len(sub)),
                    "top_fraction": 0.2,
                    "k": int(k),
                    "high_error_threshold_p80": float(high_error_cut),
                    "base_high_error_rate": base_rate,
                    "picked_high_error_rate": hit_rate,
                    "enrichment": hit_rate / base_rate if base_rate else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _add_cov80_to_eval(eval_df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    out = eval_df.copy()
    if out.empty or scores.empty:
        return out
    rows: list[dict[str, Any]] = []
    test = scores[scores["split"].eq("test")].dropna(subset=["score_value", "true_error_rmse"]).copy()
    scopes: list[tuple[str, list[str]]] = [
        ("overall", ["score_name"]),
        ("family", ["dataset_family", "score_name"]),
        ("dataset", ["dataset_family", "dataset_name", "score_name"]),
    ]
    for level, group_cols in scopes:
        for key, g in test.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            meta = dict(zip(group_cols, key))
            score_type = str(g["score_type"].iloc[0])
            ordered = g.sort_values("score_value", ascending=(score_type == "risk"))
            keep = max(1, int(math.ceil(0.8 * len(ordered))))
            full = float(ordered["true_error_rmse"].mean())
            kept = float(ordered.head(keep)["true_error_rmse"].mean())
            rows.append(
                {
                    "level": level,
                    "dataset_family": meta.get("dataset_family", "ALL"),
                    "dataset_name": meta.get("dataset_name", "ALL"),
                    "score_name": meta.get("score_name"),
                    "risk_cov_improve_pct_filled": 100.0 * (full - kept) / full if full else np.nan,
                }
            )
    cov = pd.DataFrame(rows)
    merged = out.merge(cov, on=["level", "dataset_family", "dataset_name", "score_name"], how="left")
    if "risk_cov_improve_pct" not in merged.columns:
        merged["risk_cov_improve_pct"] = merged["risk_cov_improve_pct_filled"]
    else:
        merged["risk_cov_improve_pct"] = merged["risk_cov_improve_pct"].fillna(
            merged["risk_cov_improve_pct_filled"]
        )
    return merged.drop(columns=["risk_cov_improve_pct_filled"])


def _score_dictionary(scores: pd.DataFrame) -> pd.DataFrame:
    cols = ["score_name", "score_type", "deployable", "description"]
    return scores[cols].drop_duplicates().sort_values(["deployable", "score_name"], ascending=[False, True])


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_empty_"
    shown = df if max_rows is None else df.head(max_rows)
    cols = [str(c) for c in shown.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in shown.to_dict("records"):
        vals = []
        for col in shown.columns:
            value = row[col]
            if isinstance(value, float):
                value = _fmt_float(value, 4)
            vals.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"| ... | omitted {len(df) - max_rows} rows |" + " |" * max(0, len(cols) - 2))
    return "\n".join(lines)


def _write_svg(path: Path, eval_df: pd.DataFrame) -> None:
    deploy = eval_df[
        (eval_df["level"] == "overall")
        & eval_df["score_name"].isin(
            [
                "gears_predicted_effect_l2_risk",
                "gears_predicted_effect_abs_mean_risk",
                "gears_cell_support_confidence",
                "gears_low_support_risk",
            ]
        )
    ].copy()
    deploy = deploy.sort_values("direction_aligned_spearman", ascending=False)
    width, height = 1040, 340
    x0, y0 = 360, 92
    max_abs = max(0.05, float(deploy["direction_aligned_spearman"].abs().max())) if not deploy.empty else 1
    bars = []
    for i, row in enumerate(deploy.to_dict("records")):
        y = y0 + i * 54
        rho = float(row["direction_aligned_spearman"])
        w = int(430 * abs(rho) / max_abs)
        color = "#4677C8" if rho >= 0 else "#B95C5C"
        label = str(row["score_name"]).replace("gears_", "").replace("_risk", "").replace("_confidence", "")
        bars.append(
            f'<text x="42" y="{y+23}" font-size="16" fill="#222">{html.escape(label)}</text>'
            f'<rect x="{x0}" y="{y}" width="{w}" height="30" rx="7" fill="{color}" opacity="0.88"/>'
            f'<text x="{x0+w+12}" y="{y+21}" font-size="15" fill="#333">ρ={rho:.3f}, n={int(row["n"])}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="42" y="42" font-size="24" font-weight="700" fill="#111">E26 GEARS single-model risk audit</text>
  <text x="42" y="68" font-size="15" fill="#555">Strict GEARS records from E25; no multi-model disagreement is claimed.</text>
  {''.join(bars)}
  <line x1="42" y1="300" x2="998" y2="300" stroke="#d9dde3"/>
  <text x="42" y="322" font-size="13" fill="#666">Higher aligned Spearman means the score ranks high-error GEARS predictions earlier.</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_report(
    out_dir: Path,
    enriched_records: pd.DataFrame,
    scores: pd.DataFrame,
    eval_df: pd.DataFrame,
    risk_cov: pd.DataFrame,
    partial_df: pd.DataFrame,
    enrichment: pd.DataFrame,
    validation_issues: list[str],
) -> None:
    reports = out_dir / "reports"
    top_overall = eval_df[eval_df["level"].eq("overall")].sort_values(
        "direction_aligned_spearman", ascending=False
    )
    deployable_scores = scores[scores["deployable"].astype(bool)]["score_name"].nunique()
    unavailable_uncertainty = int(enriched_records["gears_uncertainty_confidence"].notna().sum()) == 0
    report_md = f"""# E26 GEARS single-model risk audit

生成时间：{_now()}

## 结论

- 输入：E25 strict GEARS PredictionRecord 包。
- 记录：{len(enriched_records)} 条，覆盖 {enriched_records['dataset_name'].nunique()} 个数据集。
- E25 strict validator issue_count：{len(validation_issues)}。
- 可部署单模型分数：{deployable_scores} 个。
- GEARS native uncertainty：{"不可用，E25 formal records 全为空" if unavailable_uncertainty else "可用"}。

整体上，GEARS predicted-effect magnitude 与真实误差呈正相关；这说明在 GEARS 单模型场景下，效应幅度仍是重要风险线索。该结果不能写成多模型不确定性验证，因为当前没有 scGPT/CPA 对齐输出，也没有模型间 disagreement。

## Overall score summary

{_markdown_table(top_overall[["score_name", "score_type", "n", "direction_aligned_spearman", "aurc", "random_aurc", "oracle_aurc", "risk_cov_improve_pct"]])}

## Dataset-level summary

{_markdown_table(eval_df[eval_df["level"].eq("dataset")][["dataset_name", "score_name", "n", "direction_aligned_spearman", "risk_cov_improve_pct"]], max_rows=40)}

## Partial Spearman controlling true magnitude

{_markdown_table(partial_df[partial_df["scope"].eq("overall")], max_rows=20)}

## Top-20% high-error enrichment

{_markdown_table(enrichment[enrichment["scope"].eq("overall")], max_rows=20)}

## 边界

1. E26 是 GEARS-only，不能代表 GEARS/scGPT/CPA 统一多模型验证。
2. true-effect magnitude 是诊断量，部署时不可用。
3. Dixit 只有 3 条记录，只能保留在 overall 里，不能单独强解释。
"""
    (reports / "E26_GEARS_SINGLE_MODEL_RISK_AUDIT_REPORT.md").write_text(report_md, encoding="utf-8")

    cards = [
        ("Records", str(len(enriched_records))),
        ("Datasets", str(enriched_records["dataset_name"].nunique())),
        ("Deployable scores", str(deployable_scores)),
        ("Strict issues", str(len(validation_issues))),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="k">{html.escape(v)}</div><div class="l">{html.escape(k)}</div></div>'
        for k, v in cards
    )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>E26 GEARS single-model risk audit</title>
  <style>
    body{{margin:0;background:#fff;color:#1f2933;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif;}}
    main{{max-width:1120px;margin:0 auto;padding:42px 28px 72px;}}
    h1{{font-size:30px;margin:0 0 10px;letter-spacing:-.02em;}}
    h2{{font-size:21px;margin:34px 0 12px;border-top:1px solid #e5e7eb;padding-top:24px;}}
    p,li{{line-height:1.75;font-size:16px;}}
    .lead{{color:#52606d;margin-bottom:22px;}}
    .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0;}}
    .card{{border:1px solid #e5e7eb;border-radius:14px;padding:16px;background:#fafafa;}}
    .k{{font-size:26px;font-weight:760;color:#111827;}}
    .l{{font-size:13px;color:#66788a;margin-top:4px;}}
    table{{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0 22px;}}
    th,td{{border-bottom:1px solid #e5e7eb;text-align:left;padding:9px 10px;vertical-align:top;}}
    th{{background:#f7f7f7;color:#111827;}}
    .note{{border-left:4px solid #4677C8;background:#f8fbff;padding:12px 16px;border-radius:8px;}}
    .warn{{border-left-color:#B95C5C;background:#fff8f6;}}
  </style>
</head>
<body>
<main>
  <h1>E26 GEARS single-model risk audit</h1>
  <p class="lead">基于 E25 strict GEARS 包，评估单模型可用风险线索。这里不声称多模型 disagreement。</p>
  <div class="cards">{cards_html}</div>
  <p class="note">核心观察：GEARS predicted-effect magnitude 与真实误差呈正相关；true magnitude 更强但只可诊断，不可部署。</p>
  <p class="note warn">GEARS native uncertainty 在 E25 formal records 中不可用；scGPT/CPA 仍需 adapter。</p>
  <img src="../figures/gears_single_model_risk_audit.svg" alt="E26 summary" style="width:100%;max-width:1040px;border:1px solid #e5e7eb;border-radius:14px;margin:20px 0;"/>
  <h2>Overall score summary</h2>
  {top_overall.to_html(index=False, escape=False)}
  <h2>Top-20% high-error enrichment</h2>
  {enrichment[enrichment["scope"].eq("overall")].to_html(index=False, escape=False)}
  <h2>Partial Spearman controlling true magnitude</h2>
  {partial_df[partial_df["scope"].eq("overall")].to_html(index=False, escape=False)}
  <h2>Risk-coverage sample</h2>
  {risk_cov.head(40).to_html(index=False, escape=False)}
</main>
</body>
</html>
"""
    (reports / "E26_GEARS_SINGLE_MODEL_RISK_AUDIT.html").write_text(page, encoding="utf-8")


def _write_readme(out_dir: Path, status: dict[str, Any]) -> None:
    text = f"""# E26 GEARS 单模型风险审计

先看结论：E26 在 E25 strict GEARS 包上做单模型风险分析。它能说明 GEARS 自身哪些线索和误差相关，不能说明 GEARS/scGPT/CPA 三模型统一验证已经完成。

## 关键数字

- PredictionRecords：{status['n_records']}
- 数据集：{status['n_datasets']}
- 可部署分数：{status['n_deployable_scores']}
- E25 strict issue：{status['strict_issue_count']}
- GEARS native uncertainty：{status['native_uncertainty_status']}

## 文件

- `tables/GEARS_SINGLE_MODEL_ENRICHED_RECORDS.csv`
- `tables/GEARS_SINGLE_MODEL_SCORES.csv`
- `tables/GEARS_SINGLE_MODEL_EVAL_SUMMARY.csv`
- `tables/GEARS_SINGLE_MODEL_PARTIAL_SPEARMAN.csv`
- `reports/E26_GEARS_SINGLE_MODEL_RISK_AUDIT.html`

## 下一步

如果要冲更高投稿等级，E26 后面应该接 scGPT 或 CPA adapter，而不是继续只在 GEARS-only 上打磨。
"""
    (out_dir / "README_先看这个.md").write_text(text, encoding="utf-8")


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for rel in ["tables", "reports", "figures"]:
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)

    records, pred, true, strict_issues = _load_e25()
    enriched = _add_vector_diagnostics(records, pred, true)
    scores = _build_score_table(enriched)
    eval_df, risk_cov = evaluate_scores(scores)
    eval_df = _add_cov80_to_eval(eval_df, scores)
    partial_df = _partial_spearman_table(enriched, scores)
    enrichment = _top_error_enrichment(scores)
    dictionary = _score_dictionary(scores)
    dataset_summary = (
        enriched.groupby("dataset_name", as_index=False)
        .agg(
            n_records=("record_id", "count"),
            n_runs=("fold_id", "nunique"),
            mean_rmse=("true_error_rmse", "mean"),
            median_rmse=("true_error_rmse", "median"),
            mean_predicted_effect_l2=("predicted_effect_l2", "mean"),
            mean_true_effect_l2_diagnostic=("true_effect_l2_diagnostic", "mean"),
            mean_n_cells=("n_cells", "mean"),
        )
        .sort_values("dataset_name")
    )

    enriched.to_csv(OUT_DIR / "tables/GEARS_SINGLE_MODEL_ENRICHED_RECORDS.csv", index=False)
    scores.to_csv(OUT_DIR / "tables/GEARS_SINGLE_MODEL_SCORES.csv", index=False)
    eval_df.to_csv(OUT_DIR / "tables/GEARS_SINGLE_MODEL_EVAL_SUMMARY.csv", index=False)
    risk_cov.to_csv(OUT_DIR / "tables/GEARS_SINGLE_MODEL_RISK_COVERAGE.csv", index=False)
    partial_df.to_csv(OUT_DIR / "tables/GEARS_SINGLE_MODEL_PARTIAL_SPEARMAN.csv", index=False)
    enrichment.to_csv(OUT_DIR / "tables/GEARS_SINGLE_MODEL_TOP20_ENRICHMENT.csv", index=False)
    dictionary.to_csv(OUT_DIR / "tables/GEARS_SINGLE_MODEL_SCORE_DICTIONARY.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "tables/GEARS_SINGLE_MODEL_DATASET_SUMMARY.csv", index=False)

    _write_svg(OUT_DIR / "figures/gears_single_model_risk_audit.svg", eval_df)
    _write_report(OUT_DIR, enriched, scores, eval_df, risk_cov, partial_df, enrichment, strict_issues)
    native_available = bool(scores["score_name"].eq("gears_native_uncertainty_confidence").any())
    status = {
        "status": "ok",
        "generated_at": _now(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "input": os.path.relpath(E25_DIR, PROJECT_ROOT),
        "out": os.path.relpath(OUT_DIR, PROJECT_ROOT),
        "n_records": int(len(enriched)),
        "n_datasets": int(enriched["dataset_name"].nunique()),
        "n_scores": int(len(scores)),
        "n_deployable_scores": int(dictionary[dictionary["deployable"].astype(bool)]["score_name"].nunique()),
        "strict_issue_count": int(len(strict_issues)),
        "strict_issues": strict_issues,
        "native_uncertainty_status": "available" if native_available else "absent_in_e25_formal_records",
    }
    (OUT_DIR / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_readme(OUT_DIR, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
