#!/usr/bin/env python3
"""Build a CuiHacohen2023 go/no-go report from SafeConf outputs.

This script intentionally does not tune formulas or train predictors.  It only
summarizes test-set confidence/risk scores, per-fold stability, and effect
magnitude confounding for the first large dataset gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_SCORE = "protocol_v0_2_family_confidence"
SINGLE_FEATURE_SCORES = {
    "context_similarity_score",
    "support_count_score",
    "model_disagreement_risk",
    "historical_residual_risk",
    "ood_distance_risk",
    "prediction_magnitude_risk",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _raw_spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def _rank_residual(values: pd.Series, control: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"v": values, "c": control}).apply(pd.to_numeric, errors="coerce").dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(frame) < 3 or frame["c"].nunique() < 2:
        return out
    y = frame["v"].rank(method="average").to_numpy(dtype=float)
    z = frame["c"].rank(method="average").to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    out.loc[frame.index] = y - x @ beta
    return out


def _partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    rx = _rank_residual(x, control)
    ry = _rank_residual(y, control)
    mask = rx.notna() & ry.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(rx[mask].corr(ry[mask], method="pearson"))


def _risk_axis(frame: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(frame["score_value"], errors="coerce")
    score_type = frame["score_type"].astype(str)
    return score.where(score_type.eq("risk"), -score)


def _load_effect_magnitude(records: pd.DataFrame, true_effect_npz: Path) -> pd.DataFrame:
    if not true_effect_npz.exists():
        raise FileNotFoundError(true_effect_npz)
    arrays = np.load(true_effect_npz)
    rows: list[dict] = []
    for _, row in records.iterrows():
        key = str(row["true_effect_key"])
        arr = np.asarray(arrays[key], dtype=float).ravel() if key in arrays else None
        if arr is None:
            rows.append(
                {
                    "record_id": row["record_id"],
                    "true_effect_l2_norm": np.nan,
                    "true_effect_abs_mean": np.nan,
                    "effect_scale_rmse": np.nan,
                    "true_effect_key_found": False,
                }
            )
        else:
            rows.append(
                {
                    "record_id": row["record_id"],
                    "true_effect_l2_norm": float(np.linalg.norm(arr)),
                    "true_effect_abs_mean": float(np.mean(np.abs(arr))),
                    "effect_scale_rmse": float(np.linalg.norm(arr) / np.sqrt(max(arr.size, 1))),
                    "true_effect_key_found": True,
                }
            )
    return pd.DataFrame(rows)


def _assign_normalized_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["normalized_rmse"] = np.nan
    out["normalization_eps"] = np.nan
    for _, idx_obj in out.groupby(["dataset_name", "predictor_name"], dropna=False).groups.items():
        idx = list(idx_obj)
        sub = out.loc[idx]
        ref = pd.to_numeric(
            sub[sub["split"].isin(["train", "val"])]["effect_scale_rmse"], errors="coerce"
        )
        ref = ref[(ref > 0) & np.isfinite(ref)]
        eps = float(ref.quantile(0.01)) if not ref.empty else 1e-8
        if not np.isfinite(eps) or eps <= 0:
            eps = 1e-8
        denom = pd.to_numeric(sub["effect_scale_rmse"], errors="coerce").clip(lower=eps)
        out.loc[idx, "normalization_eps"] = eps
        out.loc[idx, "normalized_rmse"] = pd.to_numeric(sub["true_error_rmse"], errors="coerce") / denom
    return out


def _risk_coverage80(group: pd.DataFrame) -> float:
    valid = group.dropna(subset=["risk_axis", "true_error_rmse"])
    if len(valid) < 3:
        return float("nan")
    full = float(valid["true_error_rmse"].mean())
    keep = max(1, int(np.ceil(0.8 * len(valid))))
    kept = valid.sort_values("risk_axis", ascending=True).head(keep)
    kept_mean = float(kept["true_error_rmse"].mean())
    return 100.0 * (full - kept_mean) / full if full else float("nan")


def _summarize_group(group: pd.DataFrame) -> dict:
    return {
        "n": int(len(group)),
        "aligned_rho": _raw_spearman(group["risk_axis"], group["true_error_rmse"]),
        "raw_score_vs_rmse_rho": _raw_spearman(group["score_value"], group["true_error_rmse"]),
        "normalized_rmse_rho": _raw_spearman(group["risk_axis"], group["normalized_rmse"]),
        "magnitude_only_rho": _raw_spearman(group["true_effect_l2_norm"], group["true_error_rmse"]),
        "partial_rho_control_magnitude": _partial_spearman(
            group["risk_axis"], group["true_error_rmse"], group["true_effect_l2_norm"]
        ),
        "risk_coverage80_improve_pct": _risk_coverage80(group),
        "mean_rmse": float(pd.to_numeric(group["true_error_rmse"], errors="coerce").mean()),
    }


def _load_scores(mvp_dir: Path, protocol_dir: Path | None) -> pd.DataFrame:
    mvp_scores = _read_csv(mvp_dir / "tables" / "CONFIDENCE_SCORES.csv")
    pieces = [mvp_scores]
    if protocol_dir is not None:
        proto_path = protocol_dir / "tables" / "CONFIDENCE_SCORES.csv"
        if proto_path.exists():
            proto = pd.read_csv(proto_path)
            proto = proto[proto["score_name"].astype(str).str.startswith("protocol_v0_2")].copy()
            pieces.append(proto)
    return pd.concat(pieces, ignore_index=True)


def load_scored_records(mvp_dir: Path, protocol_dir: Path | None) -> pd.DataFrame:
    records = _read_csv(mvp_dir / "tables" / "PREDICTION_RECORDS.csv")
    scores = _load_scores(mvp_dir, protocol_dir)
    magnitudes = _load_effect_magnitude(records, mvp_dir / "input" / "true_effects.npz")

    record_cols = [
        "record_id",
        "task_id",
        "task_key",
        "dataset_name",
        "fold_id",
        "split",
        "context",
        "perturbation",
        "predictor_name",
        "true_effect_key",
    ]
    keep = [col for col in record_cols if col in records.columns]
    merged = scores.merge(
        records[keep].drop_duplicates("record_id"),
        on=["record_id", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name"],
        how="left",
        suffixes=("", "_record"),
    )
    merged = merged.merge(magnitudes, on="record_id", how="left")
    merged["risk_axis"] = _risk_axis(merged)
    merged = _assign_normalized_rmse(merged)
    return merged


def _score_summary(test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for score_name, group in test.groupby("score_name", dropna=False):
        score_type = str(group["score_type"].iloc[0])
        row = {
            "score_name": score_name,
            "score_type": score_type,
            **_summarize_group(group),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("aligned_rho", ascending=False)


def _per_fold_summary(test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (score_name, fold_id), group in test.groupby(["score_name", "fold_id"], dropna=False):
        row = {
            "score_name": score_name,
            "fold_id": int(fold_id),
            **_summarize_group(group),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["score_name", "fold_id"])


def _per_predictor_summary(test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (score_name, predictor), group in test.groupby(["score_name", "predictor_name"], dropna=False):
        row = {
            "score_name": score_name,
            "predictor_name": predictor,
            **_summarize_group(group),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["score_name", "predictor_name"])


def _decision(main_row: pd.Series | None, best_single: pd.Series | None) -> tuple[str, str]:
    if main_row is None:
        return "NO_MAIN_SCORE", f"缺少 {MAIN_SCORE}，不能按冻结 protocol v0.2 判断。"
    rho = float(main_row["aligned_rho"])
    partial = float(main_row["partial_rho_control_magnitude"])
    if rho > 0.30:
        return "GO", "冻结 protocol v0.2 在 Cui 上超过 0.30，可以全速推进下一批大数据集。"
    if rho >= 0.20:
        return "GO_WITH_MAGNITUDE_AUDIT", "总相关在 0.20-0.30，需要重点看 partial rho，排除效应大小混杂。"
    if rho >= 0.10:
        return "CAUTION_FEATURE_DIAG", "总相关在 0.10-0.20，需要看单特征是否有稳定信号。"
    if best_single is not None and float(best_single["aligned_rho"]) >= 0.20:
        return (
            "V02_WEAK_BUT_SINGLE_FEATURE_SIGNAL",
            "v0.2 原公式偏弱，但单特征有信号；只允许在 fold-val 上探索扩展公式，再 blind 到其他数据集。",
        )
    if np.isfinite(partial) and partial > 0.10:
        return (
            "V02_RAW_WEAK_BUT_PARTIAL_SIGNAL",
            "v0.2 原始相关弱，但控制效应大小后仍有信号；需要先诊断公式，不应直接换题。",
        )
    return "STOP_AND_DIAGNOSE", "v0.2 和单特征都没有足够信号，应先分析这个 task 在 Cui 上是否本身不可分。"


def write_report(
    out_dir: Path,
    score_summary: pd.DataFrame,
    per_fold: pd.DataFrame,
    per_predictor: pd.DataFrame,
    status: dict,
) -> None:
    main = score_summary[score_summary["score_name"].eq(MAIN_SCORE)]
    main_row = main.iloc[0] if not main.empty else None
    singles = score_summary[score_summary["score_name"].isin(SINGLE_FEATURE_SCORES)].copy()
    best_single = singles.iloc[0] if not singles.empty else None
    decision, reason = _decision(main_row, best_single)
    status["decision"] = decision
    status["decision_reason"] = reason

    def _fmt(v: object) -> str:
        try:
            val = float(v)
        except (TypeError, ValueError):
            return "NA"
        return "NA" if not np.isfinite(val) else f"{val:.4f}"

    main_lines = []
    if main_row is not None:
        main_lines = [
            f"- protocol v0.2 aligned rho（方向对齐相关）: {_fmt(main_row['aligned_rho'])}",
            f"- partial rho control magnitude（控制效应大小后的相关）: {_fmt(main_row['partial_rho_control_magnitude'])}",
            f"- magnitude-only rho（只看效应大小的基线）: {_fmt(main_row['magnitude_only_rho'])}",
            f"- risk-coverage@80% improvement（保留 80% 低风险预测后的误差改善）: {_fmt(main_row['risk_coverage80_improve_pct'])}%",
        ]
    else:
        main_lines = [f"- 未找到主分数 `{MAIN_SCORE}`。"]

    best_lines = []
    if best_single is not None:
        best_lines = [
            f"- best single feature（最好单特征）: {best_single['score_name']}",
            f"- best single aligned rho: {_fmt(best_single['aligned_rho'])}",
            f"- best single partial rho: {_fmt(best_single['partial_rho_control_magnitude'])}",
        ]

    fold_preview = per_fold[per_fold["score_name"].eq(MAIN_SCORE)].copy()
    if fold_preview.empty and best_single is not None:
        fold_preview = per_fold[per_fold["score_name"].eq(best_single["score_name"])].copy()

    text = [
        "# CuiHacohen2023 go/no-go report",
        "",
        "这是 CuiHacohen2023（崔，免疫细胞基因扰动大数据集）的第一关报告。",
        "",
        "核心问题：SafeConf（单细胞扰动预测结果可信度打分）在大数据集上是否真的有信号。",
        "",
        "## Decision",
        "",
        f"- decision（结论）: **{decision}**",
        f"- reason（原因）: {reason}",
        "",
        "## Main score",
        "",
        *main_lines,
        "",
        "## Single-feature diagnosis",
        "",
        *best_lines,
        "",
        "## Per-fold main-score rho",
        "",
        "```",
        fold_preview[
            [
                "score_name",
                "fold_id",
                "n",
                "aligned_rho",
                "partial_rho_control_magnitude",
                "risk_coverage80_improve_pct",
            ]
        ].to_string(index=False)
        if not fold_preview.empty
        else "NO_FOLD_TABLE",
        "```",
        "",
        "## Top score summary",
        "",
        "```",
        score_summary.head(12).to_string(index=False),
        "```",
        "",
        "## Predictor summary",
        "",
        "```",
        per_predictor[per_predictor["score_name"].isin([MAIN_SCORE, "model_disagreement_risk"])]
        .head(20)
        .to_string(index=False),
        "```",
        "",
        "## How to read this",
        "",
        "- aligned rho（方向对齐相关）越高，说明高风险/低可信预测更容易真的错。",
        "- magnitude-only rho（效应大小基线）很高时，要小心分数只是看见了扰动效应大。",
        "- partial rho（控制效应大小后的相关）还为正，才说明分数可能有独立价值。",
        "- per-fold（每折）都要看，不能只看一个 pooled（混合总数）。",
    ]
    (out_dir / "CUI_GO_NOGO_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run(mvp_dir: Path, protocol_dir: Path | None, out_dir: Path) -> dict:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    scored = load_scored_records(mvp_dir, protocol_dir)
    scored.to_csv(out_dir / "tables" / "CUI_SCORED_RECORDS_WITH_MAGNITUDE.csv", index=False)

    test = scored[scored["split"].eq("test")].dropna(subset=["score_value", "true_error_rmse"]).copy()
    summary = _score_summary(test)
    per_fold = _per_fold_summary(test)
    per_predictor = _per_predictor_summary(test)

    summary.to_csv(out_dir / "tables" / "CUI_GO_NOGO_SUMMARY.csv", index=False)
    per_fold.to_csv(out_dir / "tables" / "CUI_PER_FOLD_RHO.csv", index=False)
    per_predictor.to_csv(out_dir / "tables" / "CUI_PER_PREDICTOR_RHO.csv", index=False)
    summary[summary["score_name"].isin(SINGLE_FEATURE_SCORES)].to_csv(
        out_dir / "tables" / "CUI_SINGLE_FEATURE_DIAG.csv", index=False
    )

    status = {
        "mvp_dir": str(mvp_dir),
        "protocol_dir": str(protocol_dir) if protocol_dir else "",
        "out_dir": str(out_dir),
        "n_scored_rows": int(len(scored)),
        "n_test_scored_rows": int(len(test)),
        "main_score": MAIN_SCORE,
    }
    write_report(out_dir / "reports", summary, per_fold, per_predictor, status)
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CuiHacohen2023 go/no-go report.")
    parser.add_argument("--mvp-dir", type=Path, required=True, help="Output directory from run_blind_dataset.py.")
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=None,
        help="Optional output directory from safetrans_confidence.cli.run_benchmark.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.mvp_dir, args.protocol_dir, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
