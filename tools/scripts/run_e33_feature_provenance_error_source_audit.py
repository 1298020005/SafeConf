#!/usr/bin/env python3
"""E33: feature provenance and error-source audit.

This audit answers the questions raised by advisor Zhou:

1. Which predictor produced the error used for evaluation?
2. Which inputs are used by each SafeConf score?
3. Is true held-out effect magnitude used only retrospectively, or leaked into
   a deployable risk score?

The script is intentionally lightweight.  It consolidates existing result
tables and produces a provenance checklist before any new heavy split setting
is launched.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E33_feature_provenance_error_source_audit_20260709"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode().strip())
    except Exception:
        return True


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_feature_provenance() -> pd.DataFrame:
    rows = [
        {
            "score_or_feature": "context_similarity_score",
            "feature_group": "context",
            "input_source": "target context control expression + train-fold control expression",
            "uses_predictor_output": False,
            "uses_train_truth": False,
            "uses_heldout_truth": False,
            "prospective_status": "deployable_if_target_control_available",
            "advisor_note": "可用于新 context，但需要该 context 的 control 表达。",
        },
        {
            "score_or_feature": "perturbation_support_score",
            "feature_group": "support",
            "input_source": "train-fold task table",
            "uses_predictor_output": False,
            "uses_train_truth": False,
            "uses_heldout_truth": False,
            "prospective_status": "deployable",
            "advisor_note": "只数训练 fold 中同一 perturbation 的历史支持量。",
        },
        {
            "score_or_feature": "historical_residual_risk",
            "feature_group": "history",
            "input_source": "train-fold historical prediction residuals/errors",
            "uses_predictor_output": True,
            "uses_train_truth": True,
            "uses_heldout_truth": False,
            "prospective_status": "deployable_with_historical_records",
            "advisor_note": "可用历史真值训练/校准，不能用当前 held-out pair 真值。",
        },
        {
            "score_or_feature": "model_disagreement_risk",
            "feature_group": "disagreement",
            "input_source": "multiple predictor outputs for the target task",
            "uses_predictor_output": True,
            "uses_train_truth": False,
            "uses_heldout_truth": False,
            "prospective_status": "deployable_if_predictors_can_run",
            "advisor_note": "它是任务难度线索，不是某个模型自己的可靠性保证。",
        },
        {
            "score_or_feature": "predicted_magnitude / prediction_magnitude_risk",
            "feature_group": "predicted_magnitude",
            "input_source": "predicted effect vector norm",
            "uses_predictor_output": True,
            "uses_train_truth": False,
            "uses_heldout_truth": False,
            "prospective_status": "deployable_baseline",
            "advisor_note": "可作为前置强基线，必须和 SafeConf 并列报告。",
        },
        {
            "score_or_feature": "true_effect_magnitude / oracle_magnitude_diagnostic",
            "feature_group": "true_magnitude",
            "input_source": "held-out true effect vector norm",
            "uses_predictor_output": False,
            "uses_train_truth": False,
            "uses_heldout_truth": True,
            "prospective_status": "retrospective_control_only",
            "advisor_note": "只能事后控制混杂，不能进入前置风险打分。",
        },
        {
            "score_or_feature": "protocol_v0_2_family_confidence / SafeConf frozen",
            "feature_group": "combined_protocol",
            "input_source": "context/support/disagreement family score, standardized within fold",
            "uses_predictor_output": True,
            "uses_train_truth": False,
            "uses_heldout_truth": False,
            "prospective_status": "deployable_if_component_inputs_available",
            "advisor_note": "主张放在 task-level risk triage。",
        },
        {
            "score_or_feature": "learned_risk / safeconf_lodo_risk",
            "feature_group": "learned_extension",
            "input_source": "train-fold score/error pairs, then applied to held-out tasks",
            "uses_predictor_output": True,
            "uses_train_truth": True,
            "uses_heldout_truth": False,
            "prospective_status": "deployable_after_train_fold_calibration",
            "advisor_note": "可作扩展，不覆盖 frozen 失败边界。",
        },
        {
            "score_or_feature": "true_error_rmse",
            "feature_group": "evaluation_target",
            "input_source": "predicted effect vs held-out true effect",
            "uses_predictor_output": True,
            "uses_train_truth": False,
            "uses_heldout_truth": True,
            "prospective_status": "evaluation_only",
            "advisor_note": "必须绑定 predictor_name，不能只写“预测错误”。",
        },
    ]
    return pd.DataFrame(rows)


def collect_error_sources() -> pd.DataFrame:
    candidate_files = []
    for path in (ROOT / "docs" / "实验结果").glob("**/*.csv"):
        name = path.name.lower()
        if "prediction_records" in name or "confidence_eval_summary" in name or "per_predictor" in name or "risk_audit_summary" in name:
            candidate_files.append(path)

    rows = []
    for path in sorted(candidate_files):
        try:
            df = pd.read_csv(path, nrows=5000)
        except Exception as exc:
            rows.append(
                {
                    "source_file": rel(path),
                    "read_status": "failed",
                    "error": repr(exc),
                }
            )
            continue
        cols = set(df.columns)
        predictor_values = ""
        dataset_values = ""
        if "predictor_name" in cols:
            predictor_values = ";".join(sorted(map(str, df["predictor_name"].dropna().unique()))[:12])
        if "dataset_name" in cols:
            dataset_values = ";".join(sorted(map(str, df["dataset_name"].dropna().unique()))[:12])
        error_cols = [c for c in df.columns if "error" in c.lower() or "rmse" in c.lower()]
        rows.append(
            {
                "source_file": rel(path),
                "read_status": "ok",
                "n_preview_rows": len(df),
                "has_predictor_name": "predictor_name" in cols,
                "predictor_values_preview": predictor_values,
                "dataset_values_preview": dataset_values,
                "error_columns": ";".join(error_cols),
                "has_true_error_rmse": "true_error_rmse" in cols,
                "has_score_name": "score_name" in cols,
                "has_split_setting": "split_setting" in cols,
            }
        )
    return pd.DataFrame(rows)


def build_magnitude_split() -> pd.DataFrame:
    rows = [
        {
            "magnitude_name": "true_effect_magnitude",
            "input_vector": "held-out true effect",
            "available_before_wetlab_or_truth": False,
            "allowed_role": "retrospective confounding control",
            "forbidden_role": "prospective risk score",
            "reporting_rule": "写成事后控制，不写成可部署特征。",
        },
        {
            "magnitude_name": "predicted_magnitude",
            "input_vector": "predicted effect",
            "available_before_wetlab_or_truth": True,
            "allowed_role": "deployable strong baseline",
            "forbidden_role": "oracle control",
            "reporting_rule": "和 SafeConf 并列报告，尤其 chemical 场景。",
        },
        {
            "magnitude_name": "oracle_magnitude_diagnostic",
            "input_vector": "held-out true effect",
            "available_before_wetlab_or_truth": False,
            "allowed_role": "diagnostic upper-bound/confounding check",
            "forbidden_role": "method comparison as deployable model",
            "reporting_rule": "只能作为诊断，不和可部署方法混排为同类。",
        },
    ]
    return pd.DataFrame(rows)


def build_leakage_checklist(feature_df: pd.DataFrame, error_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in feature_df.iterrows():
        status = row["prospective_status"]
        uses_heldout = bool(row["uses_heldout_truth"])
        if uses_heldout and status not in {"retrospective_control_only", "evaluation_only"}:
            call = "FAIL"
        elif uses_heldout:
            call = "PASS_IF_NOT_USED_FOR_SCORING"
        else:
            call = "PASS"
        rows.append(
            {
                "item": row["score_or_feature"],
                "check": "heldout truth usage",
                "call": call,
                "detail": row["advisor_note"],
            }
        )
    missing_predictor = error_df[
        (error_df["read_status"].eq("ok"))
        & (error_df["error_columns"].astype(str).str.len() > 0)
        & (~error_df["has_predictor_name"].fillna(False))
    ]
    rows.append(
        {
            "item": "error source tables",
            "check": "predictor_name required for error interpretation",
            "call": "WARN" if len(missing_predictor) else "PASS",
            "detail": f"{len(missing_predictor)} previewed error/score tables lack predictor_name; aggregate summaries need explicit note.",
        }
    )
    return pd.DataFrame(rows)


def write_report(feature_df: pd.DataFrame, error_df: pd.DataFrame, mag_df: pd.DataFrame, leak_df: pd.DataFrame) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    deployable = feature_df[feature_df["prospective_status"].astype(str).str.contains("deployable", na=False)]
    retrospective = feature_df[feature_df["prospective_status"].isin(["retrospective_control_only", "evaluation_only"])]
    warn_count = int((leak_df["call"] == "WARN").sum())
    fail_count = int((leak_df["call"] == "FAIL").sum())

    text = f"""# E33 输入来源与评价对象审计

生成时间：{now}

## 核心结果

- 可前置使用的 score/feature：{len(deployable)}
- 只能事后使用的项目：{len(retrospective)}
- leakage checklist：FAIL = {fail_count}，WARN = {warn_count}

## 给周老师的回答

1. 后续所有结果都要绑定 `predictor_name`。不能只写“预测错误”，要写清这个 RMSE 来自哪个 predictor。
2. `model_disagreement_risk` 确实使用模型输出。它的定位是 task-level difficulty，不是 per-model reliability。
3. `true_effect_magnitude` 只能作为事后混杂控制；`predicted_magnitude` 才能作为前置 baseline。
4. 如果一个 prospective score 使用了 held-out true effect，就不能进入主方法。

## 下一步

E33 已经把输入来源口径锁住。下一步可以触发 E34 小矩阵和 E35 整行/整列 holdout。

## 输出表

- `tables/E33_FEATURE_PROVENANCE.csv`
- `tables/E33_ERROR_SOURCE_MAP.csv`
- `tables/E33_MAGNITUDE_SPLIT_TABLE.csv`
- `tables/E33_LEAKAGE_CHECKLIST.csv`
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "E33_ADVISOR_QA.md").write_text(text, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    feature_df = build_feature_provenance()
    error_df = collect_error_sources()
    mag_df = build_magnitude_split()
    leak_df = build_leakage_checklist(feature_df, error_df)

    save(feature_df, TABLES / "E33_FEATURE_PROVENANCE.csv")
    save(error_df, TABLES / "E33_ERROR_SOURCE_MAP.csv")
    save(mag_df, TABLES / "E33_MAGNITUDE_SPLIT_TABLE.csv")
    save(leak_df, TABLES / "E33_LEAKAGE_CHECKLIST.csv")
    write_report(feature_df, error_df, mag_df, leak_df)

    status = {
        "run_id": "E33_feature_provenance_error_source_audit_20260709",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "status": "completed",
        "outputs": [
            "tables/E33_FEATURE_PROVENANCE.csv",
            "tables/E33_ERROR_SOURCE_MAP.csv",
            "tables/E33_MAGNITUDE_SPLIT_TABLE.csv",
            "tables/E33_LEAKAGE_CHECKLIST.csv",
            "reports/E33_ADVISOR_QA.md",
        ],
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E33 feature provenance and error-source audit\n\n"
        "先看 `reports/E33_ADVISOR_QA.md`。这份审计回答周老师关于 score 输入来源、predictor error 来源和 magnitude 泄漏的问题。\n",
        encoding="utf-8",
    )
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
