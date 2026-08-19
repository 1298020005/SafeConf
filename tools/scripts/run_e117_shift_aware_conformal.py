#!/usr/bin/env python3
"""E117: nested, setting-matched one-sided conformal error bounds."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
E108 = ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv"
E109 = ROOT / "docs/实验结果/E109_inner_hard_setting_predictions_20260713/E109_ALL_INNER_ROWS.csv"
E114 = ROOT / "docs/实验结果/E114_split_conformal_error_bounds_20260713/tables/E114_TEST_TASK_BOUNDS.csv"
OUT = ROOT / "docs/实验结果/E117_shift_aware_conformal_20260713"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
ALPHA = 0.10
FEATURES = ["disagreement_z_nested", "magnitude_z_nested", "context_novelty_scaled", "perturbation_novelty", "support_scarcity"]


def order(value: str) -> str:
    return hashlib.sha256(("E117|" + str(value)).encode()).hexdigest()


def robust_params(values) -> tuple[float, float]:
    values = np.asarray(values, float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, float(np.std(values)), 1e-8)
    return center, scale


def z(values, center, scale):
    return np.clip((np.asarray(values, float) - center) / scale, -5, 5)


def conformal_q(residuals: np.ndarray) -> tuple[float, int]:
    residuals = np.sort(np.asarray(residuals, float))
    n = len(residuals)
    k = min(n, int(math.ceil((n + 1) * (1 - ALPHA))))
    return float(residuals[k - 1]), k


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(E108).copy()
    inner = pd.read_csv(E109)
    baseline = pd.read_csv(E114)[["fold_id", "task_id", "conformal_upper_error_90"]].rename(columns={"conformal_upper_error_90": "e114_upper_error_90"})
    outputs, audits = [], []
    for fold, outer in test.groupby("fold_id", sort=True):
        source = inner[(inner.outer_fold_id.eq(fold)) & inner.split.eq("test")].copy()
        if source.empty:
            raise ValueError(f"no E109 nested rows for {fold}")
        source["nested_key"] = source.inner_fold_id.astype(str) + "::" + source.context.astype(str) + "::" + source.perturbation.astype(str) + "::" + source.setting.astype(str)
        source["fit_role"] = ""
        for setting, idx in source.groupby("setting", sort=True).groups.items():
            ordered = sorted(idx, key=lambda i: order(source.loc[i, "nested_key"]))
            nfit = len(ordered) // 2
            source.loc[ordered[:nfit], "fit_role"] = "risk_fit"
            source.loc[ordered[nfit:], "fit_role"] = "conformal_calibration"
        fit = source.fit_role.eq("risk_fit")
        cal = source.fit_role.eq("conformal_calibration")
        dc, ds = robust_params(source.loc[fit, "risk_model_disagreement"])
        mc, ms = robust_params(source.loc[fit, "baseline_predicted_magnitude"])
        source["disagreement_z_nested"] = z(source.risk_model_disagreement, dc, ds)
        source["magnitude_z_nested"] = z(source.baseline_predicted_magnitude, mc, ms)
        source["support_scarcity"] = -np.log1p(source.training_support_count.astype(float))
        model = Ridge(alpha=1.0, positive=True).fit(source.loc[fit, FEATURES], source.loc[fit, "error_two_predictor_mean_rmse"])
        source["nested_base_error_prediction"] = model.predict(source[FEATURES])
        residuals = source.loc[cal, "error_two_predictor_mean_rmse"] - source.loc[cal, "nested_base_error_prediction"]
        pooled_q, pooled_k = conformal_q(residuals.to_numpy())
        q_by_setting = {}
        for setting, g in source.loc[cal].groupby("setting", sort=True):
            q_by_setting[setting] = conformal_q((g.error_two_predictor_mean_rmse - g.nested_base_error_prediction).to_numpy())
        out = outer.copy()
        out["disagreement_z_nested"] = z(out.risk_model_disagreement, dc, ds)
        out["magnitude_z_nested"] = z(out.baseline_predicted_magnitude, mc, ms)
        out["support_scarcity"] = -np.log1p(out.training_support_count.astype(float))
        out["nested_base_error_prediction"] = model.predict(out[FEATURES])
        out["nested_residual_quantile"] = [q_by_setting.get(s, (pooled_q, pooled_k))[0] for s in out.setting]
        out["nested_conformal_calibration_n"] = [int((source.loc[cal, "setting"] == s).sum()) if s in q_by_setting else int(cal.sum()) for s in out.setting]
        out["nested_conformal_source"] = ["setting_matched" if s in q_by_setting else "pooled_fallback" for s in out.setting]
        out["e117_upper_error_90"] = out.nested_base_error_prediction + out.nested_residual_quantile
        out["e117_bound_uses_outer_test_truth"] = False
        outputs.append(out)
        audits.append({"fold_id": fold, "n_inner_rows": len(source), "n_risk_fit": int(fit.sum()), "n_conformal_calibration": int(cal.sum()), "pooled_q": pooled_q, "pooled_k": pooled_k, "ridge_intercept": float(model.intercept_), "ridge_coefficients": json.dumps(dict(zip(FEATURES, map(float, model.coef_))), ensure_ascii=False), "outer_test_truth_used_for_fit_or_quantile": False})
    out = pd.concat(outputs, ignore_index=True).merge(baseline, on=["fold_id", "task_id"], how="left", validate="one_to_one")
    rows = []
    for setting, g in list(out.groupby("setting", sort=True)) + [("all_test_settings_pooled", out)]:
        rows.append({"setting": setting, "n_tasks": len(g), "nominal_coverage": 1 - ALPHA, "e117_empirical_coverage": float((g.error_two_predictor_mean_rmse <= g.e117_upper_error_90).mean()), "e114_empirical_coverage": float((g.error_two_predictor_mean_rmse <= g.e114_upper_error_90).mean()), "mean_true_error": float(g.error_two_predictor_mean_rmse.mean()), "e117_mean_upper": float(g.e117_upper_error_90.mean()), "e114_mean_upper": float(g.e114_upper_error_90.mean()), "mean_upper_reduction_vs_e114": float(g.e114_upper_error_90.mean() - g.e117_upper_error_90.mean())})
    return out, pd.DataFrame(rows), pd.DataFrame(audits)


def figure(summary: pd.DataFrame) -> None:
    labels = {"random_missing_pair": "随机缺失", "context_unseen_row": "新背景", "perturbation_unseen_column": "新扰动", "context_and_perturbation_unseen": "双未见", "all_test_settings_pooled": "全部"}
    d = summary.set_index("setting").loc[list(labels)].reset_index()
    w, h, x0, y0, pw, ph = 1050, 540, 100, 110, 850, 310
    sy = lambda v: y0 + (1 - v) * ph
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="#fff"/>', '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#27343c}.t{font-size:24px;font-weight:700}.s{font-size:13px;fill:#647078}.l{font-size:14px}</style>', '<text x="45" y="40" class="t">E117｜setting-matched conformal 覆盖率</text>', '<text x="45" y="68" class="s">青色：E117；灰色：E114；虚线：90% 名义覆盖。</text>', f'<line x1="{x0}" y1="{sy(.9):.1f}" x2="{x0+pw}" y2="{sy(.9):.1f}" stroke="#a66f55" stroke-dasharray="6 5"/>']
    for i, r in enumerate(d.itertuples(index=False)):
        c = x0 + pw / len(d) * (i + 0.5)
        for off, value, color in [(-23, r.e117_empirical_coverage, "#2f7f76"), (23, r.e114_empirical_coverage, "#9aa8ae")]:
            y = sy(value)
            parts += [f'<rect x="{c+off-18:.1f}" y="{y:.1f}" width="36" height="{sy(0)-y:.1f}" fill="{color}"/>', f'<text x="{c+off:.1f}" y="{y-6:.1f}" text-anchor="middle" class="s">{value:.2f}</text>']
        parts.append(f'<text x="{c:.1f}" y="455" text-anchor="middle" class="l">{labels[r.setting]}</text>')
    parts.append('</svg>')
    (FIGURES / "F1_shift_aware_conformal.svg").write_text("\n".join(parts))


def main() -> None:
    for d in (TABLES, REPORTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    tasks, summary, audit = run()
    tasks.to_csv(TABLES / "E117_TEST_TASK_BOUNDS.csv", index=False)
    summary.to_csv(TABLES / "E117_COVERAGE_SUMMARY.csv", index=False)
    audit.to_csv(TABLES / "E117_FOLD_AUDIT.csv", index=False)
    figure(summary)
    pooled = summary[summary.setting.eq("all_test_settings_pooled")].iloc[0]
    settings = summary[~summary.setting.eq("all_test_settings_pooled")]
    passed = bool(pooled.e117_empirical_coverage >= 0.90 and settings.e117_empirical_coverage.min() >= 0.85 and pooled.e117_mean_upper < pooled.e114_mean_upper and not audit.outer_test_truth_used_for_fit_or_quantile.any())
    lines = ["# E117｜困难设置匹配的 conformal 误差上界", "", "E117 只使用 E109 的内层 row/column/double 任务拟合基础误差并校准残差。E108 外层测试真值没有进入模型、候选选择或分位数。random-pair 没有匹配的内层 setting，按预设使用全部内层 calibration residual。", "", "| setting | n | E117 coverage | E114 coverage | E117 mean upper | E114 mean upper | upper reduction |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples(index=False):
        lines.append(f"| {r.setting} | {r.n_tasks} | {r.e117_empirical_coverage:.3f} | {r.e114_empirical_coverage:.3f} | {r.e117_mean_upper:.4f} | {r.e114_mean_upper:.4f} | {r.mean_upper_reduction_vs_e114:.4f} |")
    lines += ["", "## 预设判定", "", f"- 通过：**{'是' if passed else '否'}**。", f"- pooled coverage ≥0.90：{'是' if pooled.e117_empirical_coverage >= .90 else '否'}。", f"- 每个 setting coverage ≥0.85：{'是' if settings.e117_empirical_coverage.min() >= .85 else '否'}。", f"- mean upper 低于 E114：{'是' if pooled.e117_mean_upper < pooled.e114_mean_upper else '否'}。", "", "只有全部条件通过，E117 才能替换 E114。即使通过，理论保证仍依赖内层困难任务与未来任务的条件可交换性；这里不能写成任意分布偏移下的无条件覆盖。"]
    (REPORTS / "E117_REPORT.md").write_text("\n".join(lines) + "\n")
    status = {"experiment": "E117_shift_aware_conformal", "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete", "n_outer_folds": int(tasks.fold_id.nunique()), "n_test_tasks": len(tasks), "nominal_coverage": 0.90, "pooled_empirical_coverage": float(pooled.e117_empirical_coverage), "pooled_mean_upper": float(pooled.e117_mean_upper), "e114_pooled_mean_upper": float(pooled.e114_mean_upper), "outer_test_truth_used_for_fit_or_quantile": False, "preregistered_gate_passed": passed}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README_先看这个.md").write_text("# E117 先看这个\n\n先读 `reports/E117_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
