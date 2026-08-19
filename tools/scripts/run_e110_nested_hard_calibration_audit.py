#!/usr/bin/env python3
"""E110: fit hard-setting risk calibration on E109, audit on E108 tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
E108 = ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_ALL_TASK_RISK_TABLE.csv"
E109 = ROOT / "docs/实验结果/E109_inner_hard_setting_predictions_20260713/E109_ALL_INNER_ROWS.csv"
OUT = ROOT / "docs/实验结果/E110_nested_hard_calibration_audit_20260713"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
FEATURES = ["risk_disagreement_z", "predicted_magnitude_z", "context_novelty_scaled", "perturbation_novelty", "structural_interaction"]
ALPHA = 1.0
SEED = 202607110
N_BOOTSTRAP = 3000


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a,b=np.asarray(a,float),np.asarray(b,float); mask=np.isfinite(a)&np.isfinite(b)
    if mask.sum()<3 or np.unique(a[mask]).size<2 or np.unique(b[mask]).size<2: return float("nan")
    return float(np.corrcoef(rankdata(a[mask]),rankdata(b[mask]))[0,1])


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame=frame.copy(); frame["structural_interaction"]=frame.context_novelty_scaled*frame.perturbation_novelty
    return frame


def calibrate() -> tuple[pd.DataFrame,pd.DataFrame]:
    outer=add_features(pd.read_csv(E108)); inner=add_features(pd.read_csv(E109)); outputs=[]; params=[]
    for fold,group in outer.groupby("fold_id",sort=True):
        calibration=inner[(inner.outer_fold_id==fold)&inner.split.eq("test")].copy()
        if calibration.empty: raise RuntimeError(f"missing E109 rows for {fold}")
        model=Ridge(alpha=ALPHA,positive=True).fit(calibration[FEATURES].to_numpy(float),calibration.error_two_predictor_mean_rmse.to_numpy(float))
        group=group.copy(); group["safeconf_nested_hard_risk"]=model.predict(group[FEATURES].to_numpy(float))
        threshold=float(np.quantile(model.predict(calibration[FEATURES].to_numpy(float)),.80))
        group["accepted_by_nested_q80"]=group.safeconf_nested_hard_risk<=threshold; group["nested_q80_threshold"]=threshold
        outputs.append(group)
        params.append({"fold_id":fold,"ridge_alpha_fixed":ALPHA,"n_inner_calibration_rows":len(calibration),
                       **{f"coefficient_{name}":float(value) for name,value in zip(FEATURES,model.coef_)},
                       "intercept":float(model.intercept_),"inner_risk_q80_threshold":threshold,"outer_test_truth_used":False})
    return pd.concat(outputs,ignore_index=True),pd.DataFrame(params)


def summarize(tasks: pd.DataFrame) -> pd.DataFrame:
    scores=["safeconf_nested_hard_risk","safeconf_calibrated_pair_risk","safeconf_frozen_pair_risk","risk_model_disagreement","baseline_predicted_magnitude"]
    rows=[]; test=tasks[tasks.split.eq("test")]
    for fold,fg in test.groupby("fold_id",sort=True):
        for setting,g in list(fg.groupby("setting",sort=True))+[("all_test_settings_pooled",fg)]:
            error=g.error_two_predictor_mean_rmse.to_numpy(float)
            for score in scores:
                rows.append({"fold_id":fold,"setting":setting,"score":score,"n_tasks":len(g),"spearman":spearman(g[score],error),
                             "mean_error":float(error.mean()),"accepted_nested_q80_fraction":float(g.accepted_by_nested_q80.mean()),
                             "accepted_nested_q80_mean_error":float(g.loc[g.accepted_by_nested_q80,"error_two_predictor_mean_rmse"].mean()) if g.accepted_by_nested_q80.any() else float("nan")})
    return pd.DataFrame(rows)


def bootstrap(tasks: pd.DataFrame) -> pd.DataFrame:
    test=tasks[tasks.split.eq("test")].copy(); primary="safeconf_nested_hard_risk"
    comparators=["safeconf_calibrated_pair_risk","safeconf_frozen_pair_risk","risk_model_disagreement","baseline_predicted_magnitude"]
    cache={}
    for fold,g in test.groupby("fold_id"):
        g=g.reset_index(drop=True); clusters=[np.flatnonzero(g.perturbation.astype(str).to_numpy()==p) for p in sorted(g.perturbation.astype(str).unique())]; cache[str(fold)]=(g,clusters)
    rng=np.random.default_rng(SEED); folds=sorted(cache); rows=[]
    for comparator in comparators:
        observed=[]
        for g,_ in cache.values():
            error=g.error_two_predictor_mean_rmse.to_numpy(float); observed.append(spearman(g[primary],error)-spearman(g[comparator],error))
        samples=[]
        for _ in range(N_BOOTSTRAP):
            values=[]
            for fold in rng.choice(folds,len(folds),replace=True):
                g,clusters=cache[str(fold)]; index=np.concatenate([clusters[int(i)] for i in rng.integers(0,len(clusters),len(clusters))]); error=g.error_two_predictor_mean_rmse.to_numpy(float)[index]
                values.append(spearman(g[primary].to_numpy(float)[index],error)-spearman(g[comparator].to_numpy(float)[index],error))
            samples.append(float(np.nanmean(values)))
        sample=np.asarray(samples)
        rows.append({"primary":primary,"comparator":comparator,"observed_macro_delta_spearman":float(np.nanmean(observed)),
                     "ci95_low":float(np.nanquantile(sample,.025)),"ci95_high":float(np.nanquantile(sample,.975)),
                     "probability_delta_gt_zero":float(np.mean(sample>0)),"bootstrap_unit":"outer_fold_plus_perturbation_cluster","n_bootstrap":N_BOOTSTRAP})
    return pd.DataFrame(rows)


def write_svg(summary: pd.DataFrame) -> None:
    data=summary.groupby(["setting","score"],as_index=False).spearman.mean(); order=["random_missing_pair","context_unseen_row","perturbation_unseen_column","context_and_perturbation_unseen","all_test_settings_pooled"]
    labels=["随机缺失","新背景","新扰动","双未见","全部任务"]; scores=["safeconf_nested_hard_risk","safeconf_calibrated_pair_risk","baseline_predicted_magnitude"]; colors=["#285f78","#78a091","#b18458"]
    width,height,x0,y0,pw,ph=1180,650,95,105,1010,430; low,high=-.3,.8; sy=lambda v:y0+(high-v)/(high-low)*ph
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">','<rect width="100%" height="100%" fill="#fff"/>','<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#27343c}.t{font-size:26px;font-weight:700}.s{font-size:14px;fill:#647078}.l{font-size:15px}</style>','<text x="48" y="40" class="t">E110｜困难设置内层校准</text>','<text x="48" y="68" class="s">校准数据来自 E109；外层 Frangieh 测试标签不参与拟合。柱高为 3 个外层 fold 的 Spearman 宏平均。</text>']
    for tick in [-.2,0,.2,.4,.6,.8]:
        y=sy(tick);parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="#dde4e7"/>',f'<text x="{x0-12}" y="{y+5:.1f}" text-anchor="end" class="s">{tick:.1f}</text>']
    gw=pw/len(order)
    for i,(setting,label) in enumerate(zip(order,labels)):
        center=x0+gw*(i+.5)
        for j,(score,color) in enumerate(zip(scores,colors)):
            value=float(data[(data.setting==setting)&(data.score==score)].spearman.iloc[0]); x=center+(j-1)*46-18; z=sy(0); y=sy(value)
            parts += [f'<rect x="{x:.1f}" y="{min(y,z):.1f}" width="36" height="{max(abs(z-y),1):.1f}" fill="{color}"/>',f'<text x="{x+18:.1f}" y="{min(y,z)-6:.1f}" text-anchor="middle" class="s">{value:.2f}</text>']
        parts.append(f'<text x="{center:.1f}" y="570" text-anchor="middle" class="l">{label}</text>')
    for i,(label,color) in enumerate(zip(["内层困难设置校准","原随机 pair 校准","预测幅度"],colors)):
        x=250+i*250;parts.append(f'<rect x="{x}" y="610" width="18" height="12" fill="{color}"/><text x="{x+27}" y="622" class="s">{label}</text>')
    parts.append('</svg>');(FIGURES/"F1_nested_calibration.svg").write_text("\n".join(parts))


def main() -> None:
    for d in (TABLES,REPORTS,FIGURES): d.mkdir(parents=True,exist_ok=True)
    tasks,params=calibrate();summary=summarize(tasks);boot=bootstrap(tasks);test=tasks[tasks.split.eq("test")]
    tasks.to_csv(TABLES/"E110_ALL_TASK_RISK_TABLE.csv",index=False);test.to_csv(TABLES/"E110_TEST_TASK_RISK_TABLE.csv",index=False);params.to_csv(TABLES/"E110_CALIBRATORS.csv",index=False);summary.to_csv(TABLES/"E110_FOLD_SETTING_SUMMARY.csv",index=False);boot.to_csv(TABLES/"E110_CLUSTER_BOOTSTRAP.csv",index=False);write_svg(summary)
    macro=summary.groupby(["setting","score"],as_index=False).spearman.mean(); pooled=macro[macro.setting.eq("all_test_settings_pooled")].sort_values("spearman",ascending=False)
    status={"experiment":"E110_nested_hard_calibration_audit","generated_at":datetime.now().isoformat(timespec="seconds"),"status":"complete","fixed_ridge_alpha":ALPHA,"features":FEATURES,
            "n_inner_calibration_rows":int(params.n_inner_calibration_rows.sum()),"n_outer_test_rows":len(test),"outer_test_truth_used_for_calibrator_or_threshold":False,"pooled_macro_spearman":dict(zip(pooled.score,pooled.spearman))}
    (OUT/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    lines=["# E110｜困难设置内层校准审计","","校准器只用 E109 的内层新背景、新扰动和双未见样本拟合。固定 `Ridge(alpha=1, positive=True)`，输入为分歧、预测幅度、背景新颖度、扰动新颖度及结构交互项。外层 837 个测试标签没有参与拟合或阈值。","","## 3-fold 宏平均 Spearman","","| setting | score | ρ |","|---|---|---:|"]
    for r in macro.itertuples(index=False): lines.append(f"| {r.setting} | {r.score} | {r.spearman:.3f} |")
    lines += ["","## 聚类 bootstrap","","| comparator | Δρ | 95% CI | P(Δ>0) |","|---|---:|---:|---:|"]
    for r in boot.itertuples(index=False): lines.append(f"| {r.comparator} | {r.observed_macro_delta_spearman:.3f} | [{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.probability_delta_gt_zero:.3f} |")
    lines += ["", "## 结论", "", "困难设置校准没有替代 E108 主分数：pooled ρ 为 0.176，低于随机-pair 校准的 0.253 和冻结规则的 0.242。它在新背景上达到 0.208，接近 magnitude 的 0.218，但新扰动、双未见和随机缺失没有同步改善。当前只有三个背景，每个外层 fold 的内层训练只能使用一个背景，权重迁移不稳定。E110 作为失败边界保留，不用于回调 E108 权重。"]
    (REPORTS/"E110_REPORT.md").write_text("\n".join(lines)+"\n");(OUT/"README_先看这个.md").write_text("# E110 先看这个\n\n先读 `reports/E110_REPORT.md`。\n")
    print(json.dumps(status,ensure_ascii=False,indent=2));print(pooled.to_string(index=False));print(boot.to_string(index=False))


if __name__=="__main__": main()
