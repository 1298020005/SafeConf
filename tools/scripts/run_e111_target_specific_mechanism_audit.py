#!/usr/bin/env python3
"""E111: determine which formal predictor errors SafeConf actually ranks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv"
OUT=ROOT/"docs/实验结果/E111_target_specific_mechanism_audit_20260713"; TABLES=OUT/"tables"; REPORTS=OUT/"reports"; FIGURES=OUT/"figures"
SCORES=["safeconf_calibrated_pair_risk","safeconf_frozen_pair_risk","risk_model_disagreement","baseline_predicted_magnitude"]
TARGETS=["error_two_predictor_mean_rmse","error_two_predictor_max_rmse","error_gears_rmse","error_scgpt_rmse"]
SEED=202607111; N_BOOTSTRAP=4000


def rho(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3 or np.unique(a[m]).size<2 or np.unique(b[m]).size<2:return float("nan")
    return float(np.corrcoef(rankdata(a[m]),rankdata(b[m]))[0,1])


def rc80(score,error):
    score,error=np.asarray(score,float),np.asarray(error,float);keep=max(1,int(np.ceil(.8*len(score))));ix=np.argsort(score,kind="stable")[:keep]
    return float(100*(error.mean()-error[ix].mean())/max(error.mean(),1e-12))


def fold_summary(data):
    rows=[]
    for fold,g in data.groupby("fold_id",sort=True):
        for target in TARGETS:
            for score in SCORES:
                rows.append({"fold_id":fold,"target":target,"score":score,"n_tasks":len(g),"spearman":rho(g[score],g[target]),"risk_coverage80_improve_pct":rc80(g[score],g[target])})
    return pd.DataFrame(rows)


def bootstrap(data):
    cache={}
    for fold,g in data.groupby("fold_id"):
        g=g.reset_index(drop=True);clusters=[np.flatnonzero(g.perturbation.astype(str).to_numpy()==p) for p in sorted(g.perturbation.astype(str).unique())];cache[str(fold)]=(g,clusters)
    folds=sorted(cache);rng=np.random.default_rng(SEED);rows=[];primary="safeconf_calibrated_pair_risk"
    for target in TARGETS:
        for comparator in ["safeconf_frozen_pair_risk","risk_model_disagreement","baseline_predicted_magnitude"]:
            observed=[]
            for g,_ in cache.values():observed.append(rho(g[primary],g[target])-rho(g[comparator],g[target]))
            samples=[]
            for _ in range(N_BOOTSTRAP):
                deltas=[]
                for fold in rng.choice(folds,len(folds),replace=True):
                    g,clusters=cache[str(fold)];ix=np.concatenate([clusters[int(i)] for i in rng.integers(0,len(clusters),len(clusters))])
                    deltas.append(rho(g[primary].to_numpy(float)[ix],g[target].to_numpy(float)[ix])-rho(g[comparator].to_numpy(float)[ix],g[target].to_numpy(float)[ix]))
                samples.append(float(np.nanmean(deltas)))
            sample=np.asarray(samples)
            rows.append({"target":target,"primary":primary,"comparator":comparator,"observed_macro_delta_spearman":float(np.nanmean(observed)),"ci95_low":float(np.nanquantile(sample,.025)),"ci95_high":float(np.nanquantile(sample,.975)),"probability_delta_gt_zero":float(np.mean(sample>0)),"bootstrap_unit":"outer_fold_plus_perturbation_cluster","n_bootstrap":N_BOOTSTRAP})
    return pd.DataFrame(rows)


def write_svg(summary):
    macro=summary.groupby(["target","score"],as_index=False).spearman.mean();targets=TARGETS;labels=["两模型平均误差","两模型最坏误差","GEARS 误差","scGPT 误差"];scores=["safeconf_calibrated_pair_risk","risk_model_disagreement","baseline_predicted_magnitude"];colors=["#315f77","#759789","#ad8158"]
    w,h,x0,y0,pw,ph=1080,620,100,100,900,390;sy=lambda v:y0+(.6-v)/(.8)*ph;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">','<rect width="100%" height="100%" fill="#fff"/>','<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#27343c}.t{font-size:25px;font-weight:700}.s{font-size:14px;fill:#647078}.l{font-size:15px}</style>','<text x="48" y="40" class="t">E111｜SafeConf 在筛哪个模型的错误？</text>','<text x="48" y="68" class="s">Frangieh 正式 scGPT–GEARS；3 个外层 fold 的 Spearman 宏平均。</text>']
    for tick in [-.2,0,.2,.4,.6]:
        y=sy(tick);parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="#dde4e7"/>',f'<text x="{x0-12}" y="{y+5:.1f}" text-anchor="end" class="s">{tick:.1f}</text>']
    gw=pw/4
    for i,(target,label) in enumerate(zip(targets,labels)):
        c=x0+gw*(i+.5)
        for j,(score,color) in enumerate(zip(scores,colors)):
            v=float(macro[(macro.target==target)&(macro.score==score)].spearman.iloc[0]);x=c+(j-1)*52-20;z=sy(0);y=sy(v)
            parts += [f'<rect x="{x:.1f}" y="{min(y,z):.1f}" width="40" height="{max(abs(z-y),1):.1f}" fill="{color}"/>',f'<text x="{x+20:.1f}" y="{min(y,z)-6:.1f}" text-anchor="middle" class="s">{v:.2f}</text>']
        parts.append(f'<text x="{c:.1f}" y="530" text-anchor="middle" class="l">{label}</text>')
    for i,(label,color) in enumerate(zip(["SafeConf 校准风险","模型分歧","预测幅度"],colors)):
        x=220+i*245;parts.append(f'<rect x="{x}" y="575" width="18" height="12" fill="{color}"/><text x="{x+27}" y="587" class="s">{label}</text>')
    parts.append('</svg>');(FIGURES/"F1_target_specific_spearman.svg").write_text("\n".join(parts))


def main():
    for d in (TABLES,REPORTS,FIGURES):d.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(SOURCE);summary=fold_summary(data);boot=bootstrap(data);summary.to_csv(TABLES/"E111_FOLD_SUMMARY.csv",index=False);boot.to_csv(TABLES/"E111_CLUSTER_BOOTSTRAP.csv",index=False);write_svg(summary)
    macro=summary.groupby(["target","score"],as_index=False).agg(spearman=("spearman","mean"),rc80_improve_pct=("risk_coverage80_improve_pct","mean"));macro.to_csv(TABLES/"E111_MACRO_SUMMARY.csv",index=False)
    model_error=data.groupby("fold_id",as_index=False).agg(scgpt_mean_rmse=("error_scgpt_rmse","mean"),gears_mean_rmse=("error_gears_rmse","mean"));model_error.to_csv(TABLES/"E111_MODEL_ERROR.csv",index=False)
    status={"experiment":"E111_target_specific_mechanism_audit","generated_at":datetime.now().isoformat(timespec="seconds"),"status":"complete","n_test_rows":len(data),"targets":TARGETS,"analysis_role":"pre-specified secondary targets from earlier E98 protocol; mechanism audit, not replacement primary endpoint","test_truth_used_to_change_score":False}
    (OUT/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    lines=["# E111｜SafeConf 实际在筛哪类错误","","E108 的主目标仍是两模型平均 RMSE。E111 沿用 E98 已经列出的模型别误差与最坏误差作为次要终点，解释信号来源，不回调风险分数。","",f"scGPT 平均 RMSE 为 {data.error_scgpt_rmse.mean():.4f}，GEARS 为 {data.error_gears_rmse.mean():.4f}。","","## 3-fold 宏平均","","| target | score | ρ | RC80 改善 |","|---|---|---:|---:|"]
    for r in macro.itertuples(index=False):lines.append(f"| {r.target} | {r.score} | {r.spearman:.3f} | {r.rc80_improve_pct:.2f}% |")
    lines += ["","## 解释","","SafeConf 对 GEARS 误差和双模型最坏误差的排序明显强于对 scGPT 误差。分歧在这里更像“较弱预测器偏离较强预测器”的路由信号，而不是与模型无关的通用置信度。论文应把适用对象写成任务/预测器组合，并报告模型别结果。","","## 聚类 bootstrap","","| target | comparator | Δρ | 95% CI | P(Δ>0) |","|---|---|---:|---:|---:|"]
    for r in boot.itertuples(index=False):lines.append(f"| {r.target} | {r.comparator} | {r.observed_macro_delta_spearman:.3f} | [{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.probability_delta_gt_zero:.3f} |")
    (REPORTS/"E111_REPORT.md").write_text("\n".join(lines)+"\n");(OUT/"README_先看这个.md").write_text("# E111 先看这个\n\n先读 `reports/E111_REPORT.md`。\n")
    print(json.dumps(status,ensure_ascii=False,indent=2));print(macro.to_string(index=False));print(boot.to_string(index=False))


if __name__=="__main__":main()
