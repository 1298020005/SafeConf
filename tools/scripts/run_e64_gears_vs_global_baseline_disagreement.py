#!/usr/bin/env python3
"""E64: actual GEARS vs source-only global-effect baseline on the E60 panel.

E60 found that three GEARS seeds are too similar to form a useful uncertainty
signal.  This experiment keeps the identical fixed 24 Adamson tasks and gene
space, but replaces the second predictor with a transparent independent
baseline: the mean perturbation effect across all *training* conditions.

The baseline never uses a held-out task's cells.  The question is deliberately
narrow: does GEARS--baseline disagreement rank GEARS ensemble error better than
GEARS prediction magnitude?  A negative result is retained as a failure
boundary, not hidden.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E60 = ROOT / "docs" / "实验结果" / "E60_gears_fixed_panel_formal_20260711"
OUT = ROOT / "docs" / "实验结果" / "E64_gears_vs_global_baseline_20260711"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
PROCESSED = Path("/home/yyf/data/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad")
PRED = E60 / "arrays" / "gears_predicted_effects.npz"
TRUE = E60 / "arrays" / "gears_true_effects.npz"


def now() -> str: return datetime.now().isoformat(timespec="seconds")

def git_head() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: return "unknown"

def rmse(a: np.ndarray, b: np.ndarray) -> float: return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a)<3 or len(np.unique(a))<2 or len(np.unique(b))<2: return float("nan")
    return float(pd.Series(a).corr(pd.Series(b),method="spearman"))

def bootstrap_ci(score: np.ndarray, error: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float,float]:
    vals=[]
    for _ in range(n):
        i=rng.integers(0,len(score),len(score)); vals.append(spearman(score[i],error[i]))
    vals=np.asarray(vals); vals=vals[np.isfinite(vals)]
    return (float(np.quantile(vals,.025)),float(np.quantile(vals,.975))) if len(vals) else (float("nan"),float("nan"))

def top20(score: np.ndarray,error: np.ndarray) -> tuple[int,float]:
    k=max(1,math.ceil(.2*len(score))); return k,float(error[np.argsort(-score,kind="stable")[:k]].mean()/error.mean())

def load_e60_ensemble() -> tuple[dict[str,np.ndarray],dict[str,np.ndarray],pd.DataFrame]:
    rec=pd.read_csv(E60/"tables"/"PREDICTION_RECORDS.csv")
    with np.load(PRED) as f: pred={k:np.asarray(f[k],dtype=np.float32) for k in f.files}
    with np.load(TRUE) as f: truth={k:np.asarray(f[k],dtype=np.float32) for k in f.files}
    ensemble, shared_true={},{}
    for pert,sub in rec.groupby("perturbation",sort=True):
        vectors=[pred[k] for k in sub.predicted_effect_key]; trues=[truth[k] for k in sub.true_effect_key]
        ensemble[str(pert)]=np.mean(np.stack(vectors),axis=0).astype(np.float32); shared_true[str(pert)]=trues[0]
        if any(np.max(np.abs(v-trues[0]))>1e-7 for v in trues[1:]): raise ValueError(f"inconsistent truth for {pert}")
    return ensemble,shared_true,rec

def source_global_effect(test_conditions: set[str]) -> tuple[np.ndarray,dict]:
    a=ad.read_h5ad(PROCESSED,backed="r")
    try:
        cond=a.obs["condition"].astype(str).to_numpy(); nonctrl=sorted(c for c in set(cond) if c!="ctrl" and c not in test_conditions)
        if not nonctrl: raise ValueError("no training conditions after fixed test removal")
        ctrl=np.asarray(a[a.obs["condition"].astype(str).eq("ctrl")].X.mean(axis=0)).ravel().astype(np.float32)
        effects=[]
        for c in nonctrl:
            mean=np.asarray(a[a.obs["condition"].astype(str).eq(c)].X.mean(axis=0)).ravel().astype(np.float32)
            effects.append(mean-ctrl)
        return np.mean(np.stack(effects),axis=0).astype(np.float32),{"n_train_conditions":len(nonctrl),"n_total_conditions":len(set(cond)),"n_genes":int(a.n_vars)}
    finally:
        a.file.close()

def write_svg(df: pd.DataFrame) -> None:
    w,h,left,top=1080,700,110,90; x=df.risk_gears_globalmean_disagreement.to_numpy(float); y=df.error_gears_ensemble_rmse.to_numpy(float)
    def scale(a,lo,hi):
        amin,amax=float(a.min()),float(a.max()); d=max(amax-amin,1e-9); amin-=.08*d; amax+=.08*d; return lo+(a-amin)/(amax-amin)*(hi-lo),amin,amax
    sx,xmin,xmax=scale(x,left,w-72); sy,ymin,ymax=scale(y,h-85,top)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="#ffffff"/>','<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#23313d}.t{font-size:27px;font-weight:700}.s{font-size:16px;fill:#5c6a75}.a{font-size:15px}.sm{font-size:12px;fill:#5c6a75}</style>','<text class="t" x="55" y="43">E64｜GEARS 与全局效应基线的分歧</text>','<text class="s" x="55" y="70">Adamson 固定 24 个未见基因任务；横轴只由两个预测向量计算，纵轴仅用于事后评估。</text>',f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-85}" stroke="#74818b"/>',f'<line x1="{left}" y1="{h-85}" x2="{w-72}" y2="{h-85}" stroke="#74818b"/>',f'<text class="a" x="{(left+w-72)/2:.1f}" y="{h-35}" text-anchor="middle">GEARS–global-mean disagreement</text>',f'<text class="a" transform="translate(30 {(top+h-85)/2:.1f}) rotate(-90)" text-anchor="middle">GEARS ensemble RMSE</text>']
    for xx,yy,n in zip(sx,sy,df.perturbation): parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6" fill="#167c80" opacity=".83"><title>{escape(str(n))}</title></circle>')
    parts.append('</svg>'); (FIGURES/'F1_gears_baseline_disagreement_vs_error.svg').write_text('\n'.join(parts),encoding='utf-8')

def main() -> None:
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--n-boot',type=int,default=2000); args=p.parse_args()
    for d in [TABLES,REPORTS,FIGURES]: d.mkdir(parents=True,exist_ok=True)
    ensemble,truth,records=load_e60_ensemble(); test=set(ensemble); baseline,meta=source_global_effect(test)
    rows=[]
    for pert in sorted(ensemble):
        ge,tru=ensemble[pert],truth[pert]
        rows.append({"perturbation":pert,"error_gears_ensemble_rmse":rmse(ge,tru),"error_globalmean_rmse":rmse(baseline,tru),"risk_gears_globalmean_disagreement":rmse(ge,baseline),"risk_gears_predicted_magnitude":float(np.linalg.norm(ge)),"true_l2_diagnostic":float(np.linalg.norm(tru)),"training_support_for_heldout_gene":0})
    df=pd.DataFrame(rows); df.to_csv(TABLES/'E64_TASK_TABLE.csv',index=False)
    err=df.error_gears_ensemble_rmse.to_numpy(float); rng=np.random.default_rng(20260764); summary=[]
    for col,deploy in [('risk_gears_globalmean_disagreement',True),('risk_gears_predicted_magnitude',True),('true_l2_diagnostic',False)]:
        v=df[col].to_numpy(float); lo,hi=bootstrap_ci(v,err,rng,args.n_boot) if deploy else (np.nan,np.nan); k,enrich=top20(v,err) if deploy else (0,np.nan)
        summary.append({'predictor_under_audit':'GEARS_3seed_ensemble','second_predictor':'source_train_global_mean_effect','score_name':col,'deployable':deploy,'target_error':'error_gears_ensemble_rmse','n_tasks':len(df),'spearman':spearman(v,err),'bootstrap_rho_ci95_low':lo,'bootstrap_rho_ci95_high':hi,'top20_k':k,'top20_error_enrichment':enrich})
    s=pd.DataFrame(summary); s.to_csv(TABLES/'E64_RISK_ERROR_SUMMARY.csv',index=False); write_svg(df)
    status={'experiment':'E64_GEARS_vs_source_global_baseline','generated_at':now(),'git_head_before_run':git_head(),'source':'E60 strict GEARS outputs + Adamson GEARS processed training conditions','n_tasks':len(df),'test_conditions':sorted(test),'target_truth_used_in_scores':False,'baseline_definition':'mean effect across all non-control training conditions after removing all 24 fixed held-out conditions','baseline_meta':meta,'outputs':['tables/E64_TASK_TABLE.csv','tables/E64_RISK_ERROR_SUMMARY.csv','figures/F1_gears_baseline_disagreement_vs_error.svg','reports/E64_REPORT.md']}
    (OUT/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# E64｜GEARS 与 source-only 全局效应基线','', '## 设计','', 'E60 的三 seed GEARS 分歧没有风险信号。这里保留完全相同的 24 个固定未见基因任务和 GEARS ensemble，将第二预测器改为训练条件中所有非 control 扰动 effect 的平均值。这个 baseline 的计算时删除了全部 24 个 held-out conditions。','', '因此 `risk_gears_globalmean_disagreement` 在打分时只使用 GEARS ensemble 向量和训练域全局 effect；真实 held-out effect 只在最后计算 GEARS RMSE。','', '## 结果','', '| score | 可部署 | ρ(score, GEARS RMSE) | bootstrap 95% CI | top20 高误差富集 |','|---|---|---:|---:|---:|']
    for _,r in s.iterrows(): lines.append(f"| {r.score_name} | {'是' if r.deployable else '否（oracle）'} | {r.spearman:.3f} | {'—' if not r.deployable else f'[{r.bootstrap_rho_ci95_low:.3f}, {r.bootstrap_rho_ci95_high:.3f}]'} | {'—' if not r.deployable else f'{r.top20_error_enrichment:.3f}'} |")
    lines += ['', '## 边界','', '这不是第二个深度预测器，也不代表 scGPT/CPA 的结果。它是一个明确、可复算的独立基线，用来检测“GEARS 偏离训练域平均效应”是否与 GEARS 自身错误相关。','', '## 文件','', '- 任务表：`tables/E64_TASK_TABLE.csv`','- 汇总：`tables/E64_RISK_ERROR_SUMMARY.csv`','- 图：`figures/F1_gears_baseline_disagreement_vs_error.svg`']
    (REPORTS/'E64_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); (OUT/'README_先看这个.md').write_text('# E64 先看这个\n\n先读 `reports/E64_REPORT.md`。\n\n这是与 E60 同任务、同 GEARS 输出的独立 source-only global-effect baseline 审计。\n',encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
