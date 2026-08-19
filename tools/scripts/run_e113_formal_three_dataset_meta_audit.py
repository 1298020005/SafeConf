#!/usr/bin/env python3
"""E113: three-dataset meta-audit of formal scGPT/GEARS SafeConf outputs."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"docs/实验结果/E113_formal_three_dataset_meta_audit_20260713";TABLES=OUT/"tables";REPORTS=OUT/"reports";FIGURES=OUT/"figures"
E108=ROOT/"docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv";E112=ROOT/"docs/实验结果/E112_external_formal_dual_models_20260713/E112_ALL_TASKS.csv"
PRIMARY="safeconf_calibrated_pair_risk";COMPARATORS=["safeconf_frozen_pair_risk","risk_model_disagreement","baseline_predicted_magnitude"]
# The cross-dataset primary endpoint is the two-predictor mean error.  Model-
# specific secondary endpoints already receive a 4,000-draw audit in E111;
# repeating them here would multiply runtime without changing the protocol.
TARGETS=["error_two_predictor_mean_rmse"];SEED=202607113;N_BOOT=2000
def rho(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
 if m.sum()<3 or np.unique(a[m]).size<2 or np.unique(b[m]).size<2:return float("nan")
 return float(np.corrcoef(rankdata(a[m]),rankdata(b[m]))[0,1])
def load():
 f=pd.read_csv(E108);f["dataset"]="Frangieh";e=pd.read_csv(E112);return pd.concat([f,e],ignore_index=True,sort=False)
def summary(data):
 rows=[]
 for (dataset,fold),g in data.groupby(["dataset","fold_id"],sort=True):
  for target in TARGETS:
   for score in [PRIMARY,*COMPARATORS]:rows.append({"dataset":dataset,"fold_id":fold,"target":target,"score":score,"n_tasks":len(g),"spearman":rho(g[score],g[target])})
 return pd.DataFrame(rows)
def cache(data):
 result={}
 for dataset,dg in data.groupby("dataset"):
  result[dataset]={}
  for fold,g in dg.groupby("fold_id"):
   g=g.reset_index(drop=True);clusters=[np.flatnonzero(g.perturbation.astype(str).to_numpy()==p) for p in sorted(g.perturbation.astype(str).unique())];result[dataset][str(fold)]=(g,clusters)
 return result
def sampled(fc,target,comparator,rng):
 values=[];folds=sorted(fc)
 for fold in rng.choice(folds,len(folds),replace=True):
  g,clusters=fc[str(fold)];ix=np.concatenate([clusters[int(i)] for i in rng.integers(0,len(clusters),len(clusters))]);values.append(rho(g[PRIMARY].to_numpy(float)[ix],g[target].to_numpy(float)[ix])-rho(g[comparator].to_numpy(float)[ix],g[target].to_numpy(float)[ix]))
 return float(np.nanmean(values))
def observed(fc,target,comparator):
 return float(np.nanmean([rho(g[PRIMARY],g[target])-rho(g[comparator],g[target]) for g,_ in fc.values()]))
def bootstrap(data):
 c=cache(data);datasets=sorted(c);rng=np.random.default_rng(SEED);rows=[]
 for target in TARGETS:
  for comparator in COMPARATORS:
   obs={d:observed(c[d],target,comparator) for d in datasets};by={d:[] for d in datasets};fixed=[];population=[]
   for _ in range(N_BOOT):
    one={d:sampled(c[d],target,comparator,rng) for d in datasets}
    for d in datasets:by[d].append(one[d])
    fixed.append(float(np.mean(list(one.values()))));draw=rng.choice(datasets,len(datasets),replace=True);population.append(float(np.mean([sampled(c[str(d)],target,comparator,rng) for d in draw])))
   for d in datasets:
    x=np.asarray(by[d]);rows.append({"scope":d,"target":target,"comparator":comparator,"unit":"outer_fold_plus_perturbation_cluster","delta":obs[d],"ci95_low":float(np.quantile(x,.025)),"ci95_high":float(np.quantile(x,.975)),"p_gt_zero":float(np.mean(x>0))})
   for unit,x in [("fixed_three_datasets",np.asarray(fixed)),("dataset_population_plus_fold_plus_perturbation",np.asarray(population))]:rows.append({"scope":"three_dataset_macro","target":target,"comparator":comparator,"unit":unit,"delta":float(np.mean(list(obs.values()))),"ci95_low":float(np.quantile(x,.025)),"ci95_high":float(np.quantile(x,.975)),"p_gt_zero":float(np.mean(x>0))})
 return pd.DataFrame(rows)
def lodo(s):
 macro=s.groupby(["dataset","target","score"],as_index=False).spearman.mean();rows=[]
 for target in TARGETS:
  for removed in sorted(macro.dataset.unique()):
   v=macro[(macro.target==target)&(macro.dataset!=removed)].groupby("score").spearman.mean()
   for comp in COMPARATORS:rows.append({"target":target,"removed_dataset":removed,"comparator":comp,"delta":float(v[PRIMARY]-v[comp])})
 return pd.DataFrame(rows)
def figure(boot):
 d=boot[(boot.target=="error_two_predictor_mean_rmse")&(boot.comparator=="baseline_predicted_magnitude")&boot.unit.isin(["outer_fold_plus_perturbation_cluster","fixed_three_datasets"])].copy();order=["Frangieh","Lara_exvivo","Santinha","three_dataset_macro"];d["o"]=d.scope.map({x:i for i,x in enumerate(order)});d=d.sort_values("o");w,h,left,right=1050,500,250,80;lo,hi=-.4,.4;sx=lambda v:left+(v-lo)/(hi-lo)*(w-left-right);parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">','<rect width="100%" height="100%" fill="#fff"/>','<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#27343c}.t{font-size:24px;font-weight:700}.s{font-size:14px;fill:#647078}.l{font-size:16px}</style>','<text x="45" y="40" class="t">E113｜正式三数据集：SafeConf 相对 magnitude</text>','<text x="45" y="67" class="s">目标为双模型平均 RMSE；点为 fold 宏平均 ΔSpearman，线为聚类 bootstrap 95% CI。</text>',f'<line x1="{sx(0):.1f}" y1="90" x2="{sx(0):.1f}" y2="410" stroke="#88949a" stroke-dasharray="5 5"/>']
 for i,r in enumerate(d.itertuples(index=False)):
  y=125+i*75;label="三数据集固定集合" if r.scope=="three_dataset_macro" else r.scope;parts += [f'<text x="45" y="{y+5}" class="l">{label}</text>',f'<line x1="{sx(r.ci95_low):.1f}" y1="{y}" x2="{sx(r.ci95_high):.1f}" y2="{y}" stroke="#396d83" stroke-width="4"/>',f'<circle cx="{sx(r.delta):.1f}" cy="{y}" r="7" fill="#285f78"/>',f'<text x="{w-35}" y="{y+5}" text-anchor="end" class="s">{r.delta:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}]</text>']
 parts.append('</svg>');(FIGURES/"F1_formal_meta_forest.svg").write_text("\n".join(parts))
def main():
 for d in (TABLES,REPORTS,FIGURES):d.mkdir(parents=True,exist_ok=True)
 data=load();s=summary(data);b=bootstrap(data);l=lodo(s);s.to_csv(TABLES/"E113_FOLD_SUMMARY.csv",index=False);b.to_csv(TABLES/"E113_BOOTSTRAP.csv",index=False);l.to_csv(TABLES/"E113_LODO.csv",index=False);figure(b);macro=s.groupby(["dataset","target","score"],as_index=False).spearman.mean();macro.to_csv(TABLES/"E113_DATASET_MACRO.csv",index=False)
 status={"experiment":"E113_formal_three_dataset_meta_audit","generated_at":datetime.now().isoformat(timespec="seconds"),"status":"complete","datasets":sorted(data.dataset.unique()),"n_datasets":data.dataset.nunique(),"n_folds":data.fold_id.nunique(),"n_test_task_rows":len(data),"primary":PRIMARY,"test_truth_used_to_change_score":False,"n_bootstrap":N_BOOT};(OUT/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
 mean=macro[macro.target=="error_two_predictor_mean_rmse"].pivot(index="dataset",columns="score",values="spearman").reset_index();lines=["# E113｜三套正式 scGPT–GEARS 数据集元分析","","三套数据均使用同背景输入、端到端微调 scGPT 和训练背景专属共表达图 GEARS。E113 不重拟合分数，只在每个 fold 内计算相关，再对 fold 和数据集等权平均。","","## 双模型平均误差","","| dataset | SafeConf calibrated | frozen | disagreement | magnitude |","|---|---:|---:|---:|---:|"]
 for r in mean.itertuples(index=False):lines.append(f"| {r.dataset} | {r.safeconf_calibrated_pair_risk:.3f} | {r.safeconf_frozen_pair_risk:.3f} | {r.risk_model_disagreement:.3f} | {r.baseline_predicted_magnitude:.3f} |")
 lines += ["","## Bootstrap","","| scope | target | comparator | unit | Δρ | 95% CI | P(Δ>0) |","|---|---|---|---|---:|---:|---:|"]
 for r in b.itertuples(index=False):lines.append(f"| {r.scope} | {r.target} | {r.comparator} | {r.unit} | {r.delta:.3f} | [{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.p_gt_zero:.3f} |")
 lines += ["","Santinha 是明确的外部失败边界。三数据集固定集合与 dataset-population 区间必须分别解释；只有三个数据集时，后者通常很宽，不能声称对未来数据集已有稳定保证。"]
 (REPORTS/"E113_REPORT.md").write_text("\n".join(lines)+"\n");(OUT/"README_先看这个.md").write_text("# E113 先看这个\n\n先读 `reports/E113_REPORT.md`。\n");print(json.dumps(status,ensure_ascii=False,indent=2));print(mean.to_string(index=False));print(b[(b.scope=="three_dataset_macro")&(b.target=="error_two_predictor_mean_rmse")].to_string(index=False))
if __name__=="__main__":main()
