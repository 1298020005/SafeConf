#!/usr/bin/env python3
"""E130: frozen high-error classifier evaluated on untouched Tian task truth."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/实验结果/E130_tian_confirmatory_high_error_router_20260714';TABLES=OUT/'tables';REPORTS=OUT/'reports';FIGURES=OUT/'figures';ERROR='error_two_predictor_mean_rmse';SEED=202607130;N_BOOT=10000
FEATURES={'safeconf_rank':'safeconf_calibrated_pair_risk','disagreement_rank':'risk_model_disagreement','magnitude_rank':'baseline_predicted_magnitude','context_novelty_rank':'context_novelty_scaled','perturbation_novelty_rank':'perturbation_novelty','low_support_rank':'low_support_risk'}
DEPLOY=['dataset','fold_id','task_id','perturbation',*set(FEATURES.values())-{'low_support_risk'},'training_support_count']
def source():
 paths=[('Frangieh',ROOT/'docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv'),(None,ROOT/'docs/实验结果/E112_external_formal_dual_models_20260713/E112_ALL_TASKS.csv'),('Shifrut',ROOT/'docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut/TASK_RISK_TABLE.csv'),('Liang',ROOT/'docs/实验结果/E123_liang_formal_dual_models_20260714/Liang/TASK_RISK_TABLE.csv')];parts=[]
 for name,p in paths:
  d=pd.read_csv(p)
  if name:d['dataset']=name
  parts.append(d)
 return pd.concat(parts,ignore_index=True,sort=False)
def transform(d,with_target):
 d=d.copy();d['low_support_risk']=-np.log1p(d.training_support_count.astype(float));g=d.groupby(['dataset','fold_id'],sort=False)
 for new,old in FEATURES.items():d[new]=g[old].rank(method='average',pct=True)
 if with_target:d['high_error_label']=(g[ERROR].rank(method='average',pct=True)>.80).astype(int)
 return d
def fit(train):
 foldn=train.groupby('dataset').fold_id.nunique();sizes=train.groupby(['dataset','fold_id']).size();w=np.asarray([1/(len(foldn)*foldn.loc[r.dataset]*sizes.loc[(r.dataset,r.fold_id)]) for r in train.itertuples(index=False)])*len(train)
 model=HistGradientBoostingClassifier(loss='log_loss',learning_rate=.05,max_iter=100,max_leaf_nodes=7,max_depth=3,min_samples_leaf=40,l2_regularization=10,monotonic_cst=[1]*len(FEATURES),random_state=SEED,early_stopping=False);model.fit(train[list(FEATURES)],train.high_error_label,sample_weight=w);return model
def triage(g,score):
 x=g.sort_values([score,'task_id'],kind='stable');full=float(g[ERROR].mean());cov=np.arange(.5,1.001,.05);ret=[]
 for c in cov:ret.append(float(x.iloc[:max(1,int(np.ceil(c*len(x))))][ERROR].mean()))
 n=max(1,int(np.ceil(.2*len(x))));high=x.iloc[-n:];rho=spearmanr(g[score],g[ERROR]).statistic
 return {'spearman':float(0 if not np.isfinite(rho) else rho),'normalized_aurc_50_100':float(np.trapezoid(ret,cov)/(cov[-1]-cov[0])/full),'top20_total_error_capture':float(high[ERROR].sum()/g[ERROR].sum()),'top20_error_enrichment':float(high[ERROR].mean()/full)}
def fast_boot_metrics(error,score,task_id):
 order=np.lexsort((task_id,score));y=error[order];full=float(error.mean());cov=np.arange(.5,1.001,.05);cs=np.cumsum(y);ret=[]
 for c in cov:
  n=min(len(y),max(1,int(np.ceil(c*len(y)))));ret.append(float(cs[n-1]/n))
 n=max(1,int(np.ceil(.2*len(y))));return {'normalized_aurc_50_100':float(np.trapezoid(ret,cov)/(cov[-1]-cov[0])/full),'top20_total_error_capture':float(y[-n:].sum()/error.sum())}
def main():
 for d in (TABLES,REPORTS,FIGURES):d.mkdir(parents=True,exist_ok=True)
 train=transform(source(),True);model=fit(train);target_path=ROOT/'docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi/TASK_RISK_TABLE.csv'
 available=['fold_id','task_id','perturbation','safeconf_calibrated_pair_risk','risk_model_disagreement','baseline_predicted_magnitude','context_novelty_scaled','perturbation_novelty','training_support_count'];deploy=pd.read_csv(target_path,usecols=available);deploy['dataset']='Tian_CRISPRi';deploy=transform(deploy,False);deploy['high_error_router_risk']=model.predict_proba(deploy[list(FEATURES)])[:,1];deploy['target_truth_used_for_score_or_transform']=False;deploy.to_csv(TABLES/'E130_RISK_SCORES_BEFORE_TRUTH.csv',index=False)
 truth=pd.read_csv(target_path,usecols=['fold_id','task_id','error_two_predictor_mean_rmse']);test=deploy.merge(truth,on=['fold_id','task_id'],validate='one_to_one');scores={'HighErrorRouter':'high_error_router_risk','SafeConf':'safeconf_calibrated_pair_risk','predicted_magnitude':'baseline_predicted_magnitude','model_disagreement':'risk_model_disagreement'};rows=[]
 for fold,g in test.groupby('fold_id'):
  for name,col in scores.items():rows.append({'fold_id':fold,'score':name,**triage(g,col)})
 folds=pd.DataFrame(rows);macro=folds.groupby('score',as_index=False)[['spearman','normalized_aurc_50_100','top20_total_error_capture','top20_error_enrichment']].mean();folds.to_csv(TABLES/'E130_FOLD_METRICS.csv',index=False);macro.to_csv(TABLES/'E130_MACRO.csv',index=False);pd.DataFrame({'feature':list(FEATURES),'importance_note':['monotonic constrained; tree models have no stable linear coefficient']*len(FEATURES)}).to_csv(TABLES/'E130_MODEL_SPEC.csv',index=False)
 # Conservative fold bootstrap. Perturbation clustering is preserved by resampling folds, then perturbations within each fold.
 rng=np.random.default_rng(SEED);metrics=['normalized_aurc_50_100','top20_total_error_capture'];samples={(m,c):[] for m in metrics for c in ['SafeConf','predicted_magnitude','model_disagreement']};fold_ids=sorted(test.fold_id.unique());cache={}
 for fold,g in test.groupby('fold_id',sort=True):
  g=g.reset_index(drop=True);cluster=[np.flatnonzero(g.perturbation.astype(str).to_numpy()==p) for p in g.perturbation.astype(str).unique()];cache[fold]={'cluster':cluster,'error':g[ERROR].to_numpy(float),'task_id':g.task_id.astype(str).to_numpy(),**{s:g[col].to_numpy(float) for s,col in scores.items()}}
 for _ in range(N_BOOT):
  chosen=rng.choice(fold_ids,len(fold_ids),replace=True);vals={m:{s:[] for s in scores} for m in metrics}
  for fold in chosen:
   c=cache[fold];cluster=c['cluster'];ix=np.concatenate([cluster[int(i)] for i in rng.integers(0,len(cluster),len(cluster))])
   for s,col in scores.items():
    z=fast_boot_metrics(c['error'][ix],c[s][ix],c['task_id'][ix])
    for m in metrics:vals[m][s].append(z[m])
  for m in metrics:
   router=float(np.mean(vals[m]['HighErrorRouter']))
   for c in ['SafeConf','predicted_magnitude','model_disagreement']:
    other=float(np.mean(vals[m][c]));samples[(m,c)].append(other-router if m=='normalized_aurc_50_100' else router-other)
 point=macro.set_index('score');boot=[]
 for (m,c),v in samples.items():
  delta=float(point.loc[c,m]-point.loc['HighErrorRouter',m] if m=='normalized_aurc_50_100' else point.loc['HighErrorRouter',m]-point.loc[c,m]);x=np.asarray(v);boot.append({'metric':m,'comparator':c,'favorable_delta':delta,'ci95_low':float(np.quantile(x,.025)),'ci95_high':float(np.quantile(x,.975)),'probability_favorable':float(np.mean(x>0))})
 boot=pd.DataFrame(boot);boot.to_csv(TABLES/'E130_BOOTSTRAP.csv',index=False);primary=boot[boot.metric.isin(metrics)];point_ok=all(point.loc['HighErrorRouter','normalized_aurc_50_100']<point.loc[c,'normalized_aurc_50_100'] and point.loc['HighErrorRouter','top20_total_error_capture']>point.loc[c,'top20_total_error_capture'] for c in ['SafeConf','predicted_magnitude']);ci_ok=any((primary[primary.comparator.eq(c)].ci95_low>0).all() for c in ['SafeConf','predicted_magnitude']);passed=bool(point_ok and ci_ok)
 lines=['# E130｜Tian 第六数据确认性高错误路由','','风险分数在读取 Tian 任务真值前写入 `tables/E130_RISK_SCORES_BEFORE_TRUTH.csv`。模型只用前五套历史数据。','','| score | Spearman↑ | normalized AURC↓ | top-20% capture↑ | top-20% enrichment↑ |','|---|---:|---:|---:|---:|']
 for r in macro.itertuples(index=False):lines.append(f'| {r.score} | {r.spearman:.4f} | {r.normalized_aurc_50_100:.4f} | {r.top20_total_error_capture:.4f} | {r.top20_error_enrichment:.4f} |')
 lines+=['','## Fold × perturbation-cluster bootstrap','','正的 favorable delta 表示 HighErrorRouter 更好。','','| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |','|---|---|---:|---:|---:|']
 for r in boot.itertuples(index=False):lines.append(f'| {r.metric} | {r.comparator} | {r.favorable_delta:.4f} | [{r.ci95_low:.4f}, {r.ci95_high:.4f}] | {r.probability_favorable:.3f} |')
 lines+=['','## 预设判定','',f"- 通过：**{'是' if passed else '否'}**。",'- Tian 的 context 是技术批次，因此这里验证的是批次域偏移，不是新细胞类型泛化。'];(REPORTS/'E130_REPORT.md').write_text('\n'.join(lines)+'\n');status={'experiment':'E130_tian_confirmatory_high_error_router','generated_at':datetime.now().isoformat(timespec='seconds'),'status':'complete','n_source_datasets':5,'n_source_tasks':len(train),'n_target_folds':int(test.fold_id.nunique()),'n_target_tasks':len(test),'target_truth_used_for_model_score_transform_or_threshold':False,'preregistered_gate_passed':passed,'model_spec':'E127 frozen HistGradientBoostingClassifier'};(OUT/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');(OUT/'README_先看这个.md').write_text('# E130 先看这个\n\n先读 `reports/E130_REPORT.md`。\n');print(json.dumps(status,ensure_ascii=False,indent=2));print(macro.to_string(index=False));print(boot.to_string(index=False))
if __name__=='__main__':main()
