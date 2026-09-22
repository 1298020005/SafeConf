#!/usr/bin/env python3
"""E201 target-background holdout for raw historical evidence ablation."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

TARGETS=('K562','RPE1','hepg2','jurkat')
LABEL='family_rms_error'
RAW_FEATURES={
 'M':['predicted_magnitude'],
 'MD':['predicted_magnitude','family_disagreement'],
 'MDH':['predicted_magnitude','family_disagreement','model_source_gap',
        'source_delta_dispersion','negative_log_source_cells','support_context_deficit','dispersion_imputed']}
INPUT_HASH=''

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def percentile_train(x): return pd.Series(x).rank(method='average',pct=True).to_numpy()
def score_model(train,test,arm):
    cols=RAW_FEATURES[arm]
    a=train[cols].astype(float).copy(); b=test[cols].astype(float).copy()
    a=a.replace([np.inf,-np.inf],np.nan); b=b.replace([np.inf,-np.inf],np.nan)
    means=a.mean(); a=a.fillna(means); b=b.fillna(means)
    scale=StandardScaler().fit(a)
    x=scale.transform(a); z=scale.transform(b)
    y=percentile_train(train[LABEL].to_numpy())
    model=Ridge(alpha=10.).fit(x,y)
    return model.predict(z),{'arm':arm,'features':cols,'means':means.to_dict(),'scale_mean':scale.mean_.tolist(),'scale_scale':scale.scale_.tolist(),'coef':model.coef_.tolist(),'intercept':float(model.intercept_)}
def weights(score,fraction):
    n=len(score); k=max(1,int(np.ceil(fraction*n))); q=np.partition(score,n-k)[n-k]
    hi=np.asarray(score)>q; eq=np.asarray(score)==q; w=hi.astype(float); w[eq]=(k-hi.sum())/eq.sum(); return w
def metric(score,y,b):
    w=weights(score,b); o=weights(y,b); avg=y.mean(); denom=np.dot(o,y)/o.sum()-avg
    selected=np.dot(w,y)
    return {'utility':float((selected/w.sum()-avg)/denom) if denom>1e-15 else np.nan,
            'spearman':float(np.corrcoef(rankdata(score),rankdata(y))[0,1]) if np.std(score)>0 and np.std(y)>0 else np.nan,
            'n_tasks':len(y),'selected':float(w.sum())}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); start=time.monotonic()
    if args.output.joinpath('RUN_STATUS.json').exists(): raise ValueError('refuse overwrite')
    raw=pd.read_csv(args.input)
    required=['task_id','target',LABEL,*sorted(set(sum(RAW_FEATURES.values(),[])))]
    if not set(required)<=set(raw.columns): raise ValueError('missing columns')
    frame=raw[required].copy(); frame.dispersion_imputed=frame.dispersion_imputed.astype(float)
    if len(frame)!=2008 or set(frame.target)!=set(TARGETS): raise ValueError('population changed')
    args.output.mkdir(parents=True,exist_ok=True); rows=[]; choices=[]
    for held in TARGETS:
        train=frame[frame.target!=held].reset_index(drop=True); test=frame[frame.target==held].reset_index(drop=True)
        scores={'magnitude':test.predicted_magnitude.to_numpy()}
        for arm in ('M','MD','MDH'):
            scores[arm],rec=score_model(train,test,arm); choices.append({'heldout':held,**rec})
        for method,s in scores.items():
            for b in (.1,.2,.3): rows.append({'heldout_target':held,'method':method,'budget':b,**metric(s,test[LABEL].to_numpy(),b)})
    block=pd.DataFrame(rows); study=block.copy()
    rng=np.random.default_rng(20260922); draws=rng.integers(0,4,size=(20000,4)); intervals=[]
    for b in (.1,.2,.3):
        for base in ('magnitude','M','MD'):
            bb=study[(study.method==base)&(study.budget==b)].set_index('heldout_target')
            for method in ('M','MD','MDH'):
                aa=study[(study.method==method)&(study.budget==b)].set_index('heldout_target')
                for metric_name in ('utility','spearman'):
                    d=(aa[metric_name]-bb[metric_name]).to_numpy(); boot=d[draws].mean(1)
                    intervals.append({'method':method,'baseline':base,'budget':b,'metric':metric_name,'delta':float(d.mean()),'ci95_lower':float(np.quantile(boot,.025)),'ci95_upper':float(np.quantile(boot,.975)),'positive_targets':int((d>0).sum()),'n_targets':4})
    intervals=pd.DataFrame(intervals); primary=intervals[(intervals.method=='MDH')&(intervals.baseline=='magnitude')&(intervals.budget==.2)&(intervals.metric=='utility')].iloc[0]
    passed=bool(primary.delta>=.02 and primary.positive_targets>=3 and primary.ci95_lower>0)
    status={'experiment':'E228_target_holdout_history_ablation','status':'COMPLETE','created_at':datetime.now().astimezone().isoformat(),'input_sha256':sha(args.input),'script_sha256':sha(Path(__file__)),'n_records':len(frame),'heldout_targets':list(TARGETS),'fits':12,'e208_rows_read':0,'development_gate':'PASS' if passed else 'NOT_SUPPORTED','primary':primary.to_dict(),'evidence_class':'retrospective_target_holdout_development'}
    for name,d in [('BLOCK_RESULTS',block),('INTERVALS',intervals)]: d.to_csv(args.output/(name+'.csv'),index=False)
    (args.output/'MODEL_CHOICES.json').write_text(json.dumps(choices,ensure_ascii=False,indent=2)+'\n')
    (args.output/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    text=['# E228：E201目标背景留出历史证据消融','','主候选MDH相对幅度的20%效用：'
        f'**{primary.delta:+.6f}**，描述性95%区间 [{primary.ci95_lower:+.6f}, {primary.ci95_upper:+.6f}]，正向目标 {int(primary.positive_targets)}/4。',
        f'预定开发门：**{status["development_gate"]}**。','',
        '每次留出一个完整目标背景；评分器只在另外三个目标的已知误差上拟合。四个目标和全部任务均保留。',
        'M只用预测幅度，MD增加分歧，MDH增加原始历史参考字段；没有输入E201已有safeconf组合分数。','']
    (args.output/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(status,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
