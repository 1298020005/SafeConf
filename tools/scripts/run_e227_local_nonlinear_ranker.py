#!/usr/bin/env python3
"""Fixed nonlinear controls after E226; historical development only."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_e225_raw_evidence_residual_ranker as core
import run_e226_local_history_calibration as local

FAMILIES=('ridge','tree_direct','tree_residual')
PRIMARY='tree_direct_MDH_condition'

def inputs(features,arm):
    cols=['m'] if arm=='M' else ['m','d'] if arm=='MD' else ['m','d','cn','scarcity']
    x=features[cols].to_numpy(float)
    if arm=='MDH_condition':
        x=np.column_stack([x,*[(features.setting==s).to_numpy(float) for s in core.SETTINGS]])
    return x

def predict(train,test_features,family,arm):
    if family=='ridge':
        correction,_=core.fit_correction(train,test_features,arm,10.)
        return test_features.m.to_numpy()+.5*correction
    y,w=core.train_arrays(train)
    if family=='tree_direct': y=y+train.m.to_numpy()
    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,learning_rate=.05,
        min_samples_leaf=20,l2_regularization=10.,early_stopping=False,random_state=20260922)
    model.fit(inputs(train[core.FEATURES],arm),y,sample_weight=w)
    pred=model.predict(inputs(test_features[core.FEATURES],arm))
    return pred if family=='tree_direct' else test_features.m.to_numpy()+.5*pred

def job(args):
    frame,study=args
    with threadpool_limits(limits=1):
        mapping=local.groups_for(frame.perturbation,study)
        folds=frame.perturbation.map(mapping)
        results=[]; nfits=0
        for fold in range(5):
            test=frame.loc[folds.eq(fold)].reset_index(drop=True)
            rest=frame.loc[folds.ne(fold)]
            for seed in local.SEEDS:
                order=sorted(rest.perturbation.unique(),key=lambda p:local.digest(f'{study}|{seed}|{p}'))
                selected=set(order[:max(1,int(np.ceil(len(order)*.5)))])
                train=rest[rest.perturbation.isin(selected)].reset_index(drop=True)
                if set(train.perturbation)&set(test.perturbation): raise ValueError('Group leakage')
                scores={'magnitude':test.m.to_numpy()}
                for family in FAMILIES:
                    for arm in core.ARMS:
                        scores[family+'_'+arm]=predict(train,test[core.FEATURES],family,arm)
                        nfits+=1
                for method,score in scores.items():
                    table=core.evaluate(test,score,(.1,.2,.3))
                    table['method']=method; table['outer_fold']=fold; table['history_seed']=seed
                    results.append(table)
        return pd.concat(results,ignore_index=True),nfits

def summarize(blocks):
    metrics=['utility','capture','spearman','remaining_relative_error']
    partial=blocks.groupby(['dataset','method','history_seed','outer_fold','budget'])[metrics].mean().reset_index()
    study=partial.groupby(['dataset','method','budget'])[metrics].mean().reset_index()
    ids=sorted(study.dataset.unique())
    draws=np.random.default_rng(20260922).integers(0,len(ids),size=(10000,len(ids)))
    rows=[]
    for budget in (.1,.2,.3):
        sub=study[study.budget.eq(budget)]
        for baseline in ('magnitude','tree_direct_M','tree_direct_MD','tree_direct_MDH','ridge_MDH_condition'):
            b=sub[sub.method.eq(baseline)].set_index('dataset').loc[ids]
            for method in sorted(set(sub.method)-{'magnitude'}):
                a=sub[sub.method.eq(method)].set_index('dataset').loc[ids]
                for metric in metrics:
                    delta=(a[metric]-b[metric]).to_numpy()
                    boot=delta[draws].mean(axis=1)
                    rows.append({'method':method,'baseline':baseline,'budget':budget,'metric':metric,
                        'delta':float(delta.mean()),'ci95_lower':float(np.quantile(boot,.025)),
                        'ci95_upper':float(np.quantile(boot,.975)),
                        'positive_studies':int((delta>0).sum()),'n_studies':len(ids)})
    return study,pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    args=p.parse_args()
    if core.sha(args.input)!=core.INPUT_HASH: raise ValueError('Input changed')
    if (args.output/'RUN_STATUS.json').exists(): raise ValueError('Cannot overwrite a finished run')
    t=time.monotonic(); raw=pd.read_csv(args.input); frame=core.prepare(raw)
    frame['perturbation']=frame.task_instance_id.map(raw.set_index('task_instance_id').perturbation)
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(job,[(f.reset_index(drop=True),s) for s,f in frame.groupby('dataset')]))
    blocks=pd.concat([r[0] for r in results],ignore_index=True)
    study,intervals=summarize(blocks)
    for name,d in [('BLOCK_RESULTS',blocks),('STUDY_RESULTS',study),('INTERVALS',intervals)]:
        d.to_csv(args.output/(name+'.csv'),index=False)
    principal=intervals[(intervals.method==PRIMARY)&(intervals.budget==.2)&(intervals.metric=='utility')]
    primary=principal.set_index('baseline')
    passed=bool(primary.loc['magnitude','delta']>=.02 and primary.loc['magnitude','positive_studies']>=3
                and (primary.ci95_lower>0).all())
    status={'experiment':'E227_local_nonlinear_ranker','status':'COMPLETE',
        'created_at':datetime.now().astimezone().isoformat(),'input_sha256':core.sha(args.input),
        'script_sha256':core.sha(Path(__file__)),'helper_sha256':core.sha(Path(core.__file__)),
        'split_helper_sha256':core.sha(Path(local.__file__)),
        'fits':sum(r[1] for r in results),'elapsed_seconds':time.monotonic()-t,
        'n_records':len(frame),'n_studies':4,'primary_method':PRIMARY,
        'feature_allowlist':core.FEATURES,'evidence_class':'retrospective_development',
        'e208_rows_read':0,'target_calibrated_score_used_as_feature':False,
        'development_gate':'PASS' if passed else 'NOT_SUPPORTED','primary':principal.to_dict(orient='records')}
    (args.output/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# E227：本地校准的非线性对照','',
        '固定参数；主要候选为直接预测误差百分位的树＋全部条件特征。全部4个研究保留。','',
        '| 主要候选相对 | 20%效用差值 | 描述性95%区间 | 正向研究 |','|---|---:|---|---:|']
    for name,r in primary.iterrows():
        lines.append(f'| {name} | {r.delta:+.6f} | [{r.ci95_lower:+.6f}, {r.ci95_upper:+.6f}] | {int(r.positive_studies)}/4 |')
    lines+=['',f'预定开发门：**{status["development_gate"]}**。',
        '', '720次拟合使用同一组历史记录，不是720个独立实验。未重训上游，不是全管线未见基因验证。',
        '树是自建通用对照，不是PertEMA软件复现。所有臂见CSV；不把最好的辅助臂改称注册主要结果。','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
