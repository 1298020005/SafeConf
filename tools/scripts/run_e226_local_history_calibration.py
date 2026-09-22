#!/usr/bin/env python3
"""Local error-history calibration; retrospective, with perturbation-group holdout."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_e225_raw_evidence_residual_ranker as core
SEEDS=(11,23,47)
BUDGETS=(.1,.25,.5,1.)

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def groups_for(perturbations,study):
    ordered=sorted(set(perturbations),key=lambda p:digest(f'{study}|fold|{p}'))
    return {p:i%5 for i,p in enumerate(ordered)}

def job(args):
    study_frame,study,out=args
    with threadpool_limits(limits=1):
        mapping=groups_for(study_frame.perturbation,study)
        labels=study_frame.perturbation.map(mapping)
        logs=[]; results=[]; fits=0
        for fold in range(5):
            test=study_frame.loc[labels.eq(fold)].reset_index(drop=True)
            rest=study_frame.loc[labels.ne(fold)]
            eligible=sorted(rest.perturbation.unique())
            for seed in SEEDS:
                ordered=sorted(eligible,key=lambda p:digest(f'{study}|{seed}|{p}'))
                for fraction in BUDGETS:
                    n=max(1,int(np.ceil(len(ordered)*fraction)))
                    chosen=set(ordered[:n])
                    train=rest.loc[rest.perturbation.isin(chosen)].reset_index(drop=True)
                    if set(train.perturbation)&set(test.perturbation):
                        raise ValueError('Perturbation group leakage')
                    scores={'magnitude':test.m.to_numpy()}
                    for arm in core.ARMS:
                        correction,_=core.fit_correction(train,test[core.FEATURES],arm,10.)
                        scores[arm]=test.m.to_numpy()+.5*correction
                        fits+=1
                    for method,score in scores.items():
                        table=core.evaluate(test,score,(.1,.2,.3))
                        table['method']=method
                        table['outer_fold']=fold
                        table['history_fraction']=fraction
                        table['history_seed']=seed
                        results.append(table)
                    logs.append({'dataset':study,'outer_fold':fold,'history_fraction':fraction,
                                 'history_seed':seed,'n_calibration_perturbations':n,
                                 'n_available_perturbations':len(eligible),'n_test_perturbations':test.perturbation.nunique(),
                                 'n_calibration_rows':len(train),'n_test_rows':len(test),
                                 'calibration_perturbations':sorted(chosen),
                                 'test_perturbations':sorted(test.perturbation.unique())})
        return pd.concat(results,ignore_index=True),logs,fits

def summarize(fold):
    metrics=['utility','capture','spearman','remaining_relative_error']
    # Each choice seed and outer fold receives equal weight within each study.
    means=fold.groupby(['dataset','method','history_fraction','history_seed','outer_fold','budget'])[metrics].mean().reset_index()
    study=means.groupby(['dataset','method','history_fraction','budget'])[metrics].mean().reset_index()
    ids=sorted(study.dataset.unique())
    draws=np.random.default_rng(20260922).integers(0,len(ids),size=(10000,len(ids)))
    rows=[]
    for frac in BUDGETS:
        for budget in (.1,.2,.3):
            sub=study[(study.history_fraction==frac)&(study.budget==budget)]
            for baseline in ('magnitude','M','MD','MDH'):
                b=sub[sub.method.eq(baseline)].set_index('dataset').loc[ids]
                for method in core.ARMS:
                    a=sub[sub.method.eq(method)].set_index('dataset').loc[ids]
                    for metric in metrics:
                        delta=(a[metric]-b[metric]).to_numpy()
                        boot=delta[draws].mean(axis=1)
                        rows.append({'history_fraction':frac,'budget':budget,'method':method,
                                     'baseline':baseline,'metric':metric,'delta':float(delta.mean()),
                                     'ci95_lower':float(np.quantile(boot,.025)),
                                     'ci95_upper':float(np.quantile(boot,.975)),
                                     'positive_studies':int((delta>0).sum()),'n_studies':len(ids)})
    return study,pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4,choices=range(1,5))
    args=p.parse_args()
    if core.sha(args.input)!=core.INPUT_HASH: raise ValueError('Input changed')
    if (args.output/'RUN_STATUS.json').exists(): raise ValueError('Existing result')
    start=time.monotonic()
    raw=pd.read_csv(args.input)
    frame=core.prepare(raw)
    frame['perturbation']=frame.task_instance_id.map(raw.set_index('task_instance_id').perturbation)
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(job,[(f.reset_index(drop=True),s,args.output) for s,f in frame.groupby('dataset')]):
            results.append(result)
            print('Completed',result[1][0]['dataset'],flush=True)
    blocks=pd.concat([r[0] for r in results],ignore_index=True)
    logs=[l for r in results for l in r[1]]
    study,intervals=summarize(blocks)
    for name,d in [('BLOCK_RESULTS',blocks),('STUDY_RESULTS',study),('INTERVALS',intervals)]:
        d.to_csv(args.output/(name+'.csv'),index=False)
    (args.output/'GROUP_SPLITS.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    principal=intervals[(intervals.history_fraction==.5)&(intervals.budget==.2)&(intervals.method=='MDH_condition')&(intervals.metric=='utility')]
    pass_gate=bool(principal[principal.baseline=='magnitude'].delta.iloc[0]>=.02 and (principal.ci95_lower>0).all())
    status={'experiment':'E226_local_history_calibration','status':'COMPLETE','created_at':datetime.now().astimezone().isoformat(),
            'evidence_class':'retrospective_local_calibration_not_zero_shot_or_external_confirmation',
            'development_gate':'PASS' if pass_gate else 'NOT_SUPPORTED',
            'input_sha256':core.sha(args.input),'script_sha256':core.sha(Path(__file__)),
            'helper_sha256':core.sha(Path(core.__file__)),
            'n_records':len(frame),'n_studies':4,'fits':sum(r[2] for r in results),
            'test_perturbation_labels_used_to_fit':False,'uses_local_historical_error_labels':True,
            'e208_rows_read':0,'all_studies_retained':True,'source_data_modified':False,
            'unscorable_singleton_block_rows':int(blocks.n_tasks.eq(1).sum()),
            'undefined_utility_rows':int(blocks.utility.isna().sum()),
            'elapsed_seconds':time.monotonic()-start,'primary':principal.to_dict(orient='records')}
    if status['input_sha256']!=core.INPUT_HASH: raise ValueError('Input changed during run')
    (args.output/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# E226：本地历史误差校准','',
           '本轮允许使用目标研究中其他扰动组的历史误差，五折按扰动名隔离；这改变了部署需求，不能与无本地标签结果混称。',
           '', '| 条件模型相对 | 50%可用历史组、20%复核效用差值 | 描述性95%区间 | 正向研究 |',
           '|---|---:|---|---:|']
    for r in principal.itertuples():
        lines.append(f'| {r.baseline} | {r.delta:+.5f} | [{r.ci95_lower:+.5f}, {r.ci95_upper:+.5f}] | {r.positive_studies}/4 |')
    lines+=['',f'预定开发门：**{status["development_gate"]}**。',
            '三种历史选择种子、全部四个预算和研究均保留；同一扰动跨背景的记录不会同时进入校准与测试。',
            '原上游数据协议未重训，分组约束仅针对新增评分器；不能宣传成重新做过完整的上游扰动留出。',
            f'不可定义效用行数：{status["undefined_utility_rows"]}，单任务块行数：{status["unscorable_singleton_block_rows"]}。这些行保留，不填造提升。','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
