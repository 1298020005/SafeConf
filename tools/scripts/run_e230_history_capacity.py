#!/usr/bin/env python3
"""Real source-cell subsampling with frozen E201 predictions; no label fitting."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_e201_pretruth_task_base as original
import run_e225_raw_evidence_residual_ranker as scoring

SEEDS=(11,23,47)
FRACTIONS=(.25,.5,1.)
FEATURE_HASH='003d7dec2912f7f24b39665c3166c25cffb4670b16579e354b23764f90a2caf6'
METRIC_HASH='4a02d132cae1605a2f6f54bee8912f45e835d1fc3936e6cda0fcc98e2c6bcd35'
SCORES=('magnitude','disagreement','history_risk','m_plus_d','m_plus_history')
COMPONENTS=('family_disagreement','model_source_gap','source_delta_dispersion',
            'negative_log_source_cells','support_context_deficit')

def sha(p): return original.sha256_file(Path(p))
def now(): return datetime.now().astimezone().isoformat()
def seed_for(text): return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')
def ordered_sources(target,seed):
    return sorted(set(original.TARGETS)-{target},key=lambda c:seed_for(f'{target}|{seed}|source|{c}'))
def subset_order(n,identity,seed):
    return np.random.default_rng(seed_for(f'{identity}|cells|{seed}')).permutation(n)
def rank(v): return rankdata(v,method='average')/len(v)

def score_features(frame):
    f=frame.copy(); primary=f.analysis_stratum.eq('primary_ge30').to_numpy()
    if not primary.any(): raise ValueError('No primary tasks')
    for col in ('model_source_gap','source_delta_dispersion'):
        observed=f[col].dropna()
        f[col]=f[col].fillna(float(observed.median()) if len(observed) else 0.)
    z=[]
    for col in COMPONENTS:
        x=f[col].to_numpy(float); ref=x[primary]; sd=ref.std()
        z.append((x-ref.mean())/sd if sd>1e-14 else np.zeros(len(x)))
    s=np.mean(z,axis=0); m=rank(f.predicted_magnitude); d=rank(f.family_disagreement)
    f['magnitude']=m; f['disagreement']=d; f['history_risk']=rank(s)
    f['m_plus_d']=.8*m+.2*d
    f['m_plus_history']=np.where(f.n_source_contexts.to_numpy()>0,.8*m+.2*rank(s),m)
    return f

def extract(target,tasks,cache):
    path=cache/f'E201_blind_{target}'/'de_adata_test.h5ad'
    manifest=path.parent/'E201_BLIND_VIEW_MANIFEST.json'
    if sha(path)!=original.TRAINING_VIEW_SHA256[target]: raise ValueError('Source H5 changed')
    if sha(manifest)!=original.TRAINING_MANIFEST_SHA256[target]: raise ValueError('Source manifest changed')
    metadata=json.loads(manifest.read_text())
    if metadata['n_target_treatments']!=0 or metadata['target']!=target: raise ValueError('Manifest isolation')
    a=ad.read_h5ad(path,backed='r'); obs=a.obs.copy()
    if ((obs.cell_line.astype(str)==target)&(~obs.control.astype(bool))).any():
        a.file.close(); raise ValueError('Target perturbation in source view')
    deltas={}; counts={}; audit=[]
    try:
        for context in sorted(set(original.TARGETS)-{target}):
            mask=obs.cell_line.astype(str).eq(context)
            ctrl=np.flatnonzero(mask & obs.control.astype(bool))
            idx=np.flatnonzero(mask & ~obs.control.astype(bool) & obs.condition.astype(str).isin(tasks.condition))
            xc=sparse.csr_matrix(a.X[ctrl]); xp=sparse.csr_matrix(a.X[idx])
            cb=obs.iloc[ctrl].batch.astype(str).to_numpy()
            ctrls={b:original.row_mean(xc,np.flatnonzero(cb==b)) for b in sorted(set(cb))}
            conds=obs.iloc[idx].condition.astype(str).to_numpy(); batches=obs.iloc[idx].batch.astype(str).to_numpy()
            deltas[context]={}; counts[context]={}
            for condition in sorted(set(conds)):
                loc=np.flatnonzero(conds==condition); counts[context][condition]=len(loc)
                record={}
                for seed in SEEDS:
                    order=subset_order(len(loc),f'{target}|{context}|{condition}',seed)
                    sizes={f'f{frac:g}':max(1,int(np.ceil(len(loc)*frac))) for frac in FRACTIONS}
                    sizes.update({f'n{n}':n for n in (10,15,30) if len(loc)>=n})
                    by_size={}
                    for key,n in sizes.items():
                        if n not in by_size:
                            chosen=loc[order[:n]]
                            # Full-data mean uses original order for numerical parity.
                            if n==len(loc): chosen=loc
                            bs,ns=np.unique(batches[chosen],return_counts=True)
                            if any(b not in ctrls for b in bs): raise ValueError('Missing matched control')
                            c=sum(ctrls[b]*int(v) for b,v in zip(bs,ns))/len(chosen)
                            by_size[n]=original.row_mean(xp,chosen)-c
                        record[(seed,key)]=(by_size[n].astype(np.float32),n)
                deltas[context][condition]=record
            audit.append({'target':target,'source_context':context,'controls_read':len(ctrl),
                          'perturbed_cells_read':len(idx),'target_perturbed_cells_read':0})
            print('Extracted',target,context,len(idx),'source cells',flush=True)
    finally:
        a.file.close()
    return deltas,counts,audit

def assemble(tasks,delta,source,chosen,seed,mode,frac,k,eligible):
    rows=[]
    for i,t in enumerate(tasks.itertuples(index=False)):
        if mode=='matched30' and t.condition not in eligible: continue
        values=[]; n=0
        key=f'f{frac:g}' if mode=='fraction' else f'n{30//k}'
        for context in chosen:
            rec=source[context].get(t.condition)
            if rec is not None and (seed,key) in rec:
                v,ni=rec[(seed,key)]; values.append(v); n+=ni
        members=np.asarray(values,dtype=float)
        center=members.mean(axis=0) if len(values) else None
        gap=float(np.sqrt(np.mean((delta[i]-center)**2))) if len(values) else np.nan
        h=float(np.sqrt(np.mean((members-center)**2))) if len(values)>=2 else np.nan
        if mode=='matched30' and n!=30: raise ValueError('Matched budget broken')
        rows.append({'task_id':t.task_id,'target':t.target,'condition':t.condition,
            'analysis_stratum':t.analysis_stratum,'predicted_magnitude':t.predicted_magnitude,
            'family_disagreement':t.family_disagreement,'model_source_gap':gap,
            'source_delta_dispersion':h,'negative_log_source_cells':-np.log1p(n),
            'support_context_deficit':3-len(values),'n_source_contexts':len(values),
            'n_source_cells':n,'dispersion_imputed':len(values)<2})
    return score_features(pd.DataFrame(rows))

def target_job(args):
    target,tasks,delta,cache,out=args
    with threadpool_limits(limits=1):
        source,counts,audit=extract(target,tasks,cache)
        eligible={c for c in tasks.condition if all(counts[s].get(c,0)>=30 for s in counts)}
        frames=[]; checks=[]
        for seed in SEEDS:
            order=ordered_sources(target,seed)
            for k in (1,2,3):
                for mode,frac in [('fraction',f) for f in FRACTIONS]+[('matched30',1.)]:
                    f=assemble(tasks,delta,source,order[:k],seed,mode,frac,k,eligible)
                    f['history_mode']=mode; f['fraction']=frac; f['seed']=seed; f['k_sources']=k
                    f['chosen_sources']='|'.join(order[:k]); frames.append(f)
                    if mode=='fraction' and k==3 and frac==1:
                        for col in ('model_source_gap','source_delta_dispersion','negative_log_source_cells','support_context_deficit'):
                            error=float(np.max(np.abs(f[col].to_numpy()-tasks[col].to_numpy())))
                            if error>1e-5: raise ValueError(f'Full-source parity failed {target} {col}: {error}')
                            checks.append({'target':target,'seed':seed,'feature':col,'max_abs_difference':error})
        table=pd.concat(frames,ignore_index=True); file=out/f'FEATURES_{target}.csv.gz'
        table.to_csv(file,index=False,compression={'method':'gzip','mtime':0})
        return {'target':target,'file':file.name,'sha256':sha(file),'rows':len(table),
                'eligible_matched_tasks':len(eligible),'eligible_matched_conditions':sorted(eligible),
                'source_audit':audit,'parity_checks':checks}

def evaluate(out,seal,truth):
    if sha(truth)!=METRIC_HASH: raise ValueError('Truth changed')
    labels=pd.read_csv(truth).set_index('task_id')['family_rms_error']; rows=[]
    for item in seal:
        path=out/item['file']
        if sha(path)!=item['sha256']: raise ValueError('Feature seal changed')
        data=pd.read_csv(path)
        for keys,group in data.groupby(['target','history_mode','fraction','seed','k_sources'],sort=True):
            for scope in ('primary','all'):
                f=group[group.analysis_stratum.eq('primary_ge30')] if scope=='primary' else group
                y=labels.loc[f.task_id].to_numpy()
                for method in SCORES:
                    s=f[method].to_numpy(); sy,ss=rankdata(y),rankdata(s)
                    rho=float(np.corrcoef(ss,sy)[0,1]) if np.std(ss)>0 and np.std(sy)>0 else np.nan
                    for budget in (.1,.2,.3):
                        rows.append(dict(zip(['target','history_mode','fraction','seed','k_sources'],keys))|
                            {'scope':scope,'method':method,'budget':budget,'spearman':rho,'n_tasks':len(f),
                             'coverage':float(f.n_source_contexts.gt(0).mean()),**scoring.metrics(s,y,budget)})
    return pd.DataFrame(rows)

def contrasts(table):
    a=table[(table.scope=='primary')&(table.budget==.2)]
    def value(mode,frac,k,method):
        return a[(a.history_mode==mode)&(a.fraction==frac)&(a.k_sources==k)&(a.method==method)].groupby('target').utility.mean().sort_index()
    full=value('fraction',1.,3,'m_plus_history')
    diffs={'cell_quantity_full_minus_quarter':full-value('fraction',.25,3,'m_plus_history'),
           'diversity_3_minus_1_total30':value('matched30',1.,3,'m_plus_history')-value('matched30',1.,1,'m_plus_history'),
           'full_history_minus_magnitude':full-value('fraction',1.,3,'magnitude')}
    rng=np.random.default_rng(20260922); draws=rng.integers(0,4,(20000,4)); rows=[]
    for name,d in diffs.items():
        if len(d)!=4 or not np.isfinite(d).all(): raise ValueError('Missing target contrast')
        boot=d.to_numpy()[draws].mean(axis=1)
        rows.append({'contrast':name,'delta':float(d.mean()),'ci95_lower':float(np.quantile(boot,.025)),
                     'ci95_upper':float(np.quantile(boot,.975)),'positive_targets':int((d>0).sum()),
                     **{'delta_'+t:float(v) for t,v in d.items()}})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','e201-docs','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'RUN_STATUS.json').exists(): raise ValueError('Refuse existing run')
    feature_path=args.e201_docs/'tables/E201_PRETRUTH_RISK_FEATURES.csv'
    if sha(feature_path)!=FEATURE_HASH: raise ValueError('Feature contract changed')
    tasks=pd.read_csv(feature_path)
    if len(tasks)!=2008 or tasks.analysis_stratum.eq('primary_ge30').sum()!=1808: raise ValueError('Task population')
    vectors=args.data_root/'txpert_official_20260802/e201/pretruth_vectors'
    status=json.loads((args.e201_docs/'E201_PRETRUTH_RISK_STATUS.json').read_text())
    for name in ('E201_FAMILY_CENTROIDS.npy','E201_CONTROL_CENTROIDS.npy'):
        record=next(v for v in status['vector_files'] if v['path'].endswith(name))
        if sha(vectors/name)!=record['sha256']: raise ValueError('Frozen vector changed')
    delta=np.load(vectors/'E201_FAMILY_CENTROIDS.npy').astype(float)-np.load(vectors/'E201_CONTROL_CENTROIDS.npy').astype(float)
    if not np.allclose(np.sqrt((delta**2).mean(axis=1)),tasks.predicted_magnitude,atol=1e-5,rtol=0): raise ValueError('Prediction axis')
    run={'status':'RUNNING','created_at':now(),'script_sha256':sha(Path(__file__)),
        'feature_sha256':sha(feature_path),'workers':args.workers,'e208_rows_read':0,
        'design':'fixed_source_capacity_not_fitted_error_ranker','source':[]}
    original.atomic_json(out/'RUN_STATUS.json',run)
    jobs=[(t,tasks[tasks.target.eq(t)].reset_index(drop=True),delta[tasks.target.eq(t)],
           args.data_root/'txpert_official_20260802/cache',out) for t in original.TARGETS]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool: seal=list(pool.map(target_job,jobs))
        original.atomic_json(out/'FEATURE_SEAL.json',{'sealed_at':now(),'files':seal,'contains_target_errors':False})
        table=evaluate(out,seal,args.e201_docs/'formal_core_evaluation/tables/E201_TASK_METRICS.csv')
        table.to_csv(out/'RESULTS.csv',index=False); diffs=contrasts(table); diffs.to_csv(out/'CONTRASTS.csv',index=False)
        indexed=diffs.set_index('contrast'); full=indexed.loc['full_history_minus_magnitude']
        passed=bool((diffs.delta>0).all() and (diffs.positive_targets>=3).all() and full.delta>=.02 and full.ci95_lower>0)
        run.update(status='COMPLETE',finished_at=now(),source=seal,development_gate='PASS' if passed else 'NOT_SUPPORTED',
                   n_tasks=2008,n_primary=1808,n_feature_records=sum(s['rows'] for s in seal),
                   predictions_unchanged=True,target_feature_truth_reads=0,primary=diffs.to_dict('records'))
        lines=['# E230：真实历史细胞与来源覆盖实验','',
               '所有上游预测固定；真实抽样来源细胞；未拟合误差评分器。四目标全部保留。','',
               '| 预定主要比较 | 效用差值 | 描述性95%区间 | 正向目标 |','|---|---:|---|---:|']
        for r in diffs.itertuples(): lines.append(f'| {r.contrast} | {r.delta:+.6f} | [{r.ci95_lower:+.6f}, {r.ci95_upper:+.6f}] | {r.positive_targets}/4 |')
        lines+=['',f'开发门：**{run["development_gate"]}**。',
                '主要范围为1808个primary任务；匹配多样性限于三个来源各至少30细胞的元数据子群。',
                '效用是固定预算捕获误差的相对位置，不是预测准确率或湿实验节省比例。',
                '同一目标内历史容量干预能比较，不代表增加独立外部研究，也不能按最优配置改主结果。','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    except BaseException as error:
        run.update(status='FAILED',finished_at=now(),reason=repr(error)); raise
    finally: original.atomic_json(out/'RUN_STATUS.json',run)
    print(json.dumps({k:v for k,v in run.items() if k!='source'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
