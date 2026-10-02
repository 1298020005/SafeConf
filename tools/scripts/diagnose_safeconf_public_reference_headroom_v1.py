#!/usr/bin/env python3
"""One CPU-only DEV reconstruction diagnostic; no risk score or model fitting.

The unrestricted oracle uses the exact registered legal query/history views.
Evaluated biological truth chooses unattainable oracle weights for explanation
only. No TEST inputs, upstream fitting, GPU operations or parameter search.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from tools.scripts import run_safeconf_publicset_v1 as pub

OUT=pub.OUT/'reference_headroom_v1'

def binding(p):
    p=Path(p)
    return {'path':str(p),'sha256':pub.file_hash(p)}

def balanced_means(frame, cols):
    # Each gene has the same total weight, independent of query count.
    gene=frame.groupby('gene',sort=True)[cols].mean()
    return gene.mean().to_dict()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds',type=float,default=300.)
    args=parser.parse_args()
    if not 0<args.max_seconds<=300:raise ValueError('fixed CPU wall budget must be <=300 seconds')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RESULT_MANIFEST.json').exists():raise FileExistsError('preserve completed/failed diagnostic; no automatic reruns')
    torch.set_num_threads(4)
    started=time.monotonic();cpu_start=time.process_time();deadline=started+args.max_seconds
    fixed=pub.OUT/'full_v2_resume'
    previous=pd.read_csv(fixed/'CONSTRAINED_RECONSTRUCTION_DIAGNOSTIC.csv')
    scores=pd.read_csv(fixed/'TASK_PREDICTIONS.csv.gz',usecols=['domain','task_id','builder','seed','bio_rmse'])
    identity=['domain','task_id','builder','seed']
    spread=scores.groupby(identity).bio_rmse.agg(['min','max'])
    if np.max(np.abs(spread['max']-spread['min']))>1e-14:raise RuntimeError('biology reference changed by upstream')
    scores=scores.drop_duplicates(identity)
    scores['mse']=scores.bio_rmse**2
    actual=scores[scores.builder.isin(['B0_SupportMean','B1_HGB','B2_Pointwise','B3_DeepSets'])].groupby(
        ['domain','task_id','builder']).mse.mean().unstack('builder')
    old=previous.set_index(['domain','task_id'])
    registration={'role':'DEV_SEEN_OUTCOME_INFORMED_RECONSTRUCTION_DIAGNOSTIC',
        'scope':'Source1808/575 and McFaline DEV542/377; exact original2840 axis, outer query/history views',
        'unrestricted_constraint':'w>=0,sum(w)=1; no support floor',
        'best_single':'minimum full-effect MSE among exact legal histories; outcome-informed diagnostic only',
        'solver':{'method':'SLSQP','ftol':1e-10,'maxiter':500,'initialization':'registered support weights'},
        'CPU_seconds_cap':args.max_seconds,'threads':4,'new_model_fits':0,'GPU_hours':0,
        'MC_TEST_reads':0,'new_upstream_training':0,'risk_scores_generated':0,
        'aggregation':'equal total gene weights; macro context×fold and per-query means additionally shown',
        'biological_truth_use':'Existing DEV truth for unattainable reconstruction explanation only; never prediction/model selection',
        'original_constrained_oracle_binding':binding(fixed/'CONSTRAINED_RECONSTRUCTION_DIAGNOSTIC.csv'),
        'actual_biology_scores_binding':binding(fixed/'TASK_PREDICTIONS.csv.gz'),
        'public_runner_binding':binding(pub.__file__),'diagnostic_code_binding':binding(__file__),
        'pid':os.getpid(),'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    pub.atomic_json(OUT/'REGISTRATION.json',registration)
    pub.atomic_json(OUT/'RUN_STATUS.json',registration|{'status':'RUNNING'})
    rows=[];domains=[];failed=None
    try:
        for name in ('Source','McFaline'):
            d=pub.load_domain(name)
            expected=1808 if name=='Source' else 542
            if len(d.tasks)!=expected:raise RuntimeError('registered DEV cohort changed')
            domains.append({'domain':name,'n_queries':len(d.tasks),'n_genes':d.tasks.gene.nunique(),
                'gene_fold_hash':pub.fingerprint(d.tasks.gene.astype(str)+'|'+d.tasks.fold.astype(str))})
            for fold in range(5):
                query=np.flatnonzero(d.tasks.fold.to_numpy()==fold)
                forbidden=d.forbidden(query)
                groups=d.groups(query,forbidden)
                for g in groups:
                    if time.monotonic()>deadline:raise TimeoutError('registered CPU diagnostic wall budget reached')
                    q=g['q'];task=d.tasks.iloc[q];ix=g['ix'];k=len(ix)
                    if k==0:raise RuntimeError('unexpected missing historical query in fixed full cohort')
                    h=d.effects[ix].astype(float);y=d.truth[q].astype(float);s=g['s'];G=h.shape[1]
                    gram=h@h.T/G;target=h@y/G;constant=float(y@y/G)
                    def objective(w):return float(w@gram@w-2*w@target+constant)
                    def gradient(w):return 2*(gram@w-target)
                    single=np.mean((h-y)**2,axis=1);best_single=int(np.argmin(single))
                    if k==1:
                        weights=np.ones(1);success=True;message='single legal history';iterations=0
                    else:
                        result=minimize(objective,s,method='SLSQP',jac=gradient,bounds=[(0.,1.)]*k,
                            constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
                            options={'ftol':1e-10,'maxiter':500})
                        weights=result.x;success=bool(result.success and abs(weights.sum()-1)<1e-6 and weights.min()>=-1e-7)
                        message=str(result.message);iterations=int(result.nit)
                    support=float(np.mean((s@h-y)**2))
                    restricted=old.loc[(name,str(task.task_id))]
                    if int(restricted.n_history)!=k or abs(support-restricted.support_mean_mse)>1e-10:
                        raise RuntimeError('exact legal history/support reconstruction changed from original oracle')
                    mse=objective(weights) if success else np.nan
                    grad=gradient(weights);gap=max(0,float(grad@weights-grad.min())) if success else np.nan
                    r={'domain':name,'fold':fold,'context':str(task.context),'task_id':str(task.task_id),'gene':str(task.gene),
                        'n_history':k,'legal_history_ids_hash':pub.fingerprint(d.memory.iloc[ix].experiment_id),
                        'support_mse':support,'restricted_oracle_mse':float(restricted.constrained_best_mse),
                        'unrestricted_oracle_mse':mse,'unrestricted_oracle_lower_bound_mse':max(0,mse-gap) if success else np.nan,
                        'unrestricted_optimality_gap_bound':gap,'best_single_mse':float(single[best_single]),
                        'best_single_experiment_id':str(d.memory.iloc[ix[best_single]].experiment_id),
                        'solver_success':success,'solver_message':message,'solver_iterations':iterations,
                        'simplex_sum':float(weights.sum()),'simplex_min_weight':float(weights.min()),
                        'restricted_better_than_unrestricted_above_1e_8':bool(success and mse>restricted.constrained_best_mse+1e-8),
                        'diagnostic_only':True,'risk_oracle_claim':False}
                    for builder in ('B0_SupportMean','B1_HGB','B2_Pointwise','B3_DeepSets'):
                        r[builder+'_mse']=float(actual.loc[(name,str(task.task_id)),builder])
                    rows.append(r)
                pub.atomic_csv(OUT/'PER_QUERY_RECONSTRUCTION.csv',pd.DataFrame(rows))
                print(json.dumps({'completed':name+'/fold'+str(fold),'queries':len(rows),'elapsed_seconds':time.monotonic()-started}),flush=True)
    except Exception as error:
        failed=repr(error)
    frame=pd.DataFrame(rows)
    if not frame.empty:pub.atomic_csv(OUT/'PER_QUERY_RECONSTRUCTION.csv',frame)
    cols=['support_mse','restricted_oracle_mse','unrestricted_oracle_mse','unrestricted_oracle_lower_bound_mse',
        'unrestricted_optimality_gap_bound','best_single_mse','B1_HGB_mse','B2_Pointwise_mse','B3_DeepSets_mse']
    summaries=[]
    for name,part in frame.groupby('domain'):
        for mode,means in [('equal_gene',balanced_means(part,cols)),('equal_query',part[cols].mean().to_dict()),
                           ('equal_context_fold',part.groupby(['context','fold'])[cols].mean().mean().to_dict())]:
            row={'domain':name,'aggregation':mode,'n_tasks':len(part),'n_genes':part.gene.nunique(),
                 'solver_failures':int((~part.solver_success).sum()),'max_optimality_gap_bound':float(part.unrestricted_optimality_gap_bound.max()),**means}
            for col in cols:
                if col.endswith('_mse'):row[col.removesuffix('_mse')+'_root_mean_square']=float(np.sqrt(max(means[col],0)))
            den=means['support_mse']-means['unrestricted_oracle_mse']
            row['support_to_unrestricted_relative_MSE_reduction']=den/means['support_mse']
            row['restricted_to_unrestricted_relative_MSE_reduction']=(means['restricted_oracle_mse']-means['unrestricted_oracle_mse'])/means['restricted_oracle_mse']
            for builder in ('B1_HGB','B2_Pointwise','B3_DeepSets'):
                row[builder+'_fraction_of_unrestricted_headroom_closed']=(means['support_mse']-means[builder+'_mse'])/den if den>1e-14 else np.nan
            summaries.append(row)
    summary=pd.DataFrame(summaries);pub.atomic_csv(OUT/'AGGREGATE_RECONSTRUCTION.csv',summary)
    manifest=registration|{'status':'COMPLETE' if failed is None and len(frame)==2350 else 'FAILED_OR_PARTIAL',
        'failure':failed,'domains':domains,'queries_completed':len(frame),
        'solver_failures':int((~frame.solver_success).sum()) if not frame.empty else None,
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu_start,
        'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'solver_ordering_violations_above_1e_8':int(frame.restricted_better_than_unrestricted_above_1e_8.sum()) if not frame.empty else None,
        'result_bindings':[binding(OUT/x) for x in ('PER_QUERY_RECONSTRUCTION.csv','AGGREGATE_RECONSTRUCTION.csv')]}
    pub.atomic_json(OUT/'RESULT_MANIFEST.json',manifest);pub.atomic_json(OUT/'RUN_STATUS.json',manifest)
    lines=['# Public reference unrestricted headroom: DEV diagnostic','',
        'Unrestricted simplex and best-single use evaluated biological truth only as unattainable explanatory optima. No fitted model, inference risk score or adopted version changes.',
        '',f"Status: {manifest['status']}; completed queries {len(frame)}/2350; solver failures {manifest['solver_failures']}; wall {manifest['wall_seconds']:.2f}s.",'',
        'Gene-balanced reconstruction MSE:', '',
        '| Domain | Support | Restricted oracle | Unrestricted oracle | Best single | B2 mean over3seeds | B3 mean over3seeds | B2 headroom closed | B3 headroom closed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        if row['aggregation']!='equal_gene':continue
        lines.append(f"| {row['domain']} | {row['support_mse']:.8f} | {row['restricted_oracle_mse']:.8f} | {row['unrestricted_oracle_mse']:.8f} | {row['best_single_mse']:.8f} | {row['B2_Pointwise_mse']:.8f} | {row['B3_DeepSets_mse']:.8f} | {row['B2_Pointwise_fraction_of_unrestricted_headroom_closed']:.1%} | {row['B3_DeepSets_fraction_of_unrestricted_headroom_closed']:.1%} |")
    lines += ['', 'SLSQP returns numerical optima under fixed tolerance. Per-query Frank-Wolfe dual gaps provide an additional upper bound on the unresolved objective error; failures and ordering violations are retained.',
        'These diagnostic reconstruction limits do not bound risk-ranking Utility@20 and do not identify why risk transfer failed. A remaining reconstruction gap cannot justify outcome-driven model selection or another architecture search.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':manifest['status'],'queries':len(frame),'solver_failures':manifest['solver_failures'],
        'wall_seconds':manifest['wall_seconds'],'cpu_seconds':manifest['cpu_seconds'],'summary':summaries},ensure_ascii=False),flush=True)
    if failed:raise RuntimeError(failed)

if __name__=='__main__':main()
