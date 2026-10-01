#!/usr/bin/env python3
"""Uncertainty of existing fixed Source fits on already released McFaline tasks.

No models, Source labels, CDFs or Orion inputs are fitted/read here. The same
biological-gene multiplicities resample every fixed seed/order/method cell.
Curves average metrics of the fixed cells, never pool duplicate prediction rows.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='4'
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import metrics

RUNTIME=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
RESULTS=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis/results'
DOC=RESULTS/'source_scaling_uncertainty'
SEEDS=(20260930,20261001,20261002)
ORDERS=tuple(range(5))
BUDGETS=(.1,.25,.5,.75,1.)
REFS=('Manual','Learned')
VARIANTS=('TxPert_GAT','TxPert_Exphormer','PooledEqualRecords','PooledFullClusterWeight','SeparateRiskAverage')
METRICS=('utility20','spearman','aurc','high_risk_miss_rate','error_at_10','error_at_20','error_at_50')
REPLICATES=5000
BOOTSTRAP_SEED=20260930
META=['task_id','target','gene','true_error_rmse']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def weighted_midrank(values,weights):
    order=np.argsort(values,kind='stable')
    sorted_values=values[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_values)!=0)+1]
    sizes=np.diff(np.r_[starts,len(values)])
    frequency=np.add.reduceat(weights[:,order],starts,axis=1)
    ranks=np.cumsum(frequency,axis=1)-.5*(frequency-1)
    return np.repeat(ranks,sizes,axis=1)[:,np.argsort(order)]


def counter_metrics(truth,ids,score,weights,original_valid=True):
    """Exact scalar metrics of expanded task copies using their multiplicities."""
    truth=np.asarray(truth,float);ids=np.asarray(ids,str);score=np.asarray(score,float)
    weights=np.asarray(weights,dtype=np.int32)
    if (not np.isfinite(truth).all() or not np.isfinite(score).all()
        or weights.ndim!=2 or weights.shape[1]!=len(truth) or (weights<0).any()
        or len(set(ids))!=len(ids)):
        raise RuntimeError('Complete finite unique task cohort and nonnegative multiplicities required')
    total=weights.sum(axis=1,dtype=np.int32)
    result=np.full((len(weights),len(METRICS)),np.nan)
    with np.errstate(divide='ignore',invalid='ignore'):
        mean=(weights@truth)/total
        def take(order,fraction):
            counts=weights[:,order]
            before=np.cumsum(counts,axis=1,dtype=np.int32)-counts
            k=np.ceil(fraction*total).astype(np.int32)
            selected=np.minimum(counts,np.maximum(k[:,None]-before,0))
            accepted=(selected@truth[order])/k
            original=np.empty_like(selected);original[:,order]=selected
            return accepted,original,k
        oracle,oracle_counts,k=take(np.lexsort((ids,-truth)),.2)
        high,high_counts,_=take(np.lexsort((ids,-score)),.2)
        denominator=oracle-mean
        valid=(total>=20)&(denominator>1e-12)&bool(original_valid)
        np.divide(high-mean,denominator,out=result[:,0],where=valid)
        risk_rank=weighted_midrank(score,weights)
        truth_rank=weighted_midrank(truth,weights)
        center=(total[:,None]+1)/2
        a=risk_rank-center;b=truth_rank-center
        covariance=np.sum(weights*a*b,axis=1)
        norm=np.sqrt(np.sum(weights*a*a,axis=1)*np.sum(weights*b*b,axis=1))
        np.divide(covariance,norm,out=result[:,1],where=(total>=3)&(norm>0))
        low=np.lexsort((ids,score));counts=weights[:,low];error=truth[low]
        after=np.cumsum(counts,axis=1,dtype=np.int32);before=after-counts
        prior_sum=np.cumsum(counts*error,axis=1)-counts*error
        harmonic=np.r_[0.,np.cumsum(1/np.arange(1,int(total.max(initial=0))+1))]
        block=counts*error+(prior_sum-before*error)*(harmonic[after]-harmonic[before])
        result[:,2]=np.sum(block,axis=1)/total
        result[:,3]=1-np.minimum(high_counts,oracle_counts).sum(axis=1)/k
        for index,fraction in enumerate([.1,.2,.5],4):result[:,index]=take(low,fraction)[0]
    result[total==0]=np.nan
    return result


def synthetic_verify():
    rng=np.random.default_rng(83);maximum=0.;comparisons=0
    for _ in range(6):
        n=42;truth=rng.integers(1,10,n)/10;score=rng.integers(0,8,n)/7
        ids=np.array([f'T{i:03d}' for i in rng.permutation(n)])
        weights=np.vstack([np.ones(n,dtype=int),rng.integers(0,5,(8,n))])
        actual=counter_metrics(truth,ids,score,weights)
        for index,w in enumerate(weights):
            use=np.repeat(np.arange(n),w)
            frame=pd.DataFrame({'task_id':ids[use],'true_error_rmse':truth[use]})
            expected=np.array([metrics(frame,score[use])[key] for key in METRICS])
            if not np.allclose(actual[index],expected,atol=1e-12,rtol=0,equal_nan=True):
                raise RuntimeError('Count helper does not reproduce scalar expanded-copy metrics')
            maximum=max(maximum,float(np.nanmax(np.abs(actual[index]-expected))));comparisons+=len(METRICS)
    ids=np.array([str(i) for i in range(19)])
    frozen=counter_metrics(np.arange(19.),ids,np.arange(19.),np.full((1,19),2),original_valid=False)
    if not np.isnan(frozen[0,0]):raise RuntimeError('Bootstrap duplicates rescued original invalid U20 cohort')
    return {'expanded_scalar_metric_comparisons':comparisons,'maximum_absolute_difference':maximum,
        'original_n19_U20_stays_invalid_when_duplicated_to38':True}


def load_cells(path,group_columns,expected_keys,query=None):
    data=pd.read_csv(path)
    if set(data.line)!={'TxPert_to_McFaline'}:raise RuntimeError('Only saved common-axis McFaline evaluation is allowed')
    groups={key:part for key,part in data.groupby(group_columns,sort=True)}
    if set(groups)!=set(expected_keys):raise RuntimeError('Fixed registered source design cells differ')
    cells=[];keys=[]
    for key in expected_keys:
        part=groups[key].sort_values('task_id').reset_index(drop=True)
        if not part.task_id.is_unique:raise RuntimeError('Original prediction cell has duplicate tasks')
        if query is None:query=part[META].copy()
        if not part[META].equals(query):raise RuntimeError('Every design cell must use identical complete task identities/truth')
        score=part.risk.to_numpy(float)
        if not np.isfinite(score).all():raise RuntimeError('No nonfinite score exclusion or survivor selection allowed')
        cells.append(score);keys.append(key)
    return query,np.column_stack(cells),keys,data


def interval_records(point,draws,extra):
    records=[]
    for index,name in enumerate(METRICS):
        values=draws[:,index];finite=values[np.isfinite(values)]
        lo,hi=np.quantile(finite,[.025,.975]) if len(finite)>=2 else (np.nan,np.nan)
        records.append(extra|{'metric':name,'point_estimate':float(point[index]),
            'bootstrap_mean':float(finite.mean()) if len(finite) else np.nan,
            'ci95_lower':lo,'ci95_upper':hi,'valid_draws':len(finite),
            'bootstrap_replicates':REPLICATES,'bootstrap_seed':BOOTSTRAP_SEED})
    return records


def run(output):
    began=time.monotonic();output=Path(output).resolve()
    if not output.is_relative_to(DOC.resolve()):raise RuntimeError('Output must use new uncertainty documentation scope')
    if (output/'STATUS.json').exists():raise RuntimeError('Completed statistics cannot be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    stage=output/f'.incomplete.{os.getpid()}';stage.mkdir()
    def timeout(signum,frame):raise TimeoutError('Statistics exceeded fixed20-minute resource budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
    source=RUNTIME/'source_scaling_recheck/SOURCE_SCALING_TASK_PREDICTIONS.csv.gz'
    diversity=RUNTIME/'source_diversity_seed_completion/TASK_PREDICTIONS.csv.gz'
    originals=[RESULTS/'source_scaling_recheck/SOURCE_SCALING_MACRO.csv',RESULTS/'source_diversity_seed_completion/MACRO.csv']
    proof=synthetic_verify()
    scaling_keys=list(itertools.product([ref+'_hgb' for ref in REFS],SEEDS,BUDGETS,ORDERS))
    diversity_keys=list(itertools.product([ref+'/'+variant for ref in REFS for variant in VARIANTS],SEEDS))
    query,scaling_scores,scaling_keys,scaling_data=load_cells(source,['method','seed','budget','order'],scaling_keys)
    query,diversity_scores,diversity_keys,diversity_data=load_cells(diversity,['method','seed'],diversity_keys,query)
    contexts=sorted(query.target.unique())
    if contexts!=['a172','t98g','u87mg'] or not np.isfinite(query.true_error_rmse).all():raise RuntimeError('Registered three-context finite query cohort required')
    counts=query.groupby('target').size().to_dict()
    if min(counts.values())<20:raise RuntimeError('All original contexts must have at least20tasks before bootstrap')
    score_matrix=np.column_stack([scaling_scores,diversity_scores])
    # Equal arrays are computed once; every fixed cell retains its own index.
    unique=[];mapping=[];hashes={}
    for score in score_matrix.T:
        digest=hashlib.sha256(score.tobytes()).hexdigest()
        if digest not in hashes:hashes[digest]=len(unique);unique.append(score)
        if not np.array_equal(score,unique[hashes[digest]]):raise RuntimeError('Score hash collision')
        mapping.append(hashes[digest])
    unique=np.column_stack(unique);mapping=np.array(mapping)
    genes=sorted(query.gene.unique());gene_index={gene:i for i,gene in enumerate(genes)}
    query_gene=np.array([gene_index[gene] for gene in query.gene])
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    gene_counts=np.stack([np.bincount(rng.integers(0,len(genes),len(genes)),minlength=len(genes)) for _ in range(REPLICATES)]).astype(np.uint16)
    draws=np.zeros((REPLICATES,len(unique.T),len(METRICS)))
    point=np.zeros((len(unique.T),len(METRICS)))
    for context in contexts:
        mask=query.target.eq(context).to_numpy();part=query.loc[mask].reset_index(drop=True)
        multiplicity=gene_counts[:,query_gene[mask]]
        truth=part.true_error_rmse.to_numpy();ids=part.task_id.to_numpy(str)
        for index,score in enumerate(unique[mask].T):
            scalar=np.array([metrics(part,score)[name] for name in METRICS])
            if not np.isfinite(scalar).all():raise RuntimeError('An original planned context metric is invalid; no context omission allowed')
            check=counter_metrics(truth,ids,score,np.ones((1,len(part)),dtype=int))[0]
            if not np.allclose(scalar,check,atol=1e-12,rtol=0,equal_nan=True):raise RuntimeError('Original scalar endpoint reproduction failed')
            point[index]+=scalar/len(contexts)
            draws[:,index]+=counter_metrics(truth,ids,score,multiplicity)/len(contexts)
        print(f'Completed common gene draws for {context}; {len(unique.T)} exact unique saved score arrays',flush=True)
    all_point=point[mapping];all_draws=draws[:,mapping]
    reproduction=[];cell_records=[]
    for family,keys,offset,old_path in [('scaling',scaling_keys,0,originals[0]),('diversity',diversity_keys,len(scaling_keys),originals[1])]:
        old=pd.read_csv(old_path)
        for index,key in enumerate(keys):
            identity=dict(zip(['method','seed','budget','order'] if family=='scaling' else ['method','seed'],key))
            matching=np.ones(len(old),bool)
            for name,value in identity.items():matching&=old[name].eq(value).to_numpy()
            if matching.sum()!=1:raise RuntimeError('Original fixed point table cell absent or duplicate')
            previous=old.loc[matching,list(METRICS)].to_numpy()[0]
            if not np.allclose(previous,all_point[offset+index],atol=1e-12,rtol=0,equal_nan=True):
                raise RuntimeError('Original point values or finite/NA status did not reproduce')
            difference=np.nanmax(np.abs(previous-all_point[offset+index]))
            if difference>1e-12:raise RuntimeError('Saved original point table did not reproduce')
            reproduction.append({'family':family}|identity|{'maximum_metric_absolute_difference':difference})
            cell_records.append({'family':family}|identity|dict(zip(METRICS,all_point[offset+index])))
    curve_rows=[];curve_pairs=[];diversity_rows=[];diversity_pairs=[];worst=[]
    common={'n_tasks':len(query),'n_gene_clusters':len(genes),'n_contexts':len(contexts),
        'macro':'equal context mean','CI_condition':'fixed saved predictions; genes are sampling units, seeds/orders are fixed design cells'}
    for ref in REFS:
        means={};samples={}
        for budget in BUDGETS:
            selected=[i for i,key in enumerate(scaling_keys) if key[0]==ref+'_hgb' and key[2]==budget]
            means[budget]=all_point[selected].mean(axis=0);samples[budget]=all_draws[:,selected].mean(axis=1)
            curve_rows.extend(interval_records(means[budget],samples[budget],common|{'reference':ref,'budget':budget,'fixed_orders':5,'fixed_seeds':3}))
            for metric_index,name in enumerate(METRICS):
                values=all_point[selected,metric_index]
                worst.append({'reference':ref,'budget':budget,'metric':name,
                    'fixed_cell_minimum':values.min(),'fixed_cell_maximum':values.max(),
                    'fixed_cell_mean':values.mean(),'all15_fixed_cells_retained':True,
                    'worst_fixed_cell':values.min() if name in ['utility20','spearman'] else values.max(),
                    'best_sequence_used_for_primary':False})
        pairs=[(1.,budget,'full_minus_lower') for budget in BUDGETS[:-1]]+[(b,a,'adjacent_higher_minus_lower') for a,b in zip(BUDGETS[:-1],BUDGETS[1:])]
        for a,b,label in pairs:
            curve_pairs.extend(interval_records(means[a]-means[b],samples[a]-samples[b],common|{'reference':ref,'budget_a':a,'budget_b':b,'contrast':label,'difference':'metric_a_minus_b'}))
        base={};sample={}
        for variant in VARIANTS:
            selected=[len(scaling_keys)+i for i,key in enumerate(diversity_keys) if key[0]==ref+'/'+variant]
            base[variant]=all_point[selected].mean(axis=0);sample[variant]=all_draws[:,selected].mean(axis=1)
            diversity_rows.extend(interval_records(base[variant],sample[variant],common|{'reference':ref,'variant':variant,'fixed_seeds':3}))
        for a,b in itertools.combinations(VARIANTS,2):
            diversity_pairs.extend(interval_records(base[a]-base[b],sample[a]-sample[b],common|{'reference':ref,'variant_a':a,'variant_b':b,'difference':'metric_a_minus_b'}))
    for name,rows in [('SCALING_CURVE_INTERVALS.csv',curve_rows),('SCALING_PAIRED_BUDGET_CHANGES.csv',curve_pairs),
        ('DIVERSITY_INTERVALS.csv',diversity_rows),('DIVERSITY_ALL_PAIRED_CONTRASTS.csv',diversity_pairs),
        ('ALL_FIXED_ORDER_SEED_POINTS.csv',cell_records),('SCALING_FIXED_ORDER_RANGE.csv',worst),('POINT_REPRODUCTION.csv',reproduction)]:
        pd.DataFrame(rows).to_csv(stage/name,index=False)
    pd.DataFrame({'gene_cluster_index':range(len(genes)),'gene':genes}).to_csv(stage/'GENE_CLUSTER_AXIS.csv',index=False)
    registration={'role':'SEEN_POST_CONFIRMATION_FIXED_PREDICTION_STATISTICS','common_gene_axis':2840,
        'bootstrap_replicates':REPLICATES,'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_unit':'complete biological gene cluster jointly across all3contexts',
        'shared_draw_across_all_budgets_orders_seeds_and_diversity_methods':True,
        'gene_draw_counts_sha256':hashlib.sha256(gene_counts.tobytes()).hexdigest(),'gene_draw_counts_dtype':'uint16 little-endian C-order5000x380',
        'curve_estimand':'mean of original context-macro metrics over fixed5orders and3seeds; not metric of averaged scores or pooled duplicate rows',
        'diversity_estimand':'mean of original context-macro metrics over fixed3seeds',
        'all_original_contexts_n20_checked_before_resampling':True,'metric_definitions_unchanged':True,
        'nominal_percentile_CIs':95,'no_best_order_or_source_winner_selected':True,
        'inputs':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [source,diversity,*originals]],
        'code_sha256':sha(__file__),'research_metrics_sha256':sha(ROOT/'tools/safeconf_continual/research.py'),
        'new_fits':0,'new_upstream_calls':0,'new_error_labels':0,'Orion_access':False,'CPU_time_limit_seconds':1200,'thread_limits':4}
    write_json(stage/'REGISTRATION.json',registration)
    status={'status':'COMPLETE_FIXED_SOURCE_UNCERTAINTY','n_tasks':len(query),'n_biological_gene_clusters':len(genes),
        'contexts_task_counts':counts,'scaling_cells':len(scaling_keys),'diversity_cells':len(diversity_keys),
        'exact_unique_saved_score_arrays':len(unique.T),'bootstrap_replicates':REPLICATES,
        'all_original_points_reproduced':True,'maximum_point_absolute_difference':max(x['maximum_metric_absolute_difference'] for x in reproduction),
        'all5000_U20_draws_valid':bool(np.isfinite(draws[:,:,0]).all()),'synthetic_counter_proof':proof,
        'elapsed_seconds':time.monotonic()-began,'process_CPU_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        'peak_RSS_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'new_fits':0,'new_upstream_calls':0,'new_error_labels':0,'Orion_access':False,'original_outputs_preserved':True,
        'interpretation':'Conditional uncertainty of existing predictions, not variation over newly trained models or independent seeds/datasets'}
    write_json(stage/'STATUS.json',status)
    for path in stage.iterdir():
        if (output/path.name).exists():raise RuntimeError('Final uncertainty artifact already exists')
    for path in stage.iterdir():path.rename(output/path.name)
    stage.rmdir();signal.alarm(0)
    write_json(output/'OWNED_FILES.json',[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [Path(__file__),*sorted(output.glob('*.csv')),*sorted(output.glob('*.json'))]])
    print(json.dumps(status,indent=2),flush=True);return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DOC)
    parser.add_argument('--synthetic-only',action='store_true')
    args=parser.parse_args()
    if args.synthetic_only:print(json.dumps(synthetic_verify(),indent=2))
    else:run(args.output)
