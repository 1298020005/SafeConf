#!/usr/bin/env python3
"""Complete six fixed Source HGB-vs-Magnitude contrasts using saved gene counts."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import os
import resource
import signal
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from tools.safeconf_continual.research import metrics
from tools.scripts.run_safeconf_source_scaling_uncertainty_agent import counter_metrics,METRICS

R=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001'
D=R/'source_magnitude_comparator_completion_v1'
B=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
O=B/'source_magnitude_comparator_completion_20261002_v1'
INPUT=R/'common_gene_axis/results'
COUNT=B/'common_public_growth_statistics_20261002_v1/Source_BOOTSTRAP_GENE_COUNTS.npy'
COUNT_SHA='524ac5396accdf0892196cddcd2694a1cbf159eb0b8aff57f9c99a41ebd3908e'
GENE_SHA='f6a9d3cb6e0c2e6ba3630fb3df066392324ef8f34ebc3999001154079ebdc6c3'
LINES=['Exphormer_to_GAT','GAT_to_Exphormer']
METHODS=['Learned_hgb','Manual_hgb','Prediction_hgb','Magnitude']


def bind(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def write_json(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    path.chmod(0o444)


def source_only_csv(path):
    """Filter raw cache records by metadata before decoding any numeric column.

    This fixed cache has no quoted/comma-containing fields; source-only selected
    records are subsequently parsed by the CSV reader. Other lines are byte
    scanned solely for their line/seed/method metadata, never numerically parsed.
    """
    stream=gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')
    with stream as f:
        header=f.readline();names=header.rstrip().decode().split(',')
        li,si,mi=[names.index(k) for k in ['line','seed','method']]
        kept=[]
        for raw in f:
            fields=raw.rstrip(b'\r\n').split(b',')
            if fields[li] not in [x.encode() for x in LINES]:continue
            if fields[si]!=b'20260930' or fields[mi] not in [x.encode() for x in METHODS]:continue
            if len(fields)!=len(names) or b'"' in raw:raise RuntimeError('Unexpected fixed cache CSV encoding')
            kept.append(raw)
    return pd.read_csv(io.BytesIO(header+b''.join(kept)),float_precision='round_trip')


def main():
    began,cpu=time.monotonic(),time.process_time()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600second limit')));signal.alarm(600)
    if D.exists() or O.exists():raise RuntimeError('Fresh isolated completion roots required')
    paths=[Path(__file__),ROOT/'tools/safeconf_continual/research.py',
           ROOT/'tools/scripts/run_safeconf_source_scaling_uncertainty_agent.py',
           INPUT/'MATRIX_TASK_PREDICTIONS.csv.gz',INPUT/'MATRIX_CONTEXT_RESULTS.csv',INPUT/'MATRIX_MACRO_RESULTS.csv',
           INPUT/'PAIRED_CLUSTER_BOOTSTRAP.csv',COUNT,
           ROOT/'tools/scripts/complete_safeconf_common_public_growth_statistics.py']
    pins=[bind(p) for p in paths]
    if bind(COUNT)['sha256']!=COUNT_SHA:raise RuntimeError('Exact existing Source draw counts differ')
    D.mkdir();O.mkdir()
    write_json(D/'REGISTRATION.json',{'schema':'fixed_Source_only_Magnitude_strong_comparator_completion_v1',
        'pid':os.getpid(),'lines':LINES,'methods':METHODS,'fixed_pairs':[[a,'Magnitude'] for a in METHODS[:-1]],
        'risk_seed':20260930,'tasks_per_line':1808,'genes':575,'contexts':4,'metrics':list(METRICS),
        'bootstrap':'reuse existing5000x575uint16gene counts; bothlines share counts; no RNG',
        'existing_count_generation_seed':20261002,'not_original_main_bootstrap_seed':True,
        'sorted_gene_order_sha256':GENE_SHA,'original_main_CI_reproduction_claim':False,
        'new_fits':0,'new_upstream_calls':0,'new_CDF_or_label_fits':0,'new_sampling':0,
        'Orion_access':False,'MC_numeric_access':False,'raw_gene_vector_access':False,
        'CSV_access':'sharedcache byte scans/hashes; only allowedSource seed/method numeric records parsed',
        'resources':{'CPU_threads':4,'wall_cap_seconds':600,'RSS_cap_bytes':2147483648,'GPU_hours':0},
        'input_code_bindings':pins})
    print(json.dumps({'pid':os.getpid(),'phase':'registered_Source_only_fixed6pairs','fits':0}),flush=True)
    data=source_only_csv(INPUT/'MATRIX_TASK_PREDICTIONS.csv.gz')
    original_context=source_only_csv(INPUT/'MATRIX_CONTEXT_RESULTS.csv')
    original_macro=source_only_csv(INPUT/'MATRIX_MACRO_RESULTS.csv')
    counts=np.load(COUNT,mmap_mode='r',allow_pickle=False)
    if counts.shape!=(5000,575) or counts.dtype!=np.uint16 or not (counts.sum(axis=1)==575).all():raise RuntimeError('Draw-count metadata differs')
    all_contexts=['K562','RPE1','hepg2','jurkat'];points=[];pairs=[];checks=[];invalid=[]
    full_draws=np.full((2,5000,5,4,len(METRICS)),np.nan);full_points=np.full((2,5,4,len(METRICS)),np.nan)
    for line_index,line in enumerate(LINES):
        part=data[data.line.eq(line)]
        meta=['task_id','target','gene','fold','upstream','true_error_rmse']
        wide=part.pivot(index=meta,columns='method',values='risk').reset_index()
        genes=sorted(wide.gene.astype(str).unique())
        if (len(wide)!=1808 or wide.gene.nunique()!=575 or not wide.task_id.is_unique
            or hashlib.sha256('\n'.join(genes).encode()).hexdigest()!=GENE_SHA
            or set(wide.target)!=set(all_contexts) or not np.isfinite(wide[METHODS+['true_error_rmse']].to_numpy()).all()):
            raise RuntimeError('Full Source population or scores changed')
        gene_index=wide.gene.map({g:i for i,g in enumerate(genes)}).to_numpy(int)
        for ci,context in enumerate(all_contexts):
            ix=np.flatnonzero(wide.target.eq(context));frame=wide.iloc[ix]
            if len(frame)<20:raise RuntimeError('Original Source context invalid')
            weight=counts[:,gene_index[ix]].astype(np.int32)
            for mi,method in enumerate(METHODS):
                risk=frame[method].to_numpy(float);scalar=metrics(frame,risk)
                point=np.asarray([scalar[k] for k in METRICS])
                compact=counter_metrics(frame.true_error_rmse.to_numpy(),frame.task_id.to_numpy(str),risk,np.ones((1,len(frame)),int))[0]
                if not np.allclose(point,compact,rtol=0,atol=1e-12,equal_nan=True):raise RuntimeError('Counter point validator differs')
                old=original_context[original_context.line.eq(line)&original_context.target.eq(context)&original_context.method.eq(method)]
                if len(old)!=1 or not np.allclose(point,old[list(METRICS)].iloc[0],rtol=0,atol=1e-12,equal_nan=True):raise RuntimeError('Original scalar context points differ')
                full_points[line_index,ci,mi]=point
                full_draws[line_index,:,ci,mi]=counter_metrics(frame.true_error_rmse.to_numpy(),frame.task_id.to_numpy(str),risk,weight)
                checks.append({'line':line,'context':context,'method':method,'n_tasks':len(frame),
                    'max_abs_main_point_difference':float(np.nanmax(np.abs(point-old[list(METRICS)].iloc[0].to_numpy()))),
                    'max_abs_counter_point_difference':float(np.nanmax(np.abs(point-compact)))})
        full_points[line_index,4]=np.mean(full_points[line_index,:4],axis=0)
        full_draws[line_index,:,4]=np.mean(full_draws[line_index,:,:4],axis=1)
        for mi,method in enumerate(METHODS):
            old=original_macro[original_macro.line.eq(line)&original_macro.method.eq(method)]
            if len(old)!=1 or not np.allclose(full_points[line_index,4,mi],old[list(METRICS)].iloc[0],rtol=0,atol=1e-12,equal_nan=True):raise RuntimeError('Original macro differs')
        for ci,context in enumerate(all_contexts+['macro']):
            for mi,method in enumerate(METHODS):
                for ki,key in enumerate(METRICS):
                    d=full_draws[line_index,:,ci,mi,ki];finite=d[np.isfinite(d)]
                    lo,hi=np.quantile(finite,[.025,.975]) if len(finite)>=2 else [np.nan,np.nan]
                    points.append({'line':line,'context':context,'method':method,'metric':key,'point':full_points[line_index,ci,mi,ki],
                        'ci95_lower':lo,'ci95_upper':hi,'valid_draws':len(finite),'saved_draws':5000})
            for mi,method in enumerate(METHODS[:-1]):
                for ki,key in enumerate(METRICS):
                    delta=full_draws[line_index,:,ci,mi,ki]-full_draws[line_index,:,ci,3,ki]
                    finite=delta[np.isfinite(delta)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite)>=2 else [np.nan,np.nan]
                    pairs.append({'line':line,'context':context,'method_a':method,'method_b':'Magnitude','metric':key,
                        'point_a':full_points[line_index,ci,mi,ki],'point_b':full_points[line_index,ci,3,ki],
                        'difference_a_minus_b':full_points[line_index,ci,mi,ki]-full_points[line_index,ci,3,ki],
                        'ci95_lower':lo,'ci95_upper':hi,'valid_draws':len(finite),'saved_draws':5000,'new_RNG':0})
                    if len(finite)!=5000:invalid.append({'line':line,'context':context,'method':method,'metric':key,'valid_draws':len(finite),'total':5000})
        print(json.dumps({'phase':'Source_line_complete','line':line}),flush=True)
    if len(points)!=280 or len(pairs)!=210:raise RuntimeError('All fixed6comparators/contexts/metrics required')
    np.savez_compressed(O/'ALL_FIXED_SOURCE_METRIC_DRAWS.npz',draws=full_draws,points=full_points,
        lines=np.asarray(LINES),contexts=np.asarray(all_contexts+['macro']),methods=np.asarray(METHODS),metrics=np.asarray(METRICS))
    for name,rows in [('ALL_METHOD_INTERVALS.csv',points),('ALL6_FIXED_PAIRED_COMPARATORS.csv',pairs),
                      ('SCALAR_MAIN_AND_COUNTER_POINT_REPRODUCTION.csv',checks),('INVALID_DRAW_COUNTS.csv',invalid)]:
        pd.DataFrame(rows).to_csv(D/name,index=False);(D/name).chmod(0o444)
    for item in pins:
        if bind(item['path'])!=item:raise RuntimeError('Original matrix/cache/code changed')
    write_json(D/'RESULT_MANIFEST.json',{'status':'COMPLETE_FIXED_SOURCE_MAGNITUDE_COMPARATOR_COMPLETION',
        'fixed_pairs':6,'method_interval_rows':280,'paired_rows':210,'source_lines':2,'tasks_per_line':1808,
        'shared_gene_clusters':575,'contexts_per_line':4,'all_scalar_main_points_reproduced':True,
        'all_counter_points_validated':True,'new_fits':0,'new_CDF_fits':0,'new_RNG_draws':0,
        'saved_count_draws':5000,'count_generation_seed':20261002,'Orion_access':False,'MC_numeric_access':False,
        'original_assets_unchanged':True,'elapsed_seconds':time.monotonic()-began,'CPU_seconds':time.process_time()-cpu,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_hours':0,
        'output_bindings':[bind(p) for p in sorted(D.glob('*.csv'))]+[bind(O/'ALL_FIXED_SOURCE_METRIC_DRAWS.npz')]})
    print(json.dumps({'status':'COMPLETE_FIXED_SOURCE_MAGNITUDE_COMPARATOR_COMPLETION'}),flush=True)


if __name__=='__main__':main()
