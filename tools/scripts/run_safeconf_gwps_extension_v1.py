#!/usr/bin/env python3
"""Registered SEEN GWPS extension; immutable old bank, errors and upstream.

The whole library UMI total is used before projection. Existing K562 targets
are conservatively treated as overlapping with GWPS and never double counted.
Only Source predictions/labels train the fixed reader; Orion errors are opened
from the previously authorized cached table after all score files are sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC, fit_risk, rank_labels, shuffled_labels
from tools.scripts.seal_safeconf_orion_source_risk_agent import prediction_features

BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
SOURCE = BASE / 'orion_source_core_20261002_v1'
BANK = BASE / 'public_source_history_expanded_20261002_v2'
FIT = BASE / 'orion_published_lm_20261002_v3_decimal_tokens'
FORMAL = BASE / 'orion_registered_test_extendedbank_20261002_v1'
GWPS = Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ReplogleWeissman2022_K562_gwps.h5ad')
RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/gwps_v1')
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/gwps'
QUERY = ['query_id', 'target_gene_id', 'target_gene_symbol', 'context_id', 'role']
METRICS = ['utility20', 'spearman', 'aurc', 'high_risk_miss_rate', 'error_at_10', 'error_at_20', 'error_at_50']
NULL_SEEDS = [20260930, 20260931, 20260932, 20260933, 20260934]
BOOT_SEED = 20260930


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_): return bool(x)
    return x


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def binding(path):
    p = Path(path); return {'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def event(phase, **values):
    payload = {'phase': phase, 'pid': os.getpid(), 'UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}
    print(json.dumps(clean(payload)), flush=True)
    write_json(RUNTIME / 'LIVE_STATUS.json', payload)


def h5col(group, name):
    x = group[name]
    if isinstance(x, h5py.Group):
        cats = x['categories'].asstr()[:] if x['categories'].dtype.kind in 'OS' else x['categories'][:]
        codes = x['codes'][:]
        if (codes < 0).any(): raise ValueError('Unexpected missing categorical data: ' + name)
        return cats[codes]
    return x.asstr()[:] if x.dtype.kind in 'OS' else x[:]


def grouped_add(output, codes, values):
    if not len(codes): return
    order = np.argsort(codes, kind='stable')
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    output[sorted_codes[starts]] += np.add.reduceat(values[order], starts, axis=0)


def normalized(raw, total, columns):
    x = np.asarray(raw[:, columns], dtype=np.float64)
    x *= (4000. / total)[:, None]
    np.log1p(x, out=x)
    return x


def cosine_rows(a, b):
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def qualify():
    if (RUNTIME / 'QUALIFICATION.json').exists():
        return json.loads((RUNTIME / 'QUALIFICATION.json').read_text())
    old = pd.read_parquet(BANK / 'SOURCE_PUBLIC_MEMORY_METADATA.parquet')
    provenance = pd.read_csv(BANK / 'CANONICAL_UNIT_PROVENANCE.csv')
    axis = pd.read_csv(SOURCE / 'GENE_MANIFEST.csv')
    cohort = pd.read_csv(FORMAL / 'fixed_risk_evaluation/all_prediction_only_COHORT.csv')
    if len(cohort) != 2993 or cohort.target_gene_id.nunique() != 1750:
        raise RuntimeError('Original all-eligible metadata cohort differs')
    with h5py.File(GWPS, 'r') as f:
        gene_axis = h5col(f['var'], 'ensembl_id')
        if len(set(gene_axis)) != len(gene_axis): raise ValueError('Duplicate Ensembl IDs')
        lookup = {g: i for i, g in enumerate(gene_axis)}
        common = axis[axis.orion_ensembl_id.isin(lookup)].copy()
        common['gwps_index'] = common.orion_ensembl_id.map(lookup).astype(int)
        common['feature_axis_index'] = np.arange(len(common))
        if len(common) != 3216: raise RuntimeError('Common axis changed from metadata registration')
        genes = h5col(f['obs'], 'perturbation'); batch = h5col(f['obs'], 'batch')
        ncounts = h5col(f['obs'], 'ncounts'); totals = h5col(f['obs'], 'UMI_count')
        controls = genes == 'control'
        batches, batch_counts = np.unique(batch[controls], return_counts=True)
        if set(batch) != set(batches) or batch_counts.min() < 10:
            raise RuntimeError('Every batch requires >=10 observed controls')
        counts = pd.Series(genes).value_counts()
        eligible = counts[(counts >= 30) & counts.index.to_series().ne('control')]
        old_k562 = set(old.loc[old.context.eq('K562'), 'perturbation_target'])
        added = set(eligible.index) - old_k562
        oldgenes = set(old.perturbation_target)
        sample_rows = sorted({0, len(genes)//4, len(genes)//2, len(genes)-16})
        audits = []
        for start in sample_rows:
            raw = f['X'][start:start+16]
            total = totals[start:start+16]
            measured = raw.sum(axis=1, dtype=np.float64)
            if not np.isfinite(raw).all() or (raw < 0).any() or not np.array_equal(raw, np.rint(raw)):
                raise RuntimeError('GWPS X must be finite nonnegative integer counts')
            if not np.array_equal(measured, ncounts[start:start+16]) or (total < measured).any():
                raise RuntimeError('Stored count-total audit failed')
            audits.append({'start': start, 'n_rows': len(raw), 'full_total_min': total.min(),
                           'retained_fraction_min': (measured/total).min(), 'retained_fraction_max': (measured/total).max()})
    common.to_csv(RUNTIME / 'COMMON_GENE_AXIS.csv', index=False)
    cohort[QUERY + ['n_cells', 'source_history_n']].to_csv(RUNTIME / 'FIXED_2993_COHORT_METADATA.csv', index=False)
    q = {'schema': 'gwps_public_extension_v1', 'status': 'QUALIFIED_FOR_STREAMING_AGGREGATION',
         'gwps_asset': binding(GWPS), 'original_bank_manifest': binding(BANK/'PUBLIC_BANK_MANIFEST.json'),
         'original_truth_receipt': binding(FORMAL/'registered_test_truth/TRUTH_READER_RECEIPT.json'),
         'original_error_cache': {'path': str(FORMAL/'registered_test_truth/TEST_TASK_ERRORS.parquet'),
                                  'sha256': sha(FORMAL/'registered_test_truth/TEST_TASK_ERRORS.parquet'), 'values_read': False},
         'code': binding(__file__), 'cells': len(genes), 'raw_axis_genes': len(gene_axis), 'feature_genes': len(common),
         'error_axis_genes_unchanged': 3285, 'controls': int(controls.sum()), 'batches': len(batches),
         'minimum_batch_control_cells': batch_counts.min(), 'eligible_perturbations_ge30': len(eligible),
         'old_K562_targets_conservatively_deduplicated': len(old_k562 & set(eligible.index)),
         'eligible_new_K562_targets': len(added), 'old_bank_items': len(old),
         'old_covered_tasks': int(cohort.target_gene_symbol.isin(oldgenes).sum()),
         'potential_expanded_tasks': int(cohort.target_gene_symbol.isin(oldgenes | added).sum()),
         'normalization': 'log1p(4000 * raw retained gene UMI / original UMI_count); no retained-axis renormalization',
         'effect': 'cell-equal treated mean minus NTC mean matched to treated batch frequencies',
         'dedup': 'same Replogle2022/K562/target treated as overlapping; keep original record, append only previously absent target',
         'new_independent_studies': 0, 'sample_count_checks': audits,
         'source_training_policy': 'project original legal Source references to3216; do not introduce GWPS into Source feature training',
         'source_error_target': 'preserve original3285 errors and original per-source/context midrank CDF',
         'test_policy': 'SEEN cached2993 errors only after score sealing; no raw Orion truth reads',
         'weight': 'support weighted cell counts; quality audit separate, no new quality weight',
         'source_reader': {'hgb': {'max_iter':200,'learning_rate':.05,'max_depth':3,'min_samples_leaf':20,'l2_regularization':10},
                           'real_label_seed':20260930, 'null_seeds':NULL_SEEDS},
         'growth': {'orders':list(range(5)), 'fractions':[.1,.25,.5,.75,1.],
                    'hash_salt':'gwps-public-growth-v1', 'unit':'new K562 perturbation target'},
         'bootstrap': {'replicates':5000,'seed':BOOT_SEED,'unit':'biological gene; contexts paired'}}
    write_json(RUNTIME/'QUALIFICATION.json', q); write_json(DOC/'QUALIFICATION.json', q)
    common.to_csv(DOC/'COMMON_GENE_AXIS.csv', index=False)
    pd.DataFrame(audits).to_csv(DOC/'SMALL_SAMPLE_COUNT_AUDIT.csv', index=False)
    event('qualified', eligible_genes=len(eligible), new_genes=len(added), potential_tasks=q['potential_expanded_tasks'])
    return q


def build_bank():
    if (RUNTIME/'BANK_COMPLETE.json').exists(): return
    q = qualify(); started = time.monotonic()
    common = pd.read_csv(RUNTIME/'COMMON_GENE_AXIS.csv'); cols=common.gwps_index.to_numpy(int)
    with h5py.File(GWPS,'r') as f:
        cats = f['obs/perturbation/categories'].asstr()[:]
        gene_codes = f['obs/perturbation/codes'][:].astype(np.int32)
        gcats = f['obs/guide_id/categories'].asstr()[:]
        guide_codes=f['obs/guide_id/codes'][:].astype(np.int32)
        batch_names,batch_codes=np.unique(h5col(f['obs'],'batch'),return_inverse=True)
        totals=h5col(f['obs'],'UMI_count').astype(float); ncounts=h5col(f['obs'],'ncounts').astype(float)
        control=int(np.flatnonzero(cats=='control')[0]); is_ctrl=gene_codes==control
        n_gene,n_guide,n_batch,n_axis=len(cats),len(gcats),len(batch_names),len(cols)
        gene_count=np.bincount(gene_codes,minlength=n_gene)
        guide_count=np.bincount(guide_codes,minlength=n_guide)
        gene_batch=np.bincount(gene_codes*n_batch+batch_codes,minlength=n_gene*n_batch).reshape(n_gene,n_batch)
        guide_batch=np.bincount(guide_codes*n_batch+batch_codes,minlength=n_guide*n_batch).reshape(n_guide,n_batch)
        gene_sum=np.zeros((n_gene,n_axis),float);guide_sum=np.zeros((n_guide,n_axis),float)
        ctrl_sum=np.zeros((n_batch,n_axis),float)
        guide_gene=np.full(n_guide,-1,np.int32)
        for g in range(n_guide):
            values=np.unique(gene_codes[guide_codes==g])
            if len(values)!=1:raise ValueError('Guide construct mapped to multiple perturbations')
            guide_gene[g]=values[0]
        # A predefined batch-half agreement is technical stability, not independent-study replication.
        batch_half=np.array([int(hashlib.sha256(f'gwps-batch-half-v1|{b}'.encode()).hexdigest(),16)%2 for b in batch_names])
        half_codes=2*gene_codes+batch_half[batch_codes]
        half_count=np.bincount(half_codes,minlength=2*n_gene)
        half_sum=np.zeros((2*n_gene,n_axis),float)
        block=3886*2;max_library_gap=0.;min_retained=1.;whole_rows=0
        for start in range(0,len(gene_codes),block):
            stop=min(start+block,len(gene_codes));raw=f['X'][start:stop]
            if not np.isfinite(raw).all() or (raw<0).any() or not np.array_equal(raw,np.rint(raw)):
                raise ValueError(f'Invalid raw count block {start}')
            measured=raw.sum(1,dtype=float);total=totals[start:stop]
            if not np.array_equal(measured,ncounts[start:stop]) or not np.isfinite(total).all() or (total<=0).any() or (measured>total).any():
                raise ValueError(f'Library count inconsistency {start}')
            max_library_gap=max(max_library_gap,float(np.max(total-measured)));min_retained=min(min_retained,float(np.min(measured/total)))
            x=normalized(raw,total,cols);del raw
            grouped_add(gene_sum,gene_codes[start:stop],x)
            grouped_add(guide_sum,guide_codes[start:stop],x)
            grouped_add(half_sum,half_codes[start:stop],x)
            ctrl=is_ctrl[start:stop]
            grouped_add(ctrl_sum,batch_codes[start:stop][ctrl],x[ctrl]);del x
            whole_rows=stop
            if start==0 or (start//block)%10==0 or stop==len(gene_codes):
                event('streaming_public_counts',rows=stop,total_rows=len(gene_codes),seconds=time.monotonic()-started,
                      peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    ctrl_count=gene_batch[control]
    ctrl_mean=ctrl_sum/ctrl_count[:,None]
    matched=gene_batch@ctrl_mean/gene_count[:,None]
    means=gene_sum/gene_count[:,None]; effects=means-matched
    guide_effect=guide_sum/guide_count[:,None]-(guide_batch@ctrl_mean)/guide_count[:,None]
    half_matched=np.zeros_like(half_sum)
    for h in (0,1):
        mask=batch_half==h;half_matched[h::2]=gene_batch[:,mask]@ctrl_mean[mask]
    half_effect=np.divide(half_sum-half_matched,half_count[:,None],out=np.full_like(half_sum,np.nan),where=half_count[:,None]>0)
    quality=[]
    for i,gene in enumerate(cats):
        selected=np.flatnonzero((guide_gene==i)&(guide_count>=10))
        cos=[];rm=[]
        for a in range(len(selected)):
            for b in range(a+1,len(selected)):
                u,v=guide_effect[selected[a]],guide_effect[selected[b]]
                cos.append(float(cosine_rows(u[None,:],v[None,:])[0]));rm.append(float(np.sqrt(np.mean((u-v)**2))))
        h0,h1=2*i,2*i+1;half_ok=half_count[h0]>=10 and half_count[h1]>=10
        quality.append({'target_gene':gene,'n_cells':int(gene_count[i]),'n_batches':int((gene_batch[i]>0).sum()),
                        'n_guide_constructs':int((guide_gene==i).sum()),'n_guide_constructs_ge10':len(selected),
                        'guide_construct_cosine':np.mean(cos) if cos else np.nan,
                        'guide_construct_RMSE':np.mean(rm) if rm else np.nan,
                        'batch_half_cells_0':int(half_count[h0]),'batch_half_cells_1':int(half_count[h1]),
                        'batch_half_cosine':float(cosine_rows(half_effect[h0:h0+1],half_effect[h1:h1+1])[0]) if half_ok else np.nan,
                        'batch_half_RMSE':float(np.sqrt(np.mean((half_effect[h0]-half_effect[h1])**2))) if half_ok else np.nan})
    quality=pd.DataFrame(quality)
    old=pd.read_parquet(BANK/'SOURCE_PUBLIC_MEMORY_METADATA.parquet')
    old_k562=set(old.loc[old.context.eq('K562'),'perturbation_target'])
    keep=(gene_count>=30)&(cats!='control')
    row_ids=np.flatnonzero(keep); new_rows=np.flatnonzero(keep&~np.isin(cats,list(old_k562)))
    metadata=[]
    for outrow,i in enumerate(row_ids):
        metadata.append({'experiment_id':f'GWPS::Replogle2022::K562::{cats[i]}','original_study':'Replogle_2022',
                         'context':'K562','perturbation_target':cats[i],'effect_vector_row':outrow,
                         'n_cells':int(gene_count[i]),'n_batches':int((gene_batch[i]>0).sum()),
                         'eligible_for_extension':cats[i] not in old_k562,
                         'original_gene_code':int(i),'matched_NTC':True})
    meta=pd.DataFrame(metadata);meta.to_parquet(RUNTIME/'GWPS_PUBLIC_METADATA.parquet',index=False)
    np.save(RUNTIME/'GWPS_PUBLIC_EFFECTS.npy',effects[row_ids]);np.save(RUNTIME/'GWPS_PUBLIC_CONTROLS.npy',matched[row_ids])
    quality.to_csv(DOC/'GWPS_QUALITY_SUPPORT_AUDIT.csv',index=False)
    quality.to_parquet(RUNTIME/'GWPS_QUALITY_SUPPORT_AUDIT.parquet',index=False)
    np.save(RUNTIME/'GWPS_BATCH_NTC_MEANS.npy',ctrl_mean)
    pd.DataFrame({'batch':batch_names,'NTC_cells':ctrl_count,'registered_half':batch_half}).to_csv(DOC/'BATCH_CONTROL_COUNTS.csv',index=False)
    # Matched-study overlap is only a QC diagnostic; these duplicate candidates are never appended.
    lookup={g:i for i,g in enumerate(cats)};old_effect=np.load(BANK/'SOURCE_PUBLIC_EFFECTS.npy',mmap_mode='r')
    axis_cols=common.axis_index.to_numpy(int);overlap=[]
    for item in old[old.context.eq('K562')].itertuples(index=False):
        if item.perturbation_target not in lookup:continue
        i=lookup[item.perturbation_target];u=np.asarray(old_effect[int(item.effect_vector_row),axis_cols]);v=effects[i]
        overlap.append({'target_gene':item.perturbation_target,'old_cells':item.n_cells,'GWPS_cells':int(gene_count[i]),
                        'effect_RMSE':float(np.sqrt(np.mean((u-v)**2))), 'effect_cosine':float(cosine_rows(u[None,:],v[None,:])[0]),
                        'old_magnitude':float(np.sqrt(np.mean(u**2))),'GWPS_magnitude':float(np.sqrt(np.mean(v**2))),
                        'action':'OLD_RECORD_RETAINED_NEW_OVERLAP_NOT_APPENDED'})
    pd.DataFrame(overlap).to_csv(DOC/'OVERLAP_EFFECT_ALIGNMENT.csv',index=False)
    complete={'status':'BANK_COMPLETE','normalized_rows':whole_rows,'n_bank_records':len(meta),'new_unique_K562_targets':len(new_rows),
              'min_retained_library_fraction':min_retained,'max_missing_gene_library_UMI':max_library_gap,
              'all_count_and_control_checks_passed':True,'duplicate_K562_targets_not_double_counted':len(overlap),
              'quality_guide_constructs_not_split_into_individual_sgRNA_tokens':True,
              'batch_half_stability_is_technical_not_independent_study_reproducibility':True,
              'biological_effect_truth_or_Orion_errors_used_to_select_history':False,
              'elapsed_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    write_json(RUNTIME/'BANK_COMPLETE.json',complete);write_json(DOC/'BANK_COMPLETE.json',complete)
    event('bank_complete',**complete)


def feature_frame(pred, history_indices, effects, n_cells, base=None):
    frame=prediction_features(pred) if base is None else base.copy().reset_index(drop=True)
    frame[P]=prediction_features(pred)
    for c in PUBLIC:frame[c]=np.nan
    n=np.zeros(len(pred),int);priors=np.full_like(pred,np.nan,dtype=float)
    for row,indices in enumerate(history_indices):
        indices=np.asarray(indices,int);n[row]=len(indices)
        if not len(indices):continue
        v=np.asarray(effects[indices],float);cells=np.asarray(n_cells)[indices].astype(float);w=cells/cells.sum()
        prior=w@v;priors[row]=prior
        var=float(w@np.mean((v-prior)**2,axis=1));conflict=float(np.sqrt(np.mean((v-v.mean(0))**2,axis=1)).mean())
        frame.loc[row,PUBLIC]=[float(np.sqrt(np.mean(prior**2))),float(np.sqrt(np.mean((pred[row]-prior)**2))),
                              float(cosine_rows(pred[row:row+1],prior[None,:])[0]),math.sqrt(max(var,0)),
                              float(np.log1p(cells.sum())),float(1/np.sum(w*w)),conflict]
        identity=float(w@np.mean((v-pred[row])**2,axis=1))
        if not np.isclose(identity,float(frame.loc[row,'prediction_prior_rmse'])**2+var,rtol=1e-11,atol=1e-14):
            raise AssertionError('Weighted distance identity failed')
    return frame,priors,n


def ecdf_fit_apply(train,query):
    train=np.sort(np.asarray(train,float));train=train[np.isfinite(train)]
    query=np.asarray(query,float)
    if len(train)<2:raise RuntimeError('Score CDF needs >=2 training features')
    return (np.searchsorted(train,query,'left')+np.searchsorted(train,query,'right'))/(2*len(train))


def scores():
    if (RUNTIME/'SCORE_SEAL.json').exists():return
    build_bank();started=time.monotonic()
    cols=pd.read_csv(RUNTIME/'COMMON_GENE_AXIS.csv').axis_index.to_numpy(int)
    tasks=pd.read_parquet(SOURCE/'SOURCE_TASKS.parquet')
    pairs=pd.read_parquet(SOURCE/'SOURCE_PUBLIC_PAIR_FEATURES.parquet')
    source_memory=pd.read_parquet(SOURCE/'SOURCE_PUBLIC_MEMORY_METADATA.parquet')
    source_effect=np.asarray(np.load(SOURCE/'SOURCE_PUBLIC_EFFECTS.npy',mmap_mode='r')[:,cols])
    histories=[[] for _ in range(len(tasks))]
    memorycol='memory_row' if 'memory_row' in pairs else 'source_row'
    for i,g in pairs.groupby('task_row'):histories[int(i)]=g[memorycol].astype(int).tolist()
    train=[]
    for upstream in ('TxPert_GAT','TxPert_Exphormer'):
        base=pd.read_parquet(SOURCE/f'SOURCE_{upstream}_P_BASE.parquet')
        pred=np.asarray(np.load(SOURCE/f'SOURCE_{upstream}_PREDICTED_EFFECTS.npy',mmap_mode='r')[:,cols])
        frame,_,n=feature_frame(pred,histories,source_effect,source_memory.n_cells.to_numpy(),base)
        if not (n>0).all():raise RuntimeError('Unexpected no history Source row')
        train.append(frame)
    train=pd.concat(train,ignore_index=True)
    labels,cdf_audit=rank_labels(train,'GWPS3216_features_unchanged3285_error_target')
    original=np.load(SOURCE/'SOURCE_RISK_MIDRANK_LABELS.npy')
    if not np.array_equal(labels,original):raise RuntimeError('Original Source error rank labels changed')
    pd.DataFrame(cdf_audit).to_csv(DOC/'SOURCE_CDF_AUDIT.csv',index=False)
    train.to_parquet(RUNTIME/'SOURCE3216_TRAIN_FEATURES.parquet',index=False)
    model=fit_risk(train,labels,P+PUBLIC,'hgb',20260930)
    joblib.dump(model,RUNTIME/'SOURCE3216_REAL_HGB.joblib',compress=3)
    nulls=[];null_audit=[]
    for seed in NULL_SEEDS:
        shuffled,audit=shuffled_labels(train,labels,seed)
        m=fit_risk(train,shuffled,P+PUBLIC,'hgb',20260930)
        joblib.dump(m,RUNTIME/f'SOURCE3216_NULL_HGB_{seed}.joblib',compress=3)
        nulls.append(m);null_audit.append(audit)
    pd.DataFrame(null_audit).to_csv(DOC/'SOURCE_LABEL_NULL_AUDIT.csv',index=False)
    queries=pd.read_csv(RUNTIME/'FIXED_2993_COHORT_METADATA.csv',keep_default_na=False)
    pred=np.empty((len(queries),len(cols)),float)
    for context,part in queries.groupby('context_id',sort=True):
        path=FIT/context/'PREDICTIONS_DELTA.tsv.gz'
        a=pd.read_csv(path,sep='\t',usecols=['gene_id']+part.query_id.tolist(),keep_default_na=False)
        axis=pd.read_csv(SOURCE/'GENE_MANIFEST.csv')
        if a.gene_id.tolist()!=axis.orion_ensembl_id.tolist():raise RuntimeError('Prediction axis mismatch')
        pred[part.index]=a[part.query_id.tolist()].to_numpy(float).T[:,cols]
    np.save(RUNTIME/'FIXED2993_PREDICTED_EFFECTS_3216.npy',pred)
    old=pd.read_parquet(BANK/'SOURCE_PUBLIC_MEMORY_METADATA.parquet')
    oe=np.asarray(np.load(BANK/'SOURCE_PUBLIC_EFFECTS.npy',mmap_mode='r')[:,cols])
    new=pd.read_parquet(RUNTIME/'GWPS_PUBLIC_METADATA.parquet')
    ne=np.load(RUNTIME/'GWPS_PUBLIC_EFFECTS.npy',mmap_mode='r')
    new=new[new.eligible_for_extension].copy();nr=new.effect_vector_row.to_numpy(int)
    merged=pd.concat([old[['perturbation_target','n_cells','context']],new[['perturbation_target','n_cells','context']]],ignore_index=True)
    effects=np.concatenate([oe,np.asarray(ne[nr])],axis=0)
    merged['bank_row']=np.arange(len(merged));merged['is_added_GWPS']=merged.index>=len(old)
    merged.to_parquet(RUNTIME/'EXTENDED_PUBLIC_INDEX.parquet',index=False)
    np.save(RUNTIME/'EXTENDED_PUBLIC_EFFECTS.npy',effects)
    by_gene={gene:g.index.to_numpy() for gene,g in merged.groupby('perturbation_target')}
    newhist=[by_gene.get(g,[]) for g in queries.target_gene_symbol]
    oldhist=[np.asarray(x)[np.asarray(x)<len(old)] for x in newhist]
    out=queries.copy();out['Magnitude']=np.sqrt(np.mean(pred**2,axis=1))
    train_distance=np.sqrt(train.prediction_prior_rmse**2+train.prior_uncertainty**2)
    magcdf=ecdf_fit_apply(train.predicted_magnitude,out.Magnitude)
    out['Magnitude_SourceScoreCDF']=magcdf
    for name,hist in [('Old',oldhist),('Expanded',newhist)]:
        frame,prior,n=feature_frame(pred,hist,effects,merged.n_cells.to_numpy())
        frame.to_parquet(RUNTIME/f'{name}_FIXED2993_FEATURES.parquet',index=False)
        np.save(RUNTIME/f'{name}_FIXED2993_PRIORS.npy',prior)
        out[name+'_n_history']=n
        out[name+'_DirectDistance']=frame.prediction_prior_rmse
        out[name+'_WeightedDistance']=np.sqrt(frame.prediction_prior_rmse**2+frame.prior_uncertainty**2)
        out[name+'_NegativeSupport']=-frame.log_history_support
        valid=n>0
        for key,m in [('HGB',model)]+[(f'NullHGB{j}',m) for j,m in enumerate(nulls)]:
            out[name+'_'+key]=np.nan
            out.loc[valid,name+'_'+key]=m.predict(frame.loc[valid])
        out[name+'_DistanceCDF_or_Magnitude']=magcdf
        out.loc[valid,name+'_DistanceCDF_or_Magnitude']=ecdf_fit_apply(train_distance,out.loc[valid,name+'_WeightedDistance'])
        out[name+'_HGB_or_Magnitude']=magcdf
        out.loc[valid,name+'_HGB_or_Magnitude']=out.loc[valid,name+'_HGB']
    if int((out.Old_n_history>0).sum())!=232:raise RuntimeError('Original232 coverage changed')
    out.to_parquet(RUNTIME/'FROZEN_SCORES_BEFORE_ERROR_OPEN.parquet',index=False)
    coverage=[]
    masks={'old_covered':out.Old_n_history>0,'newly_covered':(out.Old_n_history==0)&(out.Expanded_n_history>0),
           'expanded_covered':out.Expanded_n_history>0,'all_eligible':np.ones(len(out),bool)}
    for group,mask in masks.items():
        for context in ['HCT116','HEK293T','all']:
            use=mask if context=='all' else mask&out.context_id.eq(context)
            coverage.append({'cohort':group,'context':context,'n_tasks':int(np.sum(use)),
                             'n_genes':out.loc[use,'target_gene_id'].nunique(),'fraction_of2993':float(np.sum(use)/len(out))})
    pd.DataFrame(coverage).to_csv(DOC/'ACTUAL_COVERAGE.csv',index=False)
    seal={'status':'SCORES_FROZEN_BEFORE_CACHED_ERRORS_READ','score_file':binding(RUNTIME/'FROZEN_SCORES_BEFORE_ERROR_OPEN.parquet'),
          'qualified_bank':binding(RUNTIME/'BANK_COMPLETE.json'),'feature_axis':binding(RUNTIME/'COMMON_GENE_AXIS.csv'),
          'source_training':binding(RUNTIME/'SOURCE3216_TRAIN_FEATURES.parquet'),'source_labels_original3285_exact':True,
          'n_source_fits':6,'no_GWPS_into_Source_training_references':True,'null_audits':null_audit,
          'n_eligible_tasks':len(out),'old_covered':int((out.Old_n_history>0).sum()),
          'expanded_covered':int((out.Expanded_n_history>0).sum()),
          'newly_covered':int(((out.Old_n_history==0)&(out.Expanded_n_history>0)).sum()),
          'score_CDF':'fixed Source prediction/reference features only; no Orion errors',
          'candidate_selection_on_Orion':False,'elapsed_seconds':time.monotonic()-started}
    write_json(RUNTIME/'SCORE_SEAL.json',seal);write_json(DOC/'SCORE_SEAL.json',seal)
    event('scores_sealed',**{k:seal[k] for k in ['old_covered','expanded_covered','newly_covered','elapsed_seconds']})


def metric_array(error,score,ids):
    n=len(error);result=np.full(len(METRICS),np.nan)
    if n<20 or not np.isfinite(error).all() or not np.isfinite(score).all():return result
    k=math.ceil(.2*n);high=np.lexsort((ids,-score))[:k];oracle=np.lexsort((ids,-error))[:k]
    denom=error[oracle].mean()-error.mean()
    if denom>1e-12:result[0]=(error[high].mean()-error.mean())/denom
    if np.ptp(score)>0 and np.ptp(error)>0:
        a=rankdata(error);b=rankdata(score);a-=a.mean();b-=b.mean();result[1]=a@b/np.sqrt((a@a)*(b@b))
    low=np.lexsort((ids,score));result[2]=np.mean(np.cumsum(error[low])/np.arange(1,n+1))
    result[3]=1-len(set(high)&set(oracle))/k
    for j,c in enumerate((.1,.2,.5),4):result[j]=error[low[:math.ceil(c*n)]].mean()
    return result


def stats(frame,methods,cohort):
    frame=frame.reset_index(drop=True);err=frame.true_error_rmse.to_numpy();ids=frame.query_id.to_numpy(str)
    context=frame.context_id.to_numpy(str);s=frame[methods].to_numpy(float)
    clusters=sorted(frame.target_gene_id.unique());blocks=[np.flatnonzero(frame.target_gene_id.to_numpy()==g) for g in clusters]
    def calc(ix):
        v=np.full((3,len(methods),len(METRICS)),np.nan)
        for ci,c in enumerate(('HCT116','HEK293T')):
            rows=ix[context[ix]==c]
            for mi in range(len(methods)):v[ci,mi]=metric_array(err[rows],s[rows,mi],ids[rows])
        v[2]=np.mean(v[:2],axis=0);return v
    point=calc(np.arange(len(frame)));draws=np.full((5000,3,len(methods),len(METRICS)),np.nan)
    rng=np.random.default_rng(BOOT_SEED)
    for rep in range(5000):
        rows=np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))])
        draws[rep]=calc(rows)
        if rep%1000==0:event('paired_bootstrap',cohort=cohort,draw=rep,n_tasks=len(frame),methods=len(methods))
    np.save(RUNTIME/f'{cohort}_BOOTSTRAP_DRAWS.npy',draws)
    result=[];pairs=[]
    for ci,c in enumerate(('HCT116','HEK293T','macro')):
        for mi,m in enumerate(methods):
            for ki,k in enumerate(METRICS):
                ds=draws[:,ci,mi,ki];finite=ds[np.isfinite(ds)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite)>1 else (np.nan,np.nan)
                result.append({'cohort':cohort,'context':c,'method':m,'metric':k,'point':point[ci,mi,ki],
                               'ci95_lower':lo,'ci95_upper':hi,'n_valid_draws':len(finite),'n_tasks':len(frame),'n_genes':len(clusters)})
        comparisons=[]
        for m in methods:
            if m!='Magnitude' and 'Magnitude' in methods:comparisons.append((m,'Magnitude'))
        for ref in ('Old','Expanded'):
            for b in [ref+'_WeightedDistance']+[ref+f'_NullHGB{j}' for j in range(5)]:
                if ref+'_HGB' in methods and b in methods:comparisons.append((ref+'_HGB',b))
        for suffix in ['DirectDistance','WeightedDistance','HGB','DistanceCDF_or_Magnitude','HGB_or_Magnitude']:
            if 'Expanded_'+suffix in methods and 'Old_'+suffix in methods:comparisons.append(('Expanded_'+suffix,'Old_'+suffix))
        for a,b in comparisons:
            ai,bi=methods.index(a),methods.index(b)
            for ki,k in enumerate(METRICS):
                ds=draws[:,ci,ai,ki]-draws[:,ci,bi,ki];finite=ds[np.isfinite(ds)]
                lo,hi=np.quantile(finite,[.025,.975]) if len(finite)>1 else (np.nan,np.nan)
                pairs.append({'cohort':cohort,'context':c,'method_a':a,'method_b':b,'metric':k,
                              'delta':point[ci,ai,ki]-point[ci,bi,ki], 'bootstrap_mean_delta':np.mean(finite) if len(finite) else np.nan,
                              'ci95_lower':lo,'ci95_upper':hi,'n_valid_draws':len(finite),'bootstrap_replicates':5000})
    return pd.DataFrame(result),pd.DataFrame(pairs)


def evaluate():
    if (RUNTIME/'EVALUATION_COMPLETE.json').exists():return
    scores();started=time.monotonic()
    seal=json.loads((RUNTIME/'SCORE_SEAL.json').read_text())
    if binding(RUNTIME/'FROZEN_SCORES_BEFORE_ERROR_OPEN.parquet')!=seal['score_file']:raise RuntimeError('Score seal differs')
    qualification=json.loads((RUNTIME/'QUALIFICATION.json').read_text())
    path=FORMAL/'registered_test_truth/TEST_TASK_ERRORS.parquet'
    if sha(path)!=qualification['original_error_cache']['sha256']:raise RuntimeError('Old errors changed')
    receipt=json.loads((FORMAL/'registered_test_truth/TRUTH_READER_RECEIPT.json').read_text())
    if receipt['truth_errors']['sha256']!=sha(path):raise RuntimeError('Original authorized truth receipt differs')
    score=pd.read_parquet(RUNTIME/'FROZEN_SCORES_BEFORE_ERROR_OPEN.parquet')
    errors=pd.read_parquet(path,columns=QUERY+['true_error_rmse'])
    full=score.merge(errors,on=QUERY,validate='one_to_one')
    if len(full)!=2993:raise RuntimeError('All eligible2993 preserved')
    full.to_parquet(RUNTIME/'SCORES_AND_ORIGINAL_ERRORS.parquet',index=False)
    refmethods=lambda ref:[f'{ref}_DirectDistance',f'{ref}_WeightedDistance',f'{ref}_NegativeSupport',f'{ref}_HGB']+[f'{ref}_NullHGB{j}' for j in range(5)]
    cohorts=[('old_covered',full.Old_n_history>0,['Magnitude']+refmethods('Old')+refmethods('Expanded')),
             ('newly_covered',(full.Old_n_history==0)&(full.Expanded_n_history>0),['Magnitude']+refmethods('Expanded')),
             ('expanded_covered',full.Expanded_n_history>0,['Magnitude']+refmethods('Expanded')),
             ('all_eligible',np.ones(len(full),bool),['Magnitude','Old_DistanceCDF_or_Magnitude','Expanded_DistanceCDF_or_Magnitude','Old_HGB_or_Magnitude','Expanded_HGB_or_Magnitude'])]
    results=[];pairs=[]
    for name,mask,methods in cohorts:
        pathr=DOC/f'{name}_METRICS.csv';pathp=DOC/f'{name}_PAIRED_COMPARISONS.csv'
        if pathr.exists() and pathp.exists():r=pd.read_csv(pathr);p=pd.read_csv(pathp)
        else:
            r,p=stats(full.loc[mask],methods,name);r.to_csv(pathr,index=False);p.to_csv(pathp,index=False)
        results.append(r);pairs.append(p)
    metrics=pd.concat(results,ignore_index=True);pair=pd.concat(pairs,ignore_index=True)
    metrics.to_csv(DOC/'ALL_METRICS.csv',index=False);pair.to_csv(DOC/'ALL_PAIRED_COMPARISONS.csv',index=False)
    macro=metrics[metrics.context.eq('macro')&metrics.metric.eq('utility20')]
    report={'status':'SEEN_EVALUATION_COMPLETE','score_seal':binding(RUNTIME/'SCORE_SEAL.json'),'cached_error_hash_unchanged':sha(path),
            'n_tasks':2993,'n_genes':1750,'source_fits':6,'new_upstream_fits':0,'new_GPU_hours':0,'raw_Orion_truth_reads':0,
            'old_data_or_frozen_models_changed':False,'adoption_based_on_this_SEEN_evaluation':False,
            'wall_seconds_evaluation':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'macro_U20':macro[['cohort','method','point','ci95_lower','ci95_upper']].to_dict('records')}
    write_json(RUNTIME/'EVALUATION_COMPLETE.json',report);write_json(DOC/'EVALUATION_COMPLETE.json',report)
    event('evaluation_complete',macro=report['macro_U20'])


def selftest():
    raw=np.array([[2,0,3],[0,4,1]],float);total=np.array([10,20.])
    assert np.allclose(normalized(raw,total,np.array([2,0])),np.log1p(4000*raw[:,[2,0]]/total[:,None]))
    out=np.zeros((3,2));grouped_add(out,np.array([2,1,2]),np.array([[1,2],[3,4],[5,6.]]))
    assert np.array_equal(out,np.array([[0,0],[3,4],[6,8.]]))
    pred=np.array([[1.,2.],[1.,2.],[1.,2.]])
    e=np.array([[0.,0.],[2.,4.]])
    frame,prior,n=feature_frame(pred,[[0],[0,1],[]],e,np.array([3.,1.]))
    assert frame.prior_uncertainty.iloc[0]==0 and np.isnan(prior[2]).all() and np.isnan(frame.prior_magnitude.iloc[2])
    assert np.isclose(frame.prediction_prior_rmse.iloc[1]**2+frame.prior_uncertainty.iloc[1]**2,2.5)
    y=np.arange(1,41.);ids=np.array([f'q{i:02d}' for i in range(40)])
    assert np.isclose(metric_array(y,y,ids)[0],1.)
    assert np.isnan(metric_array(y[:10],y[:10],ids[:10])[0])
    assert np.array_equal(ecdf_fit_apply([1,2,2,4],[1,2,4]),np.array([.125,.5,.875]))
    write_json(DOC/'CODE_TESTS.json',{'status':'PASS','tests':['full_library_normalization','grouped_aggregation','weighted_distance_identity','single_history_zero_dispersion','no_history_NA','U20_perfect_and_minimum_n','midrank_scoreCDF']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['qualify','bank','scores','evaluate','all','test'],default='all')
    args=parser.parse_args();RUNTIME.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();cpu=time.process_time()
    with threadpool_limits(limits=4):
        selftest()
        if args.phase=='test':return
        qualify()
        if args.phase in ['bank','scores','evaluate','all']:build_bank()
        if args.phase in ['scores','evaluate','all']:scores()
        if args.phase in ['evaluate','all']:evaluate()
    write_json(RUNTIME/'LAST_COMMAND_RECEIPT.json',{'phase':args.phase,'pid':os.getpid(),'status':'COMPLETE',
               'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,
               'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_hours':0})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write_json(RUNTIME/'FAILURE.json',{'pid':os.getpid(),'type':type(exc).__name__,'error':str(exc),'code':binding(__file__)})
        raise
