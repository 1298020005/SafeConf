#!/usr/bin/env python3
"""Audit and intervene on source quality, with frozen upstream predictions."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import itertools
import json
import os
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
import run_e230_history_capacity as capacity
import run_e225_raw_evidence_residual_ranker as scoring

BOOTSTRAPS=128
SPLITS=8
SOURCE_METHODS=('equal','precision','similarity','combined')
PRIMARY='combined_variance_shrink'
BASELINE='equal_count_fixed'
VARIANTS=('clean','shuffle11','shuffle23','shuffle47')
EPS=1e-12


def stamp(): return datetime.now().astimezone().isoformat()
def sha(path): return original.sha256_file(Path(path))
def rng_for(key): return np.random.default_rng(capacity.seed_for(key))
def rank(x): return rankdata(x,method='average')/len(x)


def parse_control(s):
    values=s.astype(str)
    if not set(values.unique()) <= {'0','1'}: raise ValueError('Non-binary control metadata')
    return values.eq('1').to_numpy()


def precision_weights(v,dist,method):
    v=np.asarray(v,float); dist=np.asarray(dist,float)
    if not len(v) or not np.isfinite(v).all() or (v<0).any(): raise ValueError('Invalid uncertainty')
    if not np.isfinite(dist).all() or (dist<0).any(): raise ValueError('Invalid similarity')
    weights=np.ones(len(v))
    if method in ('precision','combined'): weights/=v+np.median(v)+EPS
    if method in ('similarity','combined'): weights*=np.exp(-np.minimum(dist,700))
    if weights.sum()<=0: return np.ones(len(v))/len(v)
    return weights/weights.sum()


def risk_scores(features,use_variance,shrink):
    f=features.copy(); primary=f.analysis_stratum.eq('primary_ge30').to_numpy()
    cols=['family_disagreement','model_source_gap','source_delta_dispersion',
          'log_mean_variance' if use_variance else 'negative_log_source_cells','support_context_deficit']
    values=[]
    for col in cols:
        x=f[col].to_numpy(float); obs=x[np.isfinite(x)]
        x=np.where(np.isfinite(x),x,np.median(obs) if len(obs) else 0.)
        sd=x[primary].std(); values.append((x-x[primary].mean())/sd if sd>1e-14 else np.zeros(len(x)))
    m=rank(f.predicted_magnitude.to_numpy()); s=rank(np.mean(values,axis=0))
    q=f.reliability.to_numpy() if shrink else np.ones(len(f))
    q=np.where(f.n_source_contexts.to_numpy()>0,q,0.)
    return (1-.2*q)*m+.2*q*s


def source_payload(job):
    target,tasks,target_controls,cache,work=job
    path=cache/f'E201_blind_{target}'/'de_adata_test.h5ad'; manifest=path.parent/'E201_BLIND_VIEW_MANIFEST.json'
    if sha(path)!=original.TRAINING_VIEW_SHA256[target] or sha(manifest)!=original.TRAINING_MANIFEST_SHA256[target]:
        raise ValueError('Source hashes changed')
    meta=json.loads(manifest.read_text())
    if meta['target']!=target or meta['n_target_treatments']!=0: raise ValueError('Isolation manifest')
    with threadpool_limits(limits=1):
        a=ad.read_h5ad(path,backed='r'); obs=a.obs.copy(); isctrl=parse_control(obs.control)
        if ((obs.cell_line.astype(str)==target)&~isctrl).any(): raise ValueError('Target expression in source')
        if not a.var_names.is_unique or a.n_vars!=3352: raise ValueError('Invalid gene axis')
        genes_hash=hashlib.sha256('\n'.join(map(str,a.var_names)).encode()).hexdigest()
        rows=[]; effects=[]; batch_sums=[]; batch_counts=[]; offsets=[0]; diagnostics=[]
        try:
            for context in sorted(set(original.TARGETS)-{target}):
                mask=obs.cell_line.astype(str).eq(context).to_numpy()
                ci=np.flatnonzero(mask & isctrl)
                pi=np.flatnonzero(mask & ~isctrl & obs.condition.astype(str).isin(tasks.condition).to_numpy())
                xc=sparse.csr_matrix(a.X[ci]); xp=sparse.csr_matrix(a.X[pi])
                if not np.isfinite(xc.data).all() or not np.isfinite(xp.data).all(): raise ValueError('Nonfinite expression')
                cb=obs.iloc[ci].batch.astype(str).to_numpy(); pb=obs.iloc[pi].batch.astype(str).to_numpy()
                cond=obs.iloc[pi].condition.astype(str).to_numpy()
                gene=obs.iloc[pi].gene_name.astype(str).to_numpy()
                if not np.all(np.char.replace(cond.astype(str),'+ctrl','')==gene): raise ValueError('Condition/gene conflict')
                means={}; control_var={}
                for b in sorted(set(cb)):
                    x=xc[np.flatnonzero(cb==b)]; n=x.shape[0]
                    mean=original.row_mean(x,np.arange(n)); means[b]=mean
                    # Sample variance of the control mean, averaged across aligned genes.
                    variance=np.maximum(np.asarray(x.multiply(x).mean(0)).ravel()-mean**2,0.)
                    control_var[b]=float(variance.mean()/(n-1)) if n>1 else np.nan
                global_control=original.row_mean(xc,np.arange(len(ci)))
                for i,t in enumerate(tasks.itertuples(index=False)):
                    idx=np.flatnonzero(cond==t.condition)
                    if not len(idx): continue
                    labels,counts=np.unique(pb[idx],return_counts=True)
                    if any(b not in means or not np.isfinite(control_var[b]) for b in labels):
                        raise ValueError('Missing or singleton matched control')
                    x=xp[idx].toarray().astype(np.float64)
                    residual=x-np.stack([means[b] for b in pb[idx]])
                    effect=residual.mean(0); fractions=counts/len(idx)
                    sums=np.stack([residual[pb[idx]==b].sum(0) for b in labels])
                    cell_var=float(residual.var(0,ddof=1).mean()/len(idx)) if len(idx)>1 else np.nan
                    ctrl_var=float(sum(w*w*control_var[b] for w,b in zip(fractions,labels)))
                    control=sum(w*means[b] for w,b in zip(fractions,labels))
                    cos=[]; split_rms=[]
                    if len(labels)>=2:
                        for seed in range(SPLITS):
                            perm=rng_for(f'{target}|{context}|{t.condition}|split|{seed}').permutation(len(labels))
                            left,right=np.array_split(perm,2)
                            l=sums[left].sum(0)/counts[left].sum(); r=sums[right].sum(0)/counts[right].sum()
                            den=np.linalg.norm(l)*np.linalg.norm(r)
                            cos.append(float(l@r/den) if den>EPS else np.nan)
                            split_rms.append(float(np.sqrt(np.mean((l-r)**2))))
                    rows.append({'target':target,'context':context,'condition':t.condition,'task_id':t.task_id,
                        'n_cells':len(idx),'n_batches':len(labels),'largest_batch_fraction':float(fractions.max()),
                        'cell_mean_variance':cell_var,'control_mean_variance':ctrl_var,
                        'context_distance_squared':float(np.mean((control-target_controls[i])**2)),
                        'matched_vs_global_control_rmse':float(np.sqrt(np.mean((control-global_control)**2))),
                        'effect_magnitude':float(np.sqrt(np.mean(effect**2))),
                        'split_cosine_mean':float(np.nanmean(cos)) if cos else np.nan,
                        'split_rmse_mean':float(np.mean(split_rms)) if split_rms else np.nan})
                    effects.append(effect.astype(np.float32)); batch_sums.append(sums.astype(np.float32)); batch_counts.append(counts)
                    offsets.append(offsets[-1]+len(labels))
                diagnostics.append({'target':target,'context':context,'source_perturbed_cells':len(pi),
                    'source_control_cells':len(ci),'target_perturbed_rows_read':0,
                    'expression_min':float(min(xc.data.min(),xp.data.min(),0)),
                    'expression_max':float(max(xc.data.max(),xp.data.max()))})
                print('CPU source ready',target,context,flush=True)
        finally: a.file.close()
        out=work/target; out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(out/'source_metadata.csv',index=False)
        np.savez(out/'source_arrays.npz',effects=np.stack(effects),sums=np.concatenate(batch_sums),
                 counts=np.concatenate(batch_counts),offsets=np.asarray(offsets),gene_hash=np.asarray(genes_hash))
        audit={'target':target,'file_sha256':original.TRAINING_VIEW_SHA256[target],
               'gene_axis_hash':genes_hash,'source_pairs':len(rows),'source_records':diagnostics,
               'duplicate_obs_names':int(obs.index.duplicated().sum()),
               'raw_quality_columns_available':False,'processed_control_labels_valid':True,
               'condition_gene_consistency':True,'expressions_finite':True}
        original.atomic_json(out/'source_audit.json',audit)
        return audit


def gpu_uncertainty(target,work,device):
    import torch
    path=work/target
    rows=pd.read_csv(path/'source_metadata.csv'); packed=np.load(path/'source_arrays.npz',allow_pickle=False)
    effects=packed['effects']; sums=packed['sums']; counts=packed['counts']; offsets=packed['offsets']
    variances=[]; error=0.; start=time.monotonic()
    with torch.inference_mode():
        for i,row in enumerate(rows.itertuples()):
            lo,hi=offsets[i:i+2]; a=sums[lo:hi]; n=counts[lo:hi]
            if len(n)<2:
                variances.append(np.nan); continue
            rng=rng_for(f'{target}|{row.context}|{row.condition}|bootstrap')
            w=rng.multinomial(len(n),np.ones(len(n))/len(n),size=BOOTSTRAPS).astype(np.float32)
            denom=w@n; norm=(w/denom[:,None]).astype(np.float32)
            gpu=torch.as_tensor(norm,device=device)@torch.as_tensor(a,device=device)
            v=float(gpu.var(dim=0,unbiased=True).mean().cpu())
            if i==0:
                cpu=norm@a; error=float(np.max(np.abs(gpu.cpu().numpy()-cpu)))
                if error>1e-4: raise ValueError('GPU/CPU bootstrap parity')
            variances.append(v)
    rows['batch_mean_variance']=variances
    rows['estimated_mean_variance']=np.fmax(rows.batch_mean_variance,rows.cell_mean_variance)+rows.control_mean_variance
    if not np.isfinite(rows.estimated_mean_variance).all():
        # A single perturbed cell supplies no empirical variance: keep a missingness flag and conservative source median.
        missing=~np.isfinite(rows.estimated_mean_variance)
        rows['variance_imputed']=missing
        for context in rows.context.unique():
            mask=rows.context.eq(context); values=rows.loc[mask,'estimated_mean_variance'].dropna()
            if not len(values): raise ValueError('Entire source uncertainty unavailable')
            rows.loc[mask & missing,'estimated_mean_variance']=float(values.quantile(.95))
    else: rows['variance_imputed']=False
    rows['relative_estimation_noise']=np.sqrt(rows.estimated_mean_variance)/(rows.effect_magnitude+EPS)
    return rows,effects,{'device':str(device),'seconds':time.monotonic()-start,
        'bootstrap_replicates_per_pair':BOOTSTRAPS,'gpu_cpu_max_abs_difference':error}


def build_scores(tasks,pred_delta,source,effects,variant):
    original_source=source.copy(); values=effects.copy(); permutations=[]
    if variant!='clean':
        seed=int(variant.replace('shuffle',''))
        for context,idx in source.groupby('context',sort=True).indices.items():
            donor=rng_for(f'{tasks.target.iloc[0]}|{context}|shuffle|{seed}').permutation(idx)
            values[idx]=effects[donor]
            source.loc[idx,'estimated_mean_variance']=original_source.iloc[donor].estimated_mean_variance.to_numpy()
            permutations.extend({'target':tasks.target.iloc[0],'context':context,'variant':variant,
                'condition':original_source.iloc[i].condition,'donor_condition':original_source.iloc[j].condition}
                for i,j in zip(idx,donor))
    scale=float(np.median(source.context_distance_squared)); distance=source.context_distance_squared.to_numpy()/(scale+EPS)
    index=source.groupby('condition',sort=True).indices; records=[]; frames=[]
    for method in SOURCE_METHODS:
        for i,t in enumerate(tasks.itertuples(index=False)):
            idx=index.get(t.condition,np.array([],dtype=int)); n=len(idx)
            if n:
                v=source.iloc[idx].estimated_mean_variance.to_numpy()
                w=precision_weights(v,distance[idx],method); mu=w@values[idx]
                uncertainty=float(np.sum(w*w*v)); signal=float(np.mean(mu**2)); q=signal/(signal+uncertainty+EPS)
                gap=float(np.sqrt(np.mean((pred_delta[i]-mu)**2)))
                h=float(np.sqrt(np.sum(w*np.mean((values[idx]-mu)**2,axis=1)))) if n>1 else np.nan
                total=int(source.iloc[idx].n_cells.sum())
            else: gap=h=np.nan; uncertainty=np.nan; q=0.; total=0
            records.append({'task_id':t.task_id,'target':t.target,'condition':t.condition,
                'analysis_stratum':t.analysis_stratum,'variant':variant,'weighting':method,
                'predicted_magnitude':t.predicted_magnitude,'family_disagreement':t.family_disagreement,
                'model_source_gap':gap,'source_delta_dispersion':h,'negative_log_source_cells':-np.log1p(total),
                'log_mean_variance':np.log(uncertainty+EPS) if np.isfinite(uncertainty) else np.nan,
                'support_context_deficit':3-n,'n_source_contexts':n,'reliability':q,'n_source_cells':total})
        f=pd.DataFrame(records[-len(tasks):]); out=f[['task_id','target','condition','analysis_stratum','variant']].copy()
        for feature,gate in itertools.product(('count','variance'),('fixed','shrink')):
            out[method+'_'+feature+'_'+gate]=risk_scores(f,feature=='variance',gate=='shrink')
        frames.append(out)
    scores=frames[0]
    for f in frames[1:]: scores=scores.merge(f,on=['task_id','target','condition','analysis_stratum','variant'],validate='one_to_one')
    scores['magnitude']=rank(tasks.predicted_magnitude); scores['m_plus_d']=.8*scores.magnitude+.2*rank(tasks.family_disagreement)
    return scores,pd.DataFrame(records),permutations


def evaluate(predictions,truth):
    if sha(truth)!=capacity.METRIC_HASH: raise ValueError('Truth file changed')
    labels=pd.read_csv(truth).set_index('task_id').family_rms_error; output=[]
    methods=[f'{w}_{f}_{g}' for w,f,g in itertools.product(SOURCE_METHODS,('count','variance'),('fixed','shrink'))]+['magnitude','m_plus_d']
    for (target,variant),group in predictions.groupby(['target','variant'],sort=True):
        for scope in ('primary','all'):
            f=group[group.analysis_stratum.eq('primary_ge30')] if scope=='primary' else group
            y=labels.loc[f.task_id].to_numpy()
            for method in methods:
                s=f[method].to_numpy(); rho=float(np.corrcoef(rank(s),rank(y))[0,1])
                for budget in (.1,.2,.3):
                    output.append({'target':target,'variant':variant,'scope':scope,'method':method,'budget':budget,
                        'n_tasks':len(f),'spearman':rho,**scoring.metrics(s,y,budget)})
    return pd.DataFrame(output)


def summarize(results):
    a=results[(results.scope=='primary')&(results.budget==.2)]
    clean=a[a.variant=='clean'].pivot(index='target',columns='method',values='utility').sort_index()
    shuffled=a[(a.variant!='clean')&(a.method==PRIMARY)].groupby('target').utility.mean().reindex(clean.index)
    contrasts={'primary_vs_magnitude':clean[PRIMARY]-clean.magnitude,
        'primary_vs_e230':clean[PRIMARY]-clean[BASELINE],
        'clean_vs_shuffled':clean[PRIMARY]-shuffled}
    draws=np.random.default_rng(20260922).integers(0,4,(20000,4)); rows=[]
    for name,d in contrasts.items():
        boot=d.to_numpy()[draws].mean(1)
        rows.append({'contrast':name,'delta':float(d.mean()),'ci95_lower':float(np.quantile(boot,.025)),
            'ci95_upper':float(np.quantile(boot,.975)),'positive_targets':int((d>0).sum()),
            **{'delta_'+t:float(v) for t,v in d.items()}})
    return clean,pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','e201-docs','output','work'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True); args.work.mkdir(parents=True,exist_ok=True)
    if (out/'RUN_STATUS.json').exists(): raise ValueError('Refuse overwrite')
    path=args.e201_docs/'tables/E201_PRETRUTH_RISK_FEATURES.csv'
    if sha(path)!=capacity.FEATURE_HASH: raise ValueError('Task contract changed')
    tasks=pd.read_csv(path); vectors=args.data_root/'txpert_official_20260802/e201/pretruth_vectors'
    old=json.loads((args.e201_docs/'E201_PRETRUTH_RISK_STATUS.json').read_text())
    for name in ('E201_FAMILY_CENTROIDS.npy','E201_CONTROL_CENTROIDS.npy'):
        expected=next(r['sha256'] for r in old['vector_files'] if r['path'].endswith(name))
        if sha(vectors/name)!=expected: raise ValueError('Frozen vectors changed')
    controls=np.load(vectors/'E201_CONTROL_CENTROIDS.npy').astype(float)
    delta=np.load(vectors/'E201_FAMILY_CENTROIDS.npy').astype(float)-controls
    state={'status':'RUNNING','started_at':stamp(),'pid':os.getpid(),'script_sha256':sha(__file__),
        'device':args.device,'workers':args.workers,'e208_test_rows_read':0,'feature_target_truth_reads':0}
    original.atomic_json(out/'RUN_STATUS.json',state)
    try:
        jobs=[(t,tasks[tasks.target==t].reset_index(drop=True),controls[tasks.target==t],
               args.data_root/'txpert_official_20260802/cache',args.work) for t in original.TARGETS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool: audits=list(pool.map(source_payload,jobs))
        if len({v['gene_axis_hash'] for v in audits})!=1: raise ValueError('Source gene order mismatch')
        import torch
        torch.set_num_threads(4)
        if args.device.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        pieces=[]; feature_pieces=[]; quality=[]; maps=[]; compute=[]
        for target in original.TARGETS:
            source,effects,stats=gpu_uncertainty(target,args.work,torch.device(args.device))
            compute.append({'target':target,**stats}); quality.append(source)
            for variant in VARIANTS:
                selection=tasks.target==target; t=tasks[selection].reset_index(drop=True)
                pred,features,perm=build_scores(t,delta[selection],source.copy(),effects,variant)
                pieces.append(pred); feature_pieces.append(features); maps.extend(perm)
            print('GPU quality complete',target,flush=True)
        predictions=pd.concat(pieces,ignore_index=True); features=pd.concat(feature_pieces,ignore_index=True)
        if not np.isfinite(predictions.select_dtypes('number')).all().all(): raise ValueError('Nonfinite score')
        files={}
        for name,frame in [('PREDICTIONS',predictions),('FEATURES',features),('SOURCE_QUALITY',pd.concat(quality)),('SHUFFLE_MAP',pd.DataFrame(maps))]:
            file=out/f'{name}.csv.gz'; frame.to_csv(file,index=False,compression={'method':'gzip','mtime':0}); files[file.name]=sha(file)
        original.atomic_json(out/'PRE_EVALUATION_SEAL.json',{'sealed_at':stamp(),'files':files,'source_audits':audits,
            'compute':compute,'target_errors_in_features':False})
        results=evaluate(predictions,args.e201_docs/'formal_core_evaluation/tables/E201_TASK_METRICS.csv')
        results.to_csv(out/'RESULTS.csv',index=False); clean,contrasts=summarize(results)
        clean.to_csv(out/'CLEAN_COMPARATORS.csv'); contrasts.to_csv(out/'CONTRASTS.csv',index=False)
        c=contrasts.set_index('contrast')
        passed=bool((c.positive_targets>=3).all() and c.loc['primary_vs_magnitude','delta']>=.02
                    and c.loc['primary_vs_e230','delta']>=.005 and c.loc['clean_vs_shuffled','delta']>0)
        state.update(status='COMPLETE',finished_at=stamp(),development_gate='PASS' if passed else 'NOT_SUPPORTED',
            primary_method=PRIMARY,primary=contrasts.to_dict('records'),source_pairs=int(sum(a['source_pairs'] for a in audits)),
            cuda_used=args.device.startswith('cuda'),compute=compute,source_audits=audits)
    except BaseException as e:
        state.update(status='FAILED',finished_at=stamp(),reason=repr(e)); raise
    finally: original.atomic_json(out/'RUN_STATUS.json',state)
    print(contrasts.to_string(index=False),flush=True)


if __name__=='__main__': main()
