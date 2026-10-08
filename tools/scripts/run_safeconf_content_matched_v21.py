#!/usr/bin/env python3
"""Twenty fixed content nulls on the existing three-fold McFaline DEV."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,time,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.submission_evidence import (
    NULL_SEEDS,write_json,sha,digest_ids,endpoint_arrays,ClusterBootstrap,point_metrics,summarize_draws)
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN,COMMON

BASE=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/measurement/simple_risk_sensitivity_v1'


def permute_blocks(memory,allowed,seed):
    donor=np.arange(len(memory));rng=np.random.default_rng(seed)
    m=memory.iloc[allowed].copy()
    m['support_bin']=np.floor(np.log(m.n_cells.to_numpy(float))/np.log(1.1)).astype(int)
    keys=['study_id','context','condition','perturbation_type','effect_contract_id','gene_space_id','support_bin']
    for _,stratum in m.groupby(keys,sort=True,dropna=False):
        blocks={str(g):np.asarray(z.index,int) for g,z in stratum.groupby('perturbation_target',sort=True)}
        sizes={}
        for g,rows in blocks.items():sizes.setdefault(len(rows),[]).append(g)
        for names in sizes.values():
            if len(names)<2:continue
            order=np.asarray(sorted(names),object)[rng.permutation(len(names))]
            for recipient,source in zip(order,np.roll(order,1)):
                r=blocks[recipient];s=blocks[source]
                r=r[np.argsort(memory.iloc[r].experiment_id.to_numpy(str),kind='stable')]
                s=s[np.argsort(memory.iloc[s].experiment_id.to_numpy(str),kind='stable')]
                donor[r]=s
    return donor


def main():
    out=RUN/'content_matched';out.mkdir(parents=True,exist_ok=True)
    if (out/'STATUS.json').exists() and json.loads((out/'STATUS.json').read_text()).get('status')=='COMPLETE':return
    start,cpu=time.monotonic(),time.process_time()
    frozen=pd.read_csv(BASE/'FROZEN_SIMPLE_RISKS.csv.gz')
    tasks=pd.read_csv(COMMON/'VALIDATION_TASKS.csv').set_index('task_id').loc[frozen.task_id].reset_index()
    original_tasks=pd.read_csv(COMMON/'VALIDATION_TASKS.csv').reset_index().set_index('task_id')
    ii=original_tasks.loc[frozen.task_id,'index'].to_numpy(int)
    pred=np.asarray(np.load(COMMON/'VALIDATION_CALIBRATED_EFFECTS.npy',mmap_mode='r')[ii],float)
    truth=np.asarray(np.load(COMMON/'VALIDATION_TRUE_EFFECTS.npy',mmap_mode='r')[ii],float)
    error=endpoint_arrays(pred,truth)
    memory=pd.read_parquet(COMMON/'public_mcfaline_trainval/public_memory.parquet').sort_values('effect_vector_row').reset_index(drop=True)
    effects=np.load(COMMON/'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy',mmap_mode='r')
    assert effects.shape==(len(memory),2840) and frozen.task_id.is_unique
    by_gene=memory.groupby('perturbation_target').indices
    scores=frozen[['task_id','gene','context','treatment','fold']].copy();scores['target']=scores.context
    scores['Magnitude']=frozen.Magnitude;scores['SupportAll']=np.nan;scores['SupportSelected']=np.nan;scores['PublicRule']=np.nan
    groups={};audits=[]
    for fold in sorted(frozen.fold.unique()):
        forbidden={'McFaline23::'+t for t in frozen.loc[frozen.fold==fold,'task_id']}
        allowed=np.flatnonzero(~memory.experiment_id.isin(forbidden).to_numpy())
        selected=[]
        for q in np.flatnonzero(frozen.fold.to_numpy()==fold):
            t=tasks.iloc[q];ids=np.asarray(by_gene.get(str(t.gene),[]),int)
            m=memory.iloc[ids]
            legal=(~m.experiment_id.isin(forbidden))&~((m.context.astype(str)==str(t.context))&(m.condition.astype(str)==str(t.treatment)))
            ids=ids[legal.to_numpy()]
            if not len(ids):raise RuntimeError('registered DEV query lost all history')
            m=memory.iloc[ids];counts=m.n_cells.to_numpy(float)
            same=m.context.astype(str).eq(str(t.context)).to_numpy();weights=counts*(same if same.any() else 1);weights/=weights.sum()
            scores.loc[q,'SupportAll']=-np.log1p(counts.sum())
            scores.loc[q,'SupportSelected']=-np.log1p(counts[weights>0].sum())
            scores.loc[q,'PublicRule']=np.sqrt(weights@np.mean((np.asarray(effects[ids],float)-pred[q])**2,1))
            selected.append((q,ids,weights))
        groups[fold]=(allowed,selected,forbidden)
    assert np.allclose(scores.PublicRule,frozen.Public_R_history,rtol=1e-6,atol=1e-9)
    assert np.allclose(scores.SupportAll,frozen.NegativeSupport,rtol=1e-8,atol=1e-10)
    write_json(out/'EXPERIMENT_FREEZE.json',{'seeds':NULL_SEEDS,'support_log_bin_base':1.1,
        'minimum_moved_fraction':.6,'strata':['study','context','condition','perturbation_type','effect_contract','gene_axis','support_bin'],
        'block':'perturbation within compatible stratum, equal layout/size',
        'selection_uses_query_truth':False,'current_scores_reproduced':True,'risk_builder_error_labels':0,
        'input_sha256':sha(BASE/'FROZEN_SIMPLE_RISKS.csv.gz')})
    for seed in NULL_SEEDS:
        null=np.full(len(scores),np.nan)
        for fold,(allowed,selected,forbidden) in groups.items():
            donor=permute_blocks(memory,allowed,seed+int(fold))
            used=np.concatenate([ids[w>0] for _,ids,w in selected])
            if set(memory.iloc[donor[used]].experiment_id)&forbidden:raise RuntimeError('forbidden content donor')
            moved=float(np.mean(donor[used]!=used))
            audits.append({'seed':seed,'fold':fold,'moved_fraction':moved,'valid_null':moved>=.6,
                'used_histories':len(used),'forbidden_donors':0})
            for q,ids,w in selected:
                null[q]=np.sqrt(w@np.mean((np.asarray(effects[donor[ids]],float)-pred[q])**2,1))
        scores[f'ContentNull_{seed}']=null
        print(json.dumps({'content_null':seed,'moved_fractions':[r['moved_fraction'] for r in audits if r['seed']==seed]}),flush=True)
    scores.to_parquet(out/'PER_QUERY_SCORES.parquet',index=False)
    pd.DataFrame(audits).to_csv(out/'PERMUTATION_AUDIT.csv',index=False)
    results=[];intervals=[]
    for endpoint,e in error.items():
        engine=ClusterBootstrap(scores,e)
        macro=lambda s:float(np.nanmean([point_metrics(e[ix],np.asarray(s)[ix],scores.task_id.to_numpy(str)[ix])['utility'] for ix in scores.groupby('target').indices.values()]))
        real=scores.PublicRule.to_numpy(float);real_point=macro(real)
        for method in ['Magnitude','SupportAll','SupportSelected','PublicRule']+[f'ContentNull_{s}' for s in NULL_SEEDS]:
            s=scores[method].to_numpy(float)
            results.append({'endpoint':endpoint,'method':method,'utility20_macro':macro(s),
                **point_metrics(e,s,scores.task_id.to_numpy(str))})
            if method!='PublicRule':
                intervals.append({'endpoint':endpoint,'comparison':'PublicRule-minus-'+method,
                    **summarize_draws(engine.difference(real,s),real_point-macro(s))})
    pd.DataFrame(results).to_csv(out/'RESULTS.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'PAIRED_CLUSTER_BOOTSTRAP.csv',index=False)
    valid_seeds={s for s in NULL_SEEDS if all(r['valid_null'] for r in audits if r['seed']==s)}
    primary=[r for r in results if r['endpoint']=='delta_rmse']
    rp=next(r['utility20_macro'] for r in primary if r['method']=='PublicRule')
    nulls=[r['utility20_macro'] for r in primary if r['method'].startswith('ContentNull_') and int(r['method'].split('_')[-1]) in valid_seeds]
    write_json(out/'STATUS.json',{'status':'COMPLETE','n_tasks':len(scores),'n_clusters':scores.gene.nunique(),
        'permutations':20,'effective_permutations':len(nulls),'public_utility20':rp,
        'null_utility20_median':float(np.median(nulls)) if nulls else None,
        'randomization_p_nominal':(1+sum(v>=rp for v in nulls))/(1+len(nulls)) if len(nulls)==20 else None,
        'nulls_were_not_selected_by_performance':True,'new_fits':0,'new_gpu_hours':0,
        'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
        'permanent_test_truth_opened':False})


if __name__=='__main__':main()
