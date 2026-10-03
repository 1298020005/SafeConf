#!/usr/bin/env python3
"""DEV-only independent-repeat diagnosis; no predictor fitting or primary-label change.

Only registered validation treated rows and train/validation controls are read.
The supplementary WMSE uses the published squared min-max absolute score rule
with explicitly adapted Welch control scores; it is not the original full DEG
pipeline and never selects a SafeConf method.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import metrics, ids_hash

COMMON=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
RUNTIME=Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/measurement_v1')
OUT=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/measurement'
H5=Path('/home/yyf/data/perturbench_mcfaline23_official/mcfaline23_gxe_processed.h5ad')
SPLIT=H5.parent/'splits/mcfaline23_gxe_splits/full_covariate_split.csv'
MODES=('guide','plate','cell')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
    os.replace(temp,path)

def col(group,key):
    x=group[key]
    if isinstance(x,h5py.Dataset):return np.asarray(x).astype(str)
    codes=x['codes'][:];cats=np.asarray(x['categories']).astype(str)
    if np.any(codes<0):raise ValueError(f'missing required metadata: {key}')
    return cats[codes].astype(str)

def ordered(values,prefix):
    return sorted(values,key=lambda v:hashlib.sha256((prefix+str(v)).encode()).hexdigest())

def rho(a,b):
    a,b=np.asarray(a),np.asarray(b);ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.ptp(a[ok])==0 or np.ptp(b[ok])==0:return np.nan
    return float(spearmanr(a[ok],b[ok]).statistic)

def prepare():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREPARE_RECEIPT.json').exists():raise FileExistsError('completed preparation is immutable')
    tasks=pd.read_csv(COMMON/'VALIDATION_TASKS.csv');genes=json.loads((COMMON/'GENE_IDS.json').read_text())
    if (len(tasks),tasks.gene.nunique(),len(genes))!=(542,377,2840):raise ValueError('DEV contract changed')
    bindings=[{'path':str(p),'sha256':sha(p)} for p in [Path(__file__),SPLIT,COMMON/'VALIDATION_TASKS.csv',COMMON/'GENE_IDS.json',COMMON/'VALIDATION_TRUE_EFFECTS.npy',COMMON/'VALIDATION_CALIBRATED_EFFECTS.npy',RUNTIME/'official_metric_sources/wmse.py']]
    write_json(OUT/'REGISTRATION.json',{'role':'DEV_SEEN_MEASUREMENT_DIAGNOSTIC','modes':MODES,'minimum_treated_and_control_cells_per_half':5,'time_cap_seconds':14400,'primary_error_unchanged':True,'new_model_fits':0,'wmse':'Published weighted-MSE and squared min-max abs-score weighting; Welch treated-v-control score adaptation, not full authors DEG preprocessing','official_source_url':'https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines/blob/main/cellsimbench/core/data_manager.py','official_source_git_blob':'45ffda662e5be0bc4d4a086a6a363d985349015f','bindings':bindings,'h5_file_stat':{'bytes':H5.stat().st_size,'mtime_ns':H5.stat().st_mtime_ns}})
    lookup={s:i for i,s in enumerate(tasks.task_id.astype(str))};n=len(tasks);g=len(genes)
    with h5py.File(H5,'r',rdcc_nbytes=256*1024*1024) as f:
        ob=f['obs'];cell=col(ob,'_index');context=col(ob,'cell_type');treatment=col(ob,'treatment');pert=col(ob,'perturbation');plate=col(ob,'PCR_plate');guide=col(ob,'gRNA_id');control=col(ob,'control')=='1'
        split=pd.read_csv(SPLIT,header=None,names=['cell','role']).set_index('cell').role
        roles=split.reindex(cell).to_numpy();assert not pd.isna(roles).any()
        task_id=np.char.add(np.char.add(np.char.add(np.char.add(pert,'::'),context),'::'),treatment)
        qr=np.asarray([lookup.get(s,-1) for s in task_id],int)
        treated=(roles=='val')&(~control)&(qr>=0)
        states=set(zip(tasks.context.astype(str),tasks.treatment.astype(str)))
        relevant=np.asarray([(a,b) in states for a,b in zip(context,treatment)])
        controls=np.isin(roles,['train','val'])&control&relevant
        allowed=treated|controls; selected=np.flatnonzero(allowed)
        if (roles[selected]=='test').any():raise ValueError('TEST row allowed in DEV diagnostic')
        mf=pd.DataFrame({'row':selected,'task':qr[selected],'context':context[selected],'treatment':treatment[selected],'plate':plate[selected],'guide':guide[selected],'cell':cell[selected],'treated':treated[selected]})
        platekeys=sorted(set(zip(mf.context,mf.treatment,mf.plate)));pmap={key:i for i,key in enumerate(platekeys)}
        pi=np.full(len(cell),-1,int)
        pi[selected]=[pmap[x] for x in zip(mf.context,mf.treatment,mf.plate)]
        parts=np.full((3,len(cell)),-1,np.int8);ctrlhalf=np.full(len(cell),-1,np.int8)
        tm=mf[mf.treated]
        for task,rows in tm.groupby('task',sort=True):
            gids=ordered(rows.guide.unique(),f'SafeConf-measure-guide-v1|{task}|');gm={v:j%2 for j,v in enumerate(gids)}
            rr=rows.row.to_numpy();parts[0,rr]=rows.guide.map(gm).to_numpy()
        for state,rows in mf.groupby(['context','treatment'],sort=True):
            plates=ordered(rows.plate.unique(),f'SafeConf-measure-plate-v1|{state}|');pm={v:j%2 for j,v in enumerate(plates)}
            rr=rows[rows.treated].row.to_numpy();parts[1,rr]=[pm[plate[x]] for x in rr]
        for _,rows in tm.groupby(['task','guide','plate'],sort=True):
            rr=ordered(rows.row.to_list(),'SafeConf-measure-cell-v1|')
            parts[2,rr]=np.arange(len(rr))%2
        for _,rows in mf[~mf.treated].groupby(['context','treatment','plate'],sort=True):
            rr=sorted(rows.row.to_list(),key=lambda j:hashlib.sha256(('SafeConf-measure-control-v1|'+cell[j]).encode()).hexdigest())
            ctrlhalf[rr]=np.arange(len(rr))%2
        assert (parts[:,np.flatnonzero(treated)]>=0).all() and (ctrlhalf[np.flatnonzero(controls)]>=0).all()
        memberships=[]
        for m,name in enumerate(MODES):
            for half in range(2):
                use=np.flatnonzero(treated&(parts[m]==half))
                memberships.append({'mode':name,'half':half,'treated_cells':len(use),'treated_cell_ids_sha256':ids_hash(cell[use]),'overlap_other_half':len(set(use)&set(np.flatnonzero(treated&(parts[m]==1-half))))})
        pd.DataFrame(memberships).to_csv(OUT/'REPEAT_MEMBERSHIP_AUDIT.csv',index=False)
        sums=np.zeros((3,2,n,g));count=np.zeros((3,2,n),int)
        fullsum=np.zeros((n,g));fullsq=np.zeros((n,g));fullcount=np.zeros(n,int)
        cs=np.zeros((len(platekeys),2,g));css=np.zeros_like(cs);cc=np.zeros((len(platekeys),2),int)
        plate_counts={}
        names=col(f['var'],'gene_name');axis={s:i for i,s in enumerate(names)}
        projection=np.full(len(names),-1,int)
        for j,s in enumerate(genes):projection[axis[s]]=j
        x=f['X'];ptr=x['indptr'][:]
        if not isinstance(x,h5py.Group):raise ValueError('expected registered CSR expression')
        for j,r in enumerate(selected):
            if time.monotonic()-started>14400:raise TimeoutError('measurement cap')
            if roles[r]=='test':raise RuntimeError('test expression read guard')
            a,b=int(ptr[r]),int(ptr[r+1]);dest=projection[x['indices'][a:b]];values=x['data'][a:b];ok=dest>=0
            dest=dest[ok];values=values[ok].astype(float)
            if treated[r]:
                q=qr[r];fullsum[q,dest]+=values;fullsq[q,dest]+=values*values;fullcount[q]+=1
                for m in range(3):
                    h=int(parts[m,r]);sums[m,h,q,dest]+=values;count[m,h,q]+=1
                    key=(m,h,q,int(pi[r]));plate_counts[key]=plate_counts.get(key,0)+1
            else:
                p=int(pi[r]);h=int(ctrlhalf[r]);cs[p,h,dest]+=values;css[p,h,dest]+=values*values;cc[p,h]+=1
            if (j+1)%5000==0:print(json.dumps({'stage':'read_DEV_only','rows':j+1,'total':len(selected),'seconds':time.monotonic()-started}),flush=True)
    if (fullcount==0).any():raise ValueError('missing registered DEV task')
    original=np.load(COMMON/'VALIDATION_TRUE_EFFECTS.npy');ctrl0=np.load(COMMON/'VALIDATION_CONTROLS.npy')
    canonical=fullsum/fullcount[:,None]-ctrl0
    maxdiff=float(np.max(np.abs(canonical-original)))
    if not np.allclose(canonical,original,rtol=1e-5,atol=2e-6):raise ValueError(f'DEV truth aggregate changed: {maxdiff}')
    cm=np.zeros_like(sums);control_valid=np.ones((3,2,n),bool);weighted_cn=np.zeros((3,2,n))
    for (m,h,q,p),c in plate_counts.items():
        if m==1:
            nc=int(cc[p].sum());sv=cs[p].sum(0)
        else:nc=int(cc[p,h]);sv=cs[p,h]
        if nc<5:control_valid[m,h,q]=False;continue
        weight=c/max(int(count[m,h,q]),1);cm[m,h,q]+=weight*sv/nc;weighted_cn[m,h,q]+=weight*nc
    valid=(count>=5)&control_valid
    effect=np.divide(sums,count[...,None],out=np.full_like(sums,np.nan),where=count[...,None]>0)-cm
    effect[~valid]=np.nan
    pred=np.load(COMMON/'VALIDATION_CALIBRATED_EFFECTS.npy')
    # Supplementary score weights: fixed Welch effect-v-control t statistics.
    qvar=np.maximum((fullsq-fullsum**2/fullcount[:,None])/np.maximum(fullcount[:,None]-1,1),0)
    variance_ctrl=np.zeros_like(qvar);mean_ctrl=np.zeros_like(qvar);nctrl=np.zeros(n)
    for q in range(n):
        statesel=[i for i,k in enumerate(platekeys) if k[:2]==(str(tasks.iloc[q].context),str(tasks.iloc[q].treatment))]
        cn=int(cc[statesel].sum());sm=cs[statesel].sum((0,1));ss=css[statesel].sum((0,1))
        if cn<5:continue
        nctrl[q]=cn;mean_ctrl[q]=sm/cn;variance_ctrl[q]=np.maximum((ss-sm*sm/cn)/(cn-1),0)
    scores=np.abs((fullsum/fullcount[:,None]-mean_ctrl)/np.sqrt(np.maximum(qvar/fullcount[:,None]+variance_ctrl/np.maximum(nctrl[:,None],1),1e-12)))
    low=scores.min(1,keepdims=True);span=scores.max(1,keepdims=True)-low
    weights=np.divide(scores-low,span,out=np.zeros_like(scores),where=span>0)**2
    weights=np.divide(weights,weights.sum(1,keepdims=True),out=np.full_like(weights,np.nan),where=weights.sum(1,keepdims=True)>0)
    spec=importlib.util.spec_from_file_location('official_wmse',RUNTIME/'official_metric_sources/wmse.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    wmse=np.sum(weights*(pred-original)**2,axis=1)
    check=np.asarray([module.wmse(pred[i],original[i],weights[i]) for i in range(n)])
    assert np.allclose(wmse,check,equal_nan=True,atol=1e-12)
    arrays=RUNTIME/'DEV_REPEAT_EFFECTS.npz'
    np.savez_compressed(arrays,effect=effect,valid=valid,counts=count,weighted_control_count=weighted_cn,task_ids=tasks.task_id.to_numpy(str),prediction=pred,primary=original,wmse_welch=wmse)
    result=tasks[['task_id','gene','context','treatment']].copy();result['true_error_rmse']=np.sqrt(np.mean((pred-original)**2,1));result['WMSE_Welch_control_adaptation']=wmse
    for m,name in enumerate(MODES):
        for h in range(2):
            result[f'{name}_{h}_rmse']=np.sqrt(np.mean((pred-effect[m,h])**2,axis=1));result[f'{name}_{h}_cells']=count[m,h]
        result[f'{name}_duplicate_rmse']=np.sqrt(np.mean((effect[m,0]-effect[m,1])**2,1))
        result[f'{name}_control_rmse_on_half1']=np.sqrt(np.mean(effect[m,1]**2,1))
    result.to_csv(OUT/'DEV_REPEAT_TASK_ERRORS.csv.gz',index=False)
    assoc=[]
    for context,rows in result.groupby('context',sort=True):
        for name in MODES:
            a=rows[f'{name}_0_rmse'];b=rows[f'{name}_1_rmse'];ok=np.isfinite(a)&np.isfinite(b)
            assoc.append({'context':context,'mode':name,'planned_tasks':len(rows),'valid_paired_tasks':int(ok.sum()),'rho_between_repeat_errors':rho(a,b),'rho_primary_half0':rho(rows.true_error_rmse,a),'rho_primary_half1':rho(rows.true_error_rmse,b),'positive_duplicate_rmse':float(rows.loc[ok,f'{name}_duplicate_rmse'].mean()),'negative_control_rmse':float(rows.loc[ok,f'{name}_control_rmse_on_half1'].mean()),'independent_control_blocks':True})
    pd.DataFrame(assoc).to_csv(OUT/'REPEAT_ASSOCIATIONS.csv',index=False)
    write_json(OUT/'PREPARE_RECEIPT.json',{'status':'COMPLETE','role':'DEV_ONLY','n_tasks':n,'n_genes':tasks.gene.nunique(),'selected_expression_rows':len(selected),'test_expression_rows_read':0,'primary_truth_max_absolute_difference':maxdiff,'wall_seconds':time.monotonic()-started,'arrays':{'path':str(arrays),'sha256':sha(arrays)},'new_model_fits':0,'new_GPU_hours':0,'wmse_scope':'squared min-max abs Welch control score adaptation, diagnostic only','next_action':'Evaluate all preregistered feedback DEV scorers on primary and alternate repeat errors'})
    print(json.dumps({'status':'PREPARE_COMPLETE','wall_seconds':time.monotonic()-started,'output':str(OUT)},ensure_ascii=False),flush=True)

def evaluate():
    """Freeze simple DEV risk scores, then compare unchanged alternate truths."""
    started=time.monotonic();out=OUT/'simple_risk_sensitivity_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'RECEIPT.json').exists():raise FileExistsError('completed measurement evaluation is immutable')
    receipt=json.loads((OUT/'PREPARE_RECEIPT.json').read_text())
    if sha(receipt['arrays']['path'])!=receipt['arrays']['sha256']:raise ValueError('repeat arrays changed')
    tasks=pd.read_csv(COMMON/'VALIDATION_TASKS.csv');pred=np.load(COMMON/'VALIDATION_CALIBRATED_EFFECTS.npy')
    memory=pd.read_parquet(COMMON/'public_mcfaline_trainval/public_memory.parquet').sort_values('effect_vector_row').reset_index(drop=True)
    effects=np.load(COMMON/'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy',mmap_mode='r')
    foldgenes=ordered(tasks.gene.unique(),'SafeConf-target-dev-v1|');mapping={x:i%3 for i,x in enumerate(foldgenes)}
    tasks['fold']=tasks.gene.map(mapping);risks=tasks[['task_id','gene','context','treatment','fold']].copy()
    risks['Magnitude']=np.sqrt(np.mean(pred.astype(float)**2,axis=1));risks['Public_R_history']=np.nan;risks['NegativeSupport']=np.nan
    by_gene=memory.groupby('perturbation_target',sort=False).indices;membership=[]
    for fold in range(3):
        forbidden={'McFaline23::'+t for t in tasks.loc[tasks.fold==fold,'task_id']}
        for row in np.flatnonzero(tasks.fold.to_numpy()==fold):
            t=tasks.iloc[row];ix=np.asarray(by_gene.get(str(t.gene),[]),int);m=memory.iloc[ix]
            legal=(~m.experiment_id.isin(forbidden))&~((m.context.astype(str)==str(t.context))&(m.condition.astype(str)==str(t.treatment)))
            ix=ix[legal.to_numpy()];m=memory.iloc[ix]
            if not len(ix):continue
            s=m.n_cells.to_numpy(float);same=m.context.astype(str).eq(str(t.context)).to_numpy();w=s*(same if same.any() else 1);w=w/w.sum()
            vectors=np.asarray(effects[ix],float)
            risks.loc[row,'Public_R_history']=np.sqrt(w@np.mean((vectors-pred[row])**2,axis=1));risks.loc[row,'NegativeSupport']=-np.log1p(s.sum())
            membership.append({'task_id':t.task_id,'fold':fold,'history_ids_hash':ids_hash(m.experiment_id),'forbidden_ids_hash':ids_hash(forbidden),'forbidden_intersection':len(set(m.experiment_id)&forbidden),'selected_histories':int((w>0).sum())})
    risks.to_csv(out/'FROZEN_SIMPLE_RISKS.csv.gz',index=False);pd.DataFrame(membership).to_csv(out/'HISTORY_DEPENDENCY_AUDIT.csv',index=False)
    write_json(out/'PREDICTION_FREEZE.json',{'risk_file_sha256':sha(out/'FROZEN_SIMPLE_RISKS.csv.gz'),'source':'prediction and allowed Public effects only','primary_label_read_in_risk_builder':False,'development_folds':3,'query_experiments_excluded_globally':True})
    truths=pd.read_csv(OUT/'DEV_REPEAT_TASK_ERRORS.csv.gz').set_index('task_id').loc[risks.task_id].reset_index()
    truthcols=['true_error_rmse','WMSE_Welch_control_adaptation']+[f'{m}_{h}_rmse' for m in MODES for h in range(2)]
    methods=['Magnitude','Public_R_history','NegativeSupport'];allrows=[];percontext=[]
    for truthcol in truthcols:
        for method in methods:
            for context,part in risks.groupby('context',sort=True):
                use=part.index.to_numpy();frame=part.copy();frame['true_error_rmse']=truths.loc[use,truthcol].to_numpy()
                result=metrics(frame,part[method].to_numpy());percontext.append({'diagnostic_truth':truthcol,'method':method,'context':context,**result})
    context=pd.DataFrame(percontext);context.to_csv(out/'CONTEXT_METRICS.csv',index=False)
    fields=['utility20','spearman','aurc','high_risk_miss_rate']
    context.groupby(['diagnostic_truth','method'],as_index=False)[fields].mean().to_csv(out/'MACRO_METRICS.csv',index=False)
    # Paired gene uncertainty for error-rank reproducibility, independent of method adoption.
    gids=sorted(risks.gene.unique());groups=[np.flatnonzero(risks.gene.to_numpy()==gene) for gene in gids];rng=np.random.default_rng(20260930)
    contexts=risks.context.to_numpy(str);rows=[];draws=np.full((5000,3,3),np.nan)
    for b in range(5000):
        ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        for ci,c in enumerate(sorted(set(contexts))):
            use=ix[contexts[ix]==c]
            for mi,m in enumerate(MODES):draws[b,ci,mi]=rho(truths[f'{m}_0_rmse'].to_numpy()[use],truths[f'{m}_1_rmse'].to_numpy()[use])
        if (b+1)%1000==0:print(json.dumps({'stage':'repeat_rank_cluster_bootstrap','completed':b+1}),flush=True)
    for ci,c in enumerate(sorted(set(contexts))):
        use=np.flatnonzero(contexts==c)
        for mi,m in enumerate(MODES):
            vals=draws[:,ci,mi]
            rows.append({'context':c,'mode':m,'spearman':rho(truths[f'{m}_0_rmse'].to_numpy()[use],truths[f'{m}_1_rmse'].to_numpy()[use]),'lower95':float(np.nanquantile(vals,.025)),'upper95':float(np.nanquantile(vals,.975)),'valid_draws':int(np.isfinite(vals).sum()),'bootstrap_gene_clusters':len(gids)})
    pd.DataFrame(rows).to_csv(out/'REPEAT_RANK_INTERVALS.csv',index=False)
    np.savez_compressed(RUNTIME/'REPEAT_RANK_BOOTSTRAP.npz',draws=draws)
    write_json(out/'RECEIPT.json',{'status':'COMPLETE','new_model_fits':0,'new_GPU_hours':0,'role':'DEV_DIAGNOSTIC_NO_METHOD_SELECTION','primary_error_unchanged':True,'bootstrap':5000,'wall_seconds':time.monotonic()-started,'wmse_is_explicitly_control_Welch_adaptation':True,'all_preregistered_simple_scorers_reported':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','evaluate']);args=p.parse_args()
    if args.phase=='prepare':prepare()
    else:evaluate()

if __name__=='__main__':main()
