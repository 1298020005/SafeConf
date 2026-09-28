#!/usr/bin/env python3
"""One preregistered, validation-only development batch. No model expansion."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, hashlib, json, time, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/实验结果/ModelDecision_light_batch_20260928'
SOURCE=Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
P=['prediction_l2_norm','prediction_abs_mean','model_disagreement_rmse','model_disagreement_cosine']
Q=['context_similarity_max','context_similarity_mean','perturbation_support_count']
H=['prediction_norm_ratio','prediction_magnitude_deviation','historical_residual_risk','perturbation_effect_stability','perturbation_effect_variance']
GROUPS={'M':P[:1],'P':P,'PQ':P+Q,'PQH':P+Q+H}
META=['dataset_name','fold_id','split','task_key','context','perturbation','predictor_name']
CHEM={'McFarlandTsherniak2020','SrivatsanTrapnell2020_sciplex3'}
SEED=928

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def allowed_frame(path,columns,split='val'):
    """Select row numbers from metadata before parsing numerical label columns."""
    meta=pd.read_csv(path,usecols=['fold_id','split'])
    keep=set(meta.index[(meta.fold_id==0)&(meta.split==split)].to_numpy()+1)
    df=pd.read_csv(path,usecols=columns,skiprows=lambda row:row!=0 and row not in keep)
    assert df['split'].eq(split).all() and df.fold_id.eq(0).all()
    return df

def features(fit,query,cols):
    x=fit[cols].to_numpy(float); z=query[cols].to_numpy(float)
    badx=~np.isfinite(x); badz=~np.isfinite(z)
    med=np.array([np.median(x[~badx[:,j],j]) if (~badx[:,j]).any() else 0 for j in range(x.shape[1])])
    x=np.where(badx,med,x); z=np.where(badz,med,z)
    mu=x.mean(0); sd=x.std(0); sd=np.where(sd>1e-8,sd,1.)
    return np.c_[(x-mu)/sd,badx].astype('float32'),np.c_[(z-mu)/sd,badz].astype('float32')

def anchor_predict(fit,query):
    # Positive slope, one scalar input, fit-local scaling.
    x,z=features(fit,query,P[:1]); y=fit.true_error_rmse.to_numpy(float)
    model=Ridge(alpha=10.,positive=True).fit(x[:,:1],y)
    return model.predict(z[:,:1])

def anchor_oof(fit):
    out=np.full(len(fit),np.nan)
    for a,b in GroupKFold(3).split(fit,groups=fit.perturbation):
        assert not set(fit.iloc[a].perturbation)&set(fit.iloc[b].perturbation)
        out[b]=anchor_predict(fit.iloc[a],fit.iloc[b])
    assert np.isfinite(out).all()
    return out

def learn(x,z,y,kind,device):
    mu=float(np.mean(y)); scale=max(float(np.std(y)),1e-8)
    yy=(y-mu)/scale
    if kind=='Ridge':
        model=Ridge(alpha=10.).fit(x,yy); pred=model.predict(z); params=x.shape[1]+1
    elif kind=='HGB':
        model=HistGradientBoostingRegressor(max_iter=80,max_depth=3,max_leaf_nodes=7,
            min_samples_leaf=5,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=SEED)
        model.fit(x,yy); pred=model.predict(z); params=None
    elif kind=='MLP':
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        model=nn.Sequential(nn.Linear(x.shape[1],16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1)).to(device)
        params=sum(p.numel() for p in model.parameters())
        assert params==16*x.shape[1]+161
        tx=torch.as_tensor(x,device=device); tz=torch.as_tensor(z,device=device)
        ty=torch.as_tensor(yy.astype('float32'),device=device)
        opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.01)
        model.train()
        for epoch in range(120):
            opt.zero_grad(set_to_none=True)
            loss=((model(tx).squeeze(-1)-ty)**2).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite MLP loss')
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pred=model(tz).squeeze(-1).cpu().numpy()
    else: raise ValueError(kind)
    return np.asarray(pred)*scale+mu,params

def utility(s,y):
    n=len(y); k=max(1,int(np.ceil(.2*n)))
    def avg(v):
        t=np.partition(v,n-k)[n-k]; high=v>t; tie=v==t
        w=high.astype(float); w[tie]=(k-high.sum())/tie.sum()
        return np.dot(w,y)/k
    denominator=avg(y)-y.mean()
    return float((avg(s)-y.mean())/denominator) if denominator>1e-12 else float('nan')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args()
    if (OUT/'RESULTS.csv').exists(): raise FileExistsError('Never overwrite a completed batch')
    torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError('Use the existing txpert cu124 environment')
    torch.use_deterministic_algorithms(True)
    start=time.time()
    audit=json.loads((OUT/'DATA_AUDIT.json').read_text())
    assert sha(SOURCE)==audit['sha256'], 'Source changed since metadata audit'
    d=allowed_frame(SOURCE,META+P+Q+H+['true_error_rmse'])
    assert len(d)==1656 and d[['dataset_name','task_key']].drop_duplicates().shape[0]==552
    assert d.predictor_name.nunique()==3 and d.dataset_name.nunique()==7
    assert np.isfinite(d.true_error_rmse).all()
    status={'status':'RUNNING','started_unix':start,'source_sha256':sha(SOURCE),
       'script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'SAFECONF_MODEL_DECISION.md'),
       'source_fold':0,'original_split':'val','final_test_label_rows_used':0,
       'candidate_error_memory_status':'NOT_RUN_PROVENANCE','new_upstream_training_runs':0,
       'n_unique_tasks':552,'n_prediction_rows':1656,'seed':SEED,'hyperparameter_trials_per_method':1,
       'device':args.device,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
       'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
       'python':platform.python_version(),'feature_groups':GROUPS,'evidence_level':'RETROSPECTIVE_VALIDATION_DEVELOPMENT_ONLY',
       'upstream_artifact_lineage':'INCOMPLETE','outer_split':'GroupKFold(3), perturbation within dataset/predictor'}
    (OUT/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    rows=[]; predictions=[]; split_rows=[]; fits=0
    for (ds,predictor),frame in d.groupby(['dataset_name','predictor_name'],sort=True):
        frame=frame.sort_values(['perturbation','context','task_key']).reset_index(drop=True)
        for fold,(ia,ib) in enumerate(GroupKFold(3).split(frame,groups=frame.perturbation)):
            fit=frame.iloc[ia].reset_index(drop=True); query=frame.iloc[ib].reset_index(drop=True)
            assert not set(fit.perturbation)&set(query.perturbation)
            assert not set(fit.task_key)&set(query.task_key)
            y=fit.true_error_rmse.to_numpy(float); truth=query.true_error_rmse.to_numpy(float)
            for role,g in [('fit',fit),('eval',query)]:
                for r in g.itertuples(): split_rows.append(dict(dataset=ds,predictor=predictor,fold=fold,task_key=r.task_key,perturbation=r.perturbation,role=role))
            fit_digest=hashlib.sha256('\n'.join(sorted(fit.task_key)).encode()).hexdigest()
            eval_digest=hashlib.sha256('\n'.join(sorted(query.task_key)).encode()).hexdigest()
            def record(name,s,params=None,seconds=0.,regression=True):
                s=np.asarray(s,float)
                if regression: s=np.maximum(0,s)
                assert len(s)==len(truth) and np.isfinite(s).all()
                rho=float(spearmanr(s,truth).statistic) if np.ptp(s)>0 and np.ptp(truth)>0 else np.nan
                rows.append(dict(dataset=ds,type='chemical' if ds in CHEM else 'gene',predictor=predictor,fold=fold,
                    method=name,n_fit=len(fit),n_eval=len(query),utility20=utility(s,truth),spearman=rho,
                    risk_rmse=float(np.sqrt(np.mean((s-truth)**2))) if regression else np.nan,
                    parameters=params,seconds=seconds,fit_task_hash=fit_digest,eval_task_hash=eval_digest))
                for key,e,ss in zip(query.task_key,truth,s):
                    predictions.append(dict(dataset=ds,predictor=predictor,fold=fold,task_key=key,method=name,true_error_rmse=e,predicted_risk=ss))
            record('Magnitude_raw',query.prediction_l2_norm.to_numpy(float),regression=False)
            base=anchor_predict(fit,query); record('Magnitude_anchor',base,2)
            for group,cols in GROUPS.items():
                x,z=features(fit,query,cols)
                for kind in ['Ridge','HGB','MLP']:
                    t=time.time(); pred,params=learn(x,z,y,kind,args.device); fits+=1
                    record(kind+'_'+group,pred,params,time.time()-t)
            residual=y-anchor_oof(fit)
            x,z=features(fit,query,P+Q+H)
            for kind in ['Ridge','HGB','MLP']:
                t=time.time(); correction,params=learn(x,z,residual,kind,args.device); fits+=1
                record(kind+'_anchored_PQH',base+correction,params+2 if params is not None else None,time.time()-t)
        print(f'{ds} / {predictor}: completed; fits={fits}; elapsed={time.time()-start:.1f}s',flush=True)
        pd.DataFrame(rows).to_csv(OUT/'PARTIAL_RESULTS.csv',index=False)
    result=pd.DataFrame(rows)
    assert len(result)==1071 and fits==945
    for _,g in result.groupby(['dataset','predictor','fold']):
        assert g.method.nunique()==17 and g.eval_task_hash.nunique()==1 and g.fit_task_hash.nunique()==1
    result.to_csv(OUT/'RESULTS.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'PREDICTIONS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(split_rows).to_csv(OUT/'SPLIT_MANIFEST.csv',index=False)
    unit=result.groupby(['dataset','type','predictor','method'],as_index=False)[['utility20','spearman','risk_rmse']].mean()
    unit.to_csv(OUT/'DATASET_PREDICTOR_RESULTS.csv',index=False)
    dataset=unit.groupby(['dataset','type','method'],as_index=False)[['utility20','spearman','risk_rmse']].mean()
    dataset.to_csv(OUT/'DATASET_RESULTS.csv',index=False)
    dataset.groupby('method')[['utility20','spearman','risk_rmse']].mean().to_csv(OUT/'MACRO.csv')
    dataset.groupby(['type','method'])[['utility20','spearman','risk_rmse']].mean().to_csv(OUT/'DOMAIN_MACRO.csv')
    wide=dataset.pivot(index=['dataset','type'],columns='method',values='utility20')
    comp=[]
    for kind in ['Ridge','HGB','MLP']:
        for name,a,b in [('P_given_M',kind+'_P',kind+'_M'),('Q_given_P',kind+'_PQ',kind+'_P'),('H_given_PQ',kind+'_PQH',kind+'_PQ'),('anchor_given_same_info',kind+'_anchored_PQH',kind+'_PQH'),('PQH_vs_magnitude',kind+'_PQH','Magnitude_raw')]:
            delta=wide[a]-wide[b]
            comp.append(dict(comparison=kind+':'+name,delta_u20=delta.mean(),positive_dataset_groups=int((delta>0).sum()),total_dataset_groups=len(delta),gene_delta=delta.xs('gene',level='type').mean(),chemical_delta=delta.xs('chemical',level='type').mean()))
    for group in GROUPS:
        for kind in ['HGB','MLP']:
            delta=wide[kind+'_'+group]-wide['Ridge_'+group]
            comp.append(dict(comparison=group+':'+kind+'_vs_Ridge',delta_u20=delta.mean(),positive_dataset_groups=int((delta>0).sum()),total_dataset_groups=len(delta),gene_delta=delta.xs('gene',level='type').mean(),chemical_delta=delta.xs('chemical',level='type').mean()))
    pd.DataFrame(comp).to_csv(OUT/'PAIRED_COMPARISONS.csv',index=False)
    (OUT/'PARTIAL_RESULTS.csv').unlink()
    status.update(status='COMPLETE_STOPPED_AFTER_FIRST_BATCH',completed_unix=time.time(),elapsed_seconds=time.time()-start,
        n_micro_model_fits=fits,n_metric_rows=len(result),n_outer_cells=63,n_methods=17,
        all_same_outer_tasks=True,outer_perturbation_overlap=0,inner_anchor_perturbation_overlap=0,
        post_batch_jobs_started=0,result_sha256=sha(OUT/'RESULTS.csv'))
    (OUT/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(pd.read_csv(OUT/'MACRO.csv').to_string(index=False),flush=True)

if __name__=='__main__': main()
