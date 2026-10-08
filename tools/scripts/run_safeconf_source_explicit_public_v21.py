#!/usr/bin/env python3
"""One fixed Source representation repair; development uses source errors only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import P,PUBLIC,rank_labels,shuffled_labels,fit_risk
from tools.safeconf_continual.submission_evidence import ClusterBootstrap,point_metrics,summarize_draws,write_json,digest_ids
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN,SAMS,POOL,HOLD,COMMON


def main():
    out=RUN/'source_explicit_public';out.mkdir(parents=True,exist_ok=True)
    if (out/'STATUS.json').exists() and json.loads((out/'STATUS.json').read_text()).get('status')=='COMPLETE':return
    start,cpu=time.monotonic(),time.process_time()
    pool_ids=pd.read_parquet(POOL,columns=['task_id']).task_id
    memory=pd.read_parquet(COMMON/'public_mcfaline_trainval/public_memory.parquet',columns=['experiment_id'])
    if memory.experiment_id.isin('McFaline23::'+pool_ids).any():
        raise RuntimeError('query truth found in cached Source Public bank; nested regeneration required')
    columns={'P':P,'P_PUBLIC':P+PUBLIC,'P_PUBLIC_R':P+PUBLIC+['simple_history_risk']}
    seeds=[None,20260930,20261001,20261002,20261003,20261004]
    records=[];fit_ledger=[];selection=[];pairs=[];total_fits=0
    for source,target in [('DecoderOnly','SAMS_VAE'),('SAMS_VAE','DecoderOnly')]:
        full=pd.read_parquet(SAMS/f'{source}_RISK_FEATURES.parquet').set_index('task_id')
        f=full.loc[pool_ids].reset_index()
        assert len(f)==331 and f.gene.nunique()==228
        split=list(GroupKFold(3).split(f,groups=f.gene));fold=np.empty(len(f),int)
        for k,(_,va) in enumerate(split):fold[va]=k
        eval_frame=f[['task_id','gene','target','true_error_rmse']].copy()
        eval_frame['target']=f.target.astype(str)+'::fold'+pd.Series(fold).astype(str)
        engine=ClusterBootstrap(eval_frame,f.true_error_rmse.to_numpy(float))
        public=f.simple_history_risk.to_numpy(float)
        groups=eval_frame.groupby('target').indices
        def summarize(s):
            strata=[point_metrics(f.true_error_rmse.to_numpy()[ix],s[ix],f.task_id.to_numpy()[ix]) for ix in groups.values()]
            return {c:float(np.nanmean([r[c] for r in strata])) for c in ['utility','aurc','high_risk_miss_rate']}
        reference=summarize(public)
        real_scores=None;adopted='always_public';selected_scores=[]
        for shuffle_seed in seeds:
            output={name:np.full(len(f),np.nan) for name in columns}
            for outer,(tr,va) in enumerate(split):
                train=f.iloc[tr].reset_index(drop=True);query=f.iloc[va].reset_index(drop=True)
                labels,cdf=rank_labels(train,f'v21/source/{source}/fold{outer}',1.)
                shuffle_audit={'moved_clusters':0}
                if shuffle_seed is not None:labels,shuffle_audit=shuffled_labels(train,labels,shuffle_seed+outer)
                for name,cols in columns.items():
                    model=fit_risk(train,labels,cols,'hgb')
                    output[name][va]=model.predict(query,clip=False);total_fits+=1
                    if total_fits>120:raise RuntimeError('Source fit allowance exceeded')
                    fit_ledger.append({'source':source,'target':target,'fold':outer,'input':name,
                        'shuffle_seed':shuffle_seed,'train_rows':len(train),'train_genes':train.gene.nunique(),
                        'train_hash':digest_ids(train.task_id),'evaluation_rows':len(query),
                        'evaluation_gene_overlap':len(set(train.gene)&set(query.gene)),
                        'target_errors_used_for_selection':0,'cdf_groups':len(cdf),**shuffle_audit})
            results={name:summarize(s) for name,s in output.items()}
            best=max(results,key=lambda n:(results[n]['utility'],n=='P_PUBLIC',n=='P'))
            d=engine.difference(output[best],public)
            comparison=summarize_draws(d,results[best]['utility']-reference['utility'])
            # Safety checks compare fixed context x gene-fold strata.
            nonnegative=np.mean([point_metrics(f.true_error_rmse.to_numpy()[ix],output[best][ix],f.task_id.to_numpy()[ix])['utility']>=
                                 point_metrics(f.true_error_rmse.to_numpy()[ix],public[ix],f.task_id.to_numpy()[ix])['utility'] for ix in groups.values()])
            passed=(comparison['delta_point']>=.005 and comparison.get('ci95_lower',-np.inf)>=-.005 and nonnegative>=.6
                    and results[best]['aurc']<=1.05*reference['aurc']
                    and results[best]['high_risk_miss_rate']<=reference['high_risk_miss_rate']+.02)
            winner=best if passed else 'always_public'
            selected_scores.append(output[winner] if winner!='always_public' else public)
            selection.append({'source':source,'target':target,'shuffle_seed':shuffle_seed,'best_source_input':best,
                'selected_complete_system':winner,'nonnegative_strata_fraction':float(nonnegative),**comparison})
            for name,s in output.items():
                part=f[['task_id','gene','target','true_error_rmse']].copy();part['source']=source;part['target_predictor']=target
                part['shuffle_seed']=-1 if shuffle_seed is None else shuffle_seed;part['input']=name;part['fold']=fold;part['risk']=s
                records.append(part)
            if shuffle_seed is None:real_scores=output;adopted=winner
            print(json.dumps({'source':source,'target':target,'shuffle_seed':shuffle_seed,'selected':winner,
                              'delta':comparison['delta_point'],'fits':total_fits}),flush=True)
        real_vs_null=np.mean([engine.difference(selected_scores[0],s) for s in selected_scores[1:]],axis=0)
        real_null_point=summarize(selected_scores[0])['utility']-np.mean([summarize(s)['utility'] for s in selected_scores[1:]])
        evidence=summarize_draws(real_vs_null,real_null_point)
        pairs.append({'source':source,'target':target,'comparison':'selected_real-minus-selected_null_mean',**evidence})
        if adopted!='always_public' and (evidence['delta_point']<=0 or evidence.get('ci95_lower',-np.inf)<-.005):
            adopted='always_public'
        # A representation benefit is reported separately from a Public upgrade.
        for a,b in [('P_PUBLIC_R','P_PUBLIC'),('P_PUBLIC_R','P')]:
            pairs.append({'source':source,'target':target,'comparison':a+'-minus-'+b,
                **summarize_draws(engine.difference(real_scores[a],real_scores[b]),summarize(real_scores[a])['utility']-summarize(real_scores[b])['utility'])})
        if adopted!='always_public':
            target_frame=pd.read_parquet(SAMS/f'{target}_RISK_FEATURES.parquet').set_index('task_id')
            ids=pd.read_parquet(HOLD,columns=['task_id']).task_id
            query=target_frame.loc[ids].reset_index()
            labels,_=rank_labels(f,f'v21/source/{source}/final',1.)
            model=fit_risk(f,labels,columns[adopted],'hgb');total_fits+=1
            score=model.predict(query,clip=False)
            query[['task_id','gene','target','true_error_rmse']].assign(risk=score).to_parquet(out/f'{source}_to_{target}_FIXED_EVALUATION.parquet',index=False)
    pd.concat(records,ignore_index=True).to_parquet(out/'SOURCE_DEV_OOF.parquet',index=False)
    pd.DataFrame(fit_ledger).to_csv(out/'SOURCE_FIT_AND_CDF_LEDGER.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'COMPLETE_SELECTION_REAL_AND_SHUFFLED.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'REPRESENTATION_PAIRED_BOOTSTRAP.csv',index=False)
    write_json(out/'STATUS.json',{'status':'COMPLETE','new_fits':total_fits,'fit_cap':120,
        'pool_rows_per_source':331,'pool_genes_per_source':228,'target_selection_errors':0,
        'public_cache_query_experiment_intersection':0,'wall_seconds':time.monotonic()-start,
        'cpu_seconds':time.process_time()-cpu,'feature_engineering_finished':True,
        'new_gpu_hours':0,'decisions':[s for s in selection if s['shuffle_seed'] is None]})


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
