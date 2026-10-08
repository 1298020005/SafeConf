#!/usr/bin/env python3
"""Two SEEN 24-task panels: reuse frozen predictions, other-study history.

Primary errors retain the original 512-gene axis. Public features use the
505-gene common axis. No target errors train or choose the ranking rules.
Whole-cohort score scales use biological training proxies, explicitly charged
as preparation data, not mislabeled as out-of-sample predictor scores.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,time
from pathlib import Path
import h5py,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.submission_evidence import (write_json,sha,digest_ids,h5_column,
    endpoint_arrays,point_metrics,ClusterBootstrap,summarize_draws,REVIEWS)
RUN=Path('/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21')
OUT=RUN/'adamson_replication'
GWPS=Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ReplogleWeissman2022_K562_gwps.h5ad')
ADAM=Path('/home/yyf/data/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad')
PANELS={
 'panel1':Path('/home/yyf/proj/docs/实验结果/E65_scgpt_formal_fixed_panel_20260711'),
 'panel2':Path('/home/yyf/proj/docs/实验结果/E76a_adamson_scgpt_panel2_20260711')}


def load_panels():
    panels={}
    for name,path in PANELS.items():
        frame=pd.read_csv(path/'tables/PREDICTION_RECORDS.csv')
        gene_table=pd.read_csv(path/'tables/E65_GENE_PANEL.csv').sort_values('panel_index')
        split=pd.read_csv(path/'tables/E65_FIXED_SPLIT.csv')
        train=split.loc[split.split.eq('train'),'condition'].astype(str).tolist()
        frame['gene']=frame.perturbation.str.replace('+ctrl','',regex=False)
        frame['predictor']=np.where(frame.predictor_name.str.startswith('GEARS'),'GEARS','scGPT')
        if frame.gene.nunique()!=24 or len(gene_table)!=512:raise RuntimeError('panel identity changed')
        if set(train)&set(frame.perturbation):raise RuntimeError('upstream train/test identity overlap')
        panels[name]={'path':path,'frame':frame,'genes':gene_table.gene.astype(str).tolist(),'train':train}
    return panels


def prepare():
    if (OUT/'PUBLIC_CELL_MEAN_EFFECTS.npz').exists():return
    panels=load_panels()
    requested=set()
    for p in panels.values():
        requested.update(p['frame'].gene);requested.update(s.replace('+ctrl','') for s in p['train'])
    with h5py.File(GWPS,'r') as h:
        symbols=h5_column(h['var'],'gene_name').astype(str);ids=h5_column(h['var'],'ensembl_id').astype(str)
        labels=h5_column(h['obs'],'gene').astype(str);nperts=h5_column(h['obs'],'nperts').astype(str)
        target_symbols=sorted(requested&set(labels));source_index={g:i for i,g in enumerate(symbols)}
        axis=sorted(set().union(*(set(p['genes']) for p in panels.values()))&set(symbols))
        cols=np.asarray([source_index[g] for g in axis]);groups={g:i for i,g in enumerate(target_symbols)}
        codes=np.asarray([groups.get(g,-1) for g in labels],int);codes[nperts=='0']=len(groups)
        sums=np.zeros((len(groups)+1,len(cols)),float);counts=np.zeros(len(groups)+1,np.int64)
        receipt={'role':'SEEN_CROSS_STUDY_REPLICATION','panels':{},'training_error_labels_used':0,
            'source_study':'ReplogleWeissman2022_K562_GWPS','target_study_in_public':False,
            'source_normalization':'per-cell log1p(CP10k using full stored 8248-gene count library), then cell mean minus controls',
            'target_normalization':'frozen Adamson processed expression cell mean minus controls; unchanged',
            'denominator_difference':'stored source gene universe differs from original Adamson preprocessing; cross-study scale caveat retained',
            'query_truth_used_for_score_calibration':False,'score_scale_preparation':'Adamson upstream-training biological effects used as proxy predictions; errors never read',
            'source_file':str(GWPS),'source_bytes':GWPS.stat().st_size,'output_axis_hash':digest_ids(axis)}
        for name,p in panels.items():
            receipt['panels'][name]={'tasks':24,'predictors':['GEARS','scGPT'],'original_error_axis':512,
                'public_common_axis':len(set(p['genes'])&set(symbols)),'history_targets':len(set(p['frame'].gene)&set(labels)),
                'upstream_training_biological_tasks_for_score_scale':len(p['train']),
                'prediction_sha256':sha(p['path']/'arrays/predicted_effects.npz'),
                'truth_sha256':sha(p['path']/'arrays/true_effects.npz'),'status':'SEEN_REPLICATION_NOT_NEW_CONFIRMATION'}
        write_json(OUT/'EXPERIMENT_FREEZE.json',receipt)
        pd.DataFrame({'gene':axis,'ensembl_id':[ids[source_index[g]] for g in axis]}).to_csv(OUT/'COMMON_AXIS.csv',index=False)
        for begin in range(0,len(labels),4000):
            end=min(begin+4000,len(labels));mask=codes[begin:end]>=0
            if not mask.any():continue
            full=np.asarray(h['X'][begin:end,:],np.float64)[mask]
            lib=full.sum(1)
            if np.any(lib<=0):raise RuntimeError('nonpositive library in source')
            selected=np.log1p(full[:,cols]*(1e4/lib)[:,None]);gr=codes[begin:end][mask]
            order=np.argsort(gr,kind='stable');u,start=np.unique(gr[order],return_index=True)
            sums[u]+=np.add.reduceat(selected[order],start,axis=0)
            counts+=np.bincount(gr,minlength=len(groups)+1)
            if end%80000==0:print(json.dumps({'source_cells_scanned':end,'total':len(labels)}),flush=True)
    if np.any(counts==0):raise RuntimeError('empty prepared source group')
    effects=sums[:-1]/counts[:-1,None]-sums[-1]/counts[-1]
    np.savez(OUT/'PUBLIC_CELL_MEAN_EFFECTS.npz',targets=target_symbols,genes=axis,effects=effects.astype(np.float32),n_cells=counts[:-1])
    write_json(OUT/'PREPARE_STATUS.json',{'status':'COMPLETE','n_source_targets':len(target_symbols),'gpu_hours':0,'new_download_bytes':0})


def training_proxy(panel,axis):
    """Only upstream training conditions and controls; evaluation rows not read."""
    with h5py.File(ADAM,'r') as h:
        conditions=h5_column(h['obs'],'condition').astype(str)
        var_name=h['var'].attrs.get('_index','_index');var_name=var_name.decode() if isinstance(var_name,bytes) else var_name
        symbols=h5_column(h['var'],var_name).astype(str);colidx={g:i for i,g in enumerate(symbols)}
        cols=np.asarray([colidx[g] for g in axis]);conditions_order=list(dict.fromkeys(panel['train']+['ctrl']))
        index={c:i for i,c in enumerate(conditions_order)}
        sums=np.zeros((len(index),len(axis)),float);count=np.zeros(len(index),int)
        matrix=h['X'];pointer=matrix['indptr'][:]
        for row in np.flatnonzero(np.isin(conditions,list(index))):
            begin,end=int(pointer[row]),int(pointer[row+1]);ci=matrix['indices'][begin:end];x=matrix['data'][begin:end]
            dense=np.zeros(len(symbols),float);np.add.at(dense,ci,x)
            k=index[conditions[row]];sums[k]+=dense[cols];count[k]+=1
        if np.any(count==0):raise RuntimeError('empty training/control condition')
    means=sums/count[:,None]
    return means[[index[c] for c in panel['train']]]-means[index['ctrl']]


def cdf(a,x):
    v=np.sort(np.asarray(a,float));v=v[np.isfinite(v)]
    if len(v)<2:raise RuntimeError('insufficient training-only scale observations')
    return (np.searchsorted(v,x,'left')+np.searchsorted(v,x,'right'))/(2*len(v))


def score():
    panels=load_panels();bank=np.load(OUT/'PUBLIC_CELL_MEAN_EFFECTS.npz');si={g:i for i,g in enumerate(bank['targets'])}
    ci={g:i for i,g in enumerate(bank['genes'])};rows=[];metric=[];paired=[];arrays={}
    for name,panel in panels.items():
        panel_columns=np.asarray([i for i,g in enumerate(panel['genes']) if g in ci]);axis=[panel['genes'][i] for i in panel_columns]
        bank_columns=np.asarray([ci[g] for g in axis]);proxy=training_proxy(panel,axis)
        tg=[g.replace('+ctrl','') for g in panel['train']];have=np.asarray([g in si for g in tg])
        history=np.stack([bank['effects'][si[g],bank_columns] for g in np.asarray(tg)[have]])
        proxy_amp=np.sqrt(np.mean(proxy*proxy,1));proxy_public=np.sqrt(np.mean((proxy[have]-history)**2,1))
        proxy_support=-np.log1p(bank['n_cells'][[si[g] for g in np.asarray(tg)[have]]])
        np.savez(OUT/f'{name}_TRAINING_PROXY_CDFS.npz',amplitude=np.sort(proxy_amp),public=np.sort(proxy_public),support=np.sort(proxy_support))
        predz=np.load(panel['path']/'arrays/predicted_effects.npz');truez=np.load(panel['path']/'arrays/true_effects.npz')
        for predictor,q in panel['frame'].groupby('predictor',sort=True):
            q=q.sort_values('gene').reset_index(drop=True);pred=np.stack([predz[k] for k in q.predicted_effect_key]);truth=np.stack([truez[k] for k in q.true_effect_key])
            endpoints=endpoint_arrays(pred,truth);np.testing.assert_allclose(endpoints['delta_rmse'],q.true_error_rmse,rtol=1e-5,atol=1e-7)
            common_endpoints=endpoint_arrays(pred[:,panel_columns],truth[:,panel_columns]);available=np.asarray([g in si for g in q.gene])
            pp=pred[:,panel_columns];amp=np.sqrt(np.mean(pred**2,1));amp_common=np.sqrt(np.mean(pp**2,1))
            distance=np.full(len(q),np.nan);n=np.full(len(q),np.nan)
            hs=np.stack([bank['effects'][si[g],bank_columns] for g in q.loc[available,'gene']])
            distance[available]=np.sqrt(np.mean((pp[available]-hs)**2,1));n[available]=bank['n_cells'][[si[g] for g in q.loc[available,'gene']]]
            amplitude_score=cdf(proxy_amp,amp_common)
            methods={'Magnitude':amplitude_score,'HistorySupport':np.where(available,cdf(proxy_support,-np.log1p(n)),amplitude_score),
                'PublicRule':np.where(available,cdf(proxy_public,distance),amplitude_score)}
            for i,r in q.iterrows():
                rows.append({'panel':name,'predictor':predictor,'task_id':str(r.task_id),'gene':r.gene,'target':name,
                    'public_available':bool(available[i]),'evidence_status':'LEGAL_OTHER_STUDY_HISTORY' if available[i] else 'NO_HISTORY_AMPLITUDE_FALLBACK',
                    'magnitude':amp[i],'public_raw':distance[i],'n_source_cells':n[i],
                    **{e:float(v[i]) for e,v in endpoints.items()},**{e+'_common_axis':float(v[i]) for e,v in common_endpoints.items()},
                    **{m:float(v[i]) for m,v in methods.items()}})
            f=pd.DataFrame({'task_id':q.task_id.astype(str),'gene':q.gene,'target':name})
            boot=ClusterBootstrap(f,endpoints['delta_rmse'])
            for method,risk in methods.items():
                for endpoint,error in endpoints.items():
                    for review in REVIEWS:metric.append({'panel':name,'predictor':predictor,'method':method,'endpoint':endpoint,'review':review,
                        'scope':'all_24_original_axis',**point_metrics(error,risk,f.task_id,review)})
                paired.append({'panel':name,'predictor':predictor,'method':method,'reference':'Magnitude',
                    **summarize_draws(boot.difference(risk,methods['Magnitude']),point_metrics(endpoints['delta_rmse'],risk,f.task_id)['utility']-point_metrics(endpoints['delta_rmse'],methods['Magnitude'],f.task_id)['utility'])})
                metric.append({'panel':name,'predictor':predictor,'method':method,'endpoint':'delta_rmse','review':.2,
                    'scope':'covered_23_raw_rule_original_error',**point_metrics(endpoints['delta_rmse'][available],
                        {'Magnitude':amp,'HistorySupport':-np.log1p(n),'PublicRule':distance}[method][available],f.task_id[available],.2)})
    per=pd.DataFrame(rows);per.to_parquet(OUT/'PER_QUERY_RESULTS.parquet',index=False)
    pd.DataFrame(metric).to_csv(OUT/'RESULTS.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'PAIRED_CLUSTER_BOOTSTRAP.csv',index=False)
    pooled=[]
    for predictor,q in per.groupby('predictor',sort=True):
        q=q.reset_index(drop=True);b=ClusterBootstrap(q,q.delta_rmse)
        for method in ['Magnitude','HistorySupport','PublicRule']:
            values=[point_metrics(t.delta_rmse,t[method],t.task_id)['utility'] for _,t in q.groupby('panel')]
            base=[point_metrics(t.delta_rmse,t.Magnitude,t.task_id)['utility'] for _,t in q.groupby('panel')]
            pooled.append({'predictor':predictor,'method':method,'macro_u20':np.mean(values),'n_gene_clusters':q.gene.nunique(),
                **summarize_draws(b.difference(q[method],q.Magnitude),np.mean(values)-np.mean(base))})
    pd.DataFrame(pooled).to_csv(OUT/'PANEL_EQUAL_PAIRED_RESULTS.csv',index=False)
    write_json(OUT/'STATUS.json',{'status':'COMPLETE','role':'SEEN_REPLICATION','independent_gene_clusters':per.gene.nunique(),
        'prediction_rows':len(per),'risk_training_error_labels':0,'new_upstream_fits':0,'new_gpu_hours':0,
        'all_tasks_included':True,'native_truth_axis_unchanged':True,'fallback_tasks_per_panel':1,
        'score_scales_use_training_biological_proxies':True,'training_proxy_cost_is_not_zero_target_data':True})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','score','all'],default='all');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.phase in ['prepare','all']:prepare()
    if a.phase in ['score','all']:score()

if __name__=='__main__':main()
