#!/usr/bin/env python3
"""One predeclared native-similarity repair, on the existing feedback DEV pool.

Does not read or score the 212-task evaluation pool. No model selection on
holdout. All error/CDF/development uses are counted as the full 331-task pool.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import PUBLIC,cluster_weights,rank_labels,metrics,ids_hash
from tools.scripts.run_safeconf_pertema_current_truth_v1 import POOL,NATIVE,NATIVE_FEATURES,SEEDS
from tools.scripts.run_safeconf_official_pertema_feedback import factory

OUT=Path('/home/yyf/runtime_artifacts/safeconf_goal_report_20261008_v1/native_similarity_dev')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'STATUS.json').exists():raise FileExistsError('versioned diagnostic is immutable')
    start=time.monotonic();cpu=time.process_time()
    pool=pd.read_parquet(POOL)
    native=pd.read_parquet(NATIVE).set_index('task_id')
    pool=pool.join(native[NATIVE_FEATURES],on='task_id',rsuffix='_native')
    pool['native_prediction_abs_mean']=pool.prediction_abs_mean.to_numpy(float)
    assert len(pool)==331 and pool.gene.nunique()==228 and pool.task_id.is_unique
    gbt,official_hash=factory()
    (OUT/'EXPERIMENT_CARD.json').write_text(json.dumps({
        'hypothesis':'self-prototype similarity near zero in training creates a train/query representation mismatch',
        'repair':'nearest embedding of another biological gene for training; query uses training prototypes only',
        'fixed':'all other features, labels, gene weights, official tree recipe, group folds and seeds unchanged',
        'scope':'DEV on existing 331-task feedback pool; no 212 evaluation rows read or scored',
        'comparison':'two policies x Native61/Native61+Public x 3 gene folds x 3 fixed seeds',
        'new_fits':36,'no_hyperparameter_search':True,'holdout_score':False,
        'target_error_union_rows':331,'target_error_union_genes':228,
        'public_model_changed':False,'shared_risk_used':False,'official_factory_hash':official_hash},ensure_ascii=False,indent=2)+'\n')
    (OUT/'STATUS.json').write_text(json.dumps({'status':'RUNNING','pid':os.getpid()})+'\n')
    parts=[];audits=[];strata=[]
    for fold,(it,iv) in enumerate(GroupKFold(3).split(pool,groups=pool.gene)):
        tr=pool.iloc[it].reset_index(drop=True);va=pool.iloc[iv].reset_index(drop=True)
        assert not set(tr.gene)&set(va.gene)
        labels,cdf=rank_labels(tr,f'native-similarity-dev/inner{fold}',1.0)
        for name,cols in [('Native61',NATIVE_FEATURES),('Native61_Public',NATIVE_FEATURES+PUBLIC)]:
            ei=[i for i,c in enumerate(cols) if c.startswith('native_embedding_')]
            si=cols.index('native_training_similarity')
            xt=tr[cols].to_numpy(float);xv=va[cols].to_numpy(float)
            et=xt[:,ei];ev=xv[:,ei];good_t=np.isfinite(et).all(1);good_v=np.isfinite(ev).all(1)
            proto=et[good_t];owner=tr.gene.astype(str).to_numpy()[good_t]
            nn=NearestNeighbors(n_neighbors=1,algorithm='brute',n_jobs=4).fit(np.unique(proto,axis=0))
            xv[:,si]=np.nan
            xv[good_v,si]=nn.kneighbors(ev[good_v])[0].ravel()
            for policy in ['self_prototype_original','other_gene_once']:
                trainx=xt.copy();trainx[:,si]=np.nan
                if policy=='self_prototype_original':
                    trainx[good_t,si]=nn.kneighbors(et[good_t])[0].ravel()
                else:
                    dist=pairwise_distances(et[good_t],proto)
                    own=tr.gene.astype(str).to_numpy()[good_t]
                    dist[own[:,None]==owner[None,:]]=np.inf
                    mins=dist.min(axis=1);mins[~np.isfinite(mins)]=np.nan
                    trainx[good_t,si]=mins
                for seed in SEEDS:
                    valid=np.isfinite(labels)
                    model=gbt().set_params(n_jobs=4,random_state=seed)
                    model.fit(trainx[valid],labels[valid],sample_weight=cluster_weights(tr.loc[valid]))
                    risk=model.predict(xv)
                    assert np.isfinite(risk).all()
                    part=va[['task_id','target','gene','true_error_rmse']].copy()
                    part['fold']=fold;part['seed']=seed;part['feature_set']=name;part['policy']=policy;part['risk']=risk
                    parts.append(part)
                    for target,q in part.groupby('target'):
                        strata.append({'fold':fold,'seed':seed,'feature_set':name,'policy':policy,'target':target,**metrics(q,q.risk.to_numpy())})
                    audits.append({'fold':fold,'seed':seed,'feature_set':name,'policy':policy,
                        'fit_rows':len(tr),'fit_genes':tr.gene.nunique(),'query_rows':len(va),'query_genes':va.gene.nunique(),
                        'fit_id_hash':ids_hash(tr.task_id),'query_id_hash':ids_hash(va.task_id),
                        'train_similarity_min':float(np.nanmin(trainx[:,si])),
                        'train_similarity_median':float(np.nanmedian(trainx[:,si])),
                        'query_similarity_min':float(np.nanmin(xv[:,si])),
                        'query_similarity_median':float(np.nanmedian(xv[:,si])),'cdf_groups':len(cdf)})
    predictions=pd.concat(parts,ignore_index=True);s=pd.DataFrame(strata)
    predictions.to_parquet(OUT/'OOF_PREDICTIONS.parquet',index=False)
    s.to_csv(OUT/'STRATA.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'FIT_AND_FEATURE_AUDIT.csv',index=False)
    # Keep outer gene-fold x context metrics separate, then equal-weight them.
    result=s.groupby(['feature_set','policy','seed'],as_index=False)[['utility20','aurc','spearman']].mean()
    result.to_csv(OUT/'SEED_RESULTS.csv',index=False)
    summary=result.groupby(['feature_set','policy'],as_index=False)[['utility20','aurc','spearman']].mean()
    summary.to_csv(OUT/'SUMMARY.csv',index=False)
    status={'status':'COMPLETE','fits':len(audits),'wall_seconds':time.monotonic()-start,
        'cpu_seconds':time.process_time()-cpu,'new_gpu_hours':0,'new_download_bytes':0,
        'permanent_test_truth_opened':False,'evaluation_212_read':False,'default_changed':False,
        'target_errors_used':331,'target_gene_clusters':228,'role':'DEV diagnostic; no automatic adoption',
        'pool_sha256':hashlib.sha256(POOL.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(summary.to_string(index=False));print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':main()
