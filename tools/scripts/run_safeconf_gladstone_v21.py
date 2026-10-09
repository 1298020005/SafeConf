#!/usr/bin/env python3
"""Role-guarded CD4 counts protocol, competence gate, and frozen confirmation.

Target expression is materialized only for explicitly allowed roles. Predictor
queries accept gene/context metadata; Public generation never accepts truth.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,math,sys,time,importlib.util,concurrent.futures,fcntl,shutil
from pathlib import Path
import h5py,numpy as np,pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.range_h5 import MeteredHTTPFile
from tools.safeconf_continual.submission_evidence import (
    write_json,sha,digest_ids,h5_column,endpoint_arrays,point_metrics,ClusterBootstrap,
    BUDGETS,ORDERS,MODEL_SEEDS,REVIEWS,summarize_draws)
from tools.safeconf_continual.research import P,PUBLIC,cluster_weights,rank_labels
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN,native_x,official_gbt,CONFIG
from tools.safeconf_continual.frozen_scoring import score_task

OUT=RUN/'external';URL='https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.pseudobulk_merged.h5ad'
GWPS=Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ReplogleWeissman2022_K562_gwps.h5ad')
CONTEXTS=['Rest','Stim8hr','Stim48hr']


def remote():
    shared=OUT/'SHARED_RESOURCE_CACHE.json'
    paths=json.loads(shared.read_text()) if shared.exists() else {}
    return MeteredHTTPFile(URL,paths.get('cache',OUT/'range_cache/pseudobulk'),resource_ledger_path())


def resource_ledger_path():
    shared=OUT/'SHARED_RESOURCE_CACHE.json'
    return Path(json.loads(shared.read_text())['ledger']) if shared.exists() else OUT/'DOWNLOAD_RESOURCE_LEDGER.json'


def metadata():
    if (OUT/'ROLE_FREEZE.json').exists():
        frozen=json.loads((OUT/'ROLE_FREEZE.json').read_text())
        if sha(OUT/'TASK_ROLES.parquet')!=frozen['task_roles_sha256'] or sha(OUT/'OBS_ROLE_REGISTRY.parquet')!=frozen['obs_roles_sha256']:
            raise RuntimeError('frozen external role registry changed')
        return
    io=remote()
    with h5py.File(io,'r') as h:
        columns=['_index','culture_condition','donor_id','guide_id','guide_type','n_cells','perturbed_gene_id','total_counts']
        obs=pd.DataFrame({c:h5_column(h['obs'],c) for c in columns})
        genes=h5_column(h['var'],'gene_ids').astype(str)
    io.close()
    if len(set(genes))!=len(genes):raise RuntimeError('duplicate CD4 output gene identifiers')
    obs=obs.rename(columns={'_index':'observation_id','culture_condition':'context','perturbed_gene_id':'gene'})
    obs['row']=np.arange(len(obs));obs.gene=obs.gene.astype(str);obs.context=obs.context.astype(str)
    target=obs.guide_type.astype(str).eq('targeting') & (obs.n_cells>0) & (obs.total_counts>0)
    counts=obs.loc[target].groupby(['gene','context']).n_cells.sum().unstack().reindex(columns=CONTEXTS)
    eligible=counts.index[(counts>=20).all(axis=1)].astype(str).tolist()
    eligible=sorted(eligible,key=lambda g:hashlib.sha256(f'SafeConf-Gladstone-v21-cohort|{g}'.encode()).hexdigest())[:1500]
    if len(eligible)<600:raise RuntimeError('insufficient metadata-qualified genes for >=150 confirmation clusters')
    ordered=sorted(eligible,key=lambda g:hashlib.sha256(f'SafeConf-Gladstone-v21-split|{g}'.encode()).hexdigest())
    a,b=math.floor(.5*len(ordered)),math.floor(.75*len(ordered))
    roles={g:'predictor_train' if i<a else 'feedback' if i<b else 'confirmation' for i,g in enumerate(ordered)}
    tasks=pd.DataFrame([{'gene':g,'context':c,'target':c,'task_id':g+'::'+c,'role':roles[g],
        'n_cells':int(counts.loc[g,c])} for g in ordered for c in CONTEXTS])
    obs['role']=obs.gene.map(roles)
    # NTCs are permitted prediction-time material; select by IDs, not expression/QC effects.
    ntc=obs[obs.guide_type.astype(str).eq('non-targeting') & (obs.n_cells>0)&(obs.total_counts>0)]
    controls=[]
    for _,q in ntc.groupby(['donor_id','context'],sort=True):
        ordered_rows=sorted(q.index,key=lambda i:hashlib.sha256(str(q.loc[i,'observation_id']).encode()).hexdigest())[:64]
        controls.extend(ordered_rows)
    if len(controls)<100:raise RuntimeError('insufficient independent control pseudobulks for fixed embedding')
    obs['control_selected']=obs.index.isin(controls)
    with h5py.File(GWPS,'r') as h:source_genes=h5_column(h['var'],'ensembl_id').astype(str)
    common=sorted(set(genes)&set(source_genes))
    if len(common)<500:raise RuntimeError('common output axis too small')
    obs.to_parquet(OUT/'OBS_ROLE_REGISTRY.parquet',index=False)
    tasks.to_parquet(OUT/'TASK_ROLES.parquet',index=False)
    write_json(OUT/'OUTPUT_CONTRACT.json',{'gene_ids':common,'gene_axis_hash':digest_ids(common),
        'effect_definition':'log1p(CP10k pooled counts on fixed common axis) minus matched-context NTC',
        'aggregation':'pool counts before normalization, identical for both studies',
        'missing_genes_zero_filled':False,'target_study_excluded_from_public':True,
        'source_study':'ReplogleWeissman2022_K562_gwps','target_study':'Gladstone_Marson_CD4',
        'n_output_genes':len(common),'native_DESeq2_truth_not_substituted_for_counts_truth':True})
    write_json(OUT/'ROLE_FREEZE.json',{'status':'FROZEN_BEFORE_TARGET_NUMERIC_READ',
        'task_roles_sha256':sha(OUT/'TASK_ROLES.parquet'),'obs_roles_sha256':sha(OUT/'OBS_ROLE_REGISTRY.parquet'),
        'cohort_genes':len(ordered),'confirmation_genes':len(ordered)-b,
        'control_pseudobulks':len(controls),'selection_uses_error_values':False,
        'planned_contexts':CONTEXTS,'target_numeric_values_read':False,
        'source_gene_axis_sha256':digest_ids(source_genes),'target_gene_axis_sha256':digest_ids(genes)})
    print(json.dumps({'phase':'metadata_frozen','genes':len(ordered),'confirmation_genes':len(ordered)-b,
                      'common_genes':len(common),'controls':len(controls)}),flush=True)


def sparse_row(group,pointer,row,ncols):
    start,end=int(pointer[row]),int(pointer[row+1])
    indices=group['indices'][start:end];values=group['data'][start:end]
    result=np.zeros(ncols,np.float64);np.add.at(result,indices,values)
    return result


def prefetch_allowed_storage(io,group,pointer,rows,roles):
    """Inspect HDF5 chunk offsets, then fetch opaque blocks concurrently.

No expression array is returned here. The subsequent logical row selection
remains exactly the same as the frozen role registry.
"""
    blocks=set()
    for name in ['data','indices']:
        dataset=group[name]
        if not dataset.chunks:raise RuntimeError('expected chunked official CSR storage')
        step=dataset.chunks[0];chunks=set()
        for row in rows:
            begin,end=int(pointer[row]),int(pointer[row+1])
            if end>begin:chunks.update(range(begin//step,(end-1)//step+1))
        def accept(info):
            chunk=int(info.chunk_offset[0])//step
            if chunk not in chunks or info.byte_offset is None:return
            begin,end=info.byte_offset,info.byte_offset+info.size
            blocks.update(range(begin//io.block,(end-1)//io.block+1))
        if hasattr(dataset.id,'chunk_iter'):
            dataset.id.chunk_iter(accept)
        else:
            for n,chunk in enumerate(sorted(chunks)):
                accept(dataset.id.get_chunk_info_by_coord((chunk*step,)))
                if n%1000==0:print(json.dumps({'chunk_offsets':n,'total':len(chunks),'dataset':name}),flush=True)
    missing=[i for i in sorted(blocks) if not (io.cache/f'{i:08d}.bin').exists()]
    runs=[]
    for number in missing:
        if runs and number==runs[-1][-1]+1 and len(runs[-1])<8:runs[-1].append(number)
        else:runs.append([number])
    write_json(OUT/'PREFETCH_ALLOWED_STORAGE_PLAN.json',{'roles':roles,'logical_rows_hash':digest_ids(rows),
        'logical_rows':len(rows),'opaque_blocks':len(blocks),'missing_blocks':len(missing),
        'maximum_incremental_payload_bytes':len(missing)*io.block,'workers':4,'http_requests':len(runs),'max_request_blocks':8,
        'numeric_confirmation_rows_materialized':0})
    print(json.dumps({'prefetch_blocks':len(missing),'max_payload_GB':len(missing)*io.block/1e9,'roles':roles}),flush=True)
    def fetch_block(run):
        io.prefetch_blocks(run)
        return None  # Futures must not retain the entire multi-GB payload.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures={executor.submit(fetch_block,run):len(run) for run in runs}
        completed=0
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            future.result()
            completed+=futures[future]
            if n%100==0:print(json.dumps({'opaque_blocks_prefetched':completed,'total':len(missing),'requests_done':n,'requests_planned':len(runs)}),flush=True)


def prepare_target(roles):
    tasks=pd.read_parquet(OUT/'TASK_ROLES.parquet')
    obs=pd.read_parquet(OUT/'OBS_ROLE_REGISTRY.parquet');axis=json.loads((OUT/'OUTPUT_CONTRACT.json').read_text())['gene_ids']
    selected=tasks[tasks.role.isin(roles)].reset_index(drop=True)
    key_to_index={t:i for i,t in enumerate(selected.task_id)}
    allowed=obs.role.isin(roles)&obs.guide_type.astype(str).eq('targeting')
    rows=obs.index[allowed|obs.control_selected].to_numpy(int)
    if np.any(obs.loc[rows].role.eq('confirmation') & ~obs.loc[rows].control_selected) and 'confirmation' not in roles:
        raise RuntimeError('confirmation expression not authorized at this stage')
    sums=np.zeros((len(selected),len(axis)),float);control_rows=[];control_meta=[]
    io=remote();start=time.monotonic()
    with h5py.File(io,'r') as h:
        full_genes=h5_column(h['var'],'gene_ids').astype(str);colmap={g:i for i,g in enumerate(full_genes)}
        cols=np.asarray([colmap[g] for g in axis]);group=h['X'];pointer=group['indptr'][:]
        prefetch_allowed_storage(io,group,pointer,rows,roles)
        for step,row in enumerate(rows):
            record=obs.iloc[row]
            x=sparse_row(group,pointer,int(row),len(full_genes))
            if record.control_selected:
                control_rows.append(x);control_meta.append({'context':record.context,'donor_id':str(record.donor_id),
                    'observation_id':record.observation_id})
            else:sums[key_to_index[str(record.gene)+'::'+str(record.context)]]+=x[cols]
            if (step+1)%250==0:
                write_json(OUT/'TARGET_READ_STATUS.json',{'status':'READING_ALLOWED_ROWS','roles':roles,
                    'rows_read':step+1,'rows_planned':len(rows),'confirmation_read':'confirmation' in roles,
                    'wall_seconds':time.monotonic()-start})
                print(json.dumps({'target_rows':step+1,'total_rows':len(rows),'roles':roles}),flush=True)
    io.close()
    controls=np.asarray(control_rows,float);cm=pd.DataFrame(control_meta)
    output_controls={}
    for context in CONTEXTS:
        total=controls[cm.context.eq(context).to_numpy()][:,cols].sum(0)
        if total.sum()<=0:raise RuntimeError('empty context control counts')
        output_controls[context]=np.log1p(1e4*total/total.sum())
    if np.any(sums.sum(1)<=0):raise RuntimeError('empty selected query counts')
    effect=np.log1p(sums*(1e4/sums.sum(1))[:,None])-np.asarray([output_controls[c] for c in selected.context])
    name='CONFIRMATION' if roles==['confirmation'] else 'TRAIN_FEEDBACK'
    selected.to_parquet(OUT/f'{name}_TASKS.parquet',index=False);np.save(OUT/f'{name}_EFFECTS.npy',effect.astype(np.float32))
    control_payload={'counts':controls.astype(np.float32),'genes':full_genes,
        'contexts':cm.context.to_numpy(str),'donors':cm.donor_id.to_numpy(str)}
    if (OUT/'CONTROL_COUNTS.npz').exists():
        previous=np.load(OUT/'CONTROL_COUNTS.npz')
        for key,value in control_payload.items():
            if not np.array_equal(previous[key],value):raise RuntimeError('frozen control data changed')
    else:np.savez(OUT/'CONTROL_COUNTS.npz',**control_payload)
    write_json(OUT/f'{name}_READ_RECEIPT.json',{'roles':roles,'query_rows':len(selected),
        'logical_pseudobulk_rows_read':len(rows),'forbidden_query_rows_read':0,
        'forbidden_confirmation_rows_read':0,
        'authorized_confirmation_query_rows':len(selected) if 'confirmation' in roles else 0,
        'effect_contract_sha256':sha(OUT/'OUTPUT_CONTRACT.json'),'effects_sha256':sha(OUT/f'{name}_EFFECTS.npy')})
    ledger_path=resource_ledger_path()
    with ledger_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_text())
        ledger.pop('target_truth_read',None)
        ledger['training_development_truth_materialized']=(OUT/'TRAIN_FEEDBACK_READ_RECEIPT.json').exists()
        ledger['confirmation_truth_materialized']='confirmation' in roles
        ledger['materialization_status_source']='role-scoped read receipts, not opaque cache bytes'
        write_json(ledger_path,ledger)


def prepare_public():
    if (OUT/'PUBLIC_COUNTS_EFFECTS.npz').exists() and (OUT/'PUBLIC_SNAPSHOT.json').exists():return
    tasks=pd.read_parquet(OUT/'TASK_ROLES.parquet');genes=sorted(tasks.gene.unique());gi={g:i for i,g in enumerate(genes)}
    axis=json.loads((OUT/'OUTPUT_CONTRACT.json').read_text())['gene_ids']
    sums=np.zeros((len(genes)+1,len(axis)),float);cell_counts=np.zeros(len(genes)+1,np.int64)
    start=time.monotonic()
    with h5py.File(GWPS,'r') as h:
        vg=h5_column(h['var'],'ensembl_id').astype(str);cols=np.asarray([{g:i for i,g in enumerate(vg)}[g] for g in axis])
        ids=h5_column(h['obs'],'gene_id').astype(str);nperts=h5_column(h['obs'],'nperts').astype(str)
        controls=nperts=='0';code=np.asarray([gi.get(g,-1) for g in ids],int);code[controls]=len(genes)
        matrix=h['X']
        for begin in range(0,len(ids),4000):
            end=min(begin+4000,len(ids));mask=code[begin:end]>=0
            if not mask.any():continue
            block=np.asarray(matrix[begin:end,:],float)[mask][:,cols];groups=code[begin:end][mask]
            if np.any(block<0) or not np.allclose(block,np.round(block),atol=1e-5):raise RuntimeError('GWPS X is not raw counts')
            order=np.argsort(groups,kind='stable');u,first=np.unique(groups[order],return_index=True)
            sums[u]+=np.add.reduceat(block[order],first,axis=0)
            cell_counts+=np.bincount(groups,minlength=len(genes)+1)
            if end%40000==0:print(json.dumps({'public_cells_scanned':end,'total_cells':len(ids)}),flush=True)
    if sums[-1].sum()<=0:raise RuntimeError('no independent source control')
    ctrl=np.log1p(1e4*sums[-1]/sums[-1].sum());ok=sums[:-1].sum(1)>0
    effect=np.full((len(genes),len(axis)),np.nan,np.float32)
    effect[ok]=(np.log1p(sums[:-1][ok]*(1e4/sums[:-1][ok].sum(1))[:,None])-ctrl).astype(np.float32)
    np.savez(OUT/'PUBLIC_COUNTS_EFFECTS.npz',genes=np.asarray(genes),effects=effect,n_cells=cell_counts[:-1])
    write_json(OUT/'PUBLIC_SNAPSHOT.json',{'source_study':'ReplogleWeissman2022_K562_gwps','target_study_rows':0,
        'source_path':str(GWPS),'source_bytes':GWPS.stat().st_size,'output_contract_sha256':sha(OUT/'OUTPUT_CONTRACT.json'),
        'independent_record':'one screen/context effect per gene; guides are not independent experiments',
        'n_supported_genes':int(ok.sum()),'single_history_dispersion':0,
        'wall_seconds':time.monotonic()-start,'snapshot_sha256':sha(OUT/'PUBLIC_COUNTS_EFFECTS.npz')})


def control_features():
    if (OUT/'CONTROL_FEATURES.npz').exists():return
    z=np.load(OUT/'CONTROL_COUNTS.npz');counts=z['counts'].astype(float)
    normalized=np.log1p(counts*(1e6/counts.sum(1))[:,None]);context=z['contexts'];donors=z['donors']
    baseline=[];dropout=[];var=[]
    for c in CONTEXTS:
        mask=context==c;x=normalized[mask]
        baseline.append(x.mean(0));dropout.append((x==0).mean(0))
        means=np.asarray([normalized[mask&(donors==d)].mean(0) for d in sorted(set(donors[mask]))])
        var.append(means.var(0))
    m=normalized.T.astype(np.float32);m-=m.mean(1,keepdims=True);m/=m.std(1,keepdims=True)+1e-8
    embedding=TruncatedSVD(n_components=50,random_state=0).fit_transform(m).astype(np.float32)
    np.savez(OUT/'CONTROL_FEATURES.npz',genes=z['genes'],embedding=embedding,
        baseline=baseline,dropout=dropout,donor_var=var)
    write_json(OUT/'CONTROL_FEATURE_AUDIT.json',{'source':'only fixed NTC pseudobulks','perturbed_rows_used':0,
        'n_controls':len(counts),'embedding_dimensions':50,'control_sampling_cap_per_donor_context':64,
        'donor_variance_is_measured':True,'sampled_controls_are_an_explicit_official-feature-adaptation':True})


def competence_gate(tasks,prediction,truth,baseline):
    model_error=np.sqrt(np.mean((prediction-truth)**2,1));base_error=np.sqrt(np.mean((baseline-truth)**2,1))
    overall=float(model_error.mean()/base_error.mean()-1)
    genes,inverse=np.unique(tasks.gene.astype(str),return_inverse=True);rng=np.random.default_rng(20260930)
    gaps=[]
    for _ in range(5000):
        weight=np.bincount(rng.integers(0,len(genes),len(genes)),minlength=len(genes))[inverse]
        gaps.append(np.average(model_error,weights=weight)/np.average(base_error,weights=weight)-1)
    strata=[]
    for c in CONTEXTS:
        mask=tasks.context.eq(c).to_numpy();n=int(mask.sum())
        if n<20:strata.append({'context':c,'valid':False});continue
        pv=np.mean((prediction[mask]-prediction[mask].mean(0))**2)
        tv=np.mean((truth[mask]-truth[mask].mean(0))**2)
        strata.append({'context':c,'valid':bool(tv>1e-12),'relative_gap':float(model_error[mask].mean()/base_error[mask].mean()-1),
            'prediction_variance_ratio':float(pv/tv) if tv>1e-12 else None})
    valid=[s for s in strata if s['valid']]
    ci=np.quantile(gaps,[.025,.975])
    fraction=np.mean([s['relative_gap']<=.02 for s in valid]) if valid else 0
    pass_gate=(np.isfinite(prediction).all() and overall<=.02 and ci[1]<=.02 and fraction>=.6
        and len(valid)/len(CONTEXTS)>=.8 and all(s['prediction_variance_ratio']>=.01 for s in valid))
    return {'status':'PASS' if pass_gate else 'FAIL','relative_gap':overall,'ci95':ci.tolist(),
        'mean_model_rmse':float(model_error.mean()),'mean_condition_mean_rmse':float(base_error.mean()),
        'noninferior_strata_fraction':float(fraction),'valid_strata_fraction':len(valid)/len(CONTEXTS),
        'strata':strata,'qualification_error_rows':len(tasks),'qualification_gene_clusters':len(genes),
        'qualification_errors_not_risk_training_labels':True}


def train_predictors():
    if (OUT/'PREDICTOR_FREEZE.json').exists():
        if not (OUT/'PREDICTION_INPUT_ISOLATION.json').exists():raise RuntimeError('predictor freeze lacks input isolation receipt')
        return
    # Completed fits precede prediction QA. Resume from their saved weights,
    # rather than retraining all models after an inference-only fault.
    if (OUT/'PREDICTOR_FIT_LEDGER.json').exists():
        freeze_saved_predictors()
        return
    import torch
    torch.set_num_threads(4)
    control_features();tasks=pd.read_parquet(OUT/'TRAIN_FEEDBACK_TASKS.parquet')
    effect=np.load(OUT/'TRAIN_FEEDBACK_EFFECTS.npy');z=np.load(OUT/'CONTROL_FEATURES.npz')
    emb={str(g):e for g,e in zip(z['genes'],z['embedding'])}
    target_tasks=pd.read_parquet(OUT/'TASK_ROLES.parquet')
    prediction={name:np.zeros((len(target_tasks),effect.shape[1]),np.float32) for name in ['ridge','mlp']}
    mean_prediction=np.zeros_like(prediction['ridge']);fit_records=[]
    for c in CONTEXTS:
        train=np.flatnonzero(tasks.context.eq(c)&tasks.role.eq('predictor_train'))
        query=np.flatnonzero(target_tasks.context.eq(c))
        known=np.asarray([i for i in train if tasks.iloc[i].gene in emb]);x=np.asarray([emb[tasks.iloc[i].gene] for i in known])
        y=effect[known];mu=effect[train].mean(0)
        for name in prediction:prediction[name][query]=mu
        mean_prediction[query]=mu
        valid=np.asarray([i for i in query if target_tasks.iloc[i].gene in emb]);xq=np.asarray([emb[target_tasks.iloc[i].gene] for i in valid])
        ridge=Ridge(alpha=100).fit(x,y);prediction['ridge'][valid]=ridge.predict(xq)
        np.savez(OUT/f'RIDGE_{c}.npz',coef=ridge.coef_,intercept=ridge.intercept_,mean=mu)
        seed_predictions=[]
        for seed in MODEL_SEEDS:
            torch.manual_seed(seed)
            net=torch.nn.Sequential(torch.nn.Linear(50,256),torch.nn.ReLU(),torch.nn.Linear(256,y.shape[1]))
            opt=torch.optim.Adam(net.parameters(),lr=.001);xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
            losses=[]
            for epoch in range(30):
                order=torch.randperm(len(xt));epoch_loss=0
                for start in range(0,len(xt),2048):
                    ix=order[start:start+2048];opt.zero_grad();loss=torch.nn.functional.mse_loss(net(xt[ix]),yt[ix])
                    loss.backward();opt.step();epoch_loss+=float(loss.detach())*len(ix)/len(xt)
                losses.append(epoch_loss)
            net.eval()
            with torch.no_grad():seed_predictions.append(net(torch.tensor(xq,dtype=torch.float32)).numpy())
            torch.save(net.state_dict(),OUT/f'MLP_{c}_{seed}.pt')
            fit_records.append({'context':c,'seed':seed,'n_fit_rows':len(known),'epochs':30,
                'loss_first':losses[0],'loss_last':losses[-1],'all_losses':losses,
                'fit_gene_hash':digest_ids(tasks.iloc[known].gene),'query_truth_used':False})
            print(json.dumps({'predictor':'mlp','context':c,'seed':seed,'fit_rows':len(known),'loss_last':losses[-1]}),flush=True)
        prediction['mlp'][valid]=np.mean(seed_predictions,axis=0)
    target_tasks.to_parquet(OUT/'PREDICTION_TASKS.parquet',index=False)
    for name,p in prediction.items():np.save(OUT/f'{name}_ALL_PREDICTIONS.npy',p)
    np.save(OUT/'CONDITION_MEAN_ALL_PREDICTIONS.npy',mean_prediction)
    write_json(OUT/'PREDICTOR_FIT_LEDGER.json',fit_records)
    freeze_saved_predictors()


def freeze_saved_predictors():
    """Canonical inference from existing weights, without reading query truth.

    Float32 BLAS uses different reduction orders for different batch sizes.
    Evaluate the saved float32 parameters in float64 and cast the output once;
    retain the original bulk predictions and quantify that numerical change.
    The registered input-isolation/reload tolerance remains unchanged.
    """
    import torch
    torch.set_num_threads(4)
    target_tasks=pd.read_parquet(OUT/'TASK_ROLES.parquet')
    fit_tasks=pd.read_parquet(OUT/'TRAIN_FEEDBACK_TASKS.parquet')
    records=json.loads((OUT/'PREDICTOR_FIT_LEDGER.json').read_text())
    z=np.load(OUT/'CONTROL_FEATURES.npz');known_genes=set(z['genes'].astype(str))
    if len(records)!=len(CONTEXTS)*len(MODEL_SEEDS):raise RuntimeError('incomplete saved MLP fit ledger')
    artifacts=[OUT/'PREDICTOR_FIT_LEDGER.json',OUT/'CONTROL_FEATURES.npz']
    for c in CONTEXTS:
        known=fit_tasks[fit_tasks.context.eq(c)&fit_tasks.role.eq('predictor_train')&fit_tasks.gene.isin(known_genes)]
        for seed in MODEL_SEEDS:
            match=[r for r in records if r['context']==c and r['seed']==seed]
            if len(match)!=1 or match[0]['fit_gene_hash']!=digest_ids(known.gene):
                raise RuntimeError('saved predictor training identities do not match frozen roles')
            artifacts.append(OUT/f'MLP_{c}_{seed}.pt')
        artifacts.append(OUT/f'RIDGE_{c}.npz')
    artifact_hashes={p.name:sha(p) for p in artifacts}
    archive=OUT/'prefreeze_float32_predictions';archive.mkdir(exist_ok=True)
    audits={}
    for name in ['ridge','mlp']:
        path=OUT/f'{name}_ALL_PREDICTIONS.npy';original=archive/path.name
        if path.exists() and not original.exists():shutil.copy2(path,original)
        prediction=frozen_predict(name,target_tasks[['gene','context']])
        if not np.isfinite(prediction).all():raise RuntimeError('nonfinite canonical predictor output')
        np.save(path,prediction)
        # Check every context/role, including confirmation metadata, without
        # supplying or loading any confirmation expression.
        subset=target_tasks.groupby(['context','role'],sort=False).head(20)
        clean=frozen_predict(name,subset[['gene','context']])
        poisoned=frozen_predict(name,subset.assign(query_truth=1000.,true_error_rmse=-99.))
        np.testing.assert_array_equal(clean,poisoned)
        np.testing.assert_allclose(clean,prediction[subset.index],rtol=1e-5,atol=1e-7)
        audit={'canonical_subset_reload_max_abs_difference':float(np.max(np.abs(clean-prediction[subset.index]))),
               'poisoned_truth_exact_equal':True,'checked_rows':len(subset)}
        if original.exists():
            old=np.load(original);delta=np.abs(old-prediction)
            audit.update(original_prediction_sha256=sha(original),original_archive=str(original),
                float32_to_canonical_max_abs_difference=float(delta.max()),
                float32_to_canonical_rmse=float(np.sqrt(np.mean(delta.astype(float)**2))))
        audits[name]=audit
    for p in artifacts:
        if sha(p)!=artifact_hashes[p.name]:raise RuntimeError('predictor parameters changed during inference-only recovery')
    target_tasks.to_parquet(OUT/'PREDICTION_TASKS.parquet',index=False)
    write_json(OUT/'PREDICTOR_NUMERICAL_RECOVERY.json',{'cause':'float32 MLP matrix reductions differ by query batch size',
        'action':'canonical float64 CPU inference from unchanged saved float32 parameters; outputs cast once to float32',
        'new_fits':0,'confirmation_expression_rows_read':0,'predictor_artifact_sha256':artifact_hashes,
        'reload_rtol':1e-5,'reload_atol':1e-7,'prediction_audits':audits})
    write_json(OUT/'PREDICTION_INPUT_ISOLATION.json',{'input_fields':['gene','context'],
        'perturbed_query_expression_supplied':False,'confirmation_expression_rows_read_before_prediction_freeze':0,
        'training_matrix_roles':['predictor_train'],'metadata_extra_truth_columns_ignored':True,
        'inference_precision':'float64 computation / float32 saved outputs','all_context_role_groups_checked':True,
        'numerical_recovery_sha256':sha(OUT/'PREDICTOR_NUMERICAL_RECOVERY.json')})
    write_json(OUT/'PREDICTOR_FREEZE.json',{'status':'FROZEN','fit_role':'predictor_train',
        'predictor_parameters':CONFIG['external_predictors'],'confirmation_truth_read':False,
        'prediction_sha256':{name:sha(OUT/f'{name}_ALL_PREDICTIONS.npy') for name in ['ridge','mlp']},
        'predictor_artifact_sha256':artifact_hashes,
        'task_roles_sha256':sha(OUT/'TASK_ROLES.parquet'),'MLP_primary':'three-seed ensemble; seeds not independent biology',
        'input_isolation_sha256':sha(OUT/'PREDICTION_INPUT_ISOLATION.json')})


def frozen_predict(name,metadata):
    import torch
    if name not in ['ridge','mlp']:raise ValueError('unregistered frozen predictor')
    if not {'gene','context'}.issubset(metadata.columns) or not metadata.context.isin(CONTEXTS).all():
        raise ValueError('query must contain registered gene/context metadata')
    torch.set_num_threads(4)
    z=np.load(OUT/'CONTROL_FEATURES.npz');emb={str(g):e for g,e in zip(z['genes'],z['embedding'])}
    n_genes=json.loads((OUT/'OUTPUT_CONTRACT.json').read_text())['n_output_genes']
    result=np.empty((len(metadata),n_genes),np.float32)
    for c in CONTEXTS:
        selected=np.flatnonzero(metadata.context.eq(c).to_numpy())
        if not len(selected):continue
        ridge=np.load(OUT/f'RIDGE_{c}.npz');result[selected]=ridge['mean']
        known=np.asarray([i for i in selected if metadata.iloc[i].gene in emb],int)
        if not len(known):continue
        x=np.asarray([emb[metadata.iloc[i].gene] for i in known],np.float64)
        if 'x_mean' in ridge.files:x=(x-ridge['x_mean'])/ridge['x_scale']
        if name=='ridge':result[known]=x@ridge['coef'].astype(np.float64).T+ridge['intercept'].astype(np.float64)
        else:
            predictions=[]
            for seed in MODEL_SEEDS:
                net=torch.nn.Sequential(torch.nn.Linear(50,256),torch.nn.ReLU(),torch.nn.Linear(256,n_genes))
                net.load_state_dict(torch.load(OUT/f'MLP_{c}_{seed}.pt',map_location='cpu',weights_only=True));net.double().eval()
                with torch.no_grad():predictions.append(net(torch.tensor(x)).numpy())
            result[known]=np.mean(predictions,axis=0)+(ridge['mean'] if 'x_mean' in ridge.files else 0.)
    return result


def qualification():
    freeze=json.loads((OUT/'PREDICTOR_FREEZE.json').read_text())
    if sha(OUT/'TASK_ROLES.parquet')!=freeze['task_roles_sha256']:raise RuntimeError('predictor query roles changed after freeze')
    for name,digest in freeze['prediction_sha256'].items():
        if sha(OUT/f'{name}_ALL_PREDICTIONS.npy')!=digest:raise RuntimeError('frozen predictor outputs changed')
    for name,digest in freeze['predictor_artifact_sha256'].items():
        if sha(OUT/name)!=digest:raise RuntimeError('frozen predictor fit artifacts changed')
    tasks=pd.read_parquet(OUT/'TRAIN_FEEDBACK_TASKS.parquet');truth=np.load(OUT/'TRAIN_FEEDBACK_EFFECTS.npy')
    all_tasks=pd.read_parquet(OUT/'PREDICTION_TASKS.parquet').reset_index().set_index('task_id')
    feedback=tasks[tasks.role.eq('feedback')].reset_index()
    ii=all_tasks.loc[feedback.task_id,'index'].to_numpy(int);yt=truth[feedback['index'].to_numpy(int)]
    baseline=np.load(OUT/'CONDITION_MEAN_ALL_PREDICTIONS.npy')[ii]
    gates={}
    for name in ['ridge','mlp']:
        p=np.load(OUT/f'{name}_ALL_PREDICTIONS.npy')[ii]
        gates[name]=competence_gate(feedback,p,yt,baseline)
    write_json(OUT/'PREDICTOR_COMPETENCE.json',gates)
    passed=[name for name,g in gates.items() if g['status']=='PASS']
    write_json(OUT/'EXTERNAL_DECISION.json',{'passed_predictors':passed,
        'role':'DUAL_MECHANISM_CONFIRMATION_READY' if len(passed)==2 else 'SINGLE_MECHANISM_CONFIRMATION_READY' if passed else 'RELIABILITY_STRESS_TEST_BACKUP_REQUIRED',
        'confirmation_truth_read':False,'risk_algorithm_changed_using_qualification_errors':False})
    print(json.dumps({'competence':{k:{'status':v['status'],'gap':v['relative_gap'],'ci':v['ci95']} for k,v in gates.items()}}),flush=True)


def features(tasks,prediction):
    """Prediction and legal external historical responses only; no query truth."""
    f=tasks.copy();p=np.asarray(prediction,float)
    values=[np.sqrt(np.mean(p*p,1)),np.mean(np.abs(p),1),p.mean(1),p.std(1),
            np.quantile(np.abs(p),.95,axis=1),np.mean(np.abs(p)<=1e-12,1)]
    for c,v in zip(P,values):f[c]=v
    control=np.load(OUT/'CONTROL_FEATURES.npz');gene_index={str(g):i for i,g in enumerate(control['genes'])}
    gi=np.asarray([gene_index.get(str(g),-1) for g in f.gene]);ci=np.asarray([CONTEXTS.index(c) for c in f.context])
    for name,array in [('native_control_baseline','baseline'),('native_control_dropout','dropout'),('native_control_donor_variance','donor_var')]:
        out=np.full(len(f),np.nan);ok=gi>=0;out[ok]=control[array][ci[ok],gi[ok]];f[name]=out
    f['native_prediction_abs_mean']=f.prediction_abs_mean
    for k,c in enumerate(CONTEXTS):f[f'native_state_{k}']=f.context.eq(c).astype(float)
    for k in range(50):
        v=np.full(len(f),np.nan);ok=gi>=0;v[ok]=control['embedding'][gi[ok],k];f[f'native_embedding_{k}']=v
    f['native_training_similarity']=np.nan
    bank=np.load(OUT/'PUBLIC_COUNTS_EFFECTS.npz');hist_index={str(g):i for i,g in enumerate(bank['genes'])}
    hi=np.asarray([hist_index.get(str(g),-1) for g in f.gene]);hist=np.full_like(p,np.nan)
    ok=hi>=0;hist[ok]=bank['effects'][hi[ok]]
    available=np.isfinite(hist).all(1)&(bank['n_cells'][np.maximum(hi,0)]>0)
    prior_mag=np.full(len(f),np.nan);distance=np.full(len(f),np.nan);cosine=np.full(len(f),np.nan)
    prior_mag[available]=np.sqrt(np.mean(hist[available]**2,1))
    distance[available]=np.sqrt(np.mean((p[available]-hist[available])**2,1))
    den=np.sqrt(np.sum(p[available]**2,1)*np.sum(hist[available]**2,1))
    cosine[available]=np.divide(np.sum(p[available]*hist[available],1),den,
        out=np.full(int(available.sum()),np.nan),where=den>1e-12)
    f['prior_magnitude']=prior_mag;f['prediction_prior_rmse']=distance;f['prediction_prior_cosine']=cosine
    f['prior_uncertainty']=np.where(available,0.,np.nan)
    n=np.zeros(len(f));n[available]=bank['n_cells'][hi[available]]
    f['log_history_support']=np.where(available,np.log1p(n),np.nan)
    f['effective_sources']=np.where(available,1.,np.nan);f['history_conflict']=np.where(available,0.,np.nan)
    f['public_available']=available;f['public_raw']=distance
    f['support_raw']=np.where(available,-np.log1p(n),np.nan)
    f['evidence_status']=np.where(available,'LEGAL_OTHER_STUDY_HISTORY','NO_LEGAL_HISTORY_AMPLITUDE_FALLBACK')
    return f


def ecdf_fit(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    if len(v)<2:raise RuntimeError('insufficient training-side scores for channel CDF')
    return np.sort(v)


def ecdf_apply(sorted_values,values):
    a=np.asarray(sorted_values);v=np.asarray(values,float)
    return (np.searchsorted(a,v,'left')+np.searchsorted(a,v,'right'))/(2*len(a))


def risk_freeze():
    if (OUT/'RISK_FREEZE.json').exists():return
    decision=json.loads((OUT/'EXTERNAL_DECISION.json').read_text());passed=decision['passed_predictors']
    if not passed:
        stress_test_development()
        write_json(OUT/'PIPELINE_STATUS.json',{'status':'COMPETENCE_FAILED_BACKUP_REQUIRED',
            'confirmation_truth_read':False,'stress_test_scope':'development only','research_complete':False})
        return
    task=pd.read_parquet(OUT/'PREDICTION_TASKS.parquet');selected=pd.read_parquet(OUT/'TRAIN_FEEDBACK_TASKS.parquet')
    y=np.load(OUT/'TRAIN_FEEDBACK_EFFECTS.npy');truth_index=dict(zip(selected.task_id,np.arange(len(selected))))
    query=task[task.role.eq('confirmation')].reset_index(drop=True);feedback=task[task.role.eq('feedback')].reset_index(drop=True)
    feedback_idx=task.index[task.role.eq('feedback')].to_numpy(int);query_idx=task.index[task.role.eq('confirmation')].to_numpy(int)
    native=['native_prediction_abs_mean','native_control_baseline','native_control_dropout','native_control_donor_variance']
    native +=[f'native_state_{i}' for i in range(len(CONTEXTS))]+[f'native_embedding_{i}' for i in range(50)]+['native_training_similarity']
    if CONTEXTS==['KOLF2.1J']:
        native=P+['native_control_baseline','native_control_dropout','native_control_donor_variance']+[f'native_state_{i}' for i in range(len(CONTEXTS))]+[f'native_embedding_{i}' for i in range(50)]+['native_training_similarity']
    all_scores=[];ledger=[];channel_audit=[];models_written=[];source_snap=sha(OUT/'PUBLIC_COUNTS_EFFECTS.npz')
    for name in passed:
        p=np.load(OUT/f'{name}_ALL_PREDICTIONS.npy');train=features(feedback,p[feedback_idx]);q=features(query,p[query_idx])
        yy=y[np.asarray([truth_index[t] for t in train.task_id])]
        train['true_error_rmse']=np.sqrt(np.mean((p[feedback_idx]-yy)**2,1))
        # The channel transforms are fitted on feature columns only, never errors.
        amp=ecdf_fit(train.predicted_magnitude)
        pub=ecdf_fit(train.public_raw);support=ecdf_fit(train.support_raw)
        transformations={'Amplitude':amp,'Public':pub,'Support':support}
        if CONTEXTS==['KOLF2.1J']:transformations['HistoryEnergy']=ecdf_fit(train.prior_magnitude)
        native_train, native_query=native_x(train,q,native)
        # training prototypes for the unsupervised similarity heuristic
        sim_train=native_train[:,-1];sim_query=native_query[:,-1]
        sim_cdf=ecdf_fit(sim_train);transformations['Similarity']=sim_cdf
        np.savez(OUT/f'{name}_CHANNEL_CDFS.npz',**transformations)
        q['similarity_raw']=sim_query
        rules={};states={};version=name+'::v21::'+source_snap[:16]
        for method in ['Magnitude','Similarity','HistorySupport','PublicRule']+(['HistoryEnergy'] if CONTEXTS==['KOLF2.1J'] else []):
            cfg={'method':method,'version':version,'channel_cdfs':transformations}
            result=[score_task(row,cfg) for row in q.to_dict('records')]
            rules[method]=np.asarray([r[0] for r in result]);states[method]=[r[1] for r in result]
        q[['task_id','gene','context','target','role','evidence_status','public_available']].to_parquet(OUT/f'{name}_CONFIRM_QUERY_METADATA.parquet',index=False)
        for method,score in rules.items():
            if not np.isfinite(score).all():raise RuntimeError('nonfinite rule score on full confirmation cohort')
            part=q[['task_id']].copy();part['predictor']=name;part['method']=method;part['feedback_budget']=0.
            part['order_seed']=-1;part['learner_seed']=-1;part['risk']=score;part['evidence_status']=states[method];part['scorer_version']=version;all_scores.append(part)
            channel_audit.append({'predictor':name,'method':method,'cdf_role':'feedback features only',
                'cdf_error_labels_used':0,'qualification_error_rows_separate':len(train),
                'target_error_risk_training_rows':0,'n_fit_score_rows':len(train)})
        inputs={'Target_P6':P,'PertEMA_native_control':native,'PertEMA_native_control_Public':native+PUBLIC,
                'PertEMA_native_control_Support':native+['log_history_support']}
        # A fixed same-information error-regression recipe, not bundled CD4 weights.
        cache={}
        for order_seed in ORDERS:
            ordered=sorted(train.gene.unique(),key=lambda g:hashlib.sha256(f'SafeConf-v21-feedback|{order_seed}|{g}'.encode()).hexdigest())
            for budget in BUDGETS[1:]:
                genes=ordered[:math.ceil(budget*len(ordered))];fit=train[train.gene.isin(genes)].reset_index(drop=True)
                for method,cols in inputs.items():
                    xt,xq=native_x(fit,q,cols)
                    for seed in MODEL_SEEDS:
                        key=(digest_ids(fit.task_id),method,seed)
                        if key not in cache:
                            model=official_gbt(seed);model.fit(xt,fit.true_error_rmse.to_numpy(float),sample_weight=cluster_weights(fit))
                            cache[key]=model.predict(xq)
                            dest=OUT/f'RISK_{name}_{method}_o{order_seed}_b{budget}_s{seed}.json';model.get_booster().save_model(dest)
                            models_written.append({'path':str(dest),'sha256':sha(dest)})
                        part=q[['task_id']].copy();part['predictor']=name;part['method']=method;part['feedback_budget']=budget
                        part['order_seed']=order_seed;part['learner_seed']=seed
                        cfg={'method':'FrozenSupervised','version':version+'::'+method}
                        result=[score_task({'frozen_model_score':value,'evidence_status':state},cfg)
                            for value,state in zip(cache[key],q.evidence_status)]
                        part['risk']=[r[0] for r in result];part['evidence_status']=[r[1] for r in result];part['scorer_version']=cfg['version'];all_scores.append(part)
                        ledger.append({'predictor':name,'method':method,'feedback_budget':budget,'order_seed':order_seed,
                            'learner_seed':seed,'risk_training_error_rows':len(fit),'risk_training_gene_clusters':len(genes),
                            'qualification_error_rows_preparation':len(train),'qualification_gene_clusters':train.gene.nunique(),
                            'qualification_and_training_error_union_rows':len(train),
                            'training_error_record_hash':digest_ids(fit.task_id),'confirmation_errors_used':0,
                            'labels':'raw RMSE on the same current counts contract','complete_conformal_pipeline':False})
            print(json.dumps({'risk_frozen_predictor':name,'order':order_seed,'configs':len(ledger)}),flush=True)
    pd.concat(all_scores,ignore_index=True).to_parquet(OUT/'FROZEN_CONFIRMATION_SCORES.parquet',index=False)
    pd.DataFrame(ledger).to_csv(OUT/'INFORMATION_BUDGET_LEDGER.csv',index=False)
    pd.DataFrame(channel_audit).to_csv(OUT/'CHANNEL_SCORE_CDF_AUDIT.csv',index=False)
    write_json(OUT/'RISK_FREEZE.json',{'status':'FROZEN_BEFORE_CONFIRMATION_TRUTH',
        'passed_predictors':passed,'confirmation_score_sha256':sha(OUT/'FROZEN_CONFIRMATION_SCORES.parquet'),
        'predictor_freeze_sha256':sha(OUT/'PREDICTOR_FREEZE.json'),'public_snapshot_sha256':source_snap,
        'configuration_sha256':sha(RUN/'CONFIG.json'),'cohort_protocol_sha256':sha(OUT/'COHORT_AND_ACCESS_PROTOCOL.json'),
        'risk_models':models_written,'confirmation_truth_read':False,
        'zero_target_error_supervision_scope':'PublicRule risk fitting/selection/calibration',
        'adopted_system':{'method':'PublicRule','no_history':'training-feature CDF Amplitude fallback','Source_enabled':False,'Target_enabled':False},
        'scoring_entrypoint_sha256':sha(ROOT/'tools/safeconf_continual/frozen_scoring.py'),
        'qualification_error_use_separate':True,'source_study_is_other_study':True})


def stress_test_development():
    """Keep an actual DEV stress result if both predictors fail, TEST sealed."""
    if (OUT/'DEVELOPMENT_STRESS_RESULTS.csv').exists():return
    task=pd.read_parquet(OUT/'TRAIN_FEEDBACK_TASKS.parquet');truth=np.load(OUT/'TRAIN_FEEDBACK_EFFECTS.npy')
    all_tasks=pd.read_parquet(OUT/'PREDICTION_TASKS.parquet').reset_index().set_index('task_id')
    train=task[task.role.eq('predictor_train')].reset_index();dev=task[task.role.eq('feedback')].reset_index()
    rows=[];paired=[]
    for name in ['ridge','mlp']:
        p=np.load(OUT/f'{name}_ALL_PREDICTIONS.npy');ti=all_tasks.loc[train.task_id,'index'].to_numpy(int);di=all_tasks.loc[dev.task_id,'index'].to_numpy(int)
        fit=features(train,p[ti]);q=features(dev,p[di]);ep=endpoint_arrays(p[di],truth[dev['index'].to_numpy(int)])
        transforms={'Amplitude':ecdf_fit(fit.predicted_magnitude),'Public':ecdf_fit(fit.public_raw),'Support':ecdf_fit(fit.support_raw)}
        scores={}
        for method in ['Magnitude','HistorySupport','PublicRule']:
            cfg={'method':method,'version':name+'::DEV_STRESS_ONLY','channel_cdfs':transforms}
            scores[method]=np.asarray([score_task(r,cfg)[0] for r in q.to_dict('records')])
            for endpoint,error in ep.items():
                for review in REVIEWS:
                    for scope,ix in [('global',np.arange(len(q)))]+list(q.groupby('context').indices.items()):
                        rows.append({'predictor':name,'method':method,'endpoint':endpoint,'review':review,'scope':scope,
                            'evidence_role':'FAILED_COMPETENCE_DEV_STRESS_ONLY','confirmation_truth_read':False,
                            **point_metrics(error[ix],scores[method][ix],q.task_id.to_numpy(str)[ix],review)})
        boot=ClusterBootstrap(q,ep['delta_rmse'])
        def macro(s):return np.mean([point_metrics(ep['delta_rmse'][ix],s[ix],q.task_id.to_numpy(str)[ix])['utility'] for ix in q.groupby('context').indices.values()])
        for reference in ['Magnitude','HistorySupport']:
            paired.append({'predictor':name,'comparison':'PublicRule-minus-'+reference,
                **summarize_draws(boot.difference(scores['PublicRule'],scores[reference]),macro(scores['PublicRule'])-macro(scores[reference]))})
    pd.DataFrame(rows).to_csv(OUT/'DEVELOPMENT_STRESS_RESULTS.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'DEVELOPMENT_STRESS_PAIRED_BOOTSTRAP.csv',index=False)


def confirmation():
    if (OUT/'CONFIRMATION_COMPLETE.json').exists():return
    freeze=json.loads((OUT/'RISK_FREEZE.json').read_text())
    if sha(OUT/'FROZEN_CONFIRMATION_SCORES.parquet')!=freeze['confirmation_score_sha256']:
        raise RuntimeError('confirmation scores changed after freeze')
    if not (OUT/'CONFIRMATION_EVALUATION_OPEN_EVENT.json').exists():
        write_json(OUT/'CONFIRMATION_EVALUATION_OPEN_EVENT.json',{'time_utc':pd.Timestamp.now(tz='UTC').isoformat(),
            'risk_freeze_sha256':sha(OUT/'RISK_FREEZE.json'),'authorized_by':'user v2.1 fixed final confirmation protocol',
            'roles':['confirmation'],'purpose':'scoring only; no refitting/selection'})
    if not (OUT/'CONFIRMATION_READ_RECEIPT.json').exists():prepare_target(['confirmation'])
    task=pd.read_parquet(OUT/'CONFIRMATION_TASKS.parquet');truth=np.load(OUT/'CONFIRMATION_EFFECTS.npy')
    all_tasks=pd.read_parquet(OUT/'PREDICTION_TASKS.parquet').reset_index().set_index('task_id')
    ix=all_tasks.loc[task.task_id,'index'].to_numpy(int);scores=pd.read_parquet(OUT/'FROZEN_CONFIRMATION_SCORES.parquet')
    rows=[];paired=[]
    coverage_rows=[];fallback_rows=[]
    for name in freeze['passed_predictors']:
        prediction=np.load(OUT/f'{name}_ALL_PREDICTIONS.npy')[ix];endpoints=endpoint_arrays(prediction,truth)
        np.savez(OUT/f'{name}_CONFIRMATION_ENDPOINTS.npz',**endpoints)
        metadata_path=OUT/f'{name}_CONFIRM_QUERY_METADATA.parquet'
        if not metadata_path.exists():raise RuntimeError('frozen query evidence metadata is missing')
        meta=pd.read_parquet(metadata_path).set_index('task_id').loc[task.task_id]
        available=meta.public_available.to_numpy(bool)
        evidence_status=meta.evidence_status.to_numpy(str)
        coverage_rows += [
            {'predictor':name,'scope':'full_deployment','n_tasks':len(task),'n_supported':int(available.sum()),
             'n_no_history':int((~available).sum()),'coverage_fraction':float(available.mean()),
             'confirmation_truth_read':True},
            {'predictor':name,'scope':'public_supported','n_tasks':int(available.sum()),'n_supported':int(available.sum()),
             'n_no_history':0,'coverage_fraction':1.0 if available.any() else np.nan,'confirmation_truth_read':True},
            {'predictor':name,'scope':'no_history_fallback','n_tasks':int((~available).sum()),'n_supported':0,
             'n_no_history':int((~available).sum()),'coverage_fraction':0.0 if (~available).any() else np.nan,
             'confirmation_truth_read':True}]
        by_method={}
        for (method,budget,order,seed),q in scores[scores.predictor.eq(name)].groupby(['method','feedback_budget','order_seed','learner_seed']):
            risk=q.set_index('task_id').loc[task.task_id].risk.to_numpy(float)
            by_method.setdefault((method,budget),[]).append(risk)
            for endpoint,error in endpoints.items():
                for review in REVIEWS:
                    scopes=[('global',np.arange(len(task))),('full_deployment',np.arange(len(task))),
                            ('public_supported',np.flatnonzero(available)),('no_history_fallback',np.flatnonzero(~available))]
                    scopes += [(c,v) for c,v in task.groupby('context').indices.items()]
                    for scope,use in scopes:
                        rows.append({'predictor':name,'method':method,'feedback_budget':budget,'order_seed':order,
                            'learner_seed':seed,'endpoint':endpoint,'scope':scope,
                            **point_metrics(error[use],risk[use],task.task_id.to_numpy(str)[use],review)})
        engine=ClusterBootstrap(task,endpoints['delta_rmse']);reference=by_method[('PublicRule',0.)][0]
        def macro(s):return np.nanmean([point_metrics(endpoints['delta_rmse'][use],s[use],task.task_id.to_numpy(str)[use])['utility'] for use in task.groupby('context').indices.values()])
        for (method,budget),values in by_method.items():
            if method=='PublicRule':continue
            values=np.asarray(values);point=float(np.mean([macro(s) for s in values])-macro(reference))
            paired.append({'predictor':name,'method':method,'feedback_budget':budget,'comparison':'method-minus-PublicRule',
                **summarize_draws(engine.difference(values,reference),point)})
        # In the no-history stratum PublicRule must be exactly its registered
        # Magnitude fallback. This is a deployment sanity check, not a result
        # chosen after seeing the confirmation errors.
        if (~available).any():
            public_score=by_method[('PublicRule',0.)][0][~available]
            magnitude_score=by_method[('Magnitude',0.)][0][~available]
            fallback_rows.append({'predictor':name,'n_no_history':int((~available).sum()),
                'max_abs_difference_public_vs_magnitude':float(np.max(np.abs(public_score-magnitude_score))),
                'exact_equal':bool(np.array_equal(public_score,magnitude_score)),
                'evidence_status_values':sorted(set(evidence_status[~available]))})
    pd.DataFrame(rows).to_csv(OUT/'CONFIRMATION_METRICS.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'FINAL_KOLF_RESULT_TABLE.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'CONFIRMATION_PAIRED_BOOTSTRAP.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'FINAL_KOLF_PAIRED_BOOTSTRAP.csv',index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT/'PUBLIC_COVERAGE_REPORT.csv',index=False)
    write_json(OUT/'PUBLIC_FALLBACK_SANITY.json',{'status':'PASS' if all(r['exact_equal'] for r in fallback_rows) else 'FAIL',
        'rows':fallback_rows,'public_memory_excludes_target_study':True,'target_error_labels_used_by_public':0})
    if (OUT/'INFORMATION_BUDGET_LEDGER.csv').exists():
        pd.read_csv(OUT/'INFORMATION_BUDGET_LEDGER.csv').to_csv(OUT/'FINAL_INFORMATION_BUDGET_LEDGER.csv',index=False)
    label=[]
    for (name,method),q in pd.DataFrame(paired).query('feedback_budget > 0').groupby(['predictor','method']):
        for kind,col in [('primary_95','ci95_lower'),('five_budget_sensitivity','ci99_lower')]:
            passing=q[q[col]>=-.005];budget=float(passing.feedback_budget.min()) if len(passing) else None
            label.append({'predictor':name,'method':method,'definition':kind,'budget':budget,'right_censored':budget is None})
    pd.DataFrame(label).to_csv(OUT/'CONFIRMATION_LABEL_EQUIVALENT.csv',index=False)
    write_json(OUT/'CONFIRMATION_COMPLETE.json',{'status':'COMPLETE','n_tasks':len(task),'n_clusters':task.gene.nunique(),
        'predictors':freeze['passed_predictors'],'risk_configuration_changed_after_confirmation':False,
        'new_gpu_hours':0,'final_truth_role':'scoring only','label_equivalent':label})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['metadata','prepare','predictors','qualification','all-preconfirmation','risk-freeze','confirmation','complete'],required=True)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.phase in ['metadata','all-preconfirmation','complete']:metadata()
    if a.phase in ['prepare','all-preconfirmation','complete']:
        if not (OUT/'TRAIN_FEEDBACK_READ_RECEIPT.json').exists():prepare_target(['predictor_train','feedback'])
        prepare_public();control_features()
    if a.phase in ['predictors','all-preconfirmation','complete']:train_predictors()
    if a.phase in ['qualification','all-preconfirmation','complete']:qualification()
    if a.phase in ['risk-freeze','complete']:risk_freeze()
    if a.phase=='confirmation' or (a.phase=='complete' and (OUT/'RISK_FREEZE.json').exists()):confirmation()


if __name__=='__main__':main()
