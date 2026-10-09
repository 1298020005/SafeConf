#!/usr/bin/env python3
"""Role-guarded high-precision KOLF resource-panel confirmation pipeline.

Same basic-QC perturbation cohort; a Source-hashed1400-gene output panel is
fixed before response values. No quantized browser responses are used. Sparse
row indices are storage locators only; forbidden count values are never decoded.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,time,hashlib,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,h5py,torch
from sklearn.decomposition import PCA,TruncatedSVD
from tools.scripts import run_safeconf_gladstone_v21 as base
from tools.scripts import repair_safeconf_gladstone_training_v21 as repair
from tools.safeconf_continual.submission_evidence import write_json,sha,digest_ids,h5_column
from tools.safeconf_continual.range_h5 import MeteredHTTPFile

RUN=base.RUN;OUT=RUN/'external_kolf_panel1400_v1'
PREFLIGHT=RUN/'backup_asset/kolf_budget_panel1400_v1'
RAW_PREFLIGHT=RUN/'backup_asset/kolf_raw_preflight_v1'
URL='https://ndownloader.figshare.com/files/64650261'
CHECKPOINT=Path('/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human')
CONTEXT='KOLF2.1J'
ORIGINAL_FEATURES=base.features


def remote():
    return MeteredHTTPFile(URL,RAW_PREFLIGHT/'range_cache',RUN/'external/DOWNLOAD_RESOURCE_LEDGER.json')


def configure():
    OUT.mkdir(exist_ok=True)
    if json.loads((PREFLIGHT/'ASSET_QUALIFICATION.json').read_text())['status']!='RAW_ASSET_METADATA_AND_COST_QUALIFIED':
        raise RuntimeError('resource panel did not pass metadata/cost gate')
    candidate=pd.read_parquet(PREFLIGHT/'CANDIDATE_TASK_ROLES.parquet');axis=json.loads((PREFLIGHT/'CANDIDATE_AXIS.json').read_text())
    tasks=candidate.copy();tasks['context']=CONTEXT;tasks['target']=CONTEXT
    role=OUT/'TASK_ROLES.parquet'
    if role.exists():pd.testing.assert_frame_equal(pd.read_parquet(role),tasks)
    else:tasks.to_parquet(role,index=False)
    contract={'gene_ids':axis['labels'],'source_ensembl_ids':axis['ensembl_ids'],'n_output_genes':len(axis['labels']),
        'target_column_indices':axis['column_indices'],'gene_axis_hash':axis['axis_hash'],
        'effect_definition':'log1p(CP10k pooled raw counts on fixed1400-gene panel) minus same-context pooled NTC',
        'normalization_denominator':'same registered1400-gene panel in both studies; not whole-transcriptome total',
        'aggregation':'pool before normalization in both studies','target_study_excluded_from_public':True,
        'public_source':'Replogle K562 GWPS','output_panel_chosen_using_target_response':False,
        'panel_selection':'Source-only stable gene hash for remaining resource allowance',
        'registered_scope':'new KOLF resource-panel study, legacy primary endpoints unchanged',
        'quantized_viewer_truth_used':False,'missing_genes_zero_filled':False}
    p=OUT/'OUTPUT_CONTRACT.json'
    if p.exists() and json.loads(p.read_text())!=contract:raise RuntimeError('changed frozen output contract')
    write_json(p,contract)
    card={'run_id':OUT.name,'public_error_supervision':0,'source_error_model_enabled':False,
        'new_large_upstream_training':0,'new_gpu_hours':0,'cpu_threads':4,
        'predictors':'fixed frozen scGPT gene embeddings → training-gene PCA50 → Ridge100 / centered MLP50-256-output',
        'pretrained_gene_embeddings_are_not_full_scGPT_perturbation_predictions':True,
        'ridge_alpha':100,'mlp_lr':.001,'max_optimizer_steps':300,'stopping':'upstream-only10% gene holdout; patience30',
        'checkpoint_path':str(CHECKPOINT),'checkpoint_sha256':sha(CHECKPOINT/'best_model.pt'),
        'vocabulary_sha256':sha(CHECKPOINT/'vocab.json'),'checkpoint_finetuning':False,
        'fresh_role_hash':sha(role),'output_contract_sha256':sha(p),'gene_clusters':len(tasks),
        'competence_gate':'original v2.1 unchanged,95%upper<=2%; per-state prediction variance>=1%',
        'confirmation_before_risk_freeze':False,'target_capability_error_cost_separate':True,
        'sparse_locator_rows_used_for_role_mask_only':True}
    c=OUT/'EXPERIMENT_CARD.json'
    if c.exists() and json.loads(c.read_text())!=card:raise RuntimeError('changed frozen KOLF execution card')
    write_json(c,card)
    write_json(OUT/'ROLE_FREEZE.json',{'status':'FROZEN_BEFORE_TARGET_COUNT_VALUES',
        'task_roles_sha256':sha(role),'output_contract_sha256':sha(p),'gene_clusters':len(tasks),
        'confirmation_genes':int(tasks.role.eq('confirmation').sum()),
        'selection_uses_error_values':False,'panel_selected_before_target_effects':True})
    write_json(OUT/'SHARED_RESOURCE_CACHE.json',{'cache':str(RAW_PREFLIGHT/'range_cache'),
        'ledger':str(RUN/'external/DOWNLOAD_RESOURCE_LEDGER.json'),'budget_restarted':False})
    protocol={'role_registry_sha256':sha(role),'output_contract_sha256':sha(p),
        'preexisting_experiment_card_sha256':sha(c),'upstream_training_genes':600,'feedback_genes':300,'confirmation_genes':300,
        'public_target_study_rows_allowed':0,'preconfirmation_count_roles':['predictor_train','feedback','NTC'],
        'sparse_row_indices':'storage locators only; never predictor/risk features',
        'forbidden_count_value_materialization':False,'confirmation_open_requires_risk_freeze_and_score_hash':True,
        'risk_error_labels':'feedback role only; qualification preparation cost separately charged',
        'scope':'registered Source-defined1400-gene panel; no legacy endpoint modification'}
    pp=OUT/'COHORT_AND_ACCESS_PROTOCOL.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise RuntimeError('changed frozen KOLF access protocol')
    write_json(pp,protocol)
    base.OUT=OUT;base.CONTEXTS=[CONTEXT];base.remote=remote;base.features=native_features
    base.CONFIG=dict(base.CONFIG,external_predictors=card)
    repair.OUT=OUT


def permitted_csc_column(io,group,pointer,column,row_codes):
    """Gather permitted bytes before decoding float counts, avoiding HDF5's
    quadratic large fancy-index selection. Unfiltered chunks are opaque cache.
    """
    begin,end=int(pointer[column]),int(pointer[column+1])
    locators=group['indices'][begin:end]  # storage row IDs; never model features
    keep=np.flatnonzero(row_codes[locators]>=0)
    positions=begin+keep;rows=locators[keep]
    data=group['data'];step=data.chunks[0];width=data.dtype.itemsize
    if data.id.get_create_plist().get_nfilters()!=0:raise RuntimeError('expected author unfiltered count chunks')
    values=np.empty(len(positions),data.dtype);chunk_ids=positions//step
    for chunk in np.unique(chunk_ids):
        use=np.flatnonzero(chunk_ids==chunk);info=data.id.get_chunk_info_by_coord((int(chunk)*step,))
        if info.byte_offset is None:raise RuntimeError('missing allocated raw count chunk')
        io.seek(int(info.byte_offset));opaque=io.read(int(info.size))
        offsets=(positions[use]-int(chunk)*step)*width
        byte_indices=offsets[:,None]+np.arange(width)
        selected=np.frombuffer(opaque,np.uint8)[byte_indices].copy()
        values[use]=selected.reshape(-1).view(data.dtype)
    if not np.isfinite(values).all() or (values<0).any() or not np.allclose(values,np.round(values),atol=1e-5):
        raise RuntimeError('allowed counts are not finite nonnegative raw counts')
    return rows,values


def prepare(roles):
    if roles==['confirmation']:
        freeze_path=OUT/'RISK_FREEZE.json'
        if not freeze_path.exists():raise RuntimeError('confirmation requires frozen risk scores')
        frozen=json.loads(freeze_path.read_text())
        if not frozen['passed_predictors']:raise RuntimeError('confirmation requires a qualified predictor')
        if sha(OUT/'FROZEN_CONFIRMATION_SCORES.parquet')!=frozen['confirmation_score_sha256']:
            raise RuntimeError('confirmation risk scores changed after freeze')
    name='CONFIRMATION' if roles==['confirmation'] else 'TRAIN_FEEDBACK'
    if (OUT/f'{name}_READ_RECEIPT.json').exists():return
    tasks=pd.read_parquet(OUT/'TASK_ROLES.parquet');chosen=tasks[tasks.role.isin(roles)].reset_index(drop=True)
    contract=json.loads((OUT/'OUTPUT_CONTRACT.json').read_text());cols=contract['target_column_indices']
    io=remote();start=time.monotonic()
    with h5py.File(io,'r') as h:
        names=h5_column(h['obs'],'gene_target').astype(str);ntc=names=='NTC'
        stable=h5_column(h['obs'],'gene_target_ensembl_id').astype(str)
        pairs=pd.DataFrame({'gene':names,'ensembl_id':stable}).drop_duplicates()
        pairs=pairs[pairs.gene.isin(tasks.gene)]
        if pairs.groupby('gene').ensembl_id.nunique().max()>1:raise RuntimeError('ambiguous perturbation stable IDs')
        if pairs.groupby('ensembl_id').gene.nunique().max()>1:raise RuntimeError('perturbation aliases cross frozen biological gene groups')
        pairs.to_parquet(OUT/'PERTURBATION_STABLE_IDS.parquet',index=False)
        # A deterministic NTC-only reference for the native control embedding.
        control_sample=np.flatnonzero(ntc)[:2048]
        control_row_to_sample=np.full(len(names),-1,int);control_row_to_sample[control_sample]=np.arange(len(control_sample))
        idx={g:i for i,g in enumerate(chosen.gene)};row_codes=pd.Series(names).map(idx).fillna(-1).to_numpy(int)
        row_codes[ntc]=len(chosen)
        # Future confirmation roles are absent from the row-code map.
        group=h['layers/counts'];ptr=group['indptr'][:]
        partial=OUT/f'{name}_POOLED_COUNTS_PARTIAL.npz'
        if partial.exists():
            z=np.load(partial);sums=z['sums'];done=z['done'].astype(bool);dropout=z['control_nonzero'];sampled=z['sampled_control_counts']
            if str(z['task_hash'])!=digest_ids(chosen.task_id):raise RuntimeError('partial extraction role identities differ')
        else:
            sums=np.zeros((len(chosen)+1,len(cols)),float);done=np.zeros(len(cols),bool);dropout=np.zeros(len(cols),int)
            sampled=np.zeros((len(control_sample),len(cols)),np.float32)
        for j,col in enumerate(cols):
            if done[j]:continue
            rows,values=permitted_csc_column(io,group,ptr,col,row_codes)
            np.add.at(sums[:,j],row_codes[rows],values);dropout[j]=int(np.sum(ntc[rows]&(values>0)));done[j]=True
            use=control_row_to_sample[rows]>=0;sampled[control_row_to_sample[rows[use]],j]=values[use]
            if (j+1)%10==0 or done.all():
                np.savez(partial,sums=sums,done=done,control_nonzero=dropout,sampled_control_counts=sampled,task_hash=digest_ids(chosen.task_id))
                write_json(OUT/'TARGET_READ_STATUS.json',{'status':'READING_PERMITTED_COUNTS','roles':roles,
                    'columns_complete':int(done.sum()),'columns_planned':len(cols),'wall_seconds':time.monotonic()-start,
                    'forbidden_count_values_decoded':0,'locator_indices_not_given_to_predictors':True,
                    'confirmation_role_numeric_read':roles==['confirmation'],'pid':os.getpid()})
                print(json.dumps({'allowed_roles':roles,'columns_complete':int(done.sum()),'output_genes':len(cols)}),flush=True)
        if (sums.sum(1)<=0).any():raise RuntimeError('empty biological task on frozen output panel')
        ctrl=np.log1p(1e4*sums[-1]/sums[-1].sum());effect=np.log1p(sums[:-1]*(1e4/sums[:-1].sum(1))[:,None])-ctrl
        chosen.to_parquet(OUT/f'{name}_TASKS.parquet',index=False);np.save(OUT/f'{name}_EFFECTS.npy',effect.astype(np.float32))
        control=OUT/'CONTROL_PANEL_REFERENCE.npz'
        payload={'genes':np.asarray(contract['gene_ids']),'baseline':ctrl,
                 'dropout':1-dropout/max(1,int(ntc.sum())),'n_controls':int(ntc.sum()),'sampled_control_counts':sampled}
        if control.exists():
            previous=np.load(control)
            for k,v in payload.items():np.testing.assert_array_equal(previous[k],v)
        else:np.savez(control,**payload)
        write_json(OUT/f'{name}_READ_RECEIPT.json',{'roles':roles,'query_rows':len(chosen),
            'permitted_cell_rows':int(np.sum(row_codes>=0)),'forbidden_count_values_decoded':0,
            'sparse_row_locators_read_for_role_selection_only':True,'confirmation_values_before_score_freeze':0,
            'effect_contract_sha256':sha(OUT/'OUTPUT_CONTRACT.json'),'effects_sha256':sha(OUT/f'{name}_EFFECTS.npy')})
    io.close()


def public():
    if (OUT/'PUBLIC_SNAPSHOT.json').exists():return
    tasks=pd.read_parquet(OUT/'TASK_ROLES.parquet');axis=json.loads((OUT/'OUTPUT_CONTRACT.json').read_text())
    stable=pd.read_parquet(OUT/'PERTURBATION_STABLE_IDS.parquet');by_symbol=stable.set_index('gene').ensembl_id.to_dict()
    keys=sorted(tasks.gene.unique());code={by_symbol[g]:i for i,g in enumerate(keys)}
    sums=None;counts=np.zeros(len(keys)+1,int)
    with h5py.File(base.GWPS,'r') as h:
        var=h5_column(h['var'],'ensembl_id').astype(str);vi={g:i for i,g in enumerate(var)}
        requested=np.asarray(axis['source_ensembl_ids'],dtype=str)
        present=np.asarray([g in vi for g in requested],dtype=bool)
        if int(present.sum()) < 2:
            raise RuntimeError('public common axis has fewer than two response coordinates')
        cols=np.asarray([vi[g] for g in requested[present]],dtype=int);ids=h5_column(h['obs'],'gene_id').astype(str)
        sums=np.zeros((len(keys)+1,int(present.sum())),float)
        adapter={'status':'FIXED_BEFORE_PREDICTOR_FIT_AND_CONFIRMATION',
            'reason':'34 frozen response IDs absent from actual GWPS expression matrix; no symbol aliases found',
            'main_predictor_output_axis_genes':len(requested),'public_comparison_axis_genes':int(present.sum()),
            'main_endpoint_or_competence_gate_changed':False,'axis_mask':present.tolist(),
            'missing_source_ensembl_ids':requested[~present].tolist(),
            'common_axis_ids_hash':digest_ids(requested[present]),'missing_genes_zero_filled':False,
            'public_normalization':'pooled raw counts on actual common axis, log1p CP10k minus NTC',
            'prediction_view':'inverse log of frozen delta+NTC, clip negative log levels at physical zero, renormalize treated and NTC on same common axis',
            'projection_does_not_modify_original_prediction':True,'projection_fraction_saved_as_diagnostic':True,
            'confirmation_truth_used':False}
        adapter_path=OUT/'PUBLIC_AXIS_ADAPTER_CARD.json'
        if adapter_path.exists() and json.loads(adapter_path.read_text())!=adapter:
            raise RuntimeError('public axis adapter changed within fixed run')
        write_json(adapter_path,adapter)
        nperts=h5_column(h['obs'],'nperts').astype(str);groups=pd.Series(ids).map(code).fillna(-1).to_numpy(int)
        groups[nperts!='1']=-1;groups[nperts=='0']=len(keys)
        matrix=h['X']
        for start in range(0,len(ids),4000):
            end=min(start+4000,len(ids));g=groups[start:end];ok=g>=0
            if not ok.any():continue
            block=np.asarray(matrix[start:end,:],float)[ok][:,cols]
            if (block<0).any() or not np.allclose(block,np.round(block),atol=1e-5):raise RuntimeError('source data is not raw counts')
            sort=np.argsort(g[ok],kind='stable');unique,first=np.unique(g[ok][sort],return_index=True)
            sums[unique]+=np.add.reduceat(block[sort],first,axis=0);counts+=np.bincount(g[ok],minlength=len(keys)+1)
    ctrl=np.log1p(1e4*sums[-1]/sums[-1].sum());ok=sums[:-1].sum(1)>0
    effect_common=np.full_like(sums[:-1],np.nan,dtype=np.float32)
    effect_common[ok]=np.log1p(sums[:-1][ok]*(1e4/sums[:-1][ok].sum(1))[:,None])-ctrl
    effect=np.full((len(keys),len(axis['gene_ids'])),np.nan,dtype=np.float32)
    effect[:,present]=effect_common
    np.savez(OUT/'PUBLIC_COUNTS_EFFECTS.npz',genes=np.asarray(keys),effects=effect,n_cells=counts[:-1],
        axis_mask=present,source_ensembl_ids=requested)
    write_json(OUT/'PUBLIC_SNAPSHOT.json',{'source':'Replogle K562 GWPS','target_study_rows_used':0,
        'n_supported_genes':int(ok.sum()),'normalization_denominator':'actual common1366 axis; predictor comparison view renormalized identically',
        'response_axis_hash':axis['gene_axis_hash'],'snapshot_sha256':sha(OUT/'PUBLIC_COUNTS_EFFECTS.npz'),
        'single_history_dispersion':0,'independent_unit':'one screen/background effect per perturbation',
        'n_requested_output_genes':int(len(requested)),'n_public_axis_genes':int(present.sum()),
        'missing_source_ensembl_ids':requested[~present].tolist(),'missing_genes_zero_filled':False,
        'public_axis_adapter_sha256':sha(OUT/'PUBLIC_AXIS_ADAPTER_CARD.json')})


def gene_features():
    if (OUT/'CONTROL_FEATURES.npz').exists() and (OUT/'NATIVE_CONTROL_FEATURES.npz').exists() and (OUT/'CONTROL_FEATURE_AUDIT.json').exists():return
    tasks=pd.read_parquet(OUT/'TASK_ROLES.parquet');ctrl=np.load(OUT/'CONTROL_PANEL_REFERENCE.npz')
    vocab=json.loads((CHECKPOINT/'vocab.json').read_text());state=torch.load(CHECKPOINT/'best_model.pt',map_location='cpu',weights_only=True)
    weights=state['encoder.embedding.weight'].numpy();genes=sorted(set(tasks.gene)|set(ctrl['genes']))
    known=[g for g in genes if g in vocab];lookup={g:i for i,g in enumerate(known)}
    train=[g for g in tasks[tasks.role.eq('predictor_train')].gene if g in lookup]
    if len(train)<100:raise RuntimeError('insufficient pretrained gene-identity coverage')
    pca=PCA(n_components=50,svd_solver='full').fit(weights[[vocab[g] for g in train]])
    emb=pca.transform(weights[[vocab[g] for g in known]]).astype(np.float32)
    baseline=np.full((1,len(known)),np.nan);dropout=baseline.copy();variance=baseline.copy();ci={g:i for i,g in enumerate(ctrl['genes'])}
    for g,i in lookup.items():
        if g in ci:baseline[0,i]=ctrl['baseline'][ci[g]];dropout[0,i]=ctrl['dropout'][ci[g]]
    np.savez(OUT/'CONTROL_FEATURES.npz',genes=np.asarray(known),embedding=emb,baseline=baseline,dropout=dropout,donor_var=variance)
    np.savez(OUT/'FROZEN_GENE_PCA.npz',mean=pca.mean_,components=pca.components_)
    control_counts=ctrl['sampled_control_counts'].astype(float);valid=control_counts.sum(1)>0
    normalized=np.log1p(control_counts[valid]*(1e4/control_counts[valid].sum(1))[:,None])
    matrix=normalized.T;matrix=(matrix-matrix.mean(1,keepdims=True))/(matrix.std(1,keepdims=True)+1e-8)
    native_embedding=TruncatedSVD(n_components=50,random_state=0).fit_transform(matrix)
    np.savez(OUT/'NATIVE_CONTROL_FEATURES.npz',genes=ctrl['genes'],embedding=native_embedding)
    write_json(OUT/'CONTROL_FEATURE_AUDIT.json',{'predictor_features':'frozen pretrained gene embedding; training-gene-only PCA50',
        'these_are_not_control_expression_SVD_embeddings':True,'donor_variance_missing_not_zero':True,
        'perturbed_query_expression_used':False,'checkpoint_finetuned':False,
        'query_gene_coverage':float(tasks.gene.isin(known).mean()),'pca_training_gene_hash':digest_ids(train),
        'risk_native_embedding':'separate NTC-only expression SVD50, not pretrained predictor features',
        'native_control_gene_coverage':float(tasks.gene.isin(ctrl['genes']).mean()),
        'native_embedding_missing_query_genes_not_zero_filled':True})


def native_features(tasks,prediction):
    f=ORIGINAL_FEATURES(tasks,prediction);z=np.load(OUT/'NATIVE_CONTROL_FEATURES.npz')
    index={g:i for i,g in enumerate(z['genes'])};emb=np.full((len(tasks),50),np.nan)
    for i,g in enumerate(tasks.gene):
        if g in index:emb[i]=z['embedding'][index[g]]
    for j in range(50):f[f'native_embedding_{j}']=emb[:,j]
    return f


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','complete'],default='complete');a=ap.parse_args()
    configure();prepare(['predictor_train','feedback']);public();gene_features()
    if a.phase=='complete':
        repair.fit();base.qualification();base.risk_freeze()
        if (OUT/'RISK_FREEZE.json').exists():
            # Risk freeze precedes any confirmation count value decoding.
            frozen=json.loads((OUT/'RISK_FREEZE.json').read_text())
            if sha(OUT/'FROZEN_CONFIRMATION_SCORES.parquet')!=frozen['confirmation_score_sha256']:raise RuntimeError('changed risk scores')
            write_json(OUT/'CONFIRMATION_EVALUATION_OPEN_EVENT.json',{'authorized_by':'user v2.1 frozen qualified confirmation',
                'risk_freeze_sha256':sha(OUT/'RISK_FREEZE.json'),'scope':'registered1400-gene source-defined panel',
                'time_utc':pd.Timestamp.now(tz='UTC').isoformat()})
            prepare(['confirmation']);base.confirmation()


if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(OUT/'FAILURE_RECEIPT.json',{'status':'ENGINEERING_OR_RESOURCE_FAILURE','error_type':type(e).__name__,
            'error':str(e),'confirmation_opened':(OUT/'CONFIRMATION_EVALUATION_OPEN_EVENT.json').exists()})
        raise
