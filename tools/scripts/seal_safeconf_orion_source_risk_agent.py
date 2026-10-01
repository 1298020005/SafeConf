#!/usr/bin/env python3
"""Pretruth inference with fixed Source risk models and same-gene Source history.

Inputs are frozen published DELTA predictions, query identities and TRAIN NTC
means. No query truth, target error/CDF or fitting API is accepted. Missing
history leaves public features/risks unavailable; P-only risks stay separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC

SOURCE=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_source_core_20261002_v1')
DOC=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/risk_seal'
SCHEMA='safeconf_orion_pretruth_source_risk_seal_v1'
METHOD_SHA='6b8a26939c8c6134dbf2e8294ee7905fccc2122b570b046f3183a3d2f5caba97'
PAIR=['log_source_cells','log_source_batches','control_rmse','control_cosine',
      'source_effect_magnitude','source_effect_abs_mean','source_conflict','support_fraction','quality_missing']
QUERY=['query_id','target_gene_id','target_gene_symbol','context_id','role']
METHODS=[(ref,kind) for ref in ('P_only','Manual','Learned') for kind in ('ridge','hgb')]
PIN_FILES=['SOURCE_FEATURE_MANIFEST.json','MODEL_MANIFEST.json','SOURCE_PUBLIC_MEMORY_METADATA.parquet',
           'SOURCE_PUBLIC_EFFECTS.npy','SOURCE_PUBLIC_CONTROLS.npy','GENE_MANIFEST.csv','GENE_IDS.json','SOURCE_ERROR_CDF.json']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def source_core(root=SOURCE):
    root=Path(root)
    feature=json.loads((root/'SOURCE_FEATURE_MANIFEST.json').read_text())
    manifest=json.loads((root/'MODEL_MANIFEST.json').read_text())
    if feature.get('risk_P_columns')!=P or feature.get('risk_public_columns')!=PUBLIC or feature.get('public_pair_columns')!=PAIR:
        raise RuntimeError('Frozen Source feature contract differs; no replacement learner allowed')
    hashes={x['path']:x for x in json.loads((root/'ARTIFACT_HASHES.json').read_text())}
    paths=PIN_FILES+[x['path'] for x in manifest['models']]
    bindings=[]
    for name in paths:
        path=root/name; expected=hashes.get(name,{})
        if path.name!=name or expected.get('sha256')!=sha(path) or expected.get('bytes')!=path.stat().st_size:
            raise RuntimeError('Frozen Source artifact SHA/size mismatch')
        bindings.append({'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)})
    models={}; public=None
    for entry in manifest['models']:
        obj=joblib.load(root/entry['path'])
        if obj.columns!=entry['columns'] or len(obj.preprocessor.medians_)!=len(obj.columns) or obj.model.n_features_in_!=2*len(obj.columns):
            raise RuntimeError('Persisted Source preprocessor/model feature width mismatch')
        if entry['sha256']!=sha(root/entry['path']):raise RuntimeError('Source model SHA differs')
        if entry['role']=='Source_risk':
            key=(entry['reference'],entry['learner'])
            expected=P if key[0]=='P_only' else P+PUBLIC
            if key not in METHODS or obj.columns!=expected or entry['target']!='Source_per_upstream_per_context_midrank_CDF':
                raise RuntimeError('Frozen risk model feature/target differs')
            models[key]=obj
        elif entry['role']=='Public_biology_retrieval':
            if obj.columns!=PAIR or entry['target']!='biological_transfer_rmse' or entry.get('predict_clip') is not False:
                raise RuntimeError('Public learner requires exact9features and continuous transfer RMSE, clip=False')
            public=obj
        else:raise RuntimeError('Unknown Source learner role')
    if set(models)!=set(METHODS) or public is None:raise RuntimeError('Exactly six frozen Source risks and one Public model required')
    axis=pd.read_csv(root/'GENE_MANIFEST.csv')
    if len(axis)!=3285 or not np.array_equal(axis.axis_index,np.arange(3285)) or not axis.gene_name.is_unique:
        raise RuntimeError('Exact frozen Source3285 axis required')
    memory=pd.read_parquet(root/'SOURCE_PUBLIC_MEMORY_METADATA.parquet')
    effects=np.load(root/'SOURCE_PUBLIC_EFFECTS.npy',mmap_mode='r',allow_pickle=False)
    controls=np.load(root/'SOURCE_PUBLIC_CONTROLS.npy',mmap_mode='r',allow_pickle=False)
    if effects.shape!=controls.shape or effects.shape!=(len(memory),3285) or not memory.effect_vector_row.is_unique:
        raise RuntimeError('Source historical vector/metadata alignment differs')
    required={'experiment_id','perturbation_target','perturbation_type','effect_vector_row','n_cells','n_batches','eligibility'}
    if not required<=set(memory.columns):raise RuntimeError('Source history schema omits required Source-specific evidence')
    return {'root':root,'models':models,'public':public,'axis':axis,'memory':memory,
            'effects':effects,'controls':controls,'bindings':bindings}


def cosine(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denom) if denom>1e-12 else 0.


def prediction_features(delta):
    delta=np.asarray(delta,float)
    if delta.ndim!=2 or not np.isfinite(delta).all():raise RuntimeError('Finite task-by-gene DELTA predictions required')
    return pd.DataFrame({'predicted_magnitude':np.sqrt(np.mean(delta**2,axis=1)),
        'prediction_abs_mean':np.mean(np.abs(delta),axis=1),'prediction_signed_mean':np.mean(delta,axis=1),
        'prediction_std':np.std(delta,axis=1),'prediction_abs_q95':np.quantile(np.abs(delta),.95,axis=1),
        'prediction_sparsity':np.mean(np.abs(delta)<=1e-8,axis=1)})[P]


def query_pairs(queries,train_controls,memory,effects,source_controls):
    """The API has no query-truth argument and never manufactures labels."""
    if list(queries.columns)!=QUERY or not queries.query_id.is_unique:
        raise RuntimeError('Queries must contain exactly frozen identity metadata')
    if not set(queries.context_id)<={'HCT116','HEK293T'} or not set(queries.role)<={'VALIDATION','TEST'}:
        raise RuntimeError('Only registered published held-out query identities allowed')
    eligible=memory[memory.eligibility.eq(True)&memory.perturbation_type.eq('genetic_single_gene')]
    by_gene={g:group for g,group in eligible.groupby('perturbation_target',sort=False)}
    rows=[]; history=np.zeros(len(queries),dtype=np.int64)
    for q,task in enumerate(queries.itertuples(index=False)):
        control=np.asarray(train_controls.get(task.context_id),dtype=float)
        if control.shape!=(effects.shape[1],) or not np.isfinite(control).all() or (control<0).any():
            raise RuntimeError('Each query requires finite own-context TRAIN NTC control')
        group=by_gene.get(task.target_gene_symbol)
        if group is None:continue
        indices=group.effect_vector_row.to_numpy(int)
        if (indices<0).any() or (indices>=len(effects)).any():raise RuntimeError('Source memory row index out of range')
        vectors=np.asarray(effects[indices],float); ctrls=np.asarray(source_controls[indices],float)
        if not np.isfinite(vectors).all() or not np.isfinite(ctrls).all():raise RuntimeError('Missing Source vector measurement')
        cells=group.n_cells.to_numpy(float); batches=group.n_batches.to_numpy(float)
        if not np.isfinite(cells).all() or (cells<=0).any() or not np.isfinite(batches).all() or (batches<0).any():
            raise RuntimeError('Source cell/batch support is missing; no zero substitution')
        center=vectors.mean(axis=0); conflict=np.sqrt(np.mean((vectors-center)**2,axis=1))
        history[q]=len(group)
        for local,item in enumerate(group.itertuples(index=False)):
            rows.append({'query_row':q,'query_id':task.query_id,'gene':task.target_gene_symbol,
                'context':task.context_id,'memory_id':item.experiment_id,'memory_row':int(item.effect_vector_row),
                'log_source_cells':float(np.log1p(cells[local])),
                'log_source_batches':float(np.log1p(batches[local])),
                'control_rmse':float(np.sqrt(np.mean((ctrls[local]-control)**2))),
                'control_cosine':cosine(ctrls[local],control),
                'source_effect_magnitude':float(np.sqrt(np.mean(vectors[local]**2))),
                'source_effect_abs_mean':float(np.mean(np.abs(vectors[local]))),
                'source_conflict':float(conflict[local]),'support_fraction':float(cells[local]/cells.sum()),
                'quality_missing':1.})
    columns=['query_row','query_id','gene','context','memory_id','memory_row']+PAIR
    return pd.DataFrame(rows,columns=columns),history


def weighted_prior(group,effects,mode):
    vectors=np.asarray(effects[group.memory_row.to_numpy(int)],float)
    cells=np.expm1(group.log_source_cells.to_numpy(float)); cells/=cells.sum()
    if mode=='Uniform':weights=np.full(len(group),1/len(group))
    elif mode=='Manual':weights=cells
    elif mode=='Learned':
        score=group.transfer_score.to_numpy(float)
        spread=max(float(np.std(score)),1e-8)
        learned=np.exp(np.clip(-(score-score.min())/spread,-20,20)); learned/=learned.sum()
        weights=.5*learned+.5*cells
    else:raise RuntimeError('Only fixed Source prior rules allowed')
    weights/=weights.sum()
    prior=np.average(vectors,axis=0,weights=weights)
    uncertainty=float(np.sqrt(np.average(np.mean((vectors-prior)**2,axis=1),weights=weights)))
    effective=float(1/np.sum(weights**2))
    return prior,weights,uncertainty,effective


def infer(queries,delta,train_controls,core):
    if np.asarray(delta).shape!=(len(queries),len(core['axis'])):
        raise RuntimeError('Prediction/query/frozen gene axis shapes differ')
    base=prediction_features(delta)
    pairs,history=query_pairs(queries,train_controls,core['memory'],core['effects'],core['controls'])
    if len(pairs):
        pairs['transfer_score']=core['public'].predict(pairs,clip=False)
        if not np.isfinite(pairs.transfer_score).all():raise RuntimeError('Public transfer-RMSE prediction nonfinite')
    else:pairs['transfer_score']=pd.Series(dtype=float)
    priors={ref:np.full_like(np.asarray(delta,float),np.nan) for ref in ('Uniform','Manual','Learned')}
    features={ref:base.copy() for ref in priors}
    for frame in features.values():
        for col in PUBLIC:frame[col]=np.nan
    weight_records=[]
    for query_row,group in pairs.groupby('query_row',sort=True):
        query_row=int(query_row)
        for ref in priors:
            prior,weights,uncertainty,effective=weighted_prior(group,core['effects'],ref)
            priors[ref][query_row]=prior
            frame=features[ref]
            frame.loc[query_row,PUBLIC]=[float(np.sqrt(np.mean(prior**2))),
                float(np.sqrt(np.mean((delta[query_row]-prior)**2))),cosine(delta[query_row],prior),
                uncertainty,float(np.log1p(np.expm1(group.log_source_cells).sum())),effective,
                float(group.source_conflict.mean())]
            for memory_id,weight in zip(group.memory_id,weights):
                weight_records.append({'query_id':queries.iloc[query_row].query_id,'reference':ref,
                                       'memory_id':memory_id,'weight':float(weight)})
    scores=queries.copy()
    scores['source_history_n']=history
    scores['history_status']=np.where(history>0,'KNOWN_SAME_GENE_SOURCE_HISTORY','NO_SAME_GENE_SOURCE_HISTORY')
    scores['Magnitude']=base.predicted_magnitude
    valid=history>0
    for ref,kind in METHODS:
        name=f'{ref}_{kind}'
        values=np.full(len(queries),np.nan)
        if ref=='P_only':
            values=core['models'][(ref,kind)].predict(base)
            scores[name+'_status']='OK_PREDICTION_ONLY'
        else:
            if valid.any():values[valid]=core['models'][(ref,kind)].predict(features[ref].loc[valid].reset_index(drop=True))
            scores[name+'_status']=np.where(valid,'OK_SOURCE_PUBLIC_HISTORY','NO_HISTORY_PUBLIC_RISK_UNSUPPORTED')
        scores[name]=values
    for ref in ('Manual','Learned'):
        frame=features[ref]
        scores[ref+'_WeightedHistoryDistance']=np.sqrt(frame.prediction_prior_rmse**2+frame.prior_uncertainty**2)
    for ref in priors:scores[ref+'_DirectRMSE']=features[ref].prediction_prior_rmse
    return {'scores':scores,'P_features':base,'reference_features':features,'pairs':pairs,
            'priors':priors,'weights':pd.DataFrame(weight_records,columns=['query_id','reference','memory_id','weight'])}


def frozen_manifest(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_mode&0o222:raise RuntimeError('Read-only frozen pretest comparison manifest required')
    manifest=json.loads(path.read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('risk_code_sha256')!=sha(__file__):
        raise RuntimeError('Pretest manifest schema/code SHA differs')
    if not manifest.get('pretruth_inference_authorized') or manifest.get('target_errors_or_CDF_used') is not False:
        raise RuntimeError('Explicit pretruth-only inference authorization required')
    contract=Path(manifest.get('scientific_contract_path',''))
    if not contract.is_file() or contract.stat().st_mode&0o222 or sha(contract)!=METHOD_SHA or manifest.get('scientific_contract_sha256')!=METHOD_SHA:
        raise RuntimeError('Exact frozen prospective scientific contract required')
    role_registry=Path(manifest.get('data_role_registry_path',''))
    if not role_registry.is_file() or sha(role_registry)!=manifest.get('data_role_registry_sha256'):
        raise RuntimeError('Observed TEST metadata exposure/truth role registry pin required')
    return manifest


def context_bindings(predictions_root,control_root,context_id,model_root=None):
    """Hash published predictions and TRAIN-control sidecars, never query truth."""
    pred=Path(predictions_root); ctr=Path(control_root)
    model=Path(model_root) if model_root is not None else pred
    if context_id not in ('HCT116','HEK293T'):raise RuntimeError('Registered context required')
    paths={'delta':pred/context_id/'PREDICTIONS_DELTA.tsv.gz',
           'queries':pred/context_id/'QUERY_IDENTITIES.tsv',
           'axis':pred/context_id/'OUTPUT_GENE_AXIS.tsv',
           'train_control':ctr/context_id/'OWN_CONTEXT_NTC_MEAN.tsv',
           'upstream_model':model/context_id/'MODEL.rds',
           'upstream_prediction_manifest':pred/'RUN_MANIFEST.tsv',
           'upstream_status':pred/context_id/'STATUS.tsv'}
    out={'context_id':context_id,'prediction_estimand':'DELTA_CP4000_log1p_own_TRAIN_NTC'}
    for role,path in paths.items():
        out[role+'_path']=str(path.resolve()); out[role+'_sha256']=sha(path)
    return out


def load_inputs(manifest,core):
    queries=[]; predictions=[]; controls={}; bindings=[]
    for context in manifest['contexts']:
        label=context['context_id']
        if label not in ('HCT116','HEK293T') or label in controls:raise RuntimeError('Unique registered own-context inputs required')
        for name in ['delta','queries','axis','train_control','upstream_model','upstream_prediction_manifest','upstream_status']:
            path=Path(context[name+'_path'])
            if sha(path)!=context[name+'_sha256']:raise RuntimeError('Frozen LM prediction/control/parameter SHA differs')
            bindings.append({'role':name,'context':label,'path':str(path.resolve()),'sha256':sha(path)})
        q=pd.read_csv(context['queries_path'],sep='\t',keep_default_na=False)
        if list(q.columns)!=QUERY or not q.context_id.eq(label).all() or not set(q.role)<={'VALIDATION','TEST'}:
            raise RuntimeError('Queries require published pretruth identity metadata only')
        if Path(context['delta_path']).name!='PREDICTIONS_DELTA.tsv.gz' or context.get('prediction_estimand')!='DELTA_CP4000_log1p_own_TRAIN_NTC':
            raise RuntimeError('Published DELTA estimand required; treated predictions cannot be substituted')
        status_table=pd.read_csv(context['upstream_status_path'],sep='\t',keep_default_na=False)
        if status_table.columns.tolist()!=['key','value'] or not status_table.key.is_unique:
            raise RuntimeError('Published upstream status contract differs')
        status=dict(zip(status_table.key,status_table.value))
        for key,value in {'status':'PASS','context_id':label,'pca_dim':'10','ridge_penalty':'0.1','seed':'1',
                          'query_truth_read':'FALSE','input_estimand':'mean_cell_log1p_cp4000_v1','endpoint_gene_count':'3285'}.items():
            if str(status.get(key))!=value:raise RuntimeError('Published fixed upstream parameters/pretruth status differ')
        if int(status.get('query_count','-1'))!=len(q):raise RuntimeError('Published query count differs')
        upstream_manifest=pd.read_csv(context['upstream_prediction_manifest_path'],sep='\t',keep_default_na=False)
        upstream_row=upstream_manifest[upstream_manifest.context_id.eq(label)]
        if len(upstream_row)!=1 or upstream_row.iloc[0].model_sha256!=context['upstream_model_sha256']:
            raise RuntimeError('Published fitted model hash does not match upstream prediction manifest')
        axis=pd.read_csv(context['axis_path'],sep='\t',keep_default_na=False)
        if not np.array_equal(axis.gene_id,core['axis'].orion_ensembl_id) or not np.array_equal(axis.gene_symbol,core['axis'].gene_name):
            raise RuntimeError('Published LM output axis differs from fixed Source3285')
        delta=pd.read_csv(context['delta_path'],sep='\t',keep_default_na=False)
        if delta.columns.tolist()!=['gene_id']+q.query_id.tolist() or not np.array_equal(delta.gene_id,axis.gene_id):
            raise RuntimeError('Published DELTA/query order differs')
        pred=delta.drop(columns='gene_id').to_numpy(float).T
        control=pd.read_csv(context['train_control_path'],sep='\t',keep_default_na=False)
        if control.columns.tolist()!=['gene_id','mean_cell_logCP4000'] or not control.gene_id.is_unique:
            raise RuntimeError('TRAIN NTC input must contain only measured gene means')
        lookup=control.set_index('gene_id').mean_cell_logCP4000
        if not set(axis.gene_id)<=set(lookup.index):raise RuntimeError('Missing measured TRAIN control gene cannot be zero filled')
        controls[label]=lookup.loc[axis.gene_id].to_numpy(float)
        queries.append(q); predictions.append(pred)
    return pd.concat(queries,ignore_index=True),np.concatenate(predictions,axis=0),controls,bindings


def seal(manifest_path,output):
    manifest=frozen_manifest(manifest_path)
    core=source_core(manifest.get('source_core_root',str(SOURCE)))
    if manifest.get('source_core_bindings')!=core['bindings']:
        raise RuntimeError('Exact Source core parameter/array/CDF pins differ')
    queries,delta,controls,lm_bindings=load_inputs(manifest,core)
    output=Path(output)
    if output.exists():raise RuntimeError('New immutable risk seal output version required')
    stage=output.with_name(output.name+'.incomplete'); stage.mkdir(parents=True)
    started=time.monotonic(); result=infer(queries,delta,controls,core)
    result['scores'].to_csv(stage/'PRETRUTH_RISKS.tsv',sep='\t',index=False,float_format='%.17g',na_rep='NaN')
    result['P_features'].to_parquet(stage/'P_ONLY_FEATURES.parquet',index=False)
    result['pairs'].to_parquet(stage/'SOURCE_HISTORY_PAIR_FEATURES.parquet',index=False)
    result['weights'].to_csv(stage/'SOURCE_PRIOR_WEIGHTS.tsv',sep='\t',index=False,float_format='%.17g')
    for ref,frame in result['reference_features'].items():
        frame.to_parquet(stage/f'{ref}_P_PUBLIC_FEATURES.parquet',index=False)
        np.save(stage/f'{ref}_PRIOR_EFFECTS.npy',result['priors'][ref],allow_pickle=False)
    report={'schema':SCHEMA,'status':'SEALED_PRETRUTH','comparison_manifest_sha256':sha(manifest_path),
        'scientific_contract_sha256':METHOD_SHA,
        'risk_code_sha256':sha(__file__),'source_core_bindings':core['bindings'],'LM_prediction_and_control_bindings':lm_bindings,
        'n_queries':len(queries),'known_history_queries':int((result['scores'].source_history_n>0).sum()),
        'source_only_risk_models':6,'public_biology_target':'continuous_transfer_RMSE_predict_clip_False',
        'pair_features':PAIR,'P_features':P,'P_public_features':P+PUBLIC,
        'history_retrieval':'exact same perturbation_target symbol among frozen eligible Source genetic_single_gene records only',
        'learned_prior_rule':'Source: exp(-(score-min)/max(std,1e-8)), logits clipped[-20,20], then0.5learned+0.5cellweights',
        'no_history_public_policy':'all seven public features and prior vectors NaN; four public risk scores NaN with explicit UNSUPPORTED; two P-only risks remain separate',
        'fallback_claim':False,'query_truth_input_or_dummy_labels':False,'query_errors_used':False,'target_CDF_fit':False,
        'new_learners_or_parameter_fits':0,'identity_features_used':False,'target_probability_calibration_claim':False,
        'train_control_source':'own-context authorized TRAIN NTC; no VAL/TEST biology reader',
        'data_role_registry_sha256':manifest['data_role_registry_sha256'],
        'TEST_identity_metadata_status':'SEEN_METADATA_ONLY','TEST_truth_status':'CLOSED',
        'whole_method_pristine_claim':False,
        'replication_scope':'prospective truth-blind external-study replication under frozen scientific rules',
        'elapsed_seconds':round(time.monotonic()-started,3)}
    write_json(stage/'RISK_SEAL_MANIFEST.json',report)
    artifacts=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(stage.iterdir()) if p.is_file()]
    write_json(stage/'ARTIFACT_HASHES.json',artifacts); stage.rename(output)
    return report


def synthetic_test(output):
    """Hand-derived vectors plus independent original Source prior oracle."""
    from tools.scripts import run_dual_memory_txpert_public_biology as oracle
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    core=source_core(); n=len(core['axis'])
    memory=pd.DataFrame({'experiment_id':['SYNTHETIC_A','SYNTHETIC_B'],'perturbation_target':['KNOWN','KNOWN'],
        'perturbation_type':['genetic_single_gene']*2,'effect_vector_row':[0,1],'n_cells':[2,8],
        'n_batches':[1,2],'eligibility':[True,True]})
    fixture=dict(core,memory=memory,effects=np.asarray([np.ones(n),np.full(n,3.)]),
                 controls=np.asarray([np.full(n,.1),np.full(n,.2)]))
    queries=pd.DataFrame([['Q_K','E_K','KNOWN','HCT116','VALIDATION'],
                          ['Q_N','E_N','NO_HISTORY','HCT116','TEST']],columns=QUERY)
    delta=np.full((2,n),2.)
    result=infer(queries,delta,{'HCT116':np.full(n,.15)},fixture)
    group=result['pairs']; assert group.columns.tolist()==['query_row','query_id','gene','context','memory_id','memory_row']+PAIR+['transfer_score']
    assert set(PAIR)==set(oracle.PAIR_FEATURES) and PAIR==oracle.PAIR_FEATURES
    hand_pairs=np.asarray([[np.log(3),np.log(2),.05,1,1,1,1,.2,1],
                           [np.log(9),np.log(3),.05,1,3,3,1,.8,1]])
    assert np.allclose(group[PAIR].to_numpy(float),hand_pairs,rtol=0,atol=1e-12)
    assert np.array_equal(result['P_features'].to_numpy(float),np.tile([2.,2.,2.,0.,2.,0.],(2,1)))
    assert np.array_equal(group.log_source_batches,np.log1p([1.,2.]))
    assert np.array_equal(group.source_effect_abs_mean,[1.,3.])
    assert np.allclose(group.source_conflict,[1.,1.]) and np.allclose(group.support_fraction,[.2,.8])
    for ref,mode in [('Uniform','uniform'),('Manual','cells'),('Learned','learned_regularized')]:
        actual=weighted_prior(group,fixture['effects'],ref)
        expected=oracle.prior_from_scores(group,fixture['effects'],'transfer_score',mode)
        for a,b in zip(actual,expected):assert np.allclose(a,b,rtol=0,atol=1e-15)
        assert np.array_equal(result['priors'][ref][0],actual[0])
        assert np.isnan(result['priors'][ref][1]).all()
        assert result['reference_features'][ref].columns.tolist()==P+PUBLIC
        assert np.isnan(result['reference_features'][ref].loc[1,PUBLIC].to_numpy(float)).all()
    assert np.allclose(result['priors']['Manual'][0],2.6)
    assert np.allclose(result['priors']['Uniform'][0],2.)
    assert np.allclose(result['reference_features']['Manual'].loc[0,'prior_uncertainty'],.8)
    assert np.allclose(result['scores'].loc[0,'Manual_WeightedHistoryDistance'],1.)
    for ref,kind in METHODS:
        values=result['scores'][f'{ref}_{kind}'].to_numpy(float)
        assert np.isfinite(values[0])
        assert (np.isfinite(values[1]) if ref=='P_only' else np.isnan(values[1]))
    assert result['scores'].loc[1,'Manual_ridge_status']=='NO_HISTORY_PUBLIC_RISK_UNSUPPORTED'
    tied=group.copy(); tied['transfer_score']=1.
    tied_actual=weighted_prior(tied,fixture['effects'],'Learned')
    tied_expected=oracle.prior_from_scores(tied,fixture['effects'],'transfer_score','learned_regularized')
    for a,b in zip(tied_actual,tied_expected):assert np.allclose(a,b,rtol=0,atol=1e-15)
    assert np.allclose(tied_actual[0],2.3)
    result['scores'].to_csv(output/'SYNTHETIC_RISK_SCORES.tsv',sep='\t',index=False,float_format='%.17g')
    result['pairs'].to_csv(output/'SYNTHETIC_PAIR_FEATURES.tsv',sep='\t',index=False,float_format='%.17g')
    report={'schema':SCHEMA,'status':'SYNTHETIC_SOURCE_RISK_PRETRUTH_PASS','code_sha256':sha(__file__),
      'source_core_bindings':core['bindings'],'pair_feature_width':9,'P_width':6,'P_public_width':13,
      'actual_frozen_risk_models_exercised':6,'actual_public_biology_model_exercised':True,
      'manual_uniform_learned_prior_matches_original_Source_oracle':True,
      'all_nine_pair_features_match_hand_expected':True,'all_six_prediction_features_match_hand_expected':True,
      'zero_score_spread_fixed_std_floor_and_50_50_blend_match_Source':True,
      'hand_expected_manual_prior':2.6,'hand_expected_uniform_prior':2.,'hand_expected_manual_uncertainty':.8,
      'hand_expected_manual_history_distance':1.,'no_history_prior_public_features_public_risks_NaN':True,
      'no_history_P_only_risks_separate_and_finite':True,'query_truth_or_dummy_labels_used':False,
      'actual_Orion_expression_read':False,'actual_Orion_prediction_read':False,'new_fits':0}
    write_json(output/'SYNTHETIC_RISK_PROOF.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_core_bindings'},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    synthetic=sub.add_parser('synthetic-test'); synthetic.add_argument('--output',type=Path,default=DOC/'synthetic_v1')
    pins=sub.add_parser('pin-source'); pins.add_argument('--source-core',type=Path,default=SOURCE); pins.add_argument('--output',type=Path,required=True)
    bind=sub.add_parser('bind-contexts')
    bind.add_argument('--source-core',type=Path,default=SOURCE)
    bind.add_argument('--predictions-root',type=Path,required=True); bind.add_argument('--control-root',type=Path,required=True)
    bind.add_argument('--model-root',type=Path); bind.add_argument('--output',type=Path,required=True)
    actual=sub.add_parser('seal'); actual.add_argument('--comparison-manifest',type=Path,required=True); actual.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='synthetic-test':synthetic_test(args.output)
    elif args.command in ('pin-source','bind-contexts'):
        core=source_core(args.source_core)
        registry=DOC.parent/'ORION_DATA_ROLE_REGISTRY.csv'
        payload={'schema':SCHEMA,'risk_code_sha256':sha(__file__),
            'scientific_contract_path':str(DOC.parent/'ORION_METHOD_CONTRACT.json'),'scientific_contract_sha256':METHOD_SHA,
            'data_role_registry_path':str(registry),'data_role_registry_sha256':sha(registry) if registry.is_file() else None,
            'source_core_root':str(core['root']),'source_core_bindings':core['bindings'],'pretruth_inference_authorized':False,
            'target_errors_or_CDF_used':False,'contexts':[]}
        if args.command=='bind-contexts':
            payload['contexts']=[context_bindings(args.predictions_root,args.control_root,c,args.model_root) for c in ('HCT116','HEK293T')]
        write_json(args.output,payload)
    else:print(json.dumps(seal(args.comparison_manifest,args.output),indent=2))


if __name__=='__main__':main()
