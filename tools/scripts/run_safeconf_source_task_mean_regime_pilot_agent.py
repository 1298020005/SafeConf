#!/usr/bin/env python3
"""One Root-registered SourceDEV task-mean predictor; no risk fit or Orion read."""
from __future__ import annotations
import os
for _thread_key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_key]='4'
import argparse, hashlib, importlib.util, json, math, resource, signal, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001'
SOURCE=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_source_core_20261002_v1')
OUT=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/source_task_mean_regime_pilot_20261002_v1')
REPORT=DOC/'orion_preparation/source_task_mean_regime_pilot_v1'
PREP=DOC/'orion_preparation/source_task_mean_pilot_preparation_v1/PREPARATION.json'
RECIPE=PREP.parent/'CODE_PROPOSAL.py'
HELPER=ROOT/'tools/scripts/seal_safeconf_orion_source_risk_agent.py'
SCHEMA='safeconf_source_mean_treated_state_pilot_scope_v1'
MODEL_ID='SourceTaskMean_CP4000_3285_5GeneFold'
PINS={PREP:'0f9f5ac9963817e213096f9de1f82be85124c51dfd5196b323d2428bde169c55',
      RECIPE:'74698d54f19d1358f6df57074576af76001996dd317c0d4751038874db1a380c',
      HELPER:'8b69e6ce29dc92aa34150b3b08aac62ce8333a10f84ec432a39d0e81ffc64482',
      SOURCE/'SOURCE_FEATURE_MANIFEST.json':'b74b9dfbc64f5acc82b7b0d34023aa6a523bd11d1904ced7fd638c5bec4ef2d1',
      SOURCE/'MODEL_MANIFEST.json':'c8d9d5003860d8a5e117398a7d44b5c67cfb700c6fb56f37bcf04a5d549909a6',
      SOURCE/'SPLIT_MANIFEST.csv':'350e53a18f54108eef56062b96875f5ca0ad3d1095eb155aa2303eb35422aa71'}
CONTEXTS=('K562','RPE1','hepg2','jurkat')
META=['task_id','gene','target','fold']
QUANTILES=(0,.01,.05,.25,.5,.75,.95,.99,1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def binding(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def write_json(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def exact(item,path):
    actual=binding(path)
    if actual!=item:raise RuntimeError('Exact binding differs: '+str(path))
    return actual

def memory_check():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1024**3:raise RuntimeError('Source pilot1GiB RSS ceiling exceeded')

def import_fixed(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def authorize(scope_path):
    scope_path=Path(scope_path).resolve()
    if scope_path.stat().st_mode & 0o222:raise RuntimeError('Immutable Root scope contract required')
    scope=json.loads(scope_path.read_text())
    required={'schema':SCHEMA,'root_authorized':True,'Source_role':'DEV_SEEN','formal_upstream_attempt_index':2,
      'maximum_new_formal_upstream_attempts':2,'new_shared_risk_fits':0,'new_Source_CDF_fits':0,
      'Orion_numeric_access':False,'Orion_outcome_use':False,'Source_raw_X_access':False,
      'Source_mean_convention':'MEAN_TREATED_STATE_MINUS_QUERY_MATCHED_NC','CPU_threads':4,
      'max_elapsed_seconds':300,'max_RSS_bytes':1024**3,'GPU_hours':0,'new_download_bytes':0,
      'prediction_seal_before_heldout_error_evaluation':True}
    if any(scope.get(k)!=v for k,v in required.items()):raise RuntimeError('Fixed Source-only scope/attempt/recipe differs')
    if scope.get('source_root')!=str(SOURCE) or scope.get('output_root')!=str(OUT) or scope.get('report_root')!=str(REPORT):raise RuntimeError('Exact isolated Source roots required')
    exact(scope['executed_pilot'],__file__)
    for path,digest in PINS.items():
        if sha(path)!=digest:raise RuntimeError('Original code/Source/preparation pin differs: '+str(path))
    exact(scope['preparation'],PREP);exact(scope['mean_recipe'],RECIPE);exact(scope['P6_helper'],HELPER)
    exact(scope['source_artifact_registry'],SOURCE/'ARTIFACT_HASHES.json')
    if OUT.exists() or REPORT.exists() or OUT.with_name(OUT.name+'.incomplete').exists():raise RuntimeError('Fresh once-only Source pilot roots required')
    return scope, binding(scope_path)

def source_archive_hashes():
    entries=json.loads((SOURCE/'ARTIFACT_HASHES.json').read_text());observed=[]
    for entry in entries:
        name=entry['path'];path=SOURCE/name
        if Path(name).name!=name or path.resolve().parent!=SOURCE.resolve():raise RuntimeError('Unsafe original Source artifact name')
        actual=binding(path)
        if actual['bytes']!=entry['bytes'] or actual['sha256']!=entry['sha256']:raise RuntimeError('Original frozen Source archive differs: '+name)
        observed.append(actual)
    return observed

def run(scope_path):
    started=time.monotonic();cpu_started=time.process_time()
    scope,scope_binding=authorize(scope_path)
    def timeout(signum,frame):raise TimeoutError('Source pilot300s ceiling exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    stage=OUT.with_name(OUT.name+'.incomplete');stage.mkdir(parents=True)
    write_json(stage/'ATTEMPT_REGISTRATION.json',{'schema':'safeconf_Source_mean_once_attempt_v1','scope':scope_binding,'pilot_code':binding(__file__),'formal_upstream_attempt_index':2,'automatic_retry_or_release':False,'registered_before_any_Source_numeric_materialization':True,'created_utc':datetime.now(timezone.utc).isoformat()})
    (stage/'ATTEMPT_REGISTRATION.json').chmod(0o444)
    before=source_archive_hashes();memory_check()
    feature=json.loads((SOURCE/'SOURCE_FEATURE_MANIFEST.json').read_text())
    helper=import_fixed(HELPER,'fixed_Source_P6_math');recipe=import_fixed(RECIPE,'fixed_Source_task_mean_recipe')
    if feature['risk_P_columns']!=helper.P:raise RuntimeError('Exact fixed P6 contract differs')
    tasks=pd.read_csv(SOURCE/'SPLIT_MANIFEST.csv')
    archived_meta=pd.read_parquet(SOURCE/'SOURCE_TASKS.parquet',columns=META)
    if not tasks[META].equals(archived_meta[META]):raise RuntimeError('Original array task-row order differs')
    if len(tasks)!=1808 or not tasks.task_id.is_unique or tasks.gene.nunique()!=575 or not tasks.groupby('gene').fold.nunique().eq(1).all():raise RuntimeError('Original Source gene/task folds differ')
    true_delta=np.load(SOURCE/'SOURCE_TRUE_EFFECTS.npy',mmap_mode='r',allow_pickle=False)
    controls=np.load(SOURCE/'SOURCE_CONTROLS.npy',mmap_mode='r',allow_pickle=False)
    if true_delta.dtype!=np.float64 or controls.dtype!=np.float64:raise RuntimeError('Exact Sourcefloat64 vectors required')
    estimate_started=time.monotonic()
    prediction,centroids=recipe.propose_five_gene_fold_task_mean(tasks[META].to_dict('records'),true_delta,controls)
    estimate_elapsed=time.monotonic()-estimate_started;memory_check()
    np.save(stage/'SOURCE_MEAN_PREDICTED_DELTA.npy',prediction,allow_pickle=False)
    np.save(stage/'SOURCE_MEAN_FOLD_CONTEXT_ABSOLUTE_CENTROIDS.npy',np.stack([x[2] for x in centroids]),allow_pickle=False)
    identity=tasks[META].copy();identity['upstream']=MODEL_ID;identity['model_version']='v1';identity.to_csv(stage/'PREDICTION_IDENTITIES.csv',index=False)
    p6=helper.prediction_features(prediction)
    if p6.columns.tolist()!=feature['risk_P_columns']:raise RuntimeError('Fixed helper P6 width/order differs')
    pd.concat([identity,p6],axis=1).to_parquet(stage/'SOURCE_MEAN_P6_FEATURES.parquet',index=False)
    fits=[]
    for row,(fold,context,_) in enumerate(centroids):
        train=tasks[(tasks.fold!=fold)&(tasks.target==context)];query=tasks[(tasks.fold==fold)&(tasks.target==context)]
        if set(train.gene)&set(query.gene):raise RuntimeError('Heldout gene in mean fit')
        fits.append({'parameter_row':row,'heldout_fold':fold,'context':context,'train_tasks':len(train),'query_tasks':len(query),
          'train_task_ids_sha256':hashlib.sha256('\n'.join(train.task_id).encode()).hexdigest(),
          'query_task_ids_sha256':hashlib.sha256('\n'.join(query.task_id).encode()).hexdigest(),'heldout_gene_truth_in_this_fit':False})
    prediction_files=[dict(binding(stage/name),path=str(OUT/name)) for name in ['SOURCE_MEAN_PREDICTED_DELTA.npy','SOURCE_MEAN_FOLD_CONTEXT_ABSOLUTE_CENTROIDS.npy','PREDICTION_IDENTITIES.csv','SOURCE_MEAN_P6_FEATURES.parquet']]
    seal={'schema':'safeconf_Source_mean_prediction_seal_v1','status':'ALL1808_PREDICTIONS_SEALED_BEFORE_HELDOUT_ERROR_EVALUATION','scope':scope_binding,
      'model_id':MODEL_ID,'model_version':'v1','Source_axis':'3285_CP4000_log1p_matched_control_delta','mean_convention':scope['Source_mean_convention'],
      'n_query_tasks':1808,'n_gene_clusters':575,'analytic_fold_context_mean_parameter_vectors':20,'new_optimized_or_shared_risk_fits':0,
      'fold_context_parameter_records':fits,'files':prediction_files,'query_treated_truth_evaluation_started':False,'P6_helper':binding(HELPER),'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(stage/'PREDICTION_SEAL.json',seal)
    for item in prediction_files:
        current=stage/Path(item['path']).name;current.chmod(0o444)
        actual=binding(current)
        if actual['sha256']!=item['sha256'] or actual['bytes']!=item['bytes']:raise RuntimeError('Sealed Source predictions changed')
    (stage/'PREDICTION_SEAL.json').chmod(0o444)
    # No heldout error evaluation or cached old error-value read precedes seal.
    errors=np.sqrt(np.mean((prediction-np.asarray(true_delta,dtype=np.float64))**2,axis=1,dtype=np.float64))
    if not np.isfinite(errors).all():raise RuntimeError('Source heldout errors nonfinite')
    result=identity.copy();result['predicted_magnitude']=p6.predicted_magnitude;result['Source_mean_realized_DEV_error_rmse']=errors
    ranges=[];correlations=[]
    def range_rows(method,values):
        for context in CONTEXTS:
            mask=tasks.target.eq(context).to_numpy();v=values[mask]
            row={'upstream':method,'context':context,'n_tasks':len(v),'unique_magnitudes':len(np.unique(v))}
            row.update({'q'+str(int(q*100)):float(np.quantile(v,q)) for q in QUANTILES});ranges.append(row)
    range_rows(MODEL_ID,p6.predicted_magnitude.to_numpy())
    cached_columns=META+['upstream','model_version','predicted_magnitude','true_error_rmse']
    for old in ('TxPert_GAT','TxPert_Exphormer'):
        cache=pd.read_parquet(SOURCE/f'SOURCE_{old}_P_BASE.parquet',columns=cached_columns)
        if not cache[META].equals(tasks[META]) or not cache.upstream.eq(old).all():raise RuntimeError('OldSource cache row/model/axis identity differs')
        old_errors=cache.true_error_rmse.to_numpy(float);old_magnitude=cache.predicted_magnitude.to_numpy(float)
        if not np.isfinite(old_errors).all() or not np.isfinite(old_magnitude).all():raise RuntimeError('Frozen Source cache nonfinite')
        result[old+'_cached_error_rmse']=old_errors;result[old+'_cached_magnitude']=old_magnitude
        range_rows(old,old_magnitude)
        for context in CONTEXTS:
            mask=tasks.target.eq(context).to_numpy();a=errors[mask];b=old_errors[mask];m=p6.predicted_magnitude.to_numpy()[mask];oldm=old_magnitude[mask]
            rho=float(spearmanr(a,b).statistic) if len(np.unique(a))>1 and len(np.unique(b))>1 else None
            correlations.append({'context':context,'old_upstream':old,'n_matched_tasks':int(mask.sum()),'SourceMean_vs_old_error_spearman':rho,
              'Source_mean_unique_errors':len(np.unique(a)),'old_unique_errors':len(np.unique(b)),'SourceMean_below_old_min_fraction':float(np.mean(m<oldm.min())),
              'SourceMean_within_old_min_max_fraction':float(np.mean((m>=oldm.min())&(m<=oldm.max()))),
              'SourceMean_below_old_q05_fraction':float(np.mean(m<np.quantile(oldm,.05)))})
    result.to_csv(stage/'SOURCE_DEV_ERROR_AND_MAGNITUDE_DIAGNOSTIC.csv',index=False)
    pd.DataFrame(ranges).to_csv(stage/'ALL_FIXED_CONTEXT_MAGNITUDE_RANGES.csv',index=False)
    pd.DataFrame(correlations).to_csv(stage/'ALL_FIXED_CONTEXT_ERROR_RANK_CORRELATIONS.csv',index=False)
    memory_check();after=source_archive_hashes()
    if before!=after:raise RuntimeError('OriginalSource files changed during isolated pilot')
    exact(scope['source_artifact_registry'],SOURCE/'ARTIFACT_HASHES.json')
    exact(scope['executed_pilot'],__file__)
    if binding(scope_path)!=scope_binding:raise RuntimeError('Immutable Source scope changed')
    for path,digest in PINS.items():
        if sha(path)!=digest:raise RuntimeError('Original proposal/helper/Source code pin changed')
    receipt={'schema':'safeconf_Source_task_mean_regime_pilot_v1','status':'COMPLETE_SOURCE_DEV_SEEN_DIAGNOSTIC','scope':scope_binding,'pilot_code':binding(__file__),
      'model_id':MODEL_ID,'model_version':'v1','formal_upstream_attempt_index':2,'n_tasks':1808,'n_gene_clusters':575,'contexts':list(CONTEXTS),'n_P_fields':6,
      'analytic_mean_parameter_vectors':20,'new_shared_risk_CDF_Public_or_optimized_fits':0,'new_SourceMean_evaluation_error_rows':1808,'new_SourceMean_risk_training_labels':0,
      'prediction_seal':dict(binding(stage/'PREDICTION_SEAL.json'),path=str(OUT/'PREDICTION_SEAL.json')),'heldout_error_evaluation_only_after_all_prediction_hashes_frozen':True,
      'before_after_original_Source_bindings_exact':True,'original_Source_file_bindings':before,
      'mean_estimation_elapsed_seconds':estimate_elapsed,'elapsed_seconds_at_receipt_capture':time.monotonic()-started,'process_CPU_delta_seconds_at_receipt_capture':time.process_time()-cpu_started,
      'peak_RSS_bytes_at_receipt_capture':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_hours':0,'new_download_bytes':0,'Orion_numeric_or_outcome_access':False,
      'old_upstream_model_inference':False,'shared_risk_or_Source_CDF_changed':False,'Original_Source_C8_models_untouched':True,
      'limits':['Posttruth-inspired but SourceDEV-only hypothesis diagnostic; not Orion repair or new confirmation.',
        'One fixed simple mean predictor, not a third large/mature published model family.',
        'No shared-risk retraining, target evaluation, model selection, new CDF or efficacy/promotion claim.',
        'Equal-task means conditional on four original Source contexts and frozen matched controls; all fivefolds/contexts retained.',
        'Cost excludes Python imports and final receipt/registry/report writes; full final process CPU unmeasured.']}
    write_json(stage/'RECEIPT.json',receipt)
    write_json(stage/'ARTIFACT_HASHES.json',[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(stage.iterdir()) if p.is_file()])
    for p in stage.iterdir():p.chmod(0o444)
    stage.rename(OUT);REPORT.mkdir(parents=True)
    for name in ['RECEIPT.json','PREDICTION_SEAL.json','ALL_FIXED_CONTEXT_MAGNITUDE_RANGES.csv','ALL_FIXED_CONTEXT_ERROR_RANK_CORRELATIONS.csv']:
        data=(OUT/name).read_bytes();(REPORT/name).write_bytes(data);(REPORT/name).chmod(0o444)
    signal.alarm(0)
    print(json.dumps({'status':receipt['status'],'runtime':str(OUT),'report':str(REPORT),'elapsed_seconds':receipt['elapsed_seconds_at_receipt_capture'],'mean_parameter_vectors':20,'new_shared_risk_fits':0,'Orion_access':False}),flush=True)
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scope-contract',type=Path,required=True)
    run(parser.parse_args().scope_contract)
