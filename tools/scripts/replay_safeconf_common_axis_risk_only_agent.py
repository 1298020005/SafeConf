#!/usr/bin/env python3
"""Two fixed archived common2840 risk fits; no Public/upper/raw-X replay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.safeconf_continual import research as recipe
from tools.safeconf_continual import contracts,learners

BASE=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
COMMON=BASE/'common_gene_axis'
DOC=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis/risk_only_replay_v1'
RESULT=DOC.parent/'results'
OUT=BASE/'common_axis_risk_only_replay_20261002_v1'
META=['task_id','gene','target','fold','upstream','dataset_id','model_version','output_contract_id']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def binding(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}


def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    started=time.monotonic()
    if OUT.exists() or DOC.exists():raise FileExistsError('Fresh isolated replay outputs required')
    DOC.mkdir(parents=True);OUT.mkdir();fits_performed=0
    core=json.loads((DOC.parent/'COMMON_AXIS_PREDICTION_FREEZE.json').read_text())
    if core['core_parameters']!={'depth':3,'hgb_iter':200,'l2':10,'learning_rate':0.05,'min_leaf':20}:
        raise RuntimeError('Archived fixed HGB recipe is not exact')
    axis=json.loads((COMMON/'GENE_IDS.json').read_text())
    if len(axis)!=2840 or len(set(axis))!=2840:raise RuntimeError('Exact archived common2840 axis required')
    if tuple(recipe.SEEDS)!=(20260930,20261001,20261002):raise RuntimeError('Archived seed sequence differs')
    features=recipe.P+recipe.PUBLIC
    ledger=pd.read_csv(RESULT/'INFORMATION_BUDGET_LEDGER.csv',keep_default_na=False)
    archived_cdf=pd.read_csv(RESULT/'CDF_AUDIT.csv',float_precision='round_trip')
    archived_cdf=archived_cdf[archived_cdf.run_id.eq('TxPert_to_McFaline/outer-1')].reset_index(drop=True)
    scores_path=RESULT/'MATRIX_TASK_PREDICTIONS.csv.gz'
    # Never load query truth/errors, neither for fitting nor comparison.
    archived=pd.read_csv(scores_path,usecols=['task_id','gene','target','fold','upstream','risk','line','method','seed'],float_precision='round_trip')
    archived=archived[archived.line.eq('TxPert_to_McFaline')&archived.seed.eq(20260930)&archived.method.isin(['Manual_hgb','Learned_hgb'])]
    inputs={};frames={};queries={}
    for ref in ['Manual','Learned']:
        sp=COMMON/'risk_cache'/f'source_{ref}.parquet';qp=COMMON/'risk_cache'/f'external_{ref}.parquet'
        frames[ref]=pd.read_parquet(sp,columns=META+features+['true_error_rmse'])
        queries[ref]=pd.read_parquet(qp,columns=META+features)
        inputs[ref]={'source_features_errors':binding(sp),'query_features_without_errors':binding(qp)}
    manual,learned=frames['Manual'],frames['Learned']
    if (not manual[META+['true_error_rmse']].equals(learned[META+['true_error_rmse']])
        or not queries['Manual'][META].equals(queries['Learned'][META])
        or len(manual)!=3616 or manual.task_id.nunique()!=1808 or manual.gene.nunique()!=575
        or set(manual.upstream)!={'TxPert_GAT','TxPert_Exphormer'} or len(queries['Manual'])!=543):
        raise RuntimeError('Archived Source/query identities, order or error budget differ')
    labels,cdf=recipe.rank_labels(manual,'TxPert_to_McFaline/outer-1',1.)
    weights=recipe.cluster_weights(manual)
    if not np.isfinite(labels).all() or not np.isfinite(weights).all():raise RuntimeError('All exact Source-only labels/weights must be defined')
    rebuilt=pd.DataFrame(cdf).sort_values(recipe.CDF_KEYS).reset_index(drop=True)
    old=archived_cdf.sort_values(recipe.CDF_KEYS).reset_index(drop=True)
    if len(rebuilt)!=8 or len(old)!=8:raise RuntimeError('Eight Source-only CDF groups required')
    exact_fields=recipe.CDF_KEYS+['run_id','budget','n_rows','n_biological_tasks','n_clusters','n_unique_errors','training_records_hash','training_clusters_hash','status']
    for name in exact_fields:
        if rebuilt[name].tolist()!=old[name].tolist():raise RuntimeError('Archived CDF scope/hash/count differs: '+name)
    for ref in ['Manual','Learned']:
        item=ledger[ledger.line.eq('TxPert_to_McFaline')&ledger.outer_fold.eq(-1)&ledger.method.eq(ref+'_hgb')]
        if (len(item)!=1 or int(item.iloc[0].source_error_rows)!=3616 or int(item.iloc[0].source_error_clusters)!=575
            or int(item.iloc[0].source_predictors)!=2 or item.iloc[0].source_error_records_hash!=recipe.ids_hash(manual.upstream+'::'+manual.task_id)
            or int(item.iloc[0].c_risk_error_labels)!=0 or int(item.iloc[0].c_feedback_error_labels)!=0 or int(item.iloc[0].c_error_CDF_labels)!=0):
            raise RuntimeError('Original method information budget cannot be reproduced')
        a=archived[archived.method.eq(ref+'_hgb')]
        if a.duplicated('task_id').any() or a[['task_id','gene','target','fold','upstream']].to_dict('records')!=queries[ref][['task_id','gene','target','fold','upstream']].to_dict('records'):
            raise RuntimeError('Archived risk identities/order differ from exact query feature frame')
    bindings=[binding(Path(__file__)),binding(Path(recipe.__file__)),binding(Path(contracts.__file__)),binding(Path(learners.__file__)),binding(DOC.parent/'COMMON_AXIS_PREDICTION_FREEZE.json'),binding(COMMON/'GENE_IDS.json'),binding(RESULT/'INFORMATION_BUDGET_LEDGER.csv'),binding(RESULT/'CDF_AUDIT.csv'),binding(scores_path)]
    bindings += [v for d in inputs.values() for v in d.values()]
    config={'scope':'risk-only reproduction from archived Source/query features; not original70c source/Public learner/priors reconstruction','methods':['Manual_hgb','Learned_hgb'],'seed':20260930,'columns':features,'HGB':{'max_iter':200,'learning_rate':.05,'max_depth':3,'min_samples_leaf':20,'l2_regularization':10,'random_state':20260930},'preprocessing':'Source-fit finite medians, mean/std; missing flags; exact current NumericPreprocessor','labels':'Source-only midrank CDF per '+','.join(recipe.CDF_KEYS),'source_budget_fraction':1.,'sample_weights':'inverse gene record frequency normalized to mean1','predict_clip':True,'runtime_versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__},'original_command_executable':'/home/miniconda/bin/python','current_executable':sys.executable,'max_fits':2,'max_seconds':900,'target_CDF_or_error_fit':False,'new_Source_or_Orion_expression':False,'input_bindings':bindings}
    write(OUT/'CONFIG.json',config)
    record=manual[META+['true_error_rmse']].copy();record['source_CDF_label']=labels;record['gene_record_sample_weight']=weights
    record.to_parquet(OUT/'SOURCE_RECORDS_LABELS_WEIGHTS.parquet',index=False)
    np.save(OUT/'SOURCE_CDF_LABELS.npy',labels,allow_pickle=False);np.save(OUT/'SOURCE_GENE_WEIGHTS.npy',weights,allow_pickle=False)
    rebuilt.to_csv(OUT/'REBUILT_SOURCE_CDF_AUDIT.csv',index=False,float_format='%.17g')
    identity_bytes='\n'.join((manual.upstream+'::'+manual.task_id).tolist()).encode()
    summary=[];prediction_rows=[];models=[]
    for ref in ['Manual','Learned']:
        if fits_performed>=2 or time.monotonic()-started>=900:raise RuntimeError('Two-fit/time budget exceeded')
        fit=recipe.fit_risk(frames[ref],labels,features,'hgb',20260930,weighted=True);fits_performed+=1
        if fit.columns!=features or fit.model.get_params()['max_iter']!=200 or fit.model.n_features_in_!=26:
            raise RuntimeError('Exact original13numeric+13missing feature contract differs')
        score=fit.predict(queries[ref],clip=True)
        a=archived[archived.method.eq(ref+'_hgb')].reset_index(drop=True);expected=a.risk.to_numpy(float);gap=score-expected
        part=queries[ref][META].copy();part['method']=ref+'_hgb';part['archived_risk']=expected;part['replayed_risk']=score;part['signed_gap']=gap;prediction_rows.append(part)
        name=ref+'_hgb.joblib';joblib.dump(fit,OUT/name);loaded=joblib.load(OUT/name)
        if not np.array_equal(loaded.predict(queries[ref]),score):raise RuntimeError('Persisted risk/preprocessor reload differs')
        write(OUT/(ref+'_PREPROCESSOR.json'),{'columns':fit.columns,'medians':fit.preprocessor.medians_.tolist(),'center':fit.preprocessor.center_.tolist(),'scale':fit.preprocessor.scale_.tolist(),'missing_flags_appended':True,'model_params':fit.model.get_params()})
        models.append(binding(OUT/name))
        summary.append({'method':ref+'_hgb','n_Source_records':3616,'n_Source_tasks':1808,'n_Source_genes':575,'n_Source_predictors':2,'n_query_tasks':len(score),'exact_float_equal_rows':int((score==expected).sum()),'exact_float_equal_all':bool(np.array_equal(score,expected)),'max_absolute_gap':float(np.max(np.abs(gap))),'mean_absolute_gap':float(np.mean(np.abs(gap))),'RMSE_gap':float(np.sqrt(np.mean(gap**2))),'original_model_parameter_artifact_available':False,'replayed_model_parameter_binding':binding(OUT/name),'preprocessor_binding':binding(OUT/(ref+'_PREPROCESSOR.json'))})
    pd.concat(prediction_rows,ignore_index=True).to_csv(OUT/'RISK_ONLY_REPLAY_PREDICTIONS.csv.gz',index=False,float_format='%.17g')
    for b in bindings:
        if sha(b['path'])!=b['sha256']:raise RuntimeError('Input/code changed during fixed replay')
    artifacts=[binding(p) for p in sorted(OUT.iterdir()) if p.is_file()];write(OUT/'ARTIFACT_HASHES.json',artifacts)
    receipt={'schema':'safeconf_common2840_archived_risk_only_replay_v1','status':'RISK_PORTION_EXACT_REPRODUCED' if all(s['exact_float_equal_all'] for s in summary) else 'RISK_PORTION_REPLAY_COMPLETED_GAPS_REPORTED_NO_TOLERANCE_CHANGED','summary':summary,'Source_ordered_record_sha256':hashlib.sha256(identity_bytes).hexdigest(),'Source_sorted_record_hash':recipe.ids_hash(manual.upstream+'::'+manual.task_id),'Source_labels_binding':binding(OUT/'SOURCE_CDF_LABELS.npy'),'Source_weights_binding':binding(OUT/'SOURCE_GENE_WEIGHTS.npy'),'Source_records_errors_labels_weights_binding':binding(OUT/'SOURCE_RECORDS_LABELS_WEIGHTS.parquet'),'config_binding':binding(OUT/'CONFIG.json'),'CDF_budget_audit_binding':binding(OUT/'REBUILT_SOURCE_CDF_AUDIT.csv'),'artifact_registry':binding(OUT/'ARTIFACT_HASHES.json'),'input_bindings':bindings,'fits_performed':fits_performed,'upstream_models_Public_biology_learner_or_priors_fitted':0,'Source_raw_X_Orion_numeric_or_query_error_reads':0,'saved_model_and_preprocessor_reload_exact':True,'original70c_source_recovered':False,'scope':'Only TxPert-to-McFaline risk portion from already archived common2840 features and fixed current recipe. No Public learner/pair-weight/source-code recovery claim; no original asset replacement/new-method promotion.','elapsed_seconds':time.monotonic()-started,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    write(DOC/'REPLAY_RECEIPT.json',receipt);pd.DataFrame([{k:v for k,v in s.items() if not isinstance(v,dict)} for s in summary]).to_csv(DOC/'REPLAY_SUMMARY.csv',index=False,float_format='%.17g')
    write(DOC/'OWNED_FILES.json',{'scope':'two fixed SEEN archived risk-only replay fits','owned_files':[binding(Path(__file__)),binding(DOC/'REPLAY_RECEIPT.json'),binding(DOC/'REPLAY_SUMMARY.csv')],'runtime_model_preprocessor_arrays_not_in_git':str(OUT),'old_scores_assets_changed':False})
    for p in list(DOC.iterdir())+list(OUT.iterdir()):p.chmod(0o444)
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
