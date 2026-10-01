#!/usr/bin/env python3
"""Frozen Orion VALIDATION competence; no TEST truth or SafeConf scoring.

Consumes only VALIDATION effect rows and projected pretruth validation prediction
columns. The simple baseline was already selected on TRAIN. No fitting occurs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.scripts import bridge_safeconf_orion_lm_inputs_agent as bridge_api
from tools.scripts import build_safeconf_orion_allowed_biology_agent as biology
BASELINE_HELPER=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/competence/EXTRACT_FROZEN_BASELINE.R'
CONTEXTS=('HCT116','HEK293T')
MARGIN=.02
BOOTSTRAPS=5000
SEED=20260929

def sha(path):return bridge_api.sha(path)
def vector_sha(values):return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()
def load_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):bridge_api.write_json(path,value)
def read_tsv(path):return pd.read_csv(path,sep='\t',keep_default_na=False,float_precision='round_trip')

def paired_competence(task_errors):
    columns=['context_id','target_gene_id','model_rmse','baseline_rmse']
    if not set(columns)<=set(task_errors) or set(task_errors.context_id)!=set(CONTEXTS):
        raise RuntimeError('Both registered contexts with task error columns required')
    if task_errors.duplicated(['context_id','target_gene_id']).any():raise RuntimeError('Duplicate gene/context task')
    numeric=task_errors[['model_rmse','baseline_rmse']].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric<0).any():raise RuntimeError('Nonfinite/negative task errors')
    context=[]
    for name in CONTEXTS:
        rows=task_errors[task_errors.context_id==name]
        model=float(rows.model_rmse.mean());baseline=float(rows.baseline_rmse.mean())
        if baseline<=0:raise RuntimeError('Nonpositive selected baseline error; relative competence is undefined')
        context.append({'context_id':name,'n_tasks':len(rows),'model_macro_rmse':model,'baseline_macro_rmse':baseline,
            'relative_gap':model/baseline-1.,'noninferior':bool(model<=(1.+MARGIN)*baseline)})
    context=pd.DataFrame(context)
    model_macro=float(context.model_macro_rmse.mean());baseline_macro=float(context.baseline_macro_rmse.mean())
    genes=sorted(task_errors.target_gene_id.unique());gene_index={g:i for i,g in enumerate(genes)}
    arrays=[]
    for name in CONTEXTS:
        rows=task_errors[task_errors.context_id==name]
        arrays.append((np.asarray([gene_index[g]for g in rows.target_gene_id],dtype=np.int64),
                       rows.model_rmse.to_numpy(dtype=np.float64),rows.baseline_rmse.to_numpy(dtype=np.float64)))
    rng=np.random.default_rng(SEED);draws=[]
    for replicate in range(BOOTSTRAPS):
        sampled=rng.integers(0,len(genes),size=len(genes));weights=np.bincount(sampled,minlength=len(genes))
        models=[];baselines=[]
        for indices,model_error,baseline_error in arrays:
            w=weights[indices];denominator=int(w.sum())
            if denominator==0:raise RuntimeError('Empty registered context in paired gene bootstrap; no redraw allowed')
            models.append(float(np.dot(w,model_error)/denominator));baselines.append(float(np.dot(w,baseline_error)/denominator))
        m=float(np.mean(models));b=float(np.mean(baselines))
        if b<=0:raise RuntimeError('Nonpositive bootstrap baseline macro; relative competence is undefined')
        ratio=m/b
        draws.append({'replicate':replicate,'macro_model_rmse':m,'macro_baseline_rmse':b,'ratio':ratio,
            'relative_gap':ratio-1.,'gene_draw_sha256':hashlib.sha256(sampled.astype('<i8').tobytes()).hexdigest()})
    draws=pd.DataFrame(draws);low,high=np.quantile(draws.ratio,[.025,.975],method='linear')
    fraction=float(context.noninferior.mean());stable=bool(low>1.+MARGIN)
    summary={'model_macro_rmse':model_macro,'baseline_macro_rmse':baseline_macro,
        'relative_macro_error_gap':model_macro/baseline_macro-1.,'noninferior_strata_fraction':fraction,
        'noninferior_contexts':int(context.noninferior.sum()),'actual_context_count':2,
        'lower95_relative_gap':float(low-1.),'upper95_relative_gap':float(high-1.),
        'stable_disadvantage':stable,'passes_competence':bool(model_macro<=(1.+MARGIN)*baseline_macro and fraction>=.6 and not stable),
        'bootstrap_replicates':BOOTSTRAPS,'bootstrap_seed':SEED,'bootstrap_unit':'gene cluster jointly across both contexts',
        'bootstrap_quantile_method':'linear','CI_percentiles':[.025,.975],'gene_cluster_order_sha256':hashlib.sha256(('\n'.join(genes)+'\n').encode()).hexdigest(),
        'boundary_comparison':'macro_model <=1.02*macro_baseline; bootstrap lower_ratio>1.02; no epsilon',
        'inference_scope':'operational competence screen; lower guard does not establish statistical noninferiority; conditional on fixed models/truth/control estimates'}
    return summary,context,draws

def checked_artifact(root,relative,registry):
    if relative not in registry:raise RuntimeError(f'Unregistered artifact: {relative}')
    entry=registry[relative];path=root/relative
    if Path(relative).is_absolute() or '..'in Path(relative).parts or path.stat().st_size!=int(entry['bytes']) or sha(path)!=entry['sha256']:
        raise RuntimeError(f'Artifact size/hash differs: {path}')
    return path

def evaluate(fit_root,biology_root,bridge_root,permit_path,contract_path,validation_eligibility,output,
             rscript,expected_evaluator_sha256='',expected_baseline_helper_sha256='',expected_validation_eligibility_sha256=''):
    started=time.monotonic();fit=Path(fit_root).resolve();bio_root=Path(biology_root).resolve();br=Path(bridge_root).resolve();out=Path(output).resolve()
    if out.exists():raise RuntimeError('New competence output version required')
    permit_path=Path(permit_path).resolve();contract_path=Path(contract_path).resolve();permit=load_json(permit_path);bio=load_json(bio_root/'BIOLOGY_MANIFEST.json')
    synthetic=bool(bio.get('synthetic_only'))
    if not synthetic and (expected_evaluator_sha256!=sha(__file__) or expected_baseline_helper_sha256!=sha(BASELINE_HELPER)):
        raise RuntimeError('Actual validation requires exact pre-frozen evaluator and baseline helper SHA arguments')
    if not synthetic and expected_validation_eligibility_sha256!=sha(validation_eligibility):
        raise RuntimeError('Actual validation requires the exact pre-truth eligible task set SHA argument')
    contract,permit,_=biology.authorization(Path(permit['metadata_root']),Path(permit['raw_root']),contract_path,permit_path,['VALIDATION',biology.CONTROL_ROLE],CONTEXTS)
    rule=contract.get('competence',{})
    for key,expected in [('relative_macro_error_gap_max',.02),('minimum_noninferior_strata_fraction',.6),('bootstrap_replicates',5000),('bootstrap_seed',20260929)]:
        if rule.get(key)!=expected:raise RuntimeError('Scientific competence rule differs')
    if bio.get('status')!='COMPLETE' or bio.get('scientific_contract_sha256')!=sha(contract_path) or bio.get('permit_sha256')!=sha(permit_path):
        raise RuntimeError('Biology contract/permit identity differs')
    if bio.get('private_test_numeric_materialization') is not False or set(bio.get('roles',[]))-{'TRAIN','VALIDATION',biology.CONTROL_ROLE}:
        raise RuntimeError('TEST/private numerical biology is forbidden')
    scope_root=Path(permit.get('lm_query_scope',{}).get('path','')).parent
    scope_manifest=load_json(scope_root/'QUERY_SCOPE_MANIFEST.json')
    if Path(validation_eligibility).resolve()!=(scope_root/'VALIDATION_ELIGIBILITY.tsv').resolve() or sha(validation_eligibility)!=scope_manifest['validation_eligibility_sha256']:
        raise RuntimeError('Validation eligibility must be the pre-frozen query-scope sibling artifact')
    bmanifest=load_json(br/'BRIDGE_MANIFEST.json')
    if bmanifest['biology_manifest_sha256']!=sha(bio_root/'BIOLOGY_MANIFEST.json') or bmanifest['permit_sha256']!=sha(permit_path) or bmanifest['scientific_contract_sha256']!=sha(contract_path):
        raise RuntimeError('Fit bridge lineage differs')
    if not synthetic:
        for name,path in [('bridge',Path(bridge_api.__file__)),('lm_cli',bridge_api.R_CLI)]:
            binding=permit['implementation_bindings'][name]
            if Path(binding['path']).resolve()!=path.resolve() or binding['sha256']!=sha(path):raise RuntimeError('Frozen bridge/LM code differs')
    registry={r['path']:r for r in load_json(bio_root/'ARTIFACT_HASHES.json')}
    units=bridge_api.read_csv(checked_artifact(bio_root,'QUERY_METADATA_ONLY.csv',registry))
    endpoint=bridge_api.read_csv(checked_artifact(bio_root,'ENDPOINT_GENE_MANIFEST.csv',registry))
    effects_path=checked_artifact(bio_root,'ENDPOINT3285_EFFECTS.npy',registry)
    for context in CONTEXTS:
        controls=units[(units.context==context)&(units.role==biology.CONTROL_ROLE)]
        if len(controls)!=1 or controls.iloc[0].gene!='Non-Targeting' or int(controls.iloc[0].matrix_row)!=int(bio['control_matrix_rows'][context]):
            raise RuntimeError('Validation effects must use the exact own-context TRAIN NTC reference')
    if len(endpoint)!=3285 or not np.array_equal(endpoint.axis_index,np.arange(3285)) or not endpoint.orion_ensembl_id.is_unique:
        raise RuntimeError('Frozen endpoint3285 order differs')
    br_registry={r['path']:r for r in load_json(br/'ARTIFACT_HASHES.json')}
    axis=read_tsv(checked_artifact(br,'OUTPUT_GENE_AXIS.tsv',br_registry))
    if axis.gene_id.tolist()!=endpoint.orion_ensembl_id.tolist():raise RuntimeError('Bridge/effect endpoint axes differ')
    eligible=read_tsv(validation_eligibility)
    required=set(bridge_api.QUERY_COLUMNS+['n_cells_metadata'])
    if not required<=set(eligible) or eligible.empty or set(eligible.role)!={'VALIDATION'} or eligible.duplicated(['context_id','target_gene_id']).any():
        raise RuntimeError('Frozen eligible VALIDATION metadata tasks required')
    if (pd.to_numeric(eligible.n_cells_metadata)<30).any():raise RuntimeError('Ineligible validation cell counts')
    if set(eligible.context_id)!=set(CONTEXTS):raise RuntimeError('Both actual contexts required')
    effects=np.load(effects_path,mmap_mode='r',allow_pickle=False)
    if effects.dtype!=np.float64 or effects.shape!=(len(units),3285):raise RuntimeError('Effect matrix float64 shape differs')
    stage=out.with_name(out.name+f'.incomplete.{os.getpid()}');stage.mkdir(parents=True)
    inputs=[];tasks=[];read_rows=[];baseline_names={};attempt_ids=set()
    run_manifest=read_tsv(fit/'RUN_MANIFEST.tsv')
    if set(run_manifest.context_id)!=set(CONTEXTS):raise RuntimeError('Exactly two fitted context models required')
    for context in CONTEXTS:
        folder=fit/context;fregistry={r['path']:r for r in read_tsv(folder/'ARTIFACT_HASHES.tsv').to_dict('records')}
        for name in ['MODEL.rds','SIMPLE_BASELINES.rds','BASELINE_SELECTION.tsv','PREDICTIONS_DELTA.tsv.gz','OUTPUT_GENE_AXIS.tsv','QUERY_IDENTITIES.tsv','STATUS.tsv']:
            p=checked_artifact(folder,name,fregistry);inputs.append({'path':str(p),'sha256':sha(p)})
        run_row=run_manifest[run_manifest.context_id==context].iloc[0]
        if sha(folder/'MODEL.rds')!=run_row.model_sha256:raise RuntimeError('Fitted model freeze hash differs')
        status=dict(read_tsv(folder/'STATUS.tsv')[['key','value']].itertuples(index=False,name=None))
        if status.get('status')!='PASS' or status.get('mode')!='fit' or status.get('context_id')!=context or status.get('pca_dim')!='10' or status.get('ridge_penalty')!='0.1' or status.get('seed')!='1':
            raise RuntimeError('Locked fitted method/status differs')
        receipt=dict(read_tsv(br/context/'INPUT_RECEIPT.tsv')[['key','value']].itertuples(index=False,name=None));attempt_ids.add(receipt['upstream_attempt_id'])
        if receipt['biology_manifest_sha256']!=sha(bio_root/'BIOLOGY_MANIFEST.json') or receipt['method_contract_sha256']!=sha(contract_path) or receipt['expression_permit_sha256']!=sha(permit_path):
            raise RuntimeError('Per-context fitted input lineage differs')
        queries=read_tsv(folder/'QUERY_IDENTITIES.tsv');current=eligible[eligible.context_id==context].copy()
        if set(current.query_id)-set(queries[queries.role=='VALIDATION'].query_id):raise RuntimeError('Missing frozen eligible validation predictions; no attrition allowed')
        frozen_query=queries.set_index('query_id')
        for q in current.itertuples(index=False):
            matched=frozen_query.loc[q.query_id]
            if matched.role!='VALIDATION' or matched.context_id!=context or matched.target_gene_id!=q.target_gene_id or matched.target_gene_symbol!=q.target_gene_symbol:
                raise RuntimeError('Frozen validation prediction identity differs')
        if read_tsv(folder/'OUTPUT_GENE_AXIS.tsv').gene_id.tolist()!=axis.gene_id.tolist():raise RuntimeError('Prediction endpoint axis differs')
        prediction_path=folder/'PREDICTIONS_DELTA.tsv.gz'
        with gzip.open(prediction_path,'rt') as f:header=f.readline().rstrip('\n').split('\t')
        columns=['gene_id']+current.query_id.tolist()
        if len(header)!=len(set(header)) or set(columns)-set(header):raise RuntimeError('Missing/duplicate frozen prediction columns')
        # C parser converts only selected VALIDATION columns to float64.
        prediction=pd.read_csv(prediction_path,sep='\t',usecols=columns,keep_default_na=False,float_precision='round_trip')
        if prediction.gene_id.tolist()!=axis.gene_id.tolist():raise RuntimeError('Projected validation prediction axis differs')
        base_output=stage/f'{context}_SELECTED_TRAIN_BASELINE'
        with (stage/f'{context}_BASELINE_EXTRACT.log').open('w') as stream:
            subprocess.run([str(rscript),'--vanilla',str(BASELINE_HELPER),str(folder),str(folder/'OUTPUT_GENE_AXIS.tsv'),str(base_output),str(br/context/'INPUT_RECEIPT.tsv')],check=True,stdout=stream,stderr=subprocess.STDOUT)
        baseline=np.fromfile(str(base_output)+'.f64le',dtype='<f8')
        info=dict(read_tsv(str(base_output)+'.tsv')[['key','value']].itertuples(index=False,name=None));baseline_names[context]=info['selected']
        if baseline.shape!=(3285,) or not np.isfinite(baseline).all():raise RuntimeError('Frozen selected baseline vector invalid')
        for row in current.itertuples(index=False):
            matches=units[(units.context==context)&(units.role=='VALIDATION')&(units.target_ensembl_id==row.target_gene_id)]
            if len(matches)!=1:raise RuntimeError('Missing/duplicate authorized validation effect task; no attrition allowed')
            u=matches.iloc[0]
            if u.biological_unit!=row.query_id or u.gene!=row.target_gene_symbol or int(u.n_cells_metadata)!=int(row.n_cells_metadata):
                raise RuntimeError('Frozen validation unit/cell eligibility differs')
            values=np.asarray(effects[int(u.matrix_row)],dtype=np.float64)
            pred=prediction[row.query_id].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or not np.isfinite(pred).all():raise RuntimeError('Nonfinite validation prediction/truth; no silent attrition allowed')
            model_rmse=float(np.sqrt(np.mean((pred-values)**2,dtype=np.float64)))
            baseline_rmse=float(np.sqrt(np.mean((baseline-values)**2,dtype=np.float64)))
            tasks.append({'query_id':row.query_id,'target_gene_id':row.target_gene_id,'target_gene_symbol':row.target_gene_symbol,
                'context_id':context,'role':'VALIDATION','n_cells_metadata':int(row.n_cells_metadata),
                'model_rmse':model_rmse,'baseline_rmse':baseline_rmse,'selected_TRAIN_baseline':info['selected'],
                'validation_effect_vector_sha256':vector_sha(values),'pretruth_validation_prediction_vector_sha256':vector_sha(pred),
                'selected_TRAIN_baseline_vector_sha256':vector_sha(baseline)})
            read_rows.append({'context_id':context,'query_id':row.query_id,'matrix_row':int(u.matrix_row),'role':'VALIDATION'})
    if len(attempt_ids)!=1:raise RuntimeError('Two context fits must belong to one published family attempt')
    task_frame=pd.DataFrame(tasks);summary,context_frame,draws=paired_competence(task_frame)
    task_frame.to_csv(stage/'VALIDATION_TASK_ERRORS.csv',index=False,float_format='%.17g')
    context_frame.to_csv(stage/'CONTEXT_COMPETENCE.csv',index=False,float_format='%.17g')
    draws.to_csv(stage/'PAIRED_GENE_BOOTSTRAP.csv',index=False,float_format='%.17g')
    summary.update(status='PASS' if summary['passes_competence'] else 'FAIL_COMPETENCE',synthetic_only=synthetic,
        family_name='Ahlmann-Eltze2025_published_linear_PCA_adaptation',published_family_attempts=1,context_models=2,
        three_independent_model_families=False,older_three_background_requirement_fulfilled=False,
        measurement_Quality='proxy/biological_replicate_unknown',
        eligible_validation_tasks=len(eligible),evaluated_validation_tasks=len(tasks),missing_predictions=0,nonfinite_predictions=0,
        selected_TRAIN_baselines=baseline_names,validation_used_only_for_competence=True,
        validation_used_for_risk_CDF_core_baseline_or_model_selection=False,SafeConf_scores_computed=False,
        test_truth_read=False,new_model_parameters_fitted=False,precision='float64; TSV predictions parsed round_trip; 17-digit error outputs',
        evaluator_sha256=sha(__file__),baseline_helper_sha256=sha(BASELINE_HELPER),scientific_contract_sha256=sha(contract_path),permit_sha256=sha(permit_path),
        input_estimand='mean_cell_log1p_cp4000_v1; signed own-context TRAIN NTC effects',gpu_hours=0,elapsed_seconds=time.monotonic()-started)
    write_json(stage/'COMPETENCE_RESULT.json',summary)
    inputs.extend([{'path':str(p),'sha256':sha(p)}for p in [bio_root/'BIOLOGY_MANIFEST.json',bio_root/'ARTIFACT_HASHES.json',br/'BRIDGE_MANIFEST.json',fit/'RUN_MANIFEST.tsv',Path(validation_eligibility),permit_path,contract_path,effects_path]])
    write_json(stage/'INPUT_IDENTITIES.json',inputs)
    write_json(stage/'C_VALIDATION_ACCESS_LEDGER.json',{'ledger_class':'C_VALIDATION_COMPETENCE_ONLY','numeric_effect_rows_read':read_rows,
        'numeric_prediction_columns_read':[t['query_id']for t in tasks],'numeric_roles_read':['VALIDATION'],
        'TRAIN_values_read':'frozen selected baseline vector and TRAIN OOF selection only; no TRAIN effect rows',
        'TEST_truth_rows_read':0,'TEST_prediction_columns_converted':0,'Source_or_risk_parameters_modified':False,
        'allowed_use':['fixed upstream validation competence'],
        'forbidden_use':['risk fitting','target-error CDF','core fitting','baseline selection','prediction scaling','SafeConf model selection'],
        'new_upstream_attempt_started':False,'two_context_models_are_one_family_attempt':True})
    write_json(stage/'ARTIFACT_HASHES.json',[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(stage.iterdir()) if p.is_file()])
    stage.rename(out);return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fit-root','biology-root','bridge-root','permit','contract','validation-eligibility','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rscript',type=Path,default=Path('/home/yyf/.conda/envs/safeconf-orion-lm-20261002/bin/Rscript'))
    p.add_argument('--expected-evaluator-sha256',default='');p.add_argument('--expected-baseline-helper-sha256',default='');p.add_argument('--expected-validation-eligibility-sha256',default='')
    a=p.parse_args();result=evaluate(a.fit_root,a.biology_root,a.bridge_root,a.permit,a.contract,a.validation_eligibility,a.output,a.rscript,a.expected_evaluator_sha256,a.expected_baseline_helper_sha256,a.expected_validation_eligibility_sha256)
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':main()
