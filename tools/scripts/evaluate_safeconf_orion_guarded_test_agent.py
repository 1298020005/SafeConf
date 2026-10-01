#!/usr/bin/env python3
"""Postseal TEST truth reader. No raw access before every immutable gate passes.

No fitting, risk selection, target CDF, new denominator or survivor selection.
This file does not authorize TEST access: the root must issue the exact immutable
postseal operation after actual competence, full LM fit and comparison seals.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tools.scripts import build_safeconf_orion_allowed_biology_agent as biology
from tools.scripts import probe_safeconf_private_parquet_vectorized_agent as reader
from tools.scripts import seal_safeconf_orion_source_risk_agent as risk_api

DOC = ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/test_open'
SCHEMA = 'safeconf_orion_explicit_postseal_test_access_v1'
OUTPUT_SCHEMA = 'safeconf_orion_postseal_test_truth_reader_v1'
CONTEXTS = ('HCT116', 'HEK293T')
METHOD_SHA = risk_api.METHOD_SHA
META_NAMES = ['FILE_METADATA_IDENTITY.json','GENE_MANIFEST.csv','gene_metadata.parquet','PREPARATION_POLICY.json','GENE_SPLIT.csv']
RISK_COLUMNS = [f'{reference}_{kind}' for reference,kind in risk_api.METHODS]
ERROR_COLUMNS = ['query_id','target_gene_id','target_gene_symbol','context_id','role','n_cells','true_error_rmse']
SEMANTIC_QC = {'raw_integer_atol':1e-6,'library_sum_rtol':0,'library_sum_atol':0,
               'failure':'ABORT_ENTIRE_OUTPUT_NO_SURVIVOR_SELECTION'}
DEFAULT_BUDGETS = {**biology.DEFAULT_BUDGETS,'max_endpoint_sum_bytes':512*1024**2}


def sha(path):
    return biology.sha(path)


def binding(entry, readonly=False):
    if not isinstance(entry,dict) or not entry.get('sha256') or not entry.get('path'):
        raise RuntimeError('Explicit path/SHA binding required')
    path=Path(entry['path']).resolve()
    if not path.is_file() or (readonly and path.stat().st_mode&0o222) or sha(path)!=entry['sha256']:
        raise RuntimeError(f'Frozen artifact missing, writable or hash differs: {path}')
    if 'bytes' in entry and path.stat().st_size!=entry['bytes']:
        raise RuntimeError('Frozen artifact size differs')
    return path


def bound_json(entry,readonly=False):
    return json.loads(binding(entry,readonly).read_text())


def artifact_registry(root,entry,tsv=False):
    root=Path(root).resolve();path=binding(entry)
    items=pd.read_csv(path,sep='\t',keep_default_na=False).to_dict('records') if tsv else json.loads(path.read_text())
    registry={}
    for item in items:
        name=item['path']
        if Path(name).name!=name or name in registry:
            raise RuntimeError('Artifact registry requires unique safe relative filenames')
        current=binding({'path':str(root/name),'sha256':item['sha256'],'bytes':int(item['bytes'])})
        if current.stat().st_size<=0:
            raise RuntimeError('Empty sealed artifact')
        registry[name]=current
    return registry


def score_table(path):
    table=pd.read_parquet(path) if Path(path).suffix=='.parquet' else pd.read_csv(path,sep='\t',keep_default_na=False,float_precision='round_trip')
    required=set(risk_api.QUERY+RISK_COLUMNS)
    if not required<=set(table) or not table.query_id.is_unique:
        raise RuntimeError('All six fixed candidate scores and unique query identities required')
    if any('true_error' in column or column in {'truth','target_CDF','query_truth'} for column in table):
        raise RuntimeError('Pretruth score seal cannot contain target truth/errors/CDF')
    if not set(table.role)<={'VALIDATION','TEST'} or set(table.context_id)!=set(CONTEXTS):
        raise RuntimeError('Registered pretruth query role/context coverage required')
    for column in RISK_COLUMNS:
        table[column]=pd.to_numeric(table[column].replace({'NaN':np.nan,'':np.nan}),errors='raise')
    if not np.isfinite(table[['P_only_ridge','P_only_hgb']]).all().all():
        raise RuntimeError('Every frozen query requires both finite P-only risk scores')
    if 'source_history_n' in table:
        history=pd.to_numeric(table.source_history_n,errors='raise')
        for column in ['Manual_ridge','Manual_hgb','Learned_ridge','Learned_hgb']:
            known=history>0
            if not np.isfinite(table.loc[known,column]).all() or not table.loc[~known,column].isna().all():
                raise RuntimeError('Frozen no-history public risks must be unsupported NaN, not zero')
    return table


def authorize_test(receipt_path):
    """All gates precede raw file checksums, footers and numeric reader calls."""
    receipt=biology.immutable_json(receipt_path)
    if (receipt.get('schema')!=SCHEMA or receipt.get('postseal_TEST_operation_authorized') is not True
            or receipt.get('test_numeric_materialization_permitted') is not True
            or receipt.get('authorized_numeric_roles')!=['TEST']
            or set(receipt.get('authorized_contexts',[]))!=set(CONTEXTS)):
        raise RuntimeError('Explicit immutable postseal TEST-only access operation required')
    if receipt.get('TEST_metadata_before_open')!='SEEN' or receipt.get('TEST_truth_before_open')!='CLOSED':
        raise RuntimeError('TEST metadata exposure is SEEN and truth must remain CLOSED before operation')
    if receipt.get('test_semantic_QC')!=SEMANTIC_QC or receipt.get('task_min_cells')!=30 or receipt.get('normalization_scale')!=4000:
        raise RuntimeError('Exact predetermined semantic QC, task count and CP4000 rules required')
    synthetic=receipt.get('synthetic_only') is True
    mdroot=Path(receipt['metadata_root']).resolve();rawroot=Path(receipt['raw_root']).resolve()
    if synthetic:
        if (not mdroot.is_relative_to((DOC/'synthetic').resolve()) or not rawroot.is_relative_to((DOC/'synthetic').resolve())
                or not (mdroot/'SYNTHETIC_FIXTURE_ONLY.json').is_file()):
            raise RuntimeError('Synthetic access applies only to generated test_open fixture roots')
    science=bound_json(receipt['scientific_contract'],True);science_sha=receipt['scientific_contract']['sha256']
    if not synthetic and science_sha!=METHOD_SHA:
        raise RuntimeError('Exact prospective scientific contract required')
    if science.get('normalization',{}).get('formula')!='mean_cell(log1p(4000 * raw_gene_UMI / official_full_library_total_counts))':
        raise RuntimeError('Frozen normalization formula differs')
    if science.get('normalization',{}).get('unknown_token_policy')!='fail_closed':
        raise RuntimeError('Unknown tokens must fail closed')
    overrides={str(Path(x['path']).resolve()):x for x in receipt.get('technical_dependency_overrides',[])}
    if set(overrides)-{str(Path(reader.v1.__file__).resolve())}:
        raise RuntimeError('Only previously amended structural reader dependency may be overridden')
    scientific_dependencies={}
    for item in science.get('immutable_dependencies',[]):
        effective=dict(item);override=overrides.get(str(Path(item['path']).resolve()))
        if override:
            if override.get('original_sha256')!=item['sha256'] or override.get('sha256')!=sha(reader.v1.__file__):
                raise RuntimeError('Structural technical amendment must bind original and exact executed primitive SHA')
            effective['sha256']=override['sha256']
        binding(effective);scientific_dependencies[Path(item['path']).name]=item['sha256']
    for name in META_NAMES:
        expected=receipt.get('metadata_bindings',{}).get(name)
        if not expected or sha(mdroot/name)!=expected:
            raise RuntimeError(f'Frozen metadata binding differs: {name}')
        if name in scientific_dependencies and expected!=scientific_dependencies[name]:
            raise RuntimeError('TEST operation cannot replace scientific metadata/split/endpoint dependency')
    implementation={'test_reader':Path(__file__),'numeric_backend':Path(reader.__file__),
        'structural_reader':Path(reader.v1.__file__),'scientific_loader':Path(biology.__file__),'risk_sealer':Path(risk_api.__file__)}
    for name,path in implementation.items():
        entry=receipt.get('implementation_bindings',{}).get(name,{})
        if Path(entry.get('path','')).resolve()!=path.resolve() or entry.get('sha256')!=sha(path):
            raise RuntimeError(f'Exact {name} implementation binding required')
    competence=bound_json(receipt['final_competence'])
    for name in ['relative_macro_error_gap','model_macro_rmse','baseline_macro_rmse','lower95_relative_gap','upper95_relative_gap']:
        if not isinstance(competence.get(name),(int,float)) or not np.isfinite(competence[name]):
            raise RuntimeError('Final competence numeric evidence must be present and finite')
    if (competence.get('status')!='PASS' or competence.get('passes_competence') is not True
            or competence.get('actual_context_count')!=2 or competence.get('noninferior_contexts')!=2
            or competence.get('noninferior_strata_fraction')!=1.0 or competence.get('stable_disadvantage') is not False
            or competence.get('eligible_validation_tasks')!=competence.get('evaluated_validation_tasks')
            or not competence.get('evaluated_validation_tasks')
            or competence.get('missing_predictions')!=0 or competence.get('nonfinite_predictions')!=0
            or competence.get('test_truth_read') is not False or competence.get('scientific_contract_sha256')!=science_sha
            or competence.get('bootstrap_replicates')!=5000 or competence.get('bootstrap_seed')!=20260929
            or competence.get('new_model_parameters_fitted') is not False
            or bool(competence.get('synthetic_only'))!=synthetic):
        raise RuntimeError('Final actual frozen competence PASS required; incomplete/failed gate cannot open TEST')
    if (competence['baseline_macro_rmse']<=0 or competence['model_macro_rmse']<0
            or competence['model_macro_rmse']>1.02*competence['baseline_macro_rmse']
            or 1.+competence['lower95_relative_gap']>1.02):
        raise RuntimeError('Competence evidence contradicts the fixed PASS decision')
    registry_path=binding(receipt['data_role_registry'])
    role_registry=pd.read_csv(registry_path,keep_default_na=False)
    test_rows=role_registry[role_registry.task_range.str.contains('gene roles TEST',regex=False)]
    if (len(test_rows)!=2 or set(test_rows.context)!=set(CONTEXTS) or not test_rows.role.eq('SEEN').all()
            or not test_rows.metadata_seen.astype(str).eq('True').all()
            or not test_rows.result_seen.astype(str).eq('False').all()
            or not test_rows.test_truth_access_status.str.startswith('CLOSED').all()
            or not test_rows.pristine_confirmation_claim.astype(str).eq('False').all()):
        raise RuntimeError('Pinned TEST role registry must record SEEN metadata and CLOSED untouched truth')
    object_manifest=bound_json(receipt['forty_object_manifest'])
    identities=json.loads((mdroot/'FILE_METADATA_IDENTITY.json').read_text())
    expected_files={item['path']:item for item in object_manifest.get('files',[])}
    if len(expected_files)!=(2 if synthetic else 40) or len(identities)!=len(expected_files):
        raise RuntimeError('Exact registered immutable object manifest required; no shard replacements')
    if not synthetic and object_manifest.get('release_sha')!='53a5bc98d49247bcf967500292575c3d3602de31':
        raise RuntimeError('Registered publisher release differs')
    if len({i['path'] for i in identities})!=len(identities):raise RuntimeError('Duplicate object identity')
    for identity in identities:
        item=expected_files.get(identity['path'],{})
        if (identity['publisher_LFS_sha256']!=item.get('lfs_sha256') or identity['publisher_whole_file_bytes']!=item.get('bytes')
                or identity['context'] not in CONTEXTS or identity['row_groups']!=1):
            raise RuntimeError('Metadata row identities and registered complete object tuples differ')
        binding({'path':identity['metadata_parquet_path'],'sha256':identity['metadata_parquet_sha256']})
    full,endpoint,endpoint_columns,mapping=biology.gene_axes(mdroot,synthetic)
    risk_root=Path(receipt['risk_seal_root']).resolve()
    if binding(receipt['risk_seal_manifest'])!=risk_root/'RISK_SEAL_MANIFEST.json':raise RuntimeError('Exact risk seal manifest path required')
    risk_manifest=bound_json(receipt['risk_seal_manifest'])
    if (risk_manifest.get('status')!='SEALED_PRETRUTH' or risk_manifest.get('scientific_contract_sha256')!=science_sha
            or risk_manifest.get('source_only_risk_models')!=6 or risk_manifest.get('query_errors_used') is not False
            or risk_manifest.get('target_CDF_fit') is not False or risk_manifest.get('new_learners_or_parameter_fits')!=0
            or risk_manifest.get('query_truth_input_or_dummy_labels') is not False
            or risk_manifest.get('risk_code_sha256')!=sha(risk_api.__file__)):
        raise RuntimeError('All-candidate frozen pretruth Source risk seal required')
    if (risk_manifest.get('TEST_truth_status')!='CLOSED' or risk_manifest.get('TEST_identity_metadata_status')!='SEEN_METADATA_ONLY'
            or risk_manifest.get('data_role_registry_sha256')!=receipt['data_role_registry']['sha256']):
        raise RuntimeError('Risk seal must bind the same TEST metadata-SEEN/truth-CLOSED registry')
    risk_registry=artifact_registry(risk_root,receipt['risk_seal_artifacts'])
    required={'RISK_SEAL_MANIFEST.json','PRETRUTH_RISKS.tsv','P_ONLY_FEATURES.parquet','SOURCE_HISTORY_PAIR_FEATURES.parquet',
        'SOURCE_PRIOR_WEIGHTS.tsv',*[f'{r}_P_PUBLIC_FEATURES.parquet' for r in ['Uniform','Manual','Learned']],
        *[f'{r}_PRIOR_EFFECTS.npy' for r in ['Uniform','Manual','Learned']]}
    if not required<=set(risk_registry):raise RuntimeError('Core/prior/config/score artifacts are not all sealed')
    sealed_scores=score_table(risk_registry['PRETRUTH_RISKS.tsv'])
    source_bindings=risk_manifest.get('source_core_bindings',[])
    source_paths={Path(x['path']).name:binding(x) for x in source_bindings}
    if not {'MODEL_MANIFEST.json','SOURCE_FEATURE_MANIFEST.json','GENE_MANIFEST.csv','GENE_IDS.json','SOURCE_ERROR_CDF.json'}<=set(source_paths):
        raise RuntimeError('Frozen Source fitted parameter/config/axis pins missing')
    models=json.loads(source_paths['MODEL_MANIFEST.json'].read_text()).get('models',[])
    risks=[x for x in models if x.get('role')=='Source_risk'];public=[x for x in models if x.get('role')=='Public_biology_retrieval']
    if len(risks)!=6 or len(public)!=1 or {(x.get('reference'),x.get('learner')) for x in risks}!=set(risk_api.METHODS):
        raise RuntimeError('Exactly six frozen Source risk models and one Public model required')
    for model in models:
        name=model['path']
        if name not in source_paths or sha(source_paths[name])!=model.get('sha256'):
            raise RuntimeError('Frozen Source model parameter SHA missing or differs')
        columns=risk_api.PAIR if model['role']=='Public_biology_retrieval' else (risk_api.P if model['reference']=='P_only' else risk_api.P+risk_api.PUBLIC)
        if model.get('columns')!=columns or model.get('numeric_model_width_with_missing_flags')!=2*len(columns):
            raise RuntimeError('Frozen Source model feature/config width differs')
        if model['role']=='Public_biology_retrieval' and (model.get('target')!='biological_transfer_rmse' or model.get('predict_clip') is not False):
            raise RuntimeError('Public Source model must preserve raw transfer RMSE and clip=False')
        if model['role']=='Source_risk' and model.get('target')!='Source_per_upstream_per_context_midrank_CDF':
            raise RuntimeError('Risk parameter target must remain historical Source CDF, never target TEST')
    source_feature=json.loads(source_paths['SOURCE_FEATURE_MANIFEST.json'].read_text())
    if source_feature.get('risk_P_columns')!=risk_api.P or source_feature.get('risk_public_columns')!=risk_api.PUBLIC or source_feature.get('public_pair_columns')!=risk_api.PAIR:
        raise RuntimeError('Fixed Source feature contract differs')
    source_axis=pd.read_csv(source_paths['GENE_MANIFEST.csv'])
    if source_axis[['orion_ensembl_id','gene_name']].to_dict('records')!=endpoint[['orion_ensembl_id','gene_name']].to_dict('records'):
        raise RuntimeError('Frozen Source risk and TEST endpoint axis differ')
    for name in ['MODEL_MANIFEST.json','SOURCE_FEATURE_MANIFEST.json']:
        if name in scientific_dependencies and sha(source_paths[name])!=scientific_dependencies[name]:
            raise RuntimeError('Source model/config seal differs from frozen scientific dependency')
    comparison_manifest=bound_json(receipt['risk_comparison_manifest'],True)
    if (comparison_manifest.get('schema')!=risk_api.SCHEMA or comparison_manifest.get('pretruth_inference_authorized') is not True
            or comparison_manifest.get('target_errors_or_CDF_used') is not False
            or comparison_manifest.get('risk_code_sha256')!=sha(risk_api.__file__)
            or comparison_manifest.get('source_core_bindings')!=source_bindings
            or risk_manifest.get('comparison_manifest_sha256')!=receipt['risk_comparison_manifest']['sha256']):
        raise RuntimeError('Pretruth risk comparison/model configuration freeze differs')
    if comparison_manifest.get('scientific_contract_sha256')!=science_sha:raise RuntimeError('Risk comparison scientific hash differs')
    if comparison_manifest.get('data_role_registry_sha256')!=receipt['data_role_registry']['sha256']:
        raise RuntimeError('Risk comparison and TEST operation role registry pins differ')
    fit_root=Path(receipt['lm_fit_root']).resolve();fit_run_path=binding(receipt['lm_fit_run_manifest'])
    if fit_run_path!=fit_root/'RUN_MANIFEST.tsv':raise RuntimeError('Full LM fit run manifest path differs')
    run=pd.read_csv(fit_run_path,sep='\t',keep_default_na=False)
    if set(run.context_id)!=set(CONTEXTS) or len(run)!=2:raise RuntimeError('Both full LM context fits must be sealed')
    for context in CONTEXTS:
        files=artifact_registry(fit_root/context,receipt['lm_fit_artifact_registries'][context],True)
        if not {'MODEL.rds','SIMPLE_BASELINES.rds','BASELINE_SELECTION.tsv','PREDICTIONS_DELTA.tsv.gz','OUTPUT_GENE_AXIS.tsv','QUERY_IDENTITIES.tsv','STATUS.tsv'}<=set(files):
            raise RuntimeError('Missing full LM fit/prediction/config/parameter artifact seal')
        status=dict(pd.read_csv(files['STATUS.tsv'],sep='\t',keep_default_na=False)[['key','value']].itertuples(index=False,name=None))
        for key,value in {'status':'PASS','mode':'fit','context_id':context,'pca_dim':'10','ridge_penalty':'0.1','seed':'1','query_truth_read':'FALSE'}.items():
            if str(status.get(key))!=value:raise RuntimeError('Full frozen LM fit/config/status differs')
        if run[run.context_id==context].iloc[0].model_sha256!=sha(files['MODEL.rds']):raise RuntimeError('Full fitted LM model hash differs')
        context_entry=[x for x in comparison_manifest.get('contexts',[]) if x.get('context_id')==context]
        if len(context_entry)!=1 or context_entry[0].get('upstream_model_sha256')!=sha(files['MODEL.rds']):
            raise RuntimeError('Risk prediction seal must use the identical full fitted LM parameters')
    # This reused function reads only already frozen predictions/identities and
    # TRAIN control means, verifies their hashes/axes/status, and fits nothing.
    queries,delta,controls,lm_bindings=risk_api.load_inputs(comparison_manifest,{'axis':endpoint})
    if risk_manifest.get('LM_prediction_and_control_bindings')!=lm_bindings:raise RuntimeError('LM/control/parameter seal lineage differs')
    if not np.isfinite(delta).all():raise RuntimeError('Frozen LM prediction is nonfinite')
    if any(np.asarray(controls[c]).shape!=(len(endpoint),) or not np.isfinite(controls[c]).all() or (controls[c]<0).any() for c in CONTEXTS):
        raise RuntimeError('Frozen own-context TRAIN control measurements must be finite/nonnegative/aligned')
    if sealed_scores[risk_api.QUERY].to_dict('records')!=queries.to_dict('records'):
        raise RuntimeError('Every frozen LM query must have all sealed risk candidate identities')
    spec=bound_json(receipt['pretruth_comparison']['spec'],True);comparison_scores_path=binding(receipt['pretruth_comparison']['scores'],True)
    evaluator=ROOT/'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py'
    if not evaluator.is_file() or spec.get('evaluator_code_sha256')!=sha(evaluator):
        raise RuntimeError('Actual comparison evaluator implementation must be frozen before TEST opening')
    if (spec.get('scientific_contract_sha256')!=science_sha or spec.get('risk_seal_manifest_sha256')!=receipt['risk_seal_manifest']['sha256']
            or spec.get('scores_sha256')!=sha(comparison_scores_path) or spec.get('test_truth_or_errors_used') is not False
            or spec.get('new_fits')!=0 or not set(RISK_COLUMNS)<=set(spec.get('candidate_score_columns',[]))):
        raise RuntimeError('All-candidate comparison scores/config must be frozen before truth')
    comparison_scores=score_table(comparison_scores_path)
    if comparison_scores[risk_api.QUERY].to_dict('records')!=queries.to_dict('records'):
        raise RuntimeError('Frozen comparison query scope differs')
    for column in RISK_COLUMNS:
        if not np.array_equal(comparison_scores[column].to_numpy(),sealed_scores[column].to_numpy(),equal_nan=True):
            raise RuntimeError('Frozen Source risk score changed in comparison')
    budgets=dict(DEFAULT_BUDGETS);budgets.update(receipt.get('resource_budgets',{}))
    if any(k not in DEFAULT_BUDGETS or type(v) is not int or v<=0 or v>DEFAULT_BUDGETS[k] for k,v in budgets.items()):
        raise RuntimeError('Positive resource budgets within fixed ceilings required')
    return {'receipt':receipt,'receipt_path':Path(receipt_path).resolve(),'science_sha256':science_sha,'synthetic':synthetic,
        'metadata_root':mdroot,'raw_root':rawroot,'identities':identities,'full':full,'endpoint':endpoint,
        'endpoint_columns':endpoint_columns,'mapping':mapping,'queries':queries,'delta':delta,'controls':controls,
        'risk_manifest':risk_manifest,'lm_bindings':lm_bindings,'budgets':budgets}


@contextmanager
def file_deadline(seconds):
    old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    def timed_out(signum,frame):raise TimeoutError('TEST file deadline exceeded; entire output remains incomplete')
    signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,*old_timer)


def evaluate_test(receipt_path,output,forbidden_bytes=()):
    verified=authorize_test(receipt_path)
    receipt=verified['receipt'];output=Path(output).resolve()
    if output!=Path(receipt['output_path']).resolve() or output.exists():
        raise RuntimeError('Exact new operation output path required; no overwrite')
    stage=output.with_name(output.name+f'.incomplete.{os.getpid()}');stage.mkdir(parents=True)
    started=time.monotonic();test=verified['queries'][verified['queries'].role.eq('TEST')].copy().reset_index(drop=True)
    if test.empty or not test.query_id.is_unique:raise RuntimeError('Frozen TEST query scope required')
    n_genes=len(verified['endpoint']);shape=(len(test),n_genes);budgets=verified['budgets']
    if np.prod(shape,dtype=np.int64)*8>budgets['max_endpoint_sum_bytes']:raise RuntimeError('Endpoint-only sum memory budget exceeded')
    sums=np.zeros(shape,dtype=np.float64);cell_counts=np.zeros(len(test),dtype=np.uint64)
    by_unit={row.query_id:i for i,row in enumerate(test.itertuples(index=False))}
    token_to_endpoint=np.full(len(verified['full']),-1,dtype=np.int64);token_to_endpoint[verified['endpoint_columns']]=np.arange(n_genes)
    token_forbidden,value_forbidden=forbidden_bytes if forbidden_bytes else ((),())
    file_audits=[];opened=0;token_rows_opened=0
    try:
        for file_number,identity in enumerate(verified['identities']):
            with file_deadline(budgets['max_seconds_per_file']):
                # TEST task counts/coverage are first computed here, after every
                # legal-open prerequisite passed. No n>=30 preselection occurs.
                md,mask=biology.read_metadata(identity,{'TEST'})
                for row in np.flatnonzero(mask):
                    unit=md['biological_unit'][row]
                    if unit not in by_unit:raise RuntimeError('TEST unit missing from frozen all-query predictions; no silent attrition')
                    query=test.iloc[by_unit[unit]];mapped=verified['mapping'].get(md['gene_target'][row])
                    if (mapped is None or query.target_gene_id!=mapped.ensembl_id or query.target_gene_symbol!=md['gene_target'][row]
                            or query.context_id!=identity['context']):raise RuntimeError('Exact frozen TEST target/context/axis identity differs')
                if identity['rows']>budgets['max_cells_per_file']:raise RuntimeError('TEST file row budget exceeded')
                path=biology.checked_child(verified['raw_root'],identity['path'])
                if not path.is_file() or path.stat().st_size!=identity['publisher_whole_file_bytes'] or sha(path)!=identity['publisher_LFS_sha256']:
                    raise RuntimeError('Whole registered raw object identity differs before numeric read')
                footer=reader.fastparquet.ParquetFile(path)
                if len(footer.row_groups)!=1 or footer.row_groups[0].num_rows!=identity['rows']:
                    raise RuntimeError('Pinned TEST physical rowgroup identity differs')
                for name,kind in [('gene_token_id',reader.pt.Type.INT64),('gene_expression',reader.pt.Type.DOUBLE)]:
                    leaves=[c.meta_data for c in footer.row_groups[0].columns if c.meta_data.path_in_schema[0]==name]
                    if len(leaves)!=1 or leaves[0].type!=kind:
                        raise RuntimeError('Exact INT64 token and DOUBLE expression schema required')
                offsets=np.full(identity['rows'],-1,dtype=np.int64);lengths=np.full(identity['rows'],-1,dtype=np.int64)
                spool_path=stage/f'tokens_{file_number}.uint32';written=0;seen=[];tt=biology.trace_new(budgets)
                with spool_path.open('xb') as spool:
                    for row,tokens in reader.iter_selected_numeric_lists(path,'gene_token_id',mask,tt,token_forbidden):
                        token_rows_opened+=1
                        if tokens is None or not tokens or any(type(x) is not int for x in tokens) or len(tokens)>len(token_to_endpoint):
                            raise RuntimeError('Null/empty/noninteger TEST token list')
                        ids=np.asarray(tokens,dtype=np.int64)
                        if (ids<0).any() or (ids>=len(token_to_endpoint)).any() or len(np.unique(ids))!=len(ids):
                            raise RuntimeError('Unknown/duplicate/unmapped TEST token; no imputation')
                        offsets[row]=written;lengths[row]=len(ids);written+=len(ids)
                        if written*4>budgets['max_token_spool_bytes_per_file']:raise RuntimeError('TEST token spool budget exceeded')
                        spool.write(ids.astype('<u4',copy=False).tobytes());seen.append(row)
                        biology.check_memory(budgets)
                expected=np.flatnonzero(mask).tolist()
                if seen!=expected:raise RuntimeError('Token iterator did not exhaust exact TEST mask')
                mm=np.memmap(spool_path,dtype='<u4',mode='r',shape=(written,)) if written else None
                seen=[];vt=biology.trace_new(budgets)
                for row,values in reader.iter_selected_numeric_lists(path,'gene_expression',mask,vt,value_forbidden):
                    opened+=1
                    if values is None or len(values)!=lengths[row] or not values or any(x is None for x in values):
                        raise RuntimeError('Null/empty/truncated TEST measurement; abort entire output')
                    raw=np.asarray(values,dtype=np.float64)
                    if not np.isfinite(raw).all() or (raw<0).any() or np.max(np.abs(raw-np.rint(raw)))>SEMANTIC_QC['raw_integer_atol']:
                        raise RuntimeError('TEST UMI must be finite nonnegative integer measurements')
                    total=float(md['total_counts'][row])
                    if float(raw.sum(dtype=np.float64))!=total:
                        raise RuntimeError('TEST raw full-library sum differs from official total; no denominator/survivor change')
                    ids=np.asarray(mm[offsets[row]:offsets[row]+lengths[row]],dtype=np.int64)
                    endpoint_ids=token_to_endpoint[ids];keep=endpoint_ids>=0
                    normalized=np.log1p(4000.*raw[keep]/total)
                    if not np.isfinite(normalized).all():raise RuntimeError('Nonfinite endpoint normalization')
                    unit=by_unit[md['biological_unit'][row]]
                    sums[unit,endpoint_ids[keep]]+=normalized;cell_counts[unit]+=1;seen.append(row)
                    biology.check_memory(budgets)
                if seen!=expected:raise RuntimeError('Expression iterator did not exhaust exact TEST mask')
                if mm is not None:del mm
                spool_path.unlink()
                # Counts for unselected/private rows/pages are not persisted.
                safe=lambda trace:{k:v for k,v in trace.items() if k.startswith('max_') or k in {'physical_numeric_chunk_bytes_read','opaque_numeric_bytes_decompressed','numeric_materialization_count','authorized_dictionary_lookups'}}
                file_audits.append({'path':identity['path'],'raw_sha256':identity['publisher_LFS_sha256'],
                    'metadata_sha256':identity['metadata_parquet_sha256'],'TEST_cells':len(expected),
                    'token_iterator_exhausted':True,'expression_iterator_exhausted':True,'token_trace':safe(tt),'expression_trace':safe(vt)})
        test['n_cells']=cell_counts;test['eligible_n_ge30_after_legal_open']=cell_counts>=30
        test.to_csv(stage/'TEST_TASK_QC.tsv',sep='\t',index=False)
        eligible=np.flatnonzero(cell_counts>=30);effects=np.empty((len(eligible),n_genes),dtype=np.float64);errors=[]
        predictions={q:i for i,q in enumerate(verified['queries'].query_id)}
        for outrow,index in enumerate(eligible):
            query=test.iloc[index];effect=sums[index]/float(cell_counts[index])-verified['controls'][query.context_id]
            pred=verified['delta'][predictions[query.query_id]]
            effects[outrow]=effect
            errors.append({**query[risk_api.QUERY].to_dict(),'n_cells':int(cell_counts[index]),
                'true_error_rmse':float(np.sqrt(np.mean((pred-effect)**2,dtype=np.float64)))})
        error_frame=pd.DataFrame(errors,columns=ERROR_COLUMNS)
        if error_frame.empty or not np.isfinite(error_frame.true_error_rmse).all():raise RuntimeError('No eligible finite TEST evaluation tasks')
        pq.write_table(pa.Table.from_pandas(error_frame,preserve_index=False),stage/'TEST_TASK_ERRORS.parquet')
        # Endpoint truth is legal only after the exact postseal receipt. No full
        #38606-by-task sums/effects are allocated or saved.
        np.save(stage/'TEST_ENDPOINT_EFFECTS.npy',effects,allow_pickle=False)
        result={'schema':OUTPUT_SCHEMA,'status':'COMPLETE','synthetic_only':verified['synthetic'],
            'TEST_access_receipt_sha256':sha(receipt_path),'TEST_access_receipt_path':str(Path(receipt_path).resolve()),
            'test_reader_sha256':sha(__file__),'numeric_backend_sha256':sha(reader.__file__),
            'scientific_contract_sha256':verified['science_sha256'],'forty_object_manifest_sha256':receipt['forty_object_manifest']['sha256'],
            'endpoint_gene_manifest_sha256':receipt['metadata_bindings']['GENE_MANIFEST.csv'],
            'metadata_role_before_open':'SEEN','truth_status_before_open':'CLOSED','truth_status_after_open':'OPEN_REGISTERED_TEST_SCOPE',
            'role_registry_sha256':receipt['data_role_registry']['sha256'],'final_competence_sha256':receipt['final_competence']['sha256'],
            'risk_seal_manifest_sha256':receipt['risk_seal_manifest']['sha256'],'risk_seal_artifacts_sha256':receipt['risk_seal_artifacts']['sha256'],
            'risk_seal_manifest':receipt['risk_seal_manifest'],'pretruth_comparison':receipt['pretruth_comparison'],
            'PRETRUTH_COMPARISON_spec_sha256':receipt['pretruth_comparison']['spec']['sha256'],
            'PRETRUTH_COMPARISON_scores_sha256':receipt['pretruth_comparison']['scores']['sha256'],
            'Source_parameter_bindings':verified['risk_manifest']['source_core_bindings'],'LM_prediction_model_control_bindings':verified['lm_bindings'],
            'TEST_TASK_ERRORS_path':str(output/'TEST_TASK_ERRORS.parquet'),'TEST_TASK_ERRORS_sha256':sha(stage/'TEST_TASK_ERRORS.parquet'),
            'truth_errors':{'path':str(output/'TEST_TASK_ERRORS.parquet'),'sha256':sha(stage/'TEST_TASK_ERRORS.parquet')},
            'endpoint_genes':n_genes,'TEST_numeric_cells_opened':opened,'TEST_token_rows_materialized':token_rows_opened,'frozen_TEST_queries':len(test),'eligible_TEST_tasks':len(errors),
            'task_QC_applied_only_after_legal_open':True,'task_min_cells':30,'all_exact_TEST_iterators_exhausted':True,
            'semantic_QC':SEMANTIC_QC,'normalization':'equal-cell mean(log1p(4000*rawUMI/officialfulltotal)) minus frozen own TRAIN NTC',
            'new_fits':0,'target_CDF_fitted':False,'target_CDF_fit':False,'risk_or_error_based_cell_selection':False,'denominator_changed':False,
            'file_audits':file_audits,'elapsed_seconds':time.monotonic()-started,'finished_utc':datetime.now(timezone.utc).isoformat()}
        biology.json_write(stage/'TRUTH_READER_RECEIPT.json',result)
    except Exception as error:
        biology.json_write(stage/'ABORTED_TEST_OPEN.json',{'status':'ABORTED_AFTER_AUTHORIZATION_NO_COMMITTED_EVALUATION',
            'test_access_receipt_sha256':sha(receipt_path),'expression_cells_opened':opened,'token_rows_materialized':token_rows_opened,'error':str(error),
            'whole_file_completion_claim':False,'denominator_or_survivor_selection_changed':False})
        raise
    # Every committed result is immutable, including the hash ledger.
    artifacts=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(stage.iterdir()) if p.is_file() and p.name!='ARTIFACT_HASHES.json']
    (stage/'ARTIFACT_HASHES.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    for path in stage.iterdir():path.chmod(0o444)
    stage.rename(output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    validate=sub.add_parser('validate');validate.add_argument('--receipt',type=Path,required=True)
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('--receipt',type=Path,required=True);evaluate.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='validate':
        state=authorize_test(args.receipt);print(json.dumps({'status':'ALL_POSTSEAL_TEST_GATES_PASS_RAW_UNOPENED','synthetic_only':state['synthetic'],'receipt_sha256':sha(args.receipt)}))
    else:print(json.dumps(evaluate_test(args.receipt,args.output),indent=2))


if __name__=='__main__':main()
