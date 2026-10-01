#!/usr/bin/env python3
"""Freeze pretruth comparators, then evaluate only a separately receipted truth.

No expression reader, learner, target CDF or target score selection exists here.
All methods/cohorts/statistics are fixed before truth by prepare-comparison.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import metrics
from tools.scripts import seal_safeconf_orion_source_risk_agent as risk

DOC=risk.DOC.parent/'risk_evaluation'
SCHEMA='safeconf_orion_frozen_risk_evaluation_v1'
CONTEXTS=('HCT116','HEK293T')
SEED=20260929
REPLICATES=5000
METRICS=('utility20','spearman','aurc','error_at_10','error_at_20','error_at_50','high_risk_miss_rate')
BASE_METHODS=['Magnitude','P_only_ridge','P_only_hgb','Manual_ridge','Manual_hgb','Learned_ridge','Learned_hgb',
              'Manual_WeightedHistoryDistance','Learned_WeightedHistoryDistance']
OPTIONAL=['Uniform_DirectRMSE','Manual_DirectRMSE','Learned_DirectRMSE']
SUPPORT='NegativeSourceHistorySupport'
QUERY=risk.QUERY
PRIMARY=('Learned_hgb','Learned_WeightedHistoryDistance')
MANUAL=('Manual_hgb','Manual_WeightedHistoryDistance')
TEST_READER=ROOT/'tools/scripts/evaluate_safeconf_orion_guarded_test_agent.py'
NUMERIC_BACKEND=ROOT/'tools/scripts/probe_safeconf_private_parquet_vectorized_agent.py'
TRUTH_SCHEMA='safeconf_orion_postseal_test_truth_reader_v1'
OPERATION_SCHEMA='safeconf_orion_explicit_postseal_test_access_v1'
SEMANTIC_QC={'raw_integer_atol':1e-6,'library_sum_rtol':0,'library_sum_atol':0,
             'failure':'ABORT_ENTIRE_OUTPUT_NO_SURVIVOR_SELECTION'}


def binding(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':risk.sha(path),'bytes':path.stat().st_size}


def checked_bind(value):
    path=Path(value['path'])
    if not path.is_file() or risk.sha(path)!=value['sha256']:
        raise RuntimeError('Frozen comparison/truth artifact hash differs')
    return path


def json_immutable(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_mode&0o222:raise RuntimeError('Immutable read-only receipt/spec required')
    return json.loads(path.read_text())


def guarded_truth_receipt(receipt,spec):
    expected={'schema':TRUTH_SCHEMA,'status':'COMPLETE','all_exact_TEST_iterators_exhausted':True,
        'task_QC_applied_only_after_legal_open':True,'task_min_cells':30,
        'metadata_role_before_open':'SEEN','truth_status_before_open':'CLOSED',
        'truth_status_after_open':'OPEN_REGISTERED_TEST_SCOPE','semantic_QC':SEMANTIC_QC,
        'new_fits':0,'target_CDF_fit':False,'risk_or_error_based_cell_selection':False,'denominator_changed':False}
    if any(receipt.get(key)!=value for key,value in expected.items()):
        raise RuntimeError('Exact completed guarded truth-reader schema/scope required')
    operation_path=Path(receipt.get('TEST_access_receipt_path',''))
    operation=json_immutable(operation_path)
    if risk.sha(operation_path)!=receipt.get('TEST_access_receipt_sha256'):
        raise RuntimeError('Actual immutable TEST operation receipt SHA differs')
    if (operation.get('schema')!=OPERATION_SCHEMA or operation.get('postseal_TEST_operation_authorized') is not True
        or operation.get('test_numeric_materialization_permitted') is not True
        or operation.get('authorized_numeric_roles')!=['TEST'] or set(operation.get('authorized_contexts',[]))!=set(CONTEXTS)
        or operation.get('TEST_metadata_before_open')!='SEEN' or operation.get('TEST_truth_before_open')!='CLOSED'
        or operation.get('test_semantic_QC')!=SEMANTIC_QC or operation.get('task_min_cells')!=30):
        raise RuntimeError('Explicit authorized postseal TEST-only operation required')
    for name,path,receipt_key in [('test_reader',TEST_READER,'test_reader_sha256'),('numeric_backend',NUMERIC_BACKEND,'numeric_backend_sha256')]:
        entry=operation.get('implementation_bindings',{}).get(name,{})
        if Path(entry.get('path','')).resolve()!=path.resolve() or entry.get('sha256')!=risk.sha(path) or receipt.get(receipt_key)!=risk.sha(path):
            raise RuntimeError('Executed guarded TEST reader/backend code binding differs')
    if receipt.get('scientific_contract_sha256')!=spec.get('scientific_contract_sha256'):
        raise RuntimeError('Guarded truth and frozen comparison scientific contracts differ')
    science=operation.get('scientific_contract',{})
    science_path=checked_bind(science)
    json_immutable(science_path)
    if science.get('sha256')!=receipt.get('scientific_contract_sha256'):
        raise RuntimeError('Actual TEST operation scientific contract binding differs')
    for key in ['pretruth_comparison','risk_seal_manifest']:
        if operation.get(key)!=receipt.get(key):raise RuntimeError('Operation and completed reader seal bindings differ')


def write_json(path,value):
    risk.write_json(path,value)


def prepare_comparison(risk_seal,output):
    """Only prediction-time risk features and scores are read, never truth."""
    root=Path(risk_seal).resolve();output=Path(output).resolve()
    if output.exists():raise RuntimeError('New pretruth comparison version required')
    seal=json.loads((root/'RISK_SEAL_MANIFEST.json').read_text())
    if seal.get('status')!='SEALED_PRETRUTH' or seal.get('query_truth_input_or_dummy_labels') is not False or seal.get('target_CDF_fit') is not False:
        raise RuntimeError('Pretruth Source-only seal required')
    artifacts={x['path']:x for x in json.loads((root/'ARTIFACT_HASHES.json').read_text())}
    for name in ['PRETRUTH_RISKS.tsv','Manual_P_PUBLIC_FEATURES.parquet','RISK_SEAL_MANIFEST.json']:
        entry=artifacts.get(name,{})
        if entry.get('sha256')!=risk.sha(root/name):raise RuntimeError('Sealed input hash differs')
    scores=pd.read_csv(root/'PRETRUTH_RISKS.tsv',sep='\t',keep_default_na=False)
    if not set(QUERY+['source_history_n','history_status']+BASE_METHODS)<=set(scores.columns) or not scores.query_id.is_unique:
        raise RuntimeError('All frozen Source risk candidates and identity metadata required')
    if not set(scores.role)<={'VALIDATION','TEST'} or not set(scores.context_id)<=set(CONTEXTS):
        raise RuntimeError('Registered held-out identities required')
    manual=pd.read_parquet(root/'Manual_P_PUBLIC_FEATURES.parquet')
    if len(manual)!=len(scores) or list(manual.columns)!=risk.P+risk.PUBLIC or 'log_history_support' not in manual:
        raise RuntimeError('Exact Source PUBLIC feature contract/order required')
    methods=BASE_METHODS+[x for x in OPTIONAL if x in scores]
    for name in methods:scores[name]=pd.to_numeric(scores[name].replace({'NaN':np.nan,'':np.nan}),errors='raise')
    scores[SUPPORT]=-manual.log_history_support.to_numpy(float)
    methods.append(SUPPORT)
    known=scores.source_history_n.to_numpy(int)>0
    if not np.isnan(scores.loc[~known,SUPPORT]).all():raise RuntimeError('Missing history cannot become fake support zero')
    if not np.isfinite(scores.loc[known,SUPPORT]).all():raise RuntimeError('Known Source support must be finite')
    common=known&np.isfinite(scores[methods].to_numpy(float)).all(axis=1)
    scores['PRETRUTH_PRIMARY_COMMON_ELIGIBLE']=common
    output.mkdir(parents=True)
    scores_path=output/'PRETRUTH_COMPARISON_SCORES.parquet'
    scores.to_parquet(scores_path,index=False)
    pairs=list(itertools.combinations(methods,2))
    spec={'schema':SCHEMA,'status':'FROZEN_PRETRUTH_COMPARISON','evaluator_code_sha256':risk.sha(__file__),
      'scientific_contract_sha256':seal.get('scientific_contract_sha256'),
      'risk_seal_manifest_sha256':risk.sha(root/'RISK_SEAL_MANIFEST.json'),
      'scores_sha256':risk.sha(scores_path),'candidate_score_columns':methods,'test_truth_or_errors_used':False,
      'risk_seal_root':str(root),'risk_seal_manifest':binding(root/'RISK_SEAL_MANIFEST.json'),
      'risk_seal_scores':binding(root/'PRETRUTH_RISKS.tsv'),'risk_seal_artifact_hashes':binding(root/'ARTIFACT_HASHES.json'),
      'support_feature':binding(root/'Manual_P_PUBLIC_FEATURES.parquet'),
      'support_fixed_formula':'NegativeSourceHistorySupport = -log_history_support; prediction-time historical Source bank support only',
      'target_cell_count_used_as_risk_score':False,'comparison_scores':binding(scores_path),
      'methods':methods,'primary_pair':list(PRIMARY),'prespecified_manual_pair':list(MANUAL),
      'all_nominal_paired_comparisons':[list(x) for x in pairs],
      'cohorts':{'primary_common':'TEST,n_cells>=30,knownsamegeneSourcehistory,allfixedmethodsfinite; identical tasks for everymethod',
                 'all_prediction_only':'TEST,n_cells>=30,finite Magnitude/P-onlyRidge/HGB,allhistorycoverage',
                 'no_source_history_prediction_only':'sameallPcohort,source_history_n==0'},
      'metrics':list(METRICS),'metric_direction':{'utility20':'higher','spearman':'higher','aurc':'lower','error_at_10':'lower','error_at_20':'lower','error_at_50':'lower','high_risk_miss_rate':'lower'},
      'context_macro':'unweighted mean of BOTH predefined contexts; not valid if either context has fewerthan20tasks',
      'minimum_context_tasks':20,'minimum_treated_cells':30,'bootstrap_replicates':REPLICATES,'bootstrap_seed':SEED,
      'bootstrap_unit':'biological target_gene_id cluster; resample complete blocks jointly across both contexts',
      'CIs':'nominal percentile95,allprespecifiedpairs reported; no posttruthwinner selection',
      'ties':'frozen query_id lexicographic; ceil(.2*n) for highrisk/oracle and coverage10/20/50',
      'target_CDF_fit':False,'new_fits':0,'TEST_metadata_status':'SEEN_METADATA_ONLY','TEST_truth_status_at_freeze':'CLOSED',
      'whole_method_pristine_claim':False,'source_core_and_LM_prediction_parameter_bindings':seal.get('source_core_bindings',[])+seal.get('LM_prediction_and_control_bindings',[])}
    write_json(output/'COMPARISON_SPEC.json',spec)
    for p in output.iterdir():p.chmod(0o444)
    return spec


def metric_values(frame,methods):
    out={}
    for context in CONTEXTS:
        part=frame[frame.context_id.eq(context)]
        out[context]={name:dict.fromkeys(METRICS,np.nan) for name in methods}
        if len(part)<20:continue
        for name in methods:
            measured=metrics(part.assign(task_id=part.query_id),part[name].to_numpy(float))
            out[context][name]={key:measured[key] for key in METRICS}
    out['macro']={name:{key:float(np.mean([out[c][name][key] for c in CONTEXTS])) for key in METRICS} for name in methods}
    return out


def cohort_statistics(frame,methods,replicates=REPLICATES,seed=SEED):
    point=metric_values(frame,methods)
    original_valid={context:int(frame.context_id.eq(context).sum())>=20 for context in CONTEXTS}
    original_valid['macro']=all(original_valid.values())
    clusters=frame.target_gene_id.astype(str).unique()
    blocks={gene:np.flatnonzero(frame.target_gene_id.astype(str).eq(gene)) for gene in clusters}
    draws=np.full((replicates,len(CONTEXTS)+1,len(methods),len(METRICS)),np.nan)
    rng=np.random.default_rng(seed)
    if len(clusters):
        for rep in range(replicates):
            selected=rng.choice(clusters,size=len(clusters),replace=True)
            sampled=frame.iloc[np.concatenate([blocks[g] for g in selected])].reset_index(drop=True)
            values=metric_values(sampled,methods)
            for ci,context in enumerate(CONTEXTS+('macro',)):
                if not original_valid[context]:continue
                for mi,method in enumerate(methods):draws[rep,ci,mi]=[values[context][method][metric] for metric in METRICS]
    method_rows=[]; pair_rows=[]
    for ci,context in enumerate(CONTEXTS+('macro',)):
        for mi,method in enumerate(methods):
            for ki,key in enumerate(METRICS):
                array=draws[:,ci,mi,ki]; finite=array[np.isfinite(array)]
                lo,hi=np.percentile(finite,[2.5,97.5]) if len(finite)>=2 else (np.nan,np.nan)
                method_rows.append({'context':context,'method':method,'metric':key,'point':point[context][method][key],
                                    'ci95_lower':lo,'ci95_upper':hi,'valid_draws':len(finite),'bootstrap_replicates':replicates})
        for a,b in itertools.combinations(methods,2):
            ai,bi=methods.index(a),methods.index(b)
            for ki,key in enumerate(METRICS):
                array=draws[:,ci,ai,ki]-draws[:,ci,bi,ki]; finite=array[np.isfinite(array)]
                lo,hi=np.percentile(finite,[2.5,97.5]) if len(finite)>=2 else (np.nan,np.nan)
                pair_rows.append({'context':context,'method_a':a,'method_b':b,'metric':key,
                  'difference':point[context][a][key]-point[context][b][key],'ci95_lower':lo,'ci95_upper':hi,
                  'valid_draws':len(finite),'bootstrap_replicates':replicates,'primary_pair':(a,b)==PRIMARY,
                  'prespecified_manual_pair':(a,b)==MANUAL})
    return pd.DataFrame(method_rows),pd.DataFrame(pair_rows)


def evaluate(comparison_root,truth_receipt,truth_errors,output):
    comparison=Path(comparison_root).resolve(); spec_path=comparison/'COMPARISON_SPEC.json'
    spec=json_immutable(spec_path)
    if spec.get('schema')!=SCHEMA or spec.get('evaluator_code_sha256')!=risk.sha(__file__):raise RuntimeError('Frozen evaluator code/spec differs')
    if (spec.get('bootstrap_replicates')!=REPLICATES or spec.get('bootstrap_seed')!=SEED
        or spec.get('minimum_context_tasks')!=20 or spec.get('minimum_treated_cells')!=30
        or spec.get('metrics')!=list(METRICS) or spec.get('primary_pair')!=list(PRIMARY)
        or spec.get('prespecified_manual_pair')!=list(MANUAL)):
        raise RuntimeError('Fixed evaluation metrics/cohort/seed/draw contract differs')
    for key in ['risk_seal_manifest','risk_seal_scores','risk_seal_artifact_hashes','support_feature','comparison_scores']:checked_bind(spec[key])
    for value in spec['source_core_and_LM_prediction_parameter_bindings']:checked_bind(value)
    receipt=json_immutable(truth_receipt)
    guarded_truth_receipt(receipt,spec)
    for key,path in [('spec',spec_path),('scores',Path(spec['comparison_scores']['path']))]:
        expected=receipt.get('pretruth_comparison',{}).get(key,{})
        if Path(expected.get('path','')).resolve()!=path.resolve() or expected.get('sha256')!=risk.sha(path):
            raise RuntimeError('Truth receipt does not bind exactpretruthcomparison')
    truth_binding=receipt.get('truth_errors',{})
    if Path(truth_binding.get('path','')).resolve()!=Path(truth_errors).resolve() or checked_bind(truth_binding).resolve()!=Path(truth_errors).resolve():
        raise RuntimeError('Exact receipted task-error file required')
    source_bind=receipt.get('risk_seal_manifest',{})
    if source_bind.get('sha256')!=spec['risk_seal_manifest']['sha256']:
        raise RuntimeError('Truth receipt does not bind exactrisk/model/prediction seal')
    scores=pd.read_parquet(spec['comparison_scores']['path'])
    truth=pd.read_parquet(truth_errors)
    required=QUERY+['n_cells','true_error_rmse']
    if list(truth.columns)!=required or not truth.query_id.is_unique or not truth.role.eq('TEST').all():
        raise RuntimeError('Truth task schema/identity/role differs')
    if not np.isfinite(truth.true_error_rmse).all() or (truth.true_error_rmse<0).any() or not np.isfinite(truth.n_cells).all():
        raise RuntimeError('Finite observed task errors and counts required')
    if (truth.n_cells<=0).any() or not np.array_equal(truth.n_cells.to_numpy(float),np.rint(truth.n_cells.to_numpy(float))):
        raise RuntimeError('Measured positive integer TEST cell counts required')
    merged=scores.merge(truth,on=QUERY,how='inner',validate='one_to_one')
    if len(merged)!=len(truth):raise RuntimeError('Every evaluated TEST task must match frozen predictions exactly')
    eligible=merged.n_cells.ge(spec['minimum_treated_cells'])
    common=eligible&merged.PRETRUTH_PRIMARY_COMMON_ELIGIBLE
    allp=eligible&np.isfinite(merged[['Magnitude','P_only_ridge','P_only_hgb']].to_numpy(float)).all(axis=1)
    cohorts={'primary_common':(merged[common].reset_index(drop=True),spec['methods']),
             'all_prediction_only':(merged[allp].reset_index(drop=True),['Magnitude','P_only_ridge','P_only_hgb']),
             'no_source_history_prediction_only':(merged[allp&merged.source_history_n.eq(0)].reset_index(drop=True),['Magnitude','P_only_ridge','P_only_hgb'])}
    output=Path(output)
    if output.exists():raise RuntimeError('New evaluation output version required')
    stage=output.with_name(output.name+'.incomplete');stage.mkdir(parents=True)
    coverage=[]; intervals=[]; comparisons=[]
    for cohort,(frame,methods) in cohorts.items():
        for context in CONTEXTS:
            denominator=int((eligible&merged.context_id.eq(context)).sum()); n=int(frame.context_id.eq(context).sum())
            coverage.append({'cohort':cohort,'context':context,'n_tasks':n,'n_biological_gene_clusters':frame[frame.context_id.eq(context)].target_gene_id.nunique(),
                'eligible_TEST_tasks':denominator,'fraction_of_eligible_TEST':n/denominator if denominator else np.nan,
                'status':'OK' if n>=20 else 'INSUFFICIENT_CONTEXT_TASKS'})
        points,pairs=cohort_statistics(frame,methods,spec['bootstrap_replicates'],spec['bootstrap_seed'])
        points['cohort']=cohort;pairs['cohort']=cohort;intervals.append(points);comparisons.append(pairs)
        frame[QUERY+['n_cells','source_history_n']].to_csv(stage/f'{cohort}_COHORT.csv',index=False)
    pd.DataFrame(coverage).to_csv(stage/'COHORT_COVERAGE.csv',index=False)
    pd.concat(intervals,ignore_index=True).to_csv(stage/'METHOD_METRIC_INTERVALS.csv',index=False)
    pd.concat(comparisons,ignore_index=True).to_csv(stage/'ALL_PRESPECIFIED_PAIRED_COMPARISONS.csv',index=False)
    report={'schema':SCHEMA,'status':'COMPLETE_FIXED_COMPARISON','comparison_spec_sha256':risk.sha(spec_path),
       'truth_reader_receipt_sha256':risk.sha(truth_receipt),'truth_errors_sha256':risk.sha(truth_errors),
       'candidate_selection_after_truth':False,'new_fits':0,'target_CDF_fit':False,'bootstrap_unit':spec['bootstrap_unit'],
       'bootstrap_replicates':spec['bootstrap_replicates'],'bootstrap_seed':spec['bootstrap_seed'],
       'primary_pair':spec['primary_pair'],'prespecified_manual_pair':spec['prespecified_manual_pair'],
       'all_nominal_CIs_reported':True,'TEST_metadata_status':'SEEN_METADATA_ONLY','whole_method_pristine_claim':False,
       'truth_status':'OPENED_ONLY_UNDER_SEPARATE_RECEIPT','no_history_public_fallback_claim':False}
    write_json(stage/'EVALUATION_MANIFEST.json',report)
    write_json(stage/'ARTIFACT_HASHES.json',[{'path':p.name,'bytes':p.stat().st_size,'sha256':risk.sha(p)} for p in sorted(stage.iterdir()) if p.is_file()])
    stage.rename(output);return report


def synthetic_test(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for context in CONTEXTS:
        for i in range(30):rows.append({'query_id':f'{context}|G{i:02d}','target_gene_id':f'E{i:02d}','target_gene_symbol':f'G{i:02d}',
           'context_id':context,'role':'TEST','true_error_rmse':float(i+1),'n_cells':40,'perfect':float(i+1),'reverse':float(30-i)})
    frame=pd.DataFrame(rows); point=metric_values(frame,['perfect','reverse'])
    assert point['macro']['perfect']['utility20']==1 and point['macro']['perfect']['spearman']==1
    assert np.isclose(point['macro']['perfect']['aurc'],8.25)
    assert point['macro']['perfect']['error_at_20']==3.5 and point['macro']['perfect']['high_risk_miss_rate']==0
    assert point['macro']['reverse']['high_risk_miss_rate']==1
    assert np.isnan(metric_values(frame.iloc[:19],['perfect'])['macro']['perfect']['utility20'])
    intervals,pairs=cohort_statistics(frame,['perfect','reverse'],replicates=40,seed=SEED)
    assert pairs.query("context=='macro' and metric=='utility20'").difference.iloc[0]>1.9
    assert intervals.bootstrap_replicates.eq(40).all()
    insufficient=pd.concat([frame[frame.context_id.eq('HCT116')].iloc[:19],frame[frame.context_id.eq('HEK293T')]],ignore_index=True)
    insufficient_intervals,insufficient_pairs=cohort_statistics(insufficient,['perfect','reverse'],replicates=40,seed=SEED)
    invalid=insufficient_intervals.context.isin(['HCT116','macro'])
    assert insufficient_intervals.loc[invalid,'valid_draws'].eq(0).all()
    assert insufficient_intervals.loc[invalid,['point','ci95_lower','ci95_upper']].isna().all().all()
    assert insufficient_pairs.loc[insufficient_pairs.context.isin(['HCT116','macro']),'valid_draws'].eq(0).all()
    try:guarded_truth_receipt({'status':'COMPLETE'},{})
    except RuntimeError:pass
    else:raise AssertionError('Generic COMPLETE JSON cannot authorize truth/error reads')
    # The fixture uses40draws to verify resampling mechanics; production spec
    # is fixed at5000 and the CLI exposes no lower draw override.
    intervals.to_csv(output/'SYNTHETIC_INTERVALS.csv',index=False)
    pairs.to_csv(output/'SYNTHETIC_PAIRED.csv',index=False)
    report={'schema':SCHEMA,'status':'SYNTHETIC_METRICS_AND_GENE_BLOCK_BOOTSTRAP_PASS',
       'code_sha256':risk.sha(__file__),'hand_perfect_U20':1,'hand_perfect_AURC':8.25,'hand_error_at20':3.5,
       'full_two_context_gene_blocks_resampled':True,'minimum20_context_gate_pass':True,
       'original19task_context_and_macro_CIs_remain_NA_despite_bootstrap_duplicates':True,
       'generic_COMPLETE_receipt_rejected_before_truth_open':True,
       'fixture_draws':40,'production_draws_fixed':REPLICATES,'actual_TEST_expression_or_truth_read':False,
       'newfits':0,'target_CDF_fit':False}
    write_json(output/'SYNTHETIC_EVALUATION_PROOF.json',report);print(json.dumps(report,indent=2));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare-comparison');prep.add_argument('--risk-seal',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    actual=sub.add_parser('evaluate');actual.add_argument('--comparison',type=Path,required=True);actual.add_argument('--truth-receipt',type=Path,required=True);actual.add_argument('--truth-errors',type=Path,required=True);actual.add_argument('--output',type=Path,required=True)
    synthetic=sub.add_parser('synthetic-test');synthetic.add_argument('--output',type=Path,default=DOC/'synthetic_v1')
    args=parser.parse_args()
    if args.command=='prepare-comparison':result=prepare_comparison(args.risk_seal,args.output)
    elif args.command=='evaluate':result=evaluate(args.comparison,args.truth_receipt,args.truth_errors,args.output)
    else:result=synthetic_test(args.output)
    if args.command!='synthetic-test':print(json.dumps(result,indent=2))


if __name__=='__main__':main()
