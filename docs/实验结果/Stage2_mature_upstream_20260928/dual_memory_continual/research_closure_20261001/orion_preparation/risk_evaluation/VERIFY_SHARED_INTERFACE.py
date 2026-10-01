"""Generated fixture proof only: pretruth comparison -> guarded reader -> metrics."""
from pathlib import Path
import importlib.util
import json
import sys
import uuid

import numpy as np
import pandas as pd

REPO=Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
sys.path.insert(0,str(REPO))
from tools.scripts import evaluate_safeconf_orion_frozen_risk_agent as evaluation
from tools.scripts import evaluate_safeconf_orion_guarded_test_agent as truth


def load_fixture_api():
    path=truth.DOC/'VERIFY_GUARDED_TEST.py'
    spec=importlib.util.spec_from_file_location('guarded_fixture_api',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    fixture_api=load_fixture_api()
    base=truth.DOC/'synthetic'/('shared_interface_'+uuid.uuid4().hex)
    operation,truth_output,sentinels=fixture_api.fixture(base)
    permit=json.loads(operation.read_text())
    risk_root=Path(permit['risk_seal_root'])
    scores=pd.read_csv(risk_root/'PRETRUTH_RISKS.tsv',sep='\t',keep_default_na=False)
    assert len(scores)==6
    scores['history_status']='KNOWN_SOURCE_SAME_GENE'
    for i,name in enumerate(evaluation.BASE_METHODS+evaluation.OPTIONAL):
        if name not in scores:scores[name]=np.linspace(.1,.6,len(scores))+i/100
    fixture_api.table(risk_root/'PRETRUTH_RISKS.tsv',scores)
    features=pd.DataFrame({name:np.arange(1.,7.)/10 for name in evaluation.risk.P+evaluation.risk.PUBLIC})
    features['log_history_support']=np.log1p(np.arange(1,7)*10)
    for reference in ['Uniform','Manual','Learned']:
        features.to_parquet(risk_root/f'{reference}_P_PUBLIC_FEATURES.parquet',index=False)
    features[evaluation.risk.P].to_parquet(risk_root/'P_ONLY_FEATURES.parquet',index=False)
    pd.DataFrame({name:np.arange(1.,7.)/10 for name in evaluation.risk.PAIR}).to_parquet(
        risk_root/'SOURCE_HISTORY_PAIR_FEATURES.parquet',index=False)
    (risk_root/'ARTIFACT_HASHES.json').chmod(0o600)
    fixture_api.ledger(risk_root)
    for path in risk_root.iterdir():
        if path.is_file():path.chmod(0o444)
    permit['risk_seal_artifacts']=fixture_api.bind(risk_root/'ARTIFACT_HASHES.json')
    comparison=base/'shared_pretruth_comparison'
    spec=evaluation.prepare_comparison(risk_root,comparison)
    assert spec['test_truth_or_errors_used'] is False and spec['bootstrap_replicates']==5000
    frozen=pd.read_parquet(comparison/'PRETRUTH_COMPARISON_SCORES.parquet')
    np.testing.assert_array_equal(frozen[evaluation.SUPPORT],-features.log_history_support)
    assert len(frozen)==6 and set(evaluation.BASE_METHODS+evaluation.OPTIONAL+[evaluation.SUPPORT])<=set(frozen)
    permit['pretruth_comparison']={'spec':fixture_api.bind(comparison/'COMPARISON_SPEC.json'),
        'scores':fixture_api.bind(comparison/'PRETRUTH_COMPARISON_SCORES.parquet')}
    operation.chmod(0o600)
    fixture_api.immutable(operation,permit)
    truth_result=truth.evaluate_test(operation,truth_output,sentinels)
    evaluation_output=base/'shared_evaluation'
    result=evaluation.evaluate(comparison,truth_output/'TRUTH_READER_RECEIPT.json',
        truth_output/'TEST_TASK_ERRORS.parquet',evaluation_output)
    errors=pd.read_parquet(truth_output/'TEST_TASK_ERRORS.parquet')
    assert len(errors)==2 and errors.n_cells.eq(30).all()
    coverage=pd.read_csv(evaluation_output/'COHORT_COVERAGE.csv')
    assert coverage.status.eq('INSUFFICIENT_CONTEXT_TASKS').all()
    intervals=pd.read_csv(evaluation_output/'METHOD_METRIC_INTERVALS.csv')
    assert intervals[['point','ci95_lower','ci95_upper']].isna().all().all()
    assert intervals.valid_draws.eq(0).all() and intervals.bootstrap_replicates.eq(5000).all()
    pairs=pd.read_csv(evaluation_output/'ALL_PRESPECIFIED_PAIRED_COMPARISONS.csv')
    assert pairs[['difference','ci95_lower','ci95_upper']].isna().all().all()
    assert pairs.valid_draws.eq(0).all()
    proof={'status':'SHARED_PRETRUTH_GUARD_EVALUATOR_SYNTHETIC_PASS',
        'evaluator_sha256':evaluation.risk.sha(evaluation.__file__),
        'test_reader_sha256':evaluation.risk.sha(truth.__file__),
        'numeric_backend_sha256':evaluation.risk.sha(truth.reader.__file__),
        'fixture_api_sha256':evaluation.risk.sha(truth.DOC/'VERIFY_GUARDED_TEST.py'),
        'generated_fixture_root':str(base),'six_row_P_PUBLIC_frames_exact_contract':True,
        'all_fixed_candidates_before_truth':spec['candidate_score_columns'],
        'comparison_spec':fixture_api.bind(comparison/'COMPARISON_SPEC.json'),
        'comparison_scores':fixture_api.bind(comparison/'PRETRUTH_COMPARISON_SCORES.parquet'),
        'operation':fixture_api.bind(operation),
        'truth_reader_receipt':fixture_api.bind(truth_output/'TRUTH_READER_RECEIPT.json'),
        'evaluation_manifest':fixture_api.bind(evaluation_output/'EVALUATION_MANIFEST.json'),
        'eligible_tasks':2,'eligible_tasks_per_context':1,
        'all_metrics_and_CIs_NA_because_original_contexts_below20':True,
        'private_NON_TEST_sentinels_not_materialized':True,
        'actual_Orion_TEST_expression_or_truth_access':False,'new_fits':0,'target_CDF_fit':False}
    out=evaluation.DOC/'SHARED_INTERFACE_PROOF.json'
    if out.exists():out.chmod(0o600)
    fixture_api.immutable(out,proof)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':run()
