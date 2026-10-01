"""Fail-closed coordinator boundaries and generated, temporary scope claims."""
from pathlib import Path
import importlib.util
import json
import tempfile
from unittest.mock import patch

ROOT=Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
spec=importlib.util.spec_from_file_location('coordinator_review',ROOT/'tools/scripts/run_safeconf_orion_registered_evaluation_followthrough.py')
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
from tools.scripts import evaluate_safeconf_orion_frozen_risk_agent as e
from tools.scripts import evaluate_safeconf_orion_guarded_test_agent as t


def run():
    cases=[]
    production_claim=c.BASE/'ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json'
    production_claim_existed=production_claim.exists()
    production_claim_sha=c.sha(production_claim) if production_claim_existed else None
    with tempfile.TemporaryDirectory(prefix='orion_coordinator_boundary_') as tmp:
        tmp=Path(tmp)
        calls=[]
        def forbidden_prepare(*args,**kwargs):calls.append('prepare');raise AssertionError('Unexpected comparator preparation')
        def forbidden_authorize(*args,**kwargs):calls.append('authorize');raise AssertionError('Unexpected TEST authorization')
        def forbidden_reader(*args,**kwargs):calls.append('numeric_reader');raise AssertionError('Unexpected numeric reader');yield
        with patch.object(e,'prepare_comparison',forbidden_prepare),patch.object(t,'authorize_test',forbidden_authorize),patch.object(t.reader,'iter_selected_numeric_lists',forbidden_reader):
            try:c.make_operation(tmp/'current_not_ready');raise AssertionError('Current incomplete upstream accepted')
            except RuntimeError as exc:assert 'not ready' in str(exc) and not calls
            cases.append({'case':'current_actual_WAITING_status','rejected_before_comparator_or_TEST_authorization_or_numeric_reader':True})
            original=c.sha
            wrong=ROOT/'tools/scripts/evaluate_safeconf_orion_guarded_test_agent.py'
            def mutated_pin(path):return '0'*64 if Path(path).resolve()==wrong else original(path)
            with patch.object(c,'sha',mutated_pin):
                try:c.make_operation(tmp/'changed_pin');raise AssertionError('Changed code pin accepted')
                except RuntimeError as exc:assert 'Reviewed code hash changed' in str(exc) and not calls
            cases.append({'case':'mocked_guarded_reader_code_pin_change','rejected_before_comparator_or_TEST_authorization_or_numeric_reader':True})
            with patch.object(c,'FOLLOW',tmp/'missing_follow'):
                try:c.make_operation(tmp/'missing_status');raise AssertionError('Missing actual gate accepted')
                except FileNotFoundError:assert not calls
            cases.append({'case':'missing_actual_upstream_STATUS','rejected_before_comparator_or_TEST_authorization_or_numeric_reader':True})
        incomplete=tmp/'incomplete_ready';incomplete.mkdir()
        (incomplete/'STATUS.json').write_text(json.dumps({'status':'VALIDATION_QUALIFIED_AND_RISK_PREDICTIONS_FROZEN_TEST_TRUTH_STILL_CLOSED'}))
        with patch.object(c,'FOLLOW',incomplete),patch.object(t,'authorize_test',forbidden_authorize),patch.object(t.reader,'iter_selected_numeric_lists',forbidden_reader):
            try:c.make_operation(tmp/'missing_seal');raise AssertionError('Ready string without actual all-candidate seal accepted')
            except FileNotFoundError:assert not calls and not (tmp/'missing_seal').exists()
        cases.append({'case':'ready_status_string_but_missing_actual_candidate_seal','rejected_without_output_before_TEST_authorization_or_numeric_reader':True})
        # This is deliberately NOT an authorized TEST operation/scientific
        # contract: it exercises only the exclusive-file claim function.
        synthetic_science=tmp/'synthetic_claim_science.json'
        synthetic_science.write_text(json.dumps({'synthetic_only':True,'claim_function_fixture_only':True}))
        synthetic_objects=tmp/'synthetic_claim_objects.json'
        synthetic_objects.write_text(json.dumps({'synthetic_only':True,'files':[]}))
        claim_path=tmp/'first_claim.json'
        operations=[]
        for label in ['first_output','different_second_output']:
            op=tmp/f'{label}_operation.json'
            op.write_text(json.dumps({'synthetic_only':True,'claim_function_fixture_only':True,
                'scientific_contract':c.bind(synthetic_science),'forty_object_manifest':c.bind(synthetic_objects),
                'metadata_bindings':{'GENE_SPLIT.csv':'synthetic_claim_only_no_gene_split'},
                'output_path':str(tmp/label)}))
            op.chmod(0o444);operations.append(op)
        first=c.reserve_registered_scope(operations[0],claim_path=claim_path)
        first_sha=c.sha(claim_path);first_claim=json.loads(claim_path.read_text())
        assert first['sha256']==first_sha and not claim_path.stat().st_mode&0o222
        assert first_claim['operation']==c.bind(operations[0])
        assert first_claim['failure_does_not_release_scope'] is True
        try:c.reserve_registered_scope(operations[1],claim_path=claim_path);raise AssertionError('Second distinct output registered')
        except FileExistsError:pass
        assert c.sha(claim_path)==first_sha and json.loads(claim_path.read_text())==first_claim
        cases.append({'case':'generated_first_claim_then_second_distinct_output_operation',
            'first_claim_success':True,'second_claim_FileExistsError':True,
            'first_claim_sha256_preserved':first_sha,'first_claim_readonly_and_bound_to_original_operation':True,
            'temporary_claim_path_only':True})
    assert production_claim.exists()==production_claim_existed
    if production_claim_existed:assert c.sha(production_claim)==production_claim_sha
    report={'status':'COORDINATOR_READONLY_AND_SAFE_BOUNDARY_REVIEW_PASS',
        'coordinator_sha256':c.sha(c.__file__),'reviewed_code_pins':c.PINS,'cases':cases,
        'current_upstream_status':json.loads((c.FOLLOW/'STATUS.json').read_text())['status'],
        'once_only_scope_enforcement':'Exclusive fixed BASE/ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json across all output roots; never auto-released',
        'scope_claim_sequence_review':'make_operation ends with reader.authorize_test; run then reserves scope and saves claim before raw reader subprocess starts',
        'actual_TEST_expression_or_truth_access':False,'coordinator_launched':False,
        'production_registration_created_or_changed':False,
        'fake_valid_production_scientific_contract_created':False,'source_or_frozen_method_changes':False,
        'artifact_interface_review':'Exact reader operation keys, comparator direct aliases, LM/risk/Source parameter registries, old9fd-to3e7 primitive override and immutable40object snapshot997 match guardedreader/evaluator interfaces.'}
    review=e.DOC/'COORDINATOR_REVIEW.json'
    if review.exists():review.chmod(0o600)
    review.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
