"""Operational waiting/sealing only. Never imports or invokes TEST opening."""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
DOC = REPO / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation'
RUN = BASE / 'orion_public_bank_extension_20261002_v1'
FOLLOW = BASE / 'orion_postfit_followthrough_20261002_v2'
SCIENCE = '6b8a26939c8c6134dbf2e8294ee7905fccc2122b570b046f3183a3d2f5caba97'
READY = 'VALIDATION_QUALIFIED_AND_RISK_PREDICTIONS_FROZEN_TEST_TRUTH_STILL_CLOSED'
FAIL = {'UPSTREAM_NOT_QUALIFIED_NO_FORMAL_RISK_TEST_OR_TEST_OPEN', 'FAILED_OR_BOUNDED_INCOMPLETE', 'COMPETENCE_DONE_SCHEMA_REQUIRES_PARENT_READ'}
PINS = {
    'tools/safeconf_continual/orion_public_bank_extension.py': '9857b96b98460346bce334313c1a212466c85dc450c07f16e5d934e1ef045454',
    'tools/scripts/seal_safeconf_orion_public_bank_extension_agent.py': '81ca3f44f147880b19fc9b8225ac8c63572e9a44dfc78e1d6d2208b3770b0414',
    'tools/scripts/run_safeconf_orion_public_bank_extension_chain_agent.py': '2df89381e10075ccad73c3072dba610a77159257f80c8adf5162a1bc63ef1dff',
    'tools/scripts/seal_safeconf_orion_source_risk_agent.py': '8b69e6ce29dc92aa34150b3b08aac62ce8333a10f84ec432a39d0e81ffc64482',
    'tools/scripts/evaluate_safeconf_orion_guarded_test_agent.py': '852fcd764b567a63f04c4a5539a118c8a715f664bf534184018010ac269087b2',
    'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py': 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c',
    'tools/scripts/run_safeconf_orion_registered_evaluation_followthrough.py': 'f08f27279d8a0f45f67ba97d539bfbe27ed9351a75fd57c1eaab57a91c5ece85'}
BANK = BASE / 'public_source_history_expanded_20261002_v2/PUBLIC_BANK_MANIFEST.json'
CONTRACT = DOC / 'historical_bank_v1/registered_v3_sorted_csr/PUBLIC_BANK_EXPANSION_CONTRACT.json'
REVIEW = DOC / 'public_bank_extension_independent_review_v1/FINAL_ADAPTER_CHAIN_REVIEW.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024**2), b''): digest.update(chunk)
    return digest.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def immutable_json(path, value):
    with Path(path).open('x') as file: file.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    Path(path).chmod(0o444)


def checked(path, expected):
    if Path(path).stat().st_mode & 0o222 or sha(path) != expected:
        raise RuntimeError('Exact immutable pretruth dependency differs: ' + str(path))
    return json.loads(Path(path).read_text())


def verify():
    for name, expected in PINS.items():
        if sha(REPO / name) != expected: raise RuntimeError('Registered implementation changed: ' + name)
    if sha(Path(__file__)) != registration['runtime_waiter']['sha256']:
        raise RuntimeError('Operational waiter bytes changed')
    bank = checked(BANK, '8053a6118bc2e3cace841e1498cee3c1366ce5b49e9c79328e7cd34a180b6607')
    contract = checked(CONTRACT, '0b6b9b86ad18da6f748ff4fc7cca4e20253c6b43a2d9533bb2b585664c79487f')
    review = checked(REVIEW, 'b39093a42a5afd27be491afd7c54c3fba414e28733eb5f013f0890000a66bfe9')
    if bank.get('status') != 'COMPLETE' or review.get('status') != 'PASS' or contract.get('new_parameter_fits') != 0:
        raise RuntimeError('Reviewed completed no-fit pretruth bank required')
    if (BASE / 'ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json').exists():
        raise RuntimeError('A TEST scope already exists; this pretruth extension queue must stop')


def competence_pass(path):
    if Path(path).stat().st_mode & 0o222: raise RuntimeError('Actual competence must be immutable')
    value = json.loads(Path(path).read_text())
    for name in ['relative_macro_error_gap', 'model_macro_rmse', 'baseline_macro_rmse', 'lower95_relative_gap', 'upper95_relative_gap']:
        if type(value.get(name)) not in (int, float) or not math.isfinite(value[name]):
            raise RuntimeError('Finite actual competence evidence required')
    expected = {'status': 'PASS', 'passes_competence': True, 'actual_context_count': 2, 'noninferior_contexts': 2,
        'noninferior_strata_fraction': 1., 'stable_disadvantage': False, 'missing_predictions': 0, 'nonfinite_predictions': 0,
        'test_truth_read': False, 'scientific_contract_sha256': SCIENCE, 'bootstrap_replicates': 5000, 'bootstrap_seed': 20260929,
        'new_model_parameters_fitted': False}
    if (any(value.get(key) != expected_value for key, expected_value in expected.items())
        or value.get('synthetic_only') is not False or not value.get('evaluated_validation_tasks')
        or value.get('eligible_validation_tasks') != value.get('evaluated_validation_tasks')
        or value['baseline_macro_rmse'] <= 0 or value['model_macro_rmse'] < 0
        or value['model_macro_rmse'] > 1.02 * value['baseline_macro_rmse'] or 1. + value['lower95_relative_gap'] > 1.02):
        raise RuntimeError('Actual competence is not the exact fixed PASS; no extension scoring')
    return value


def save():
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    path = RUN / '.STATUS.tmp'
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    path.replace(RUN / 'STATUS.json')


def stage(name, argv):
    verify(); state['status'] = name; save(); started = time.monotonic()
    with (RUN / (name + '.log')).open('xb') as log:
        result = subprocess.run(argv, cwd=REPO, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1'),
            stdout=log, stderr=log, timeout=1800)
    state['stages'].append({'name': name, 'argv': argv, 'exit_code': result.returncode, 'elapsed_seconds': time.monotonic() - started}); save()
    if result.returncode: raise RuntimeError(name + ' failed; preserve all outputs, no retry')


registration = json.loads((RUN / 'WAIT_REGISTRATION.json').read_text())
state = {'schema': 'safeconf_orion_pretruth_extension_waiter_status_v1', 'status': 'WAITING_ACTUAL_ORIGINAL_COMPETENCE_AND_SEAL',
    'pid': os.getpid(), 'TEST_truth_opened': False, 'first_open_claim_created': False, 'new_fits': 0, 'stages': []}
try:
    verify(); save(); started = time.monotonic()
    while True:
        if (FOLLOW / 'STATUS.json').exists():
            upstream = json.loads((FOLLOW / 'STATUS.json').read_text())
            if upstream.get('TESTtruth_opened') is not False: raise RuntimeError('Original followthrough no longer truth closed')
            if upstream.get('status') in FAIL: raise RuntimeError('Original upstream competence/pipeline failed or is unqualified')
            if upstream.get('status') == READY: break
        if time.monotonic() - started > 16 * 3600: raise TimeoutError('Original prerequisite wait exceeds16h')
        time.sleep(30)
    verify(); competence = FOLLOW / 'competence/COMPETENCE_RESULT.json'; competence_pass(competence)
    old_manifest = FOLLOW / 'risk_seal/RISK_SEAL_MANIFEST.json'; old_registry = FOLLOW / 'risk_seal/ARTIFACT_HASHES.json'
    old = json.loads(old_manifest.read_text())
    if (old.get('status') != 'SEALED_PRETRUTH' or old.get('scientific_contract_sha256') != SCIENCE
        or old.get('risk_code_sha256') != PINS['tools/scripts/seal_safeconf_orion_source_risk_agent.py']
        or old_manifest.stat().st_mode & 0o222 or old_registry.stat().st_mode & 0o222):
        raise RuntimeError('Original fixed2008 Source pretruth package not sealed')
    for item in json.loads(old_registry.read_text()):
        path = old_registry.parent / item['path']
        if path.name != item['path'] or path.stat().st_size != item['bytes'] or sha(path) != item['sha256'] or path.stat().st_mode & 0o222:
            raise RuntimeError('Original package artifact changed before extension registration')
    original_input = FOLLOW / 'FROZEN_PRETEST_COMPARISON.json'
    if original_input.stat().st_mode & 0o222: raise RuntimeError('Actual original pretruth inputs must be immutable')
    seal_registration = {'schema': 'safeconf_orion_public_bank_extension_seal_registration_v1',
        'implementation_bindings': {'extension_sealer': bind(REPO / 'tools/scripts/seal_safeconf_orion_public_bank_extension_agent.py'),
                                    'extension_helper': bind(REPO / 'tools/safeconf_continual/orion_public_bank_extension.py')},
        'extension_contract': bind(CONTRACT), 'bank_manifest': bind(BANK), 'base_comparison_manifest': bind(original_input),
        'actual_competence': bind(competence), 'preserved_original2008_risk_package': {'manifest': bind(old_manifest), 'artifacts': bind(old_registry)},
        'reviewed_adapter_chain': bind(REVIEW), 'execution_scope': 'PRETRUTH_SEAL_AND_COMPARISON_ONLY_TEST_TRUTH_CLOSED'}
    immutable_json(RUN / 'SEAL_REGISTRATION.json', seal_registration)
    state['actual_competence'] = bind(competence); state['original_fixed_archive'] = seal_registration['preserved_original2008_risk_package']; save()
    stage('EXPANDED_PRETRUTH_SEAL', [sys.executable, str(REPO / 'tools/scripts/seal_safeconf_orion_public_bank_extension_agent.py'),
        '--registration', str(RUN / 'SEAL_REGISTRATION.json'), '--output', str(RUN / 'expanded_risk_seal')])
    stage('EXPANDED_PRETRUTH_COMPARISON', [sys.executable, str(REPO / 'tools/scripts/run_safeconf_orion_public_bank_extension_chain_agent.py'),
        'prepare-comparison', '--seal', str(RUN / 'expanded_risk_seal'), '--output', str(RUN / 'expanded_comparison')])
    verify(); state['status'] = 'EXPANDED_BANK_SCORES_COMPARISON_FROZEN_TEST_TRUTH_CLOSED_AWAIT_ACTUAL_ROOT_REVIEW'
    state['extension_seal_receipt'] = bind(RUN / 'expanded_risk_seal/EXTENSION_SEAL_RECEIPT.json')
    state['extension_comparison_receipt'] = bind(RUN / 'expanded_comparison/EXTENSION_COMPARISON_RECEIPT.json'); save()
except Exception as error:
    state['status'] = 'STOPPED_CLOSED_NO_TEST_OPEN_NO_RETRY'; state['error_type'] = type(error).__name__; state['error'] = str(error); save()
    print(state['error'], flush=True); sys.exit(1)
