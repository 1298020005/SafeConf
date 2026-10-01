"""Continue the existing fixed family attempt after token-format bridge repair.

No aggregation retry, new candidate selection or TEST opening is implemented.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
DOC = REPO / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation'
RUN = BASE / 'orion_decimal_token_recovery_20261002_v2'
BIO = BASE / 'orion_allowed_biology_20261002_v2'
BRIDGE = BASE / 'orion_lm_bridge_20261002_v1/actual_authorized_v3_decimal_tokens'
FIT = BASE / 'orion_published_lm_20261002_v3_decimal_tokens'
FOLLOW = BASE / 'orion_postfit_followthrough_20261002_v4_decimal_tokens'
EXPANDED = BASE / 'orion_public_bank_extension_20261002_v3_decimal_tokens'
AMENDMENT = DOC / 'lm_core/decimal_token_bridge_recovery_v1/ROOT_APPROVED_TECHNICAL_AMENDMENT.json'
PERMIT = DOC / 'ORION_VECTORIZED_PIPELINE_ACCESS_PERMIT.json'
CONTRACT = DOC / 'ORION_METHOD_CONTRACT.json'
BANK = BASE / 'public_source_history_expanded_20261002_v2/PUBLIC_BANK_MANIFEST.json'
EXT_CONTRACT = DOC / 'historical_bank_v1/registered_v3_sorted_csr/PUBLIC_BANK_EXPANSION_CONTRACT.json'
SCOPE = BASE / 'orion_lm_bridge_20261002_v1/query_scope_metadata_only'
RSCRIPT = '/home/yyf/.conda/envs/safeconf-orion-lm-20261002/bin/Rscript'
SCIENCE_SHA = '6b8a26939c8c6134dbf2e8294ee7905fccc2122b570b046f3183a3d2f5caba97'
PINS = {
 'tools/scripts/bridge_safeconf_orion_lm_inputs_decimal_token_agent.py': 'caa4212788e89190efa93980c41314a7892393a8d19bc809e427f23d4859c315',
 'tools/scripts/bridge_safeconf_orion_lm_inputs_agent.py': 'dcc7d6afcd812d89ddb372e511cb9607e216da8c1f57e484317db4b3eda52a38',
 'tools/scripts/run_safeconf_orion_lm_blind.R': '086fb5489bc88520b1854d0101db117f66f913d7ce94064a6df18772633aaee1',
 'tools/scripts/evaluate_safeconf_orion_competence_agent.py': 'da06a174c69339b4afb1cfc7ab33015e34c8d27f2d7264a045c799e7f75b942c',
 'tools/scripts/seal_safeconf_orion_source_risk_agent.py': '8b69e6ce29dc92aa34150b3b08aac62ce8333a10f84ec432a39d0e81ffc64482',
 'tools/safeconf_continual/orion_public_bank_extension.py': '9857b96b98460346bce334313c1a212466c85dc450c07f16e5d934e1ef045454',
 'tools/scripts/seal_safeconf_orion_public_bank_extension_agent.py': '81ca3f44f147880b19fc9b8225ac8c63572e9a44dfc78e1d6d2208b3770b0414',
 'tools/scripts/run_safeconf_orion_public_bank_extension_chain_agent.py': '2df89381e10075ccad73c3072dba610a77159257f80c8adf5162a1bc63ef1dff',
 'tools/scripts/evaluate_safeconf_orion_guarded_test_agent.py': '852fcd764b567a63f04c4a5539a118c8a715f664bf534184018010ac269087b2',
 'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py': 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c'}


def sha(path):
 digest = hashlib.sha256()
 with Path(path).open('rb') as file:
  for chunk in iter(lambda: file.read(8 * 1024**2), b''): digest.update(chunk)
 return digest.hexdigest()


def bind(path):
 path = Path(path).resolve(); return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def immutable(path, value):
 with Path(path).open('x') as file: file.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
 Path(path).chmod(0o444)


def verify():
 for name, expected in PINS.items():
  if sha(REPO / name) != expected: raise RuntimeError('Fixed implementation differs: ' + name)
 for path, expected in [(AMENDMENT, 'a7d3162aa36c3276e9cbbd0b0a5a679716dd7c640aaf2123ab412517b2781846'),
  (PERMIT, 'aad3ef022968bb3d5a2d9f3b4baf59a548698bd20f2996f249b7153570af1a76'), (CONTRACT, SCIENCE_SHA),
  (BIO / 'BIOLOGY_MANIFEST.json', 'be4113822254cfb0824a464bd87300d6ff33388069500e14f3229d8acc9fba16'),
  (BANK, '8053a6118bc2e3cace841e1498cee3c1366ce5b49e9c79328e7cd34a180b6607'),
  (EXT_CONTRACT, '0b6b9b86ad18da6f748ff4fc7cca4e20253c6b43a2d9533bb2b585664c79487f')]:
  if sha(path) != expected or (path != BIO / 'BIOLOGY_MANIFEST.json' and path.stat().st_mode & 0o222): raise RuntimeError('Exact preserved contract/input differs: ' + str(path))
 if sha(Path(__file__)) != registration['runtime_controller']['sha256']: raise RuntimeError('Controller bytes changed')
 if (BASE / 'ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json').exists(): raise RuntimeError('TEST scope already exists; pretruth recovery stops')


def save():
 state['updated_utc'] = datetime.now(timezone.utc).isoformat()
 for root in (RUN, FOLLOW, EXPANDED):
  path = root / '.STATUS.tmp'; path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); path.replace(root / 'STATUS.json')


def stage(name, argv, seconds):
 verify(); state['status'] = name; save(); started = time.monotonic()
 with (RUN / (name + '.log')).open('xb') as log:
  process = subprocess.Popen(argv, cwd=REPO, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1'), stdout=log, stderr=log)
  state['child_pid'] = process.pid; save()
  try: result = process.wait(timeout=seconds)
  except subprocess.TimeoutExpired:
   process.terminate()
   try: process.wait(timeout=15)
   except subprocess.TimeoutExpired: process.kill(); process.wait()
   raise RuntimeError(name + ' exceeded original stage bound; no retry/TEST')
 state['stages'].append({'name': name, 'pid': process.pid, 'exit_code': result, 'elapsed_seconds': time.monotonic() - started, 'argv': argv})
 state.pop('child_pid', None); save()
 if result: raise RuntimeError(name + ' failed; preserve outputs, no retry/TEST')


registration = json.loads((RUN / 'CONTINUATION_REGISTRATION.json').read_text())
state = {'schema': 'safeconf_orion_decimal_token_recovery_pretruth_v1', 'status': 'WAITING_ALREADY_STARTED_TECHNICAL_BRIDGE',
 'pid': os.getpid(), 'TEST_truth_opened': False, 'test_truth_opened': False, 'first_open_claim_created': False,
 'original_scientific_attempt_id': 'Orion_PublishedLM_family_attempt_20261002_v2', 'new_family_or_candidate_count': 0,
 'planned_context_model_fits': 2, 'formal_validation_prediction_attempts': 1, 'gpu_hours': 0, 'stages': []}
try:
 verify(); save(); started = time.monotonic(); bridge_process = json.loads((RUN / 'BRIDGE_PROCESS.json').read_text())
 while not (BRIDGE / 'BRIDGE_MANIFEST.json').is_file():
  try: os.kill(bridge_process['pid'], 0)
  except ProcessLookupError: raise RuntimeError('Actual technical bridge exited without completed output; no fit/TEST')
  if time.monotonic() - started > 1800: raise TimeoutError('Actual bridge continuation exceeds original30min budget')
  time.sleep(5)
 bridge = json.loads((BRIDGE / 'BRIDGE_MANIFEST.json').read_text())
 if (bridge.get('status') != 'COMPLETE' or bridge.get('bridge_sha256') != PINS['tools/scripts/bridge_safeconf_orion_lm_inputs_decimal_token_agent.py']
  or bridge.get('permit_sha256') != sha(PERMIT) or bridge.get('biology_manifest_sha256') != sha(BIO / 'BIOLOGY_MANIFEST.json')
  or bridge.get('scientific_contract_sha256') != SCIENCE_SHA
  or bridge.get('integral_token_technical_amendment', {}).get('sha256') != sha(AMENDMENT)):
  raise RuntimeError('Actual repaired bridge producer/amendment/original provenance differs')
 state['actual_bridge_manifest'] = bind(BRIDGE / 'BRIDGE_MANIFEST.json'); save()
 stage('FIXED_PUBLISHED_CPU_LINEAR_FIT_PRETRUTH', [RSCRIPT, str(REPO / 'tools/scripts/run_safeconf_orion_lm_blind.R'),
  '--mode', 'fit', '--manifest', str(BRIDGE / 'FIT_MANIFEST.tsv'), '--output-dir', str(FIT)], 3600)
 stage('FIXED_VALIDATION_COMPETENCE_ONLY', [sys.executable, '-u', str(REPO / 'tools/scripts/evaluate_safeconf_orion_competence_agent.py'),
  '--fit-root', str(FIT), '--biology-root', str(BIO), '--bridge-root', str(BRIDGE), '--permit', str(PERMIT), '--contract', str(CONTRACT),
  '--validation-eligibility', str(SCOPE / 'VALIDATION_ELIGIBILITY.tsv'), '--output', str(FOLLOW / 'competence'),
  '--expected-evaluator-sha256', PINS['tools/scripts/evaluate_safeconf_orion_competence_agent.py'],
  '--expected-baseline-helper-sha256', 'd89ab0f92d8ef971a4476d7e13f17ce19e332e7075508db1ed057be81e52f5fe',
  '--expected-validation-eligibility-sha256', '274bc9f946d09229334cda7b78e4c39d94f2e6f4dc18888a8e58da7f27d16e6c'], 1800)
 competence = FOLLOW / 'competence/COMPETENCE_RESULT.json'; result = json.loads(competence.read_text()); state['actual_competence'] = bind(competence)
 if (result.get('status') != 'PASS' or result.get('passes_competence') is not True or result.get('test_truth_read') is not False
  or result.get('scientific_contract_sha256') != SCIENCE_SHA or result.get('new_model_parameters_fitted') is not False):
  state['status'] = 'UPSTREAM_NOT_QUALIFIED_NO_RISK_SEAL_NO_TEST_OPEN'; save(); sys.exit(0)
 sys.path.insert(0, str(REPO))
 from tools.scripts import seal_safeconf_orion_source_risk_agent as original
 original_input = json.loads((DOC / 'risk_seal/SOURCE_COMPARISON_TEMPLATE_FINAL.json').read_text())
 original_input.update(risk_code_sha256=sha(Path(original.__file__)), pretruth_inference_authorized=True, target_errors_or_CDF_used=False,
  contexts=[original.context_bindings(FIT, BRIDGE, context, FIT) for context in ['HCT116', 'HEK293T']],
  data_role_registry_path=str(DOC / 'ORION_DATA_ROLE_REGISTRY.csv'), data_role_registry_sha256=sha(DOC / 'ORION_DATA_ROLE_REGISTRY.csv'),
  competence_gate_reference=bind(competence), technical_bridge_amendment=bind(AMENDMENT), actual_bridge_manifest=bind(BRIDGE / 'BRIDGE_MANIFEST.json'))
 immutable(FOLLOW / 'FROZEN_PRETEST_COMPARISON.json', original_input)
 stage('ORIGINAL2008_SOURCE_PRETRUTH_ARCHIVE', [sys.executable, '-u', str(REPO / 'tools/scripts/seal_safeconf_orion_source_risk_agent.py'),
  'seal', '--comparison-manifest', str(FOLLOW / 'FROZEN_PRETEST_COMPARISON.json'), '--output', str(FOLLOW / 'risk_seal')], 1800)
 immutable(EXPANDED / 'SEAL_REGISTRATION.json', {'schema': 'safeconf_orion_public_bank_extension_seal_registration_v1',
  'implementation_bindings': {'extension_sealer': bind(REPO / 'tools/scripts/seal_safeconf_orion_public_bank_extension_agent.py'),
                             'extension_helper': bind(REPO / 'tools/safeconf_continual/orion_public_bank_extension.py')},
  'extension_contract': bind(EXT_CONTRACT), 'bank_manifest': bind(BANK), 'base_comparison_manifest': bind(FOLLOW / 'FROZEN_PRETEST_COMPARISON.json'),
  'actual_competence': bind(competence), 'technical_bridge_amendment': bind(AMENDMENT),
  'actual_bridge_manifest': bind(BRIDGE / 'BRIDGE_MANIFEST.json'),
  'preserved_original2008_risk_package': {'manifest': bind(FOLLOW / 'risk_seal/RISK_SEAL_MANIFEST.json'), 'artifacts': bind(FOLLOW / 'risk_seal/ARTIFACT_HASHES.json')},
  'execution_scope': 'PRETRUTH_ONLY_NO_TEST_FIRST_OPEN_CLAIM'})
 stage('EXPANDED_SOURCE_PRETRUTH_SEAL', [sys.executable, '-u', str(REPO / 'tools/scripts/seal_safeconf_orion_public_bank_extension_agent.py'),
  '--registration', str(EXPANDED / 'SEAL_REGISTRATION.json'), '--output', str(EXPANDED / 'expanded_risk_seal')], 1800)
 stage('EXPANDED_FIXED_PRETRUTH_COMPARISON', [sys.executable, '-u', str(REPO / 'tools/scripts/run_safeconf_orion_public_bank_extension_chain_agent.py'),
  'prepare-comparison', '--seal', str(EXPANDED / 'expanded_risk_seal'), '--output', str(EXPANDED / 'expanded_comparison')], 1800)
 verify(); state['status'] = 'EXPANDED_BANK_SCORES_COMPARISON_FROZEN_TEST_CLOSED_AWAIT_ROOT_ACTUAL_REVIEW'
 state['extension_seal_receipt'] = bind(EXPANDED / 'expanded_risk_seal/EXTENSION_SEAL_RECEIPT.json')
 state['extension_comparison_receipt'] = bind(EXPANDED / 'expanded_comparison/EXTENSION_COMPARISON_RECEIPT.json'); save()
except Exception as error:
 state['status'] = 'FAILED_TECHNICAL_CONTINUATION_NO_TEST_OPEN_NO_RETRY'; state['error_type'] = type(error).__name__; state['error'] = str(error); save()
 print(state['error'], flush=True); sys.exit(1)
