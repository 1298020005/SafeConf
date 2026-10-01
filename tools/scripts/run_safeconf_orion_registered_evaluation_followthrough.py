#!/usr/bin/env python3
"""Continue the authorized frozen Orion replication after actual competence.

The conditional user authorization is freeze all parameters/predictions first,
then read the registered test truth. This coordinator never opens TEST on a
failed or incomplete gate and never fits/selects a method on TEST results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation'
MD = Path('/home/yyf/data/safeconf_orion_frozen40_20261002/metadata_preparation_20261002_v1')
FOLLOW = BASE / 'orion_postfit_followthrough_20261002_v2'
FIT = BASE / 'orion_published_lm_20261002_v2'
PINS = {
    'evaluate_safeconf_orion_guarded_test_agent.py': '852fcd764b567a63f04c4a5539a118c8a715f664bf534184018010ac269087b2',
    'evaluate_safeconf_orion_frozen_risk_agent.py': 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c',
    'seal_safeconf_orion_source_risk_agent.py': '8b69e6ce29dc92aa34150b3b08aac62ce8333a10f84ec432a39d0e81ffc64482',
    'probe_safeconf_private_parquet_vectorized_agent.py': '636455cb30fc0e8c8ce6ac92db82756eff7508322e92862cf23ffad861d2e435',
    'probe_safeconf_private_parquet_agent.py': '3e7c506f62d384073f2c20af383a371a2a23511c306d2ac046bd72c691f41312',
    'build_safeconf_orion_allowed_biology_agent.py': 'cdf261ffcae854f49d469437492a12b7f835aae7b720bbe565044d8a89369e8b',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify_code():
    for name, expected in PINS.items():
        if sha(ROOT / 'tools/scripts' / name) != expected:
            raise RuntimeError('Reviewed code hash changed; no automatic TEST access: ' + name)


def reserve_registered_scope(operation_path, claim_path=None):
    """Exclusive registration across output directories; never auto-release.

    The claim binds the already validated operation and survives any partial
    read/failure. A technical retry must reuse/audit the original frozen scope;
    another output directory cannot silently register a second test opening.
    """
    operation_path = Path(operation_path).resolve()
    operation = json.loads(operation_path.read_text())
    claim_path = Path(claim_path) if claim_path is not None else BASE / 'ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json'
    claim = {'schema': 'safeconf_orion_exclusive_registered_test_scope_v1',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
             'operation': bind(operation_path), 'scientific_contract': operation['scientific_contract'],
             'forty_object_manifest': operation['forty_object_manifest'],
             'gene_split_sha256': operation['metadata_bindings']['GENE_SPLIT.csv'],
             'output_path': operation['output_path'],
             'automatic_second_registration_allowed': False,
             'failure_does_not_release_scope': True}
    with claim_path.open('x') as file:
        file.write(json.dumps(claim, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    claim_path.chmod(0o444)
    return bind(claim_path)


def make_operation(output_root):
    """Prepare and validate the complete access receipt; no raw TEST read."""
    verify_code()
    from tools.scripts import evaluate_safeconf_orion_guarded_test_agent as reader
    from tools.scripts import evaluate_safeconf_orion_frozen_risk_agent as evaluator
    status = json.loads((FOLLOW / 'STATUS.json').read_text())
    if status.get('status') != 'VALIDATION_QUALIFIED_AND_RISK_PREDICTIONS_FROZEN_TEST_TRUTH_STILL_CLOSED':
        raise RuntimeError('Actual final competence and all-candidate pretruth seals are not ready')
    comparison = output_root / 'pretruth_comparison'
    evaluator.prepare_comparison(FOLLOW / 'risk_seal', comparison)
    prior_permit = json.loads((DOC / 'ORION_VECTORIZED_PIPELINE_ACCESS_PERMIT.json').read_text())
    implementation = {
        'test_reader': Path(reader.__file__), 'numeric_backend': Path(reader.reader.__file__),
        'structural_reader': Path(reader.reader.v1.__file__), 'scientific_loader': Path(reader.biology.__file__),
        'risk_sealer': Path(reader.risk_api.__file__),
    }
    operation = {
        'schema': reader.SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization_basis': 'Existing user 72-hour plan authorizes freeze all parameters/configurations/predictions, then once-only registered TEST evaluation. No fresh method selection or target fitting.',
        'postseal_TEST_operation_authorized': True, 'test_numeric_materialization_permitted': True,
        'authorized_numeric_roles': ['TEST'], 'authorized_contexts': list(reader.CONTEXTS),
        'TEST_metadata_before_open': 'SEEN', 'TEST_truth_before_open': 'CLOSED', 'synthetic_only': False,
        'scientific_contract': bind(DOC / 'ORION_METHOD_CONTRACT.json'),
        'forty_object_manifest': bind(MD / 'snapshots/ORION_CANDIDATE_FILES.997bdbf33c079746457646cd5fe433e3812063ebc1aefcb6f590376086f15900.json'),
        'metadata_root': str(MD), 'raw_root': str(MD.parent),
        'metadata_bindings': {name: sha(MD / name) for name in reader.META_NAMES},
        'data_role_registry': bind(DOC / 'ORION_DATA_ROLE_REGISTRY.csv'),
        'final_competence': bind(FOLLOW / 'competence/COMPETENCE_RESULT.json'),
        'lm_fit_root': str(FIT), 'lm_fit_run_manifest': bind(FIT / 'RUN_MANIFEST.tsv'),
        'lm_fit_artifact_registries': {c: bind(FIT / c / 'ARTIFACT_HASHES.tsv') for c in reader.CONTEXTS},
        'risk_seal_root': str(FOLLOW / 'risk_seal'),
        'risk_seal_manifest': bind(FOLLOW / 'risk_seal/RISK_SEAL_MANIFEST.json'),
        'risk_seal_artifacts': bind(FOLLOW / 'risk_seal/ARTIFACT_HASHES.json'),
        'risk_comparison_manifest': bind(FOLLOW / 'FROZEN_PRETEST_COMPARISON.json'),
        'pretruth_comparison': {'spec': bind(comparison / 'COMPARISON_SPEC.json'),
                               'scores': bind(comparison / 'PRETRUTH_COMPARISON_SCORES.parquet')},
        'implementation_bindings': {key: bind(path) for key, path in implementation.items()},
        'technical_dependency_overrides': prior_permit['technical_dependency_overrides'],
        'test_semantic_QC': reader.SEMANTIC_QC, 'task_min_cells': 30, 'normalization_scale': 4000,
        'resource_budgets': reader.DEFAULT_BUDGETS, 'output_path': str(output_root / 'registered_test_truth'),
        'new_methods_or_fits_after_test': False, 'whole_method_pristine_claim': False,
        'coordinator': bind(__file__),
    }
    path = output_root / 'IMMUTABLE_POSTSEAL_TEST_ACCESS.json'
    with path.open('x') as f:
        f.write(json.dumps(operation, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    path.chmod(0o444)
    reader.authorize_test(path)
    return path


def run(output_root):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'safeconf_orion_registered_external_evaluation_followthrough_v1',
             'status': 'WAITING_ACTUAL_FINAL_PRETRUTH_SEAL', 'pid': os.getpid(),
             'TEST_truth_opened': False, 'stages': [], 'new_fits': 0, 'new_large_upstream_training': 0}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = output_root / '.status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output_root / 'STATUS.json')
    def stage(name, command, timeout):
        state['status'] = name
        save()
        started = time.monotonic()
        with (output_root / (name + '.log')).open('xb') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, timeout=timeout)
        state['stages'].append({'name': name, 'exit_code': result.returncode,
                                'elapsed_seconds': time.monotonic() - started})
        save()
        if result.returncode:
            raise RuntimeError(name + ' did not complete; preserve frozen assets and failure')
    try:
        verify_code()
        save()
        started = time.monotonic()
        while True:
            status_path = FOLLOW / 'STATUS.json'
            if status_path.exists():
                upstream = json.loads(status_path.read_text())
                status = upstream.get('status')
                if status == 'VALIDATION_QUALIFIED_AND_RISK_PREDICTIONS_FROZEN_TEST_TRUTH_STILL_CLOSED':
                    break
                if status == 'UPSTREAM_NOT_QUALIFIED_NO_FORMAL_RISK_TEST_OR_TEST_OPEN':
                    state['status'] = 'UPSTREAM_NOT_QUALIFIED_TEST_TRUTH_REMAINS_CLOSED'
                    save()
                    return
                if status == 'FAILED_OR_BOUNDED_INCOMPLETE':
                    raise RuntimeError('Prerequisite pipeline failed; TEST truth not authorized')
            if time.monotonic() - started > 16 * 3600:
                raise TimeoutError('Prerequisite wait exceeded16h; no TEST opening')
            time.sleep(30)
        operation = make_operation(output_root)
        state['validated_operation'] = bind(operation)
        state['exclusive_registered_TEST_scope'] = reserve_registered_scope(operation)
        state['status'] = 'ALL_ACTUAL_HASH_GATES_PASS_TEST_NOT_YET_READ'
        save()
        state['TEST_truth_opened'] = 'AUTHORIZED_READER_STARTING_SCOPE_MAY_BE_PARTIAL'
        save()
        stage('REGISTERED_TEST_TRUTH_READER', [sys.executable, '-u', str(ROOT / 'tools/scripts/evaluate_safeconf_orion_guarded_test_agent.py'),
              'evaluate', '--receipt', str(operation), '--output', str(output_root / 'registered_test_truth')], 8 * 3600)
        state['TEST_truth_opened'] = True
        stage('FIXED_5000_CLUSTER_RISK_EVALUATION', [sys.executable, '-u', str(ROOT / 'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py'),
              'evaluate', '--comparison', str(output_root / 'pretruth_comparison'),
              '--truth-receipt', str(output_root / 'registered_test_truth/TRUTH_READER_RECEIPT.json'),
              '--truth-errors', str(output_root / 'registered_test_truth/TEST_TASK_ERRORS.parquet'),
              '--output', str(output_root / 'fixed_risk_evaluation')], 2 * 3600)
        state['status'] = 'REGISTERED_EXTERNAL_REPLICATION_COMPLETE_NO_METHOD_SELECTION'
        state['whole_method_pristine_claim'] = False
        save()
    except Exception as error:
        state['status'] = 'FAILED_OR_BOUNDED_INCOMPLETE'
        state['error_type'] = type(error).__name__
        state['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root)
