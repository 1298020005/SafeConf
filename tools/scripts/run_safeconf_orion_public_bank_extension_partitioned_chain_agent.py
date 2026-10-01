#!/usr/bin/env python3
"""Explicit bank-extension receipts around unchanged comparison/TEST functions.

No automatic waiting or opening coordinator is included. Actual TEST requires
an immutable root-and-independent-review approval of the completed chain.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual import orion_public_bank_extension as extension
from tools.scripts import evaluate_safeconf_orion_frozen_risk_agent as evaluator
from tools.scripts import evaluate_safeconf_orion_guarded_test_agent as reader
from tools.scripts import seal_safeconf_orion_public_bank_extension_partitioned_agent as sealer
from tools.safeconf_continual import orion_legacy_partition as partition
from tools.scripts import seal_safeconf_orion_source_risk_agent as base
from tools.scripts import run_safeconf_orion_registered_evaluation_followthrough as coordinator

BASE_READER_SHA = '852fcd764b567a63f04c4a5539a118c8a715f664bf534184018010ac269087b2'
BASE_EVALUATOR_SHA = 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c'
BASE_COORDINATOR_SHA = 'f08f27279d8a0f45f67ba97d539bfbe27ed9351a75fd57c1eaab57a91c5ece85'
OUTER_OPERATION_SCHEMA = 'safeconf_orion_extended_bank_postseal_test_operation_v1'


def implementation_bindings():
    return {name: extension.binding(path) for name, path in {
        'extension_chain': __file__, 'extension_helper': extension.__file__,
        'extension_sealer': sealer.__file__, 'partition_helper': partition.__file__, 'base_risk_math': base.__file__,
        'base_test_reader': reader.__file__, 'base_risk_evaluator': evaluator.__file__,
        'base_once_only_coordinator': coordinator.__file__}.items()}


def pinned_base():
    for path, expected in [(reader.__file__, BASE_READER_SHA), (evaluator.__file__, BASE_EVALUATOR_SHA),
                           (base.__file__, extension.BASE_RISK_SHA),
                           (coordinator.__file__, BASE_COORDINATOR_SHA)]:
        if extension.sha(path) != expected: raise RuntimeError('Original delegated code pin changed')


def prepare_comparison(seal_root, output):
    pinned_base()
    seal_root, output = Path(seal_root).resolve(), Path(output).resolve()
    if output.is_relative_to(seal_root) or seal_root.is_relative_to(output):
        raise RuntimeError('Comparison output must be isolated from its immutable input seal')
    receipt = extension.read_json(extension.binding(seal_root / 'EXTENSION_SEAL_RECEIPT.json'))
    if receipt.get('schema') != 'safeconf_orion_expanded_bank_pretruth_seal_receipt_v1' or receipt.get('status') != 'COMPLETE':
        raise RuntimeError('Actual expanded pretruth seal must complete before comparison')
    registration = extension.read_json(receipt['seal_registration'])
    if partition.checked_amendment(receipt['legacy_partition_technical_amendment']['path'], registration, sealer.__file__, chain_path=__file__) != receipt['legacy_partition_technical_amendment']:
        raise RuntimeError('Actual partition technical amendment differs')
    if receipt.get('executed_partition_helper') != extension.binding(partition.__file__):
        raise RuntimeError('Actual partition helper differs')
    extension.contract(receipt['extension_contract'])
    bank = extension.read_json(receipt['bank_manifest'])
    if bank.get('status') != 'COMPLETE': raise RuntimeError('Completed actual bank required')
    for item in receipt['retrieval_bank_bindings']: extension.checked(item)
    if (receipt['executed_sealer'] != extension.binding(sealer.__file__)
        or receipt['executed_helper'] != extension.binding(extension.__file__)
        or receipt['base_risk_math'] != extension.binding(base.__file__)):
        raise RuntimeError('Executed sealer/helper/base code changed')
    extension.checked(receipt['base_risk_seal'])
    extension.checked(receipt['original_fixed_scores'])
    if extension.checked(receipt['base_risk_seal']) != seal_root / 'RISK_SEAL_MANIFEST.json':
        raise RuntimeError('Expanded seal receipt belongs to another risk input directory')
    if output.exists(): raise RuntimeError('New immutable expanded comparison output required')
    stage = output.with_name(output.name + '.incomplete')
    spec = evaluator.prepare_comparison(seal_root, stage)
    # The frozen evaluator's default TSV parser can change final float bits;
    # the frozen reader compares against its round-trip parser exactly. Keep
    # the sealed values, without changing math, rounding or equality tolerance.
    scores_path = stage / 'PRETRUTH_COMPARISON_SCORES.parquet'
    scores = pd.read_parquet(scores_path)
    exact = reader.score_table(seal_root / 'PRETRUTH_RISKS.tsv')
    if exact[base.QUERY].to_dict('records') != scores[base.QUERY].to_dict('records'):
        raise RuntimeError('Serialization reconciliation query identity differs')
    columns = [name for name in spec['methods'] if name != evaluator.SUPPORT]
    changed, maximum_gap = {}, {}
    for name in columns:
        values = pd.to_numeric(exact[name].replace({'NaN': np.nan, '': np.nan}), errors='raise').to_numpy(float)
        previous = scores[name].to_numpy(float)
        changed[name] = int(np.sum(~((values == previous) | (np.isnan(values) & np.isnan(previous)))))
        finite = np.isfinite(values) & np.isfinite(previous)
        maximum_gap[name] = float(np.max(np.abs(values[finite] - previous[finite]))) if finite.any() else 0.
        scores[name] = values
    known = scores.source_history_n.to_numpy(int) > 0
    scores['PRETRUTH_PRIMARY_COMMON_ELIGIBLE'] = known & np.isfinite(scores[spec['methods']].to_numpy(float)).all(axis=1)
    scores_path.chmod(0o600); scores.to_parquet(scores_path, index=False); scores_path.chmod(0o444)
    reloaded = pd.read_parquet(scores_path)
    for name in columns:
        if not np.array_equal(reloaded[name].to_numpy(float), scores[name].to_numpy(float), equal_nan=True):
            raise RuntimeError('Final comparison Parquet changed the sealed candidate values')
    support = -pd.read_parquet(seal_root / 'Manual_P_PUBLIC_FEATURES.parquet').log_history_support.to_numpy(float)
    if not np.array_equal(reloaded[evaluator.SUPPORT].to_numpy(float), support, equal_nan=True):
        raise RuntimeError('Fixed negative Source support formula changed')
    coverage = extension.checked_coverage(bank['coverage_certificate'], reloaded[base.QUERY], reloaded)
    if coverage != receipt.get('coverage'):
        raise RuntimeError('Final13-candidate finite primary differs from expanded pretruth seal')
    spec['scores_sha256'] = extension.sha(scores_path)
    spec['comparison_scores'] = dict(extension.binding(scores_path), path=str(output / scores_path.name))
    reconciliation = {'source': extension.binding(seal_root / 'PRETRUTH_RISKS.tsv'),
        'parser': 'unchanged guarded reader.score_table: float_precision=round_trip then existing numeric conversion',
        'changed_last_bit_values_by_candidate': changed, 'numeric_formula_or_tolerance_changed': False,
        'maximum_default_parser_gap_by_candidate': maximum_gap,
        'all_existing_numeric_candidates_exact_after_parquet_reload': True,
        'support_formula_unchanged': True, 'executed_adapter': extension.binding(__file__)}
    spec['expanded_bank_serialization_receipt'] = reconciliation
    spec['expanded_bank_final_primary_coverage'] = coverage
    spec_path = stage / 'COMPARISON_SPEC.json'; spec_path.chmod(0o600); spec_path.unlink()
    base.write_json(spec_path, spec); spec_path.chmod(0o444)
    stage.rename(output)
    comparison = {'spec': extension.binding(output / 'COMPARISON_SPEC.json'),
                  'scores': extension.binding(output / 'PRETRUTH_COMPARISON_SCORES.parquet')}
    value = {'schema': 'safeconf_orion_extended_bank_comparison_receipt_v1', 'status': 'FROZEN_PRETRUTH',
             'extension_contract': receipt['extension_contract'], 'bank_manifest': receipt['bank_manifest'],
             'extension_seal_receipt': extension.binding(seal_root / 'EXTENSION_SEAL_RECEIPT.json'),
             'legacy_partition_technical_amendment': receipt['legacy_partition_technical_amendment'],
             'base_comparison': comparison,
             'legacy_original2008_fixed_scores': receipt['original_fixed_scores'],
             'legacy_comparison_scope': 'separate fixed archive/legacy-known cohort; excluded from expanded-primary finite intersection',
             'expanded_primary_uses_original13_candidate_ids_and_metrics': True,
             'no_new_parameter_fits': True, 'no_TEST_truth_or_error': True,
             'serialization_reconciliation': reconciliation,
             'final_primary_coverage': coverage,
             'implementation_bindings': implementation_bindings()}
    path = output / 'EXTENSION_COMPARISON_RECEIPT.json'
    base.write_json(path, value); path.chmod(0o444)
    return value


def reviewed_section(section, comparison):
    seal = extension.read_json(comparison['extension_seal_receipt'])
    registered = extension.read_json(seal['seal_registration'])
    item = partition.checked_amendment(seal['legacy_partition_technical_amendment']['path'], registered, sealer.__file__, chain_path=__file__)
    if comparison.get('legacy_partition_technical_amendment') != item or seal.get('executed_partition_helper') != extension.binding(partition.__file__):
        raise RuntimeError('Exact technical partition lineage required before delegation/evaluation')
    if section.get('implementation_bindings') != implementation_bindings():
        raise RuntimeError('Actual extension/delegated implementation chain differs')
    if comparison.get('implementation_bindings') != implementation_bindings():
        raise RuntimeError('Frozen extension comparison implementation pins differ')
    review = extension.read_json(section['root_and_independent_review'])
    if (review.get('schema') != 'safeconf_orion_public_bank_extension_final_review_v1'
        or review.get('status') != 'APPROVED_FOR_ONCE_ONLY_REGISTERED_TEST'
        or review.get('root_approved') is not True or review.get('independent_review_passed') is not True
        or review.get('bank_manifest') != section['bank_manifest']
        or review.get('extension_contract') != section['extension_contract']
        or review.get('comparison_receipt') != section['comparison_receipt']
        or review.get('implementation_bindings') != implementation_bindings()):
        raise RuntimeError('Root and independent review of actual completed bank/seal/chain is required')
    independent = extension.read_json(review['independent_review_receipt'])
    if independent.get('status') != 'PASS':
        raise RuntimeError('Exact independent review receipt must report PASS')


def delegated_payload(receipt_path, operation, claim=None):
    value = dict(operation, schema=reader.SCHEMA,
                 validated_extension_outer_operation=extension.binding(receipt_path),
                 delegated_after_extension_validation=True)
    if claim is not None: value['exclusive_registered_scope'] = claim
    return value


def authorize_extended_test(receipt_path):
    """Every explicit extension check runs before original raw-input guards."""
    pinned_base()
    operation = extension.read_json(extension.binding(receipt_path))
    if operation.get('schema') != OUTER_OPERATION_SCHEMA:
        raise RuntimeError('Expanded-bank launch requires the new outer operation schema; base CLI is not an extension validator')
    section, comparison, seal_receipt = extension.exact_extension_chain(operation)
    reviewed_section(section, comparison)
    # Revalidate actual arrays/prefix, not only claims in the bank JSON.
    core = base.source_core()
    _, bank, bank_bindings, _ = extension.load_bank(section['bank_manifest'], section['extension_contract'], core)
    expected = seal_receipt['retrieval_bank_bindings']
    if bank_bindings != expected:
        raise RuntimeError('Actual retrieval bank differs from sealed inference bank')
    # The unchanged old CLI rejects the outer schema. Only this wrapper creates
    # a base-schema delegation after every extension/actual-bank review gate.
    # Authorization exposes no lasting receipt usable by the base CLI. The
    # persistent delegation is created only after the once-only claim below.
    path = Path(receipt_path).resolve().parent / f'.BASE_GUARD_VALIDATION.{os.getpid()}.{uuid.uuid4().hex}.json'
    base.write_json(path, delegated_payload(receipt_path, operation)); path.chmod(0o444)
    try: result = reader.authorize_test(path)
    finally: path.unlink()
    result['public_bank_extension_verified'] = True
    result['validated_outer_operation'] = extension.binding(receipt_path)
    return result


def evaluate_extended_test(receipt_path, output):
    authorized = authorize_extended_test(receipt_path)
    operation = extension.read_json(extension.binding(receipt_path))
    output = Path(output).resolve()
    if output != Path(operation['output_path']).resolve() or output.exists():
        raise RuntimeError('Exact new registered truth output required before claiming once-only scope')
    exclusive_scope = coordinator.reserve_registered_scope(receipt_path)
    delegated = delegated_payload(receipt_path, operation, exclusive_scope)
    path = Path(receipt_path).resolve().parent / ('DELEGATED_BASE_OPERATION.' + extension.sha(receipt_path) + '.json')
    if path.exists(): raise RuntimeError('A persistent delegation already exists; no second opening')
    base.write_json(path, delegated); path.chmod(0o444)
    delegated_binding = extension.binding(path)
    result = reader.evaluate_test(path, output)
    value = {'schema': 'safeconf_orion_extended_bank_truth_receipt_v1', 'status': 'COMPLETE',
             'operation': extension.binding(receipt_path),
             'delegated_base_operation': delegated_binding,
             'exclusive_registered_scope': exclusive_scope,
             'extension_operation': operation['public_bank_extension'],
             'base_truth_receipt': extension.binding(output / 'TRUTH_READER_RECEIPT.json'),
             'truth_errors': extension.binding(output / 'TEST_TASK_ERRORS.parquet'),
             'executed_extension_chain': extension.binding(__file__),
             'delegated_unchanged_reader': extension.binding(reader.__file__),
             'bank_verified_before_any_TEST_reader_call': True,
             'new_fits': 0, 'target_CDF_fit': False}
    path = output / 'EXTENSION_TRUTH_RECEIPT.json'
    base.write_json(path, value); path.chmod(0o444)
    return result


def evaluate_extended_risk(comparison_root, extension_truth_receipt, output):
    pinned_base()
    receipt = extension.read_json(extension.binding(extension_truth_receipt))
    if (receipt.get('schema') != 'safeconf_orion_extended_bank_truth_receipt_v1'
        or receipt.get('status') != 'COMPLETE' or receipt.get('bank_verified_before_any_TEST_reader_call') is not True
        or receipt.get('executed_extension_chain') != extension.binding(__file__)
        or receipt.get('delegated_unchanged_reader') != extension.binding(reader.__file__)
        or receipt.get('new_fits') != 0 or receipt.get('target_CDF_fit') is not False):
        raise RuntimeError('Actual completed extension-wrapped truth receipt required')
    operation_path = extension.checked(receipt['operation'])
    operation = extension.read_json(receipt['operation'])
    if operation.get('schema') != OUTER_OPERATION_SCHEMA:
        raise RuntimeError('Actual expanded truth must bind the new outer operation schema')
    section, comparison, _ = extension.exact_extension_chain(operation)
    reviewed_section(section, comparison)
    if section != receipt['extension_operation']:
        raise RuntimeError('Actual operation extension differs from truth receipt')
    comparison_root = Path(comparison_root).resolve()
    if extension.checked(section['comparison_receipt']) != comparison_root / 'EXTENSION_COMPARISON_RECEIPT.json':
        raise RuntimeError('Exact registered expanded comparison root required')
    claim_path = extension.checked(receipt['exclusive_registered_scope'])
    claim = extension.read_json(receipt['exclusive_registered_scope'])
    if (claim_path != (coordinator.BASE / 'ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json').resolve()
        or claim.get('schema') != 'safeconf_orion_exclusive_registered_test_scope_v1'
        or claim.get('operation') != receipt['operation']
        or claim.get('scientific_contract') != operation.get('scientific_contract')
        or claim.get('forty_object_manifest') != operation.get('forty_object_manifest')
        or claim.get('gene_split_sha256') != operation['metadata_bindings']['GENE_SPLIT.csv']
        or claim.get('output_path') != operation.get('output_path')
        or claim.get('automatic_second_registration_allowed') is not False
        or claim.get('failure_does_not_release_scope') is not True):
        raise RuntimeError('Actual immutable once-only claim lineage differs')
    delegated = extension.read_json(receipt['delegated_base_operation'])
    if delegated != delegated_payload(operation_path, operation, receipt['exclusive_registered_scope']):
        raise RuntimeError('Base-schema operation was not an exact validated outer-operation delegation')
    truth = extension.checked(receipt['base_truth_receipt'])
    base_receipt = json.loads(truth.read_text())
    if (base_receipt.get('TEST_access_receipt_path') != receipt['delegated_base_operation']['path']
        or base_receipt.get('TEST_access_receipt_sha256') != receipt['delegated_base_operation']['sha256']):
        raise RuntimeError('Delegated truth reader belongs to another TEST operation')
    # The exact base completion/semantic gates also precede hashing errors.
    # A generic COMPLETE sidecar must not cause any error-label byte access.
    spec = extension.read_json(comparison['base_comparison']['spec'])
    evaluator.guarded_truth_receipt(base_receipt, spec)
    error_binding = base_receipt.get('truth_errors', {})
    if any(error_binding.get(key) != receipt['truth_errors'].get(key) for key in ('path', 'sha256')):
        raise RuntimeError('Wrapped truth errors differ from exact completed base-reader receipt')
    errors = extension.checked(receipt['truth_errors'])
    result = evaluator.evaluate(comparison_root, truth, errors, output)
    value = {'schema': 'safeconf_orion_extended_bank_evaluation_receipt_v1', 'status': 'COMPLETE',
             'extension_truth_receipt': extension.binding(extension_truth_receipt),
             'comparison_receipt': section['comparison_receipt'], 'bank_manifest': section['bank_manifest'],
             'extension_contract': section['extension_contract'], 'executed_chain': extension.binding(__file__),
             'delegated_evaluator': extension.binding(evaluator.__file__), 'new_fits': 0,
             'legacy_scores_remain_separate_archive': True}
    path = Path(output) / 'EXTENSION_EVALUATION_RECEIPT.json'
    base.write_json(path, value); path.chmod(0o444)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    compare = sub.add_parser('prepare-comparison'); compare.add_argument('--seal', type=Path, required=True); compare.add_argument('--output', type=Path, required=True)
    authorize = sub.add_parser('authorize'); authorize.add_argument('--receipt', type=Path, required=True)
    test = sub.add_parser('evaluate-test'); test.add_argument('--receipt', type=Path, required=True); test.add_argument('--output', type=Path, required=True)
    evaluate = sub.add_parser('evaluate-risk'); evaluate.add_argument('--comparison', type=Path, required=True); evaluate.add_argument('--extension-truth-receipt', type=Path, required=True); evaluate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare-comparison': prepare_comparison(args.seal, args.output)
    elif args.command == 'authorize': authorize_extended_test(args.receipt)
    elif args.command == 'evaluate-test': evaluate_extended_test(args.receipt, args.output)
    else: evaluate_extended_risk(args.comparison, args.extension_truth_receipt, args.output)


if __name__ == '__main__': main()
