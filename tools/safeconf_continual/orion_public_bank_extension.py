"""Explicit immutable Public-bank extension around frozen Orion Source math.

The base scientific contract, seven trained models and numerical functions are
preserved. This module has no fitting or raw-expression reader API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.scripts import seal_safeconf_orion_source_risk_agent as base

BASE_METHOD_SHA = '6b8a26939c8c6134dbf2e8294ee7905fccc2122b570b046f3183a3d2f5caba97'
BASE_RISK_SHA = '8b69e6ce29dc92aa34150b3b08aac62ce8333a10f84ec432a39d0e81ffc64482'
BANK_SCHEMA = 'safeconf_source_public_history_expansion_v1'
CONTRACT_SCHEMA = 'safeconf_orion_pretruth_public_bank_extension_v1'
CONTEXTS = ('HCT116', 'HEK293T')
ARTIFACTS = ('metadata', 'effects', 'controls', 'axis', 'gene_ids')
RECEIPTS = ('eligible_gene_set', 'canonical_unit_provenance', 'metadata_audit_counts',
            'compatibility_diagnostic', 'read_audit', 'coverage_certificate')


def sha(path):
    return base.sha(path)


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def checked(item, immutable=True):
    path = Path(item.get('path', '')).resolve()
    if (not path.is_file() or sha(path) != item.get('sha256')
        or ('bytes' in item and path.stat().st_size != item['bytes'])):
        raise RuntimeError('Exact extension artifact hash/size binding required')
    if immutable and path.stat().st_mode & 0o222:
        raise RuntimeError('Extension artifacts must be immutable read-only files')
    return path


def read_json(item, immutable=True):
    return json.loads(checked(item, immutable).read_text())


def contract(item):
    value = read_json(item)
    if value.get('schema') != CONTRACT_SCHEMA:
        raise RuntimeError('Explicit versioned Public-bank extension contract required')
    original = value.get('base_scientific_contract', {})
    if original.get('sha256') != BASE_METHOD_SHA:
        raise RuntimeError('Original scientific contract must remain unchanged')
    read_json(original)
    expected = {'new_parameter_fits': 0, 'append_only_absent_original_eligible_genes': True,
                'legacy_query_invariance_required': True, 'TEST_truth_used_for_bank_selection': False,
                'SafeConf_scores_used_for_bank_selection': False, 'minimum_primary_gene_clusters_union': 100,
                'minimum_context_tasks': 20}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError('Frozen absent-only/no-fit/pretruth/coverage extension policy differs')
    if sha(base.__file__) != BASE_RISK_SHA:
        raise RuntimeError('Original risk math code changed')
    for entry in value.get('immutable_dependencies', []):
        checked(entry, immutable=False)
    return value


def load_bank(bank_binding, extension_binding, core):
    """Validate actual bank and legacy invariance before frozen inference."""
    registered = contract(extension_binding)
    manifest = read_json(bank_binding)
    if manifest.get('schema') != BANK_SCHEMA or manifest.get('status') != 'COMPLETE':
        raise RuntimeError('Actual completed expanded Public bank required')
    if manifest.get('extension_contract', {}).get('sha256') != extension_binding['sha256']:
        raise RuntimeError('Bank belongs to a different extension contract')
    if checked(manifest['extension_contract']) != checked(extension_binding):
        raise RuntimeError('Bank extension contract path differs')
    original = {entry['path']: entry for entry in core['bindings']}
    for entry in manifest.get('base_source_core_bindings', []):
        path = checked(entry, immutable=False)
        expected = original.get(str(path))
        # The original artifact registry may be separately bound by the builder.
        if expected is None and path.name != 'ARTIFACT_HASHES.json':
            raise RuntimeError('Bank tries to replace an original Source training artifact')
        if expected is None and path != (core['root'] / 'ARTIFACT_HASHES.json').resolve():
            raise RuntimeError('Original Source artifact registry path differs')
        if expected and entry['sha256'] != expected['sha256']:
            raise RuntimeError('Original Source parameters/configuration differ')
    required_models = {Path(entry['path']).name for entry in core['bindings']
                       if Path(entry['path']).suffix == '.joblib'}
    provided = {Path(entry['path']).name for entry in manifest.get('base_source_core_bindings', [])}
    if not required_models <= provided:
        raise RuntimeError('All seven original model hashes must be bound by the bank')
    parent = manifest['parent_bank']
    for name in ARTIFACTS:
        checked(parent[name], immutable=False)
    parent_expected = {'metadata': 'SOURCE_PUBLIC_MEMORY_METADATA.parquet', 'effects': 'SOURCE_PUBLIC_EFFECTS.npy',
                       'controls': 'SOURCE_PUBLIC_CONTROLS.npy', 'axis': 'GENE_MANIFEST.csv', 'gene_ids': 'GENE_IDS.json'}
    for role, filename in parent_expected.items():
        if checked(parent[role], False) != (core['root'] / filename).resolve():
            raise RuntimeError('Expanded bank parent is not the original frozen Source bank')
    extended = manifest['extended_bank']
    paths = {name: checked(extended[name]) for name in ARTIFACTS}
    for name in RECEIPTS:
        checked(manifest[name])
    for name in ['builder_code_binding', 'source_read_contract_binding']:
        checked(manifest[name], immutable=(name != 'builder_code_binding'))
    if (manifest['builder_code_binding'] != registered['builder_code_binding']
        or manifest['source_read_contract_binding'] != registered['source_read_contract_binding']
        or manifest.get('new_fits') != 0 or manifest.get('target_orion_truth_expression_used') is not False):
        raise RuntimeError('Actual bank builder/read-contract/no-fit provenance differs')
    read_contract = read_json(manifest['source_read_contract_binding'])
    if (read_contract.get('schema') != 'safeconf_source_public_historical_read_contract_v1'
        or read_contract.get('source') != manifest['source_asset_binding']
        or read_contract.get('builder') != manifest['builder_code_binding']
        or read_contract.get('no_Orion_expression_or_truth') is not True
        or read_contract.get('no_model_fits') is not True):
        raise RuntimeError('Historical Source read authorization differs')
    policy = manifest.get('append_policy', {})
    if (policy.get('target_absent_old_eligible_gene_set') is not True
        or policy.get('minimum_historical_cells') != 30
        or set(policy.get('allowed_contexts', [])) != {'K562', 'RPE1', 'hepg2', 'jurkat'}):
        raise RuntimeError('Exact registered historical-record addition policy required')
    memory = pd.read_parquet(paths['metadata'])
    effects = np.load(paths['effects'], mmap_mode='r', allow_pickle=False)
    controls = np.load(paths['controls'], mmap_mode='r', allow_pickle=False)
    axis = pd.read_csv(paths['axis'])
    gene_ids = json.loads(paths['gene_ids'].read_text())
    if isinstance(gene_ids, dict): gene_ids = gene_ids.get('gene_ids')
    if (not axis.equals(core['axis']) or gene_ids != core['axis'].gene_name.astype(str).tolist()
        or effects.shape != (len(memory), 3285) or controls.shape != effects.shape):
        raise RuntimeError('Expanded history must use the exact measured frozen Source3285 axis')
    old = core['memory'].reset_index(drop=True)
    count = len(old)
    if count != 2008 or len(memory) < count or not set(old.columns) <= set(memory.columns):
        raise RuntimeError('Original2008 history prefix is missing')
    if not memory.iloc[:count][old.columns].reset_index(drop=True).equals(old):
        raise RuntimeError('Legacy metadata changed')
    for values, original_values in [(effects, core['effects']), (controls, core['controls'])]:
        if values.dtype != original_values.dtype:
            raise RuntimeError('Legacy vector dtype changed')
        for start in range(0, count, 128):
            stop = min(count, start + 128)
            if values[start:stop].tobytes(order='C') != original_values[start:stop].tobytes(order='C'):
                raise RuntimeError('Legacy history vector bits changed')
    old_genes = set(old.loc[old.eligibility.eq(True) & old.perturbation_type.eq('genetic_single_gene'),
                            'perturbation_target'].astype(str))
    if len(old_genes) != 580:
        raise RuntimeError('Original eligible gene set differs')
    tasks_path = checked(manifest['source_training_tasks_binding'], False)
    if (tasks_path != (core['root'] / 'SOURCE_TASKS.parquet').resolve()
        or manifest['source_training_tasks_binding'] != read_contract.get('source_training_tasks_binding')):
        raise RuntimeError('Original Source risk-training task identity differs')
    training_genes = set(pd.read_parquet(tasks_path, columns=['gene']).gene.astype(str))
    if len(training_genes) != 575 or not training_genes <= old_genes:
        raise RuntimeError('Original Source575 training history is not preserved')
    added = memory.iloc[count:]
    if set(added.perturbation_target.astype(str)) & old_genes:
        raise RuntimeError('New records for original eligible genes are forbidden')
    if (not memory.experiment_id.is_unique or not np.array_equal(memory.effect_vector_row, np.arange(len(memory)))
        or not added.eligibility.eq(True).all() or not added.perturbation_type.eq('genetic_single_gene').all()
        or not set(added.context) <= set(policy['allowed_contexts'])
        or not np.isfinite(added.n_cells).all() or (added.n_cells < 30).any()
        or not np.isfinite(added.n_batches).all() or (added.n_batches < 1).any()):
        raise RuntimeError('New history metadata/row alignment/eligibility invalid')
    for start in range(0, len(memory), 128):
        if (not np.isfinite(effects[start:start + 128]).all()
            or not np.isfinite(controls[start:start + 128]).all()
            or (controls[start:start + 128] < 0).any()):
            raise RuntimeError('Measured history/control vectors must be finite with nonnegative controls')
    all_genes = set(memory.loc[memory.eligibility.eq(True), 'perturbation_target'].astype(str))
    eligible = read_json(manifest['eligible_gene_set'])
    expected_sets = {'old_eligible_gene_symbols': old_genes, 'new_eligible_gene_symbols': all_genes - old_genes,
                     'eligible_gene_symbols': all_genes}
    if any(eligible.get(name) != sorted(genes) for name, genes in expected_sets.items()):
        raise RuntimeError('Actual eligible-gene identity certificate differs')
    diagnostic = pd.read_csv(checked(manifest['compatibility_diagnostic']))
    if (len(diagnostic) != count or not diagnostic.experiment_id.astype(str).equals(old.experiment_id.astype(str))
        or not diagnostic.n_cells_match.eq(True).all() or not diagnostic.n_batches_match.eq(True).all()
        or not np.isfinite(diagnostic[['max_abs_effect_gap', 'max_abs_control_gap']].to_numpy(float)).all()
        or (diagnostic[['max_abs_effect_gap', 'max_abs_control_gap']].to_numpy(float) > 1e-5).any()):
        raise RuntimeError('Historical cache compatibility proof differs from fixed legacy check')
    provenance = pd.read_csv(checked(manifest['canonical_unit_provenance']), keep_default_na=False)
    if (len(provenance) != len(memory) or not provenance.physical_unit_id.is_unique
        or provenance.experiment_id.astype(str).tolist() != memory.experiment_id.astype(str).tolist()
        or not provenance.source_sha256.eq(manifest['source_asset_binding']['sha256']).all()):
        raise RuntimeError('Physical historical-unit provenance is incomplete or repeated')
    audit = read_json(manifest['read_audit'])
    if (audit.get('all_exact_selected_Source_rows_completed') is not True
        or audit.get('contract_sha256') != manifest['source_read_contract_binding']['sha256']
        or audit.get('builder_code_sha256') != manifest['builder_code_binding']['sha256']
        or audit.get('source_sha256') != manifest['source_asset_binding']['sha256']
        or audit.get('excluded_context_rows_decoded') != 0 or audit.get('Orion_expression_truth_values_read') != 0
        or audit.get('normalization_altered') is not False or audit.get('new_fits') != 0):
        raise RuntimeError('Exact completed Source-only expression-access audit required')
    counts = read_json(manifest['metadata_audit_counts'])
    if (counts.get('Source575_risk_training_gene_history_unchanged') is not True
        or counts.get('physical_units_unique') is not True or counts.get('new_fits') != 0
        or counts.get('Orion_expression_truth_read') is not False):
        raise RuntimeError('Source training invariance or no-TEST access proof missing')
    claimed = manifest.get('legacy_prefix', {})
    for key, expected in {'n_items': 2008, 'metadata_exact': True, 'effects_bitwise_equal': True,
                          'controls_bitwise_equal': True, 'old_eligible_gene_count': 580,
                          'no_old_eligible_target_appended': True}.items():
        if claimed.get(key) != expected: raise RuntimeError('Actual legacy prefix differs from bank receipt')
    coverage_ids = read_json(manifest['coverage_certificate'])['qualified_query_ids']
    checked(coverage_ids)
    # Raw Source X is not reopened here. Its exact opaque asset binding is part
    # of the immutable bank/read contract and the completed builder audit.
    # Executable code and original task hashes are checked above/by contract;
    # retrieval inputs below consist only of immutable new bank/receipt files.
    bindings = ([bank_binding, extension_binding] + list(extended.values()) + [manifest[name] for name in RECEIPTS]
                + [coverage_ids, manifest['source_read_contract_binding']])
    return dict(core, memory=memory, effects=effects, controls=controls), manifest, bindings, old_genes


def frame_equal_bits(left, right):
    if not left.equals(right): return False
    for column in left.select_dtypes(include=[np.number]).columns:
        if left[column].to_numpy().tobytes() != right[column].to_numpy().tobytes(): return False
    return True


def assert_legacy_invariance(queries, original, expanded, old_genes):
    use = queries.target_gene_symbol.astype(str).isin(old_genes).to_numpy()
    for name in original['scores'].columns:
        left = original['scores'].loc[use, name].reset_index(drop=True)
        right = expanded['scores'].loc[use, name].reset_index(drop=True)
        if not frame_equal_bits(left.to_frame(), right.to_frame()):
            raise RuntimeError('Legacy-gene query score/status/identity changed: ' + name)
    for reference in ('Uniform', 'Manual', 'Learned'):
        if (not frame_equal_bits(original['reference_features'][reference].loc[use].reset_index(drop=True),
                expanded['reference_features'][reference].loc[use].reset_index(drop=True))
            or original['priors'][reference][use].tobytes() != expanded['priors'][reference][use].tobytes()):
            raise RuntimeError('Legacy-gene query Public feature/prior changed')
    old_pairs = original['pairs'][original['pairs'].query_id.isin(queries.loc[use, 'query_id'])].reset_index(drop=True)
    new_pairs = expanded['pairs'][expanded['pairs'].query_id.isin(queries.loc[use, 'query_id'])].reset_index(drop=True)
    old_weights = original['weights'][original['weights'].query_id.isin(queries.loc[use, 'query_id'])].reset_index(drop=True)
    new_weights = expanded['weights'][expanded['weights'].query_id.isin(queries.loc[use, 'query_id'])].reset_index(drop=True)
    if not frame_equal_bits(old_pairs, new_pairs) or not frame_equal_bits(old_weights, new_weights):
        raise RuntimeError('Legacy history pair feature/learned score/weight changed')
    return {'status': 'EXACT_LEGACY_QUERY_INVARIANCE_PASS', 'legacy_query_rows': int(use.sum()),
            'scope': 'all registered VALIDATION and TEST query identities for original eligible genes',
            'all_score_columns_features_priors_unchanged': True,
            'all_pair_features_transfer_scores_weights_unchanged': True}


def checked_coverage(certificate_binding, queries, scores, manual_features=None):
    certificate = read_json(certificate_binding)
    if certificate.get('schema') != 'safeconf_public_bank_metadata_coverage_certificate_v1':
        raise RuntimeError('Explicit metadata-only coverage certificate required')
    if certificate.get('TEST_truth_read') is not False or certificate.get('risk_scores_used') is not False:
        raise RuntimeError('Bank coverage certificate may use metadata only')
    eligible = pd.read_csv(checked(certificate['qualified_query_ids']), sep='\t', keep_default_na=False)
    if list(eligible.columns) != ['query_id', 'context_id'] or eligible.query_id.duplicated().any():
        raise RuntimeError('Coverage eligibility contains identity IDs only')
    if not queries.query_id.is_unique or not scores.query_id.is_unique:
        raise RuntimeError('Unique frozen query/prediction identities required')
    if scores[base.QUERY].to_dict('records') != queries[base.QUERY].to_dict('records'):
        raise RuntimeError('Coverage score/query identity axis differs')
    all_queries = queries.set_index('query_id')
    if not set(eligible.query_id) <= set(all_queries.index):
        raise RuntimeError('Metadata coverage query IDs outside frozen scope')
    selected = all_queries.loc[eligible.query_id]
    if (not selected.role.eq('TEST').all()
        or not np.array_equal(selected.context_id.to_numpy(), eligible.context_id.to_numpy())):
        raise RuntimeError('Metadata qualification query role/context differs')
    methods = ['Magnitude'] + [f'{ref}_{kind}' for ref, kind in base.METHODS] + [
        'Manual_WeightedHistoryDistance', 'Learned_WeightedHistoryDistance',
        'Uniform_DirectRMSE', 'Manual_DirectRMSE', 'Learned_DirectRMSE']
    scores = scores.copy()
    support = 'NegativeSourceHistorySupport'
    if manual_features is not None:
        if len(manual_features) != len(scores): raise RuntimeError('Manual support feature row axis differs')
        scores[support] = -manual_features.log_history_support.to_numpy(float)
    if support not in scores: raise RuntimeError('All13 primary candidates including fixed negative support required')
    methods.append(support)
    table = scores.set_index('query_id').loc[eligible.query_id]
    valid = (table.source_history_n.to_numpy() > 0) & np.isfinite(table[methods].to_numpy(float)).all(axis=1)
    counts = {context: int((valid & table.context_id.eq(context).to_numpy()).sum()) for context in CONTEXTS}
    union = int(queries.set_index('query_id').loc[eligible.query_id].loc[valid, 'target_gene_id'].nunique())
    if union < 100 or any(value < 20 for value in counts.values()):
        raise RuntimeError('Expanded primary requires100independentgeneclusters union and original20task/context validity')
    return {'status': 'PRETRUTH_PRIMARY_COVERAGE_PASS', 'minimum_gene_clusters_union': 100,
            'minimum_tasks_per_context': 20, 'primary_gene_clusters_union': union,
            'primary_tasks_per_context': counts, 'metadata_qualification_binding': certificate['qualified_query_ids'],
            'metadata_counts_not_numeric_risk_features': True}


def exact_extension_chain(operation):
    """Validate every extension receipt before delegating any TEST operation."""
    section = operation.get('public_bank_extension', {})
    if section.get('schema') != 'safeconf_orion_public_bank_extension_operation_v1':
        raise RuntimeError('Explicit expanded-bank operation chain required')
    extension = section['extension_contract']
    contract(extension)
    bank = read_json(section['bank_manifest'])
    if bank.get('schema') != BANK_SCHEMA or bank.get('status') != 'COMPLETE':
        raise RuntimeError('No TEST operation before actual bank completion')
    if bank.get('extension_contract') != extension:
        raise RuntimeError('Operation bank/extension identity differs')
    comparison = read_json(section['comparison_receipt'])
    if (comparison.get('schema') != 'safeconf_orion_extended_bank_comparison_receipt_v1'
        or comparison.get('status') != 'FROZEN_PRETRUTH'
        or comparison.get('no_new_parameter_fits') is not True
        or comparison.get('no_TEST_truth_or_error') is not True
        or comparison.get('expanded_primary_uses_original13_candidate_ids_and_metrics') is not True):
        raise RuntimeError('Expanded comparison receipt required')
    if (comparison.get('extension_contract') != extension or comparison.get('bank_manifest') != section['bank_manifest']
        or comparison.get('base_comparison') != operation.get('pretruth_comparison')):
        raise RuntimeError('Operation comparison/bank/contract chain differs')
    sealer = read_json(comparison['extension_seal_receipt'])
    if (sealer.get('schema') != 'safeconf_orion_expanded_bank_pretruth_seal_receipt_v1'
        or sealer.get('status') != 'COMPLETE' or sealer.get('new_parameter_fits') != 0
        or sealer.get('extension_contract') != extension or sealer.get('bank_manifest') != section['bank_manifest']
        or sealer.get('base_risk_seal') != operation.get('risk_seal_manifest')
        or sealer.get('legacy_invariance', {}).get('status') != 'EXACT_LEGACY_QUERY_INVARIANCE_PASS'
        or sealer.get('coverage', {}).get('status') != 'PRETRUTH_PRIMARY_COVERAGE_PASS'):
        raise RuntimeError('Actual completed invariant expanded sealer/coverage proof missing')
    checked(sealer['base_risk_seal'])
    checked(sealer['original_fixed_scores'])
    if comparison.get('legacy_original2008_fixed_scores') != sealer['original_fixed_scores']:
        raise RuntimeError('Separate fixed original2008 score archive identity differs')
    registration = read_json(sealer['seal_registration'])
    implementations = section.get('implementation_bindings', {})
    if (registration.get('schema') != 'safeconf_orion_public_bank_extension_seal_registration_v1'
        or registration.get('extension_contract') != extension
        or registration.get('bank_manifest') != section['bank_manifest']
        or registration.get('base_comparison_manifest') != operation.get('risk_comparison_manifest')
        or sealer.get('executed_sealer') != implementations.get('extension_sealer')
        or sealer.get('executed_helper') != implementations.get('extension_helper')
        or sealer.get('base_risk_math') != implementations.get('base_risk_math')
        or registration.get('implementation_bindings') !=
           {name: implementations.get(name) for name in ('extension_sealer', 'extension_helper')}):
        raise RuntimeError('Exact actual sealer registration/executed implementation lineage required')
    coverage = sealer['coverage']
    if comparison.get('final_primary_coverage') != coverage:
        raise RuntimeError('Final13-candidate primary coverage differs from expanded seal')
    scores = pd.read_parquet(checked(comparison['base_comparison']['scores']))
    actual_coverage = checked_coverage(bank['coverage_certificate'], scores[base.QUERY], scores)
    primary = (scores.source_history_n.to_numpy() > 0) & np.isfinite(scores[
        ['Magnitude'] + [f'{ref}_{kind}' for ref, kind in base.METHODS] + [
            'Manual_WeightedHistoryDistance', 'Learned_WeightedHistoryDistance',
            'Uniform_DirectRMSE', 'Manual_DirectRMSE', 'Learned_DirectRMSE',
            'NegativeSourceHistorySupport']].to_numpy(float)).all(axis=1)
    if (actual_coverage != coverage
        or not np.array_equal(scores.PRETRUTH_PRIMARY_COMMON_ELIGIBLE.to_numpy(bool), primary)):
        raise RuntimeError('Actual frozen primary13-candidate finite intersection differs')
    if (set(coverage['primary_tasks_per_context']) != set(CONTEXTS)
        or not isinstance(coverage.get('primary_gene_clusters_union'), int)
        or coverage['primary_gene_clusters_union'] < 100):
        raise RuntimeError('Primary coverage requires100independentgeneclusters union')
    for count in coverage['primary_tasks_per_context'].values():
        if not isinstance(count, int) or count < 20: raise RuntimeError('Original context-validity gate not satisfied')
    for entry in sealer['retrieval_bank_bindings']:
        checked(entry)
    for name, item in section.get('implementation_bindings', {}).items():
        checked(item, immutable=False)
    return section, comparison, sealer
