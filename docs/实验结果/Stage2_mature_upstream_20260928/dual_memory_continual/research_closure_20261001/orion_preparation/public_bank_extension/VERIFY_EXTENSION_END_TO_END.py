"""Generated full extension chain. No actual Source or Orion input is read.

Only fixture-local scientific SHA/root constants and bootstrap draw count are
changed in this process. Every production guard/inference/reader function runs
unchanged; seven generated deterministic predictors are never fitted.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
sys.path.insert(0, str(REPO))
from tools.safeconf_continual import orion_public_bank_extension as e
from tools.scripts import run_safeconf_orion_public_bank_extension_chain_agent as c
from tools.scripts import seal_safeconf_orion_source_risk_agent as b
from tools.scripts import seal_safeconf_orion_public_bank_extension_agent as s


class GeneratedPredictor:
    def __init__(self, columns):
        self.columns = columns
        self.preprocessor = SimpleNamespace(medians_=np.zeros(len(columns)))
        self.model = SimpleNamespace(n_features_in_=2 * len(columns))
    def predict(self, frame, clip=True):
        value = .2 + .001 * frame[self.columns].to_numpy(float).sum(axis=1)
        return np.clip(value, 0, 1) if clip else value


def immutable(path, value):
    b.write_json(path, value); Path(path).chmod(0o444)
    return e.binding(path)


def table(path, frame):
    frame.to_csv(path, sep='\t', index=False, float_format='%.17g')
    Path(path).chmod(0o444)
    return e.binding(path)


def registry(root, tsv=False):
    entries = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': e.sha(p)}
               for p in sorted(root.iterdir()) if p.is_file() and not p.name.startswith('ARTIFACT_HASHES')]
    path = root / ('ARTIFACT_HASHES.tsv' if tsv else 'ARTIFACT_HASHES.json')
    return table(path, pd.DataFrame(entries)) if tsv else immutable(path, entries)


def make_fixture(root):
    source, bank, md, raw, fit = [root / name for name in ['source', 'bank', 'metadata', 'raw', 'fit']]
    for folder in [source, bank, md, raw, fit]: folder.mkdir()
    axis_names = ['G1', 'G3'] + [f'MEASURED{i}' for i in range(3283)]
    new_genes = ['G2'] + [f'NEW{i}' for i in range(99)]
    full_names = axis_names + new_genes
    full = pd.DataFrame({'ensembl_id': [f'E{i}' for i in range(len(full_names))],
                         'gene_name': full_names, 'gene_token_id': np.arange(len(full_names))})
    axis = pd.DataFrame({'axis_index': np.arange(3285), 'source_index': np.arange(3285),
                         'gene_name': axis_names, 'orion_gene_token_id': np.arange(3285),
                         'orion_ensembl_id': full.ensembl_id.iloc[:3285].to_numpy()})
    pq.write_table(pa.Table.from_pandas(full, preserve_index=False), md / 'gene_metadata.parquet')
    axis.to_csv(md / 'GENE_MANIFEST.csv', index=False); axis.to_csv(source / 'GENE_MANIFEST.csv', index=False)
    pd.DataFrame({'gene': full_names, 'gene_role': ['TEST' if x in ['G1'] + new_genes else 'VALIDATION' for x in full_names]}).to_csv(md / 'GENE_SPLIT.csv', index=False)
    immutable(md / 'SYNTHETIC_FIXTURE_ONLY.json', {'synthetic_only': True})
    immutable(md / 'PREPARATION_POLICY.json', {'synthetic_only': True, 'normalization_scale': 4000})
    science = immutable(root / 'SCIENCE.json', {'synthetic_only': True, 'normalization': {
        'formula': 'mean_cell(log1p(4000 * raw_gene_UMI / official_full_library_total_counts))', 'unknown_token_policy': 'fail_closed'}})
    old_genes = ['G1', 'G3'] + [f'OLD{i}' for i in range(578)]
    contexts = ['K562', 'RPE1', 'hepg2', 'jurkat']
    old = pd.DataFrame({'experiment_id': [f'OLD_RECORD{i}' for i in range(2008)],
        'perturbation_target': [old_genes[i % 580] for i in range(2008)],
        'perturbation_type': ['genetic_single_gene'] * 2008, 'effect_vector_row': np.arange(2008),
        'n_cells': [30] * 2008, 'n_batches': [1] * 2008, 'eligibility': [True] * 2008,
        'context': [contexts[i // 580] for i in range(2008)]})
    old['condition'] = old.perturbation_target + '+ctrl'
    old.to_parquet(source / 'SOURCE_PUBLIC_MEMORY_METADATA.parquet', index=False)
    np.save(source / 'SOURCE_PUBLIC_EFFECTS.npy', np.full((2008, 3285), .1))
    np.save(source / 'SOURCE_PUBLIC_CONTROLS.npy', np.full((2008, 3285), .2))
    immutable(source / 'GENE_IDS.json', axis_names)
    immutable(source / 'SOURCE_ERROR_CDF.json', {'scope': 'GENERATED_HISTORICAL_SOURCE_ONLY'})
    immutable(source / 'SOURCE_FEATURE_MANIFEST.json', {'risk_P_columns': b.P, 'risk_public_columns': b.PUBLIC, 'public_pair_columns': b.PAIR})
    model_rows = []
    for ref, kind in b.METHODS:
        name = f'SOURCE_RISK_{ref}_{kind}.joblib'; columns = b.P if ref == 'P_only' else b.P + b.PUBLIC
        joblib.dump(GeneratedPredictor(columns), source / name)
        model_rows.append({'role': 'Source_risk', 'reference': ref, 'learner': kind, 'path': name,
            'sha256': e.sha(source / name), 'columns': columns, 'target': 'Source_per_upstream_per_context_midrank_CDF',
            'numeric_model_width_with_missing_flags': 2 * len(columns)})
    name = 'SOURCE_PUBLIC_BIOLOGY_FULL_HGB.joblib'; joblib.dump(GeneratedPredictor(b.PAIR), source / name)
    model_rows.append({'role': 'Public_biology_retrieval', 'path': name, 'sha256': e.sha(source / name),
        'columns': b.PAIR, 'target': 'biological_transfer_rmse', 'predict_clip': False, 'numeric_model_width_with_missing_flags': 18})
    immutable(source / 'MODEL_MANIFEST.json', {'models': model_rows})
    pd.DataFrame({'gene': old_genes[:575]}).to_parquet(source / 'SOURCE_TASKS.parquet', index=False)
    registry(source)
    for p in source.iterdir(): p.chmod(0o444)
    core = b.source_core(source)
    parent = {key: e.binding(source / name) for key, name in {
        'metadata': 'SOURCE_PUBLIC_MEMORY_METADATA.parquet', 'effects': 'SOURCE_PUBLIC_EFFECTS.npy',
        'controls': 'SOURCE_PUBLIC_CONTROLS.npy', 'axis': 'GENE_MANIFEST.csv', 'gene_ids': 'GENE_IDS.json'}.items()}
    asset = root / 'GENERATED_HISTORICAL_SOURCE.bin'; asset.write_bytes(b'GENERATED_SOURCE_ONLY_NO_ACTUAL_X'); asset.chmod(0o444)
    read = immutable(root / 'SOURCE_READ_CONTRACT.json', {'schema': 'safeconf_source_public_historical_read_contract_v1',
        'source': e.binding(asset), 'builder': e.binding(Path(__file__)), 'no_Orion_expression_or_truth': True,
        'no_model_fits': True, 'source_training_tasks_binding': e.binding(source / 'SOURCE_TASKS.parquet')})
    contract = immutable(root / 'EXTENSION_CONTRACT.json', {'schema': e.CONTRACT_SCHEMA,
        'base_scientific_contract': science, 'builder_code_binding': e.binding(Path(__file__)), 'source_read_contract_binding': read,
        'new_parameter_fits': 0, 'append_only_absent_original_eligible_genes': True, 'legacy_query_invariance_required': True,
        'TEST_truth_used_for_bank_selection': False, 'SafeConf_scores_used_for_bank_selection': False,
        'minimum_primary_gene_clusters_union': 100, 'minimum_context_tasks': 20, 'immutable_dependencies': []})
    added = pd.DataFrame({'experiment_id': [f'NEW_RECORD{i}' for i in range(100)], 'perturbation_target': new_genes,
        'perturbation_type': ['genetic_single_gene'] * 100, 'effect_vector_row': np.arange(2008, 2108),
        'n_cells': [30] * 100, 'n_batches': [1] * 100, 'eligibility': [True] * 100,
        'context': ['K562'] * 100, 'condition': [x + '+ctrl' for x in new_genes]})
    memory = pd.concat([old, added], ignore_index=True); memory.to_parquet(bank / 'SOURCE_PUBLIC_MEMORY_METADATA.parquet', index=False)
    np.save(bank / 'SOURCE_PUBLIC_EFFECTS.npy', np.concatenate([core['effects'], np.full((100, 3285), .15)]))
    np.save(bank / 'SOURCE_PUBLIC_CONTROLS.npy', np.concatenate([core['controls'], np.full((100, 3285), .25)]))
    axis.to_csv(bank / 'GENE_MANIFEST.csv', index=False); immutable(bank / 'GENE_IDS.json', axis_names)
    immutable(bank / 'ELIGIBLE_GENE_SET.json', {'old_eligible_gene_symbols': sorted(old_genes),
        'new_eligible_gene_symbols': sorted(new_genes), 'eligible_gene_symbols': sorted(old_genes + new_genes)})
    pd.DataFrame({'experiment_id': old.experiment_id, 'context': old.context, 'condition': old.condition,
        'n_cells_match': True, 'n_batches_match': True, 'max_abs_effect_gap': 0., 'max_abs_control_gap': 0.}).to_csv(bank / 'LEGACY_COMPATIBILITY_DIAGNOSTIC.csv', index=False)
    pd.DataFrame({'experiment_id': memory.experiment_id, 'physical_unit_id': memory.experiment_id,
                  'source_sha256': e.sha(asset)}).to_csv(bank / 'CANONICAL_UNIT_PROVENANCE.csv', index=False)
    immutable(bank / 'SOURCE_HISTORICAL_READ_AUDIT.json', {'all_exact_selected_Source_rows_completed': True,
        'contract_sha256': read['sha256'], 'builder_code_sha256': e.sha(__file__), 'source_sha256': e.sha(asset),
        'excluded_context_rows_decoded': 0, 'Orion_expression_truth_values_read': 0, 'normalization_altered': False, 'new_fits': 0})
    immutable(bank / 'METADATA_AUDIT_COUNTS.json', {'Source575_risk_training_gene_history_unchanged': True,
        'physical_units_unique': True, 'new_fits': 0, 'Orion_expression_truth_read': False})
    identities, objects, fit_rows, fit_registries, all_queries = [], [], [], {}, []
    target_map = dict(zip(full.gene_name, full.ensembl_id))
    for context in e.CONTEXTS:
        folder = fit / context; folder.mkdir()
        queries = pd.DataFrame([[f'SyntheticStudy|{gene}|{context}', target_map[gene], gene, context, role]
            for gene, role in [(x, 'TEST') for x in ['G1'] + new_genes] + [('G3', 'VALIDATION')]], columns=b.QUERY)
        all_queries.append(queries); table(folder / 'QUERY_IDENTITIES.tsv', queries)
        table(folder / 'OUTPUT_GENE_AXIS.tsv', axis.rename(columns={'orion_ensembl_id': 'gene_id', 'gene_name': 'gene_symbol'})[['gene_id', 'gene_symbol']])
        pd.DataFrame({'gene_id': axis.orion_ensembl_id, **{qid: np.full(3285, .5) for qid in queries.query_id}}).to_csv(folder / 'PREDICTIONS_DELTA.tsv.gz', sep='\t', index=False, compression='gzip')
        table(folder / 'OWN_CONTEXT_NTC_MEAN.tsv', pd.DataFrame({'gene_id': axis.orion_ensembl_id, 'mean_cell_logCP4000': np.full(3285, .2)}))
        (folder / 'MODEL.rds').write_bytes(b'GENERATED_IMMUTABLE_MODEL_' + context.encode()); (folder / 'SIMPLE_BASELINES.rds').write_bytes(b'GENERATED_ZERO_BASELINE')
        table(folder / 'BASELINE_SELECTION.tsv', pd.DataFrame({'selected': ['zero_effect']}))
        status = {'status': 'PASS', 'mode': 'fit', 'context_id': context, 'pca_dim': '10', 'ridge_penalty': '0.1', 'seed': '1',
            'query_truth_read': 'FALSE', 'input_estimand': 'mean_cell_log1p_cp4000_v1', 'endpoint_gene_count': '3285', 'query_count': str(len(queries))}
        table(folder / 'STATUS.tsv', pd.DataFrame({'key': list(status), 'value': list(status.values())}))
        fit_rows.append({'context_id': context, 'model_sha256': e.sha(folder / 'MODEL.rds')}); fit_registries[context] = registry(folder, True)
        units = [qid for qid in queries.query_id.iloc[:-1] for _ in range(30)] + [queries.query_id.iloc[-1]] * 3
        genes = [gene for gene in queries.target_gene_symbol.iloc[:-1] for _ in range(30)] + ['G3'] * 3
        roles = ['TEST'] * (len(units) - 3) + ['TRAIN', 'VALIDATION', 'DEV_EXPOSED_CELL']
        rows = len(units); private_token = 918273645; private_value = 918273645.125
        ids = [[0, 1]] * (rows - 3) + [[private_token]] * 3; values = [[1., 3.]] * (rows - 3) + [[private_value]] * 3
        path = raw / f'{context}_Batch1.parquet'
        pq.write_table(pa.table({'gene_token_id': pa.array(ids, type=pa.list_(pa.int64())), 'gene_expression': pa.array(values, type=pa.list_(pa.float64()))}), path, compression='snappy', use_dictionary=True, data_page_version='2.0', row_group_size=rows)
        metadata = md / f'{context}_rows.parquet'
        pq.write_table(pa.table({'original_row_index': np.arange(rows), 'original_row_group': [0] * rows, 'row_role': roles,
            'gene_target': genes, 'sample': [context + '_Batch1'] * rows, 'cell_barcode': [f'{context}:{i}' for i in range(rows)],
            'total_counts': [4.] * (rows - 3) + [1.] * 3, 'biological_unit': units, 'fixed_metadata_QC_pass': [True] * rows}), metadata)
        metadata.chmod(0o444)
        identities.append({'path': path.name, 'context': context, 'sample': context + '_Batch1', 'rows': rows, 'row_groups': 1,
            'publisher_whole_file_bytes': path.stat().st_size, 'publisher_LFS_sha256': e.sha(path), 'metadata_parquet_path': str(metadata), 'metadata_parquet_sha256': e.sha(metadata)})
        objects.append({'path': path.name, 'bytes': path.stat().st_size, 'lfs_sha256': e.sha(path)})
        for p in folder.iterdir(): p.chmod(0o444)
    table(fit / 'RUN_MANIFEST.tsv', pd.DataFrame(fit_rows))
    immutable(md / 'FILE_METADATA_IDENTITY.json', identities)
    object_binding = immutable(root / 'OBJECTS.json', {'release_sha': 'SYNTHETIC', 'files': objects})
    queries = pd.concat(all_queries, ignore_index=True)
    table(bank / 'QUALIFIED_IDS.tsv', queries.loc[queries.role.eq('TEST'), ['query_id', 'context_id']])
    immutable(bank / 'METADATA_COVERAGE_CERTIFICATE.json', {'schema': 'safeconf_public_bank_metadata_coverage_certificate_v1',
        'TEST_truth_read': False, 'risk_scores_used': False, 'qualified_query_ids': e.binding(bank / 'QUALIFIED_IDS.tsv')})
    for p in bank.iterdir(): p.chmod(0o444)
    extended = {key: e.binding(bank / Path(item['path']).name) for key, item in parent.items()}
    manifest = {'schema': e.BANK_SCHEMA, 'status': 'COMPLETE', 'extension_contract': contract, 'parent_bank': parent,
        'base_source_core_bindings': core['bindings'] + [e.binding(source / 'ARTIFACT_HASHES.json')], 'extended_bank': extended,
        'builder_code_binding': e.binding(__file__), 'source_read_contract_binding': read, 'source_asset_binding': e.binding(asset),
        'source_training_tasks_binding': e.binding(source / 'SOURCE_TASKS.parquet'), 'new_fits': 0, 'target_orion_truth_expression_used': False,
        'append_policy': {'target_absent_old_eligible_gene_set': True, 'minimum_historical_cells': 30, 'allowed_contexts': contexts},
        'legacy_prefix': {'n_items': 2008, 'metadata_exact': True, 'effects_bitwise_equal': True, 'controls_bitwise_equal': True,
            'old_eligible_gene_count': 580, 'no_old_eligible_target_appended': True}}
    for key, name in {'eligible_gene_set': 'ELIGIBLE_GENE_SET.json', 'compatibility_diagnostic': 'LEGACY_COMPATIBILITY_DIAGNOSTIC.csv',
        'canonical_unit_provenance': 'CANONICAL_UNIT_PROVENANCE.csv', 'read_audit': 'SOURCE_HISTORICAL_READ_AUDIT.json',
        'metadata_audit_counts': 'METADATA_AUDIT_COUNTS.json', 'coverage_certificate': 'METADATA_COVERAGE_CERTIFICATE.json'}.items(): manifest[key] = e.binding(bank / name)
    bank_binding = immutable(bank / 'PUBLIC_BANK_MANIFEST.json', manifest)
    role = root / 'ROLES.csv'; pd.DataFrame([{'context': ctx, 'task_range': 'all gene roles TEST', 'role': 'SEEN',
        'metadata_seen': True, 'result_seen': False, 'test_truth_access_status': 'CLOSED_UNTOUCHED_LABELS', 'pristine_confirmation_claim': False} for ctx in e.CONTEXTS]).to_csv(role, index=False); role.chmod(0o444)
    comparison_manifest = immutable(root / 'BASE_PRETRUTH_INPUTS.json', {'schema': b.SCHEMA, 'risk_code_sha256': e.sha(b.__file__),
        'scientific_contract_path': science['path'], 'scientific_contract_sha256': science['sha256'], 'pretruth_inference_authorized': True,
        'target_errors_or_CDF_used': False, 'source_core_root': str(source), 'source_core_bindings': core['bindings'],
        'data_role_registry_path': str(role), 'data_role_registry_sha256': e.sha(role),
        'contexts': [b.context_bindings(fit, fit, ctx, fit) for ctx in e.CONTEXTS]})
    registration = immutable(root / 'SEAL_REGISTRATION.json', {'schema': 'safeconf_orion_public_bank_extension_seal_registration_v1',
        'implementation_bindings': {'extension_sealer': e.binding(s.__file__), 'extension_helper': e.binding(e.__file__)},
        'extension_contract': contract, 'bank_manifest': bank_binding, 'base_comparison_manifest': comparison_manifest})
    competence = immutable(root / 'COMPETENCE.json', {'status': 'PASS', 'passes_competence': True, 'actual_context_count': 2,
        'noninferior_contexts': 2, 'noninferior_strata_fraction': 1., 'stable_disadvantage': False, 'relative_macro_error_gap': 0.,
        'model_macro_rmse': 1., 'baseline_macro_rmse': 1., 'lower95_relative_gap': 0., 'upper95_relative_gap': 0.,
        'eligible_validation_tasks': 2, 'evaluated_validation_tasks': 2, 'missing_predictions': 0, 'nonfinite_predictions': 0,
        'test_truth_read': False, 'scientific_contract_sha256': science['sha256'], 'bootstrap_replicates': 5000, 'bootstrap_seed': 20260929,
        'new_model_parameters_fitted': False, 'synthetic_only': True})
    return locals()


def run():
    before = c.implementation_bindings(); started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='extension_generated_', dir=c.reader.DOC / 'synthetic') as temp:
        root = Path(temp); f = make_fixture(root)
        # These fixture constants never alter production file bytes or functions.
        with patch.object(e, 'BASE_METHOD_SHA', f['science']['sha256']), patch.object(b, 'METHOD_SHA', f['science']['sha256']), \
             patch.object(b.source_core, '__defaults__', (f['source'],)), patch.object(c.coordinator, 'BASE', root), \
             patch.object(c.evaluator, 'REPLICATES', 4):
            validated = e.load_bank(f['bank_binding'], f['contract'], f['core'])
            assert len(validated[0]['memory']) == 2108 and len(validated[3]) == 580
            seal = root / 'expanded_seal'; s.seal(f['registration']['path'], seal)
            compared = root / 'expanded_comparison'; c.prepare_comparison(seal, compared)
            serialization = json.loads((compared / 'EXTENSION_COMPARISON_RECEIPT.json').read_text())['serialization_reconciliation']
            assert serialization['all_existing_numeric_candidates_exact_after_parquet_reload'] is True
            assert any(serialization['changed_last_bit_values_by_candidate'].values())
            comparison_receipt = e.binding(compared / 'EXTENSION_COMPARISON_RECEIPT.json')
            independent = immutable(root / 'GENERATED_INDEPENDENT_REVIEW.json', {'status': 'PASS', 'bank_manifest': f['bank_binding']})
            review = immutable(root / 'ROOT_REVIEW.json', {'schema': 'safeconf_orion_public_bank_extension_final_review_v1',
                'status': 'APPROVED_FOR_ONCE_ONLY_REGISTERED_TEST', 'root_approved': True, 'independent_review_passed': True,
                'bank_manifest': f['bank_binding'], 'extension_contract': f['contract'], 'comparison_receipt': comparison_receipt,
                'implementation_bindings': c.implementation_bindings(), 'independent_review_receipt': independent})
            reader = c.reader
            operation = {'schema': c.OUTER_OPERATION_SCHEMA, 'postseal_TEST_operation_authorized': True,
                'test_numeric_materialization_permitted': True, 'authorized_numeric_roles': ['TEST'], 'authorized_contexts': list(e.CONTEXTS),
                'TEST_metadata_before_open': 'SEEN', 'TEST_truth_before_open': 'CLOSED', 'synthetic_only': True,
                'scientific_contract': f['science'], 'forty_object_manifest': f['object_binding'], 'metadata_root': str(f['md']), 'raw_root': str(f['raw']),
                'metadata_bindings': {name: e.sha(f['md'] / name) for name in reader.META_NAMES}, 'data_role_registry': e.binding(f['role']),
                'final_competence': f['competence'], 'lm_fit_root': str(f['fit']), 'lm_fit_run_manifest': e.binding(f['fit'] / 'RUN_MANIFEST.tsv'),
                'lm_fit_artifact_registries': f['fit_registries'], 'risk_seal_root': str(seal), 'risk_seal_manifest': e.binding(seal / 'RISK_SEAL_MANIFEST.json'),
                'risk_seal_artifacts': e.binding(seal / 'ARTIFACT_HASHES.json'), 'risk_comparison_manifest': f['comparison_manifest'],
                'pretruth_comparison': {'spec': e.binding(compared / 'COMPARISON_SPEC.json'), 'scores': e.binding(compared / 'PRETRUTH_COMPARISON_SCORES.parquet')},
                'implementation_bindings': {name: e.binding(path) for name, path in {'test_reader': reader.__file__,
                    'numeric_backend': reader.reader.__file__, 'structural_reader': reader.reader.v1.__file__,
                    'scientific_loader': reader.biology.__file__, 'risk_sealer': b.__file__}.items()},
                'test_semantic_QC': reader.SEMANTIC_QC, 'task_min_cells': 30, 'normalization_scale': 4000, 'output_path': str(root / 'truth'),
                'public_bank_extension': {'schema': 'safeconf_orion_public_bank_extension_operation_v1', 'extension_contract': f['contract'],
                    'bank_manifest': f['bank_binding'], 'comparison_receipt': comparison_receipt,
                    'implementation_bindings': c.implementation_bindings(), 'root_and_independent_review': review}}
            op = immutable(root / 'OUTER_OPERATION.json', operation)
            try: reader.authorize_test(op['path'])
            except RuntimeError: pass
            else: raise AssertionError('Old CLI accepted new outer schema')
            for label, mutate in [('wrong_bank', lambda x: x['public_bank_extension']['bank_manifest'].__setitem__('sha256', '0' * 64)),
                ('missing_review', lambda x: x['public_bank_extension'].pop('root_and_independent_review')),
                ('wrong_implementation', lambda x: x['public_bank_extension']['implementation_bindings']['extension_helper'].__setitem__('sha256', '0' * 64))]:
                altered = json.loads(json.dumps(operation)); mutate(altered); item = immutable(root / (label + '.json'), altered)
                try: c.authorize_extended_test(item['path'])
                except (RuntimeError, KeyError): pass
                else: raise AssertionError('Bad extension gate accepted: ' + label)
                assert not list(root.glob('DELEGATED_BASE_OPERATION.' + item['sha256'] + '*'))
            authorized = c.authorize_extended_test(op['path'])
            assert authorized['public_bank_extension_verified'] and not (root / 'truth').exists()
            assert not list(root.glob('DELEGATED_BASE_OPERATION.*')) and not list(root.glob('.BASE_GUARD_VALIDATION.*'))
            try: c.evaluate_extended_test(op['path'], root / 'wrong_output')
            except RuntimeError: pass
            else: raise AssertionError('Wrong CLI output claimed once-only scope')
            assert not (root / 'ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json').exists()
            truth = c.evaluate_extended_test(op['path'], root / 'truth')
            assert truth['TEST_numeric_cells_opened'] == 6060
            errors = pd.read_parquet(root / 'truth' / 'TEST_TASK_ERRORS.parquet')
            assert len(errors) == 202 and errors.n_cells.eq(30).all()
            wrong_claim = json.loads((root / 'truth' / 'EXTENSION_TRUTH_RECEIPT.json').read_text())
            wrong_claim['exclusive_registered_scope']['sha256'] = '0' * 64
            wrong_claim_binding = immutable(root / 'WRONG_CLAIM_TRUTH_RECEIPT.json', wrong_claim)
            try: c.evaluate_extended_risk(compared, wrong_claim_binding['path'], root / 'bad_evaluation')
            except RuntimeError: pass
            else: raise AssertionError('Wrong once-only truth claim accepted')
            assert not (root / 'bad_evaluation').exists()
            generic_path = root / 'GENERIC_COMPLETE.json'
            generic = immutable(generic_path, {'status': 'COMPLETE',
                'TEST_access_receipt_path': str(root / ('DELEGATED_BASE_OPERATION.' + op['sha256'] + '.json')),
                'TEST_access_receipt_sha256': e.sha(root / ('DELEGATED_BASE_OPERATION.' + op['sha256'] + '.json'))})
            wrong_base = json.loads((root / 'truth' / 'EXTENSION_TRUTH_RECEIPT.json').read_text())
            wrong_base['base_truth_receipt'] = generic
            wrong_base_binding = immutable(root / 'GENERIC_BASE_TRUTH_SIDE_RECEIPT.json', wrong_base)
            checked_paths = []; checked = e.checked
            def traced_check(item, immutable=True):
                checked_paths.append(item.get('path')); return checked(item, immutable)
            with patch.object(e, 'checked', traced_check):
                try: c.evaluate_extended_risk(compared, wrong_base_binding['path'], root / 'generic_bad_evaluation')
                except RuntimeError: pass
                else: raise AssertionError('Generic base COMPLETE accepted')
            assert str(root / 'truth' / 'TEST_TASK_ERRORS.parquet') not in checked_paths
            result = c.evaluate_extended_risk(compared, root / 'truth' / 'EXTENSION_TRUTH_RECEIPT.json', root / 'evaluation')
            assert result['status'] == 'COMPLETE_FIXED_COMPARISON' and result['candidate_selection_after_truth'] is False
            comparison = pd.read_parquet(compared / 'PRETRUTH_COMPARISON_SCORES.parquet')
            assert comparison.loc[comparison.role.eq('TEST'), 'PRETRUTH_PRIMARY_COMMON_ELIGIBLE'].all()
            assert len(json.loads((compared / 'COMPARISON_SPEC.json').read_text())['methods']) == 13
            assert (seal / 'ORIGINAL2008_FIXED_PRETRUTH_RISKS.tsv').exists()
            archived = pd.read_csv(seal / 'ORIGINAL2008_FIXED_PRETRUTH_RISKS.tsv', sep='\t')
            assert archived.loc[archived.target_gene_symbol.isin(f['new_genes']), 'Manual_hgb'].isna().all()
            try: c.coordinator.reserve_registered_scope(op['path'])
            except FileExistsError: pass
            else: raise AssertionError('Duplicate once-only scope accepted')
        assert c.implementation_bindings() == before
    proof = {'status': 'GENERATED_COMPLETE_EXTENSION_CHAIN_PASS', 'elapsed_seconds': time.monotonic() - started,
        'implementation_bindings': before, 'generated_bank_original_rows': 2008, 'generated_original_gene_count': 580,
        'generated_Source_training_gene_count': 575, 'generated_endpoint_gene_count': 3285, 'generated_query_count': 204,
        'generated_TEST_tasks': 202, 'generated_TEST_cells_read': 6060, 'registered_candidates': 13,
        'checks': ['real_helper_full_prefix_axis_sets_provenance_audit_validation', 'original_math_and_7_generated_fixed_predictors_no_fit',
            'all_legacy_QUERY_invariance', 'coverage_gene_union_101_two_contexts_101_each', 'old_reader_outer_schema_rejection',
            '3_faulty_extension_chains_rejected_before_base_delegation', 'root_review_then_exact_base_schema_delegation',
            'authorize_removes_temporary_base_receipt_no_persistent_delegation', 'wrong_output_rejected_before_once_claim',
            'wrong_once_claim_truth_receipt_rejected_before_error_evaluation',
            'generic_base_COMPLETE_rejected_before_error_hash_access',
            'unchanged_private_reader_skips_TRAIN_VAL_DEV_sentinels', 'complete_wrapper_truth_to_original_evaluator',
            'expanded_primary_keeps13_methods_original2008_NaNs_only_separate_archive', 'once_only_scope_second_registration_rejected',
            'all_production_code_bytes_unchanged'],
        'fixture_only_constant_substitutions': ['synthetic_scientific_SHA', 'generated_Source_core_default_root', 'isolated_once_only_claim_root', '4_bootstrap_draws'],
        'serialization_changed_value_counts': serialization['changed_last_bit_values_by_candidate'],
        'serialization_maximum_default_parser_gaps': serialization['maximum_default_parser_gap_by_candidate'],
        'production_bootstrap_draws_remain': c.evaluator.REPLICATES, 'new_fits': 0, 'actual_Source_expression_read': False,
        'actual_Orion_prediction_or_expression_read': False, 'actual_TEST_access': False, 'temporary_generated_arrays_removed': True}
    path = Path(__file__).parent / 'GENERATED_END_TO_END_PROOF_FINAL.json'; immutable(path, proof)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__': run()
