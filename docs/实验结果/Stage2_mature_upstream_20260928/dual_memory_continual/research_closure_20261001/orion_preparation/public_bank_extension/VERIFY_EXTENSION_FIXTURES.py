"""Generated extension fixtures only; no actual expression/prediction input."""
from pathlib import Path
import copy
import json
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
sys.path.insert(0, str(REPO))
from tools.safeconf_continual import orion_public_bank_extension as ext
from tools.scripts import seal_safeconf_orion_source_risk_agent as base
from tools.scripts import run_safeconf_orion_public_bank_extension_chain_agent as chain


class FixedPredictor:
    def __init__(self, columns): self.columns = columns
    def predict(self, frame, clip=True):
        values = frame[self.columns].to_numpy(float)
        out = .2 + .001 * np.sum(values, axis=1)
        return np.clip(out, 0, 1) if clip else out


def immutable(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n'); path.chmod(0o444)
    return ext.binding(path)


def run():
    checks = []
    with tempfile.TemporaryDirectory(prefix='safeconf_extension_generated_') as temp:
        root = Path(temp)
        n = 3285
        core = {'axis': pd.DataFrame({'gene_name': [f'MEASURED{i}' for i in range(n)]}),
                'memory': pd.DataFrame({'experiment_id': ['OLD_A', 'OLD_B'], 'perturbation_target': ['OLD', 'OLD'],
                    'perturbation_type': ['genetic_single_gene'] * 2, 'effect_vector_row': [0, 1],
                    'n_cells': [30, 60], 'n_batches': [1, 2], 'eligibility': [True, True]}),
                'effects': np.asarray([np.ones(n), np.full(n, 2.)]),
                'controls': np.asarray([np.full(n, .1), np.full(n, .2)]),
                'models': {key: FixedPredictor(base.P if key[0] == 'P_only' else base.P + base.PUBLIC)
                           for key in base.METHODS}, 'public': FixedPredictor(base.PAIR)}
        new = dict(core)
        added = pd.DataFrame({'experiment_id': ['NEW_A'], 'perturbation_target': ['NEW'],
                             'perturbation_type': ['genetic_single_gene'], 'effect_vector_row': [2],
                             'n_cells': [40], 'n_batches': [1], 'eligibility': [True]})
        new['memory'] = pd.concat([core['memory'], added], ignore_index=True)
        new['effects'] = np.vstack([core['effects'], np.full(n, 3.)])
        new['controls'] = np.vstack([core['controls'], np.full(n, .3)])
        queries = pd.DataFrame([['OLD|HCT', 'E_OLD', 'OLD', 'HCT116', 'TEST'],
                                ['NEW|HEK', 'E_NEW', 'NEW', 'HEK293T', 'TEST'],
                                ['OLD|VAL', 'E_OLD', 'OLD', 'HEK293T', 'VALIDATION']], columns=base.QUERY)
        delta = np.full((3, n), .5)
        controls = {'HCT116': np.full(n, .15), 'HEK293T': np.full(n, .25)}
        original, expanded = base.infer(queries, delta, controls, core), base.infer(queries, delta, controls, new)
        invariant = ext.assert_legacy_invariance(queries, original, expanded, {'OLD'})
        assert invariant['legacy_query_rows'] == 2
        checks.append('all_legacy_VAL_TEST_scores_priors_features_pairs_and_weights_invariant')
        assert np.isnan(original['scores'].loc[1, 'Manual_hgb']) and np.isfinite(expanded['scores'].loc[1, 'Manual_hgb'])
        checks.append('new_gene_history_becomes_supported_without_a_fit')
        for what in ['score', 'prior', 'pair', 'weight']:
            altered = copy.deepcopy(expanded)
            if what == 'score': altered['scores'].loc[0, 'Manual_hgb'] += .001
            elif what == 'prior': altered['priors']['Manual'][0, 0] += .001
            elif what == 'pair': altered['pairs'].loc[0, 'log_source_cells'] += .001
            else: altered['weights'].loc[0, 'weight'] += .001
            try: ext.assert_legacy_invariance(queries, original, altered, {'OLD'})
            except RuntimeError: checks.append('legacy_' + what + '_change_rejected')
            else: raise AssertionError(what)
        coverage_queries = pd.DataFrame([[f'Q{i}|{ctx}', f'E{i}', f'GENE{i}', ctx, 'TEST']
            for ctx in ext.CONTEXTS for i in range(100)], columns=base.QUERY)
        eligible = coverage_queries[['query_id', 'context_id']]
        ids = root / 'QUALIFIED_IDS.tsv'; eligible.to_csv(ids, sep='\t', index=False); ids.chmod(0o444)
        certificate = immutable(root / 'COVERAGE.json', {'schema': 'safeconf_public_bank_metadata_coverage_certificate_v1',
            'TEST_truth_read': False, 'risk_scores_used': False, 'qualified_query_ids': ext.binding(ids)})
        methods = ['Magnitude'] + [f'{ref}_{kind}' for ref, kind in base.METHODS] + [
            'Manual_WeightedHistoryDistance', 'Learned_WeightedHistoryDistance', 'Uniform_DirectRMSE', 'Manual_DirectRMSE', 'Learned_DirectRMSE']
        scores = coverage_queries.copy(); scores['source_history_n'] = 1
        for method in methods + ['NegativeSourceHistorySupport']: scores[method] = .2
        good = ext.checked_coverage(certificate, coverage_queries, scores)
        assert good['primary_gene_clusters_union'] == 100
        checks.append('coverage_counts_independent_gene_union_not_double_context_rows')
        for bad in ['n99', 'nonfinite', 'nonfinite_support', 'missinghistory', 'context19']:
            altered = scores.copy()
            if bad == 'n99': altered.loc[altered.target_gene_id.eq('E99'), 'source_history_n'] = 0
            elif bad == 'nonfinite': altered.loc[altered.target_gene_id.eq('E99'), 'Manual_hgb'] = np.nan
            elif bad == 'nonfinite_support': altered.loc[altered.target_gene_id.eq('E99'), 'NegativeSourceHistorySupport'] = np.nan
            elif bad == 'missinghistory': altered['source_history_n'] = 0
            else: altered.loc[altered.context_id.eq('HEK293T') & ~altered.target_gene_id.isin([f'E{i}' for i in range(19)]), 'source_history_n'] = 0
            try: ext.checked_coverage(certificate, coverage_queries, altered)
            except RuntimeError: checks.append('coverage_' + bad + '_fails_closed')
            else: raise AssertionError(bad)
        # Old reader must reject the outer extension operation before raw guards.
        operation = immutable(root / 'OUTER.json', {'schema': chain.OUTER_OPERATION_SCHEMA})
        try: chain.reader.authorize_test(operation['path'])
        except RuntimeError: checks.append('unchanged_old_reader_rejects_outer_operation_schema')
        else: raise AssertionError('old CLI accepted outer extension operation')
        try: chain.authorize_extended_test(operation['path'])
        except (RuntimeError, KeyError): checks.append('incomplete_extension_chain_rejected_before_base_reader')
        else: raise AssertionError('incomplete extension operation accepted')
        bad_bank = immutable(root / 'INCOMPLETE_BANK.json', {'schema': ext.BANK_SCHEMA, 'status': 'RUNNING'})
        source_contract = REPO / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/historical_bank_v1/registered_v2/PUBLIC_BANK_EXPANSION_CONTRACT.json'
        try: ext.load_bank(bad_bank, ext.binding(source_contract), {})
        except RuntimeError: checks.append('bank_RUNNING_cannot_enter_inference_or_test_chain')
        else: raise AssertionError('incomplete bank accepted')
    proof = {'status': 'GENERATED_EXTENSION_FIXTURES_PASS', 'checks': checks, 'check_count': len(checks),
             'helper_sha256': ext.sha(ext.__file__), 'chain_sha256': ext.sha(chain.__file__),
             'sealer_sha256': ext.sha(chain.sealer.__file__), 'original_risk_math_sha256': ext.sha(base.__file__),
             'new_fits': 0, 'actual_Source_expression_read': False, 'actual_Orion_prediction_or_expression_read': False,
             'actual_TEST_access': False, 'fixtures_temp_only': True}
    path = Path(__file__).parent / 'GENERATED_FIXTURE_PROOF_FINAL.json'
    if path.exists(): raise FileExistsError('Preserve prior proof; use a new proof version')
    immutable(path, proof)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__': run()
