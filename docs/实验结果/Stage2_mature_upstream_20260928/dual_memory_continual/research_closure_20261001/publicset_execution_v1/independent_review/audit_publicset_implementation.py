#!/usr/bin/env python3
"""Low-cost independent PublicSet checks; no model training or MC TEST reads."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
DOC = Path(__file__).resolve().parent
RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/publicset_execution_v1/independent_review')


def main():
    start = time.monotonic()
    script = ROOT / 'tools/scripts/run_safeconf_publicset_v1.py'
    spec = importlib.util.spec_from_file_location('publicset_review_target', script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    actual = pd.read_csv(RUNTIME / 'SCOPED_QUERY_HISTORY_EDGES.csv')
    expected_pca = pd.read_csv(RUNTIME / 'PREPROCESSOR_FIT_UNIQUE_HISTORY_IDS.csv')
    metadata_results = []
    for name in ['Source', 'McFaline']:
        d = module.load_domain(name)
        for outer in range(5):
            fit_rows = np.flatnonzero(d.tasks.fold.to_numpy() != outer)
            query_rows = np.flatnonzero(d.tasks.fold.to_numpy() == outer)
            forbidden = d.forbidden(query_rows)
            fit = d.groups(fit_rows, forbidden)
            query = d.groups(query_rows, forbidden)
            used = np.unique(np.concatenate([g['ix'] for g in fit if len(g['ix'])]))
            actual_used_ids = set(d.memory.iloc[used].experiment_id)
            contexts = sorted(d.tasks.context.unique()) if name == 'Source' else [None]
            for context in contexts:
                scope = f'Source/outer{outer}/eval_{context}' if name == 'Source' else f'McFaline/outer{outer}'
                sub = [g for g in query if context is None or d.tasks.iloc[g['q']].context == context]
                pairs = {(str(d.tasks.iloc[g['q']].task_id), str(d.memory.iloc[i].experiment_id)) for g in sub for i in g['ix']}
                reference = actual.loc[actual.scope.eq(scope) & actual.phase.eq('predict')]
                reference_pairs = set(zip(reference.task_id, reference.history_experiment_id))
                assert pairs == reference_pairs, (scope, 'query edges differ')
                expected_ids = set(expected_pca.loc[expected_pca.scope.eq(scope), 'experiment_id'])
                assert actual_used_ids == expected_ids, (scope, 'PCA candidate IDs differ')
                assert not set(d.memory.iloc[used].perturbation_target) & set(d.tasks.iloc[query_rows].gene)
                metadata_results.append({'scope': scope, 'query_pairs': len(pairs), 'fit_history_items': len(used), 'independent_metadata_match': True})

            genes = sorted(d.tasks.iloc[fit_rows].gene.unique(), key=lambda g: hashlib.sha256(f'PublicSet-inner-v1|{name}|{outer}|{g}'.encode()).hexdigest())
            val_genes = set(genes[:max(1, int(np.ceil(.2 * len(genes))))])
            val_rows = np.asarray([i for i in fit_rows if d.tasks.iloc[i].gene in val_genes])
            inner_rows = np.asarray([i for i in fit_rows if d.tasks.iloc[i].gene not in val_genes])
            inner_forbidden = forbidden | d.forbidden(val_rows)
            inner_groups = d.groups(inner_rows, inner_forbidden)
            val_groups = d.groups(val_rows, inner_forbidden)
            inner_history = np.unique(np.concatenate([g['ix'] for g in inner_groups if len(g['ix'])]))
            assert not set(d.memory.iloc[inner_history].perturbation_target) & set(d.tasks.iloc[np.r_[query_rows, val_rows]].gene)
            assert not set(d.memory.iloc[inner_history].experiment_id) & inner_forbidden
            if name == 'McFaline':
                for g in val_groups:
                    assert not set(d.memory.iloc[g['ix']].experiment_id) & inner_forbidden
            if outer == 0:
                prep = module.fit_preprocessor(d, inner_groups)
                assert np.array_equal(prep['history_rows'], inner_history)
                assert prep['pca'].n_samples_ == len(inner_history)
                assert np.allclose(prep['pca'].mean_, d.effects[inner_history].mean(0), atol=1e-6)
                packed = module.pack(d, inner_groups, prep, 'cpu')
                weights = packed['weights'].numpy()
                totals = pd.DataFrame({'gene': d.tasks.iloc[packed['q']].gene.to_numpy(), 'w': weights}).groupby('gene').w.sum()
                assert np.allclose(totals, totals.iloc[0], atol=1e-6)
                torch.manual_seed(20261002); pointwise = module.SetScorer(packed['tokens'].shape[-1], False)
                torch.manual_seed(20261002); collective = module.SetScorer(packed['tokens'].shape[-1], True)
                assert all(torch.equal(v, collective.state_dict()[k]) for k, v in pointwise.state_dict().items())
                assert sum(p.numel() for p in pointwise.parameters()) < 100000

    # Independently test actual set context: change only third history, leaving
    # the first two token encodings/support fixed. The learned log-weight ratio
    # may change in DeepSets but must not change in the pointwise arm.
    torch.manual_seed(19)
    b2 = module.SetScorer(8, False)
    torch.manual_seed(19)
    b3 = module.SetScorer(8, True)
    x = torch.randn(2, 3, 8)
    x[1] = x[0]
    x[1, 2] += 8
    mask = torch.ones((2, 3), dtype=torch.bool)
    support = torch.ones((2, 3)) / 3
    with torch.no_grad():
        a2 = (b2(x, mask, support) - .5 * support) * 2
        a3 = (b3(x, mask, support) - .5 * support) * 2
    ratio2 = torch.log(a2[:, 0] / a2[:, 1])
    ratio3 = torch.log(a3[:, 0] / a3[:, 1])
    assert torch.allclose(ratio2[0], ratio2[1], atol=1e-6)
    assert abs(float(ratio3[0] - ratio3[1])) > 1e-6
    module.semantic_tests()
    pd.DataFrame(metadata_results).to_csv(DOC / 'IMPLEMENTATION_SCOPE_CHECKS.csv', index=False)
    summary = {
        'status': 'PASS_INPUT_SCOPE_AND_SYNTHETIC_SEMANTICS', 'code_sha256': script_sha,
        'script_path': str(script), 'checked_deployment_scopes': len(metadata_results),
        'mc_TEST_values_read': 0, 'model_training_fits': 0,
        'preprocessor_PCA_fits': 2, 'preprocessor_fits_are_audit_only_not_risk_or_biology_model_training': True,
        'early_validation_and_outer_query_gene_exclusion': True,
        'PCA_fit_only_unique_actual_training_history_rows': True,
        'equal_initial_parameters_B2_B3_same_seed': True,
        'gene_equal_training_weight_totals': True,
        'pointwise_logratio_change': float(ratio2[0] - ratio2[1]),
        'deepsets_logratio_change': float(ratio3[0] - ratio3[1]),
        'Source_empty_forbidden_helper_equivalent_only_under_existing_same_gene_other_context_eligibility': True,
        'elapsed_seconds': time.monotonic() - start,
        'remaining_runtime_checks': ['completed early/refit FIT_AUDIT and MODEL artifacts', 'reload reproduces actual predictions', 'four-arm same-query output matrices', 'GPU cost cap checked before each stage or inner-loop deadline'],
        'notes': ['No neural model was trained by this audit.', 'All input arrays loaded by runner are Source or McFaline DEV; no TEST files loaded.', 'PCA.transform currently computes unused codes for forbidden bank rows; these do not influence fitted parameters or selected training tokens. Prefer restricting transform to actual used fit/query rows in future cache cleanup.']}
    (DOC / 'IMPLEMENTATION_REVIEW.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
