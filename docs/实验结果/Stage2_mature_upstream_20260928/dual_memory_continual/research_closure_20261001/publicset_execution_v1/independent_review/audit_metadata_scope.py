#!/usr/bin/env python3
"""Metadata-only PublicSet scope audit; no expression arrays or TEST values."""
from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
DOC = Path(__file__).resolve().parent
COMMON = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
RUNTIME = COMMON.parent / 'publicset_execution_v1/independent_review'
SOURCE_BANK = Path('/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256('\n'.join(sorted(map(str, ids))).encode()).hexdigest()


def write_csv(name, value, public=False):
    path = (DOC if public else RUNTIME) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False)
    return {'path': str(path), 'sha256': sha(path), 'rows': len(value)}


def main():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    inputs = {
        'source_queries': COMMON / 'SOURCE_TASKS.csv',
        'source_bank_metadata': SOURCE_BANK / 'public_memory.parquet',
        'source_eligibility': SOURCE_BANK / 'eligibility.parquet',
        'source_pretruth_metadata': ROOT / 'docs/实验结果/E201_txpert_multitarget_retraining_20260802/tables/E201_PRETRUTH_TASK_BASE.csv',
        'mc_dev_queries': COMMON / 'VALIDATION_BIOLOGY_TASKS.csv',
        'mc_bank_metadata': COMMON / 'public_mcfaline_trainval/public_memory.parquet',
    }
    s = pd.read_csv(inputs['source_queries'])
    sm = pd.read_parquet(inputs['source_bank_metadata'])
    eligibility = pd.read_parquet(inputs['source_eligibility'])
    target_meta = pd.read_csv(inputs['source_pretruth_metadata'])
    assert s.task_id.is_unique and sm.experiment_id.is_unique
    s['own_experiment_id'] = 'E201::' + s.target + '::' + s.condition
    identity = s.merge(sm, left_on='own_experiment_id', right_on='experiment_id', validate='one_to_one', suffixes=('', '_memory'))
    assert len(identity) == len(s) == 1808
    identity = identity.merge(target_meta[['task_id', 'n_target_cells', 'n_target_batches']], on='task_id', validate='one_to_one')
    assert identity.n_target_cells.eq(identity.n_cells).all()
    assert identity.n_target_batches.eq(identity.n_batches).all()
    assert identity.gene.eq(identity.perturbation_target).all()
    assert identity.target.eq(identity.context).all()
    assert identity.condition.eq(identity.condition_memory).all()
    se = s.merge(eligibility, left_on=['target', 'condition'], right_on=['target_context', 'condition'], validate='one_to_many')
    se = se.rename(columns={'public_experiment_id': 'history_experiment_id'})
    assert len(se) == 4737 and not se.own_experiment_id.eq(se.history_experiment_id).any()
    sg = sm.set_index('experiment_id').perturbation_target
    assert se.gene.eq(se.history_experiment_id.map(sg)).all()
    source_views, forbidden, allowed, fits, edges_out, naive = [], [], [], [], [], []
    for fold in sorted(s.fold.unique()):
        train = s.loc[s.fold.ne(fold)]
        eval_all = s.loc[s.fold.eq(fold)]
        train_edges = se.loc[se.fold.ne(fold)]
        eval_edges = se.loc[se.fold.eq(fold)]
        all_forbidden = set(eval_all.own_experiment_id)
        globally_filtered = eval_edges.loc[~eval_edges.history_experiment_id.isin(all_forbidden)]
        naive.append({'fold': int(fold), 'query_count': len(eval_all), 'query_supported': globally_filtered.task_id.nunique(), 'edges_before': len(eval_edges), 'edges_after': len(globally_filtered)})
        for context in sorted(s.target.unique()):
            name = f'Source/outer{fold}/eval_{context}'
            query = eval_all.loc[eval_all.target.eq(context)]
            blocked = set(query.own_experiment_id)
            fit_edges = train_edges.loc[~train_edges.history_experiment_id.isin(blocked)]
            inference_edges = eval_edges.loc[eval_edges.target.eq(context) & ~eval_edges.history_experiment_id.isin(blocked)]
            used_ids = set(fit_edges.history_experiment_id)
            assert len(fit_edges) == len(train_edges)
            assert len(inference_edges) == len(eval_edges.loc[eval_edges.target.eq(context)])
            assert not used_ids & blocked
            fit_signature = ids_sha(train.task_id) + ':' + ids_sha(used_ids)
            source_views.append({'scope': name, 'fold': int(fold), 'evaluation_context': context,
                'train_queries': len(train), 'train_clusters': train.gene.nunique(), 'train_edges': len(fit_edges),
                'fit_unique_history_items': len(used_ids), 'fit_signature': fit_signature,
                'query_count': len(query), 'query_clusters': query.gene.nunique(), 'query_edges': len(inference_edges),
                'query_supported': inference_edges.task_id.nunique(), 'forbidden_items': len(blocked),
                'allowed_bank_items': len(sm) - len(blocked)})
            forbidden.extend({'scope': name, 'experiment_id': x} for x in sorted(blocked))
            allowed.extend({'scope': name, 'experiment_id': x} for x in sorted(set(sm.experiment_id) - blocked))
            fits.extend({'scope': name, 'experiment_id': x} for x in sorted(used_ids))
            edges_out.extend({'scope': name, 'phase': 'predict', 'task_id': row.task_id, 'history_experiment_id': row.history_experiment_id} for row in inference_edges.itertuples())
            edges_out.extend({'scope': name, 'phase': 'fit', 'task_id': row.task_id, 'history_experiment_id': row.history_experiment_id} for row in fit_edges.itertuples())
    source_views = pd.DataFrame(source_views)
    assert source_views.groupby('fold').fit_signature.nunique().eq(1).all()

    mc = pd.read_csv(inputs['mc_dev_queries'])
    mm = pd.read_parquet(inputs['mc_bank_metadata'])
    mc['fold'] = mc.perturbation.map(lambda g: int.from_bytes(hashlib.sha256(f'SafeConf-McFaline-Decoder-repair-v1\0{g}'.encode()).digest()[:8], 'big') % 5)
    mc['own_experiment_id'] = 'McFaline23::' + mc.task_id
    assert len(mc) == 542 and mc.task_id.is_unique and mm.experiment_id.is_unique
    assert mc.own_experiment_id.isin(mm.experiment_id).all()
    roles = mm.provenance.str.rsplit('::', n=1).str[-1]
    assert dict(roles.value_counts()) == {'train': 6631, 'val': 542}
    assert set(mc.own_experiment_id) == set(mm.loc[roles.eq('val'), 'experiment_id'])
    me = mc.merge(mm, left_on='perturbation', right_on='perturbation_target', suffixes=('_query', '_memory'))
    me = me.loc[~(me.context_query.eq(me.context_memory) & me.treatment.eq(me.condition))]
    assert len(me) == 7157
    mc_views = []
    for fold in range(5):
        name = f'McFaline/outer{fold}'
        query = mc.loc[mc.fold.eq(fold)]
        train = mc.loc[mc.fold.ne(fold)]
        blocked = set(query.own_experiment_id)
        train_edges = me.loc[me.fold.ne(fold) & ~me.experiment_id.isin(blocked)]
        query_before = me.loc[me.fold.eq(fold)]
        query_edges = query_before.loc[~query_before.experiment_id.isin(blocked)]
        used_ids = set(train_edges.experiment_id)
        counts = query_edges.groupby('task_id').size()
        assert train_edges.task_id.nunique() == len(train)
        assert query_edges.task_id.nunique() == len(query)
        assert not used_ids & blocked
        mc_views.append({'scope': name, 'fold': fold, 'train_queries': len(train), 'train_clusters': train.perturbation.nunique(),
            'train_edges': len(train_edges), 'fit_unique_history_items': len(used_ids),
            'query_count': len(query), 'query_clusters': query.perturbation.nunique(), 'query_edges_before': len(query_before),
            'query_edges': len(query_edges), 'query_supported': len(counts), 'min_query_K': int(counts.min()),
            'max_query_K': int(counts.max()), 'forbidden_items': len(blocked), 'allowed_bank_items': len(mm) - len(blocked)})
        forbidden.extend({'scope': name, 'experiment_id': x} for x in sorted(blocked))
        allowed.extend({'scope': name, 'experiment_id': x} for x in sorted(set(mm.experiment_id) - blocked))
        fits.extend({'scope': name, 'experiment_id': x} for x in sorted(used_ids))
        edges_out.extend({'scope': name, 'phase': 'predict', 'task_id': row.task_id, 'history_experiment_id': row.experiment_id} for row in query_edges.itertuples())
        edges_out.extend({'scope': name, 'phase': 'fit', 'task_id': row.task_id, 'history_experiment_id': row.experiment_id} for row in train_edges.itertuples())
    paths = {
        'source_identity': write_csv('SOURCE_QUERY_EXPERIMENT_MAPPING.csv', identity[['task_id', 'gene', 'target', 'condition', 'fold', 'own_experiment_id', 'effect_vector_row', 'n_target_cells', 'n_cells', 'n_target_batches', 'n_batches', 'provenance']]),
        'mc_identity': write_csv('MC_DEV_QUERY_EXPERIMENT_MAPPING.csv', mc),
        'source_views': write_csv('SOURCE_SCOPE_COUNTS.csv', source_views, public=True),
        'source_naive_global': write_csv('SOURCE_ALL_CONTEXT_GLOBAL_EXCLUSION_DIAGNOSTIC.csv', pd.DataFrame(naive), public=True),
        'mc_views': write_csv('MC_SCOPE_COUNTS.csv', pd.DataFrame(mc_views), public=True),
        'forbidden_ids': write_csv('FORBIDDEN_EXPERIMENT_IDS.csv', pd.DataFrame(forbidden)),
        'allowed_ids': write_csv('ALLOWED_EXPERIMENT_IDS.csv', pd.DataFrame(allowed)),
        'preprocessor_fit_ids': write_csv('PREPROCESSOR_FIT_UNIQUE_HISTORY_IDS.csv', pd.DataFrame(fits)),
        'scoped_query_edges': write_csv('SCOPED_QUERY_HISTORY_EDGES.csv', pd.DataFrame(edges_out)),
    }
    report = {'status': 'METADATA_SCOPE_AUDIT_PASS_WITH_REQUIRED_SOURCE_CONTEXT_VIEWS',
        'no_expression_or_prediction_arrays_read': True, 'mc_TEST_values_read': 0, 'model_fits': 0,
        'source_exact_query_memory_units': len(identity), 'source_equal_cell_count_units': int(identity.n_target_cells.eq(identity.n_cells).sum()),
        'source_equal_batch_count_units': int(identity.n_target_batches.eq(identity.n_batches).sum()),
        'source_legal_evaluation_edges': int(source_views.query_edges.sum()),
        'source_naive_all_context_query_supported': int(pd.DataFrame(naive).query_supported.sum()),
        'source_naive_all_context_edges': int(pd.DataFrame(naive).edges_after.sum()),
        'source_fit_parameter_scopes': int(source_views.fit_signature.nunique()),
        'source_inference_history_views': len(source_views),
        'mc_dev_query_units': len(mc), 'mc_original_edges': len(me), 'mc_filtered_evaluation_edges': sum(x['query_edges'] for x in mc_views),
        'mc_bank_roles': {str(k): int(v) for k, v in roles.value_counts().items()},
        'source_scope_interpretation': 'Separate deployment episodes indexed by outer gene fold and evaluation context; other contexts are observed Public experiments in that episode. Parameters fit only non-evaluation genes and can be shared when the exact training data/preprocessor IDs match.',
        'pca_rule': 'Fit only unique history items actually referenced by permitted fitting queries; never fit every item in the permitted inference bank.',
        'inner_rule': 'Repeat experiment exclusions at each early-stop/OOF scope. Source inner validation is also split into context views; MC additionally removes every inner-validation experiment ID. Outer forbidden IDs remain excluded. Query features/conflict are rebuilt after history filtering.',
        'derivative_rule': 'Any future guide/plate/split-half record inherits the same canonical parent experiment key and is excluded with that parent. Current bank stores one row per canonical experiment.',
        'source_identity_evidence': 'Exact context-condition-gene join and matching treated-cell/batch counts for all 1808 queries, plus existing builder canonicalization across physical blind-view copies; no new cell-expression equality claim.',
        'input_bindings': {key: {'path': str(value), 'sha256': sha(value)} for key, value in inputs.items()},
        'outputs': paths, 'audit_script_sha256': sha(__file__)}
    (DOC / 'INPUT_SCOPE_AUDIT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'outputs', 'input_bindings'}}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
