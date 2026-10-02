#!/usr/bin/env python3
"""Identity/physical-null feasibility and original reference replay; no query truth."""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import seal_safeconf_orion_source_risk_agent as math_core

BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
SEAL = BASE / 'orion_public_bank_extension_20261002_v4_partitioned/expanded_risk_seal'
COMPARISON = SEAL.parent / 'expanded_comparison'
BANK = BASE / 'public_source_history_expanded_20261002_v2'
FIT = BASE / 'orion_published_lm_20261002_v3_decimal_tokens'
PRIMARY = BASE / 'orion_registered_test_extendedbank_20261002_v1/fixed_risk_evaluation/primary_common_COHORT.csv'
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/physical_content_null_preparation_v1'
QUERY = math_core.QUERY
MATH_SHA = '8b69e6ce29dc92aa34150b3b08aac62ce8333a10f84ec432a39d0e81ffc64482'
GROUP_COLS = ['original_study', 'context', 'perturbation_type', 'effect_contract_id',
              'gene_space_id', 'control_source', 'n_batches', 'cell_support_quartile']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def check(item):
    path = Path(item['path'])
    if bind(path) != item: raise RuntimeError(f'Frozen artifact binding differs: {path}')
    return path


def bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def write_json(path, obj):
    with Path(path).open('x') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def main():
    started, cpu = time.monotonic(), time.process_time()
    if DOC.exists(): raise RuntimeError('Fresh preparation namespace required')
    DOC.mkdir()
    write_json(DOC / 'EXECUTION_REGISTRATION.json', {'pid': os.getpid(), 'script': bind(__file__),
        'scope': 'known232query predictions and cached historicalPublic vectors only', 'query_truth_read': False,
        'actual_permutations': 0, 'fits': 0, 'start_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print(json.dumps({'pid': os.getpid(), 'phase': 'identity_and_reference_replay', 'query_truth_read': False}), flush=True)
    try:
        if sha(math_core.__file__) != MATH_SHA: raise RuntimeError('Original inference math changed')
        seal = json.loads((SEAL / 'RISK_SEAL_MANIFEST.json').read_text())
        manifest = json.loads((BANK / 'PUBLIC_BANK_MANIFEST.json').read_text())
        if bind(BANK / 'PUBLIC_BANK_MANIFEST.json') != seal['public_bank_manifest']:
            raise RuntimeError('Original expanded retrieval bank differs')
        registry = {x['path']: x for x in json.loads((SEAL / 'ARTIFACT_HASHES.json').read_text())}
        paths = [Path(__file__), Path(math_core.__file__), PRIMARY, SEAL / 'RISK_SEAL_MANIFEST.json', SEAL / 'ARTIFACT_HASHES.json',
                 COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet', BANK / 'PUBLIC_BANK_MANIFEST.json']
        for name in ['metadata', 'effects', 'axis', 'gene_ids']:
            paths.append(check(manifest['extended_bank'][name]))
        paths.append(check(manifest['canonical_unit_provenance']))
        for name in ['SOURCE_HISTORY_PAIR_FEATURES.parquet', 'SOURCE_PRIOR_WEIGHTS.tsv', 'Manual_P_PUBLIC_FEATURES.parquet',
                     'Uniform_P_PUBLIC_FEATURES.parquet', 'Manual_PRIOR_EFFECTS.npy', 'Uniform_PRIOR_EFFECTS.npy']:
            path = SEAL / name
            if sha(path) != registry[name]['sha256'] or path.stat().st_size != registry[name]['bytes']:
                raise RuntimeError('Frozen original reference cache differs: ' + name)
            paths.append(path)
        queries = pd.read_csv(PRIMARY, usecols=QUERY, keep_default_na=False)
        scores = pd.read_parquet(COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet', columns=QUERY +
            ['Uniform_DirectRMSE', 'Manual_DirectRMSE', 'Manual_WeightedHistoryDistance'], use_threads=False)
        scores['global_query_row'] = np.arange(len(scores))
        queries = queries.merge(scores, on=QUERY, validate='one_to_one')
        if len(queries) != 232 or queries.target_gene_id.nunique() != 144 or not queries.role.eq('TEST').all():
            raise RuntimeError('Exact frozen primary identity scope required')
        memory = pd.read_parquet(BANK / 'SOURCE_PUBLIC_MEMORY_METADATA.parquet', use_threads=False)
        provenance = pd.read_csv(BANK / 'CANONICAL_UNIT_PROVENANCE.csv', keep_default_na=False)
        provenance = provenance.set_index('experiment_id').loc[memory.experiment_id].reset_index()
        if (not memory.experiment_id.is_unique or not provenance.physical_unit_id.is_unique
            or not np.array_equal(memory.effect_vector_row, np.arange(5365))
            or not np.array_equal(memory.context, provenance.source_context)
            or not np.array_equal(memory.condition, provenance.condition)
            or not np.array_equal(memory.perturbation_target, provenance.target_gene)):
            raise RuntimeError('Canonical physical-unit/row identity mismatch')
        if not memory.eligibility.eq(True).all() or not memory.perturbation_type.eq('genetic_single_gene').all():
            raise RuntimeError('Original historical eligibility differs')
        memory['original_study'] = provenance.original_study.to_numpy()
        memory['physical_unit_id'] = provenance.physical_unit_id.to_numpy()
        if memory.groupby(['context', 'perturbation_target']).size().max() != 1:
            raise RuntimeError('Distinct target derangement feasibility needs unique context-target units')
        pair = pd.read_parquet(SEAL / 'SOURCE_HISTORY_PAIR_FEATURES.parquet', columns=
            ['query_id', 'memory_id', 'memory_row', 'log_source_cells'], use_threads=False)
        pair = pair[pair.query_id.isin(queries.query_id)].sort_values(['query_id', 'memory_row'], kind='stable')
        weight = pd.read_csv(SEAL / 'SOURCE_PRIOR_WEIGHTS.tsv', sep='\t', keep_default_na=False, float_precision='round_trip')
        weight = weight[weight.query_id.isin(queries.query_id) & weight.reference.isin(['Uniform', 'Manual'])]
        if pair.duplicated(['query_id', 'memory_id']).any(): raise RuntimeError('Duplicate query-history link')
        effects = np.load(BANK / 'SOURCE_PUBLIC_EFFECTS.npy', mmap_mode='r', allow_pickle=False)
        if effects.shape != (5365, 3285) or effects.dtype != np.float64: raise RuntimeError('Full fixed historical axis required')
        axis = pd.read_csv(BANK / 'GENE_MANIFEST.csv', keep_default_na=False)
        delta = {}
        for context in ['HCT116', 'HEK293T']:
            folder = FIT / context
            selected = queries[queries.context_id.eq(context)].query_id.tolist()
            upstream_ids = pd.read_csv(folder / 'QUERY_IDENTITIES.tsv', sep='\t', keep_default_na=False)
            upstream_axis = pd.read_csv(folder / 'OUTPUT_GENE_AXIS.tsv', sep='\t', keep_default_na=False)
            if upstream_axis.gene_id.tolist() != axis.orion_ensembl_id.tolist(): raise RuntimeError('Prediction axis differs')
            matched = upstream_ids[upstream_ids.query_id.isin(selected)]
            expected = queries[queries.context_id.eq(context)][QUERY]
            if set(map(tuple, matched[QUERY].to_numpy())) != set(map(tuple, expected.to_numpy())):
                raise RuntimeError('Published prediction query identity differs')
            prediction_path = folder / 'PREDICTIONS_DELTA.tsv.gz'
            recorded = next(x for x in seal['LM_prediction_and_control_bindings'] if x['role'] == 'delta' and x['context'] == context)
            if sha(prediction_path) != recorded['sha256']: raise RuntimeError('Frozen predicted DELTA differs')
            # Same default C parser as original8b69.load_inputs; only fixed232 prediction columns convert.
            pred = pd.read_csv(prediction_path, sep='\t', usecols=['gene_id'] + selected, keep_default_na=False)
            if pred.gene_id.tolist() != axis.orion_ensembl_id.tolist(): raise RuntimeError('Prediction gene row order differs')
            for qid in selected: delta[qid] = pred[qid].to_numpy(float)
            paths += [folder / 'QUERY_IDENTITIES.tsv', folder / 'OUTPUT_GENE_AXIS.tsv', prediction_path]
        before = [bind(p) for p in paths]
        features = {ref: pd.read_parquet(SEAL / f'{ref}_P_PUBLIC_FEATURES.parquet', use_threads=False) for ref in ['Uniform', 'Manual']}
        cached_priors = {ref: np.load(SEAL / f'{ref}_PRIOR_EFFECTS.npy', mmap_mode='r', allow_pickle=False) for ref in features}
        replay = []
        for item in queries.itertuples(index=False):
            group = pair[pair.query_id.eq(item.query_id)].copy()
            rows = group.memory_row.to_numpy(int)
            actual = memory.iloc[rows]
            expected_ids = memory[memory.perturbation_target.eq(item.target_gene_symbol)].experiment_id.tolist()
            if group.memory_id.tolist() != expected_ids or not actual.eligibility.all():
                raise RuntimeError('Exact same-gene query history eligibility changed')
            if not np.array_equal(group.memory_id, actual.experiment_id): raise RuntimeError('Memory row/ID map differs')
            for ref in ['Uniform', 'Manual']:
                prior, weights, uncertainty, effective = math_core.weighted_prior(group, effects, ref)
                saved_w = weight[weight.query_id.eq(item.query_id) & weight.reference.eq(ref)].set_index('memory_id').loc[group.memory_id].weight.to_numpy(float)
                feature = features[ref].iloc[int(item.global_query_row)]
                direct = float(np.sqrt(np.mean((delta[item.query_id] - prior)**2)))
                weighted = float(np.sqrt(direct**2 + uncertainty**2))
                known_direct = getattr(item, ref + '_DirectRMSE')
                checks = {'prior_bits': bits(prior, cached_priors[ref][int(item.global_query_row)]), 'weight_bits': bits(weights, saved_w),
                          'uncertainty_bits': bits([uncertainty], [feature.prior_uncertainty]),
                          'effective_sources_bits': bits([effective], [feature.effective_sources]),
                          'direct_feature_bits': bits([direct], [feature.prediction_prior_rmse]),
                          'direct_score_bits': bits([direct], [known_direct]),
                          'weighted_frozen_feature_formula_bits': bits([weighted], [np.sqrt(feature.prediction_prior_rmse**2 + feature.prior_uncertainty**2)])}
                if ref == 'Manual': checks['registered_weighted_score_bits'] = bits([weighted], [item.Manual_WeightedHistoryDistance])
                if not all(checks.values()): raise RuntimeError('Original exact reference reproduction failed: ' + json.dumps({'query_id': item.query_id, 'reference': ref, 'checks': checks}))
                replay.append({'query_id': item.query_id, 'context': item.context_id, 'reference': ref,
                               'n_history_records': len(group), **checks, 'max_abs_prior_gap': float(np.max(np.abs(prior - cached_priors[ref][int(item.global_query_row)])))})
        # Only metadata matching feasibility; no RNG or actual donor assignment here.
        preliminary = GROUP_COLS[:-1]
        memory['cell_support_quartile'] = -1
        for _, group in memory.groupby(preliminary, sort=True, dropna=False):
            bins = pd.qcut(np.log1p(group.n_cells), 4, labels=False, duplicates='drop').fillna(0).astype(int)
            memory.loc[group.index, 'cell_support_quartile'] = bins
        group_rows = []
        movable = set()
        for number, (key, group) in enumerate(memory.groupby(GROUP_COLS, sort=True, dropna=False)):
            if len(group) >= 2: movable.update(group.effect_vector_row)
            group_rows.append(dict(zip(GROUP_COLS, key)) | {'group_id': number, 'n_records': len(group),
                'n_distinct_targets': group.perturbation_target.nunique(), 'derangement_all_distinct_targets_possible': len(group) >= 2,
                'min_cells': int(group.n_cells.min()), 'max_cells': int(group.n_cells.max())})
        coverage = []
        for qid, group in pair.groupby('query_id', sort=True):
            moved = group.memory_row.isin(movable).to_numpy()
            manual = weight[weight.query_id.eq(qid) & weight.reference.eq('Manual')].set_index('memory_id').loc[group.memory_id].weight.to_numpy(float)
            coverage.append({'query_id': qid, 'history_records': len(group), 'movable_records': int(moved.sum()),
                             'movable_manual_weight': float(manual[moved].sum()), 'movable_uniform_weight': float(moved.mean())})
        pd.DataFrame(replay).to_csv(DOC / 'EXACT_ORIGINAL_REFERENCE_REPLAY.csv', index=False)
        pd.DataFrame(group_rows).to_csv(DOC / 'PHYSICAL_PERMUTATION_GROUP_FEASIBILITY.csv', index=False)
        pd.DataFrame(coverage).to_csv(DOC / 'FIXED_PRIMARY_HISTORY_MOVABILITY.csv', index=False)
        queries[QUERY + ['global_query_row']].to_csv(DOC / 'FIXED_PRIMARY_QUERY_IDENTITIES.csv', index=False)
        memory[['experiment_id', 'effect_vector_row', 'perturbation_target', 'physical_unit_id'] + GROUP_COLS].to_csv(DOC / 'FROZEN_RECIPIENT_GROUP_METADATA.csv', index=False)
        if before != [bind(p) for p in paths]: raise RuntimeError('Original inputs/code changed during replay')
        result = {'schema': 'safeconf_orion_physical_content_null_preparation_v1', 'status': 'PREPARATION_AND_EXACT_REFERENCE_REPLAY_PASS',
            'fixed_primary_queries': 232, 'independent_query_gene_clusters': 144, 'bank_records': 5365, 'bank_manifest': bind(BANK / 'PUBLIC_BANK_MANIFEST.json'),
            'replayed_query_reference_records': len(replay), 'all_original_prior_weights_uncertainty_direct_and_Manualweighted_bits_exact': True,
            'Uniform_weighted_is_only_internal_formula_check_not_registered13score': True,
            'contract_group_columns': GROUP_COLS, 'same_context_target_unique': True,
            'recipient_support_weights_query_history_IDs_and_eligibility_unchanged': True,
            'allowed_donor_source': 'entire frozen eligible historical5365bank; only same listed group, different perturbation_target/physicalunit',
            'permutation_operation_if_later_registered': 'cyclic derangement of randomly ordered members in each nonsingleton group; actual effect vector reassignment, not score shuffling',
            'support_matching': 'exact n_batches plus within-group quartile log1p n_cells; not exact donor cellcount or biological/measurement variance matching',
            'historical_source_study': 'canonical physical-unit sidecar; legacy generic study_id not treated as real study',
            'literal_gene_condition_not_grouped': 'all genetic-single-gene entries; actual context-target uniqueness checked; literalcondition would block target-breaking null',
            'raw_legacy_gene_space_labels_retained_in_matching': True,
            'matching_groups': len(group_rows), 'singleton_groups': sum(x['n_records'] == 1 for x in group_rows),
            'movable_bank_records': len(movable), 'primary_queries_with_any_movable_history': sum(x['movable_records'] > 0 for x in coverage),
            'queries_with_no_movable_history_remain_in_fixed_cohort': [x['query_id'] for x in coverage if x['movable_records'] == 0],
            'five_formal_permutation_seeds_not_executed_or_selected': True,
            'new_fits': 0, 'model_inference_calls': 0, 'raw_expression_reads': 0, 'query_truth_values_read': 0,
            'actual_permutations': 0, 'bootstrap_draws': 0, 'new_confirmation': False, 'original13methods_or_Source_parameters_changed': False,
            'elapsed_seconds': time.monotonic() - started, 'CPU_seconds': time.process_time() - cpu,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'input_bindings': before, 'output_bindings': [bind(p) for p in sorted(DOC.glob('*.csv'))]}
        write_json(DOC / 'PREPARATION_RESULT.json', result)
        print(json.dumps({k: result[k] for k in ['status', 'replayed_query_reference_records', 'movable_bank_records', 'primary_queries_with_any_movable_history', 'elapsed_seconds', 'query_truth_values_read']}), flush=True)
    except BaseException as e:
        write_json(DOC / 'ABORT.json', {'status': 'ABORT', 'type': type(e).__name__, 'message': str(e), 'query_truth_values_read': 0, 'new_fits': 0})
        raise


if __name__ == '__main__': main()
