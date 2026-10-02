#!/usr/bin/env python3
"""Fixed SEEN physical-history nulls; no training, original three rules only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import prepare_safeconf_orion_physical_content_null_scorecodec_agent as prep

BASE = prep.BASE
DOC = prep.DOC.parent / 'physical_content_null_actual_v1'
OUT = BASE / 'orion_fixed_physical_content_null_20261002_v1'
CONTROL = BASE / 'orion_existing_validation_risk_control_20261002_v1'
TRUTH = BASE / 'orion_registered_test_extendedbank_20261002_v1/registered_test_truth/TEST_TASK_ERRORS.parquet'
METRIC_CODE = ROOT / 'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py'
METRIC_SHA = 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c'
PREP_SHA = '3898f0441c5ae97b2cd5d0dc9520ba17c33e15bc7b72f59b9128b59ea5a0aa5a'
SEEDS = [20260930, 20261001, 20261002, 20261003, 20261004]
RULES = ['Uniform_DirectRMSE', 'Manual_DirectRMSE', 'Manual_WeightedHistoryDistance']


def row_scores(query, pair, effects, prediction):
    values = []
    for item in query.itertuples(index=False):
        group = pair[pair.query_id.eq(item.query_id)]
        scores = {}
        for ref in ['Uniform', 'Manual']:
            prior, _, uncertainty, _ = prep.math_core.weighted_prior(group, effects, ref)
            distance = float(np.sqrt(np.mean((prediction[item.query_id] - prior)**2)))
            scores[ref + '_DirectRMSE'] = prep.original_score_codec(distance)
            if ref == 'Manual':
                scores['Manual_WeightedHistoryDistance'] = prep.original_score_codec(np.sqrt(distance**2 + uncertainty**2))
        values.append(scores)
    return pd.DataFrame(values, index=query.index)


def main():
    began, cpu = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1200 seconds exceeded')))
    signal.alarm(1200)
    if DOC.exists() or OUT.exists():
        raise RuntimeError('Fresh immutable diagnostic roots required')
    if prep.sha(prep.__file__) != PREP_SHA or prep.sha(METRIC_CODE) != METRIC_SHA:
        raise RuntimeError('Verified original preparation or metric code differs')
    preparation = json.loads((prep.DOC / 'PREPARATION_RESULT.json').read_text())
    if preparation['status'] != 'PREPARATION_AND_EXACT_REFERENCE_REPLAY_PASS':
        raise RuntimeError('Exact original reference replay gate missing')
    for item in preparation['input_bindings'] + preparation['output_bindings']:
        prep.check(item)
    if prep.sha(prep.math_core.__file__) != prep.MATH_SHA:
        raise RuntimeError('Original inference math changed')
    inputs = [Path(__file__), Path(prep.__file__), Path(prep.math_core.__file__), METRIC_CODE,
              prep.DOC / 'PREPARATION_RESULT.json', prep.DOC / 'FROZEN_RECIPIENT_GROUP_METADATA.csv',
              prep.DOC / 'FIXED_PRIMARY_QUERY_IDENTITIES.csv', CONTROL / 'BOOTSTRAP_GENE_INDICES.npy',
              CONTROL / 'ALL_PRIMARY_PREDICTIONS.parquet', TRUTH]
    inputs += [Path(v['path']) for v in preparation['input_bindings'] if Path(v['path']).suffix in ['.npy', '.parquet', '.gz']]
    inputs = list(dict.fromkeys(inputs))
    before = [prep.bind(path) for path in inputs]
    DOC.mkdir(); OUT.mkdir()
    prep.write_json(DOC / 'DIAGNOSTIC_REGISTRATION.json', {
        'role': 'SEEN_postconfirmation_fixed_physical_content_negative_control_not_confirmation',
        'same_fixed232tasks144genes_and_original3rules': RULES,
        'permutation_seeds_all_retained': SEEDS, 'matching_group_columns': prep.GROUP_COLS,
        'operation': 'Cyclic derangement within each nonsingleton eligible Source group; actual effect vectors reassigned',
        'support': 'Recipient counts/batches/weights retained; donor counts quartile matched, not exact variance matched',
        'query_exclusions': 'No task or nonmovable history exclusion; original232query identity unchanged',
        'fixed_paired_bootstrap': 'Reuse saved5000canonical-sorted144gene indices; no new RNG for bootstrap',
        'permutation_RNG_is_not_bootstrap_RNG': True, 'new_fits': 0, 'model_inference_calls': 0,
        'Source_parameters_or_original13primary_changed': False, 'raw_expression_reads': 0,
        'truth_scope': 'Cached232alreadyopenedTESTerrors for statistics only after null score sealing',
        'primary_change': False, 'causal_or_measurement_variance_isolation_claim': False,
        'input_bindings': before, 'resources': {'wall_cap_seconds': 1200, 'CPU_threads': 4, 'GPU_hours': 0}})
    query = pd.read_csv(prep.DOC / 'FIXED_PRIMARY_QUERY_IDENTITIES.csv', keep_default_na=False)
    memory = pd.read_csv(prep.DOC / 'FROZEN_RECIPIENT_GROUP_METADATA.csv', keep_default_na=False)
    bank = pd.read_parquet(prep.BANK / 'SOURCE_PUBLIC_MEMORY_METADATA.parquet')
    if not np.array_equal(memory.experiment_id, bank.experiment_id) or len(query) != 232:
        raise RuntimeError('Original bank/query identities differ')
    memory['n_cells'] = bank.n_cells.to_numpy()
    groups = [part.effect_vector_row.to_numpy(int) for _, part in memory.groupby(prep.GROUP_COLS, sort=True, dropna=False)]
    pairs = pd.read_parquet(prep.SEAL / 'SOURCE_HISTORY_PAIR_FEATURES.parquet',
                           columns=['query_id', 'memory_id', 'memory_row', 'log_source_cells'])
    pairs = pairs[pairs.query_id.isin(query.query_id)].sort_values(['query_id', 'memory_row'], kind='stable')
    effects = np.load(prep.BANK / 'SOURCE_PUBLIC_EFFECTS.npy', mmap_mode='r', allow_pickle=False)
    predictions = {}
    for context in ['HCT116', 'HEK293T']:
        ids = query[query.context_id.eq(context)].query_id.tolist()
        frame = pd.read_csv(prep.FIT / context / 'PREDICTIONS_DELTA.tsv.gz', sep='\t', usecols=['gene_id'] + ids, keep_default_na=False)
        for qid in ids: predictions[qid] = frame[qid].to_numpy(float)
    known = pd.read_parquet(prep.COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet', columns=prep.QUERY + RULES)
    scores = query.merge(known, on=prep.QUERY, validate='one_to_one')
    replay = row_scores(query, pairs, effects, predictions)
    if replay[RULES].to_numpy().tobytes() != scores[RULES].to_numpy().tobytes():
        raise RuntimeError('All three original rule scores must reproduce bitwise before nulls')
    donor_records, shift_records = [], []
    score_names = RULES.copy()
    moved_sets = []
    for seed in SEEDS:
        donor = np.arange(len(memory))
        rng = np.random.default_rng(seed)
        for group in groups:
            if len(group) >= 2:
                ordered = group[rng.permutation(len(group))]
                donor[ordered] = np.roll(ordered, 1)
        moved = donor != np.arange(len(memory))
        if moved.sum() != preparation['movable_bank_records']:
            raise RuntimeError('All and only nonsingleton Source records must move')
        for col in prep.GROUP_COLS:
            if not np.array_equal(memory[col].to_numpy(), memory.iloc[donor][col].to_numpy()):
                raise RuntimeError('Donor crossed a frozen matching/normalization contract')
        if (memory.perturbation_target.to_numpy()[moved] == memory.iloc[donor].perturbation_target.to_numpy()[moved]).any():
            raise RuntimeError('Moved historical effect must break gene pairing')
        moved_sets.append(moved)
        recipient_rows = pairs.memory_row.to_numpy(int)
        null_pairs = pairs.copy()
        null_pairs['memory_row'] = donor[recipient_rows]
        null = row_scores(query, null_pairs, effects, predictions)
        for rule in RULES:
            name = f'Null{seed}_{rule}'
            score_names.append(name)
            scores[name] = null[rule].to_numpy()
        ratio = memory.n_cells.to_numpy(float)[donor] / memory.n_cells.to_numpy(float)
        for row, source in enumerate(donor):
            donor_records.append({'seed': seed, 'recipient_effect_row': row, 'donor_effect_row': int(source),
                'recipient_experiment_id': memory.experiment_id.iloc[row], 'donor_experiment_id': memory.experiment_id.iloc[source],
                'moved': bool(moved[row]), 'recipient_n_cells': int(memory.n_cells.iloc[row]),
                'donor_n_cells': int(memory.n_cells.iloc[source]), 'donor_to_recipient_cell_ratio': ratio[row]})
        shift_records.append({'seed': seed, 'moved_bank_rows': int(moved.sum()),
            'recipient_weight_and_support_unchanged': True, 'minimum_donor_cell_ratio': float(ratio.min()),
            'maximum_donor_cell_ratio': float(ratio.max())})
        print(json.dumps({'completed_physical_null_seed': seed, 'moved_records': int(moved.sum())}), flush=True)
    if not np.isfinite(scores[score_names].to_numpy()).all():
        raise RuntimeError('All232tasks must remain finite; no survivor selection')
    score_path = OUT / 'ALL_FIXED_ACTUAL_AND_NULL_RULE_SCORES.parquet'
    scores[prep.QUERY + score_names].to_parquet(score_path, index=False)
    pd.DataFrame(donor_records).to_csv(OUT / 'PHYSICAL_DONOR_ASSIGNMENTS.csv.gz', index=False)
    pd.DataFrame(shift_records).to_csv(DOC / 'PHYSICAL_NULL_MOVEMENT_AND_SUPPORT_AUDIT.csv', index=False)
    prep.write_json(DOC / 'ALL_NULL_SCORES_SEAL.json', {
        'scores': prep.bind(score_path), 'physical_donors': prep.bind(OUT / 'PHYSICAL_DONOR_ASSIGNMENTS.csv.gz'),
        'fixed_query_rows': 232, 'gene_clusters': 144, 'scores_count': 18,
        'all_original_actual_rules_bits_match': True, 'all5nulls_saved_before_cached_truth_parse': True,
        'new_fits': 0, 'new_confirmation': False})
    # The whole cohort and all null predictions are fixed; only now parse cached SEEN errors.
    truth = pd.read_parquet(TRUTH, columns=prep.QUERY + ['true_error_rmse'], filters=[('query_id', 'in', query.query_id.tolist())])
    frame = scores.merge(truth, on=prep.QUERY, validate='one_to_one')
    if len(frame) != 232 or not np.isfinite(frame.true_error_rmse).all():
        raise RuntimeError('Same complete232cachedTESTerrors required')
    spec = importlib.util.spec_from_file_location('frozen_physical_null_metrics', METRIC_CODE)
    metric = importlib.util.module_from_spec(spec); spec.loader.exec_module(metric)
    genes = sorted(frame.target_gene_id.astype(str).unique())
    control_query = pd.read_parquet(CONTROL / 'ALL_PRIMARY_PREDICTIONS.parquet', columns=prep.QUERY)
    if genes != sorted(control_query.target_gene_id.astype(str).unique()):
        raise RuntimeError('Saved144gene index order differs')
    if set(map(tuple, frame[prep.QUERY].to_numpy())) != set(map(tuple, control_query[prep.QUERY].to_numpy())):
        raise RuntimeError('Saved draws must be used on the identical fixed biological population')
    indices = np.load(CONTROL / 'BOOTSTRAP_GENE_INDICES.npy', mmap_mode='r')
    if indices.shape != (5000, 144): raise RuntimeError('All5000original draw indices required')
    blocks = [np.flatnonzero(frame.target_gene_id.astype(str).eq(g)) for g in genes]
    contexts = ['HCT116', 'HEK293T', 'macro']
    points = metric.metric_values(frame, score_names)
    draws = np.full((5000, 3, 18, 7), np.nan)
    for index, sampled in enumerate(indices):
        selected = frame.iloc[np.concatenate([blocks[int(i)] for i in sampled])].reset_index(drop=True)
        values = metric.metric_values(selected, score_names)
        for ci, context in enumerate(contexts):
            for mi, name in enumerate(score_names):
                draws[index, ci, mi] = [values[context][name][m] for m in metric.METRICS]
        if index % 1000 == 0:
            print(json.dumps({'saved_bootstrap_draws_completed': index, 'total': 5000}), flush=True)
    np.save(OUT / 'ALL_FIXED_ACTUAL_NULL_METRIC_DRAWS.npy', draws)
    intervals, pairs_out = [], []
    for ci, context in enumerate(contexts):
        for mi, name in enumerate(score_names):
            for ki, key in enumerate(metric.METRICS):
                finite = draws[:, ci, mi, ki][np.isfinite(draws[:, ci, mi, ki])]
                lower, upper = np.quantile(finite, [.025, .975]) if len(finite) >= 2 else (np.nan, np.nan)
                intervals.append({'context': context, 'method': name, 'metric': key, 'point': points[context][name][key],
                    'ci95_lower': lower, 'ci95_upper': upper, 'valid_draws': len(finite), 'saved_draws': 5000})
        for rule in RULES:
            for seed in SEEDS:
                null_name = f'Null{seed}_{rule}'
                delta = draws[:, ci, score_names.index(rule)] - draws[:, ci, score_names.index(null_name)]
                for ki, key in enumerate(metric.METRICS):
                    finite = delta[:, ki][np.isfinite(delta[:, ki])]
                    lower, upper = np.quantile(finite, [.025, .975]) if len(finite) >= 2 else (np.nan, np.nan)
                    pairs_out.append({'context': context, 'reference': rule, 'null_seed': seed, 'metric': key,
                        'actual_point': points[context][rule][key], 'null_point': points[context][null_name][key],
                        'difference_actual_minus_null': points[context][rule][key] - points[context][null_name][key],
                        'ci95_lower': lower, 'ci95_upper': upper, 'valid_draws': len(finite), 'saved_draws': 5000,
                        'interpretation': 'SEEN association under matched support groups; not pure biological causality'})
    pd.DataFrame(intervals).to_csv(DOC / 'ALL_ACTUAL_NULL_METRIC_INTERVALS.csv', index=False)
    pd.DataFrame(pairs_out).to_csv(DOC / 'ALL_ACTUAL_MINUS_PHYSICAL_NULL_PAIRED_INTERVALS.csv', index=False)
    if len(intervals) != 378 or len(pairs_out) != 315:
        raise RuntimeError('All fixed references/seeds/contexts/metrics must remain reported')
    if before != [prep.bind(path) for path in inputs]:
        raise RuntimeError('Original code/artifacts changed during diagnostic')
    result = {'status': 'COMPLETE_FIXED_SEEN_PHYSICAL_CONTENT_NULL', 'original_reference_bits_reproduced': True,
        'query_tasks': 232, 'gene_clusters': 144, 'fixed_rules': RULES, 'all_seeds': SEEDS,
        'same5200bankrows_moved_each_seed': True, 'retained_queries_with_no_movable_history': preparation['queries_with_no_movable_history_remain_in_fixed_cohort'],
        'metric_rows': 378, 'paired_rows': 315, 'existing_bootstrap_draws': 5000, 'new_bootstrap_draws': 0,
        'actual_physical_permutations': 5, 'new_fits': 0, 'model_inference_calls': 0,
        'raw_expression_reads': 0, 'new_confirmation': False, 'original13methods_changed': False,
        'elapsed_seconds': time.monotonic() - began, 'CPU_seconds': time.process_time() - cpu,
        'reported_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'original_input_bindings': before, 'outputs': [prep.bind(p) for p in sorted(DOC.glob('*.csv'))]}
    prep.write_json(DOC / 'RESULT_MANIFEST.json', result)
    print(json.dumps({'status': result['status'], 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
