#!/usr/bin/env python3
"""Complete fixed common-axis Public coverage statistics, without new fitting."""
from pathlib import Path
import hashlib
import importlib.util
import itertools
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
from tools.safeconf_continual.research import metrics

RESULTS = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis/results'
SOURCE = RESULTS / 'public_mechanisms'
DOC = RESULTS / 'public_growth_uncertainty_v1'
OUT = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_public_growth_statistics_20261002_v1')
HELPER = ROOT / 'tools/scripts/run_safeconf_source_scaling_uncertainty_agent.py'
ORDERS = tuple(range(5))
BUDGETS = (.1, .25, .5, .75, 1.)
KEYS = list(itertools.product(ORDERS, BUDGETS))
META = ['task_id', 'target', 'gene', 'fold', 'upstream', 'true_error_rmse']
PAIRS = [(.25, .1), (.5, .25), (.75, .5), (1., .75), (1., .1)]
LINES = ('GAT_to_Exphormer', 'Exphormer_to_GAT', 'TxPert_to_McFaline')


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def json_once(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    began, cpu = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1200 seconds exceeded')))
    signal.alarm(1200)
    if OUT.exists() or (DOC / 'STATISTICS_REGISTRATION.json').exists():
        raise RuntimeError('Fresh statistics output required; no overwriting/restarting')
    inputs = [Path(__file__), HELPER, ROOT / 'tools/safeconf_continual/research.py',
              DOC / 'REGISTRATION.json', DOC / 'FIT_COMPLETION_RECEIPT.json']
    inputs += [SOURCE / n for n in ('GROWTH_TASK_PREDICTIONS.csv.gz', 'GROWTH_MACRO.csv',
                                   'GROWTH_STRATA.csv', 'GROWTH_COVERAGE.csv', 'GROWTH_BANK_AUDIT.csv')]
    before = [bind(path) for path in inputs]
    OUT.mkdir()
    json_once(DOC / 'STATISTICS_REGISTRATION.json', {
        'role': 'SEEN_fixed_growth_statistics_not_new_confirmation_or_method',
        'comparison_scope': 'HGB all fixed tasks; same draws across all budgets/orders and Source architectures',
        'bootstrap_replicates_per_study': 5000, 'bootstrap_seed': 20261002,
        'contrasts_a_minus_b': PAIRS, 'orders': ORDERS, 'budgets': BUDGETS,
        'macro': 'equal context mean then equal five fixed order mean; preserve NA, no invalid context/order omission',
        'distance_scope': 'per-order common-history point/invariance; all25snapshot common cohort reported even if empty',
        'new_fits': 0, 'GPU_hours': 0, 'raw_expression_reads': 0, 'no_Orion_access': True,
        'input_bindings': before, 'selection_status': 'descriptive SEEN followup; not pretruth algorithm selection'})
    spec = importlib.util.spec_from_file_location('fixed_count_statistics', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.BOOTSTRAP_SEED = 20261002  # audit label only; draws are created below
    data = pd.read_csv(SOURCE / 'GROWTH_TASK_PREDICTIONS.csv.gz', float_precision='round_trip')
    old = pd.read_csv(SOURCE / 'GROWTH_MACRO.csv', float_precision='round_trip')
    if set(data.line) != set(LINES) or set(data.method) != {'ManualHGB', 'ManualHistoryDistance'}:
        raise RuntimeError('Fixed full design differs')
    rng = np.random.default_rng(20261002)
    study_counts, study_genes = {}, {}
    curves, differences, order_points, common_rows, validity, reproductions = [], [], [], [], [], []
    common_invariance_max = 0.
    for line in LINES:
        rows = data[data.line.eq(line)]
        groups = dict(tuple(rows[rows.method.eq('ManualHGB')].groupby(['order', 'budget'], sort=True)))
        if set(groups) != set(KEYS):
            raise RuntimeError('All25registered HGB snapshots required')
        query, scores = None, []
        for key in KEYS:
            part = groups[key].sort_values('task_id').reset_index(drop=True)
            if not part.task_id.is_unique or not np.isfinite(part.risk).all():
                raise RuntimeError('HGB must evaluate the complete fixed population')
            if query is None:
                query = part[META].copy()
            if not part[META].equals(query):
                raise RuntimeError('All budgets must share exact tasks/true error')
            scores.append(part.risk.to_numpy(float))
        score_matrix = np.column_stack(scores)
        contexts = sorted(query.target.unique())
        study = 'Source' if line != 'TxPert_to_McFaline' else 'McFaline'
        genes = sorted(query.gene.unique())
        if study not in study_counts:
            study_genes[study] = genes
            counts = np.stack([np.bincount(rng.integers(0, len(genes), len(genes)), minlength=len(genes))
                               for _ in range(5000)]).astype(np.uint16)
            study_counts[study] = counts
            np.save(OUT / f'{study}_BOOTSTRAP_GENE_COUNTS.npy', counts)
        elif genes != study_genes[study]:
            raise RuntimeError('Same biological Source units must have identical draw mapping')
        counts = study_counts[study]
        gi = {gene: i for i, gene in enumerate(genes)}
        point = np.full((len(contexts) + 1, 25, 7), np.nan)
        draws = np.full((5000, len(contexts) + 1, 25, 7), np.nan)
        # Duplicate full-bank predictions retain all design labels but compute once.
        unique, mapping, lookup = [], [], {}
        for score in score_matrix.T:
            digest = hashlib.sha256(score.tobytes()).hexdigest()
            if digest not in lookup:
                lookup[digest] = len(unique)
                unique.append(score)
            mapping.append(lookup[digest])
        for ci, context in enumerate(contexts):
            mask = query.target.eq(context).to_numpy()
            q = query.loc[mask].reset_index(drop=True)
            weights = counts[:, [gi[g] for g in q.gene]]
            values, sampled = [], []
            for score in unique:
                scalar = np.array([metrics(q, score[mask])[k] for k in helper.METRICS])
                check = helper.counter_metrics(q.true_error_rmse.to_numpy(), q.task_id.to_numpy(str),
                                               score[mask], np.ones((1, len(q)), dtype=int))[0]
                if not np.allclose(scalar, check, rtol=0, atol=1e-12, equal_nan=True):
                    raise RuntimeError('Original scalar point must match expanded-count implementation')
                values.append(scalar)
                chunks = []
                for start in range(0, 5000, 500):
                    chunks.append(helper.counter_metrics(q.true_error_rmse.to_numpy(), q.task_id.to_numpy(str),
                                  score[mask], weights[start:start + 500], original_valid=len(q) >= 20))
                sampled.append(np.concatenate(chunks))
            point[ci] = np.array(values)[mapping]
            draws[:, ci] = np.stack(sampled, axis=1)[:, mapping]
        point[-1] = np.mean(point[:-1], axis=0)
        draws[:, -1] = np.mean(draws[:, :-1], axis=1)
        np.save(OUT / f'{line}_FIXED_HGB_METRIC_DRAWS.npy', draws)
        for index, (order, budget) in enumerate(KEYS):
            match = old[old.line.eq(line) & old.method.eq('ManualHGB') & old.order.eq(order) & old.budget.eq(budget)]
            if len(match) != 1 or not np.allclose(match[list(helper.METRICS)].to_numpy()[0], point[-1, index],
                                                rtol=0, atol=1e-12, equal_nan=True):
                raise RuntimeError('All75original HGB macro points must reproduce')
            reproductions.append({'line': line, 'order': order, 'budget': budget,
                'maximum_difference': float(np.max(np.abs(match[list(helper.METRICS)].to_numpy()[0] - point[-1, index])))})
        for ci, context in enumerate(contexts + ['macro']):
            averaged = {}
            for budget in BUDGETS:
                indices = [KEYS.index((order, budget)) for order in ORDERS]
                actual = np.mean(point[ci, indices], axis=0)
                averaged[budget] = np.mean(draws[:, ci, indices], axis=1)
                curves += helper.interval_records(actual, averaged[budget], {
                    'line': line, 'context': context, 'method': 'ManualHGB', 'budget': budget,
                    'orders': 5, 'n_fixed_tasks': len(query), 'n_gene_clusters': len(genes),
                    'bootstrap_seed_actual': 20261002, 'scope': 'fixed whole-task population, OOF concatenated within context'})
                for index in indices:
                    order_points.append({'line': line, 'context': context, 'order': KEYS[index][0],
                        'budget': budget, **dict(zip(helper.METRICS, point[ci, index]))})
            for a, b in PAIRS:
                ia = [KEYS.index((order, a)) for order in ORDERS]
                ib = [KEYS.index((order, b)) for order in ORDERS]
                actual = np.mean(point[ci, ia] - point[ci, ib], axis=0)
                differences += helper.interval_records(actual, averaged[a] - averaged[b], {
                    'line': line, 'context': context, 'method': 'ManualHGB', 'budget_a': a, 'budget_b': b,
                    'orders': 5, 'bootstrap_seed_actual': 20261002, 'sampling_unit': 'biological gene; not order or prediction row'})
        history = rows[rows.method.eq('ManualHistoryDistance')]
        wide = history.pivot(index=META, columns=['order', 'budget'], values='risk').reindex(columns=KEYS)
        common_all = np.isfinite(wide.to_numpy()).all(axis=1)
        common_rows.append({'line': line, 'scope': 'all25snapshots', 'n_tasks': int(common_all.sum()),
                            'order': '', 'context': 'all', 'delta_distance_allbudgets': 0. if common_all.any() else np.nan})
        for order in ORDERS:
            selected = wide.loc[:, [(order, b) for b in BUDGETS]]
            mask = np.isfinite(selected.to_numpy()).all(axis=1)
            sub = selected.loc[mask]
            frame = sub.reset_index()[META]
            gap = float(np.ptp(sub.to_numpy(), axis=1).max()) if len(sub) else np.nan
            if len(sub) and gap != 0:
                raise RuntimeError('Whole-gene inclusion must preserve existing-task direct distances')
            common_invariance_max = max(common_invariance_max, gap if len(sub) else 0)
            common_rows.append({'line': line, 'scope': 'perorder_all5budgets', 'order': order,
                'context': 'all', 'n_tasks': len(sub), 'delta_distance_allbudgets': gap})
            for context in contexts:
                local = frame.target.eq(context).to_numpy()
                value = metrics(frame.loc[local], sub.to_numpy()[local, 0])
                common_rows.append({'line': line, 'scope': 'perorder_all5budgets', 'order': order,
                    'context': context, 'delta_distance_allbudgets': gap, **value})
        for (method, order, budget, context, fold), group in rows.groupby(['method', 'order', 'budget', 'target', 'fold']):
            n = int(np.isfinite(group.risk).sum())
            validity.append({'line': line, 'method': method, 'order': order, 'budget': budget,
                'context': context, 'outer_fold': fold, 'n_planned': len(group), 'n_evaluable': n,
                'U20_original_n_at_least20': n >= 20, 'evaluation_scope': 'fold/context validity companion, not silently macro-selected'})
        print(json.dumps({'completed_statistics_line': line, 'fixed_tasks': len(query), 'genes': len(genes)}), flush=True)
    for name, records in [('FIXED_TASK_HGB_CURVE_INTERVALS.csv', curves), ('FIXED_TASK_HGB_GROWTH_CONTRASTS.csv', differences),
                          ('FIXED_TASK_ORDER_POINTS.csv', order_points), ('DISTANCE_COMMON_COHORT_INVARIANCE.csv', common_rows),
                          ('FOLD_CONTEXT_AVAILABILITY.csv', validity), ('ORIGINAL_HGB_POINT_REPRODUCTION.csv', reproductions)]:
        pd.DataFrame(records).to_csv(DOC / name, index=False)
    if before != [bind(path) for path in inputs]:
        raise RuntimeError('Original code or fixed outputs changed during statistical analysis')
    result = {'status': 'COMPLETE_FIXED_SEEN_PUBLIC_COVERAGE_STATISTICS', 'new_fits': 0,
        'replicates_per_study': 5000, 'biological_studies_resampled': 2, 'independent_Source_families': 1,
        'original_HGB_macro_points_reproduced': len(reproductions),
        'same_gene_direct_distance_max_difference': common_invariance_max,
        'new_confirmation': False, 'new_methods': 0, 'GPU_hours': 0, 'no_Orion_access': True,
        'elapsed_seconds': time.monotonic() - began, 'CPU_seconds': time.process_time() - cpu,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'original_input_bindings': before, 'outputs': [bind(p) for p in sorted(DOC.glob('*.csv'))]}
    json_once(DOC / 'STATISTICS_RESULT_MANIFEST.json', result)
    print(json.dumps({'status': result['status'], 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
