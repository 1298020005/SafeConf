"""Fixed SEEN E170 strong-reference completion; no fitting or truth reopening."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pathlib import Path
import sys
import json
import hashlib
import time
import resource
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import metrics, ids_hash
from tools.scripts.run_safeconf_source_scaling_uncertainty_agent import counter_metrics, METRICS

OLD = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/confirmation/e170_primary_cd4_four_panel'
REPORT = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/e170_legacy_strong_references_v1'
RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/e170_legacy_strong_references_20261002_v1')
OLD_METHODS = ['Magnitude_raw', 'Ridge_U', 'Ridge_US', 'Ridge_USR', 'Ridge_USRCH', 'V2_nested']
RULES = ['DirectHistoryDistance', 'DistancePlusHistoryDispersion', 'NegativeLogHistorySupport']
COHORTS = ['FULL_ORIGINAL_2400', 'KNOWN_HISTORY_1920', 'NO_HISTORY_480']
META = ['task_id', 'panel_id', 'culture_condition', 'gene', 'target_stratum']


def bind(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    started, cpu = time.monotonic(), time.process_time()
    REPORT.mkdir(parents=True, exist_ok=False)
    RUNTIME.mkdir(parents=True, exist_ok=False)
    inputs = [p for p in OLD.iterdir() if p.is_file()] + [Path(__file__),
              ROOT / 'tools/safeconf_continual/research.py',
              ROOT / 'tools/scripts/run_safeconf_source_scaling_uncertainty_agent.py',
              ROOT / 'tools/scripts/run_e170_safeconf_v4_confirmation.py',
              ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/E170_SAFECONF_V4_CONFIRMATION_CONTRACT.md',
              ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/E170_V4_CONFIRMATION_AUTHORIZATION.json']
    pins = [bind(p) for p in inputs]
    write_json(REPORT / 'REGISTRATION.json', {
        'role': 'SEEN_after_original_confirmation_strong_comparator_completion',
        'fixed_rules': {'DirectHistoryDistance': '-negative_model_source_gap',
                        'DistancePlusHistoryDispersion': 'sqrt(gap**2 + source_delta_dispersion**2)',
                        'NegativeLogHistorySupport': '-log1p(n_source_cells)'},
        'old_scores_not_refitted': OLD_METHODS, 'cohorts': COHORTS,
        'primary_old_gate_unchanged': True, 'undefined_direct_and_uncertainty_scores_remain_NA': True,
        'support_full_population_extra_control': 'n_source_cells=0 is observed absence of history, not a zero effect; -log1p(0) is finite. No-history score is constant, so this arm has no ranking information there.',
        'bootstrap': {'draws': 5000, 'unit': 'whole gene across three states',
                      'panels': 4, 'independent_cohort_seeds': [20261002, 20261003, 20261004]},
        'fixed_contrasts': 'V2 minus every other available method; known-history direct minus magnitude, uncertainty minus direct, support minus magnitude',
        'new_fits': 0, 'new_upstream_calls': 0,
        'information': 'old risk learners consumed 1920 target ensemble validation errors; rules and magnitude consume zero error labels',
        'gene_generalization': 'known-history640genes also have target validation errors; heldout donor within same study, not unseen-gene source-only transfer',
        'input_code_bindings': pins})
    feature = pd.read_csv(OLD / 'CONFIRMATION_FEATURES.csv.gz')
    scores = pd.read_csv(OLD / 'CONFIRMATION_PREDICTIONS.csv.gz', usecols=META + ['method', 'predicted_risk'])
    if (len(feature) != 2400 or feature.task_id.nunique() != 2400 or feature.gene.nunique() != 800
            or feature.perturbed_gene_id.nunique() != 800
            or not feature.groupby('perturbed_gene_id').panel_id.nunique().eq(1).all()
            or len(scores) != 14400 or set(scores.method) != set(OLD_METHODS)):
        raise RuntimeError('Original full identity cohort or frozen methods changed')
    wide = scores.pivot(index='task_id', columns='method', values='predicted_risk')
    frame = feature[META].copy()
    for method in OLD_METHODS:
        frame[method] = frame.task_id.map(wide[method])
    known = feature.target_stratum.eq('DONOR_UNSEEN_ONLY')
    missing = feature.target_stratum.eq('COLUMN_UNSEEN')
    if (known.sum() != 1920 or missing.sum() != 480 or not (known | missing).all()
            or not feature.loc[known, 'n_source_contexts'].eq(2).all()
            or not feature.loc[missing, 'n_source_contexts'].eq(0).all()):
        raise RuntimeError('Frozen known/no-history counts differ')
    gap = -feature.negative_model_source_gap.to_numpy(float)
    sigma = feature.source_delta_dispersion.to_numpy(float)
    if (not np.isfinite(gap[known]).all() or not np.isfinite(sigma[known]).all()
            or (gap[known] < 0).any() or (sigma[known] < 0).any()
            or not np.isnan(gap[missing]).all() or not np.isnan(sigma[missing]).all()):
        raise RuntimeError('History fields are not valid or missing scores were filled')
    frame[RULES[0]] = gap
    frame[RULES[1]] = np.sqrt(gap**2 + sigma**2)
    frame[RULES[2]] = -np.log1p(feature.n_source_cells.to_numpy(float))
    if not np.array_equal(frame.Magnitude_raw.to_numpy(), feature.predicted_magnitude.to_numpy()):
        raise RuntimeError('Original magnitude scores differ from frozen feature cache')
    frame.to_parquet(RUNTIME / 'ALL2400_FROZEN_SUPPLEMENTAL_SCORES.parquet', index=False)
    write_json(REPORT / 'SCORE_FREEZE.json', {
        'score_binding': bind(RUNTIME / 'ALL2400_FROZEN_SUPPLEMENTAL_SCORES.parquet'),
        'frozen_before_cached_error_numeric_parse': True,
        'original_query_tasks': 2400, 'all_original_methods_retained': True,
        'each_direct_history_rule_undefined_tasks': 480})
    print(json.dumps({'phase': 'scores_frozen', 'pid': os.getpid(), 'new_fits': 0}), flush=True)
    truth = pd.read_csv(OLD / 'CONFIRMATION_PREDICTIONS.csv.gz', usecols=['task_id', 'true_error_rmse'])
    if not truth.groupby('task_id').true_error_rmse.nunique().eq(1).all():
        raise RuntimeError('Original methods have inconsistent query error')
    frame['true_error_rmse'] = frame.task_id.map(truth.drop_duplicates('task_id').set_index('task_id').true_error_rmse)
    if not np.isfinite(frame.true_error_rmse).all():
        raise RuntimeError('Original cached truth is incomplete')
    original = pd.read_csv(OLD / 'STRATUM_RESULTS.csv')
    original_full = pd.read_csv(OLD / 'SUMMARY.csv').set_index('method')
    method_rows, intervals, strata_rows, invalid = [], [], [], []
    selections = [np.ones(2400, bool), known.to_numpy(), missing.to_numpy()]
    for ci, (cohort, mask) in enumerate(zip(COHORTS, selections)):
        part = frame.loc[mask].reset_index(drop=True)
        methods = OLD_METHODS + (RULES if ci == 1 else [RULES[2]])
        if part.task_id.nunique() != [2400, 1920, 480][ci] or part.gene.nunique() != [800, 640, 160][ci]:
            raise RuntimeError('Predetermined cohort differs')
        rng = np.random.default_rng(20261002 + ci)
        counts, panels = {}, {}
        for panel, group in part.groupby('panel_id', sort=True):
            genes = sorted(group.gene.unique())
            if len(genes) != [200, 160, 40][ci]:
                raise RuntimeError('Original panel gene cluster count differs')
            counts[panel] = np.asarray([np.bincount(rng.integers(0, len(genes), len(genes)), minlength=len(genes))
                                       for _ in range(5000)], dtype=np.uint16)
            panels[panel] = genes
        np.savez_compressed(RUNTIME / f'{cohort}_BOOTSTRAP_COUNTS.npz',
                            **{f'{p}_counts': counts[p] for p in panels},
                            **{f'{p}_genes': np.asarray(panels[p]) for p in panels})
        points, draws = [], []
        for (panel, state), group in part.groupby(['panel_id', 'culture_condition'], sort=True):
            group = group.reset_index(drop=True)
            mapping = {g: row for row, g in enumerate(panels[panel])}
            multiplicity = counts[panel][:, group.gene.map(mapping).to_numpy(int)].astype(np.int32)
            point, draw = [], []
            for method in methods:
                risk = group[method].to_numpy(float)
                measured = metrics(group, risk)
                vector = np.asarray([measured[k] for k in METRICS])
                numerical = counter_metrics(group.true_error_rmse, group.task_id, risk, np.ones((1, len(group)), int))[0]
                if not np.allclose(vector, numerical, rtol=0, atol=1e-12, equal_nan=True):
                    raise RuntimeError('Count metric differs from frozen scalar definition')
                if ci == 0 and method in OLD_METHODS:
                    row = original[original.panel_id.eq(panel) & original.culture_condition.eq(state) & original.method.eq(method)]
                    for key in METRICS:
                        old_key = key.replace('error_at_', 'risk_at_')
                        if not np.isclose(measured[key], row.iloc[0][old_key], rtol=0, atol=1e-12, equal_nan=True):
                            raise RuntimeError('Original full-cohort point changed')
                point.append(vector)
                draw.append(counter_metrics(group.true_error_rmse, group.task_id, risk, multiplicity))
                strata_rows.append({'cohort': cohort, 'panel': panel, 'state': state, 'method': method,
                                    'n_tasks': len(group), 'n_gene_clusters': group.gene.nunique(), **measured})
            points.append(point)
            draws.append(np.stack(draw, axis=1))
        macro_point, macro_draw = np.mean(points, axis=0), np.mean(draws, axis=0)
        np.savez_compressed(RUNTIME / f'{cohort}_METRIC_DRAWS.npz',
                            points=macro_point, draws=macro_draw, methods=np.asarray(methods), metrics=np.asarray(METRICS))
        for mi, method in enumerate(methods):
            method_rows.append({'cohort': cohort, 'method': method, 'tasks': len(part),
                                'gene_clusters': part.gene.nunique(), 'strata': 12,
                                **dict(zip(METRICS, macro_point[mi]))})
            if ci == 0 and method in OLD_METHODS:
                assert np.isclose(macro_point[mi, 0], original_full.loc[method, 'utility20'], atol=1e-12, rtol=0)
        comparisons = [('V2_nested', method) for method in methods if method != 'V2_nested']
        if ci == 1:
            comparisons += [(RULES[0], 'Magnitude_raw'), (RULES[1], RULES[0]), (RULES[2], 'Magnitude_raw')]
        for a, b in comparisons:
            ai, bi = methods.index(a), methods.index(b)
            delta = macro_draw[:, ai] - macro_draw[:, bi]
            for ki, key in enumerate(METRICS):
                finite = delta[:, ki][np.isfinite(delta[:, ki])]
                lower, upper = np.quantile(finite, [.025, .975]) if len(finite) >= 2 else (np.nan, np.nan)
                intervals.append({'cohort': cohort, 'method_a': a, 'method_b': b, 'metric': key,
                                  'point_difference': macro_point[ai, ki] - macro_point[bi, ki],
                                  'ci95_lower': lower, 'ci95_upper': upper, 'valid_draws': len(finite), 'total_draws': 5000})
                if len(finite) != 5000:
                    invalid.append({'cohort': cohort, 'comparison': f'{a}-{b}', 'metric': key,
                                    'valid_draws': len(finite), 'reason': 'undefined metric retained; not imputed'})
        print(json.dumps({'phase': 'cohort_complete', 'cohort': cohort, 'methods': len(methods)}), flush=True)
    for name, rows in [('ALL_COHORT_MACRO.csv', method_rows), ('ALL_COHORT_STRATA.csv', strata_rows),
                       ('ALL_FIXED_PAIRED_INTERVALS.csv', intervals), ('INVALID_METRIC_DRAWS.csv', invalid)]:
        pd.DataFrame(rows).to_csv(REPORT / name, index=False, lineterminator='\n')
    for pin in pins:
        if bind(pin['path']) != pin:
            raise RuntimeError('Original input, result or code changed')
    write_json(REPORT / 'RESULT_MANIFEST.json', {
        'status': 'COMPLETE_FIXED_SEEN_E170_STRONG_RULES', 'new_fits': 0, 'new_upstream_calls': 0,
        'original_full_points_reproduced': True, 'original_gate_and_predictions_unchanged': True,
        'macro_rows': len(method_rows), 'strata_rows': len(strata_rows), 'paired_rows': len(intervals),
        'rule_missing_tasks_retained': 480, 'new_risk_training_labels': 0,
        'original_supervised_methods_target_validation_labels': 1920,
        'elapsed_seconds': time.monotonic() - started, 'CPU_delta_seconds': time.process_time() - cpu,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'GPU_hours': 0, 'runtime_artifacts': [bind(p) for p in RUNTIME.iterdir() if p.is_file()]})


if __name__ == '__main__':
    main()
