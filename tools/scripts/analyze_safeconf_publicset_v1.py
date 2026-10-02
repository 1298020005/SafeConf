#!/usr/bin/env python3
"""Paired gene-cluster statistics for completed PublicSet development runs.

The primary estimand computes each seed's metrics, averages seeds within each
context x outer fold, then averages strata. It never averages risk scores.
Repeated genes are evaluated with integer multiplicities, exactly equivalent
to repeating their tasks, including partially selected top-k tie blocks.
"""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '4'

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import warnings

import numpy as np
import pandas as pd

from run_safeconf_publicset_v1 import atomic_csv, atomic_json, metric_values

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1'
SEEDS = (20260930, 20261001, 20261002)
BUILDERS = ('B0_SupportMean', 'B1_HGB', 'B2_Pointwise', 'B3_DeepSets', 'Magnitude', 'NegativeHistorySupport')
STRONG = ('NearestControl', 'SameContextSupport', 'SameConditionSupport')
ALL_BUILDERS = BUILDERS + STRONG
NEURAL = ('B2_Pointwise', 'B3_DeepSets')
OLD = 'AlignmentControl_OldGuideMean'
KEYS = ['fold', 'context', 'task_id', 'gene']
METRICS = ('utility20', 'aurc', 'high_risk_miss_rate', 'error_at_10', 'error_at_20', 'error_at_50', 'bio_rmse', 'bio_cosine')
READOUTS = ('risk', 'direct_risk')
COMPARISONS = (('B2_Pointwise', 'B1_HGB'), ('B3_DeepSets', 'B1_HGB'), ('B3_DeepSets', 'B2_Pointwise'),
               ('B2_Pointwise', 'B0_SupportMean'), ('B3_DeepSets', 'B0_SupportMean'),
               ('B2_Pointwise', 'Magnitude'), ('B3_DeepSets', 'Magnitude'),
               ('B2_Pointwise', 'NegativeHistorySupport'), ('B3_DeepSets', 'NegativeHistorySupport'))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def nanmean(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(values, axis=axis)


def selected_counts(weights, order, k):
    ordered = weights[:, order]
    before = np.cumsum(ordered, axis=1) - ordered
    return np.minimum(ordered, np.maximum(0, k[:, None] - before))


def weighted_metrics(frame, multiplicities, score_col='risk'):
    """Vectorized exact repeat metrics; each row in multiplicities is one draw."""
    ids = frame.task_id.astype(str).to_numpy()
    error = frame.true_error_rmse.to_numpy(float)
    score = frame[score_col].to_numpy(float)
    weights = np.asarray(multiplicities, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[1] != len(frame) or (weights < 0).any():
        raise ValueError('Invalid task multiplicities')
    if not np.isfinite(np.c_[error, score, frame.bio_rmse, frame.bio_cosine]).all():
        raise ValueError('Metrics require finite paired records')
    n = weights.sum(1)
    safe_n = np.maximum(n, 1)
    k = np.maximum(1, np.ceil(.2 * n).astype(int))
    high = np.lexsort((ids, -score))
    low = np.lexsort((ids, score))
    oracle = np.lexsort((ids, -error))
    selected = selected_counts(weights, high, k)
    ideal = selected_counts(weights, oracle, k)
    average = (weights @ error) / safe_n
    denominator = (ideal @ error[oracle]) / k - average
    utility = np.divide((selected @ error[high]) / k - average, denominator,
                        out=np.full(len(weights), np.nan), where=(n >= 20) & (denominator > 1e-12))
    selected_original = np.empty_like(selected)
    selected_original[:, high] = selected
    ideal_original = np.empty_like(ideal)
    ideal_original[:, oracle] = ideal
    miss = 1 - np.minimum(selected_original, ideal_original).sum(1) / k
    # Sum the cumulative means for all repeated task copies without expansion.
    ordered = weights[:, low]
    count = np.cumsum(ordered, axis=1)
    before_count = count - ordered
    cumulative_error = np.cumsum(ordered * error[low], axis=1)
    before_error = cumulative_error - ordered * error[low]
    harmonic = np.r_[0., np.cumsum(1. / np.arange(1, max(1, int(n.max())) + 1))]
    contribution = ordered * error[low] + (before_error - before_count * error[low]) * (harmonic[count] - harmonic[before_count])
    aurc = contribution.sum(1) / safe_n
    output = [utility, aurc, miss]
    for coverage in (.1, .2, .5):
        low_k = np.maximum(1, np.ceil(coverage * n).astype(int))
        output.append((selected_counts(weights, low, low_k) @ error[low]) / low_k)
    output.extend(((weights @ frame.bio_rmse.to_numpy(float)) / safe_n,
                   (weights @ frame.bio_cosine.to_numpy(float)) / safe_n))
    result = np.column_stack(output)
    result[n == 0] = np.nan
    return result


def semantic_check():
    """Check integer multiplicities, score ties, partial ties, and small strata."""
    rng = np.random.default_rng(20261002)
    checked = 0
    for size in (7, 20, 31):
        frame = pd.DataFrame({'task_id': [f't{i:03d}' for i in range(size)],
                              'gene': [f'g{i // 2}' for i in range(size)],
                              'risk': rng.integers(0, 4, size), 'true_error_rmse': rng.integers(1, 8, size) / 10,
                              'bio_rmse': rng.random(size), 'bio_cosine': rng.random(size)})
        for _ in range(20):
            counts = rng.integers(0, 5, size)
            if not counts.any():
                counts[0] = 1
            literal = frame.iloc[np.repeat(np.arange(size), counts)].reset_index(drop=True)
            expected = metric_values(literal)
            actual = weighted_metrics(frame, counts[None])[0]
            for j, name in enumerate(METRICS[:6]):
                if not np.isclose(actual[j], expected.get(name, np.nan), atol=2e-12, rtol=2e-12, equal_nan=True):
                    raise AssertionError((name, counts.tolist(), actual[j], expected.get(name)))
            np.testing.assert_allclose(actual[6:], [literal.bio_rmse.mean(), literal.bio_cosine.mean()], atol=2e-12)
            checked += 1
    return {'status': 'PASS', 'literal_repeat_cases': checked, 'ties': True, 'partial_topk_blocks': True,
            'checks': list(METRICS), 'maximum_tolerance': 2e-12}


def scope_name(domain, upstream):
    return f'{domain}/{upstream}'


def validate_and_align(frame, run_dir, strong_frame):
    required = set(KEYS + ['domain', 'upstream', 'builder', 'seed', *READOUTS, 'true_error_rmse', 'bio_rmse', 'bio_cosine'])
    if required - set(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    if set(strong_frame.builder) != set(STRONG):
        raise ValueError('All three fixed strong simple references required')
    frame = pd.concat([frame, strong_frame], ignore_index=True)
    for name in ('domain', 'upstream', 'builder', 'context', 'task_id', 'gene'):
        frame[name] = frame[name].astype(str)
    frame['seed'] = frame.seed.astype(int)
    frame['fold'] = frame.fold.astype(int)
    if frame.duplicated(['domain', 'upstream', 'builder', 'seed', *KEYS]).any():
        raise ValueError('Duplicate prediction keys')
    if frame.groupby(['domain', 'gene']).fold.nunique().max() != 1:
        raise ValueError('A gene crosses outer folds')
    domains = set(frame.domain)
    if domains != {'Source', 'McFaline'}:
        raise ValueError(f'Expected both complete domains, found {domains}')
    expected = {}
    for domain, n_tasks, n_genes in [('Source', 1808, 575), ('McFaline', 542, 377)]:
        tasks = pd.read_csv(run_dir / f'{domain}_QUERY_SPLIT.csv')
        if domain == 'McFaline':
            tasks = tasks.rename(columns={'perturbation': 'gene'})
        if len(tasks) != n_tasks or tasks.gene.nunique() != n_genes or set(tasks.fold) != set(range(5)):
            raise ValueError(f'Registered {domain} complete task set changed')
        tasks['task_id'] = tasks.task_id.astype(str)
        tasks['gene'] = tasks.gene.astype(str)
        tasks['context'] = tasks.context.astype(str)
        expected[domain] = tasks.set_index(KEYS).sort_index()
    coverage = []
    aligned = {}
    for (domain, upstream), group in frame.groupby(['domain', 'upstream'], sort=True):
        required_arms = ALL_BUILDERS + ((OLD,) if domain == 'McFaline' else ())
        arm_frames = {}
        common = expected[domain].index
        for builder in required_arms:
            for seed in (SEEDS if builder in NEURAL else (0,)):
                arm = group[(group.builder == builder) & (group.seed == seed)].set_index(KEYS).sort_index()
                arm_frames[builder, seed] = arm
                common = common.intersection(arm.index)
        for (builder, seed), arm in arm_frames.items():
            coverage.append({'domain': domain, 'upstream': upstream, 'builder': builder, 'seed': seed,
                             'expected_tasks': len(expected[domain]), 'available_tasks': len(arm), 'paired_common_tasks': len(common),
                             'available_genes': arm.reset_index().gene.nunique(), 'common_genes': len(set(common.get_level_values('gene'))),
                             'missing_expected_tasks': len(expected[domain].index.difference(arm.index)),
                             'unexpected_tasks': len(arm.index.difference(expected[domain].index)),
                             'common_task_coverage': len(common) / len(expected[domain])})
            if len(arm.index.difference(expected[domain].index)):
                raise ValueError('Unexpected prediction task')
        if not len(common):
            raise ValueError(f'No common evaluation tasks for {domain}/{upstream}')
        views = {key: arm.loc[common].reset_index() for key, arm in arm_frames.items()}
        reference = views['B0_SupportMean', 0]
        for key, arm in views.items():
            if not np.allclose(arm.true_error_rmse, reference.true_error_rmse, atol=1e-12, rtol=0):
                raise ValueError(f'Outcome changed across methods: {domain}/{upstream}/{key}')
        aligned[domain, upstream] = views
    if set(aligned) != {('Source', 'TxPert_GAT'), ('Source', 'TxPert_Exphormer'), ('McFaline', 'DecoderOnly')}:
        raise ValueError('Complete registered upstream scopes changed')
    return aligned, pd.DataFrame(coverage), expected


def make_counts(expected, replicates, seed):
    children = np.random.SeedSequence(seed).spawn(2)
    counts, genes = {}, {}
    for domain, child in zip(('Source', 'McFaline'), children):
        genes[domain] = sorted(set(expected[domain].index.get_level_values('gene')))
        size = len(genes[domain])
        rng = np.random.default_rng(child)
        # One joint gene draw for all Source upstreams/seeds/outer folds.
        counts[domain] = rng.multinomial(size, np.full(size, 1 / size), size=replicates).astype(np.int16)
    return counts, genes


def evaluate(aligned, expected, counts, genes, batch, deadline):
    point_rows, seed_rows = [], []
    draws, points, strata_by_scope, valid_masks = {}, {}, {}, {}
    seed_draws = {}
    for (domain, upstream), arms in aligned.items():
        scope = scope_name(domain, upstream)
        full_strata = sorted(set(zip(expected[domain].index.get_level_values('fold'), expected[domain].index.get_level_values('context'))))
        strata_by_scope[scope] = full_strata
        gene_lookup = {g: i for i, g in enumerate(genes[domain])}
        for readout in READOUTS:
            for builder in sorted(set(k[0] for k in arms)):
                seeds = SEEDS if builder in NEURAL else (0,)
                method_points, method_draws = [], []
                for seed in seeds:
                    arm = arms[builder, seed]
                    seed_point, seed_boot = [], []
                    for fold, context in full_strata:
                        group = arm[(arm.fold == fold) & (arm.context == context)].reset_index(drop=True)
                        if len(group):
                            metric = metric_values(group, readout)
                            p = weighted_metrics(group, np.ones((1, len(group)), dtype=int), readout)[0]
                            indices = np.asarray([gene_lookup[g] for g in group.gene])
                            pieces = []
                            for first in range(0, len(counts[domain]), batch):
                                if time.monotonic() > deadline:
                                    raise TimeoutError('Statistics CPU wall budget reached')
                                pieces.append(weighted_metrics(group, counts[domain][first:first + batch, indices], readout))
                            boot = np.concatenate(pieces)
                        else:
                            metric = {'n_tasks': 0, 'n_clusters': 0, 'spearman': np.nan}
                            p = np.full(len(METRICS), np.nan)
                            boot = np.full((len(counts[domain]), len(METRICS)), np.nan)
                        seed_point.append(p)
                        seed_boot.append(boot)
                        seed_rows.append({'scope': scope, 'domain': domain, 'upstream': upstream, 'readout': readout,
                                          'builder': builder, 'seed': seed, 'fold': fold, 'context': context,
                                          **metric, **dict(zip(METRICS, p))})
                    seed_point = np.asarray(seed_point)
                    seed_boot = np.asarray(seed_boot)
                    valid = np.isfinite(seed_point[:, 0])
                    seed_macro = nanmean(seed_point[valid]) if valid.any() else np.full(len(METRICS), np.nan)
                    seed_rows.append({'scope': scope, 'domain': domain, 'upstream': upstream, 'readout': readout,
                                      'builder': builder, 'seed': seed, 'fold': 'MACRO', 'context': 'MACRO',
                                      'n_tasks': len(arm), 'n_clusters': arm.gene.nunique(),
                                      'valid_strata': int(valid.sum()), 'planned_strata': len(full_strata),
                                      **dict(zip(METRICS, seed_macro))})
                    seed_draws[scope, readout, builder, seed] = nanmean(seed_boot[valid], axis=0)
                    method_points.append(seed_point)
                    method_draws.append(seed_boot)
                stratum_point = nanmean(np.asarray(method_points), axis=0)
                stratum_draws = nanmean(np.asarray(method_draws), axis=0)
                valid = np.isfinite(stratum_point[:, 0])
                # Keep the planned point-valid strata fixed; invalidate very sparse draws.
                macro = nanmean(stratum_point[valid]) if valid.any() else np.full(len(METRICS), np.nan)
                macro_draws = nanmean(stratum_draws[valid], axis=0)
                draw_valid_strata = np.isfinite(stratum_draws[valid, :, 0]).sum(0)
                macro_draws[draw_valid_strata < math.ceil(.8 * len(full_strata))] = np.nan
                key = scope, readout, builder
                points[key] = macro
                draws[key] = macro_draws
                valid_masks[key] = valid
                for i, (fold, context) in enumerate(full_strata):
                    point_rows.append({'scope': scope, 'domain': domain, 'upstream': upstream, 'readout': readout,
                                       'builder': builder, 'fold': fold, 'context': context, 'seed_count': len(seeds),
                                       'valid': bool(valid[i]), **dict(zip(METRICS, stratum_point[i]))})
                print(json.dumps({'phase': 'statistics', 'scope': scope, 'readout': readout, 'builder': builder,
                                  'valid_strata': int(valid.sum()), 'planned_strata': len(full_strata)}), flush=True)
    # Upstreams are dependent views. Preserve that dependence in all aggregates.
    for scope, members in [('Source/MACRO', [('Source/TxPert_GAT', .5), ('Source/TxPert_Exphormer', .5)]),
                           ('ALL/DOMAIN_MACRO', [('Source/TxPert_GAT', .25), ('Source/TxPert_Exphormer', .25), ('McFaline/DecoderOnly', .5)])]:
        for readout in READOUTS:
            for builder in ALL_BUILDERS:
                key = scope, readout, builder
                points[key] = sum(weight * points[member, readout, builder] for member, weight in members)
                draws[key] = sum(weight * draws[member, readout, builder] for member, weight in members)
                valid_masks[key] = np.concatenate([valid_masks[member, readout, builder] for member, _ in members])
        strata_by_scope[scope] = [(member, fold, context) for member, _ in members for fold, context in strata_by_scope[member]]
    return pd.DataFrame(point_rows), pd.DataFrame(seed_rows), points, draws, valid_masks, strata_by_scope, seed_draws


def intervals(points, draws, valid_masks):
    rows = []
    for (scope, readout, builder), point in points.items():
        values = draws[scope, readout, builder]
        lower, upper = np.nanquantile(values, [.025, .975], axis=0)
        for i, metric in enumerate(METRICS):
            rows.append({'scope': scope, 'readout': readout, 'builder': builder, 'metric': metric,
                         'value': point[i], 'ci95_lower': lower[i], 'ci95_upper': upper[i],
                         'valid_strata': int(valid_masks[scope, readout, builder].sum()),
                         'planned_strata': len(valid_masks[scope, readout, builder]),
                         'bootstrap_valid_draws': int(np.isfinite(values[:, i]).sum()), 'bootstrap_replicates': len(values)})
    return pd.DataFrame(rows)


def paired_results(points, draws, valid_masks, strata, alignment=False, strong=False):
    rows, stratum_rows = [], []
    scopes = sorted(set(k[0] for k in points))
    for scope in scopes:
        if alignment and scope != 'McFaline/DecoderOnly':
            continue
        for readout in READOUTS:
            pairs = ([('B0_SupportMean', OLD)] if alignment else
                     [(candidate, comparator) for candidate in NEURAL for comparator in STRONG] if strong else COMPARISONS)
            for candidate, comparator in pairs:
                a = points[scope, readout, candidate]
                b = points[scope, readout, comparator]
                delta = a - b
                paired_draws = draws[scope, readout, candidate] - draws[scope, readout, comparator]
                lower, upper = np.nanquantile(paired_draws, [.025, .975], axis=0)
                if scope in ('Source/MACRO', 'ALL/DOMAIN_MACRO'):
                    members = ['Source/TxPert_GAT', 'Source/TxPert_Exphormer'] + (['McFaline/DecoderOnly'] if scope.startswith('ALL/') else [])
                    sub = strata[(strata.scope.isin(members)) & (strata.readout == readout)]
                else:
                    sub = strata[(strata.scope == scope) & (strata.readout == readout)]
                x = sub[sub.builder == candidate].set_index(['scope', 'fold', 'context'])
                y = sub[sub.builder == comparator].set_index(['scope', 'fold', 'context'])
                merged = x[list(METRICS)].join(y[list(METRICS)], lsuffix='_candidate', rsuffix='_comparator')
                valid = merged.utility20_candidate.notna() & merged.utility20_comparator.notna()
                valid_n = int(valid.sum())
                planned = len(merged)
                nonnegative = float(((merged.utility20_candidate - merged.utility20_comparator)[valid] >= -1e-12).mean()) if valid_n else np.nan
                safety = {m: float((a[METRICS.index(m)] - b[METRICS.index(m)]) / b[METRICS.index(m)])
                          if b[METRICS.index(m)] > 1e-12 else np.nan for m in ('aurc', 'error_at_10', 'error_at_20', 'error_at_50')}
                miss_delta = float(delta[METRICS.index('high_risk_miss_rate')])
                safe = all(np.isfinite(v) and v <= .05 + 1e-12 for v in safety.values()) and miss_delta <= .02 + 1e-12
                stable = bool(lower[0] >= -.005)
                actual_gain = bool(delta[0] >= .005 - 1e-12)
                validity_pass = planned > 0 and valid_n / planned >= .8
                strata_pass = bool(nonnegative >= .6 - 1e-12)
                gate = actual_gain and stable and validity_pass and strata_pass and safe
                for i, metric in enumerate(METRICS):
                    rows.append({'scope': scope, 'readout': readout, 'candidate': candidate, 'comparator': comparator,
                                 'metric': metric, 'candidate_value': a[i], 'comparator_value': b[i], 'delta': delta[i],
                                 'ci95_lower': lower[i], 'ci95_upper': upper[i],
                                 'bootstrap_valid_draws': int(np.isfinite(paired_draws[:, i]).sum()),
                                 'bootstrap_replicates': len(paired_draws), 'valid_strata': valid_n, 'planned_strata': planned,
                                 'valid_strata_fraction': valid_n / planned if planned else np.nan,
                                 'nonnegative_strata_fraction': nonnegative,
                                 'actual_gain_ge_0_005': actual_gain, 'ci_lower_ge_minus_0_005': stable,
                                 'nonnegative_ge_60pct': strata_pass, 'valid_ge_80pct': validity_pass,
                                 'macro_safety_pass': safe, 'adoption_gate_pass': gate,
                                 'aurc_relative_change': safety['aurc'], 'error_at_10_relative_change': safety['error_at_10'],
                                 'error_at_20_relative_change': safety['error_at_20'], 'error_at_50_relative_change': safety['error_at_50'],
                                 'miss_rate_change': miss_delta,
                                 'role': 'DATA_ALIGNMENT_CONTROL' if alignment else 'FIXED_STRONG_SIMPLE_COMPARISON' if strong else 'DEV_SEEN_PAIRED_COMPARISON'})
                for idx, row in merged.iterrows():
                    changes = {m: row[f'{m}_candidate'] - row[f'{m}_comparator'] for m in METRICS}
                    stratum_safety = {m: changes[m] / row[f'{m}_comparator'] if row[f'{m}_comparator'] > 1e-12 else np.nan for m in safety}
                    stratum_rows.append({'summary_scope': scope, 'scope': idx[0], 'fold': idx[1], 'context': idx[2],
                                         'readout': readout, 'candidate': candidate, 'comparator': comparator,
                                         'valid': bool(np.isfinite(changes['utility20'])), **{f'delta_{m}': v for m, v in changes.items()},
                                         **{f'relative_change_{m}': v for m, v in stratum_safety.items()},
                                         'stratum_safety_pass': all(np.isfinite(v) and v <= .05 + 1e-12 for v in stratum_safety.values())
                                         and changes['high_risk_miss_rate'] <= .02 + 1e-12})
    return pd.DataFrame(rows), pd.DataFrame(stratum_rows)


def decisions(paired, strong_paired):
    utility = paired[paired.metric == 'utility20']
    primitive = ['Source/TxPert_GAT', 'Source/TxPert_Exphormer', 'McFaline/DecoderOnly']
    rows = []
    for readout in READOUTS:
        for builder in NEURAL:
            comparisons = utility[(utility.readout == readout) & (utility.candidate == builder)
                                  & utility.scope.isin(primitive)
                                  & utility.comparator.isin(['B0_SupportMean', 'B1_HGB', 'Magnitude', 'NegativeHistorySupport'])]
            strong_comparisons = strong_paired[(strong_paired.metric == 'utility20') & (strong_paired.readout == readout)
                                              & (strong_paired.candidate == builder) & strong_paired.scope.isin(primitive)]
            core_passed = len(comparisons) == 12 and comparisons.adoption_gate_pass.all()
            strong_passed = len(strong_comparisons) == 9 and strong_comparisons.adoption_gate_pass.all()
            passed = core_passed and strong_passed
            structure = utility[(utility.readout == readout) & (utility.candidate == 'B3_DeepSets')
                                & (utility.comparator == 'B2_Pointwise') & utility.scope.isin(primitive)]
            rows.append({'readout': readout, 'builder': builder, 'practical_comparisons_passed': int(comparisons.adoption_gate_pass.sum()),
                         'practical_comparisons_planned': 12, 'practical_all_scopes_pass': bool(passed),
                         'original_simple_all_scopes_pass': bool(core_passed),
                         'strong_simple_comparisons_passed': int(strong_comparisons.adoption_gate_pass.sum()),
                         'strong_simple_comparisons_planned': 9, 'strong_simple_all_scopes_pass': bool(strong_passed),
                         'structure_all_scopes_pass': bool(len(structure) == 3 and structure.adoption_gate_pass.all())
                         if builder == 'B3_DeepSets' else None})
    table = pd.DataFrame(rows)
    primary = table[table.readout == 'risk'].set_index('builder')
    if primary.loc['B3_DeepSets', 'practical_all_scopes_pass'] and primary.loc['B3_DeepSets', 'structure_all_scopes_pass']:
        selected, label = 'B3_DeepSets', 'KEEP_DEEPSETS_CANDIDATE'
    elif primary.loc['B2_Pointwise', 'practical_all_scopes_pass']:
        selected, label = 'B2_Pointwise', 'KEEP_POINTWISE_CANDIDATE'
    else:
        selected, label = 'B1_HGB', 'STOP_PUBLICSET_ARCHITECTURE_RETAIN_B1'
    direct = table[(table.readout == 'direct_risk') & table.practical_all_scopes_pass]
    return table, {'primary_readout': 'R_hist', 'decision': label, 'selected_builder': selected,
                   'readout_dependent': bool(selected == 'B1_HGB' and len(direct)),
                   'direct_only_candidates': direct.builder.tolist() if selected == 'B1_HGB' else [],
                   'claim_scope': 'DEV/SEEN; no independent confirmation or upstream retraining',
                   'interpretation': 'Practical gate must hold against the original four and all three fixed matching references in all three upstream/domain views; B3 additionally needs the B3-B2 structure gate.'}


def plot_results(summary, paired, strata, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'figure.facecolor': 'white'})
    scopes = ['Source/TxPert_GAT', 'Source/TxPert_Exphormer', 'McFaline/DecoderOnly']
    names = {'B0_SupportMean': 'Support mean', 'B1_HGB': 'HGB', 'B2_Pointwise': 'Pointwise MLP',
             'B3_DeepSets': 'DeepSets', 'Magnitude': 'Magnitude', 'NegativeHistorySupport': 'History support'}
    colors = ['#8a8a8a', '#323b49', '#2284a1', '#d28136', '#9b70ab', '#648f5f']
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('pdf', 'png'):
            fig.savefig(output / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, scope in zip(axes, scopes):
        selected = summary[(summary.scope == scope) & (summary.readout == 'risk') & (summary.metric == 'utility20')].set_index('builder')
        for i, (builder, color) in enumerate(zip(BUILDERS, colors)):
            row = selected.loc[builder]
            ax.errorbar(i, row.value, yerr=[[max(0, row.value - row.ci95_lower)], [max(0, row.ci95_upper - row.value)]],
                        fmt='o', color=color, capsize=3)
        ax.axhline(0, color='#bbbbbb', lw=.8)
        ax.set_title(scope.replace('Source/', '').replace('McFaline/', 'McFaline / '))
        ax.set_xticks(range(len(BUILDERS)), [names[b] for b in BUILDERS], rotation=48, ha='right')
        ax.set_ylabel('U20; primary R_hist')
    fig.suptitle('PublicSet: seed metrics averaged within context x outer fold; paired gene bootstrap 95% CI', fontsize=10)
    save(fig, 'FIG01_PRIMARY_UTILITY')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.0), sharex=True)
    labels = [('B2_Pointwise', 'B1_HGB', 'MLP - HGB'), ('B3_DeepSets', 'B1_HGB', 'DeepSets - HGB'),
              ('B3_DeepSets', 'B2_Pointwise', 'DeepSets - MLP')]
    for row_i, readout in enumerate(READOUTS):
        for ax, scope in zip(axes[row_i], scopes):
            for i, (candidate, comparator, label) in enumerate(labels):
                row = paired[(paired.scope == scope) & (paired.readout == readout) & (paired.metric == 'utility20')
                             & (paired.candidate == candidate) & (paired.comparator == comparator)].iloc[0]
                ax.errorbar(row.delta, i, xerr=[[max(0, row.delta - row.ci95_lower)], [max(0, row.ci95_upper - row.delta)]],
                            fmt='o', color=colors[i + 1], capsize=3)
            ax.axvline(0, color='#9a9a9a', lw=.8)
            ax.axvline(.005, color='#447347', lw=.8, ls='--')
            ax.axvline(-.005, color='#b7794a', lw=.8, ls=':')
            ax.set_yticks(range(3), [label for _, _, label in labels])
            ax.set_title(scope.replace('Source/', '').replace('McFaline/', 'McFaline / '))
            ax.set_xlabel('Paired delta U20: ' + ('R_hist' if readout == 'risk' else 'direct RMSE (auxiliary)'))
    save(fig, 'FIG02_PAIRED_MAIN_AUXILIARY')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for col, scope in enumerate(scopes):
        ax = axes[0, col]
        selected = summary[(summary.scope == scope) & (summary.readout == 'risk') & (summary.metric == 'bio_rmse')].set_index('builder')
        for i, builder in enumerate(BUILDERS[:4]):
            row = selected.loc[builder]
            ax.errorbar(i, row.value, yerr=[[max(0, row.value - row.ci95_lower)], [max(0, row.ci95_upper - row.value)]], fmt='o', color=colors[i], capsize=3)
        ax.set_xticks(range(4), [names[b] for b in BUILDERS[:4]], rotation=25, ha='right')
        ax.set_ylabel('Biological reconstruction RMSE')
        ax.set_title(scope.replace('Source/', '').replace('McFaline/', 'McFaline / '))
        sub = strata[(strata.summary_scope == scope) & (strata.readout == 'risk')
                     & (strata.candidate == 'B3_DeepSets') & (strata.comparator == 'B2_Pointwise')]
        heat = sub.pivot(index='context', columns='fold', values='delta_utility20').sort_index()
        ax = axes[1, col]
        bound = max(.05, float(np.nanmax(np.abs(heat.to_numpy()))))
        img = ax.imshow(heat.to_numpy(), aspect='auto', cmap='RdBu', vmin=-bound, vmax=bound)
        ax.set_xticks(range(len(heat.columns)), heat.columns)
        ax.set_yticks(range(len(heat.index)), heat.index)
        ax.set_xlabel('Outer gene fold')
        ax.set_title('Stratum delta U20: DeepSets - MLP')
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f'{heat.iloc[i,j]:.3f}', ha='center', va='center', fontsize=7)
        fig.colorbar(img, ax=ax, shrink=.8)
    save(fig, 'FIG03_BIOLOGY_STRATA')


def write_readme(output, run_dir, manifest, decision, decision_table, paired, coverage, check):
    utility = paired[paired.metric == 'utility20']
    lines = ['# PublicSet 完整五折真实统计', '',
             f"采用决定：**{decision['decision']}**；保留 `{decision['selected_builder']}`。", '',
             '这是 DEV/SEEN 新架构比较，不能作为未参与设计的独立确认。所有数值来自完整运行，未新增上游训练、未读取新 TEST 真值。', '',
             f"输入运行：`{run_dir}`；训练 PID：`{manifest.get('training_pid')}`；统计 PID：`{manifest['pid']}`。", '',
             '主指标先分别计算三个神经种子的指标，在各 context × outer fold 内平均，再对有效分层等权平均。B0/B1/简单规则只有 seed0。主结果不使用平均风险分数排序；未把两个 Source upstream 作为独立生物样本。', '',
             f"配对 bootstrap {manifest['bootstrap_replicates']} 次：Source 全体 575 gene 的同一个重采样同步应用到所有上游、种子和 outer folds；McFaline 377 gene 独立重采样。gene 重复次数被完整保留。{check['literal_repeat_cases']} 个含并列、部分 top-k 和小分层的显式复制检查通过。", '',
             f"全臂共同任务覆盖：{coverage.paired_common_tasks.min()}–{coverage.paired_common_tasks.max()} 条；各 scope 完整数量和缺失见 `COMMON_TASK_COVERAGE.csv`。", '',
             '采用规则同时要求 ΔU20 ≥ 0.005、CI 下界 ≥ −0.005、≥60% 有效分层非负、≥80% 分层有效；macro AURC/error@10/20/50 相对恶化 ≤5%，高风险漏检率增量 ≤0.02。每条门限分别保存，不以“不显著”代替收益或稳定性。分层安全变化另表报告。跨域采用须在每个上游/域对 B0/B1/Magnitude/HistorySupport 及预登记 NearestControl/SameContextSupport/SameConditionSupport 都通过；DeepSets 集合贡献还须对 Pointwise 通过结构门。', '',
             'U20 和低风险 error@ 的任务数均使用 ceil；并列按任务 ID。AURC沿用原定义（累计平均误差的平均，未归一）。漏检按重复任务副本的 top20 交集计算。抽样后不足80%计划分层有效的 macro draw 记 NA；置信区间为有效配对draw的百分位区间。', '',
             '| Scope | Readout | Comparison | ΔU20 | 95% CI | Nonnegative strata | Gate |',
             '|---|---|---|---:|---|---:|---|']
    for scope in ['Source/TxPert_GAT', 'Source/TxPert_Exphormer', 'McFaline/DecoderOnly']:
        for readout in READOUTS:
            for candidate, comparator in COMPARISONS[:3]:
                row = utility[(utility.scope == scope) & (utility.readout == readout) & (utility.candidate == candidate)
                              & (utility.comparator == comparator)].iloc[0]
                lines.append(f'| {scope} | {readout} | {candidate}−{comparator} | {row.delta:.5f} | [{row.ci95_lower:.5f}, {row.ci95_upper:.5f}] | {row.nonnegative_strata_fraction:.1%} | {bool(row.adoption_gate_pass)} |')
    lines.extend(['', '对 B0/Magnitude/HistorySupport 的全部比较及实际收益门见 `PAIRED_COMPARISONS.csv`；三个固定匹配参照另列 `STRONG_SIMPLE_COMPARISONS.csv`，规则不按评价结果选择。单种子结果见 `SEED_METRICS.csv`。McFaline cell-weighted B0−old-guide B0 仅为数据对齐变化，见 `MC_DATA_ALIGNMENT_COMPARISONS.csv`，不作为学习架构增益。', '',
                  f"辅助读出依赖状态：`{decision['readout_dependent']}`；仅 direct 通过的候选：`{decision['direct_only_candidates']}`。辅助读出结果不得替代主读出宣布成功。", '',
                  '本轮具体处理：' + ('冻结通过门限的构建器候选，再按登记合同做内容置乱与固定合法 Source HGB 读出验证。' if decision['selected_builder'] != 'B1_HGB'
                                      else '停止把本轮 Pointwise/DeepSets 作为整体替换候选，继续使用 B1；已计算的凸组合重建诊断只解释生物重建空间。继续完成合法 Source 错误输入资格、同预算风险层和反馈工作。'), '',
                  '文件：`MACRO_METRIC_INTERVALS.csv`、`STRATUM_METRICS.csv`、`SEED_METRICS.csv`、`PAIRED_COMPARISONS.csv`、`PAIRED_STRATUM_DELTAS.csv`、`STRONG_SIMPLE_COMPARISONS.csv`、`STRONG_SIMPLE_STRATUM_DELTAS.csv`、`COMMON_TASK_COVERAGE.csv`、`MC_DATA_ALIGNMENT_COMPARISONS.csv`、`DECISIONS.csv`、`DECISION.json`、`BOOTSTRAP_GENE_MULTIPLICITIES.npz`、`BOOTSTRAP_MACRO_DRAWS.npz`、`BOOTSTRAP_SEED_MACRO_DRAWS.npz`、三张 `FIG*.pdf/png`、`RUN_STATUS.json`。', '',
                  '复现：', '', '```bash',
                  f'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python {Path(__file__).resolve()} --run-dir {run_dir} --output-dir {output} --replicates 5000 --threads 4 --max-seconds 600',
                  '```', ''])
    (output / 'README.md').write_text('\n'.join(lines))


def run(args):
    started = time.monotonic()
    deadline = started + args.max_seconds
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'RUN_STATUS.json').exists():
        raise FileExistsError('Statistics run is immutable; use a new output directory')
    status_path = args.run_dir / 'RUN_STATUS.json'
    status = json.loads(status_path.read_text())
    if status.get('status') != 'COMPLETE' or status.get('phase') != 'full':
        raise ValueError('Statistics require a COMPLETE full run')
    if set(status.get('outer_folds', [])) != set(range(5)) or set(status.get('seeds', [])) != set(SEEDS):
        raise ValueError('Full five folds and all registered seeds required')
    strong_status = json.loads((args.strong_simple_run / 'RUN_STATUS.json').read_text())
    if strong_status.get('status') != 'COMPLETE':
        raise ValueError('Fixed strong references must be COMPLETE')
    manifest = {'status': 'RUNNING', 'pid': os.getpid(), 'started_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                'training_run': str(args.run_dir), 'training_pid': status.get('pid'), 'training_status': 'COMPLETE',
                'bootstrap_replicates': args.replicates, 'bootstrap_seed': args.seed, 'cpu_threads': args.threads,
                'wall_seconds_cap': args.max_seconds, 'new_upstream_training': 0, 'new_truth_reads': 0,
                'role': 'DEV_SEEN_NEW_PUBLICSET', 'code_sha256': file_hash(__file__),
                'numpy_version': np.__version__, 'pandas_version': pd.__version__,
                'input_sha256': file_hash(args.run_dir / 'TASK_PREDICTIONS.csv.gz'),
                'strong_simple_run': str(args.strong_simple_run),
                'strong_simple_input_sha256': file_hash(args.strong_simple_run / 'TASK_PREDICTIONS.csv.gz'),
                'seed_aggregation': 'metrics per seed; average seeds within context x outer-fold; then equal-stratum macro',
                'cluster_resampling': 'Source 575 shared genes synchronously; McFaline 377 genes independently; integer multiplicity exact repeats',
                'safety_gate_scope': 'macro; per-stratum violations separately reported'}
    atomic_json(args.output_dir / 'RUN_MANIFEST.json', manifest)
    try:
        check = semantic_check()
        atomic_json(args.output_dir / 'SEMANTIC_CHECK.json', check)
        aligned, coverage, expected = validate_and_align(pd.read_csv(args.run_dir / 'TASK_PREDICTIONS.csv.gz'), args.run_dir,
                                                       pd.read_csv(args.strong_simple_run / 'TASK_PREDICTIONS.csv.gz'))
        atomic_csv(args.output_dir / 'COMMON_TASK_COVERAGE.csv', coverage)
        counts, genes = make_counts(expected, args.replicates, args.seed)
        np.savez_compressed(args.output_dir / 'BOOTSTRAP_GENE_MULTIPLICITIES.npz', Source_counts=counts['Source'],
                            Source_genes=np.asarray(genes['Source']), McFaline_counts=counts['McFaline'], McFaline_genes=np.asarray(genes['McFaline']))
        strata, seed_metrics, points, draws, masks, _, seed_draws = evaluate(aligned, expected, counts, genes, args.batch_size, deadline)
        summary = intervals(points, draws, masks)
        paired, stratum_deltas = paired_results(points, draws, masks, strata)
        strong_paired, strong_strata = paired_results(points, draws, masks, strata, strong=True)
        alignment, alignment_strata = paired_results(points, draws, masks, strata, alignment=True)
        decision_table, decision = decisions(paired, strong_paired)
        for name, frame in [('MACRO_METRIC_INTERVALS.csv', summary), ('STRATUM_METRICS.csv', strata), ('SEED_METRICS.csv', seed_metrics),
                            ('PAIRED_COMPARISONS.csv', paired), ('PAIRED_STRATUM_DELTAS.csv', stratum_deltas),
                            ('STRONG_SIMPLE_COMPARISONS.csv', strong_paired), ('STRONG_SIMPLE_STRATUM_DELTAS.csv', strong_strata),
                            ('MC_DATA_ALIGNMENT_COMPARISONS.csv', alignment), ('MC_DATA_ALIGNMENT_STRATUM_DELTAS.csv', alignment_strata),
                            ('DECISIONS.csv', decision_table)]:
            atomic_csv(args.output_dir / name, frame)
        atomic_json(args.output_dir / 'DECISION.json', decision)
        array_names = [json.dumps(key) for key in draws]
        np.savez_compressed(args.output_dir / 'BOOTSTRAP_MACRO_DRAWS.npz', keys=np.asarray(array_names),
                            metric_names=np.asarray(METRICS), values=np.asarray(list(draws.values())))
        np.savez_compressed(args.output_dir / 'BOOTSTRAP_SEED_MACRO_DRAWS.npz', keys=np.asarray([json.dumps(key) for key in seed_draws]),
                            metric_names=np.asarray(METRICS), values=np.asarray(list(seed_draws.values())))
        plot_results(summary, paired, stratum_deltas, args.output_dir)
        write_readme(args.output_dir, args.run_dir, manifest, decision, decision_table, paired, coverage, check)
        if time.monotonic() > deadline:
            raise TimeoutError('Statistics CPU wall budget exceeded')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        manifest.update(status='COMPLETE', elapsed_seconds=time.monotonic() - started,
                        cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
                        maximum_rss_kib=usage.ru_maxrss, completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                        semantic_check=check, decision=decision, output_files=sorted(set(p.name for p in args.output_dir.iterdir()) | {'RUN_STATUS.json'}))
        atomic_json(args.output_dir / 'RUN_STATUS.json', manifest)
        atomic_json(args.output_dir / 'RUN_MANIFEST.json', manifest)
        print(json.dumps({'status': 'COMPLETE', 'pid': manifest['pid'], 'elapsed_seconds': manifest['elapsed_seconds'],
                          'output_dir': str(args.output_dir), 'decision': decision}), flush=True)
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_seconds=time.monotonic() - started)
        atomic_json(args.output_dir / 'RUN_STATUS.json', manifest)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=BASE / 'full_v2_resume')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'statistics_v1')
    parser.add_argument('--strong-simple-run', type=Path, default=BASE / 'strong_simple_reference_v1')
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--threads', type=int, choices=[4], default=4)
    parser.add_argument('--max-seconds', type=float, default=600.)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--semantic-tests', action='store_true')
    args = parser.parse_args()
    if args.replicates != 5000 and not args.semantic_tests:
        raise ValueError('Registered analysis requires exactly 5000 paired bootstraps')
    return args


if __name__ == '__main__':
    _args = parse_args()
    if _args.semantic_tests:
        print(json.dumps(semantic_check()))
    else:
        run(_args)
