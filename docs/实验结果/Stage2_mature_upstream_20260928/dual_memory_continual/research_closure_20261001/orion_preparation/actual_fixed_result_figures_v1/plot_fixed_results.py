#!/usr/bin/env python3
"""Render saved fixed summaries; no statistical or model computation."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
EVAL = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_registered_test_extendedbank_20261002_v1/fixed_risk_evaluation')
DOC = OUT.parents[1]
SPEC = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_public_bank_extension_20261002_v4_partitioned/expanded_comparison/COMPARISON_SPEC.json')
REGISTRY = DOC/'orion_preparation/risk_evaluation/ORION_CLAIM_INTERPRETATION_REGISTRY.json'
PINS = {
    EVAL/'EXTENSION_EVALUATION_RECEIPT.json': '803a11ea87b9019b3cdac73d2cfe76dcd392747a452e57cf02a174582c51b73d',
    EVAL/'EVALUATION_MANIFEST.json': '75c1c03b146498d88f5c0f5a4f8d3a980eab7366771d4933e16aad553039b225',
    EVAL/'ARTIFACT_HASHES.json': '6e0f9aaf720f657de07078b6b5623286cca944d375e0116ead48038a42f1d6ff',
    EVAL/'METHOD_METRIC_INTERVALS.csv': '64fe43165ce83149e60104856159b74fd8e07842dcd8fa4082319fb1ac402418',
    EVAL/'ALL_PRESPECIFIED_PAIRED_COMPARISONS.csv': '77fc4acd321a1abe95af9973e2b5b6f5649231445b5a9323ee50d9eefc739f50',
    EVAL/'COHORT_COVERAGE.csv': 'f3e4ab32edd715073d1a938f6df501b74b759d136ec2649445cf008d168530d3',
    EVAL/'primary_common_COHORT.csv': '59385312a8a5d25ff4435678535f1495514fcb5eb8bbf43e94e6c2911e2ab1e5',
    SPEC: '47c76a048159e4e6d964730507001136e59fb0185d6364887710f68f1ca0d2e4',
    REGISTRY: 'd227bf92e1cc4396a332b82f0ae5a3ffd90a03ce21dadac50ed8d1e0c9382bea',
}
CONTEXTS = ('HCT116', 'HEK293T', 'macro')
COLORS = ('#0072B2', '#D55E00', '#374151')


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def one(table, **keys):
    found = [row for row in table if all(row.get(key) == value for key, value in keys.items())]
    if len(found) != 1:
        raise ValueError('Missing/duplicate fixed summary row: '+str(keys))
    return found[0]


def forest_axes(axes, labels, limits):
    for index, ax in enumerate(axes):
        ax.set_ylim(len(labels)-0.5, -0.5)
        ax.set_xlim(*limits)
        ax.set_yticks(range(len(labels)), labels if index == 0 else [])
        ax.axvline(0, color='#9CA3AF', linewidth=0.85, zorder=0)
        ax.grid(axis='x', color='#E5E7EB', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.tick_params(axis='x', labelsize=9)


def interval(ax, y, point, lo, hi, color):
    ax.hlines(y, lo, hi, color=color, linewidth=1.35)
    ax.vlines([lo, hi], y-0.065, y+0.065, color=color, linewidth=1)
    ax.plot(point, y, 'o', color=color, markersize=4.6)


def save(fig, stem):
    for suffix in ('png', 'pdf'):
        path = OUT/(stem+'.'+suffix)
        if path.exists():
            raise FileExistsError('Immutable figure already exists: '+str(path))
        fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


def main():
    for path, digest in PINS.items():
        if binding(path)['sha256'] != digest:
            raise ValueError('Pinned completed input changed: '+str(path))
    receipt = json.loads((EVAL/'EXTENSION_EVALUATION_RECEIPT.json').read_text())
    if receipt.get('status') != 'COMPLETE':
        raise ValueError('Completed extension evaluation required')
    registry = json.loads(REGISTRY.read_text())
    spec = json.loads(SPEC.read_text())
    methods = spec['methods']
    points = rows(EVAL/'METHOD_METRIC_INTERVALS.csv')
    pairs = rows(EVAL/'ALL_PRESPECIFIED_PAIRED_COMPARISONS.csv')
    coverage = [row for row in rows(EVAL/'COHORT_COVERAGE.csv') if row['cohort'] == 'primary_common']
    tasks = sum(int(row['n_tasks']) for row in coverage)
    eligible = sum(int(row['eligible_TEST_tasks']) for row in coverage)
    clusters = len({row['target_gene_id'] for row in rows(EVAL/'primary_common_COHORT.csv')})
    assert tasks == 232 and eligible == 2993 and clusters == 144 and len(methods) == 13
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'pdf.fonttype': 42, 'axes.labelsize': 10, 'axes.titlesize': 11})

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 8.4))
    forest_axes(axes, methods, (-0.45, 0.70))
    titles = ('HCT116 · 107 tasks', 'HEK293T · 125 tasks', 'Unweighted macro · 232 tasks')
    for ax, context, color, title in zip(axes, CONTEXTS, COLORS, titles):
        for y, method in enumerate(methods):
            row = one(points, cohort='primary_common', context=context, method=method, metric='utility20')
            interval(ax, y, float(row['point']), float(row['ci95_lower']), float(row['ci95_upper']), color)
        ax.set_title(title, loc='left', pad=12, fontweight='bold')
        ax.set_xlabel('U20 (higher is better)')
    fig.suptitle('Orion: all 13 frozen methods on the same primary cohort', x=0.04, y=0.965, ha='left', fontsize=14, fontweight='bold')
    fig.text(0.04, 0.92, f'{tasks} tasks / {clusters} target-gene clusters; history-supported coverage {tasks} / {eligible:,} eligible TEST tasks ({100*tasks/eligible:.1f}%).', fontsize=10)
    fig.text(0.04, 0.075, 'Dots: saved point estimates. Bars: nominal 95% gene-cluster bootstrap intervals (5,000 shared draws), conditional on fixed fits.\nFrozen method order retained. Individual-method intervals are not pairwise superiority tests; no method is promoted as a winner.', fontsize=9, linespacing=1.55)
    fig.subplots_adjust(left=0.265, right=0.975, top=0.845, bottom=0.17, wspace=0.13)
    save(fig, 'ALL13_PRIMARY_U20')

    contrasts = [tuple(registry['fixed_primary_contrast']), tuple(registry['fixed_manual_secondary_contrast']), ('Learned_hgb', registry['strong_metadata_rule']), tuple(registry['Public_increment_with_source_budget_fixed']['contrast'])]
    labels = ['Primary: Learned HGB\n− Learned history distance', 'Manual secondary: Manual HGB\n− Manual history distance', 'Support contrast: Learned HGB\n− historical support', 'Public contrast: Learned HGB\n− P-only HGB']
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.6))
    forest_axes(axes, labels, (-0.55, 0.80))
    for ax, context, color, title in zip(axes, CONTEXTS, COLORS, titles):
        ax.axvline(registry['practical_margin_U20'], color='#6B7280', linestyle='--', linewidth=0.75, zorder=0)
        for y, (a, b) in enumerate(contrasts):
            found = [row for row in pairs if row['cohort'] == 'primary_common' and row['context'] == context and row['metric'] == 'utility20' and (row['method_a'], row['method_b']) in [(a, b), (b, a)]]
            if len(found) != 1:
                raise ValueError('Missing/duplicate fixed contrast')
            row = found[0]
            point, lo, hi = (float(row[key]) for key in ('difference', 'ci95_lower', 'ci95_upper'))
            if row['method_a'] != a:
                point, lo, hi = -point, -hi, -lo
            interval(ax, y, point, lo, hi, color)
        ax.set_title(title, loc='left', pad=12, fontweight='bold')
        ax.set_xlabel('Paired ΔU20 (A − B)')
    fig.suptitle('Orion: four prespecified paired contrasts', x=0.04, y=0.955, ha='left', fontsize=14, fontweight='bold')
    fig.text(0.04, 0.88, f'Same {tasks} tasks / {clusters} target-gene clusters; primary cohort covers {tasks} / {eligible:,} eligible TEST tasks.', fontsize=10)
    fig.text(0.04, 0.075, 'Nominal paired 95% intervals; 5,000 shared gene-cluster draws, conditional on fixed fits. Dashed line: +0.005 practical point margin.\nPrimary does not pass. The support contrast passes its registered checks, but cannot establish increment beyond strong history rules.\nManual and Public intervals include zero; neither replaces the primary contrast.', fontsize=9, linespacing=1.5)
    fig.subplots_adjust(left=0.265, right=0.975, top=0.775, bottom=0.255, wspace=0.13)
    save(fig, 'FIXED4_PAIRED_U20_CONTRASTS')

    source_receipt = {'schema': 'safeconf_actual_fixed_result_figure_source_binding_v1', 'status': 'COMPLETE', 'plot_code': binding(Path(__file__)), 'matplotlib_version': matplotlib.__version__, 'inputs': [binding(path) for path in PINS], 'cohort': 'primary_common', 'tasks': tasks, 'target_gene_clusters_union': clusters, 'eligible_TEST_tasks': eligible, 'method_order': methods, 'fixed_contrasts': [list(pair) for pair in contrasts], 'data_transformations': 'Numeric CSV parsing, identity/count descriptions, and prespecified signed contrast reversal only; points and CI endpoints copied from completed fixed summaries. No averaging, statistical-estimate recomputation, bootstrap, model calls, fits or truth/error-label reads.', 'no_winner_selection': True}
    path = OUT/'SOURCE_BINDING_RECEIPT.json'
    with path.open('x') as stream:
        json.dump(source_receipt, stream, indent=2); stream.write('\n')
    files = [Path(__file__), path]+[OUT/(stem+'.'+suffix) for stem in ('ALL13_PRIMARY_U20', 'FIXED4_PAIRED_U20_CONTRASTS') for suffix in ('png', 'pdf')]
    with (OUT/'OWNED_PLOTS_MANIFEST.json').open('x') as stream:
        json.dump({'schema': 'safeconf_owned_plot_files_v1', 'files': [binding(file) for file in files]}, stream, indent=2); stream.write('\n')
    for path in files+[OUT/'OWNED_PLOTS_MANIFEST.json']:
        path.chmod(0o444)


if __name__ == '__main__':
    main()
