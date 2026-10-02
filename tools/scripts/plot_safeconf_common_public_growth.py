#!/usr/bin/env python3
"""Plot the fixed full-task Public target-coverage evidence, without fitting."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis/results'
DOC = RESULTS / 'public_growth_uncertainty_v1'
OUT = DOC / 'figures'


def main():
    if OUT.exists():
        raise FileExistsError('Existing figure version cannot be overwritten')
    curve = pd.read_csv(DOC / 'FIXED_TASK_HGB_CURVE_INTERVALS.csv')
    orders = pd.read_csv(DOC / 'FIXED_TASK_ORDER_POINTS.csv')
    coverage = pd.read_csv(RESULTS / 'public_mechanisms/GROWTH_COVERAGE.csv')
    OUT.mkdir()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    lines = [('GAT_to_Exphormer', 'GAT to Exphormer', '#2b6cb0'),
             ('Exphormer_to_GAT', 'Exphormer to GAT', '#2b6cb0'),
             ('TxPert_to_McFaline', 'TxPert to McFaline', '#b54632')]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), sharey=True)
    for ax, (line, title, color) in zip(axes, lines):
        for _, part in orders[orders.line.eq(line) & orders.context.eq('macro')].groupby('order'):
            part = part.sort_values('budget')
            ax.plot(100 * part.budget, part.utility20, color=color, alpha=.2, lw=1)
        part = curve[curve.line.eq(line) & curve.context.eq('macro') & curve.metric.eq('utility20')].sort_values('budget')
        x = part.budget.to_numpy(float) * 100
        ax.fill_between(x, part.ci95_lower.to_numpy(float), part.ci95_upper.to_numpy(float), color=color, alpha=.16)
        ax.plot(x, part.point_estimate, 'o-', color=color, lw=2, label='Mean of five fixed orders')
        ax.axhline(0, color='0.6', ls=':', lw=.8)
        ax.set_title(title)
        ax.set_xticks([10, 25, 50, 75, 100])
        ax.set_xlabel('Public target clusters available (%)')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Full-task macro Utility@20')
    axes[0].set_ylim(-.2, .95)
    fig.text(.5, .015, 'Source supervision fixed at 100%; support-weighted priors; 5000 gene-cluster draws; SEEN analysis',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'FIXED_TASK_PUBLIC_COVERAGE_RISK_CURVES.{extension}', dpi=170, metadata={'Creator': 'SafeConf fixed evidence'})
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, domain in zip(axes, ['TxPert', 'McFaline']):
        part = coverage[coverage.domain.eq(domain)]
        for _, values in part.groupby('order'):
            grouped = values.groupby('budget').agg(n=('n_planned_tasks', 'sum'), available=('n_tasks_with_history', 'sum')).reset_index()
            ax.plot(100 * grouped.budget, 100 * grouped.available / grouped.n, color='#2b6cb0', alpha=.35, marker='o', lw=1)
        ax.set_title(domain)
        ax.set_xticks([10, 25, 50, 75, 100])
        ax.set_xlabel('Public target clusters available (%)')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Fixed evaluation tasks with history (%)')
    axes[0].set_ylim(0, 105)
    fig.tight_layout()
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'PUBLIC_TASK_AVAILABILITY.{extension}', dpi=170, metadata={'Creator': 'SafeConf fixed evidence'})
    plt.close(fig)
    files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(OUT.iterdir())]
    (OUT / 'FIGURE_MANIFEST.json').write_text(json.dumps({'scope': 'SEEN fixed evidence; no method selection',
        'recipe': str(Path(__file__).resolve()), 'recipe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'coverage_lines': 'Five fixed orders, not independent studies or confidence intervals', 'files': files}, indent=2) + '\n')


if __name__ == '__main__':
    main()
