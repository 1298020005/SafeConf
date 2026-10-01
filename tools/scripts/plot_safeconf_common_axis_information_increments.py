#!/usr/bin/env python3
"""Render existing fixed 2840-gene paired comparisons without refitting."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis'
RESULTS = DOC / 'results'
OUT = DOC / 'figures/information_increments_v1'
LINES = ['Exphormer_to_GAT', 'GAT_to_Exphormer', 'TxPert_to_McFaline']
LABELS = ['Exphormer → GAT\n1,808 tasks / 575 genes',
          'GAT → Exphormer\n1,808 tasks / 575 genes',
          'TxPert → McFaline\n543 tasks / 380 genes']


def main():
    if OUT.exists():
        raise RuntimeError('Existing scientific figures cannot be overwritten')
    paired_path = RESULTS / 'PAIRED_CLUSTER_BOOTSTRAP.csv'
    support_path = RESULTS / 'SUPPORT_CONTENT_ATTRIBUTION_COMPARISONS.csv'
    paired = pd.read_csv(paired_path)
    support = pd.read_csv(support_path)
    selections = [
        ('History distance − magnitude', paired,
         'Learned_WeightedHistoryDistance', 'Magnitude', 'delta_utility20'),
        ('History distance − support only', support[support.metric.eq('utility20')],
         'Learned_WeightedHistoryDistance', 'NegativeHistorySupport', 'point_difference'),
        ('Source HGB − history distance', paired,
         'Learned_hgb', 'Learned_WeightedHistoryDistance', 'delta_utility20'),
        ('Public + source HGB − prediction HGB', paired,
         'Learned_hgb', 'Prediction_hgb', 'delta_utility20'),
    ]
    extracted = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    for index, (title, frame, a, b, point) in enumerate(selections):
        ax = axes[index]
        for position, line in enumerate(LINES):
            row = frame[frame.line.eq(line) & frame.method_a.eq(a) & frame.method_b.eq(b)]
            if len(row) != 1:
                raise RuntimeError('Exactly one archived paired comparison required')
            row = row.iloc[0]
            if int(row.bootstrap_replicates) != 5000:
                raise RuntimeError('Registered 5000-draw statistics required')
            value, lo, hi = float(row[point]), float(row.ci95_lower), float(row.ci95_upper)
            color = '#285781' if position < 2 else '#a64b32'
            ax.plot([lo, hi], [position, position], color=color, linewidth=2)
            ax.plot(value, position, 'o', color=color, markersize=6)
            extracted.append({'panel': title, 'line': line, 'method_a': a, 'method_b': b,
                              'delta_u20': value, 'ci95_lower': lo, 'ci95_upper': hi,
                              'bootstrap_replicates': 5000, 'n_tasks': int(row.n_tasks),
                              'n_clusters': int(row.n_clusters)})
        ax.axvline(0, color='#777777', linewidth=1, linestyle='--')
        ax.set_title(title, fontsize=10, pad=14)
        ax.set_xlabel('Δ Utility@20 (higher is better)')
        ax.grid(axis='x', color='#dddddd', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(3), LABELS)
    axes[0].invert_yaxis()
    fig.suptitle('The value of history and source supervision depends on the comparison', fontsize=13)
    fig.text(.5, .01, 'Fixed 2,840-gene effect axis; context macro; paired 95% gene-cluster bootstrap intervals. DEV/SEEN analysis.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .91))
    OUT.mkdir(parents=True)
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'INFORMATION_INCREMENTS.{suffix}', dpi=180)
    svg = OUT / 'INFORMATION_INCREMENTS.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    pd.DataFrame(extracted).to_csv(OUT / 'SOURCE_DATA.csv', index=False, lineterminator='\n')
    manifest = {'status': 'COMPLETE_EXISTING_FIXED_RESULTS_ONLY', 'new_fits': 0,
                'new_evaluations_or_target_truth': 0, 'Orion_access': False,
                'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in [paired_path, support_path]],
                'comparisons': len(extracted), 'all_archived_intervals_unchanged': True}
    (OUT / 'FIGURE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
