#!/usr/bin/env python3
"""Scientific figures and source sampling diagnostics from fixed saved results."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import SEEDS
from tools.scripts import run_safeconf_research_closure as closure


def main():
    common = closure.OUT / 'common_gene_axis'
    results = common / 'results'
    plots = common / 'figures'; plots.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(common / 'truth_reproducibility/cell_sampling/CELL_SAMPLING_TASK_DIAGNOSTICS.csv')
    diagnostic = pd.read_csv(common / 'truth_reproducibility/cell_sampling/CELL_SAMPLING_RISK_DIAGNOSTICS.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for name, part in cells.groupby('target', sort=True):
        axes[0].scatter(part.n_test_cells, part.true_error_rmse, s=12, alpha=.6, label=name)
    axes[0].set(xlabel='Observed TEST cells per task', ylabel='Primary task RMSE',
                title='A. Observed error depends on sampling size', xscale='log')
    axes[0].legend(frameon=False)
    names = ['NegativeHistorySupport', 'Learned_WeightedHistoryDistance', 'Learned_hgb']
    labels = ['Historical support only', 'Weighted history distance', 'Source-error HGB']
    primary, balanced, low, high = [], [], [], []
    for method in names:
        part = diagnostic[diagnostic.method.eq(method)]
        primary.append(part[part.diagnostic_truth.eq('true_error_rmse')].utility20.mean())
        values = part[part.diagnostic_truth.str.startswith('equal20_error')].groupby('diagnostic_truth').utility20.mean()
        balanced.append(values.mean()); low.append(values.mean() - values.min()); high.append(values.max() - values.mean())
    x = np.arange(len(names))
    axes[1].bar(x - .18, primary, .36, label='Original cell counts')
    axes[1].bar(x + .18, balanced, .36, yerr=[low, high], capsize=3,
                label='Fixed 20 cells; all five hash orders')
    axes[1].axhline(0, color='black', lw=.6)
    axes[1].set_xticks(x, labels, rotation=15, ha='right')
    axes[1].set(ylabel='Macro Utility@20', title='B. Fixed-size truth sensitivity (diagnostic)')
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ['png', 'pdf']:
        fig.savefig(plots / f'SAMPLING_SIZE_SENSITIVITY.{suffix}', dpi=180)
    plt.close(fig)
    feedback = pd.read_csv(results / 'STRICT_FEEDBACK_MACRO.csv')
    native = pd.read_csv(results / 'native_control_pertema_adaptation/MACRO.csv')
    feedback = pd.concat([feedback, native], ignore_index=True)
    feedback = feedback[feedback.seed.eq(SEEDS[0])]
    fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    methods = [('Shared', 'Shared'), ('TargetOnly_HGB', 'Target only HGB'),
               ('PublicTarget_HGB', 'Public + Target HGB'), ('SharedTarget_HGB', 'Shared + Target HGB'),
               ('PertEMA_Native_adapted', 'PertEMA native adaptation'),
               ('PertEMA_Native_Public_adapted', 'PertEMA native + Public')]
    for method, label in methods:
        part = feedback[feedback.method.eq(method)].sort_values('budget')
        ax.plot(part.budget * 100, part.utility20, marker='o', label=label)
    ax.set(xlabel='Opened feedback clusters (% of registered feedback pool)', ylabel='Macro Utility@20',
           title='Common-axis feedback: same 212 holdout tasks')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.axhline(0, color='black', lw=.6)
    for suffix in ['png', 'pdf']:
        fig.savefig(plots / f'COMMON_AXIS_FEEDBACK.{suffix}', dpi=180)
    plt.close(fig)
    source = pd.read_csv(closure.tx.TASK_PATH)
    source = source[source.analysis_stratum.eq('primary_ge30')]
    metadata = source.set_index('task_id')
    frame = pd.read_parquet(closure.RUNTIME / 'common_gene_axis/risk_cache/source_Manual.parquet')
    frame['n_target_cells'] = frame.task_id.map(metadata.n_target_cells)
    rows = []
    for (upstream, context), part in frame.groupby(['upstream', 'target'], sort=True):
        for column in ['n_target_cells', 'log_history_support', 'prior_uncertainty', 'predicted_magnitude']:
            rows.append({'upstream': upstream, 'context': context, 'field': column,
                         'rho_with_true_error': float(spearmanr(part[column], part.true_error_rmse).statistic),
                         'n_tasks': len(part), 'role': 'SEEN_METADATA_DIAGNOSTIC'})
    closure.tx.atomic_csv(results / 'SOURCE_SAMPLING_ASSOCIATIONS.csv', pd.DataFrame(rows))
    closure.tx.atomic_json(plots / 'FIGURE_PROVENANCE.json', {
        'role': 'SEEN_POST_CONFIRMATION', 'sampling_tasks': len(cells),
        'sampling_error_bars': 'minimum/maximum over all five fixed hash orders; not confidence intervals',
        'equal20_is_new_primary_endpoint': False, 'feedback_tasks': 212,
        'feedback_primary_seed': SEEDS[0], 'pertEMA_native_is_adaptation': True,
        'no_model_selection_from_figures': True})


if __name__ == '__main__':
    main()
