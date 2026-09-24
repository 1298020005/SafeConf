#!/usr/bin/env python3
"""Descriptive gene-cluster bootstrap for retrospective E253 comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_e253_lara_history_transfer import utility


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/实验结果/E253_lara_history_transfer_20260924'


def main() -> None:
    data = pd.read_csv(OUT / 'E253_ALL_120_TASKS.csv')
    genes = sorted(data.perturbation.astype(str).unique())
    folds = {fold: part.reset_index(drop=True) for fold, part in data.groupby('fold_id')}
    rng = np.random.default_rng(20260924)
    draws = []
    for iteration in range(2000):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        count = pd.Series(sampled).value_counts().to_dict()
        for endpoint in ('absolute_error', 'excess_error'):
            differences = []
            for fold, part in folds.items():
                repeat = part.perturbation.map(count).fillna(0).to_numpy(int)
                boot = part.iloc[np.repeat(np.arange(len(part)), repeat)]
                if len(boot) < 8:
                    continue
                error = boot[endpoint].to_numpy(float)
                differences.append(utility(boot.M_plus_H.to_numpy(float), error)
                                   - utility(boot.rank_M.to_numpy(float), error))
            if len(differences) != 5:
                raise ValueError('missing fold after gene-cluster resampling')
            draws.append({'draw': iteration, 'endpoint': endpoint,
                          'delta_MH_vs_M': float(np.mean(differences))})
    all_draws = pd.DataFrame(draws)
    all_draws.to_csv(OUT / 'E253_GENE_CLUSTER_BOOTSTRAP_DRAWS.csv', index=False)
    intervals = {}
    for endpoint, part in all_draws.groupby('endpoint'):
        intervals[endpoint] = {'lower_95': float(part.delta_MH_vs_M.quantile(.025)),
                               'upper_95': float(part.delta_MH_vs_M.quantile(.975)),
                               'fraction_positive': float((part.delta_MH_vs_M > 0).mean())}
    (OUT / 'E253_BOOTSTRAP_AUDIT.json').write_text(
        json.dumps({'status': 'RETROSPECTIVE_DESCRIPTIVE_INTERVAL',
                    'cluster': 'perturbation gene across all five folds',
                    'n_unique_genes': len(genes), 'n_draws': 2000,
                    'intervals': intervals}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(intervals, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
