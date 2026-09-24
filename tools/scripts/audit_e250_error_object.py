#!/usr/bin/env python3
"""Post-result E250 diagnostic: distinguish absolute from excess model error.

This audit was specified only after seeing E250 absolute-error results and is
therefore exploratory. It must not be represented as an independent test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path('/home/yyf/proj')
OUT = ROOT / 'docs/实验结果/E250_frangieh_history_transfer_20260924'
METRICS = {
    'scGPT': INPUT_ROOT / 'docs/实验结果/E106_frangieh_context_scgpt_20260713/E106_ALL_TEST_TASK_METRICS.csv',
    'GEARS': INPUT_ROOT / 'docs/实验结果/E107_frangieh_context_gears_20260713/E107_ALL_TEST_TASK_METRICS.csv',
}


def utility(score: np.ndarray, error: np.ndarray) -> float:
    n = len(score)
    k = max(1, int(np.ceil(.2 * n)))
    # Stable tie handling via average membership at the cutoff.
    def chosen(values: np.ndarray) -> np.ndarray:
        threshold = np.partition(values, n-k)[n-k]
        above = values > threshold
        equal = values == threshold
        weights = above.astype(float)
        weights[equal] = (k - above.sum()) / equal.sum()
        return weights
    pick, oracle = chosen(score), chosen(error)
    average = float(np.mean(error))
    denom = float(np.dot(oracle, error) / k - average)
    return float((np.dot(pick, error) / k - average) / denom) if denom > 1e-12 else float('nan')


def evaluate() -> None:
    scores = pd.read_csv(OUT / 'SCORES.csv')
    truths = []
    for predictor, path in METRICS.items():
        frame = pd.read_csv(path, usecols=['fold_id', 'task_id', 'true_error_rmse', 'true_effect_l2_diagnostic'])
        frame['predictor'] = predictor
        truths.append(frame)
    data = scores.merge(pd.concat(truths, ignore_index=True),
                        on=['predictor', 'fold_id', 'task_id'], validate='one_to_one')
    if len(data) != len(scores) or data[['M', 'H', 'M_plus_H']].isna().any().any():
        raise ValueError('unmatched or missing scores')
    data['no_change_rmse'] = data.true_effect_l2_diagnostic / np.sqrt(512)
    data['excess_rmse'] = data.true_error_rmse - data.no_change_rmse
    records = []
    for (predictor, fold), part in data.groupby(['predictor', 'fold_id']):
        for endpoint in ('true_error_rmse', 'no_change_rmse', 'excess_rmse'):
            y = part[endpoint].to_numpy(float)
            for method in ('M', 'H', 'M_plus_H'):
                x = part[method].to_numpy(float)
                records.append({'predictor': predictor, 'fold_id': fold, 'endpoint': endpoint,
                                'method': method, 'n_tasks': len(part),
                                'spearman': float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
                                'utility_20': utility(x, y),
                                'selected_mean_endpoint': float(np.dot((rankdata(x) > .8 * len(x)).astype(float), y)
                                                                / max(1, (rankdata(x) > .8 * len(x)).sum()))})
    output = pd.DataFrame(records)
    output.to_csv(OUT / 'ERROR_OBJECT_DIAGNOSTIC.csv', index=False)
    pivot = output.pivot(index=['predictor', 'fold_id', 'endpoint'], columns='method', values='utility_20')
    pivot['delta_MH_vs_M'] = pivot.M_plus_H - pivot.M
    pivot.to_csv(OUT / 'ERROR_OBJECT_PAIRED.csv')
    summary = {'status': 'POST_RESULT_EXPLORATORY_DIAGNOSTIC',
               'endpoints': {endpoint: {'mean_delta_MH_vs_M': float(frame.delta_MH_vs_M.mean()),
                                        'positive_pairs': int((frame.delta_MH_vs_M > 0).sum()),
                                        'n_pairs': len(frame)}
                             for endpoint, frame in pivot.reset_index().groupby('endpoint')},
               'n_model_beats_no_change_taskwise': int((data.excess_rmse < 0).sum()),
               'n_task_model_rows': len(data),
               'warning': 'Analysis chosen after absolute-error E250 results; upstream capability failed in all six fold-model pairs.'}
    (OUT / 'ERROR_OBJECT_AUDIT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    evaluate()
