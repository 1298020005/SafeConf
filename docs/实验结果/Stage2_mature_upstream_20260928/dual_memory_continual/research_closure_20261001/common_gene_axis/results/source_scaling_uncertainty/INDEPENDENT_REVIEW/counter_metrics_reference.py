"""Synthetic review reference: exact original metrics from integer row counts.

This is a slow scalar oracle, not a production data/statistics runner.
"""
import math
import numpy as np

NAMES = ('utility20', 'spearman', 'aurc', 'high_risk_miss_rate',
         'error_at_10', 'error_at_20', 'error_at_50')


def counted_metrics(frame, scores, weights, freeze_original=True):
    truth = frame.true_error_rmse.to_numpy(float)
    ids = frame.task_id.to_numpy(str)
    scores = np.asarray(scores, float)
    weights = np.asarray(weights, int)
    valid = np.isfinite(truth) & np.isfinite(scores)
    original_n = int(valid.sum())
    truth, ids, scores, weights = truth[valid], ids[valid], scores[valid], weights[valid]
    n = int(weights.sum())
    out = dict.fromkeys(NAMES, np.nan)
    if not n:
        return out
    k = math.ceil(.2 * n)
    mean = np.dot(weights, truth) / n

    def top_takes(order, count):
        cs = weights[order]
        takes = np.clip(count - (np.cumsum(cs) - cs), 0, cs)
        restore = np.empty_like(takes)
        restore[order] = takes
        return restore

    risk = top_takes(np.lexsort((ids, -scores)), k)
    oracle = top_takes(np.lexsort((ids, -truth)), k)
    den = np.dot(truth, oracle) / k - mean
    if n >= 20 and (not freeze_original or original_n >= 20) and den > 1e-12:
        out['utility20'] = (np.dot(truth, risk) / k - mean) / den

    def ranks(values):
        order = np.argsort(values, kind='stable')
        v, c = values[order], weights[order]
        starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
        tiecount = np.add.reduceat(c, starts)
        before = np.cumsum(tiecount) - tiecount
        tied = (2 * before + tiecount + 1) / 2
        rowties = np.cumsum(np.r_[0, v[1:] != v[:-1]])
        ordered = tied[rowties]
        result = np.empty_like(ordered)
        result[order] = ordered
        return result

    if n >= 3:
        a, b = ranks(scores) - (n + 1) / 2, ranks(truth) - (n + 1) / 2
        norm = np.sqrt(np.dot(weights, a * a) * np.dot(weights, b * b))
        if norm > 0:
            out['spearman'] = np.dot(weights, a * b) / norm

    low = np.lexsort((ids, scores))
    c, e = weights[low], truth[low]
    before = np.cumsum(c) - c
    esum = np.cumsum(c * e) - c * e
    harmonic = np.r_[0., np.cumsum(1 / np.arange(1, n + 1))]
    out['aurc'] = np.sum(c * e + (esum - before * e) *
                         (harmonic[before + c] - harmonic[before])) / n
    out['high_risk_miss_rate'] = 1 - np.minimum(risk, oracle).sum() / k
    for fraction in (10, 20, 50):
        covered = math.ceil(fraction / 100 * n)
        take = top_takes(low, covered)
        out[f'error_at_{fraction}'] = np.dot(truth, take) / covered
    return out
