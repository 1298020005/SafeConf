"""Versioned v2.1 evaluation helpers; frozen older experiments are unchanged.

Bootstrap weights represent biological gene clusters. Algorithm runs remain
fixed functions of those clusters, never extra biological observations.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BUDGETS = (0., .1, .25, .5, .75, 1.)
REVIEWS = (.05, .1, .2, .3)
ORDERS = tuple(range(2026101001, 2026101011))
MODEL_SEEDS = (20260930, 20261001, 20261002)
NULL_SEEDS = tuple(range(2026100901, 2026100921))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest_ids(ids):
    return hashlib.sha256('\n'.join(sorted(map(str, ids))).encode()).hexdigest()


def clean_json(x):
    if isinstance(x, dict): return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def ordered_genes(frame, seed):
    return sorted(frame.gene.astype(str).unique(), key=lambda g: hashlib.sha256(
        f'SafeConf-v21-feedback|{seed}|{g}'.encode()).hexdigest())


def endpoint_arrays(prediction, truth):
    p, y = np.asarray(prediction, float), np.asarray(truth, float)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('finite aligned prediction/truth required for evaluation')
    pc, yc = p - p.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    den = np.sqrt(np.sum(pc * pc, 1) * np.sum(yc * yc, 1))
    corr = np.divide(np.sum(pc * yc, 1), den, out=np.full(len(p), np.nan), where=den > 1e-12)
    # Stable column order is the predefined gene-ID tie key. Truth selection
    # is only used here, in the evaluator, and is never a model feature.
    top = np.argsort(-np.abs(y), axis=1, kind='stable')[:, :min(200, y.shape[1])]
    return {'delta_rmse': np.sqrt(np.mean((p-y)**2, 1)),
            'pearson_error': 1 - np.clip(corr, -1, 1),
            'effect_top200_rmse': np.sqrt(np.mean(np.take_along_axis((p-y)**2, top, axis=1), 1))}


def point_metrics(error, scores, ids, review=.2):
    e, s, ids = np.asarray(error, float), np.asarray(scores, float), np.asarray(ids, str)
    ok = np.isfinite(e) & np.isfinite(s)
    e, s, ids = e[ok], s[ok], ids[ok]
    n = len(e)
    result = {'n_tasks': n, 'n_planned': len(ok), 'utility': np.nan, 'aurc': np.nan,
              'spearman': np.nan, 'high_error_found': 0, 'high_error_precision': np.nan,
              'high_error_recall': np.nan, 'high_risk_miss_rate': np.nan,
              'remaining_mean_error': np.nan, 'review_fraction': review}
    if not n: return result
    k, severe_k = math.ceil(review*n), math.ceil(.2*n)
    high = np.lexsort((ids, -s))[:k]
    oracle = np.lexsort((ids, -e))[:k]
    severe = np.lexsort((ids, -e))[:severe_k]
    denominator = e[oracle].mean()-e.mean()
    if n >= 20 and denominator > 1e-12:
        result['utility'] = float((e[high].mean()-e.mean())/denominator)
    low = np.lexsort((ids, s))
    result['aurc'] = float(np.mean(np.cumsum(e[low])/np.arange(1, n+1)))
    if n >= 3 and np.ptp(e) and np.ptp(s): result['spearman'] = float(spearmanr(e, s).statistic)
    found = len(set(high) & set(severe))
    result.update(review_k=k, severe_k=severe_k, high_error_found=found,
                  high_error_precision=found/k, high_error_recall=found/severe_k,
                  high_risk_miss_rate=1-found/severe_k,
                  remaining_mean_error=float((e.sum()-e[high].sum())/(n-k)) if n > k else np.nan)
    return result


class ClusterBootstrap:
    """Exact count-weight equivalent of resampling whole genes, vectorized.

At a bootstrap top-k boundary only the necessary copies of a row are kept.
Every method uses the same draws; order/learner seeds are not resampled.
"""
    def __init__(self, frame, error, replicates=5000, seed=20260930):
        self.frame = frame.reset_index(drop=True)
        self.error = np.asarray(error, float)
        self.ids = self.frame.task_id.astype(str).to_numpy()
        genes, inv = np.unique(self.frame.gene.astype(str), return_inverse=True)
        rng = np.random.default_rng(seed)
        counts = np.zeros((replicates, len(genes)), np.uint16)
        draws = rng.integers(0, len(genes), (replicates, len(genes)))
        np.add.at(counts, (np.repeat(np.arange(replicates), len(genes)), draws.ravel()), 1)
        self.weights = counts[:, inv]
        self.replicates, self.n_genes = replicates, len(genes)

    def utility(self, score, review=.2, macro=True):
        s = np.asarray(score, float)
        groups = self.frame.groupby('target', sort=True).indices.values() if macro else [np.arange(len(s))]
        out = []
        for rows in groups:
            rows = np.asarray(rows, int)
            rows = rows[np.isfinite(self.error[rows]) & np.isfinite(s[rows])]
            if len(rows) < 20:
                out.append(np.full(self.replicates, np.nan)); continue
            w = self.weights[:, rows].astype(np.int32)
            n = w.sum(1); k = np.ceil(review*n).astype(int)
            e = self.error[rows]
            mean = np.divide(w @ e, n, out=np.full(len(n), np.nan), where=n > 0)
            def selected(order):
                cw = w[:, order]; prior = np.cumsum(cw, axis=1)-cw
                take = np.clip(k[:, None]-prior, 0, cw)
                return np.divide(take @ e[order], k, out=np.full(len(k), np.nan), where=k > 0)
            risk_order = np.lexsort((self.ids[rows], -s[rows]))
            true_order = np.lexsort((self.ids[rows], -e))
            numerator, denominator = selected(risk_order)-mean, selected(true_order)-mean
            out.append(np.divide(numerator, denominator, out=np.full(len(n), np.nan),
                                 where=(n >= 20) & (denominator > 1e-12)))
        values = np.asarray(out)
        count = np.isfinite(values).sum(0)
        return np.divide(np.nansum(values, 0), count, out=np.full(self.replicates, np.nan),
                         where=count >= math.ceil(.8*len(out)))

    def difference(self, scores, baseline, review=.2):
        scores = np.asarray(scores, float)
        if scores.ndim == 1: scores = scores[None, :]
        draws = np.asarray([self.utility(s, review) for s in scores])
        n=np.isfinite(draws).sum(0)
        mean=np.divide(np.nansum(draws,axis=0),n,out=np.full(self.replicates,np.nan),where=n>0)
        return mean-self.utility(baseline, review)


def summarize_draws(draws, point):
    d = np.asarray(draws, float); valid = d[np.isfinite(d)]
    if not len(valid): return {'delta_point': point, 'bootstrap_status': 'NO_VALID_DRAWS'}
    return {'delta_point': point, 'bootstrap_mean_delta': float(valid.mean()),
            'ci95_lower': float(np.quantile(valid, .025)), 'ci95_upper': float(np.quantile(valid, .975)),
            'ci99_lower': float(np.quantile(valid, .005)), 'ci99_upper': float(np.quantile(valid, .995)),
            'bootstrap_valid_draws': len(valid), 'bootstrap_replicates': len(d)}


def h5_column(group, name):
    import h5py
    item = group[name]
    if isinstance(item, h5py.Group):
        cats, codes = item['categories'][:], item['codes'][:]
        cats = np.asarray([x.decode() if isinstance(x, bytes) else x for x in cats])
        if np.any(codes < 0):
            result = np.full(len(codes), '', object); ok = codes >= 0; result[ok] = cats[codes[ok]]
            return result
        return cats[codes]
    a = item[:]
    return np.asarray([x.decode() if isinstance(x, bytes) else x for x in a]) if a.dtype.kind in 'SO' else a
