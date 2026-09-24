#!/usr/bin/env python3
"""Retrospective conditional-distance diagnostic on E253's fixed Lara scope."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
E253 = ROOT / 'docs/实验结果/E253_lara_history_transfer_20260924'
OUT = ROOT / 'docs/实验结果/E254_lara_history_consistency_20260924'
SOURCE_ROOT = Path('/home/yyf/proj/docs/实验结果/E112_external_formal_dual_models_20260713/Lara_exvivo')
MANIFEST = Path('/home/yyf/proj/docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv')
CACHE = Path('/home/yyf/data/safeconf_e112_external/Lara_exvivo_CONTROL_ONLY_512.npz')


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a-b)**2)))


def utility(score: np.ndarray, errors: np.ndarray) -> float:
    n = len(score)
    k = max(1, int(np.ceil(.2*n)))
    def chosen(values: np.ndarray) -> np.ndarray:
        cutoff = np.partition(values, n-k)[n-k]
        above = values > cutoff
        tied = values == cutoff
        out = above.astype(float)
        out[tied] = (k-int(above.sum())) / int(tied.sum())
        return out
    pick, oracle = chosen(score), chosen(errors)
    average = float(errors.mean())
    denom = float(np.dot(oracle, errors)/k-average)
    return float((np.dot(pick, errors)/k-average)/denom) if denom > 1e-12 else float('nan')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(E253 / 'E253_ALL_120_TASKS.csv')
    if len(frame) != 116:
        raise ValueError('E253 task scope changed')
    manifest = pd.read_csv(MANIFEST)
    allowed = set(map(tuple, manifest.loc[manifest.dataset.eq('Lara_exvivo') & manifest.split.eq('train')
                                          & manifest.in_train_fraction_100,
                                          ['fold_id', 'context', 'perturbation']].astype(str).itertuples(index=False, name=None)))
    with np.load(CACHE, allow_pickle=False) as saved:
        contexts = saved['contexts'].astype(str).tolist()
        perts = saved['perturbations'].astype(str).tolist()
        effects = saved['effects'].reshape(len(contexts), len(perts), 512)
    ctx_idx = {name: i for i, name in enumerate(contexts)}
    pert_idx = {name: i for i, name in enumerate(perts)}
    records = pd.read_csv(SOURCE_ROOT / 'PREDICTION_RECORDS.csv')
    records = records.loc[records.predictor_name.eq('GEARS_context_mean_trainonly_graphs'),
                          ['fold_id', 'task_id', 'true_effect_key']]
    frame = frame.merge(records, on=['fold_id', 'task_id'], validate='one_to_one')
    radius, target_distance, covered, bound_holds = [], [], [], []
    with np.load(SOURCE_ROOT / 'arrays/true_effects.npz', allow_pickle=False) as truths:
        for item in frame.itertuples(index=False):
            names = [c for c in contexts if (item.fold_id, c, item.perturbation) in allowed]
            if len(names) < 2 or item.context in names:
                raise ValueError('source scope violated')
            source = np.stack([effects[ctx_idx[c], pert_idx[item.perturbation]] for c in names])
            mu = source.mean(axis=0)
            leave_one_out = [rmse(source[i], np.delete(source, i, axis=0).mean(axis=0))
                                 for i in range(len(names))]
            r = max(leave_one_out)
            true = np.asarray(truths[str(item.true_effect_key)], dtype=np.float64)
            cache_true = effects[ctx_idx[item.context], pert_idx[item.perturbation]]
            if rmse(true, cache_true) > 1e-5:
                raise ValueError('cached true effect does not match E112 truth archive')
            distance = rmse(true, mu)
            l = max(0.0, float(item.G)-r)
            radius.append(r)
            target_distance.append(distance)
            covered.append(distance <= r + 1e-12)
            bound_holds.append(float(item.absolute_error) + 1e-10 >= l)
    frame['source_loo_radius'] = radius
    frame['target_to_source_mean_distance_diagnostic'] = target_distance
    frame['target_within_source_radius_diagnostic'] = covered
    frame['conditional_bound_holds_diagnostic'] = bound_holds
    frame['L'] = np.maximum(0, frame.G-frame.source_loo_radius)
    for fold, index in frame.groupby('fold_id').indices.items():
        part = frame.iloc[index]
        for name in ('L',):
            frame.loc[frame.index[index], 'rank_'+name] = rankdata(part[name].to_numpy(float))/len(part)
        frame.loc[frame.index[index], 'M_plus_L'] = .8*part.rank_M.to_numpy(float)+.2*frame.loc[frame.index[index], 'rank_L'].to_numpy(float)
    frame.to_csv(OUT / 'E254_TASK_DIAGNOSTICS.csv', index=False)
    methods = {'M': 'rank_M', 'G': 'rank_G', 'L': 'rank_L',
               'M_plus_H': 'M_plus_H', 'M_plus_G': 'M_plus_G', 'M_plus_L': 'M_plus_L'}
    records = []
    for fold, part in frame.groupby('fold_id'):
        for endpoint in ('absolute_error', 'excess_error'):
            y = part[endpoint].to_numpy(float)
            for method, column in methods.items():
                x = part[column].to_numpy(float)
                records.append({'fold_id': fold, 'endpoint': endpoint, 'method': method,
                                'n_tasks': len(part), 'utility_20': utility(x, y),
                                'spearman': float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
                                'source_radius_coverage': float(part.target_within_source_radius_diagnostic.mean()),
                                'L_nonzero_fraction': float((part.L>0).mean())})
    results = pd.DataFrame(records)
    results.to_csv(OUT / 'E254_FOLD_RESULTS.csv', index=False)
    pivot = results.pivot(index=['fold_id', 'endpoint'], columns='method', values='utility_20').reset_index()
    for name in ('G', 'L', 'M_plus_H', 'M_plus_G', 'M_plus_L'):
        pivot['delta_'+name+'_vs_M'] = pivot[name]-pivot.M
    pivot.to_csv(OUT / 'E254_PAIRED.csv', index=False)
    summary = pivot.groupby('endpoint', as_index=False).agg(
        mean_M=('M', 'mean'), mean_G=('G', 'mean'), mean_L=('L', 'mean'),
        mean_M_plus_L=('M_plus_L', 'mean'),
        mean_delta_ML_vs_M=('delta_M_plus_L_vs_M', 'mean'),
        positive_folds_ML_vs_M=('delta_M_plus_L_vs_M', lambda x: int((x>0).sum())))
    summary.to_csv(OUT / 'E254_SUMMARY.csv', index=False)
    status = {'status': 'POST_E253_RETROSPECTIVE_DEVELOPMENT',
              'created_at': datetime.now().astimezone().isoformat(),
              'n_tasks': len(frame), 'source_radius_coverage': float(frame.target_within_source_radius_diagnostic.mean()),
              'bound_holds_fraction': float(frame.conditional_bound_holds_diagnostic.mean()),
              'L_nonzero_fraction': float((frame.L>0).mean()),
              'warning': 'A conditional triangle bound is not a distribution-free certificate; target coverage must be established separately.'}
    (OUT / 'E254_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
