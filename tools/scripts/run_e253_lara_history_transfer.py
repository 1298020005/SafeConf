#!/usr/bin/env python3
"""Retrospective source-train-only history transfer to E112 Lara GEARS."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path('/home/yyf/proj/docs/实验结果/E112_external_formal_dual_models_20260713/Lara_exvivo')
MANIFEST = Path('/home/yyf/proj/docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv')
CACHE = Path('/home/yyf/data/safeconf_e112_external/Lara_exvivo_CONTROL_ONLY_512.npz')
COMPETENCE = ROOT / 'docs/实验结果/E251_e112_upstream_capability_20260924/E112_TASK_CAPABILITY.csv'
OUT = ROOT / 'docs/实验结果/E253_lara_history_transfer_20260924'
PREDICTOR = 'GEARS_context_mean_trainonly_graphs'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utility(score: np.ndarray, errors: np.ndarray) -> float:
    n = len(score)
    k = max(1, int(np.ceil(.2*n)))
    def chosen(values: np.ndarray) -> np.ndarray:
        threshold = np.partition(values, n-k)[n-k]
        above = values > threshold
        tied = values == threshold
        weights = above.astype(float)
        weights[tied] = (k - int(above.sum())) / int(tied.sum())
        return weights
    selected, oracle = chosen(score), chosen(errors)
    average = float(np.mean(errors))
    denominator = float(np.dot(oracle, errors) / k - average)
    return float((np.dot(selected, errors) / k - average) / denominator) if denominator > 1e-12 else float('nan')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest.loc[manifest.dataset.eq('Lara_exvivo')]
    task = pd.read_csv(SOURCE_ROOT / 'TASK_RISK_TABLE.csv')
    task = task.loc[task.setting.eq('context_unseen_row')].copy()
    if len(task) != 120 or task.fold_id.nunique() != 5:
        raise ValueError('Lara main task inventory changed')
    capability = pd.read_csv(COMPETENCE)
    capability = capability.loc[capability.dataset.eq('Lara_exvivo') & capability.predictor.eq(PREDICTOR)]
    task = task.merge(capability[['fold_id', 'task_id', 'model_rmse', 'no_change_rmse']],
                      on=['fold_id', 'task_id'], validate='one_to_one')
    if len(task) != 120:
        raise ValueError('missing capability records')
    records = pd.read_csv(SOURCE_ROOT / 'PREDICTION_RECORDS.csv')
    records = records.loc[records.predictor_name.eq(PREDICTOR),
                          ['fold_id', 'task_id', 'predicted_effect_key']]
    task = task.merge(records, on=['fold_id', 'task_id'], validate='one_to_one')
    with np.load(CACHE, allow_pickle=False) as cached:
        contexts = cached['contexts'].astype(str).tolist()
        perts = cached['perturbations'].astype(str).tolist()
        effects = cached['effects'].reshape(len(contexts), len(perts), 512)
    context_index = {name: i for i, name in enumerate(contexts)}
    pert_index = {name: i for i, name in enumerate(perts)}
    allowed = set(map(tuple, manifest.loc[manifest.split.eq('train') & manifest.in_train_fraction_100,
                                          ['fold_id', 'context', 'perturbation']].astype(str).itertuples(index=False, name=None)))
    rows = []
    unsupported = []
    with np.load(SOURCE_ROOT / 'arrays/predicted_effects.npz', allow_pickle=False) as pred:
        for item in task.itertuples(index=False):
            source_contexts = [name for name in contexts if (item.fold_id, name, item.perturbation) in allowed]
            if item.context in source_contexts:
                raise ValueError(f'{item.task_id}: held-out context in source history')
            if len(source_contexts) < 2:
                unsupported.append({'fold_id': str(item.fold_id), 'task_id': str(item.task_id),
                                    'perturbation': str(item.perturbation),
                                    'n_train_source_contexts': len(source_contexts),
                                    'reason': 'H requires at least two train source contexts'})
                continue
            source = np.stack([effects[context_index[name], pert_index[item.perturbation]]
                               for name in source_contexts])
            pred_effect = np.asarray(pred[str(item.predicted_effect_key)], dtype=np.float64)
            if pred_effect.shape != (512,):
                raise ValueError('prediction gene axis mismatch')
            source_mean = source.mean(axis=0)
            rows.append({'fold_id': str(item.fold_id), 'task_id': str(item.task_id),
                         'context': str(item.context), 'perturbation': str(item.perturbation),
                         'n_train_source_contexts': len(source_contexts),
                         'M': float(np.sqrt(np.mean(pred_effect**2))),
                         'D': float(item.risk_model_disagreement),
                         'H': float(np.sqrt(np.mean((source-source_mean)**2))),
                         'G': float(np.sqrt(np.mean((pred_effect-source_mean)**2))),
                         'absolute_error': float(item.model_rmse),
                         'no_change_error': float(item.no_change_rmse),
                         'excess_error': float(item.model_rmse-item.no_change_rmse)})
    frame = pd.DataFrame(rows)
    if len(frame) != 116 or len(unsupported) != 4 or frame.isna().any().any():
        raise ValueError('incomplete feature table')
    pd.DataFrame(unsupported).to_csv(OUT / 'E253_UNSUPPORTED_TASKS.csv', index=False)
    for fold, index in frame.groupby('fold_id').indices.items():
        part = frame.iloc[index]
        if len(part) < 22 or len(part) > 24:
            raise ValueError(f'{fold}: unexpected supported count {len(part)}')
        ranks = {name: rankdata(part[name].to_numpy(float)) / len(part) for name in ('M', 'D', 'H', 'G')}
        for name in ('M', 'D', 'H', 'G'):
            frame.loc[frame.index[index], f'rank_{name}'] = ranks[name]
        frame.loc[frame.index[index], 'M_plus_H'] = .8*ranks['M'] + .2*ranks['H']
        frame.loc[frame.index[index], 'M_plus_D'] = .8*ranks['M'] + .2*ranks['D']
        frame.loc[frame.index[index], 'M_plus_G'] = .8*ranks['M'] + .2*ranks['G']
    frame.to_csv(OUT / 'E253_ALL_120_TASKS.csv', index=False)
    methods = {'M': 'rank_M', 'D': 'rank_D', 'H': 'rank_H', 'G': 'rank_G',
               'M_plus_H': 'M_plus_H', 'M_plus_D': 'M_plus_D', 'M_plus_G': 'M_plus_G'}
    results = []
    for fold, endpoint, part in ((fold, endpoint, group) for fold, group in frame.groupby('fold_id')
                                 for endpoint in ('absolute_error', 'excess_error')):
        y = part[endpoint].to_numpy(float)
        for method, column in methods.items():
            x = part[column].to_numpy(float)
            results.append({'fold_id': fold, 'endpoint': endpoint, 'method': method,
                            'n_tasks': len(part), 'utility_20': utility(x, y),
                            'spearman': float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
                            'model_mean_rmse': float(part.absolute_error.mean()),
                            'no_change_mean_rmse': float(part.no_change_error.mean())})
    out = pd.DataFrame(results)
    out.to_csv(OUT / 'E253_FOLD_RESULTS.csv', index=False)
    paired = out.pivot(index=['fold_id', 'endpoint'], columns='method', values='utility_20').reset_index()
    paired['delta_MH_vs_M'] = paired.M_plus_H - paired.M
    paired['delta_MH_vs_MD'] = paired.M_plus_H - paired.M_plus_D
    paired['delta_MH_vs_MG'] = paired.M_plus_H - paired.M_plus_G
    paired.to_csv(OUT / 'E253_PAIRED.csv', index=False)
    summary = paired.groupby('endpoint', as_index=False).agg(
        mean_M=('M', 'mean'), mean_M_plus_H=('M_plus_H', 'mean'),
        mean_delta_MH_vs_M=('delta_MH_vs_M', 'mean'),
        positive_folds_MH_vs_M=('delta_MH_vs_M', lambda x: int((x > 0).sum())),
        mean_delta_MH_vs_MD=('delta_MH_vs_MD', 'mean'),
        mean_delta_MH_vs_MG=('delta_MH_vs_MG', 'mean'))
    summary.to_csv(OUT / 'E253_SUMMARY.csv', index=False)
    status = {'status': 'RETROSPECTIVE_EXTERNAL_STUDY_AUDIT',
              'created_at': datetime.now().astimezone().isoformat(),
              'n_tasks': len(frame), 'n_unsupported_tasks': len(unsupported),
              'n_folds': frame.fold_id.nunique(),
              'all_folds_GEARS_beats_no_change': bool((out.loc[out.method.eq('M')].model_mean_rmse
                                                      < out.loc[out.method.eq('M')].no_change_mean_rmse).all()),
              'manifest_sha256': sha(MANIFEST), 'cache_sha256': sha(CACHE),
              'task_table_sha256': sha(OUT / 'E253_ALL_120_TASKS.csv'),
              'truth_previously_public': True,
              'cache_physical_isolation': False}
    (OUT / 'E253_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
