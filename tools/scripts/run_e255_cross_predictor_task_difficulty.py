#!/usr/bin/env python3
"""E255 retrospective task-error sharing across GEARS and scGPT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/yyf/proj/docs/实验结果/E112_external_formal_dual_models_20260713')
OUT = ROOT / 'docs/实验结果/E255_cross_predictor_task_difficulty_20260924'
E251 = ROOT / 'docs/实验结果/E251_e112_upstream_capability_20260924/E112_TASK_CAPABILITY.csv'
A1_SOURCE = Path('/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_reliability_model_corrected_20260610/tables/RELIABILITY_ALL_SCORES.csv')
E205 = ROOT / 'docs/实验结果/E205_cross_family_disagreement_20260830/formal_evaluation/E205_TASK_METRICS.csv'
E201 = ROOT / 'docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/tables/E201_TASK_METRICS.csv'
SEED = 25520260924


def rank_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    xr, yr, zr = (rankdata(v).astype(float) for v in (x, y, z))
    design = np.column_stack((np.ones(len(zr)), zr))
    rx = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    ry = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def quantile_ci(values: list[float]) -> list[float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) < 100:
        return [float('nan'), float('nan')]
    return np.quantile(valid, [.025, .975]).astype(float).tolist()


def bootstrap_rho(frame: pd.DataFrame, unit: str, seed: int) -> tuple[list[float], int]:
    rng = np.random.default_rng(seed)
    if unit == 'task':
        groups = [np.asarray([i]) for i in range(len(frame))]
    elif unit == 'fold':
        groups = [np.flatnonzero(frame.fold_id.to_numpy() == key)
                  for key in sorted(frame.fold_id.unique())]
    else:
        raise ValueError(unit)
    scores = []
    x = frame.GEARS.to_numpy(float)
    y = frame.scGPT.to_numpy(float)
    for _ in range(2000):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in picked])
        scores.append(float(spearmanr(x[idx], y[idx]).statistic))
    return quantile_ci(scores), len(groups)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a1 = pd.read_csv(A1_SOURCE, usecols=['dataset_name', 'fold_id', 'split', 'task_key',
                                          'predictor_name', 'score_name', 'true_error_rmse',
                                          'true_effect_l2_norm'])
    a1 = a1.loc[a1.split.eq('test') & a1.score_name.eq('random')].copy()
    a1_pair = a1.pivot(index=['dataset_name','fold_id','task_key'],
                       columns='predictor_name', values='true_error_rmse').reset_index()
    effect = a1.drop_duplicates(['dataset_name','fold_id','task_key'])[
        ['dataset_name','fold_id','task_key','true_effect_l2_norm']]
    a1_pair = a1_pair.merge(effect, on=['dataset_name','fold_id','task_key'], validate='one_to_one')
    if len(a1_pair) != 4584 or a1_pair.isna().any().any():
        raise ValueError('A1 paired inventory changed')
    simple = []
    for dataset, sample in [('__pooled__', a1_pair), *list(a1_pair.groupby('dataset_name'))]:
        x = sample.V0StrongBaseline.to_numpy(float)
        y = sample.ContextSimBaseline.to_numpy(float)
        z = sample.true_effect_l2_norm.to_numpy(float)
        simple.append({'dataset': dataset, 'n_tasks': len(sample),
                       'rho_raw': float(spearmanr(x, y).statistic),
                       'rho_partial_control_true_effect_length': rank_partial(x,y,z)})
    pd.DataFrame(simple).to_csv(OUT / 'E255_ORIGINAL_SIMPLE_BASELINES.csv', index=False)
    modern = pd.read_csv(E205)
    modern = modern.loc[modern.analysis_stratum.eq('primary_ge30')].copy()
    controls = pd.read_csv(E201, usecols=['task_id', 'control_error'])
    modern = modern.merge(controls, on='task_id', validate='one_to_one')
    if len(modern) != 1808 or modern.task_id.nunique() != 1808:
        raise ValueError('E205 modern task inventory changed')
    modern['GAT_error'] = modern[[f'gat_seed_{i}_rmse' for i in range(1,5)]].mean(axis=1)
    modern['Exphormer_error'] = modern[[f'exphormer_seed_{i}_rmse' for i in range(1,5)]].mean(axis=1)
    modern_summary = []
    for target, sample in [('__pooled__',modern), *list(modern.groupby('target'))]:
        x = sample.GAT_error.to_numpy(float)
        y = sample.Exphormer_error.to_numpy(float)
        z = sample.control_error.to_numpy(float)
        modern_summary.append({'target': target, 'n_tasks': len(sample),
                    'rho_raw': float(spearmanr(x,y).statistic),
                    'rho_excess_over_nochange': float(spearmanr(x-z,y-z).statistic),
                    'rho_partial_control_nochange': rank_partial(x,y,z),
                    'GAT_vs_nochange_relative_gain': float(1-x.mean()/z.mean()),
                    'Exphormer_vs_nochange_relative_gain': float(1-y.mean()/z.mean())})
    pd.DataFrame(modern_summary).to_csv(OUT / 'E255_E205_MODERN_CROSS_ARCHITECTURE.csv', index=False)
    capability = pd.read_csv(E251)
    records = []
    summaries = []
    for dataset in ('Lara_exvivo', 'Santinha'):
        src = SOURCE / dataset
        raw = pd.read_csv(src / 'PREDICTION_RECORDS.csv')
        if raw.groupby(['fold_id', 'task_id']).predictor_name.nunique().ne(2).any():
            raise ValueError(f'{dataset}: expected paired two-model tasks')
        paired = raw.pivot(index=['fold_id', 'task_id', 'context'],
                           columns='predictor_name', values='true_error_rmse').reset_index()
        predictor_names = [name for name in paired.columns if name.endswith('finetuned') or name.endswith('trainonly_graphs')]
        if len(predictor_names) != 2:
            raise ValueError(f'{dataset}: predictor mapping changed: {predictor_names}')
        paired = paired.rename(columns={next(name for name in predictor_names if name.startswith('GEARS')): 'GEARS',
                                        next(name for name in predictor_names if name.startswith('scGPT')): 'scGPT'})
        truth_keys = raw.drop_duplicates(['fold_id', 'task_id'])[['fold_id', 'task_id', 'true_effect_key']]
        with np.load(src / 'arrays/true_effects.npz', allow_pickle=False) as truths:
            truth_keys['no_change'] = [float(np.sqrt(np.mean(np.asarray(truths[key], float)**2)))
                                       for key in truth_keys.true_effect_key]
        paired = paired.merge(truth_keys[['fold_id', 'task_id', 'no_change']],
                              on=['fold_id', 'task_id'], validate='one_to_one')
        setting = pd.read_csv(src / 'TASK_RISK_TABLE.csv')[['fold_id', 'task_id', 'setting']]
        paired = paired.merge(setting, on=['fold_id', 'task_id'], validate='one_to_one')
        if paired.isna().any().any():
            raise ValueError(f'{dataset}: missing paired values')
        paired['dataset'] = dataset
        paired['GEARS_excess'] = paired.GEARS - paired.no_change
        paired['scGPT_excess'] = paired.scGPT - paired.no_change
        records.append(paired)
        for scope, frame in [('all', paired), *[(name, part) for name, part in paired.groupby('setting')]]:
            if len(frame) < 10:
                continue
            row = {'dataset': dataset, 'setting': scope, 'n_tasks': len(frame),
                   'n_folds': frame.fold_id.nunique(),
                   'rho_raw': float(spearmanr(frame.GEARS, frame.scGPT).statistic),
                   'rho_excess_over_nochange': float(spearmanr(frame.GEARS_excess, frame.scGPT_excess).statistic),
                   'rho_partial_control_true_effect_nochange': rank_partial(frame.GEARS.to_numpy(float),
                                                                            frame.scGPT.to_numpy(float),
                                                                            frame.no_change.to_numpy(float)),
                   'GEARS_vs_nochange_relative_gain': float(1-frame.GEARS.mean()/frame.no_change.mean()),
                   'scGPT_vs_nochange_relative_gain': float(1-frame.scGPT.mean()/frame.no_change.mean())}
            ci, n = bootstrap_rho(frame.reset_index(drop=True), 'task', SEED + len(summaries))
            row['rho_raw_task_boot_ci_low'], row['rho_raw_task_boot_ci_high'] = ci
            row['n_task_clusters'] = n
            ci, n = bootstrap_rho(frame.reset_index(drop=True), 'fold', SEED + 100 + len(summaries))
            row['rho_raw_fold_boot_ci_low'], row['rho_raw_fold_boot_ci_high'] = ci
            row['n_fold_clusters'] = n
            summaries.append(row)
    pairs = pd.concat(records, ignore_index=True)
    pairs.to_csv(OUT / 'E255_PAIRED_TASK_ERRORS.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'E255_SUMMARY.csv', index=False)
    folds = capability.groupby(['dataset', 'predictor'], as_index=False).agg(
        n_folds=('fold_id', 'size'), n_competent=('model_rmse', lambda s: int((s.to_numpy() < capability.loc[s.index, 'no_change_rmse'].to_numpy()).sum())))
    folds.to_csv(OUT / 'E255_UPSTREAM_COMPETENCE.csv', index=False)
    status = {'status': 'RETROSPECTIVE_DIAGNOSTIC', 'n_matched_task_pairs': len(pairs),
              'n_original_simple_baseline_pairs': len(a1_pair),
              'n_modern_E205_cross_architecture_pairs': len(modern),
              'datasets': ['Lara_exvivo', 'Santinha'], 'bootstrap_reps': 2000,
              'nochange_uses_truth_and_is_not_deployable': True,
              'test_truth_previously_public': True}
    (OUT / 'E255_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
