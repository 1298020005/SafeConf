#!/usr/bin/env python3
"""Nested cross-study expected-error fitting; development only, never E208."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_e221_nested_study_adaptation import load, metrics, sha256

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'docs/实验结果/E222_error_regression_transfer_20260921'
ERROR = 'error_two_predictor_mean_rmse'
FEATURES = {'magnitude': ['m'], 'full': ['m', 's', 'd']}
CANDIDATES = ('ridge_1', 'ridge_10', 'hgb_7')
METHODS = ('magnitude', 'fixed_80m_20s', 'learned_magnitude', 'learned_full', 'guarded_full')
METRICS = ('spearman', 'utility_20', 'capture_20', 'remaining_relative_error_80')
SEED = 20260921


def training_arrays(frame: pd.DataFrame, columns: list[str]):
    """All labels, normalization and sample weights come from training only."""
    y = frame[ERROR].to_numpy(float)
    scale = frame.groupby(['dataset', 'fold_id'])[ERROR].transform('mean').to_numpy(float)
    counts = frame.groupby(['dataset', 'fold_id']).dataset.transform('size').to_numpy(float)
    folds = frame[['dataset', 'fold_id']].drop_duplicates().groupby('dataset').size()
    nfold = frame.dataset.map(folds).to_numpy(float)
    if np.any(scale <= 0) or not np.isfinite(y).all():
        raise ValueError('Training errors must be finite with positive fold mean')
    weights = 1.0 / (counts * nfold)
    weights *= len(frame) / weights.sum()
    return frame[columns].to_numpy(float), y / scale, weights


def fit_predict(train: pd.DataFrame, test_features: pd.DataFrame, family: str, candidate: str):
    """test_features needs only m/s/d, never any held-out error column."""
    columns = FEATURES[family]
    x, y, weight = training_arrays(train, columns)
    xtest = test_features[columns].to_numpy(float)
    if candidate.startswith('ridge_'):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        x = poly.fit_transform(x)
        xtest = poly.transform(xtest)
        model = Ridge(alpha=float(candidate.split('_')[1]))
    elif candidate == 'hgb_7':
        model = HistGradientBoostingRegressor(max_iter=80, learning_rate=.05,
                    max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=1,
                    early_stopping=False, random_state=SEED)
    else:
        raise ValueError(candidate)
    model.fit(x, y, sample_weight=weight)
    result = model.predict(xtest)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite model prediction')
    return result


def evaluate_blocks(frame: pd.DataFrame, score: np.ndarray):
    rows = []
    for (study, fold), idx in frame.groupby(['dataset', 'fold_id'], sort=True).indices.items():
        block = frame.iloc[idx]
        y = block[ERROR].to_numpy(float)
        record = metrics(np.asarray(score)[idx], y)
        record['remaining_relative_error_80'] = record.pop('remaining_error_80') / y.mean()
        rows.append({'dataset': study, 'fold_id': fold, 'n_tasks': len(idx), **record})
    return rows


def select_inner(train: pd.DataFrame, outer: str):
    rows = []
    for inner in sorted(train.dataset.unique()):
        fit = train.loc[train.dataset.ne(inner)].copy()
        valid = train.loc[train.dataset.eq(inner)].copy().reset_index(drop=True)
        base = evaluate_blocks(valid, valid.m.to_numpy())
        base_utility = float(np.mean([x['utility_20'] for x in base]))
        for family in FEATURES:
            for candidate in CANDIDATES:
                pred = fit_predict(fit, valid[['m', 's', 'd']], family, candidate)
                measures = pd.DataFrame(evaluate_blocks(valid, pred))
                rows.append({'outer_study': outer, 'inner_study': inner,
                             'training_studies': '|'.join(sorted(fit.dataset.unique())),
                             'feature_group': family, 'candidate': candidate,
                             **{x: float(measures[x].mean()) for x in METRICS},
                             'delta_utility_20': float(measures.utility_20.mean() - base_utility)})
    table = pd.DataFrame(rows)
    chosen = {}
    for family in FEATURES:
        means = table.loc[table.feature_group.eq(family)].groupby('candidate').utility_20.mean()
        chosen[family] = max(CANDIDATES, key=lambda key: (means[key], -CANDIDATES.index(key)))
    full = table[(table.feature_group == 'full') & (table.candidate == chosen['full'])]
    admitted = bool(full.delta_utility_20.mean() >= .02 and (full.delta_utility_20 > 0).sum() >= 5)
    return chosen, admitted, table


def outer_job(args):
    frame, outer = args
    with threadpool_limits(limits=1):
        train = frame.loc[frame.dataset.ne(outer)].copy()
        test = frame.loc[frame.dataset.eq(outer)].copy().reset_index(drop=True)
        chosen, admitted, inner = select_inner(train, outer)
        # Freeze all decisions and predictions before consulting outer outcomes.
        predicted = {family: fit_predict(train, test[['m', 's', 'd']], family, chosen[family])
                     for family in FEATURES}
        scores = {'magnitude': test.m.to_numpy(), 'fixed_80m_20s': .8*test.m.to_numpy()+.2*test.s.to_numpy(),
                  'learned_magnitude': predicted['magnitude'], 'learned_full': predicted['full'],
                  'guarded_full': predicted['full'] if admitted else test.m.to_numpy()}
        selected = {'outer_study': outer, 'training_studies': '|'.join(sorted(train.dataset.unique())),
                    'n_train': len(train), 'n_test': len(test), 'magnitude_candidate': chosen['magnitude'],
                    'full_candidate': chosen['full'], 'full_admitted': admitted}
        prediction = test[['dataset', 'fold_id', 'task_id', 'setting', 'perturbation', ERROR]].copy()
        rows, settings = [], []
        for name, value in scores.items():
            prediction[name] = value
            rows.extend({'method': name, **r} for r in evaluate_blocks(test, value))
            # Descriptive within-setting budgets; never used to select models.
            for (fold, setting), idx in test.groupby(['fold_id', 'setting'], sort=True).indices.items():
                if len(idx) < 4:
                    settings.append({'dataset': outer, 'fold_id': fold, 'setting': setting,
                                     'method': name, 'n_tasks': len(idx), 'status': 'TOO_SMALL'})
                    continue
                y = test.iloc[idx][ERROR].to_numpy(float)
                v = metrics(value[idx], y)
                v['remaining_relative_error_80'] = v.pop('remaining_error_80') / y.mean()
                settings.append({'dataset': outer, 'fold_id': fold, 'setting': setting,
                                 'method': name, 'n_tasks': len(idx), 'status': 'EVALUATED', **v})
        return pd.DataFrame(rows), inner, selected, prediction, pd.DataFrame(settings)


def summarize(folds: pd.DataFrame, n_bootstrap=5000):
    studies = folds.groupby(['dataset', 'method'])[list(METRICS)].mean().reset_index()
    ids = sorted(studies.dataset.unique())
    draws = np.random.default_rng(SEED).integers(len(ids), size=(n_bootstrap, len(ids)))
    rows = []
    for base in ('magnitude', 'learned_magnitude'):
        b = studies[studies.method.eq(base)].set_index('dataset').loc[ids, list(METRICS)].to_numpy()
        for method in METHODS:
            a = studies[studies.method.eq(method)].set_index('dataset').loc[ids, list(METRICS)].to_numpy()
            delta = a - b
            sample = delta[draws].mean(axis=1)
            for j, metric in enumerate(METRICS):
                rows.append({'method': method, 'baseline': base, 'metric': metric,
                             'mean': float(a[:, j].mean()), 'delta': float(delta[:, j].mean()),
                             'ci95_lower': float(np.quantile(sample[:, j], .025)),
                             'ci95_upper': float(np.quantile(sample[:, j], .975)),
                             'positive_studies': int((delta[:, j] > 0).sum()),
                             'favorable_studies': int((delta[:, j] < 0).sum()) if metric.startswith('remaining') else int((delta[:, j] > 0).sum()),
                             'n_studies': len(ids)})
    return studies, pd.DataFrame(rows)


def figures(studies: pd.DataFrame, intervals: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'pdf.fonttype': 42})
    order = sorted(studies.dataset.unique())
    baseline = studies[studies.method.eq('magnitude')].set_index('dataset').loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for method, color, offset, label in [('fixed_80m_20s','#70859a',-.16,'Fixed fusion'),
                ('learned_magnitude','#b8895b',0,'Learned magnitude'),
                ('learned_full','#20877d',.16,'Learned full features')]:
        block = studies[studies.method.eq(method)].set_index('dataset').loc[order]
        axes[0].scatter(block.utility_20-baseline.utility_20, np.arange(len(order))+offset,
                        s=22, color=color, label=label)
    axes[0].axvline(0, color='#888888', linewidth=.7)
    axes[0].set_yticks(np.arange(len(order)), order)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Review utility difference vs magnitude')
    axes[0].legend(frameon=False, fontsize=8, loc='best')
    names = ['fixed_80m_20s','learned_magnitude','learned_full','guarded_full']
    block = intervals[(intervals.baseline == 'magnitude') & (intervals.metric == 'utility_20')].set_index('method').loc[names]
    axes[1].hlines(np.arange(4), block.ci95_lower, block.ci95_upper, color='#20877d')
    axes[1].scatter(block.delta, np.arange(4), color='#20877d', s=26)
    axes[1].axvline(0, color='#888888', linewidth=.7)
    axes[1].set_yticks(np.arange(4), ['Fixed fusion','Learned magnitude','Learned full','Guarded full'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Equal-study difference (descriptive 95% interval)')
    for suffix in ('svg','pdf','png'):
        fig.savefig(output / ('study_transfer_controls.'+suffix), dpi=300, facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('workers must be 1--4')
    if (args.output / 'RUN_STATUS.json').exists() and not args.overwrite:
        raise ValueError('Refuse to overwrite an existing run; supply a new output path')
    start = time.monotonic()
    frame = load(args.input).sort_values(['dataset','fold_id','task_id'], kind='stable').reset_index(drop=True)
    fingerprint = sha256(args.input)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    outer_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(outer_job, [(frame, study) for study in sorted(frame.dataset.unique())]):
            outer_results.append(result)
            print('Completed',result[2]['outer_study'],flush=True)
    folds = pd.concat([x[0] for x in outer_results], ignore_index=True)
    studies, intervals = summarize(folds)
    tables = {'FOLD_RESULTS': folds, 'STUDY_RESULTS': studies, 'INTERVALS': intervals,
              'INNER_VALIDATION': pd.concat([x[1] for x in outer_results], ignore_index=True),
              'OUTER_SELECTIONS': pd.DataFrame([x[2] for x in outer_results]),
              'HELDOUT_PREDICTIONS': pd.concat([x[3] for x in outer_results], ignore_index=True),
              'SETTING_RESULTS': pd.concat([x[4] for x in outer_results], ignore_index=True)}
    for name, data in tables.items():
        data.to_csv(output / (name+'.csv'), index=False)
    principal = intervals[(intervals.method == 'learned_full') & (intervals.metric == 'utility_20')].set_index('baseline')
    passed = bool(principal.loc['magnitude','delta'] >= .02 and
                  principal.loc['magnitude','ci95_lower'] > 0 and
                  principal.loc['learned_magnitude','ci95_lower'] > 0 and
                  principal.loc['magnitude','positive_studies'] >= 6)
    figures(studies, intervals, output)
    if sha256(args.input) != fingerprint:
        raise ValueError('Input changed during execution')
    status = {'experiment':'E222_error_regression_transfer','status':'COMPLETE',
              'development_gate':'PASS' if passed else 'NOT_SUPPORTED',
              'evidence_class':'retrospective_nested_study_validation',
              'created_at':datetime.now().astimezone().isoformat(),
              'input_sha256':fingerprint,'script_sha256':sha256(Path(__file__)),
              'helper_sha256':sha256(Path(__file__).with_name('run_e221_nested_study_adaptation.py')),
              'n_tasks':len(frame),'n_studies':frame.dataset.nunique(),
              'candidate_algorithms':list(CANDIDATES),'feature_groups':FEATURES,
              'fit_count':8*(7*2*3+2),'e208_rows_read':0,'e208_truth_used':False,
              'source_data_modified':False,'all_studies_retained':True,'workers':args.workers,
              'elapsed_seconds':time.monotonic()-start, 'versions':{'python':platform.python_version(),
                  'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
              'utility20':principal.reset_index().to_dict(orient='records')}
    (output/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# E222：直接拟合误差的跨研究验证','',
             '这次实际训练了轻量风险评分器，未重训上游扰动预测器。每次外层留一研究；内层六研究训练、一个研究验证。',
             '', '| 输入与比较 | utility@20% 差值 | 描述性 95% 区间 | 正方向研究 |',
             '|---|---:|---|---:|']
    for base, row in principal.iterrows():
        lines.append(f'| 完整特征相对 {base} | {row.delta:+.5f} | [{row.ci95_lower:+.5f}, {row.ci95_upper:+.5f}] | {int(row.positive_studies)}/8 |')
    lines += ['', f'预定开发门：**{status["development_gate"]}**。所有研究保留，未根据结果改变主指标或删除任务。',
              '', '## 该结果的范围', '',
              '- 历史评分标签来自双预测器平均 RMSE；不等同于 TxPert 平均预测误差，也不等同于五特征 E201 分数。',
              '- 幅度单项学习组得到相同的标签和调参预算。完整组胜过原幅度而未胜过它时，不能把提高归给 SafeConf 特征。',
              '- 区间在固定外层预测上按研究重采样；八个研究的训练集重叠，未包含完整重训练不确定性。',
              '- 本轮是已公开数据上的开发检验。即使通过，也需要新的外部确认；E208 的主公式未修改。',
              '- 正文图为白底 SVG/PDF/PNG；每个研究都画出，负方向不隐藏。',
              '', '## 复算', '',
              '```bash', 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/scripts/run_e222_error_regression_transfer.py --input <E153_ABSOLUTE_TASK_INPUT.csv> --output <new_output_dir> --workers 4','```','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
