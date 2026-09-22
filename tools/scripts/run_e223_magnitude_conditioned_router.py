#!/usr/bin/env python3
"""Nested leave-study-out development of magnitude-anchored routing rules.

This is a finite, predeclared rule dictionary, not a fitted upstream predictor.
Outer-study labels are never used to choose the formula.  E208 is untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path('/home/yyf/proj/docs/实验结果/E187_advisor_difficulty_certificate_20260726/tables/E187_CARTESIAN_TASK_CERTIFICATES.csv')
DEFAULT_OUTPUT = ROOT / 'docs/实验结果/E223_magnitude_conditioned_router_20260922'
INPUT_SHA256 = 'c84cb0f2b8c36c27b33d62cfbad7e98d2228288a85937e18351ab13d069d7ba0'
SEED = 20260922
BUDGET = 0.20
ERROR = 'error_two_predictor_mean_rmse'
GROUPS = ['dataset', 'fold_id', 'setting', 'train_fraction']
SCENARIOS = (
    'context_unseen_row', 'perturbation_unseen_column',
    'random_missing_pair', 'context_and_perturbation_unseen',
)


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pct(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if len(values) < 4 or not np.isfinite(values).all():
        raise ContractError('invalid feature block')
    return rankdata(values, method='average') / len(values)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, block in frame.groupby(GROUPS, sort=True):
        block = block.copy()
        block['m'] = pct(block.baseline_predicted_magnitude.to_numpy())
        block['s'] = pct(block.safeconf_calibrated_pair_risk.to_numpy())
        block['d'] = pct(block.model_disagreement_rmse.to_numpy())
        block['cn'] = pct(block.context_novelty_scaled.to_numpy())
        block['pn'] = pct(block.perturbation_novelty.to_numpy())
        # Low historical support is higher scarcity and therefore higher risk.
        block['ss'] = pct(-np.log1p(block.training_support_count.to_numpy(float)))
        out.append(block)
    return pd.concat(out, ignore_index=True)


def load(path: Path) -> pd.DataFrame:
    if not path.is_file() or sha256(path) != INPUT_SHA256:
        raise ContractError('E187 input is missing or has changed')
    frame = pd.read_csv(path)
    required = set(GROUPS + ['dataset', 'setting', 'train_fraction', ERROR,
        'baseline_predicted_magnitude', 'safeconf_calibrated_pair_risk',
        'model_disagreement_rmse', 'context_novelty_scaled',
        'perturbation_novelty', 'training_support_count', 'task_instance_id'])
    if not required.issubset(frame.columns) or len(frame) != 8196:
        raise ContractError('E187 schema or row count changed')
    if frame.dataset.nunique() != 4 or set(frame.setting.unique()) != set(SCENARIOS):
        raise ContractError('study or scenario identity changed')
    if frame.task_instance_id.duplicated().any() or not np.isfinite(frame[ERROR]).all():
        raise ContractError('task identity or outcome invalid')
    if frame[['certificate_uses_target_truth', 'truth_used_for_evaluation_only']].isna().any().any():
        raise ContractError('truth-use audit fields missing')
    return add_features(frame.sort_values(GROUPS + ['task_instance_id'], kind='stable').reset_index(drop=True))


def rules() -> list[dict]:
    out = [
        {'name': 'magnitude', 'kind': 'linear', 'params': {}},
        {'name': 'fixed_80m_20s', 'kind': 'linear', 'params': {'s': .2}},
    ]
    for lam in (.25, .5, .75, 1.0):
        out.append({'name': f'one_sided_s{lam:.2f}', 'kind': 'one_sided', 'params': {'s': lam}})
    maps = {
        'scenario_a': {'context_unseen_row': .25, 'perturbation_unseen_column': .75,
                       'random_missing_pair': .25, 'context_and_perturbation_unseen': 0.0},
        'scenario_b': {'context_unseen_row': .50, 'perturbation_unseen_column': .50,
                       'random_missing_pair': .50, 'context_and_perturbation_unseen': .25},
        'scenario_c': {'context_unseen_row': .25, 'perturbation_unseen_column': 1.0,
                       'random_missing_pair': .25, 'context_and_perturbation_unseen': 0.0},
        'scenario_d': {'context_unseen_row': .50, 'perturbation_unseen_column': 1.0,
                       'random_missing_pair': .50, 'context_and_perturbation_unseen': .0},
    }
    for name, mapping in maps.items():
        out.append({'name': name, 'kind': 'scenario_one_sided', 'params': mapping})
    # One fixed evidence extension tests whether SafeConf's extra information
    # survives after magnitude is kept as the first-stage anchor.
    out.append({'name': 'anchored_evidence', 'kind': 'anchored_evidence',
                'params': {'s': .25, 'd': .25, 'cn': .25, 'pn': .15, 'ss': .10}})
    return out


def score(block: pd.DataFrame, rule: dict) -> np.ndarray:
    m = block.m.to_numpy(float)
    if rule['kind'] == 'linear':
        if rule['name'] == 'magnitude':
            return m
        return (1 - rule['params']['s']) * m + rule['params']['s'] * block.s.to_numpy(float)
    if rule['kind'] == 'one_sided':
        return m + rule['params']['s'] * np.maximum(block.s.to_numpy(float) - m, 0)
    if rule['kind'] == 'scenario_one_sided':
        lam = block.setting.map(rule['params']).to_numpy(float)
        return m + lam * np.maximum(block.s.to_numpy(float) - m, 0)
    if rule['kind'] == 'anchored_evidence':
        p = rule['params']
        return m + p['s'] * np.maximum(block.s.to_numpy(float) - m, 0) \
            + p['d'] * np.maximum(block.d.to_numpy(float) - m, 0) \
            + p['cn'] * np.maximum(block.cn.to_numpy(float) - m, 0) \
            + p['pn'] * np.maximum(block.pn.to_numpy(float) - m, 0) \
            + p['ss'] * np.maximum(block.ss.to_numpy(float) - m, 0)
    raise ContractError(f'unknown rule {rule}')


def metrics(values: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    values, outcome = np.asarray(values, float), np.asarray(outcome, float)
    if len(values) < 4 or not np.isfinite(values).all() or not np.isfinite(outcome).all():
        raise ContractError('invalid metric input')
    rho = np.corrcoef(rankdata(values, method='average'), rankdata(outcome, method='average'))[0, 1]
    k = max(1, math.ceil(BUDGET * len(outcome)))
    top = np.argsort(-values, kind='mergesort')[:k]
    oracle = np.argsort(-outcome, kind='mergesort')[:k]
    mean = float(outcome.mean())
    denom = float(outcome[oracle].mean() - mean)
    utility = (float(outcome[top].mean() - mean) / denom) if denom > 1e-15 else float('nan')
    capture = float(outcome[top].sum() / outcome.sum()) if outcome.sum() > 0 else float('nan')
    remaining = float((outcome.sum() - outcome[top].sum()) / max(1, len(outcome) - k))
    return {'spearman': float(rho), 'utility_20': utility, 'capture_20': capture,
            'remaining_error_80': remaining}


METRICS = ('spearman', 'utility_20', 'capture_20', 'remaining_error_80')


def eval_blocks(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    rows = []
    values = score(frame, rule)
    for key, idx in frame.groupby(GROUPS, sort=True).indices.items():
        block = frame.iloc[idx]
        rows.append(dict(zip(GROUPS, key), rule=rule['name'], n_tasks=len(block),
                             **metrics(values[idx], block[ERROR].to_numpy(float))))
    return pd.DataFrame(rows)


def choose_inner(train: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for study in sorted(train.dataset.unique()):
        valid = train.loc[train.dataset.eq(study)]
        for rule in rules():
            vals = eval_blocks(valid, rule)
            rows.append({'inner_heldout_study': study, **{k: float(vals[k].mean()) for k in METRICS},
                         'rule': rule['name']})
    table = pd.DataFrame(rows)
    ranking = []
    for rule in rules():
        sub = table.loc[table.rule.eq(rule['name'])]
        complexity = len(rule['params'])
        ranking.append((float(sub.utility_20.mean()), float(sub.spearman.mean()), -complexity,
                        rule['name'], rule))
    chosen = max(ranking, key=lambda row: (row[0], row[1], row[2], row[3]))
    return chosen[4], table


def outer_eval(frame: pd.DataFrame):
    fold_rows, inner_rows, selections, setting_rows = [], [], [], []
    for heldout in sorted(frame.dataset.unique()):
        train = frame.loc[frame.dataset.ne(heldout)]
        test = frame.loc[frame.dataset.eq(heldout)].copy()
        chosen, inner = choose_inner(train)
        inner['outer_heldout_study'] = heldout
        inner_rows.append(inner)
        selections.append({'outer_heldout_study': heldout, 'selected_rule': chosen['name'],
                           'n_train_tasks': len(train), 'n_test_tasks': len(test)})
        for rule in rules():
            vals = eval_blocks(test, rule)
            vals['outer_heldout_study'] = heldout
            fold_rows.append(vals)
            selected = vals.loc[vals.rule.eq(rule['name'])].copy()
            if rule['name'] == chosen['name']:
                selected['nested_selected'] = True
            for setting, block in test.groupby('setting', sort=True):
                # Descriptive only: settings never select the outer rule.
                sub = block.copy()
                v = metrics(score(sub, rule), sub[ERROR].to_numpy(float))
                setting_rows.append({'outer_heldout_study': heldout, 'setting': setting,
                                     'rule': rule['name'], 'n_tasks': len(sub), **v})
    return (pd.concat(fold_rows, ignore_index=True), pd.concat(inner_rows, ignore_index=True),
            pd.DataFrame(selections), pd.DataFrame(setting_rows))


def bootstrap(fold_results: pd.DataFrame, n: int = 10000) -> pd.DataFrame:
    # Average folds within study, then resample four studies as clusters.
    study = fold_results.groupby(['outer_heldout_study', 'rule'])[list(METRICS)].mean().reset_index()
    studies = sorted(study.outer_heldout_study.unique())
    base = study.loc[study.rule.eq('magnitude')].set_index('outer_heldout_study').loc[studies]
    rng = np.random.default_rng(SEED)
    rows = []
    for rule in sorted(study.rule.unique()):
        cur = study.loc[study.rule.eq(rule)].set_index('outer_heldout_study').loc[studies]
        delta = cur[list(METRICS)].to_numpy() - base[list(METRICS)].to_numpy()
        draws = delta[rng.integers(0, len(studies), size=(n, len(studies)))].mean(axis=1)
        for j, metric in enumerate(METRICS):
            rows.append({'rule': rule, 'metric': metric, 'delta_vs_magnitude': float(delta[:, j].mean()),
                         'ci95_lower': float(np.quantile(draws[:, j], .025)),
                         'ci95_upper': float(np.quantile(draws[:, j], .975)),
                         'positive_studies': int((delta[:, j] > 0).sum()), 'n_studies': len(studies)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.bootstrap < 1000:
        raise ValueError('use at least 1000 bootstrap draws')
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise ContractError(f'refusing to overwrite non-empty {args.output}')
    frame = load(args.input.resolve())
    input_hash = sha256(args.input.resolve())
    fold, inner, selections, setting = outer_eval(frame)
    interval = bootstrap(fold, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    fold.to_csv(args.output / 'E223_OUTER_BLOCK_RESULTS.csv', index=False)
    inner.to_csv(args.output / 'E223_INNER_RULE_SELECTION.csv', index=False)
    selections.to_csv(args.output / 'E223_OUTER_SELECTIONS.csv', index=False)
    setting.to_csv(args.output / 'E223_SETTING_RESULTS.csv', index=False)
    interval.to_csv(args.output / 'E223_STUDY_BOOTSTRAP_INTERVALS.csv', index=False)
    selected = interval.loc[(interval.rule.isin(selections.selected_rule.unique())) & (interval.metric.eq('utility_20'))]
    # The nested candidate is reported separately by outer selection; this gate
    # is deliberately descriptive and cannot turn a mixed result into a pass.
    nested_rules = selections.selected_rule.value_counts().to_dict()
    nested_rows = fold.loc[fold.rule.isin(nested_rules)].copy()
    nested_study = nested_rows.groupby(['outer_heldout_study', 'rule']).utility_20.mean().reset_index()
    base = nested_study.loc[nested_study.rule.eq('magnitude')].set_index('outer_heldout_study')
    nested_delta = []
    for study in sorted(nested_study.outer_heldout_study.unique()):
        rule = selections.loc[selections.outer_heldout_study.eq(study), 'selected_rule'].iloc[0]
        if rule == 'magnitude':
            val = 0.0
        else:
            val = float(nested_study.loc[(nested_study.outer_heldout_study.eq(study)) & (nested_study.rule.eq(rule)), 'utility_20'].iloc[0] - base.loc[study, 'utility_20'])
        nested_delta.append({'outer_heldout_study': study, 'selected_rule': rule, 'delta_utility_20_vs_magnitude': val})
    nested_table = pd.DataFrame(nested_delta)
    nested_mean = float(nested_table.delta_utility_20_vs_magnitude.mean())
    status = {'experiment': 'E223_magnitude_conditioned_router', 'status': 'COMPLETE',
              'evidence_class': 'retrospective_nested_leave_study_out_rule_dictionary',
              'input_sha256': input_hash, 'script_sha256': sha256(Path(__file__)),
              'n_tasks': len(frame), 'n_studies': int(frame.dataset.nunique()),
              'n_rules': len(rules()), 'n_outer_selections': len(selections),
              'outer_truth_used_for_selection': False, 'e208_truth_used': False,
              'source_data_modified': False, 'all_studies_retained': True,
              'nested_selected_rule_counts': nested_rules,
              'nested_mean_utility_delta': nested_mean,
              'nested_positive_outer_studies': int((nested_table.delta_utility_20_vs_magnitude > 0).sum()),
              'bootstrap': args.bootstrap, 'seed': SEED}
    nested_table.to_csv(args.output / 'E223_NESTED_SELECTED_DELTAS.csv', index=False)
    (args.output / 'RUN_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# E223：幅度锚定与场景条件化路由', '',
             '本轮固定候选规则，用外层留一研究评估；未删除不利任务，未读取 E208 测试答案。', '',
             '## 读取方法', '',
             f'- 输入 {len(frame)} 个任务，4 个研究、4 种缺失场景、4 个训练比例。',
             '- 每个任务块内把幅度、SafeConf、分歧、新颖度和支持稀缺度转为百分位。',
             '- 只在外层训练研究内部选规则，外层研究的真实误差只在最后一次评估。',
             '', '## 解释规则', '',
             f'- 嵌套选择后的平均 top-20% 复核效用差值：**{nested_mean:+.5f}**；外层正向研究：**{int((nested_table.delta_utility_20_vs_magnitude > 0).sum())}/4**。',
             '- 这是开发证据，不是外部确认。若区间跨 0，报告为条件性改善，不写成普遍优于幅度。',
             '- 幅度被保留为第一阶段锚点；SafeConf 和其他特征只能补充幅度以上的信息。',
             '', '## 复算', '',
             '```bash',
             'python tools/scripts/run_e223_magnitude_conditioned_router.py --input <E187_CARTESIAN_TASK_CERTIFICATES.csv> --output <new_output_dir> --bootstrap 10000',
             '```', '']
    (args.output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
