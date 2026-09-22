#!/usr/bin/env python3
"""Nested leave-study-out test of a predeclared scenario gate."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e223_magnitude_conditioned_router as core

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path('/home/yyf/proj/docs/实验结果/E187_advisor_difficulty_certificate_20260726/tables/E187_CARTESIAN_TASK_CERTIFICATES.csv')
DEFAULT_OUTPUT = ROOT / 'docs/实验结果/E224_scenario_gate_router_20260922'
SEED = 20260922


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rules() -> list[dict]:
    return [
        {'name': 'magnitude', 'kind': 'linear', 'params': {}},
        {'name': 'fixed_80m_20s', 'kind': 'linear', 'params': {'s': .2}},
        {'name': 'global_max_m_s', 'kind': 'one_sided', 'params': {'s': 1.0}},
        {'name': 'gate_hard_context_only', 'kind': 'scenario_one_sided', 'params': {
            'context_unseen_row': 0., 'perturbation_unseen_column': 1.,
            'random_missing_pair': 1., 'context_and_perturbation_unseen': 1.}},
        {'name': 'gate_soft_context_only', 'kind': 'scenario_one_sided', 'params': {
            'context_unseen_row': 0., 'perturbation_unseen_column': .75,
            'random_missing_pair': .75, 'context_and_perturbation_unseen': .5}},
        {'name': 'gate_half_context_only', 'kind': 'scenario_one_sided', 'params': {
            'context_unseen_row': 0., 'perturbation_unseen_column': .5,
            'random_missing_pair': .5, 'context_and_perturbation_unseen': .25}},
        {'name': 'scenario_b', 'kind': 'scenario_one_sided', 'params': {
            'context_unseen_row': .5, 'perturbation_unseen_column': .5,
            'random_missing_pair': .5, 'context_and_perturbation_unseen': .25}},
    ]


def eval_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    return core.eval_blocks(frame, rule)


def choose_inner(train: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for study in sorted(train.dataset.unique()):
        valid = train.loc[train.dataset.eq(study)]
        for rule in rules():
            vals = eval_rule(valid, rule)
            rows.append({'inner_heldout_study': study, 'rule': rule['name'],
                         'utility_20': float(vals.utility_20.mean()),
                         'spearman': float(vals.spearman.mean()),
                         'capture_20': float(vals.capture_20.mean())})
    table = pd.DataFrame(rows)
    ranking = []
    for rule in rules():
        sub = table.loc[table.rule.eq(rule['name'])]
        ranking.append((float(sub.utility_20.mean()), float(sub.spearman.mean()),
                        -len(rule['params']), rule['name'], rule))
    return max(ranking, key=lambda x: (x[0], x[1], x[2], x[3]))[4], table


def outer_eval(frame: pd.DataFrame):
    fold, inner, selections, settings = [], [], [], []
    for heldout in sorted(frame.dataset.unique()):
        train = frame.loc[frame.dataset.ne(heldout)]
        test = frame.loc[frame.dataset.eq(heldout)].copy()
        chosen, inner_table = choose_inner(train)
        inner_table['outer_heldout_study'] = heldout
        inner.append(inner_table)
        selections.append({'outer_heldout_study': heldout, 'selected_rule': chosen['name'],
                           'n_train_tasks': len(train), 'n_test_tasks': len(test)})
        for rule in rules():
            vals = eval_rule(test, rule)
            vals['outer_heldout_study'] = heldout
            fold.append(vals)
            for setting, block in test.groupby('setting', sort=True):
                v = core.metrics(core.score(block, rule), block[core.ERROR].to_numpy(float))
                settings.append({'outer_heldout_study': heldout, 'setting': setting,
                                 'rule': rule['name'], 'n_tasks': len(block), **v})
    return (pd.concat(fold, ignore_index=True), pd.concat(inner, ignore_index=True),
            pd.DataFrame(selections), pd.DataFrame(settings))


def bootstrap(fold: pd.DataFrame, n: int) -> pd.DataFrame:
    study = fold.groupby(['outer_heldout_study', 'rule'])[list(core.METRICS)].mean().reset_index()
    ids = sorted(study.outer_heldout_study.unique())
    base = study.loc[study.rule.eq('magnitude')].set_index('outer_heldout_study').loc[ids]
    rng = np.random.default_rng(SEED)
    rows = []
    for rule in sorted(study.rule.unique()):
        cur = study.loc[study.rule.eq(rule)].set_index('outer_heldout_study').loc[ids]
        delta = cur[list(core.METRICS)].to_numpy() - base[list(core.METRICS)].to_numpy()
        draws = delta[rng.integers(0, len(ids), size=(n, len(ids)))].mean(axis=1)
        for j, metric in enumerate(core.METRICS):
            rows.append({'rule': rule, 'metric': metric, 'delta_vs_magnitude': float(delta[:, j].mean()),
                         'ci95_lower': float(np.quantile(draws[:, j], .025)),
                         'ci95_upper': float(np.quantile(draws[:, j], .975)),
                         'positive_studies': int((delta[:, j] > 0).sum()), 'n_studies': len(ids)})
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
        raise RuntimeError(f'refusing to overwrite non-empty {args.output}')
    frame = core.load(args.input.resolve())
    fold, inner, selections, settings = outer_eval(frame)
    intervals = bootstrap(fold, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    fold.to_csv(args.output / 'E224_OUTER_BLOCK_RESULTS.csv', index=False)
    inner.to_csv(args.output / 'E224_INNER_RULE_SELECTION.csv', index=False)
    selections.to_csv(args.output / 'E224_OUTER_SELECTIONS.csv', index=False)
    settings.to_csv(args.output / 'E224_SETTING_RESULTS.csv', index=False)
    intervals.to_csv(args.output / 'E224_STUDY_BOOTSTRAP_INTERVALS.csv', index=False)
    selected = []
    base = fold.loc[fold.rule.eq('magnitude')].groupby('outer_heldout_study').utility_20.mean()
    for _, row in selections.iterrows():
        val = fold.loc[(fold.outer_heldout_study.eq(row.outer_heldout_study)) & (fold.rule.eq(row.selected_rule)), 'utility_20'].mean()
        selected.append({'outer_heldout_study': row.outer_heldout_study, 'selected_rule': row.selected_rule,
                         'delta_utility_20_vs_magnitude': float(val - base[row.outer_heldout_study])})
    selected = pd.DataFrame(selected)
    selected.to_csv(args.output / 'E224_NESTED_SELECTED_DELTAS.csv', index=False)
    nested_mean = float(selected.delta_utility_20_vs_magnitude.mean())
    status = {'experiment': 'E224_scenario_gate_router', 'status': 'COMPLETE',
              'evidence_class': 'retrospective_nested_leave_study_out_rule_dictionary',
              'input_sha256': sha256(args.input.resolve()),
              'script_sha256': sha256(Path(__file__)),
              'helper_sha256': sha256(Path(__file__).with_name('run_e223_magnitude_conditioned_router.py')),
              'n_tasks': len(frame), 'n_studies': int(frame.dataset.nunique()), 'n_rules': len(rules()),
              'outer_truth_used_for_selection': False, 'e208_truth_used': False,
              'source_data_modified': False, 'all_studies_retained': True,
              'nested_selected_rule_counts': selections.selected_rule.value_counts().to_dict(),
              'nested_mean_utility_delta': nested_mean,
              'nested_positive_outer_studies': int((selected.delta_utility_20_vs_magnitude > 0).sum()),
              'bootstrap': args.bootstrap, 'seed': SEED,
              'created_at': datetime.now().astimezone().isoformat()}
    (args.output / 'RUN_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# E224：场景门控的幅度—SafeConf 路由', '',
             'E223 之后固定的假设：只缺细胞背景时不追加 SafeConf，其余三种场景允许补充幅度以上的风险。', '',
             f'- 嵌套选择后的平均 top-20% 效用差值：**{nested_mean:+.5f}**。',
             f'- 外层正向研究：**{status["nested_positive_outer_studies"]}/4**。',
             '- 所有 8196 个任务均保留；外层研究答案没有参与规则选择。',
             '- 仍属于历史开发证据；需用 E208 或新研究按冻结规则做外部确认。', '',
             '```bash',
             'python tools/scripts/run_e224_scenario_gate_router.py --input <E187_CARTESIAN_TASK_CERTIFICATES.csv> --output <new_output_dir> --bootstrap 10000',
             '```', '']
    (args.output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
