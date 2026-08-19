#!/usr/bin/env python3
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('/home/yyf/proj')
CODE_ROOT = PROJECT_ROOT / 'code' / '20260426_154505_perturb_transport_final_push'
FORMAL_TABLE = CODE_ROOT / 'outputs' / 'safeconf_formal_main_20260604' / 'formal_audit' / 'tables' / 'FORMAL_SCORED_RECORDS.csv'
DOC_ROOT = PROJECT_ROOT / '\u534f\u4f5c\u8bb0\u5f55' / '\u684c\u9762GPT' / '2026-06-07_\u7a81\u7834\u5b9e\u9a8c\u8bbe\u8ba1'
RUNS = DOC_ROOT / 'runs'
IDX = ['dataset_name', 'fold_id', 'split', 'context', 'perturbation']
PRED_KEY = IDX + ['predictor_name']
V0 = 'V0StrongBaseline'
CS = 'ContextSimBaseline'

ALLOWED_FEATURES = [
    'context_similarity_max', 'context_similarity_mean', 'perturbation_support_count',
    'perturbation_effect_stability', 'prediction_magnitude_deviation',
    'model_disagreement_rmse', 'ood_nearest_distance', 'historical_residual_risk',
]
SCORE_NAMES = [
    'protocol_v0_2_family_confidence', 'learned_risk_score', 'historical_residual_risk',
    'model_disagreement_risk', 'prediction_magnitude_risk', 'simple_combined_confidence',
    'random_score',
]
FORBIDDEN = ['true_error', 'true_effect', 'test_label', 'label', 'target', 'token', 'secret', 'password', 'key']
USECOLS = [
    'record_id', 'dataset_name', 'dataset_family', 'fold_id', 'split', 'context', 'perturbation',
    'predictor_name', 'score_name', 'score_type', 'score_value', 'risk_axis', 'true_error_rmse',
    'true_effect_l2_norm', 'run_dir', 'normalized_rmse',
]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def safe_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in name)


def spearman(x: pd.Series, y: pd.Series) -> float:
    x = num(x)
    y = num(y)
    m = x.notna() & y.notna()
    if int(m.sum()) < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float('nan')
    return float(x[m].corr(y[m], method='spearman'))


def load_scored() -> pd.DataFrame:
    return pd.read_csv(FORMAL_TABLE, usecols=USECOLS)


def load_features(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    frames = []
    for _, row in scored[['dataset_name', 'run_dir']].dropna().drop_duplicates().iterrows():
        path = Path(str(row['run_dir'])) / 'tables' / 'CONFIDENCE_FEATURES.csv'
        rec = {'dataset_name': row['dataset_name'], 'feature_path': str(path), 'exists': path.exists(), 'status': 'missing', 'n_rows': 0}
        if path.exists():
            try:
                use = set(PRED_KEY + ['record_id'] + ALLOWED_FEATURES)
                f = pd.read_csv(path, usecols=lambda c: c in use)
                rec['n_rows'] = len(f)
                missing = [c for c in ALLOWED_FEATURES if c not in f.columns]
                rec['missing_features'] = ';'.join(missing)
                rec['status'] = 'ok' if not missing else 'partial'
                frames.append(f.drop_duplicates(PRED_KEY))
            except Exception as exc:
                rec['status'] = f'failed:{type(exc).__name__}'
                rec['error'] = str(exc)[:200]
        rows.append(rec)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), pd.DataFrame(rows))


def make_pairs(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = PRED_KEY + ['dataset_family', 'record_id', 'true_error_rmse', 'true_effect_l2_norm', 'normalized_rmse', 'run_dir']
    base = scored[base_cols].drop_duplicates(PRED_KEY).copy()
    v0 = base[base['predictor_name'].eq(V0)].drop(columns=['predictor_name']).rename(columns={
        'record_id': 'v0_record_id', 'true_error_rmse': 'v0_error', 'normalized_rmse': 'v0_normalized_rmse', 'run_dir': 'v0_run_dir'
    })
    cs = base[base['predictor_name'].eq(CS)].drop(columns=['predictor_name']).rename(columns={
        'record_id': 'cs_record_id', 'true_error_rmse': 'cs_error', 'normalized_rmse': 'cs_normalized_rmse', 'run_dir': 'cs_run_dir'
    })
    pairs = v0.merge(cs[IDX + ['cs_record_id', 'cs_error', 'cs_normalized_rmse', 'cs_run_dir']], on=IDX, how='inner')
    pairs['winner'] = np.where(num(pairs['cs_error']) < num(pairs['v0_error']), CS, V0)
    pairs['winner_is_contextsim'] = (pairs['winner'] == CS).astype(int)
    pairs['oracle_error'] = np.minimum(num(pairs['v0_error']), num(pairs['cs_error']))
    pairs['error_gap_abs'] = (num(pairs['v0_error']) - num(pairs['cs_error'])).abs()

    features, audit = load_features(scored)
    if not features.empty:
        for pred, prefix in [(V0, 'v0'), (CS, 'cs')]:
            cols = IDX + [c for c in ALLOWED_FEATURES if c in features.columns]
            sub = features[features['predictor_name'].eq(pred)][cols].copy().drop_duplicates(IDX)
            sub = sub.rename(columns={c: f'{prefix}_{c}' for c in ALLOWED_FEATURES if c in sub.columns})
            pairs = pairs.merge(sub, on=IDX, how='left')
        for feat in ALLOWED_FEATURES:
            a = f'cs_{feat}'
            b = f'v0_{feat}'
            if a in pairs.columns and b in pairs.columns:
                pairs[f'delta_{feat}'] = num(pairs[a]) - num(pairs[b])
                pairs[f'abs_delta_{feat}'] = (num(pairs[a]) - num(pairs[b])).abs()

    score_base = scored[scored['score_name'].isin(SCORE_NAMES)][IDX + ['predictor_name', 'score_name', 'risk_axis']].copy()
    for score in SCORE_NAMES:
        safe = safe_name(score)
        for pred, prefix in [(V0, 'v0'), (CS, 'cs')]:
            sub = score_base[score_base['score_name'].eq(score) & score_base['predictor_name'].eq(pred)][IDX + ['risk_axis']].copy().drop_duplicates(IDX)
            sub = sub.rename(columns={'risk_axis': f'{prefix}_risk_{safe}'})
            pairs = pairs.merge(sub, on=IDX, how='left')
        a = f'cs_risk_{safe}'
        b = f'v0_risk_{safe}'
        if a in pairs.columns and b in pairs.columns:
            pairs[f'delta_risk_{safe}'] = num(pairs[a]) - num(pairs[b])
            pairs[f'abs_delta_risk_{safe}'] = (num(pairs[a]) - num(pairs[b])).abs()
    return pairs, audit


def choose_error(df: pd.DataFrame, choice: np.ndarray | pd.Series) -> pd.Series:
    choice = pd.Series(choice, index=df.index)
    return pd.Series(np.where(choice.eq(CS), num(df['cs_error']), num(df['v0_error'])), index=df.index)


def eval_choice(df: pd.DataFrame, strategy: str, choice: np.ndarray | pd.Series) -> dict:
    err = choose_error(df, choice)
    v0_mean = float(num(df['v0_error']).mean())
    cs_mean = float(num(df['cs_error']).mean())
    oracle = float(num(df['oracle_error']).mean())
    chosen = float(err.mean())
    best_fixed = min(v0_mean, cs_mean)
    winner_acc = float(pd.Series(choice, index=df.index).eq(df['winner']).mean()) if len(df) else float('nan')
    return {
        'strategy': strategy, 'n': int(len(df)), 'mean_rmse': chosen, 'fixed_v0_rmse': v0_mean,
        'fixed_contextsim_rmse': cs_mean, 'best_fixed_rmse': best_fixed, 'oracle_rmse': oracle,
        'improve_vs_best_fixed_pct': 100.0 * (best_fixed - chosen) / best_fixed if best_fixed > 0 else float('nan'),
        'improve_vs_v0_pct': 100.0 * (v0_mean - chosen) / v0_mean if v0_mean > 0 else float('nan'),
        'improve_vs_contextsim_pct': 100.0 * (cs_mean - chosen) / cs_mean if cs_mean > 0 else float('nan'),
        'winner_accuracy': winner_acc,
        'contextsim_choice_rate': float(pd.Series(choice, index=df.index).eq(CS).mean()) if len(df) else float('nan'),
    }


def add_eval_rows(rows: list[dict], pairs: pd.DataFrame, strategy: str, choice: pd.Series | np.ndarray) -> None:
    all_row = eval_choice(pairs, strategy, choice)
    all_row.update({'dataset_name': 'ALL'})
    rows.append(all_row)
    choice_s = pd.Series(choice, index=pairs.index)
    for dataset, idx in pairs.groupby('dataset_name').groups.items():
        row = eval_choice(pairs.loc[idx], strategy, choice_s.loc[idx])
        row.update({'dataset_name': dataset})
        rows.append(row)


def feature_association(pairs: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for split_name, sub in [('trainval', pairs[pairs['split'].isin(['train', 'val'])]), ('test', pairs[pairs['split'].eq('test')]), ('all', pairs)]:
        y = num(sub['winner_is_contextsim'])
        for col in feature_cols:
            if col not in sub.columns:
                continue
            x = num(sub[col])
            rows.append({'split_scope': split_name, 'feature': col, 'n': int((x.notna() & y.notna()).sum()), 'spearman_with_contextsim_winner': spearman(x, y)})
    return pd.DataFrame(rows).sort_values(['split_scope', 'spearman_with_contextsim_winner'], ascending=[True, False])


def classifier_choices(pairs: pd.DataFrame, feature_cols: list[str], score_name: str) -> tuple[pd.Series, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    choices = pd.Series(V0, index=pairs.index, dtype=object)
    status = []
    clean_cols = [c for c in feature_cols if c in pairs.columns]
    forbidden_hits = [c for c in clean_cols if any(p in c.lower() for p in FORBIDDEN)]
    for (dataset, fold), sub_idx in pairs.groupby(['dataset_name', 'fold_id']).groups.items():
        sub = pairs.loc[sub_idx]
        train = sub[sub['split'].isin(['train', 'val'])]
        test = sub[sub['split'].eq('test')]
        rec = {'score_name': score_name, 'dataset_name': dataset, 'fold_id': int(fold), 'n_train': len(train), 'n_test': len(test), 'n_features': len(clean_cols), 'status': 'ok'}
        if forbidden_hits:
            rec['status'] = 'blocked_forbidden_feature_name'
            rec['forbidden_hits'] = ';'.join(forbidden_hits)
            status.append(rec)
            continue
        if len(train) < 20 or len(test) == 0 or len(clean_cols) == 0 or train['winner_is_contextsim'].nunique() < 2:
            rec['status'] = 'skipped_too_few_or_one_class'
            status.append(rec)
            continue
        x_train = train[clean_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        x_test = test[clean_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        med = x_train.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_train = x_train.fillna(med)
        x_test = x_test.fillna(med)
        y = train['winner_is_contextsim'].astype(int)
        try:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
            model.fit(x_train, y)
            pred = model.predict(x_test)
            choices.loc[test.index] = np.where(pred == 1, CS, V0)
        except Exception as exc:
            rec['status'] = f'failed:{type(exc).__name__}'
            rec['error'] = str(exc)[:200]
        status.append(rec)
    return choices, pd.DataFrame(status)


def run() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    scored = load_scored()
    pairs, feature_audit = make_pairs(scored)
    feature_audit.to_csv(RUNS / 'runB5_feature_load_audit.csv', index=False)
    pairs.to_csv(RUNS / 'runB5_predictor_winner_table.csv', index=False)

    test = pairs[pairs['split'].eq('test')].copy()
    eval_rows = []
    add_eval_rows(eval_rows, test, 'fixed_v0', pd.Series(V0, index=test.index))
    add_eval_rows(eval_rows, test, 'fixed_contextsim', pd.Series(CS, index=test.index))
    add_eval_rows(eval_rows, test, 'oracle_best', test['winner'])

    for score in SCORE_NAMES:
        safe = safe_name(score)
        vcol = f'v0_risk_{safe}'
        ccol = f'cs_risk_{safe}'
        if vcol in test.columns and ccol in test.columns:
            choice = pd.Series(np.where(num(test[ccol]) < num(test[vcol]), CS, V0), index=test.index)
            add_eval_rows(eval_rows, test, f'score_guided_{score}', choice)

    feature_cols = [c for c in pairs.columns if c.startswith('delta_') or c.startswith('abs_delta_')]
    assoc = feature_association(pairs, feature_cols)
    assoc.to_csv(RUNS / 'runB5_feature_winner_association.csv', index=False)

    status_frames = []
    for name, cols in [
        ('logistic_all_pair_features', feature_cols),
        ('logistic_no_magnitude_pair_features', [c for c in feature_cols if 'magnitude' not in c.lower()]),
    ]:
        choices_all, status = classifier_choices(pairs, cols, name)
        status_frames.append(status)
        choices_test = choices_all.loc[test.index]
        add_eval_rows(eval_rows, test, name, choices_test)
    pd.concat(status_frames, ignore_index=True).to_csv(RUNS / 'runB5_selection_fit_status.csv', index=False)

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(RUNS / 'runB5_selection_eval_summary.csv', index=False)
    write_docs(eval_df, assoc)


def write_docs(eval_df: pd.DataFrame, assoc: pd.DataFrame) -> None:
    all_rows = eval_df[eval_df['dataset_name'].eq('ALL')].copy()
    all_rows = all_rows.sort_values('mean_rmse')
    best = all_rows.iloc[0] if len(all_rows) else None
    lines = []
    lines.append('# Run B5 predictor selection result')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    if best is not None:
        lines.append(f'- Best overall strategy: {best["strategy"]}, mean RMSE={best["mean_rmse"]:.4f}, improvement vs best fixed={best["improve_vs_best_fixed_pct"]:.2f}%.')
    for strat in ['fixed_v0', 'fixed_contextsim', 'score_guided_protocol_v0_2_family_confidence', 'score_guided_learned_risk_score', 'logistic_all_pair_features', 'logistic_no_magnitude_pair_features', 'oracle_best']:
        row = all_rows[all_rows['strategy'].eq(strat)]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(f'- {strat}: mean RMSE={r["mean_rmse"]:.4f}, improve_vs_best_fixed={r["improve_vs_best_fixed_pct"]:.2f}%, winner_accuracy={r["winner_accuracy"]:.3f}.')
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    learned = all_rows[all_rows['strategy'].eq('score_guided_learned_risk_score')]
    logistic = all_rows[all_rows['strategy'].eq('logistic_all_pair_features')]
    if not learned.empty and float(learned.iloc[0]['improve_vs_best_fixed_pct']) >= 2.0:
        lines.append('- Score-guided selection has enough signal to be considered as a practical add-on.')
    elif not logistic.empty and float(logistic.iloc[0]['improve_vs_best_fixed_pct']) >= 2.0:
        lines.append('- Learned selector has enough signal, but it should stay optional until stability is checked per dataset.')
    else:
        lines.append('- Predictor selection is weaker than risk filtering; keep it as exploratory unless per-dataset results show a clear niche.')
    lines.append('')
    lines.append('## Top feature associations on test')
    lines.append('')
    top = assoc[assoc['split_scope'].eq('test')].copy()
    top['abs_corr'] = top['spearman_with_contextsim_winner'].abs()
    for _, r in top.sort_values('abs_corr', ascending=False).head(8).iterrows():
        lines.append(f'- {r["feature"]}: rho={r["spearman_with_contextsim_winner"]:.3f}, n={int(r["n"])}')
    lines.append('')
    lines.append('## Outputs')
    lines.append('')
    lines.append('- runB5_predictor_winner_table.csv')
    lines.append('- runB5_feature_winner_association.csv')
    lines.append('- runB5_selection_eval_summary.csv')
    lines.append('- runB5_selection_fit_status.csv')
    (RUNS / 'runB5_result.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    cmd = f'''# Run B5 command log\n\nGenerated: {datetime.now().isoformat(timespec="seconds")}\n\n```bash\ncd /home/yyf/proj\n/home/yyf/.conda/envs/scgpt_env/bin/python code/20260426_154505_perturb_transport_final_push/scripts/run_breakthrough_b5_predictor_selection.py\n```\n\nInput: {FORMAL_TABLE}\n\nOutputs are in: {RUNS}\n'''
    (RUNS / 'runB5_command_log.md').write_text(cmd, encoding='utf-8')

    stage = []
    stage.append('# B5 predictor selection stage summary')
    stage.append('')
    stage.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    stage.append('')
    stage.append('## Decision')
    stage.append('')
    if best is not None:
        stage.append(f'- Best strategy: {best["strategy"]}.')
        stage.append(f'- Improvement vs best fixed predictor: {best["improve_vs_best_fixed_pct"]:.2f}%.')
    stage.append('- Use this as a practical add-on only if it beats the best fixed predictor clearly and consistently; otherwise keep SafeConf focused on risk filtering and calibration.')
    (DOC_ROOT / '08_B5_predictor_selection_summary.md').write_text('\n'.join(stage) + '\n', encoding='utf-8')


if __name__ == '__main__':
    run()
