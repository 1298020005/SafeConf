#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path('/home/yyf/proj')
CODE_ROOT = PROJECT_ROOT / 'code' / '20260426_154505_perturb_transport_final_push'
FORMAL_ROOT = CODE_ROOT / 'outputs' / 'safeconf_formal_main_20260604'
FORMAL_TABLE = FORMAL_ROOT / 'formal_audit' / 'tables' / 'FORMAL_SCORED_RECORDS.csv'
DOC_ROOT = PROJECT_ROOT / '\u534f\u4f5c\u8bb0\u5f55' / '\u684c\u9762GPT' / '2026-06-07_\u7a81\u7834\u5b9e\u9a8c\u8bbe\u8ba1'
RUNS = DOC_ROOT / 'runs'

PRED_KEY = ['dataset_name', 'fold_id', 'split', 'context', 'perturbation', 'predictor_name']

ALLOWED_FEATURES = [
    'context_similarity_max',
    'context_similarity_mean',
    'perturbation_support_count',
    'perturbation_effect_stability',
    'prediction_magnitude_deviation',
    'model_disagreement_rmse',
    'ood_nearest_distance',
    'historical_residual_risk',
]

NO_MAGNITUDE_FEATURES = [
    'context_similarity_max',
    'context_similarity_mean',
    'perturbation_support_count',
    'perturbation_effect_stability',
    'model_disagreement_rmse',
    'ood_nearest_distance',
    'historical_residual_risk',
]

BASELINE_SCORES = [
    'protocol_v0_2_family_confidence',
    'learned_risk_score',
    'historical_residual_risk',
    'model_disagreement_risk',
    'prediction_magnitude_risk',
    'simple_combined_confidence',
    'random_score',
]

MODEL_SPECS = [
    ('b3plus_elasticnet_all_features', 'elasticnet', ALLOWED_FEATURES),
    ('b3plus_elasticnet_no_magnitude', 'elasticnet', NO_MAGNITUDE_FEATURES),
    ('b3plus_isotonic_historical_residual', 'isotonic', ['historical_residual_risk']),
    ('b3plus_isotonic_disagreement', 'isotonic', ['model_disagreement_rmse']),
    ('b3plus_isotonic_magnitude_deviation', 'isotonic', ['prediction_magnitude_deviation']),
]

FORBIDDEN_PATTERNS = [
    'true_error',
    'true_effect',
    'test_label',
    'label',
    'target',
    'rmse_target',
]

USECOLS = [
    'record_id',
    'dataset_name',
    'dataset_family',
    'fold_id',
    'split',
    'context',
    'perturbation',
    'predictor_name',
    'score_name',
    'score_type',
    'score_value',
    'true_error_rmse',
    'true_effect_l2_norm',
    'run_dir',
    'risk_axis',
    'normalized_rmse',
]


def ensure_dirs() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def spearman(x: pd.Series, y: pd.Series) -> float:
    x = num(x)
    y = num(y)
    m = x.notna() & y.notna()
    if int(m.sum()) < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float('nan')
    return float(x[m].corr(y[m], method='spearman'))


def rank_residual(values: pd.Series, control: pd.Series) -> pd.Series:
    frame = pd.DataFrame({'v': values, 'c': control}).apply(pd.to_numeric, errors='coerce').dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(frame) < 3 or frame['c'].nunique() < 2:
        return out
    y = frame['v'].rank(method='average').to_numpy(dtype=float)
    z = frame['c'].rank(method='average').to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    out.loc[frame.index] = y - design @ beta
    return out


def partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    rx = rank_residual(x, control)
    ry = rank_residual(y, control)
    m = rx.notna() & ry.notna()
    if int(m.sum()) < 3 or rx[m].nunique() < 2 or ry[m].nunique() < 2:
        return float('nan')
    return float(rx[m].corr(ry[m], method='pearson'))


def risk_coverage_improve(group: pd.DataFrame, coverage: float = 0.8) -> float:
    g = group[['risk_axis', 'true_error_rmse']].apply(pd.to_numeric, errors='coerce').dropna()
    if len(g) < 5:
        return float('nan')
    base = float(g['true_error_rmse'].mean())
    if not math.isfinite(base) or base <= 0:
        return float('nan')
    keep_n = max(1, int(math.ceil(len(g) * coverage)))
    kept = g.sort_values('risk_axis', ascending=True).head(keep_n)
    kept_mean = float(kept['true_error_rmse'].mean())
    return 100.0 * (base - kept_mean) / base


def magnitude_controlled_rc(group: pd.DataFrame, coverage: float = 0.8, bins: int = 5) -> float:
    g = group[['risk_axis', 'true_error_rmse', 'true_effect_l2_norm']].apply(pd.to_numeric, errors='coerce').dropna()
    if len(g) < 20 or g['true_effect_l2_norm'].nunique() < 3:
        return float('nan')
    base = float(g['true_error_rmse'].mean())
    if not math.isfinite(base) or base <= 0:
        return float('nan')
    try:
        g = g.copy()
        q = min(bins, int(g['true_effect_l2_norm'].nunique()))
        g['mag_bin'] = pd.qcut(g['true_effect_l2_norm'].rank(method='first'), q=q, labels=False, duplicates='drop')
    except Exception:
        return float('nan')
    kept_parts = []
    for _, bg in g.groupby('mag_bin', dropna=False):
        if len(bg) == 0:
            continue
        keep_n = max(1, int(math.ceil(len(bg) * coverage)))
        kept_parts.append(bg.sort_values('risk_axis', ascending=True).head(keep_n))
    if not kept_parts:
        return float('nan')
    kept = pd.concat(kept_parts, ignore_index=True)
    kept_mean = float(kept['true_error_rmse'].mean())
    return 100.0 * (base - kept_mean) / base


def metric_row(group: pd.DataFrame) -> dict:
    return {
        'n': int(len(group)),
        'aligned_rho': spearman(group['risk_axis'], group['true_error_rmse']),
        'partial_rho_control_magnitude': partial_spearman(group['risk_axis'], group['true_error_rmse'], group['true_effect_l2_norm']),
        'normalized_rmse_rho': spearman(group['risk_axis'], group['normalized_rmse']) if 'normalized_rmse' in group.columns else float('nan'),
        'magnitude_only_rho': spearman(group['true_effect_l2_norm'], group['true_error_rmse']),
        'risk_coverage80_improve_pct': risk_coverage_improve(group, 0.8),
        'magnitude_controlled_rc80_improve_pct': magnitude_controlled_rc(group, 0.8, 5),
        'mean_rmse': float(num(group['true_error_rmse']).mean()),
        'median_rmse': float(num(group['true_error_rmse']).median()),
        'mean_effect_magnitude': float(num(group['true_effect_l2_norm']).mean()),
    }


def load_scored() -> pd.DataFrame:
    return pd.read_csv(FORMAL_TABLE, usecols=USECOLS)


def base_prediction_rows(scored: pd.DataFrame) -> pd.DataFrame:
    cols = PRED_KEY + [
        'record_id',
        'dataset_family',
        'true_error_rmse',
        'true_effect_l2_norm',
        'normalized_rmse',
        'run_dir',
    ]
    return scored[cols].drop_duplicates(PRED_KEY).copy()


def read_feature_tables(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_rows = []
    frames = []
    mapping = scored[['dataset_name', 'run_dir']].dropna().drop_duplicates()
    for _, row in mapping.iterrows():
        dataset = row['dataset_name']
        run_dir = Path(str(row['run_dir']))
        path = run_dir / 'tables' / 'CONFIDENCE_FEATURES.csv'
        record = {
            'dataset_name': dataset,
            'run_dir': str(run_dir),
            'feature_path': str(path),
            'exists': bool(path.exists()),
            'n_rows': 0,
            'n_missing_feature_columns': 0,
            'status': 'missing',
        }
        if path.exists():
            try:
                usecols = PRED_KEY + ['record_id'] + [c for c in ALLOWED_FEATURES]
                feat = pd.read_csv(path, usecols=lambda c: c in set(usecols))
                record['n_rows'] = int(len(feat))
                missing = [c for c in ALLOWED_FEATURES if c not in feat.columns]
                record['n_missing_feature_columns'] = int(len(missing))
                record['missing_feature_columns'] = ';'.join(missing)
                record['status'] = 'ok' if not missing else 'partial'
                frames.append(feat)
            except Exception as exc:
                record['status'] = f'failed:{type(exc).__name__}'
                record['error'] = str(exc)[:200]
        path_rows.append(record)
    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not features.empty:
        features = features.drop_duplicates(PRED_KEY)
    return features, pd.DataFrame(path_rows)


def make_feature_base(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = base_prediction_rows(scored)
    features, audit = read_feature_tables(scored)
    if features.empty:
        base['feature_merge_status'] = 'no_features_loaded'
        return base, audit
    feature_cols = [c for c in ALLOWED_FEATURES if c in features.columns]
    merged = base.merge(features[PRED_KEY + feature_cols], on=PRED_KEY, how='left')
    merge_rows = []
    for dataset, g in merged.groupby('dataset_name', dropna=False):
        row = {
            'dataset_name': dataset,
            'n_predictions': int(len(g)),
            'n_rows_with_all_features': int(g[feature_cols].notna().all(axis=1).sum()) if feature_cols else 0,
        }
        for col in feature_cols:
            row[f'missing_{col}'] = int(g[col].isna().sum())
        merge_rows.append(row)
    audit = pd.concat([audit, pd.DataFrame(merge_rows)], ignore_index=True, sort=False)
    return merged, audit


def median_fill(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_num = train_x.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    test_num = test_x.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    med = train_num.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_num.fillna(med), test_num.fillna(med)


def fit_elasticnet(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], random_state: int = 5201):
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x, test_x = median_fill(train[cols], test[cols])
    y = num(train['true_error_rmse'])
    valid = y.notna()
    train_x = train_x.loc[valid]
    y = y.loc[valid]
    if len(train_x) < 12 or float(y.std(ddof=0)) <= 1e-12:
        raise ValueError('too_few_rows_or_degenerate_target')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if len(train_x) >= 25:
            cv = max(2, min(5, int(len(train_x) // 5)))
            model = make_pipeline(
                StandardScaler(),
                ElasticNetCV(
                    l1_ratio=[0.2, 0.5, 0.8, 1.0],
                    alphas=np.logspace(-4, 1, 30),
                    cv=cv,
                    max_iter=20000,
                    random_state=random_state,
                ),
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=random_state),
            )
        model.fit(train_x, y)
    pred = model.predict(test_x)
    coef_rows = []
    final = model.steps[-1][1]
    if hasattr(final, 'coef_'):
        for col, coef in zip(cols, final.coef_):
            coef_rows.append({'feature': col, 'coefficient': float(coef)})
    return pred, coef_rows


def fit_isotonic(train: pd.DataFrame, test: pd.DataFrame, col: str):
    from sklearn.isotonic import IsotonicRegression

    x = num(train[col])
    y = num(train['true_error_rmse'])
    valid = x.notna() & y.notna()
    x = x.loc[valid]
    y = y.loc[valid]
    if len(x) < 12 or x.nunique() < 3 or float(y.std(ddof=0)) <= 1e-12:
        raise ValueError('too_few_rows_or_degenerate_target')
    rho = spearman(x, y)
    increasing = bool(rho >= 0) if math.isfinite(rho) else True
    order = np.argsort(x.to_numpy())
    model = IsotonicRegression(out_of_bounds='clip', increasing=increasing)
    model.fit(x.to_numpy()[order], y.to_numpy()[order])
    test_x = num(test[col]).fillna(float(x.median()))
    pred = model.predict(test_x.to_numpy())
    return pred, [{'feature': col, 'coefficient': 1.0 if increasing else -1.0, 'train_feature_error_rho': rho}]


def run_b3plus(scored: pd.DataFrame) -> pd.DataFrame:
    base, feature_audit = make_feature_base(scored)
    feature_audit.to_csv(RUNS / 'runB3plus_feature_merge_audit.csv', index=False)

    forbidden_hits = []
    for _, _, cols in MODEL_SPECS:
        forbidden_hits.extend([c for c in cols if any(p in c.lower() for p in FORBIDDEN_PATTERNS)])

    score_rows = []
    status_rows = []
    coef_rows = []
    for score_name, model_type, cols in MODEL_SPECS:
        cols = [c for c in cols if c in base.columns]
        for (dataset, fold, predictor), sub in base.groupby(['dataset_name', 'fold_id', 'predictor_name'], dropna=False):
            train = sub[sub['split'].isin(['train', 'val'])].copy()
            test = sub[sub['split'].eq('test')].copy()
            status = {
                'score_name': score_name,
                'model_type': model_type,
                'dataset_name': dataset,
                'fold_id': int(fold),
                'predictor_name': predictor,
                'n_train_pool': int(len(train)),
                'n_test': int(len(test)),
                'n_features': int(len(cols)),
                'features': ';'.join(cols),
            }
            if forbidden_hits:
                status['status'] = 'blocked_forbidden_feature_name'
                status_rows.append(status)
                continue
            if len(cols) == 0 or len(test) == 0 or len(train) < 12:
                status['status'] = 'skipped_too_few_rows_or_features'
                status_rows.append(status)
                continue
            try:
                if model_type == 'elasticnet':
                    pred, coefs = fit_elasticnet(train, test, cols)
                elif model_type == 'isotonic':
                    pred, coefs = fit_isotonic(train, test, cols[0])
                else:
                    raise ValueError(f'unknown_model_type:{model_type}')
            except Exception as exc:
                status['status'] = f'failed:{type(exc).__name__}'
                status['error'] = str(exc)[:200]
                status_rows.append(status)
                continue
            for (_, row), value in zip(test.iterrows(), pred):
                score_rows.append({
                    'record_id': row['record_id'],
                    'dataset_name': row['dataset_name'],
                    'dataset_family': row.get('dataset_family', ''),
                    'fold_id': int(row['fold_id']),
                    'split': row['split'],
                    'context': row['context'],
                    'perturbation': row['perturbation'],
                    'predictor_name': row['predictor_name'],
                    'score_name': score_name,
                    'score_type': 'risk',
                    'score_value': float(value),
                    'risk_axis': float(value),
                    'true_error_rmse': float(row['true_error_rmse']),
                    'true_effect_l2_norm': float(row['true_effect_l2_norm']),
                    'normalized_rmse': float(row['normalized_rmse']) if pd.notna(row['normalized_rmse']) else float('nan'),
                })
            for coef in coefs:
                coef_row = dict(coef)
                coef_row.update({
                    'score_name': score_name,
                    'dataset_name': dataset,
                    'fold_id': int(fold),
                    'predictor_name': predictor,
                })
                coef_rows.append(coef_row)
            status['status'] = 'ok'
            status_rows.append(status)

    b3_scores = pd.DataFrame(score_rows)
    pd.DataFrame(status_rows).to_csv(RUNS / 'runB3plus_fit_status.csv', index=False)
    pd.DataFrame(coef_rows).to_csv(RUNS / 'runB3plus_model_coefficients.csv', index=False)
    if not b3_scores.empty:
        b3_scores.to_csv(RUNS / 'runB3plus_scored_records.csv', index=False)
    else:
        pd.DataFrame(columns=['dataset_name', 'score_name']).to_csv(RUNS / 'runB3plus_scored_records.csv', index=False)

    baseline = scored[scored['split'].eq('test') & scored['score_name'].isin(BASELINE_SCORES)].copy()
    combined = pd.concat([baseline, b3_scores], ignore_index=True, sort=False)
    eval_rows = []
    for (dataset, score_name), g in combined.groupby(['dataset_name', 'score_name'], dropna=False):
        row = metric_row(g)
        row.update({'dataset_name': dataset, 'score_name': score_name})
        eval_rows.append(row)
    eval_df = pd.DataFrame(eval_rows).sort_values(['score_name', 'dataset_name'])
    eval_df.to_csv(RUNS / 'runB3plus_eval_summary.csv', index=False)
    bootstrap_ci(combined, RUNS / 'runB3plus_bootstrap_ci.csv', n_boot=200, seed=5301)
    write_b3plus_docs(eval_df, status_rows, forbidden_hits)
    return combined


def bootstrap_ci(scored_like: pd.DataFrame, out_path: Path, n_boot: int = 200, seed: int = 5201) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for (dataset, score_name), g in scored_like.groupby(['dataset_name', 'score_name'], dropna=False):
        gg = g[['risk_axis', 'true_error_rmse', 'true_effect_l2_norm']].apply(pd.to_numeric, errors='coerce').dropna()
        n = len(gg)
        if n < 20:
            continue
        aligned = []
        rc80 = []
        controlled = []
        values = gg.reset_index(drop=True)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = values.iloc[idx]
            aligned.append(spearman(sample['risk_axis'], sample['true_error_rmse']))
            rc80.append(risk_coverage_improve(sample, 0.8))
            controlled.append(magnitude_controlled_rc(sample, 0.8, 5))
        rows.append({
            'dataset_name': dataset,
            'score_name': score_name,
            'n': int(n),
            'n_bootstrap': int(n_boot),
            'aligned_rho_ci_low': float(np.nanpercentile(aligned, 2.5)),
            'aligned_rho_ci_high': float(np.nanpercentile(aligned, 97.5)),
            'rc80_ci_low': float(np.nanpercentile(rc80, 2.5)),
            'rc80_ci_high': float(np.nanpercentile(rc80, 97.5)),
            'controlled_rc80_ci_low': float(np.nanpercentile(controlled, 2.5)),
            'controlled_rc80_ci_high': float(np.nanpercentile(controlled, 97.5)),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_b3plus_docs(eval_df: pd.DataFrame, status_rows: list[dict], forbidden_hits: list[str]) -> None:
    def count_positive(score: str, col: str) -> tuple[int, int]:
        sub = eval_df[eval_df['score_name'].eq(score)]
        return int((num(sub[col]) > 0).sum()), int(len(sub))

    lines = []
    lines.append('# Run B3-plus result')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append('')
    lines.append('## Safety')
    lines.append('')
    lines.append('- v0.2 formula was not changed.')
    lines.append('- Models were fit within dataset/fold/predictor using train+val only, then evaluated on test.')
    lines.append('- This is risk calibration for existing predictions, not a new perturbation predictor.')
    lines.append(f'- Forbidden feature name hits: {forbidden_hits if forbidden_hits else "none"}.')
    lines.append('')
    lines.append('## Main checks')
    lines.append('')
    for score in ['b3plus_elasticnet_all_features', 'b3plus_elasticnet_no_magnitude', 'b3plus_isotonic_historical_residual', 'protocol_v0_2_family_confidence', 'learned_risk_score']:
        if score not in set(eval_df['score_name']):
            continue
        a = count_positive(score, 'aligned_rho')
        p = count_positive(score, 'partial_rho_control_magnitude')
        r = count_positive(score, 'risk_coverage80_improve_pct')
        c = count_positive(score, 'magnitude_controlled_rc80_improve_pct')
        lines.append(f'- {score}: aligned {a[0]}/{a[1]}, partial {p[0]}/{p[1]}, RC80 {r[0]}/{r[1]}, controlled RC80 {c[0]}/{c[1]}.')
    lines.append('')
    mc = eval_df[eval_df['dataset_name'].eq('McFarlandTsherniak2020')]
    lines.append('## McFarland')
    lines.append('')
    for score in ['protocol_v0_2_family_confidence', 'learned_risk_score', 'b3plus_elasticnet_all_features', 'b3plus_elasticnet_no_magnitude', 'b3plus_isotonic_historical_residual']:
        sub = mc[mc['score_name'].eq(score)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        lines.append(f'- {score}: rho={row["aligned_rho"]:.3f}, partial={row["partial_rho_control_magnitude"]:.3f}, RC80={row["risk_coverage80_improve_pct"]:.2f}%, controlled_RC80={row["magnitude_controlled_rc80_improve_pct"]:.2f}%.')
    lines.append('')
    ok_count = sum(1 for r in status_rows if r.get('status') == 'ok')
    lines.append('## Outputs')
    lines.append('')
    lines.append(f'- Fit groups ok: {ok_count}/{len(status_rows)}')
    lines.append('- runB3plus_eval_summary.csv')
    lines.append('- runB3plus_bootstrap_ci.csv')
    lines.append('- runB3plus_fit_status.csv')
    lines.append('- runB3plus_feature_merge_audit.csv')
    lines.append('- runB3plus_model_coefficients.csv')
    lines.append('- runB3plus_scored_records.csv')
    (RUNS / 'runB3plus_result.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def rc_table(scored_like: pd.DataFrame, coverage: float, controlled: bool = False) -> pd.DataFrame:
    rows = []
    for (dataset, score_name), g in scored_like.groupby(['dataset_name', 'score_name'], dropna=False):
        g2 = g[['risk_axis', 'true_error_rmse', 'true_effect_l2_norm']].apply(pd.to_numeric, errors='coerce').dropna()
        if len(g2) < 5:
            continue
        base = float(g2['true_error_rmse'].mean())
        improve = magnitude_controlled_rc(g2, coverage, 5) if controlled else risk_coverage_improve(g2, coverage)
        rows.append({
            'dataset_name': dataset,
            'score_name': score_name,
            'coverage': float(coverage),
            'controlled_by_magnitude_bin': bool(controlled),
            'n': int(len(g2)),
            'baseline_mean_rmse': base,
            'improve_pct': improve,
        })
    return pd.DataFrame(rows)


def aurc_table(scored_like: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    rows = []
    for (dataset, score_name), g in scored_like.groupby(['dataset_name', 'score_name'], dropna=False):
        g2 = g[['risk_axis', 'true_error_rmse']].apply(pd.to_numeric, errors='coerce').dropna()
        if len(g2) < 5:
            continue
        base = float(g2['true_error_rmse'].mean())
        risks = []
        for cov in coverages:
            keep_n = max(1, int(math.ceil(len(g2) * cov)))
            kept = g2.sort_values('risk_axis', ascending=True).head(keep_n)
            risks.append(float(kept['true_error_rmse'].mean()))
        aurc = float(np.trapz(risks, coverages) / (max(coverages) - min(coverages))) if len(coverages) > 1 else risks[0]
        rows.append({
            'dataset_name': dataset,
            'score_name': score_name,
            'n': int(len(g2)),
            'baseline_mean_rmse': base,
            'aurc_rmse': aurc,
            'relative_aurc_improve_pct': 100.0 * (base - aurc) / base if base > 0 else float('nan'),
        })
    return pd.DataFrame(rows)


def run_b4(scored_like: pd.DataFrame) -> None:
    coverages = [0.5, 0.6, 0.7, 0.8, 0.9]
    standard_frames = [rc_table(scored_like, cov, controlled=False) for cov in coverages]
    controlled_frames = [rc_table(scored_like, cov, controlled=True) for cov in coverages]
    standard = pd.concat(standard_frames, ignore_index=True) if standard_frames else pd.DataFrame()
    controlled = pd.concat(controlled_frames, ignore_index=True) if controlled_frames else pd.DataFrame()
    aurc = aurc_table(scored_like, coverages)
    standard.to_csv(RUNS / 'runB4_standard_risk_coverage.csv', index=False)
    controlled.to_csv(RUNS / 'runB4_magnitude_controlled_risk_coverage.csv', index=False)
    aurc.to_csv(RUNS / 'runB4_aurc_summary.csv', index=False)
    write_b4_docs(standard, controlled, aurc)


def write_b4_docs(standard: pd.DataFrame, controlled: pd.DataFrame, aurc: pd.DataFrame) -> None:
    def pos_count(df: pd.DataFrame, score: str, coverage: float = 0.8) -> tuple[int, int]:
        sub = df[df['score_name'].eq(score) & np.isclose(num(df['coverage']), coverage)] if 'coverage' in df.columns else pd.DataFrame()
        return int((num(sub['improve_pct']) > 0).sum()), int(len(sub))

    lines = []
    lines.append('# Run B4 result')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append('')
    lines.append('## What this checks')
    lines.append('')
    lines.append('- Standard RC asks whether removing the riskiest 20% predictions lowers RMSE.')
    lines.append('- Magnitude-controlled RC repeats the filtering inside effect-magnitude bins, so a score cannot win only by throwing away large-effect records.')
    lines.append('')
    lines.append('## RC@80 summary')
    lines.append('')
    for score in ['protocol_v0_2_family_confidence', 'learned_risk_score', 'b3plus_elasticnet_all_features', 'b3plus_elasticnet_no_magnitude', 'b3plus_isotonic_historical_residual', 'prediction_magnitude_risk']:
        if score not in set(standard['score_name']):
            continue
        s = pos_count(standard, score, 0.8)
        c = pos_count(controlled, score, 0.8)
        lines.append(f'- {score}: standard {s[0]}/{s[1]}, magnitude-controlled {c[0]}/{c[1]}.')
    lines.append('')
    lines.append('## McFarland RC@80')
    lines.append('')
    mc_std = standard[standard['dataset_name'].eq('McFarlandTsherniak2020') & np.isclose(num(standard['coverage']), 0.8)]
    mc_ctl = controlled[controlled['dataset_name'].eq('McFarlandTsherniak2020') & np.isclose(num(controlled['coverage']), 0.8)]
    for score in ['protocol_v0_2_family_confidence', 'learned_risk_score', 'b3plus_elasticnet_all_features', 'b3plus_elasticnet_no_magnitude', 'b3plus_isotonic_historical_residual', 'prediction_magnitude_risk']:
        srow = mc_std[mc_std['score_name'].eq(score)]
        crow = mc_ctl[mc_ctl['score_name'].eq(score)]
        if srow.empty or crow.empty:
            continue
        lines.append(f'- {score}: standard={float(srow.iloc[0]["improve_pct"]):.2f}%, controlled={float(crow.iloc[0]["improve_pct"]):.2f}%.')
    lines.append('')
    lines.append('## Outputs')
    lines.append('')
    lines.append('- runB4_standard_risk_coverage.csv')
    lines.append('- runB4_magnitude_controlled_risk_coverage.csv')
    lines.append('- runB4_aurc_summary.csv')
    (RUNS / 'runB4_result.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_stage_summary() -> None:
    b3 = pd.read_csv(RUNS / 'runB3plus_eval_summary.csv') if (RUNS / 'runB3plus_eval_summary.csv').exists() else pd.DataFrame()
    b4 = pd.read_csv(RUNS / 'runB4_magnitude_controlled_risk_coverage.csv') if (RUNS / 'runB4_magnitude_controlled_risk_coverage.csv').exists() else pd.DataFrame()
    lines = []
    lines.append('# B3-plus / B4 unattended stage summary')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append('')
    lines.append('## Version safety')
    lines.append('')
    lines.append('- Branch: exp/breakthrough-20260607')
    lines.append('- Main branch was not merged or edited.')
    lines.append('- v0.2 formula was not changed.')
    lines.append('- No authorization/key/token paths are touched by this script.')
    lines.append('')
    lines.append('## Tomorrow first read')
    lines.append('')
    if not b3.empty:
        for score in ['b3plus_elasticnet_all_features', 'b3plus_elasticnet_no_magnitude', 'learned_risk_score', 'protocol_v0_2_family_confidence']:
            sub = b3[b3['score_name'].eq(score)]
            if sub.empty:
                continue
            aligned = int((num(sub['aligned_rho']) > 0).sum())
            partial = int((num(sub['partial_rho_control_magnitude']) > 0).sum())
            controlled = int((num(sub['magnitude_controlled_rc80_improve_pct']) > 0).sum())
            lines.append(f'- {score}: aligned positive {aligned}/{len(sub)}, partial positive {partial}/{len(sub)}, controlled RC80 positive {controlled}/{len(sub)}.')
    if not b4.empty:
        ctl80 = b4[np.isclose(num(b4['coverage']), 0.8)]
        lines.append('')
        lines.append('## Magnitude-controlled RC@80')
        lines.append('')
        for score in ['protocol_v0_2_family_confidence', 'learned_risk_score', 'b3plus_elasticnet_all_features', 'b3plus_elasticnet_no_magnitude']:
            sub = ctl80[ctl80['score_name'].eq(score)]
            if sub.empty:
                continue
            lines.append(f'- {score}: positive {int((num(sub["improve_pct"]) > 0).sum())}/{len(sub)}, mean improve {float(num(sub["improve_pct"]).mean()):.2f}%.')
    lines.append('')
    lines.append('## Next decision')
    lines.append('')
    lines.append('- If ElasticNet is stable, treat learned risk as an interpretable optional calibration layer.')
    lines.append('- If only the old learned_risk_score is strong, keep it as supplement until leakage/fold-local audit is stronger.')
    lines.append('- Tahoe and larger predictors remain postponed until these tables are reviewed.')
    (DOC_ROOT / '07_B3plus_B4_stage_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_command_log() -> None:
    text = f'''# Run B3-plus / B4 command log

Generated: {datetime.now().isoformat(timespec="seconds")}

```bash
cd /home/yyf/proj
/home/yyf/.conda/envs/scgpt_env/bin/python code/20260426_154505_perturb_transport_final_push/scripts/run_breakthrough_b3plus_b4.py
```

Input:

- {FORMAL_TABLE}
- per-dataset `tables/CONFIDENCE_FEATURES.csv` referenced by `run_dir`

Outputs:

- {RUNS / 'runB3plus_eval_summary.csv'}
- {RUNS / 'runB4_magnitude_controlled_risk_coverage.csv'}
- {DOC_ROOT / '07_B3plus_B4_stage_summary.md'}
'''
    (RUNS / 'runB3plus_B4_command_log.md').write_text(text, encoding='utf-8')


def main() -> None:
    ensure_dirs()
    scored = load_scored()
    combined = run_b3plus(scored)
    run_b4(combined)
    write_stage_summary()
    write_command_log()


if __name__ == '__main__':
    main()
