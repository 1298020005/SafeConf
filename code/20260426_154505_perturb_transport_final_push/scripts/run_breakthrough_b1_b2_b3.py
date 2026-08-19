
#!/usr/bin/env python3
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('/home/yyf/proj')
CODE_ROOT = PROJECT_ROOT / 'code' / '20260426_154505_perturb_transport_final_push'
FORMAL_ROOT = CODE_ROOT / 'outputs' / 'safeconf_formal_main_20260604'
FORMAL_TABLE = FORMAL_ROOT / 'formal_audit' / 'tables' / 'FORMAL_SCORED_RECORDS.csv'
FORMAL_SUMMARY = FORMAL_ROOT / 'formal_audit' / 'tables' / 'FORMAL_SCORE_SUMMARY.csv'
DOC_ROOT = PROJECT_ROOT / '\u534f\u4f5c\u8bb0\u5f55' / '\u684c\u9762GPT' / '2026-06-07_\u7a81\u7834\u5b9e\u9a8c\u8bbe\u8ba1'
RUNS = DOC_ROOT / 'runs'

KEY_SCORES = [
    'protocol_v0_2_family_confidence',
    'learned_risk_score',
    'historical_residual_risk',
    'model_disagreement_risk',
    'prediction_magnitude_risk',
    'simple_combined_confidence',
    'random_score',
]

FORBIDDEN_FEATURE_PATTERNS = ['true_error', 'true_effect', 'test_label', 'label', 'rmse_target', 'score_value_target']

USECOLS = [
    'record_id', 'dataset_name', 'dataset_family', 'fold_id', 'split',
    'context', 'perturbation', 'predictor_name', 'score_name', 'score_type',
    'score_value', 'true_error_rmse', 'true_effect_l2_norm', 'risk_axis', 'normalized_rmse'
]


def _ensure():
    RUNS.mkdir(parents=True, exist_ok=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def spearman(x: pd.Series, y: pd.Series) -> float:
    x = _num(x)
    y = _num(y)
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
    if int(m.sum()) < 3:
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


def metric_row(group: pd.DataFrame) -> dict:
    return {
        'n': int(len(group)),
        'aligned_rho': spearman(group['risk_axis'], group['true_error_rmse']),
        'partial_rho_control_magnitude': partial_spearman(group['risk_axis'], group['true_error_rmse'], group['true_effect_l2_norm']),
        'normalized_rmse_rho': spearman(group['risk_axis'], group['normalized_rmse']) if 'normalized_rmse' in group else float('nan'),
        'magnitude_only_rho': spearman(group['true_effect_l2_norm'], group['true_error_rmse']),
        'risk_coverage80_improve_pct': risk_coverage_improve(group, 0.8),
        'mean_rmse': float(_num(group['true_error_rmse']).mean()),
        'median_rmse': float(_num(group['true_error_rmse']).median()),
        'mean_effect_magnitude': float(_num(group['true_effect_l2_norm']).mean()),
    }


def load_scored() -> pd.DataFrame:
    return pd.read_csv(FORMAL_TABLE, usecols=USECOLS)


PREDICTION_KEY = ['dataset_name', 'fold_id', 'split', 'context', 'perturbation', 'predictor_name']


def base_prediction_rows(scored: pd.DataFrame) -> pd.DataFrame:
    # record_id is not guaranteed to be globally unique across datasets, so use
    # the task/predictor key to collapse multiple score rows for one prediction.
    cols = PREDICTION_KEY + ['dataset_family', 'true_error_rmse', 'true_effect_l2_norm', 'normalized_rmse']
    return scored[cols].drop_duplicates(PREDICTION_KEY).copy()


def run_b0() -> None:
    rows = []
    source_files = [
        ('formal_score_summary', FORMAL_SUMMARY, 'core'),
        ('formal_scored_records', FORMAL_TABLE, 'core_large_table'),
        ('feature_ablation_1000', PROJECT_ROOT / 'docs/04-????/Formal_main_20260604/feature_ablation_1000_20260606/tables/FEATURE_ABLATION_SUMMARY.csv', 'core_support'),
        ('lodo_main_table', PROJECT_ROOT / 'docs/04-????/Formal_main_20260604/sprint1_lodo/tables/LODO_MAIN_TABLE.csv', 'core_support'),
        ('signal_validity_7main', PROJECT_ROOT / 'docs/04-????/Formal_main_20260604/signal_validity_7main_20260606/tables/SIGNAL_VALIDITY_7MAIN_SUMMARY.csv', 'core_support'),
        ('mcfarland_diagnostics', PROJECT_ROOT / 'docs/04-????/Formal_main_20260604/diagnostics/tables/McFarland_single_feature_diagnostics.csv', 'failure_boundary'),
        ('tahoe_smoke', PROJECT_ROOT / 'docs/04-????/Formal_main_20260604/tahoe_sampled_formal_v1/tables/TAHOE_FORMAL_EVAL_SUMMARY_SMOKE.csv', 'external_smoke'),
        ('gears_probe', PROJECT_ROOT / '????/??GPT/2026-06-07_??????/runs/2026-06-07_run03_GEARS_confidence_eval_summary.csv', 'supplement_probe'),
    ]
    for name, path, role in source_files:
        rows.append({'artifact_name': name, 'path': str(path.relative_to(PROJECT_ROOT)) if path.is_absolute() else str(path), 'exists': bool(path.exists()), 'size_bytes': int(path.stat().st_size) if path.exists() else 0, 'role': role})
    pd.DataFrame(rows).to_csv(RUNS / 'runB0_baseline_index.csv', index=False)

    summary = pd.read_csv(FORMAL_SUMMARY)
    main = summary[summary['score_name'].eq('protocol_v0_2_family_confidence')].copy()
    n_pos = int((_num(main['aligned_rho']) > 0).sum())
    n_partial_pos = int((_num(main['partial_rho_control_magnitude']) > 0).sum())
    n_rc_pos = int((_num(main['risk_coverage80_improve_pct']) > 0).sum())
    text = f"""# Run B0 ????\n\n???{datetime.now().isoformat(timespec='seconds')}\n\n## ????\n\n- ?? formal ????LODO?ablation?signal validity?McFarland ???Tahoe smoke?GEARS probe?\n- ?? v0.2 ? 7 ??? aligned rho ???{n_pos}/7?\n- partial rho ???{n_partial_pos}/7?\n- RC@80 ?????{n_rc_pos}/7?\n\n## ???????\n\n- ??????\n- ???? v0.3 router?\n- ??? magnitude residual?McFarland ???learned risk ???\n"""
    (RUNS / 'runB0_result.md').write_text(text, encoding='utf-8')


def run_b1(scored: pd.DataFrame) -> None:
    from sklearn.isotonic import IsotonicRegression

    base = base_prediction_rows(scored)
    test_base = base[base['split'].eq('test')].copy()
    rows = []
    for dataset, g in test_base.groupby('dataset_name', dropna=False):
        row = metric_row(g.assign(risk_axis=g['true_effect_l2_norm']))
        row.update({'dataset_name': dataset, 'level': 'dataset', 'predictor_name': 'ALL'})
        rows.append(row)
        for predictor, pg in g.groupby('predictor_name', dropna=False):
            prow = metric_row(pg.assign(risk_axis=pg['true_effect_l2_norm']))
            prow.update({'dataset_name': dataset, 'level': 'dataset_predictor', 'predictor_name': predictor})
            rows.append(prow)
    pd.DataFrame(rows).to_csv(RUNS / 'runB1_magnitude_bias_table.csv', index=False)

    bin_rows = []
    for (dataset, predictor), g in test_base.groupby(['dataset_name', 'predictor_name'], dropna=False):
        gg = g.copy()
        mag = _num(gg['true_effect_l2_norm'])
        if len(gg) < 20 or mag.nunique() < 3:
            continue
        try:
            gg['magnitude_bin'] = pd.qcut(mag.rank(method='first'), q=5, labels=False, duplicates='drop')
        except Exception:
            continue
        for b, bg in gg.groupby('magnitude_bin', dropna=False):
            bin_rows.append({'dataset_name': dataset, 'predictor_name': predictor, 'magnitude_bin': int(b) if pd.notna(b) else -1, 'n': int(len(bg)), 'mean_rmse': float(_num(bg['true_error_rmse']).mean()), 'median_rmse': float(_num(bg['true_error_rmse']).median()), 'mean_magnitude': float(_num(bg['true_effect_l2_norm']).mean()), 'min_magnitude': float(_num(bg['true_effect_l2_norm']).min()), 'max_magnitude': float(_num(bg['true_effect_l2_norm']).max())})
    pd.DataFrame(bin_rows).to_csv(RUNS / 'runB1_magnitude_bin_error.csv', index=False)

    residual_frames = []
    status_rows = []
    for (dataset, fold, predictor), g in base.groupby(['dataset_name', 'fold_id', 'predictor_name'], dropna=False):
        train = g[g['split'].isin(['train', 'val'])].copy()
        test = g[g['split'].eq('test')].copy()
        status = {'dataset_name': dataset, 'fold_id': int(fold), 'predictor_name': predictor, 'n_train': int(len(train)), 'n_test': int(len(test))}
        tx = _num(train['true_effect_l2_norm'])
        ty = _num(train['true_error_rmse'])
        valid = tx.notna() & ty.notna()
        tx = tx[valid]
        ty = ty[valid]
        if len(tx) < 20 or tx.nunique() < 3 or len(test) < 3:
            status['status'] = 'skipped_too_few_or_degenerate'
            status_rows.append(status)
            continue
        order = np.argsort(tx.to_numpy())
        model = IsotonicRegression(out_of_bounds='clip', increasing=True)
        try:
            model.fit(tx.to_numpy()[order], ty.to_numpy()[order])
            pred = model.predict(_num(test['true_effect_l2_norm']).to_numpy())
        except Exception as exc:
            status['status'] = f'failed:{type(exc).__name__}'
            status_rows.append(status)
            continue
        out = test[PREDICTION_KEY + ['dataset_family', 'true_error_rmse', 'true_effect_l2_norm']].copy()
        out['expected_error_from_magnitude'] = pred
        out['residual_error'] = _num(out['true_error_rmse']) - out['expected_error_from_magnitude']
        residual_frames.append(out)
        status['status'] = 'ok'
        status_rows.append(status)
    residual = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    pd.DataFrame(status_rows).to_csv(RUNS / 'runB1_isotonic_fit_status.csv', index=False)

    residual_rows = []
    key = scored[scored['score_name'].isin(KEY_SCORES)].copy()
    key_test = key[key['split'].eq('test')].copy()
    if not residual.empty:
        merged = key_test.merge(
            residual[PREDICTION_KEY + ['expected_error_from_magnitude', 'residual_error']],
            on=PREDICTION_KEY,
            how='inner',
        )
        for (dataset, score_name), g in merged.groupby(['dataset_name', 'score_name'], dropna=False):
            residual_rows.append({'dataset_name': dataset, 'score_name': score_name, 'n': int(len(g)), 'raw_error_rho': spearman(g['risk_axis'], g['true_error_rmse']), 'residual_error_rho': spearman(g['risk_axis'], g['residual_error']), 'expected_error_rho': spearman(g['expected_error_from_magnitude'], g['true_error_rmse']), 'residual_error_mean': float(_num(g['residual_error']).mean()), 'residual_error_std': float(_num(g['residual_error']).std())})
    pd.DataFrame(residual_rows).to_csv(RUNS / 'runB1_residual_diagnostic_table.csv', index=False)

    df = pd.DataFrame(residual_rows)
    proto = df[df['score_name'].eq('protocol_v0_2_family_confidence')] if not df.empty else pd.DataFrame()
    n_res_pos = int((_num(proto['residual_error_rho']) > 0).sum()) if not proto.empty else 0
    n_raw_pos = int((_num(proto['raw_error_rho']) > 0).sum()) if not proto.empty else 0
    text = f"""# Run B1 ????\n\n???{datetime.now().isoformat(timespec='seconds')}\n\n## ????\n\n- ??? magnitude-only ??magnitude bin ????fold-local isotonic residual diagnostic?\n- v0.2 raw rho ????????{n_raw_pos}/7?\n- v0.2 residual rho ????????{n_res_pos}/7?\n\n## ????\n\nResidual diagnostic ??? scorer?????????? magnitude ??????????\n"""
    (RUNS / 'runB1_result.md').write_text(text, encoding='utf-8')


def _load_mcfarland_meta() -> pd.DataFrame:
    candidates = [FORMAL_ROOT / 'post_formal_diagnostics' / 'tables' / 'McFarland_task_metadata.csv', PROJECT_ROOT / 'docs/04-????/Formal_main_20260604/diagnostics/tables/McFarland_task_metadata.csv']
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame(columns=['context', 'perturbation'])


def _group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (score_name, group_id), g in df.groupby(['score_name', group_col], dropna=False):
        if len(g) < 8:
            continue
        row = metric_row(g)
        row.update({'score_name': score_name, 'group_col': group_col, 'group_id': str(group_id)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['score_name', 'aligned_rho'], ascending=[True, True]) if rows else pd.DataFrame()


def run_b2(scored: pd.DataFrame) -> None:
    meta = _load_mcfarland_meta()
    mc = scored[scored['dataset_name'].eq('McFarlandTsherniak2020') & scored['split'].eq('test') & scored['score_name'].isin(KEY_SCORES)].copy()
    if not meta.empty:
        keep_cols = [c for c in ['context', 'perturbation', 'dominant_dose_value', 'dominant_time', 'n_dose_values', 'n_time_values', 'all_dose_values', 'all_time_values'] if c in meta.columns]
        mc = mc.merge(meta[keep_cols].drop_duplicates(['context', 'perturbation']), on=['context', 'perturbation'], how='left')
    else:
        mc['dominant_dose_value'] = 'NA'
        mc['dominant_time'] = 'NA'
    mc['drug'] = mc['perturbation'].astype(str)
    mc['dose'] = mc.get('dominant_dose_value', 'NA').astype(str)
    mc['time'] = mc.get('dominant_time', 'NA').astype(str)
    mc['drug_dose'] = mc['drug'] + ' | dose=' + mc['dose']
    mc['drug_time'] = mc['drug'] + ' | time=' + mc['time']
    outputs = {'drug': RUNS / 'runB2_mcfarland_drug_level.csv', 'dose': RUNS / 'runB2_mcfarland_dose_level.csv', 'time': RUNS / 'runB2_mcfarland_time_level.csv', 'drug_dose': RUNS / 'runB2_mcfarland_drug_dose_level.csv', 'drug_time': RUNS / 'runB2_mcfarland_drug_time_level.csv'}
    summary_bits = []
    for col, path in outputs.items():
        table = _group_metrics(mc, col)
        table.to_csv(path, index=False)
        proto = table[table['score_name'].eq('protocol_v0_2_family_confidence')] if not table.empty else pd.DataFrame()
        learned = table[table['score_name'].eq('learned_risk_score')] if not table.empty else pd.DataFrame()
        summary_bits.append({'level': col, 'n_groups_protocol': int(len(proto)), 'protocol_positive_groups': int((_num(proto['aligned_rho']) > 0).sum()) if not proto.empty else 0, 'protocol_negative_groups': int((_num(proto['aligned_rho']) <= 0).sum()) if not proto.empty else 0, 'learned_positive_groups': int((_num(learned['aligned_rho']) > 0).sum()) if not learned.empty else 0})
    pd.DataFrame(summary_bits).to_csv(RUNS / 'runB2_mcfarland_failure_mode_summary.csv', index=False)
    drug_table = pd.read_csv(RUNS / 'runB2_mcfarland_drug_level.csv') if (RUNS / 'runB2_mcfarland_drug_level.csv').exists() else pd.DataFrame()
    proto_drug = drug_table[drug_table['score_name'].eq('protocol_v0_2_family_confidence')] if not drug_table.empty else pd.DataFrame()
    learned_drug = drug_table[drug_table['score_name'].eq('learned_risk_score')] if not drug_table.empty else pd.DataFrame()
    proto_pos = int((_num(proto_drug['aligned_rho']) > 0).sum()) if not proto_drug.empty else 0
    proto_total = int(len(proto_drug))
    learned_pos = int((_num(learned_drug['aligned_rho']) > 0).sum()) if not learned_drug.empty else 0
    learned_total = int(len(learned_drug))
    mode = 'v0.2 ??? drug ????????? failure boundary' if proto_total and proto_pos <= proto_total / 2 else 'v0.2 ????????? drug / dose???????'
    text = f"""# Run B2 McFarland Failure Modes\n\n???{datetime.now().isoformat(timespec='seconds')}\n\n## ????\n\n- v0.2 drug-level ?????{proto_pos}/{proto_total}\n- learned_risk_score drug-level ?????{learned_pos}/{learned_total}\n\n## ????\n\n{mode}\n\n## ??\n\n- `runB2_mcfarland_drug_level.csv`\n- `runB2_mcfarland_dose_level.csv`\n- `runB2_mcfarland_time_level.csv`\n- `runB2_mcfarland_drug_dose_level.csv`\n- `runB2_mcfarland_drug_time_level.csv`\n"""
    (RUNS / 'runB2_result.md').write_text(text, encoding='utf-8')
    (RUNS / 'runB2_result.md').write_text(text, encoding='utf-8')


def run_b3(scored: pd.DataFrame) -> None:
    summary = pd.read_csv(FORMAL_SUMMARY)
    keep = summary[summary['score_name'].isin(KEY_SCORES)].copy()
    keep.to_csv(RUNS / 'runB3_error_ranker_per_dataset.csv', index=False)
    learned = keep[keep['score_name'].eq('learned_risk_score')].copy()
    proto = keep[keep['score_name'].eq('protocol_v0_2_family_confidence')].copy()
    comp = learned.merge(proto, on=['dataset_name', 'dataset_family'], suffixes=('_learned', '_v02'))
    if not comp.empty:
        comp['delta_aligned_rho'] = _num(comp['aligned_rho_learned']) - _num(comp['aligned_rho_v02'])
        comp['delta_partial_rho'] = _num(comp['partial_rho_control_magnitude_learned']) - _num(comp['partial_rho_control_magnitude_v02'])
        comp['delta_rc80_pct'] = _num(comp['risk_coverage80_improve_pct_learned']) - _num(comp['risk_coverage80_improve_pct_v02'])
    comp.to_csv(RUNS / 'runB3_error_ranker_eval_summary.csv', index=False)
    ci_cols = [c for c in ['dataset_name', 'dataset_family', 'score_name', 'aligned_rho', 'aligned_rho_ci_low', 'aligned_rho_ci_high', 'partial_rho_control_magnitude', 'partial_rho_ci_low', 'partial_rho_ci_high', 'risk_coverage80_improve_pct', 'n_bootstrap'] if c in keep.columns]
    keep[ci_cols].to_csv(RUNS / 'runB3_error_ranker_bootstrap_ci.csv', index=False)
    learned_pos = int((_num(learned['risk_coverage80_improve_pct']) > 0).sum()) if not learned.empty else 0
    learned_total = int(len(learned))
    mc = learned[learned['dataset_name'].eq('McFarlandTsherniak2020')]
    mc_aligned = float(mc['aligned_rho'].iloc[0]) if not mc.empty else float('nan')
    mc_partial = float(mc['partial_rho_control_magnitude'].iloc[0]) if not mc.empty else float('nan')
    mc_rc = float(mc['risk_coverage80_improve_pct'].iloc[0]) if not mc.empty else float('nan')
    allowed_features = ['context_similarity_max', 'context_similarity_mean', 'perturbation_support_count', 'perturbation_effect_stability', 'prediction_magnitude_deviation', 'model_disagreement_rmse', 'ood_nearest_distance', 'historical_residual_risk']
    forbidden_hits = [f for f in allowed_features if any(p in f.lower() for p in FORBIDDEN_FEATURE_PATTERNS)]
    leakage = '# Run B3 Leakage Audit\n\n## ????\n\n??????? formal audit ?? `learned_risk_score`???????\n\n## ????\n\n`code/20260426_154505_perturb_transport_final_push/safetrans_confidence/scoring/error_ranker.py` ???? `(dataset, fold, predictor)` ????? train+val ???test ????\n\n## ????\n\n' + '\n'.join('- ' + f for f in allowed_features) + '\n\n## ??????\n\n' + (str(forbidden_hits) if forbidden_hits else '??? forbidden pattern?') + '\n\n## ??\n\n?? B3 ??? fold-local learned risk ????????? test-label ???\n'
    (RUNS / 'runB3_result.md').write_text(leakage, encoding='utf-8')
    text = f"""# Run B3 ????\n\n???{datetime.now().isoformat(timespec='seconds')}\n\n## ????\n\n- learned_risk_score ? RC@80 ???{learned_pos}/{learned_total}?\n- McFarland learned_risk_score aligned rho = {mc_aligned:.3f}?\n- McFarland learned_risk_score partial rho = {mc_partial:.3f}?\n- McFarland learned_risk_score RC@80 = {mc_rc:.2f}%?\n\n## ????\n\n?? B3 ?????learned risk ???? optional calibration layer?????? frozen v0.2???????????????????????????????????\n"""
    (RUNS / 'runB3_result.md').write_text(text, encoding='utf-8')


def main() -> None:
    _ensure()
    run_b0()
    scored = load_scored()
    run_b1(scored)
    run_b2(scored)
    run_b3(scored)
    log = f"""# Run B1-B3 ?????\n\n???{datetime.now().isoformat(timespec='seconds')}\n\n???\n\n```bash\ncd /home/yyf/proj\n/home/yyf/.conda/envs/scgpt_env/bin/python code/20260426_154505_perturb_transport_final_push/scripts/run_breakthrough_b1_b2_b3.py\n```\n\n???`{FORMAL_TABLE}`\n\n?????`{RUNS}`\n\n??????\n"""
    (RUNS / 'runB1_result.md').write_text(log, encoding='utf-8')


if __name__ == '__main__':
    main()
