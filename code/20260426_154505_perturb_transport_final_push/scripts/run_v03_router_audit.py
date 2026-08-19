#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from safetrans_confidence.data.records import load_merged_records
from safetrans_confidence.scoring.protocol_v0_2 import build_protocol_v0_2_scores, zscore_by_ref

CODE_ROOT = Path('/home/yyf/proj/code/20260426_154505_perturb_transport_final_push')
DATA_ROOT = Path('/home/yyf/data/singlecell_perturbation_atlas')
DEFAULT_OUT_DIR = Path('/home/yyf/safeconf_runtime/outputs/run_v03_router_audit')
AUDIT_TABLE = CODE_ROOT / 'outputs/safeconf_phaseB_full_dataset_audit/tables/FULL_DATASET_ELIGIBILITY.csv'

STANDARD_RUNS = {
    'McFarlandTsherniak2020': CODE_ROOT / 'outputs/safeconf_formal_main_20260604/McFarlandTsherniak2020',
    'SantinhaPlatt2023': CODE_ROOT / 'outputs/safeconf_phase1_main/SantinhaPlatt2023',
    'SrivatsanTrapnell2020_sciplex3': CODE_ROOT / 'outputs/safeconf_phase1_main/SrivatsanTrapnell2020_sciplex3',
    'SrivatsanTrapnell2020_sciplex4': CODE_ROOT / 'outputs/safeconf_supplement_v02_20260605/SrivatsanTrapnell2020_sciplex4',
}

H5AD_FILES = {
    'McFarlandTsherniak2020': DATA_ROOT / 'official_scperturb/McFarlandTsherniak2020.h5ad',
    'SantinhaPlatt2023': DATA_ROOT / 'official_scperturb/SantinhaPlatt2023.h5ad',
    'SrivatsanTrapnell2020_sciplex3': DATA_ROOT / 'official_scperturb/SrivatsanTrapnell2020_sciplex3.h5ad',
    'SrivatsanTrapnell2020_sciplex4': DATA_ROOT / 'official_scperturb/SrivatsanTrapnell2020_sciplex4.h5ad',
}

TAHOE_RUN = CODE_ROOT / 'outputs/safeconf_tahoe_sampled_formal_v1_20260605'
GENETIC_LINES = {'genetic', 'enhancer_or_regulatory', 'cytokine', 'gene'}
DOSE_COLS = ['dose_value', 'dose', 'dose_uM', 'concentration']
TIME_COLS = ['time', 'timepoint_hr', 'time_value', 'time_hr']
DOSE_UNIT_COLS = ['dose_unit', 'concentration_unit']
PERT_TYPE_COLS = ['perturbation_type', 'perturbation_type_2']


def _mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    counts = s.astype(str).value_counts()
    return counts.index[0] if len(counts) else np.nan


def _num(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return np.nan
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except Exception:
        return np.nan


def _norm_text(value: object) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip()


def _exact_key(row: pd.Series) -> str:
    drug = _norm_text(row.get('drug_name'))
    dose = row.get('dose_value')
    time = row.get('time_value')
    unit = _norm_text(row.get('dose_unit'))
    dose_key = 'NA' if pd.isna(dose) else f'{float(dose):.8g}'
    time_key = 'NA' if pd.isna(time) else f'{float(time):.8g}'
    return f'{drug}|{dose_key}|{unit}|{time_key}'


def _safe_log10(value: float) -> float:
    if pd.isna(value) or not np.isfinite(value) or value <= 0:
        return np.nan
    return float(math.log10(value))


def _raw_spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float('nan')
    return float(x[mask].corr(y[mask], method='spearman'))


def _rank_residual(values: pd.Series, control: pd.Series) -> pd.Series:
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


def _partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    rx = _rank_residual(x, control)
    ry = _rank_residual(y, control)
    mask = rx.notna() & ry.notna()
    if int(mask.sum()) < 3:
        return float('nan')
    return float(rx[mask].corr(ry[mask], method='pearson'))


def _risk_coverage80(group: pd.DataFrame) -> float:
    valid = group.dropna(subset=['risk_axis', 'true_error_rmse'])
    if len(valid) < 3:
        return float('nan')
    full = float(valid['true_error_rmse'].mean())
    keep = max(1, int(np.ceil(0.8 * len(valid))))
    kept = valid.sort_values('risk_axis', ascending=True).head(keep)
    kept_mean = float(kept['true_error_rmse'].mean())
    return 100.0 * (full - kept_mean) / full if full else float('nan')


def _load_effect_magnitudes(records: pd.DataFrame, npz_path: Path) -> pd.DataFrame:
    if not npz_path.exists():
        return pd.DataFrame({'record_id': records['record_id'], 'true_effect_l2_norm': np.nan})
    arrays = np.load(npz_path)
    rows = []
    for _, row in records.iterrows():
        key = str(row['true_effect_key'])
        arr = np.asarray(arrays[key], dtype=float).ravel() if key in arrays else None
        rows.append({'record_id': row['record_id'], 'true_effect_l2_norm': float(np.linalg.norm(arr)) if arr is not None else np.nan})
    return pd.DataFrame(rows)


def _load_dataset_meta(dataset: str, audit_row: pd.Series) -> pd.DataFrame:
    h5ad = H5AD_FILES.get(dataset)
    if h5ad is None or not h5ad.exists():
        return pd.DataFrame(columns=['context', 'perturbation', 'drug_name', 'dose_value', 'dose_unit', 'time_value', 'perturbation_type'])
    context_col = str(audit_row.get('context_col_used', 'context'))
    perturb_col = str(audit_row.get('perturbation_col_used', 'perturbation'))
    adata = ad.read_h5ad(h5ad, backed='r')
    obs_cols = set(adata.obs.columns)
    wanted = [context_col, perturb_col]
    dose_col = next((c for c in DOSE_COLS if c in obs_cols), None)
    time_col = next((c for c in TIME_COLS if c in obs_cols), None)
    unit_col = next((c for c in DOSE_UNIT_COLS if c in obs_cols), None)
    ptype_col = next((c for c in PERT_TYPE_COLS if c in obs_cols), None)
    for col in [dose_col, time_col, unit_col, ptype_col]:
        if col and col not in wanted:
            wanted.append(col)
    obs = adata.obs[wanted].copy()
    adata.file.close()
    obs = obs.rename(columns={context_col: 'context', perturb_col: 'perturbation'})
    obs['context'] = obs['context'].astype(str)
    obs['perturbation'] = obs['perturbation'].astype(str)
    if dose_col:
        obs['dose_value'] = pd.to_numeric(
            obs[dose_col].astype(str).map(_num), errors='coerce'
        )
    else:
        obs['dose_value'] = np.nan
    if time_col:
        obs['time_value'] = pd.to_numeric(
            obs[time_col].astype(str).map(_num), errors='coerce'
        )
    else:
        obs['time_value'] = np.nan
    obs['dose_unit'] = obs[unit_col].map(_norm_text) if unit_col else ''
    obs['perturbation_type'] = obs[ptype_col].map(_norm_text) if ptype_col else ''
    grouped = obs.groupby(['context', 'perturbation'], dropna=False).agg(
        drug_name=('perturbation', 'first'),
        dose_value=('dose_value', 'median'),
        dose_unit=('dose_unit', _mode),
        time_value=('time_value', 'median'),
        perturbation_type=('perturbation_type', _mode),
        n_cells_for_condition=('perturbation', 'size'),
    ).reset_index()
    return grouped


def _load_standard_dataset(dataset: str, run_dir: Path, audit_row: pd.Series) -> pd.DataFrame:
    records = pd.read_csv(run_dir / 'tables/PREDICTION_RECORDS.csv')
    base = load_merged_records(run_dir)
    protocol_scores, _ = build_protocol_v0_2_scores(base)
    v02 = protocol_scores[protocol_scores['score_name'].eq('protocol_v0_2_family_confidence')][['record_id', 'score_value']].rename(columns={'score_value': 'v02_score_value'})
    meta = _load_dataset_meta(dataset, audit_row)
    out = base.merge(v02, on='record_id', how='left')
    out = out.merge(meta, on=['context', 'perturbation'], how='left')
    mags = _load_effect_magnitudes(records, run_dir / 'input/true_effects.npz')
    out = out.merge(mags, on='record_id', how='left')
    out['dataset_line'] = str(audit_row.get('dataset_line', 'unknown'))
    out['source_kind'] = 'standard_safeconf_run'
    return out


def _load_tahoe() -> pd.DataFrame:
    rec = pd.read_csv(TAHOE_RUN / 'tables/TAHOE_PREDICTION_RECORDS_SMOKE.csv')
    scores = pd.read_csv(TAHOE_RUN / 'tables/TAHOE_PROTOCOL_SCORES_SMOKE.csv')
    v02 = scores[['record_id', 'score_value']].rename(columns={'score_value': 'v02_score_value'})
    out = rec.merge(v02, on='record_id', how='left')
    out['dataset_line'] = 'chemical'
    out['source_kind'] = 'tahoe_pseudobulk_smoke'
    out = out.rename(columns={
        'drug': 'drug_name',
        'concentration': 'dose_value',
        'concentration_unit': 'dose_unit',
        'exact_dose_support_count': 'exact_drug_dose_time_support_count',
        'drug_across_dose_support_count': 'same_drug_other_dose_support_count',
    })
    for col in ['context_similarity_max', 'perturbation_effect_stability', 'perturbation_effect_variance', 'historical_residual_risk', 'prediction_magnitude_deviation']:
        if col not in out.columns:
            out[col] = np.nan
    if 'time_value' not in out.columns:
        out['time_value'] = np.nan
    return out


def _assign_routes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['dose_value'] = pd.to_numeric(out.get('dose_value'), errors='coerce')
    out['time_value'] = pd.to_numeric(out.get('time_value'), errors='coerce')
    out['drug_name'] = out.get('drug_name', out['perturbation']).fillna(out['perturbation']).astype(str)
    has_dose_time = out['dose_value'].notna() | out['time_value'].notna()
    is_genetic = out['dataset_line'].astype(str).isin(GENETIC_LINES)
    is_chemical = out['dataset_line'].astype(str).eq('chemical')
    out['route_name'] = np.select(
        [is_genetic, is_chemical & has_dose_time, is_chemical & ~has_dose_time],
        ['gene_main_v02', 'drug_dose_time_v03', 'chem_robust_v02'],
        default='gene_main_v02',
    )
    out['condition_parse_quality'] = np.select(
        [is_genetic, is_chemical & has_dose_time, is_chemical & ~has_dose_time],
        ['genetic_or_nonchemical', 'drug_dose_time_metadata', 'chemical_no_reliable_dose_time'],
        default='fallback_gene_like',
    )
    return out


def _compute_support_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'exact_drug_dose_time_support_count' not in out.columns:
        out['exact_drug_dose_time_support_count'] = np.nan
    if 'same_drug_other_dose_support_count' not in out.columns:
        out['same_drug_other_dose_support_count'] = np.nan
    out['nearest_log_dose_gap'] = np.nan
    out['nearest_time_gap'] = np.nan
    out['_exact_key'] = out.apply(_exact_key, axis=1)
    out['_log_dose'] = out['dose_value'].map(_safe_log10)
    for (_, _fold, _predictor), idx_obj in out.groupby(['dataset_name', 'fold_id', 'predictor_name'], dropna=False).groups.items():
        idx = list(idx_obj)
        sub = out.loc[idx]
        train = sub[sub['split'].eq('train')].copy()
        if train.empty:
            continue
        for i in idx:
            row = out.loc[i]
            exact = train[train['_exact_key'].eq(row['_exact_key'])]
            exact = exact[~exact['context'].astype(str).eq(str(row['context']))]
            same_drug = train[train['drug_name'].astype(str).eq(str(row['drug_name']))]
            other = same_drug[~same_drug['_exact_key'].eq(row['_exact_key'])]
            if pd.isna(out.at[i, 'exact_drug_dose_time_support_count']):
                out.at[i, 'exact_drug_dose_time_support_count'] = int(exact['context'].nunique())
            if pd.isna(out.at[i, 'same_drug_other_dose_support_count']):
                out.at[i, 'same_drug_other_dose_support_count'] = int(other['context'].nunique())
            if pd.notna(row['_log_dose']):
                logs = pd.to_numeric(same_drug['_log_dose'], errors='coerce').dropna()
                out.at[i, 'nearest_log_dose_gap'] = float(np.min(np.abs(logs.to_numpy() - float(row['_log_dose'])))) if len(logs) else np.nan
            if pd.notna(row['time_value']):
                times = pd.to_numeric(same_drug['time_value'], errors='coerce').dropna()
                out.at[i, 'nearest_time_gap'] = float(np.min(np.abs(times.to_numpy() - float(row['time_value'])))) if len(times) else np.nan
    if 'perturbation_effect_stability' in out.columns:
        out['drug_effect_instability'] = -pd.to_numeric(out['perturbation_effect_stability'], errors='coerce')
    else:
        out['drug_effect_instability'] = np.nan
    if 'perturbation_effect_variance' in out.columns:
        out['drug_effect_instability'] = out['drug_effect_instability'].fillna(pd.to_numeric(out['perturbation_effect_variance'], errors='coerce'))
    return out.drop(columns=['_exact_key', '_log_dose'])


def _score_v03(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['v03_drug_dose_time_confidence'] = np.nan
    out['router_confidence_score'] = out['v02_score_value']
    out['log_exact_drug_dose_time_support'] = np.log1p(pd.to_numeric(out['exact_drug_dose_time_support_count'], errors='coerce').fillna(0))
    out['log_same_drug_other_dose_support'] = np.log1p(pd.to_numeric(out['same_drug_other_dose_support_count'], errors='coerce').fillna(0))
    for (_, _fold, _predictor), idx_obj in out.groupby(['dataset_name', 'fold_id', 'predictor_name'], dropna=False).groups.items():
        idx = list(idx_obj)
        sub = out.loc[idx]
        route_idx = sub[sub['route_name'].eq('drug_dose_time_v03')].index.tolist()
        if not route_idx:
            continue
        train = sub[sub['split'].eq('train') & sub['route_name'].eq('drug_dose_time_v03')]
        if train.empty:
            train = sub[sub['split'].eq('train')]
        if train.empty:
            train = sub
        z_exact = zscore_by_ref(sub['log_exact_drug_dose_time_support'], train['log_exact_drug_dose_time_support'])
        z_other = zscore_by_ref(sub['log_same_drug_other_dose_support'], train['log_same_drug_other_dose_support'])
        z_ctx = zscore_by_ref(sub['context_similarity_max'], train['context_similarity_max'])
        z_dis = zscore_by_ref(sub['model_disagreement_rmse'], train['model_disagreement_rmse'])
        z_dose_gap = zscore_by_ref(sub['nearest_log_dose_gap'], train['nearest_log_dose_gap'])
        z_time_gap = zscore_by_ref(sub['nearest_time_gap'], train['nearest_time_gap'])
        z_inst = zscore_by_ref(sub['drug_effect_instability'], train['drug_effect_instability'])
        score = z_exact + 0.5 * z_other + 0.5 * z_ctx - z_dis - 0.5 * z_dose_gap - 0.5 * z_time_gap - 0.5 * z_inst
        out.loc[route_idx, 'v03_drug_dose_time_confidence'] = score.loc[route_idx]
        out.loc[route_idx, 'router_confidence_score'] = score.loc[route_idx]
    return out


def _make_score_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        base = {
            'record_id': row['record_id'],
            'dataset_name': row['dataset_name'],
            'fold_id': row['fold_id'],
            'split': row['split'],
            'context': row['context'],
            'perturbation': row['perturbation'],
            'predictor_name': row['predictor_name'],
            'route_name': row['route_name'],
            'true_error_rmse': row['true_error_rmse'],
            'true_effect_l2_norm': row.get('true_effect_l2_norm', np.nan),
        }
        for name, val in [
            ('protocol_v0_2_family_confidence', row.get('v02_score_value')),
            ('router_confidence_score', row.get('router_confidence_score')),
            ('v03_drug_dose_time_confidence', row.get('v03_drug_dose_time_confidence')),
        ]:
            if pd.notna(val):
                rows.append({**base, 'score_name': name, 'score_type': 'confidence', 'score_value': float(val), 'risk_axis': -float(val)})
    return pd.DataFrame(rows)


def _eval_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = scores[scores['split'].eq('test')].copy()
    for (dataset, route, score), group in test.groupby(['dataset_name', 'route_name', 'score_name'], dropna=False):
        rows.append({
            'dataset_name': dataset,
            'route_name': route,
            'score_name': score,
            'n': int(len(group)),
            'aligned_rho': _raw_spearman(group['risk_axis'], group['true_error_rmse']),
            'partial_rho_control_magnitude': _partial_spearman(group['risk_axis'], group['true_error_rmse'], group['true_effect_l2_norm']),
            'risk_coverage80_improve_pct': _risk_coverage80(group),
            'mean_rmse': float(pd.to_numeric(group['true_error_rmse'], errors='coerce').mean()),
        })
    return pd.DataFrame(rows).sort_values(['dataset_name', 'route_name', 'score_name'])


def _missingness(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for (dataset, route), group in df.groupby(['dataset_name', 'route_name'], dropna=False):
        for feat in features:
            s = group[feat] if feat in group.columns else pd.Series([np.nan] * len(group))
            rows.append({'dataset_name': dataset, 'route_name': route, 'feature_name': feat, 'n_rows': int(len(group)), 'missing_rate': float(pd.isna(s).mean()) if len(group) else 1.0})
    return pd.DataFrame(rows)


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT_TABLE)
    audit_map = {row['dataset_name']: row for _, row in audit.iterrows()}
    frames = []
    status = []
    for dataset, run_dir in STANDARD_RUNS.items():
        try:
            frame = _load_standard_dataset(dataset, run_dir, audit_map[dataset])
            frames.append(frame)
            status.append({'dataset_name': dataset, 'status': 'ok', 'n_rows': int(len(frame)), 'run_dir': str(run_dir)})
        except Exception as exc:
            status.append({'dataset_name': dataset, 'status': 'failed', 'error': repr(exc), 'run_dir': str(run_dir)})
    try:
        tahoe = _load_tahoe()
        frames.append(tahoe)
        status.append({'dataset_name': 'Tahoe100M_pseudobulk_smoke', 'status': 'ok', 'n_rows': int(len(tahoe)), 'run_dir': str(TAHOE_RUN)})
    except Exception as exc:
        status.append({'dataset_name': 'Tahoe100M_pseudobulk_smoke', 'status': 'failed', 'error': repr(exc), 'run_dir': str(TAHOE_RUN)})
    if not frames:
        raise RuntimeError('No usable inputs for v0.3 router audit')
    base = pd.concat(frames, ignore_index=True, sort=False)
    base = _assign_routes(base)
    base = _compute_support_features(base)
    base = _score_v03(base)
    score_long = _make_score_long(base)
    eval_table = _eval_scores(score_long)
    candidate_cols = [
        'record_id', 'dataset_name', 'source_kind', 'fold_id', 'split', 'context', 'perturbation', 'predictor_name', 'dataset_line', 'route_name', 'condition_parse_quality',
        'drug_name', 'dose_value', 'dose_unit', 'time_value', 'exact_drug_dose_time_support_count', 'same_drug_other_dose_support_count', 'nearest_log_dose_gap', 'nearest_time_gap',
        'drug_effect_instability', 'context_similarity_max', 'model_disagreement_rmse', 'v02_score_value', 'v03_drug_dose_time_confidence', 'router_confidence_score', 'true_error_rmse', 'true_effect_l2_norm',
    ]
    candidate_cols = [c for c in candidate_cols if c in base.columns]
    base[candidate_cols].to_csv(out_dir / '2026-06-07_run02_v03_candidate_features.csv', index=False)
    score_long.to_csv(out_dir / '2026-06-07_run02_v03_scores_long.csv', index=False)
    eval_table.to_csv(out_dir / '2026-06-07_run02_v02_vs_v03_drug_table.csv', index=False)
    route_dist = base.groupby(['dataset_name', 'route_name', 'condition_parse_quality'], dropna=False).size().reset_index(name='n_records')
    route_dist.to_csv(out_dir / '2026-06-07_run02_router_distribution.csv', index=False)
    features = ['drug_name', 'dose_value', 'dose_unit', 'time_value', 'exact_drug_dose_time_support_count', 'same_drug_other_dose_support_count', 'nearest_log_dose_gap', 'nearest_time_gap', 'drug_effect_instability', 'context_similarity_max', 'model_disagreement_rmse']
    miss = _missingness(base, features)
    miss.to_csv(out_dir / '2026-06-07_run02_feature_missingness.csv', index=False)
    pd.DataFrame(status).to_csv(out_dir / '2026-06-07_run02_input_status.csv', index=False)
    main = eval_table[eval_table['score_name'].isin(['protocol_v0_2_family_confidence', 'router_confidence_score'])]
    result_lines = [
        '# Run 02 v0.3 入口选择审计结果',
        '',
        '## 一句话结论',
        '',
        'v0.3 router 已按预注册规则生成候选特征和初步分流分数；它是 v0.2 的补充审计，不替代冻结主协议。',
        '',
        '## 路由分布',
        '',
        '`	ext',
        route_dist.to_string(index=False),
        '`',
        '',
        '## v0.2 vs router 初步对照',
        '',
        '`	ext',
        main.to_string(index=False),
        '`',
        '',
        '## 解释边界',
        '',
        '- McFarland / sciplex / Tahoe 这类带 dose/time 元数据的 chemical 任务进入 drug_dose_time_v03。',
        '- Santinha 当前没有可靠 dose/time 元数据，保留 chem_robust_v02，不硬分流。',
        '- 本轮只证明入口选择和候选特征可计算；是否进入论文主表，要等 Run 04 做更严格的药物线验证。',
        '- v0.2 主表不改，不为 McFarland 单独调参。',
    ]
    (out_dir / '2026-06-07_run02_结果解读.md').write_text('\n'.join(result_lines) + '\n', encoding='utf-8')
    design = '# Run 02 v0.3 入口选择设计\n\n## 目的\n\n把 v0.2 不擅长的 drug + dose + time 场景单独识别出来，生成候选 v0.3 特征。\n\n## 固定路由\n\n`	ext\ngenetic / cytokine / gene-like -> gene_main_v02\nchemical without reliable dose-time -> chem_robust_v02\ndrug + dose and/or time -> drug_dose_time_v03\n`\n\n## 预注册公式\n\n`	ext\nconfidence =\n  z(log_exact_drug_dose_time_support)\n  + 0.5 * z(log_same_drug_other_dose_support)\n  + 0.5 * z(context_similarity)\n  - z(model_disagreement)\n  - 0.5 * z(nearest_log_dose_gap)\n  - 0.5 * z(nearest_time_gap)\n  - 0.5 * z(drug_effect_instability)\n`\n\n所有 z-score 只使用 train split 作为参考。\n\n## 禁止项\n\n不使用 true_error_rmse、真实 effect magnitude、test label 派生字段作为特征。\n'
    (out_dir / '2026-06-07_run02_v03_router_design.md').write_text(design, encoding='utf-8')
    status_obj = {
        'out_dir': str(out_dir),
        'n_rows': int(len(base)),
        'n_scores': int(len(score_long)),
        'n_eval_rows': int(len(eval_table)),
        'inputs': status,
        'outputs': [
            '2026-06-07_run02_v03_candidate_features.csv',
            '2026-06-07_run02_router_distribution.csv',
            '2026-06-07_run02_feature_missingness.csv',
            '2026-06-07_run02_v02_vs_v03_drug_table.csv',
            '2026-06-07_run02_结果解读.md',
        ],
    }
    (out_dir / '2026-06-07_run02_RUN_STATUS.json').write_text(json.dumps(status_obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return status_obj


def main() -> None:
    parser = argparse.ArgumentParser(description='Run SafeConf v0.3 router audit without modifying frozen v0.2.')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
