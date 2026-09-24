#!/usr/bin/env python3
"""E257 modern upstream fixed; post-prediction risk models only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs/实验结果/E205_cross_family_disagreement_20260830/formal_evaluation/E205_TASK_METRICS.csv'
OUT = ROOT / 'docs/实验结果/E257_modern_small_risk_model_20260924'
FEATURES = {
    'A_M': ['predicted_magnitude'],
    'A_D': ['predicted_magnitude', 'registered_family_disagreement'],
    'A_D_Bio': ['predicted_magnitude', 'registered_family_disagreement',
                'model_source_gap', 'source_delta_dispersion',
                'negative_log_source_cells', 'support_context_deficit',
                'source_delta_dispersion_observed', 'n_source_contexts'],
}


def utility(score: np.ndarray, errors: np.ndarray) -> float:
    n = len(score)
    k = max(1,int(np.ceil(.2*n)))
    def picked(values: np.ndarray) -> np.ndarray:
        threshold = np.partition(values,n-k)[n-k]
        above, ties = values > threshold, values == threshold
        weights = above.astype(float)
        weights[ties] = (k-int(above.sum()))/int(ties.sum())
        return weights
    average = float(errors.mean())
    oracle = float(np.dot(picked(errors),errors)/k-average)
    return float((np.dot(picked(score),errors)/k-average)/oracle) if oracle > 1e-12 else float('nan')


def prepare(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray,np.ndarray]:
    tr = train[features].astype(float).replace([np.inf,-np.inf],np.nan).copy()
    te = test[features].astype(float).replace([np.inf,-np.inf],np.nan).copy()
    missing_tr = tr.isna().astype(float)
    missing_te = te.isna().astype(float)
    medians = tr.median().fillna(0.0)
    tr = tr.fillna(medians)
    te = te.fillna(medians)
    return (np.concatenate([tr.to_numpy(),missing_tr.to_numpy()],axis=1),
            np.concatenate([te.to_numpy(),missing_te.to_numpy()],axis=1))


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    raw = pd.read_csv(SOURCE)
    data = raw.loc[raw.analysis_stratum.eq('primary_ge30')].copy()
    if len(data) != 1808 or data.task_id.nunique() != 1808 or data.target.nunique() != 4:
        raise ValueError('E205 primary inventory changed')
    data['source_delta_dispersion_observed'] = data['source_delta_dispersion_observed'].astype(float)
    rows = []
    for target in sorted(data.target.unique()):
        tr = data.loc[data.target.ne(target)]
        te = data.loc[data.target.eq(target)]
        if set(tr.task_id) & set(te.task_id):
            raise ValueError(f'{target}: train/test task leakage')
        errors = te.family_rms_error.to_numpy(float)
        for name,column in (('M', 'predicted_magnitude'),
                            ('frozen_M_plus_S','safeconf_m_4to1')):
            score = te[column].to_numpy(float)
            rows.append({'target':target,'method':name,'feature_group':'reference',
                         'n_source':len(tr),'n_target':len(te),
                         'spearman':float(spearmanr(score,errors).statistic),
                         'utility_20':utility(score,errors)})
        for group,features in FEATURES.items():
            x_train,x_test = prepare(tr,te,features)
            y_train = tr.family_rms_error.to_numpy(float)
            models = {
                'ridge':Pipeline([('scale',StandardScaler()),('reg',Ridge(alpha=10))]),
                'histgb':HistGradientBoostingRegressor(max_iter=60,max_depth=3,
                         learning_rate=.05,min_samples_leaf=20,l2_regularization=.1,
                         early_stopping=False,random_state=257),
            }
            for model_name,model in models.items():
                model.fit(x_train,y_train)
                score = model.predict(x_test)
                rows.append({'target':target,'method':model_name,'feature_group':group,
                             'n_source':len(tr),'n_target':len(te),
                             'spearman':float(spearmanr(score,errors).statistic),
                             'utility_20':utility(score,errors)})
        print(f'{target}: three input groups × two small models complete',flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'E257_ALL_RESULTS.csv',index=False)
    comparison = []
    for (method,group),part in result.groupby(['method','feature_group']):
        baseline = result.loc[result.method.eq('M')].set_index('target')
        frozen = result.loc[result.method.eq('frozen_M_plus_S')].set_index('target')
        now = part.set_index('target')
        row = {'method':method,'feature_group':group,'n_targets':len(now)}
        for metric in ('spearman','utility_20'):
            row[f'mean_{metric}'] = float(now[metric].mean())
            row[f'delta_vs_M_{metric}'] = float((now[metric]-baseline.loc[now.index,metric]).mean())
            row[f'positive_targets_vs_M_{metric}'] = int((now[metric]>baseline.loc[now.index,metric]).sum())
            row[f'delta_vs_frozen_{metric}'] = float((now[metric]-frozen.loc[now.index,metric]).mean())
            row[f'positive_targets_vs_frozen_{metric}'] = int((now[metric]>frozen.loc[now.index,metric]).sum())
        comparison.append(row)
    summary = pd.DataFrame(comparison)
    summary.to_csv(OUT/'E257_SUMMARY.csv',index=False)
    status = {'status':'RETROSPECTIVE_MODEL_DEVELOPMENT','n_tasks':len(data),
              'n_targets':4,'n_small_model_fits':24,'source_model_predictions_reused':True,
              'target_truth_previously_public':True,'test_task_overlap_with_training':0}
    (OUT/'E257_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
