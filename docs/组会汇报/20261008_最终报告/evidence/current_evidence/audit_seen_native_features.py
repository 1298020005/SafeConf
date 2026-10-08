"""Read only previously open McFaline features and saved XGBoost trees.

No model fitting, inference, new truth, raw expression, or TEST access.
All new files are confined to this agent's report directory.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
FB = Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1')
NATIVE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/native_control_reference/TASK_FEATURES.parquet')
RUN = Path('/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/native_public_current_20261007_v1')
native = pd.read_parquet(NATIVE).set_index('task_id')
config = json.loads((RUN/'EXECUTION_CONFIG.json').read_text())
cols = config['feature_sets']['Native61']
public_cols = [c for c in config['feature_sets']['Native61_Public'] if c not in cols]
inputs = [NATIVE, RUN/'EXECUTION_CONFIG.json']
stats = []
coverage = []
for role, file in [('feedback_pool', 'FIXED_POOL_FEATURES.parquet'), ('seen_holdout', 'HOLDOUT_FEATURES.parquet')]:
    path = FB/file
    inputs.append(path)
    base = pd.read_parquet(path)
    joined = base.join(native[cols], on='task_id', how='left', rsuffix='_old_native')
    joined['native_prediction_abs_mean'] = joined.prediction_abs_mean
    for target, group in joined.groupby('target'):
        coverage.append({'role':role,'target':target,'rows':len(group), 'genes':group.gene.nunique(),
                         'missing_control_embedding_rows':group.native_embedding_0.isna().sum(),
                         'missing_control_embedding_genes':group.loc[group.native_embedding_0.isna(), 'gene'].nunique(),
                         'exact_states':group[['context','treatment']].drop_duplicates().shape[0]})
        for feature in cols + public_cols + ['simple_history_risk','support_risk','predicted_magnitude']:
            values = group[feature].to_numpy(float)
            truth = group.true_error_rmse.to_numpy(float)
            ok = np.isfinite(values) & np.isfinite(truth)
            rho = float(spearmanr(values[ok],truth[ok]).statistic) if ok.sum()>2 and np.ptp(values[ok])>0 else np.nan
            stats.append({'role':role,'target':target,'feature':feature, 'n_rows':len(group),
                          'finite_rows':int(ok.sum()),'unique_finite_values':len(np.unique(values[ok])),
                          'spearman_with_existing_rmse':rho,
                          'interpretation':'SEEN descriptive association; no direction or feature selected'})
pd.DataFrame(stats).to_csv(OUT/'NATIVE_PUBLIC_FEATURE_ASSOCIATIONS.csv',index=False)
pd.DataFrame(coverage).to_csv(OUT/'NATIVE_FEATURE_COVERAGE.csv',index=False)
splits = []
for path in sorted(RUN.glob('model_*.json')):
    v = json.loads(path.read_text())
    name = path.name
    feature_set = 'Native61_Public' if name.startswith('model_Native61_Public_') else 'Native61'
    names = config['feature_sets'][feature_set]
    counts = np.zeros(len(names), dtype=int)
    gains = np.zeros(len(names), dtype=float)
    trees = v['learner']['gradient_booster']['model']['trees']
    for tree in trees:
        for feature, child, gain in zip(tree['split_indices'],tree['left_children'],tree['loss_changes']):
            if child >= 0:
                counts[feature] += 1
                gains[feature] += gain
    for i, feature in enumerate(names):
        splits.append({'model_file':str(path),'feature_set':feature_set,'feature':feature,
                       'split_count':int(counts[i]),'total_split_gain':float(gains[i]),
                       'n_trees':len(trees),'interpretation':'saved-tree structure; not causal feature importance'})
    inputs.append(path)
pd.DataFrame(splits).to_csv(OUT/'NATIVE_SAVED_TREE_SPLITS.csv',index=False)
summary = {'role':'SEEN_READ_ONLY_FEATURE_AND_MODEL_STRUCTURE_AUDIT',
           'source_predictions_or_truth_changed':False,'new_model_fits':0,'new_inference_calls':0,
           'raw_expression_reads':0,'new_TEST_truth_reads':0,
           'missing_similarity_note':'cached native similarity is intentionally NaN; NativePredictor reconstructs it per fit',
           'similarity_fit_note':'training prototypes include each training gene itself; compare official CV and deployment conventions separately',
           'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
(OUT/'NATIVE_FEATURE_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(pd.DataFrame(coverage).to_string(index=False))
frame = pd.DataFrame(stats)
print(frame[frame.feature.isin(['native_control_baseline','native_control_dropout','native_control_plate_variance_proxy',
                              'native_prediction_abs_mean','predicted_magnitude','log_history_support',
                              'prediction_prior_rmse','simple_history_risk','support_risk'])].to_string(index=False))
frame = pd.DataFrame(splits)
print(frame.groupby(['feature_set','feature']).split_count.sum().unstack(0).loc[
    ['native_training_similarity','native_prediction_abs_mean']+public_cols].to_string())
