#!/usr/bin/env python3
"""Metadata-only recovery inventory; no errors, vectors, or model fitting."""
from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/实验结果/ModelDecision_light_batch_20260928'
SRC=Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
m=pd.read_csv(SRC,usecols=['dataset_name','task_key','predictor_name','fold_id','split'])
same=m.groupby(['dataset_name','task_key','fold_id'])
assert same.split.nunique().eq(1).all() and same.predictor_name.nunique().eq(3).all()
x=m[m.predictor_name=='PertMeanPredictor'].copy()
b=x[(x.fold_id==0)&x.split.isin(['train','val'])][['dataset_name','task_key','split']].rename(columns={'split':'fold0_role'})
assert len(b)==3668 and not b.duplicated(['dataset_name','task_key']).any()
x=x.merge(b,on=['dataset_name','task_key'],validate='many_to_one')
rows=[]; details=[]
for ds,g in x.groupby('dataset_name'):
 base=b[b.dataset_name==ds]; initial=set(base.loc[base.fold0_role=='val','task_key'])
 vals=set(g.loc[g.split=='val','task_key']); tests=set(g.loc[g.split=='test','task_key']); held=vals|tests
 for key in base.task_key:
  k=g[g.task_key==key]
  details.append(dict(dataset=ds,task_key=key,fold0_role=k.fold0_role.iloc[0],existing_val_folds=';'.join(str(i) for i in sorted(k.loc[k.split=='val','fold_id'])),existing_test_folds=';'.join(str(i) for i in sorted(k.loc[k.split=='test','fold_id'])),recovery_status='CANDIDATE_METADATA_ONLY'))
 rows.append(dict(dataset=ds,fold0_train_val_tasks=len(base),current_val_tasks=len(initial),any_fold_val_candidate_tasks=len(vals),additional_val_candidate_tasks=len(vals-initial),any_fold_heldout_candidate_tasks=len(held),additional_requires_test_role_metadata=len(held-vals),no_heldout_metadata_tasks=len(set(base.task_key)-held),currently_newly_verified_formal_tasks=0))
r=pd.DataFrame(rows);r.to_csv(OUT/'RECOVERABLE_TASKS.csv',index=False)
pd.DataFrame(details).to_csv(OUT/'RECOVERABLE_TASK_KEYS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
status={'metadata_columns_read':['dataset_name','task_key','predictor_name','fold_id','split'],'labels_or_expression_read':False,'new_training_runs':0,'unique_original_development_tasks':3668,'all_other_fold_val_candidates':int(r.any_fold_val_candidate_tasks.sum()),'additional_val_candidates':int(r.additional_val_candidate_tasks.sum()),'any_heldout_role_upper_bound':int(r.any_fold_heldout_candidate_tasks.sum()),'new_provenance_verified_formal_tasks':0,'caution':'Different outer folds have different upstream training sets/states. Do not combine as one frozen model. Test-role metadata are not permission to consume sealed truth. Repeated tasks/predictors/folds are not independent samples.'}
(OUT/'RECOVERY_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print(r.to_string(index=False));print(json.dumps(status,indent=2))
