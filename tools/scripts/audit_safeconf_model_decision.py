from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from sklearn.model_selection import GroupKFold
root=Path(__file__).resolve().parents[2]
out=root/'docs/实验结果/ModelDecision_light_batch_20260928'
out.mkdir(parents=True,exist_ok=True)
src=Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
meta=['dataset_name','fold_id','split','task_key','context','perturbation','predictor_name','run_dir']
x=pd.read_csv(src,usecols=meta)
allowed=x.index[(x.fold_id==0)&x.split.isin(['train','val'])].to_numpy()
keep=set(allowed+1)
f=['prediction_l2_norm','prediction_abs_mean','model_disagreement_rmse','model_disagreement_cosine','context_similarity_max','context_similarity_mean','perturbation_support_count','prediction_norm_ratio','prediction_magnitude_deviation','historical_residual_risk','perturbation_effect_stability','perturbation_effect_variance']
z=pd.read_csv(src,usecols=meta+f,skiprows=lambda i:i!=0 and i not in keep)
rows=[]; folds=[]
for ds,g in z[z.predictor_name=='PertMeanPredictor'].groupby('dataset_name'):
 v=g[g.split=='val']; masks=[]
 for k,(a,b) in enumerate(GroupKFold(3).split(v,groups=v.perturbation)):
  folds.append(dict(dataset=ds,fold=k,n_fit=len(a),n_eval=len(b),fit_perts=v.iloc[a].perturbation.nunique(),eval_perts=v.iloc[b].perturbation.nunique()))
 chem=ds in ['McFarlandTsherniak2020','SrivatsanTrapnell2020_sciplex3']
 rows.append(dict(dataset=ds,type='chemical' if chem else 'gene',tasks_train_val=len(g),tasks_val=len(v),contexts=g.context.nunique(),perturbations=g.perturbation.nunique(),val_contexts=v.context.nunique(),val_perturbations=v.perturbation.nunique(),public_same_pert_support_pct=100*(v.perturbation_support_count>0).mean(),public_multicontext_content_pct=100*v[['perturbation_effect_stability','perturbation_effect_variance']].notna().all(axis=1).mean(),history_residual_finite_pct=100*v.historical_residual_risk.notna().mean(),verified_checkpoint_linked_error_records=0,recorded_heldout_error_candidates=3*len(v)))
pd.DataFrame(rows).to_csv(out/'DATA_COUNTS.csv',index=False)
pd.DataFrame(folds).to_csv(out/'SPLIT_COUNTS.csv',index=False)
z[z.split=='val'].groupby(['dataset_name','predictor_name'])[f].agg(lambda c:1-np.isfinite(c).mean()).to_csv(out/'FEATURE_MISSINGNESS.csv')
manifest=dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),metadata_rows=len(x),feature_rows_train_val=len(z),risk_training_candidate_rows=len(z[z.split=='val']),unique_validation_tasks=552,final_test_label_rows_used=0,error_labels_loaded_by_audit=False,source_fold=0,n_splits=3,feature_order=f)
manifest['original_run_paths_present']={p: (Path('/home/yyf/safeconf_runtime')/p).exists() for p in z.run_dir.unique()}
(out/'DATA_AUDIT.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(folds).to_string(index=False))
