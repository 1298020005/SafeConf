#!/usr/bin/env python3
"""Read released assets, check predictions, and report upstream competence."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];PUB=Path('/home/yyf/proj/docs/实验结果');OUT=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928'
rows=[]; errors=[];manifest=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def scan(label,kind,path,records,ckpts,split_manifest):
 p=Path(path);f=p/records;r=pd.read_csv(f)
 assert (p/'arrays/predicted_effects.npz').exists() and (p/'arrays/true_effects.npz').exists()
 with np.load(p/'arrays/predicted_effects.npz') as pred,np.load(p/'arrays/true_effects.npz') as truth:
  for predictor,g in r.groupby('predictor_name'):
   for fold,gg in g.groupby('fold_id'):
    e=[];b=[];diff=[]
    for t in gg.itertuples():
     y=truth[t.true_effect_key].astype(float);q=pred[t.predicted_effect_key].astype(float)
     assert y.shape==q.shape and np.isfinite(y).all() and np.isfinite(q).all()
     actual=float(np.sqrt(np.mean((q-y)**2)));zero=float(np.sqrt(np.mean(y*y)))
     e.append(actual);b.append(zero);diff.append(abs(actual-t.true_error_rmse))
    errors.append(dict(asset=label,type=kind,predictor=predictor,fold=str(fold),n_records=len(gg),n_unique_tasks=gg.task_id.nunique(),rmse_mean=np.mean(e),zero_rmse_mean=np.mean(b),relative_rmse_improvement_vs_zero=1-np.mean(e)/np.mean(b),fraction_tasks_beats_zero=np.mean(np.array(e)<np.array(b)),error_recompute_max_absdiff=max(diff),passes_2pct_zero_gate=np.mean(e)<=.98*np.mean(b)))
   rows.append(dict(asset=label,type=kind,predictor=predictor,n_prediction_records=len(g),n_unique_biological_tasks=g.task_id.nunique(),n_upstream_states=g.fold_id.nunique(),n_contexts=g.context.nunique(),prediction_vectors=True,released_truth_vectors=True,checkpoint_files=sum(Path(c).exists() for c in ckpts),split_manifest_exists=Path(split_manifest).exists(),records_path=str(f),prediction_archive=str(p/'arrays/predicted_effects.npz'),split_manifest=str(split_manifest),reuse_without_training=True,evidence_role='RELEASED_RETROSPECTIVE_DEVELOPMENT'))
 manifest.append(dict(asset=label,records_sha256=sha(f),predictions_sha256=sha(p/'arrays/predicted_effects.npz'),truth_sha256=sha(p/'arrays/true_effects.npz'),split_sha256=sha(split_manifest),checkpoints=[{'path':str(c),'exists':Path(c).exists(),'sha256':sha(c) if Path(c).exists() else None} for c in ckpts]))

for ds in ['Lara_exvivo','Santinha']:
 scan('E112_'+ds,'gene',PUB/'E112_external_formal_dual_models_20260713'/ds,'PREDICTION_RECORDS.csv',[],PUB/'E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv')
parent=PUB/'E84_cpa_rdkit_cartesian_formal_20260712'
for p in sorted((parent/'manifests').iterdir()):
 if not p.is_dir():continue
 scan('E84_'+p.name,'chemical',p,'tables/PREDICTION_RECORDS.csv',[p/'raw_cpa/cpa_rdkit_state.pt'],PUB/'E81_sciplex_cartesian_contract_20260712/tables/E81_SPLIT_MANIFEST.csv')
for exp,folder,ckpt,contract in [('E87','E87_sciplex_to_openproblems_cpa_20260712','cpa_cross_dataset_state.pt','E86_sciplex_to_openproblems_contract_20260712/tables/E86_CROSS_DATASET_MANIFEST.csv'),('E89','E89_sciplex3_to_sciplex4_cpa_20260712','cpa_state.pt','E88_sciplex4_dose_response_contract_20260712/tables/E88_CROSS_DATASET_MANIFEST.csv')]:
 p=PUB/folder
 # E89 contract filename is discovered explicitly if the old name differs.
 c=PUB/contract
 if exp=='E89' and not c.exists():
  candidates=list(PUB.glob('E88*/tables/*MANIFEST.csv'))
  if len(candidates)!=1:raise ValueError(candidates)
  c=candidates[0]
 scan(exp,'chemical',p,'tables/PREDICTION_RECORDS.csv',[p/'raw_cpa'/ckpt],c)
pd.DataFrame(rows).to_csv(OUT/'MATURE_ASSETS.csv',index=False)
e=pd.DataFrame(errors);e.to_csv(OUT/'UPSTREAM_COMPETENCE.csv',index=False)
(OUT/'MATURE_ASSET_HASHES.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(e[['asset','predictor','fold','n_records','relative_rmse_improvement_vs_zero','passes_2pct_zero_gate']].to_string(index=False))
