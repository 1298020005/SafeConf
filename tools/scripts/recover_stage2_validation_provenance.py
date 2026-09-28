#!/usr/bin/env python3
"""Forensic replay of old validation records from fold-permitted train caches."""
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928'
SRC=Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
BASE=Path('/home/yyf/safeconf_runtime');TOL=1e-5

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def selected_csv(path,meta_cols,select,columns):
 meta=pd.read_csv(path,usecols=meta_cols);ix=set(meta.index[select(meta)].to_numpy()+1)
 return pd.read_csv(path,usecols=columns,skiprows=lambda i:i!=0 and i not in ix)

def main():
 start=time.time(); rows=[]; provenance=[]
 meta=pd.read_csv(SRC,usecols=['dataset_name','fold_id','split','task_key','predictor_name','run_dir'])
 base=meta[(meta.fold_id==0)&meta.split.isin(['train','val'])][['dataset_name','task_key']].drop_duplicates()
 for ds,b in base.groupby('dataset_name'):
  ids=set(b.task_key);ref=meta[meta.dataset_name==ds];run=BASE/ref.run_dir.iloc[0]
  source_cols=['dataset_name','task_key','fold_id','split','predictor_name','true_error_rmse','prediction_l2_norm','prediction_abs_mean']
  want=lambda x:(x.dataset_name==ds)&x.task_key.isin(ids)&x.split.eq('val')
  vals=selected_csv(SRC,['dataset_name','task_key','split'],want,source_cols)
  spl=pd.read_csv(run/'input/HELDOUT_PAIR_SPLITS.csv')
  recmeta=pd.read_csv(run/'tables/PREDICTION_RECORDS.csv',usecols=['task_key','task_id','context','perturbation','fold_id','split','predictor_name','predicted_effect_key','true_effect_key','target_control_key'])
  sm=spl[['fold_id','task_key','split']].merge(recmeta[['fold_id','task_key','split']].drop_duplicates(),on=['fold_id','task_key'],suffixes=('_split','_record'),validate='one_to_one')
  assert len(sm)==len(spl) and sm.split_split.eq(sm.split_record).all()
  with np.load(run/'input/true_effects.npz') as tz,np.load(run/'input/predicted_effects.npz') as pz,np.load(run/'input/target_control_means.npz') as cz:
   for fold,v in vals.groupby('fold_id'):
    train=recmeta[(recmeta.fold_id==fold)&recmeta.split.eq('train')&recmeta.predictor_name.eq('V0StrongBaseline')].sort_values('task_id')
    allowed=set(spl[(spl.fold_id==fold)&spl.split.eq('train')].task_key)
    assert set(train.task_key)==allowed and not (set(v.task_key)&allowed)
    y=np.stack([tz[k] for k in train.true_effect_key]);ctrl=np.stack([cz[k] for k in train.target_control_key])
    pp=train.perturbation.to_numpy(str);cc=train.context.to_numpy(str);global_mean=y.mean(0)
    query=recmeta[(recmeta.fold_id==fold)&recmeta.split.eq('val')&recmeta.task_key.isin(ids)]
    identity=hashlib.sha256('\n'.join(sorted(allowed)).encode()).hexdigest()
    for key,vv in v.groupby('task_key'):
     raw=query[query.task_key==key];first=raw.iloc[0];context=str(first.context);pert=str(first.perturbation)
     true=np.asarray(tz[first.true_effect_key],np.float32);control=np.asarray(cz[first.target_control_key],np.float64)
     same=np.flatnonzero(pp==pert);basepred=y[same].mean(0) if len(same) else global_mean
     cp=np.flatnonzero(cc==context)
     v0=(.85*basepred+.15*y[cp].mean(0)) if len(cp) else basepred
     idx=np.flatnonzero((pp==pert)&(cc!=context));idx=idx if len(idx) else same
     if len(idx):
      sims=np.array([float(np.dot(control,ctrl[j])/(np.linalg.norm(control)*np.linalg.norm(ctrl[j])+1e-8)) for j in idx])
      w=np.exp(5*(sims-sims.max()));w/=w.sum();sim=(w[:,None]*y[idx]).sum(0).astype(np.float32)
     else: sim=global_mean
     for predictor,pred in [('V0StrongBaseline',v0),('ContextSimBaseline',sim),('PertMeanPredictor',basepred)]:
      target=vv[vv.predictor_name==predictor].iloc[0]
      reconstructed=float(np.sqrt(np.mean((pred.astype(np.float32)-true)**2)))
      errdiff=abs(reconstructed-float(target.true_error_rmse));normdiff=abs(float(np.linalg.norm(pred))-float(target.prediction_l2_norm))
      old=raw[raw.predictor_name==predictor]
      vd=float(np.max(np.abs(np.asarray(pz[old.iloc[0].predicted_effect_key])-pred))) if len(old) else np.nan
      match=errdiff<=TOL and normdiff<=TOL and (not len(old) or vd<=TOL)
      rows.append(dict(dataset=ds,task_key=key,original_fold=int(fold),predictor=predictor,upstream_state_id=f'{ds}/fold{fold}/{predictor}/{identity[:12]}',split='val',n_upstream_train=len(train),query_in_train=False,split_manifest_match=True,error_replay_absdiff=errdiff,magnitude_replay_absdiff=normdiff,vector_replay_maxdiff=vd,status='REPLAY_VERIFIED_DEVELOPMENT' if match else 'REPLAY_MISMATCH',history_features_recovered=False,error_memory_accepted=False))
    print(f'{ds} fold {fold}: replayed {len(v)} validation records',flush=True)
  provenance.append(dict(dataset=ds,run_dir=str(run),split_sha256=sha(run/'input/HELDOUT_PAIR_SPLITS.csv'),prediction_archive_sha256=sha(run/'input/predicted_effects.npz'),true_archive_sha256=sha(run/'input/true_effects.npz')))
  pd.DataFrame(rows).to_csv(OUT/'RECOVERY_REPLAY_ROWS.csv',index=False)
 r=pd.DataFrame(rows); r.groupby(['dataset','predictor','status']).size().rename('prediction_records').to_csv(OUT/'RECOVERY_REPLAY_COUNTS.csv')
 unique=r.groupby(['dataset','task_key']).status.apply(lambda x:x.eq('REPLAY_VERIFIED_DEVELOPMENT').all())
 status=dict(status='COMPLETE',candidate_unique_tasks=int(len(unique)),all_records_pass_unique_tasks=int(unique.sum()),n_prediction_records=len(r),passed_prediction_records=int(r.status.eq('REPLAY_VERIFIED_DEVELOPMENT').sum()),elapsed_seconds=time.time()-start,vector_tolerance=TOL,error_tolerance=TOL,source_sha256=sha(SRC),new_deep_upstream_training_runs=0,test_error_records_used=0,history_features_recovered=False,can_pool_different_upstream_folds_for_risk_without_further_audit=False,limits=['Replay demonstrates fold-source compatibility, not historical temporal attestation.','Records remain retrospective development; source folds are different upstream states.','Q/H and cross-model Error Memory are not automatically recovered.'])
 (OUT/'RECOVERY_REPLAY_STATUS.json').write_text(json.dumps(status,indent=2)+'\n');(OUT/'RECOVERY_REPLAY_SOURCES.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':main()
