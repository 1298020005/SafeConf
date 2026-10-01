"""Read-only E195 validation competence check; no new fitting or test truth access.

Only X rows belonging to the already registered validation sets or training
controls are materialized. The original cache and checkpoints are never mutated.
All six previously frozen checkpoints are reported without selection by outcomes.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pickle
import time
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from gears.model import GEARS_Model

OUT = Path(__file__).resolve().parent
REPO = Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
OLD = Path('/home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730')
H5 = Path('/home/yyf/data/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def decode(v):
    return v.decode() if isinstance(v, bytes) else str(v)

def rows_dense(f, ids):
    """Read CSR values for explicitly permitted row IDs, never a full X array."""
    ids = np.asarray(ids, dtype=int)
    ptr = f['X/indptr'][:]
    x = np.zeros((len(ids), int(f['X'].attrs['shape'][1])), dtype=np.float32)
    for out_i, row_i in enumerate(ids):
        a,b = int(ptr[row_i]),int(ptr[row_i+1])
        ix = f['X/indices'][a:b]
        x[out_i, ix] = f['X/data'][a:b]
    return x

start = time.time()
registry = pd.read_csv(REPO/'docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/tables/E195_RAW_ARTIFACT_HASHES.csv')
hashes = dict(zip(registry.local_path, registry.sha256))
rows = []
manifest = []
vectors = {}
with h5py.File(H5, 'r') as f:
    cats = np.array([decode(x) for x in f['obs/condition/categories'][:]])
    labels = cats[f['obs/condition/codes'][:]]
    names = [decode(x) for x in f['var/gene_name'][:]]
    ng = len(names)
    gene_to_idx = {g:i for i,g in enumerate(names)}
    ctrl_ids = np.flatnonzero(labels=='ctrl')
    # Fixed 32-control inference batch and all training controls for baseline.
    ctrl = rows_dense(f, ctrl_ids)
    ctrl_mean = ctrl.mean(0)
    ctrl_batch = ctrl[:32]
    ctrl_batch_mean = ctrl_batch.mean(0)
    accessed_validation = set()
    for panel in ['P1','P2']:
        for seed in [11,22,33]:
            status_path = OLD/f'panels/{panel}/raw_gears/seed_{seed}/E195_SEED_STATUS.json'
            status = json.loads(status_path.read_text())['child_status']
            conditions = status['actual_condition_sets']['val']
            test = set(status['actual_condition_sets']['test'])
            assert not set(conditions)&test
            assert status['n_genes']==ng==5025
            model_dir = OLD/f'panels/{panel}/raw_gears/seed_{seed}/norman/seed_{seed}/model'
            for name in ['config.pkl','model.pt']:
                path=model_dir/name
                actual=sha(path)
                assert actual==hashes[str(path)], (path,actual,hashes.get(str(path)))
                manifest.append({'panel':panel,'seed':seed,'role':name,'path':str(path),'sha256':actual,'hash_match':True})
            with (model_dir/'config.pkl').open('rb') as c:
                config=pickle.load(c)
            config['device']=DEVICE
            model=GEARS_Model(config).to(DEVICE)
            model.load_state_dict(torch.load(model_dir/'model.pt',map_location='cpu'))
            model.eval()
            batch=torch.repeat_interleave(torch.arange(32,device=DEVICE),ng)
            for condition in conditions:
                ids=np.flatnonzero(labels==condition)
                assert len(ids)>0
                accessed_validation.update(ids.tolist())
                true=rows_dense(f,ids).mean(0)-ctrl_mean
                pert=np.zeros(ng,dtype=np.float32)
                for g in condition.split('+'):
                    if g!='ctrl': pert[gene_to_idx[g]]=1
                xx=np.stack([ctrl_batch,np.broadcast_to(pert,ctrl_batch.shape)],axis=-1)
                inp=SimpleNamespace(x=torch.from_numpy(xx.reshape(-1,2)).to(DEVICE),batch=batch)
                with torch.no_grad():
                    pred,logvar=model(inp)
                # GEARS adds control expression to learned delta, so subtract the
                # exact inference batch mean to compare centered effects.
                effect=pred.cpu().numpy().mean(0)-ctrl_batch_mean
                vectors[f'{panel}__seed{seed}__{condition}__pred'] = effect
                vectors[f'{panel}__{condition}__val_truth'] = true
                rmse=float(np.sqrt(np.mean((effect-true)**2)))
                zero_rmse=float(np.sqrt(np.mean(true**2)))
                p=float(pearsonr(effect,true)[0])
                rows.append({'panel':panel,'seed':seed,'condition':condition,'n_validation_cells':len(ids),'n_genes':ng,'model_rmse':rmse,'control_rmse':zero_rmse,'rmse_gain_control_minus_model':zero_rmse-rmse,'relative_rmse_gain':(zero_rmse-rmse)/zero_rmse,'effect_pearson':p,'native_logvar_mean':float(logvar.cpu().numpy().mean()),'scope':'registered_validation_only','new_model_fit':False})
            print(json.dumps({'panel':panel,'seed':seed,'completed_validation_conditions':len(conditions)}),flush=True)
            del model
    access={'expression_path':str(H5),'n_gene_columns':ng,'n_control_rows':len(ctrl_ids),'n_distinct_validation_rows':len(accessed_validation),'test_expression_rows_materialized':0,'full_X_materialized':False,'test_array_files_opened':0,'fixed_inference_control_rows':32,'device':DEVICE,'new_training_gpu_hours':0,'new_download_bytes':0,'test_used_for_model_selection':False,'elapsed_seconds':time.time()-start}

tasks=pd.DataFrame(rows)
tasks.to_csv(OUT/'ASSET_AGENT_AUDIT.validation_tasks.csv',index=False)
summary=[]
for (panel,seed),df in tasks.groupby(['panel','seed']):
    summary.append({'panel':panel,'seed':int(seed),'n_validation_conditions':len(df),'mean_model_rmse':float(df.model_rmse.mean()),'mean_control_rmse':float(df.control_rmse.mean()),'mean_relative_rmse_gain':float(df.relative_rmse_gain.mean()),'win_rate_vs_control':float((df.model_rmse<df.control_rmse).mean()),'mean_effect_pearson':float(df.effect_pearson.mean()),'median_effect_pearson':float(df.effect_pearson.median())})
pd.DataFrame(summary).to_csv(OUT/'ASSET_AGENT_AUDIT.validation_summary.csv',index=False)
family_rows=[]
for (panel,condition),df in tasks.groupby(['panel','condition']):
    effect=np.stack([vectors[f'{panel}__seed{s}__{condition}__pred'] for s in [11,22,33]]).mean(0)
    true=vectors[f'{panel}__{condition}__val_truth']
    rmse=float(np.sqrt(np.mean((effect-true)**2)))
    zero_rmse=float(np.sqrt(np.mean(true**2)))
    family_rows.append({'panel':panel,'condition':condition,'n_genes':ng,'n_members':3,'family_centroid_rmse':rmse,'control_rmse':zero_rmse,'relative_rmse_gain':(zero_rmse-rmse)/zero_rmse,'effect_pearson':float(pearsonr(effect,true)[0])})
family=pd.DataFrame(family_rows)
family.to_csv(OUT/'ASSET_AGENT_AUDIT.validation_family_tasks.csv',index=False)
family.groupby('panel').agg(n_conditions=('condition','size'),family_rmse=('family_centroid_rmse','mean'),control_rmse=('control_rmse','mean'),effect_pearson=('effect_pearson','mean'),win_rate_vs_control=('relative_rmse_gain',lambda x:float((x>0).mean()))).reset_index().to_csv(OUT/'ASSET_AGENT_AUDIT.validation_family_summary.csv',index=False)
np.savez_compressed(OUT/'ASSET_AGENT_AUDIT.validation_vectors.npz',**vectors)
pd.DataFrame(manifest).to_csv(OUT/'ASSET_AGENT_AUDIT.validation_checkpoint_hashes.csv',index=False)
(OUT/'ASSET_AGENT_AUDIT.validation_access.json').write_text(json.dumps(access,indent=2))
print(json.dumps(summary,indent=2),flush=True)
