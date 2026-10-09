#!/usr/bin/env python3
"""One bounded CPU training repair; original failed predictor package preserved.

No new architecture or target expression is introduced. Control embedding
features are standardized on upstream training genes; MLP learns centered
responses from a zero residual head with an upstream-only stopping split.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from sklearn.linear_model import Ridge
from tools.scripts import run_safeconf_gladstone_v21 as base
from tools.safeconf_continual.submission_evidence import write_json,sha,digest_ids

ORIGINAL=base.OUT
OUT=base.RUN/'external_predictor_training_repair_v1'
MAX_STEPS=300
PATIENCE=30


def prepare():
    OUT.mkdir(exist_ok=True)
    card={'status':'FIXED_BEFORE_REPAIR_DEV_SCORING','reason':str(ORIGINAL/'PREDICTOR_FIRST_FAILURE_DIAGNOSIS.json'),
        'original_package':str(ORIGINAL),'original_predictors_and_failure_preserved':True,
        'repair_number':1,'new_architectures':0,'new_downloads':0,'new_gpu_hours':0,
        'fixed_inputs':'same 50-dimensional control-only embedding',
        'ridge':{'alpha':100,'input_standardization':'training genes only'},
        'mlp':{'hidden':256,'learning_rate':.001,'max_optimizer_steps':MAX_STEPS,'patience':PATIENCE,
            'initial_last_layer':'zero','target':'response minus upstream-training context mean',
            'stopping':'first 10% of upstream gene hashes only, including step-zero baseline',
            'full_refit':'all allowed upstream genes; chosen step count; three original seeds'},
        'competence_gate':'unchanged user v2.1; no variance or error tolerance changes',
        'confirmation_truth_stays_sealed_until_capability_and_risk_freeze':True,
        'cpu_thread_cap':4,'wall_seconds_cap':1800,'sample_or_axis_filtering_after_results':False}
    p=OUT/'REPAIR_EXPERIMENT_CARD.json'
    if p.exists() and json.loads(p.read_text())!=card:raise RuntimeError('different repair configuration at fixed run id')
    write_json(p,card)
    input_sources={}
    for name in ['TASK_ROLES.parquet','OBS_ROLE_REGISTRY.parquet','ROLE_FREEZE.json','OUTPUT_CONTRACT.json',
                 'TRAIN_FEEDBACK_TASKS.parquet','TRAIN_FEEDBACK_EFFECTS.npy','TRAIN_FEEDBACK_READ_RECEIPT.json',
                 'CONTROL_COUNTS.npz','CONTROL_FEATURES.npz','CONTROL_FEATURE_AUDIT.json',
                 'PUBLIC_COUNTS_EFFECTS.npz','PUBLIC_SNAPSHOT.json']:
        src=ORIGINAL/name;dst=OUT/name
        # KOLF is a separate frozen study root. Its role registry, public
        # snapshot, and control features are owned regular files, not links to
        # the earlier CD4 study. Accept them only in the already materialized
        # run root and bind their hashes below. Missing legacy-only control
        # arrays are not needed by the repair path.
        if dst.exists():
            input_sources[name]={'path':str(dst),'kind':'owned_frozen_file','sha256':sha(dst)}
            continue
        if not src.exists():
            if name=='CONTROL_COUNTS.npz':continue
            raise RuntimeError(f'missing shared input: {src}')
        dst.symlink_to(src)
        input_sources[name]={'path':str(dst),'kind':'shared_immutable_symlink','sha256':sha(dst)}
    write_json(OUT/'SHARED_RESOURCE_CACHE.json',{'cache':str(ORIGINAL/'range_cache/pseudobulk'),
        'ledger':str(ORIGINAL/'DOWNLOAD_RESOURCE_LEDGER.json'),'budget_restarted':False})
    write_json(OUT/'REPAIR_INPUT_MANIFEST.json',{'inputs':input_sources,
        'selection_truth_role':'predictor_train only; feedback error used only for unchanged qualification',
        'original_competence_sha256':sha(ORIGINAL/'PREDICTOR_COMPETENCE.json')})
    base.OUT=OUT
    base.CONFIG=dict(base.CONFIG,external_predictors=card)


def network(seed,n_outputs):
    torch.manual_seed(seed)
    net=torch.nn.Sequential(torch.nn.Linear(50,256),torch.nn.ReLU(),torch.nn.Linear(256,n_outputs))
    torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    return net


def fit():
    if (OUT/'PREDICTOR_FREEZE.json').exists():return
    if (OUT/'PREDICTOR_FIT_LEDGER.json').exists():base.freeze_saved_predictors();return
    torch.set_num_threads(4);start=time.monotonic()
    task=pd.read_parquet(OUT/'TRAIN_FEEDBACK_TASKS.parquet');effect=np.load(OUT/'TRAIN_FEEDBACK_EFFECTS.npy')
    roles=pd.read_parquet(OUT/'TASK_ROLES.parquet');z=np.load(OUT/'CONTROL_FEATURES.npz')
    emb={str(g):e for g,e in zip(z['genes'],z['embedding'])};records=[];diagnoses=[];means=np.zeros((len(roles),effect.shape[1]),np.float32)
    def check_budget():
        if time.monotonic()-start>1800:raise RuntimeError('fixed CPU training-repair wall budget reached')
    for c in base.CONTEXTS:
        rows=np.flatnonzero(task.context.eq(c)&task.role.eq('predictor_train'))
        known=np.asarray([i for i in rows if task.iloc[i].gene in emb])
        x=np.asarray([emb[task.iloc[i].gene] for i in known],np.float32);y=effect[known]
        mu=effect[rows].mean(0);xm=x.mean(0);xs=x.std(0);xs=np.where(xs>1e-8,xs,1.)
        standardized=(x-xm)/xs
        ridge=Ridge(alpha=100).fit(standardized,y)
        np.savez(OUT/f'RIDGE_{c}.npz',coef=ridge.coef_,intercept=ridge.intercept_,mean=mu,x_mean=xm,x_scale=xs)
        means[np.flatnonzero(roles.context.eq(c))]=mu
        order=sorted(range(len(known)),key=lambda i:hashlib.sha256(f'SafeConf-v21-MLP-upstream-repair|{task.iloc[known[i]].gene}'.encode()).hexdigest())
        nval=max(1,len(order)//10);vi=np.asarray(order[:nval]);ti=np.asarray(order[nval:])
        # Transform and mean for the stopping experiment exclude its validation genes.
        inner_mean=x[ti].mean(0);inner_scale=x[ti].std(0);inner_scale=np.where(inner_scale>1e-8,inner_scale,1.)
        inner_mu=y[ti].mean(0)
        xt=torch.tensor((x[ti]-inner_mean)/inner_scale);yt=torch.tensor(y[ti]-inner_mu)
        xv=torch.tensor((x[vi]-inner_mean)/inner_scale);yv=torch.tensor(y[vi]-inner_mu)
        for seed in base.MODEL_SEEDS:
            check_budget();net=network(seed,y.shape[1]);opt=torch.optim.Adam(net.parameters(),lr=.001)
            best_loss=float(torch.mean(yv*yv));best_step=0;stale=0;history=[]
            for step in range(1,MAX_STEPS+1):
                net.train();opt.zero_grad();loss=torch.nn.functional.mse_loss(net(xt),yt);loss.backward();opt.step()
                net.eval()
                with torch.no_grad():vl=float(torch.nn.functional.mse_loss(net(xv),yv))
                history.append({'step':step,'train_mse':float(loss.detach()),'inner_validation_mse':vl})
                if vl<best_loss-1e-10:best_loss=vl;best_step=step;stale=0
                else:stale+=1
                if stale>=PATIENCE:break
                check_budget()
            # Refit on all allowed upstream genes; feedback/confirmation answers never read here.
            net=network(seed,y.shape[1]);opt=torch.optim.Adam(net.parameters(),lr=.001)
            full_x=torch.tensor(standardized);full_y=torch.tensor(y-mu)
            losses=[]
            for step in range(best_step):
                opt.zero_grad();loss=torch.nn.functional.mse_loss(net(full_x),full_y);loss.backward();opt.step()
                losses.append(float(loss.detach()));check_budget()
            net.eval();torch.save(net.state_dict(),OUT/f'MLP_{c}_{seed}.pt')
            records.append({'context':c,'seed':seed,'n_fit_rows':len(known),'epochs':best_step,
                'loss_first':losses[0] if losses else float(torch.mean(full_y*full_y)),
                'loss_last':losses[-1] if losses else float(torch.mean(full_y*full_y)),
                'all_losses':losses,'fit_gene_hash':digest_ids(task.iloc[known].gene),'query_truth_used':False,
                'selected_by_upstream_inner_split':True,'inner_fit_genes':len(ti),'inner_validation_genes':len(vi)})
            diagnoses.append({'context':c,'seed':seed,'best_step':best_step,'inner_best_mse':best_loss,
                'inner_baseline_mse':float(torch.mean(yv*yv)),'inner_learning_curve':history})
            print(json.dumps({'repair_context':c,'seed':seed,'best_step':best_step,
                'inner_best_mse':best_loss,'inner_baseline_mse':float(torch.mean(yv*yv))}),flush=True)
    roles.to_parquet(OUT/'PREDICTION_TASKS.parquet',index=False)
    np.save(OUT/'CONDITION_MEAN_ALL_PREDICTIONS.npy',means)
    write_json(OUT/'PREDICTOR_FIT_LEDGER.json',records);write_json(OUT/'UPSTREAM_ONLY_STOPPING_LEDGER.json',diagnoses)
    base.freeze_saved_predictors()
    write_json(OUT/'TRAINING_REPAIR_RESOURCE_COST.json',{'wall_seconds':time.monotonic()-start,
        'new_gpu_hours':0,'new_download_bytes':0,'initial_failed_upstream_preserved':True,'cpu_threads':4})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['fit','complete'],default='complete');a=ap.parse_args()
    prepare();fit();base.qualification()
    if a.phase=='complete':
        base.risk_freeze()
        if (OUT/'RISK_FREEZE.json').exists():base.confirmation()


if __name__=='__main__':main()
