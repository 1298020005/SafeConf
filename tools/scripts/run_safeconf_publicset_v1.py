#!/usr/bin/env python3
"""DEV-only PublicSet: aligned biology, grouped fits, four matched builders.

No McFaline TEST arrays or raw expression are read by this runner. Source is
evaluated as rotating-context deployment with legal other-context history.
PCA is fitted on unique history rows actually used by fitting queries only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
COMMON = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
RUNTIME = COMMON.parent / 'publicset_execution_v1'
OUT = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1'
SOURCE_BANK = Path('/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201')
SEEDS = (20260930, 20261001, 20261002)
SOURCE_FIELDS = ['log_source_cells', 'log_source_batches', 'control_rmse', 'control_cosine', 'source_effect_magnitude', 'source_effect_abs_mean', 'source_conflict', 'support_fraction', 'quality_missing']
MC_FIELDS = ['same_context', 'same_condition', 'control_rmse', 'control_cosine', 'log_source_cells', 'log_source_guides', 'log_source_plates', 'guide_reproducibility', 'plate_reproducibility', 'split_half_stability', 'batch_agreement', 'source_effect_magnitude', 'source_conflict', 'support_fraction', 'quality_missing']


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def atomic_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    frame.to_csv(tmp, index=False, compression={'method': 'gzip', 'mtime': 0} if str(path).endswith('.gz') else None)
    os.replace(tmp, path)


def fingerprint(values):
    return hashlib.sha256('\n'.join(sorted(map(str, values))).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


@dataclass
class Domain:
    name: str
    tasks: pd.DataFrame
    memory: pd.DataFrame
    effects: np.ndarray
    controls: np.ndarray
    truth: np.ndarray
    query_controls: np.ndarray
    predictions: dict
    fields: list
    eligibility: dict | None
    old_effects: np.ndarray | None = None

    def forbidden(self, query_rows):
        # Source has independently declared deployment views for each context.
        # Training genes and their PCA rows are disjoint from ALL query genes.
        if self.name == 'Source': return set()
        return {'McFaline23::' + str(self.tasks.iloc[i].task_id) for i in query_rows}

    def groups(self, rows, forbidden):
        index = self.memory.set_index('experiment_id').effect_vector_row.to_dict()
        by_gene = self.memory.groupby('perturbation_target', sort=False).indices
        output = []
        for q in rows:
            task = self.tasks.iloc[int(q)]
            if self.eligibility is not None:
                ids = self.eligibility.get((str(task.context), str(task.condition)), [])
                ix = np.asarray([index[x] for x in ids if x not in forbidden], int)
            else:
                ix = np.asarray(by_gene.get(str(task.gene), []), int)
                if len(ix):
                    m = self.memory.iloc[ix]
                    ok = ~m.experiment_id.astype(str).isin(forbidden).to_numpy()
                    ok &= ~((m.context.astype(str) == str(task.context)) & (m.condition.astype(str) == str(task.condition))).to_numpy()
                    ix = ix[ok]
            if not len(ix):
                output.append({'q': int(q), 'ix': ix, 'x': np.empty((0, len(self.fields))), 's': np.empty(0), 'conflict': np.nan})
                continue
            m = self.memory.iloc[ix]
            h, control = self.effects[ix], self.controls[ix]
            raw_conflict = np.sqrt(np.mean((h - h.mean(0)) ** 2, axis=1))
            cells = m.n_cells.to_numpy(float)
            if not np.isfinite(cells).all() or (cells <= 0).any(): raise ValueError('invalid registered support')
            quality = m[['guide_reproducibility','plate_reproducibility','split_half_stability','batch_agreement']].to_numpy(float)
            features = dict(log_source_cells=np.log1p(cells), log_source_batches=np.log1p(m.n_batches.to_numpy(float)),
                control_rmse=np.sqrt(np.mean((control - self.query_controls[q]) ** 2, axis=1)),
                control_cosine=np.asarray([cosine(v, self.query_controls[q]) for v in control]),
                source_effect_magnitude=np.sqrt(np.mean(h ** 2, axis=1)), source_effect_abs_mean=np.mean(np.abs(h), axis=1),
                source_conflict=raw_conflict, support_fraction=cells/cells.sum(), quality_missing=(~np.isfinite(quality).all(1)).astype(float),
                same_context=(m.context.astype(str) == str(task.context)).to_numpy(float),
                same_condition=(m.condition.astype(str) == str(task.condition)).to_numpy(float),
                log_source_guides=np.log1p(m.n_guides.to_numpy(float)), log_source_plates=np.log1p(m.n_plates.to_numpy(float)))
            for j, col in enumerate(['guide_reproducibility','plate_reproducibility','split_half_stability','batch_agreement']): features[col] = quality[:, j]
            output.append({'q': int(q), 'ix': ix, 'x': np.column_stack([features[c] for c in self.fields]),
                's': cells/cells.sum(), 'conflict': float(raw_conflict.mean()), 'support': float(cells.sum())})
        return output


def load_domain(name):
    if name == 'Source':
        tasks = pd.read_csv(COMMON / 'SOURCE_TASKS.csv').rename(columns={'target':'context'})
        memory = pd.read_parquet(SOURCE_BANK / 'public_memory.parquet').sort_values('effect_vector_row').reset_index(drop=True)
        effects = np.asarray(np.load(COMMON/'SOURCE_PUBLIC_EFFECTS.npy', mmap_mode='r'), np.float32)
        controls = np.asarray(np.load(COMMON/'SOURCE_PUBLIC_CONTROLS.npy', mmap_mode='r'), np.float32)
        truth = np.asarray(np.load(COMMON/'SOURCE_TRUE_EFFECTS.npy', mmap_mode='r'), np.float32)
        base = pd.read_csv(ROOT/'docs/实验结果/E201_txpert_multitarget_retraining_20260802/tables/E201_PRETRUTH_TASK_BASE.csv').set_index('task_id').loc[tasks.task_id]
        source_genes = json.loads((SOURCE_BANK/'gene_ids.json').read_text())['gene_ids']
        target_genes = json.loads((COMMON/'GENE_IDS.json').read_text())
        lookup = {g:i for i,g in enumerate(source_genes)}
        cols = np.asarray([lookup[g] for g in target_genes])
        native_controls = np.load('/home/yyf/data/txpert_official_20260802/e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy', mmap_mode='r')
        qcontrol = np.asarray(native_controls[base.source_mean_delta_row.to_numpy(int)][:,cols],np.float32)
        pred = {model: np.asarray(np.load(COMMON/f'SOURCE_{model}_PREDICTED_EFFECTS.npy',mmap_mode='r'),np.float32) for model in ['TxPert_GAT','TxPert_Exphormer']}
        eligibility = pd.read_parquet(SOURCE_BANK/'eligibility.parquet').groupby(['target_context','condition']).public_experiment_id.apply(list).to_dict()
        result = Domain(name,tasks,memory,effects,controls,truth,qcontrol,pred,SOURCE_FIELDS,eligibility)
    else:
        tasks = pd.read_csv(COMMON/'VALIDATION_BIOLOGY_TASKS.csv').rename(columns={'perturbation':'gene','treatment':'condition'})
        tasks['fold'] = tasks.gene.map(lambda g:int.from_bytes(hashlib.sha256(f'SafeConf-McFaline-Decoder-repair-v1\0{g}'.encode()).digest()[:8],'big') % 5)
        bank = COMMON/'public_mcfaline_trainval'
        memory = pd.read_parquet(bank/'public_memory.parquet').sort_values('effect_vector_row').reset_index(drop=True)
        effects = np.asarray(np.load(COMMON/'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy',mmap_mode='r'),np.float32)
        controls = np.asarray(np.load(bank/'control_vectors.npy',mmap_mode='r'),np.float32)
        truth = np.asarray(np.load(COMMON/'VALIDATION_TRUE_EFFECTS.npy',mmap_mode='r'),np.float32)
        qcontrol = np.asarray(np.load(COMMON/'VALIDATION_CONTROLS.npy',mmap_mode='r'),np.float32)
        pred = {'DecoderOnly':np.asarray(np.load(COMMON/'VALIDATION_CALIBRATED_EFFECTS.npy',mmap_mode='r'),np.float32)}
        old = np.asarray(np.load(bank/'effect_vectors.npy',mmap_mode='r'),np.float32)
        result = Domain(name,tasks,memory,effects,controls,truth,qcontrol,pred,MC_FIELDS,None,old)
    if not np.array_equal(result.memory.effect_vector_row.to_numpy(),np.arange(len(result.memory))): raise ValueError('bank row mapping changed')
    if result.effects.shape[1] != 2840 or result.truth.shape != (len(result.tasks),2840): raise ValueError('registered axis changed')
    if result.tasks.groupby('gene').fold.nunique().max() != 1: raise ValueError('gene crossed folds')
    return result


class SetScorer(nn.Module):
    def __init__(self, n_features, collective):
        super().__init__(); self.collective = collective
        self.encoder = nn.Sequential(nn.Linear(n_features,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self, tokens, mask, support):
        u = self.encoder(tokens)
        if self.collective:
            c = (u * mask[...,None]).sum(1) / mask.sum(1,keepdim=True).clamp_min(1)
        else: c = torch.zeros_like(u[:,0])
        score = self.head(torch.cat([u,c[:,None].expand_as(u)],dim=-1)).squeeze(-1)
        score = score.masked_fill(~mask,-torch.inf)
        return .5*torch.softmax(score,dim=1)+.5*support


def fit_preprocessor(domain, groups):
    groups = [g for g in groups if len(g['ix'])]
    x = np.concatenate([g['x'] for g in groups]); missing = ~np.isfinite(x)
    median = np.asarray([np.median(x[np.isfinite(x[:,j]),j]) if np.isfinite(x[:,j]).any() else 0.0 for j in range(x.shape[1])])
    x = np.where(missing,median,x); mean=x.mean(0); scale=x.std(0); scale[scale<1e-8]=1
    rows = np.unique(np.concatenate([g['ix'] for g in groups]))
    components = min(32,len(rows)-1,domain.effects.shape[1])
    if components < 1: raise ValueError('insufficient fitting biology for PCA')
    pca = PCA(n_components=components,svd_solver='randomized',random_state=20260930).fit(domain.effects[rows])
    codes = pca.transform(domain.effects).astype(np.float32)
    pc_scale = codes[rows].std(0); pc_scale[pc_scale<1e-8]=1
    codes = codes/pc_scale
    return dict(median=median,mean=mean,scale=scale,pca=pca,codes=codes,pc_scale=pc_scale,history_rows=rows)


def pack(domain, groups, prep, device):
    groups = [g for g in groups if len(g['ix'])]
    if not groups: raise ValueError('no eligible queries')
    k = max(len(g['ix']) for g in groups); f=len(prep['median'])*2+prep['codes'].shape[1]
    tokens=np.zeros((len(groups),k,f),np.float32); indices=np.zeros((len(groups),k),int)
    mask=np.zeros((len(groups),k),bool); support=np.zeros((len(groups),k),np.float32)
    trainable=[]
    for i,g in enumerate(groups):
        n=len(g['ix']); miss=~np.isfinite(g['x']); x=np.where(miss,prep['median'],g['x'])
        tokens[i,:n]=np.c_[(x-prep['mean'])/prep['scale'],miss,prep['codes'][g['ix']]]
        indices[i,:n]=g['ix']; mask[i,:n]=True; support[i,:n]=g['s']
        h=domain.effects[g['ix']]; trainable.append(bool(n>1 and np.any(h[1:] != h[0])))
    q=np.asarray([g['q'] for g in groups],int)
    counts=domain.tasks.iloc[q].groupby('gene').gene.transform('size').to_numpy(float)
    weights=1/counts; weights/=weights.mean()
    return dict(groups=groups,q=q,tokens=torch.as_tensor(tokens,device=device),indices=torch.as_tensor(indices,device=device),
        mask=torch.as_tensor(mask,device=device),support=torch.as_tensor(support,device=device),
        weights=torch.as_tensor(weights,dtype=torch.float32,device=device),
        truth=torch.as_tensor(domain.truth[q],device=device),trainable=np.asarray(trainable))


def training_run(domain, groups, val_groups, collective, seed, epochs, device, early, stage_dir, deadline=None):
    start=time.monotonic(); prep=fit_preprocessor(domain,groups)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=SetScorer(len(prep['median'])*2+prep['codes'].shape[1],collective).to(device)
    fit=pack(domain,groups,prep,device); valid=pack(domain,val_groups,prep,device) if early else None
    bank=torch.tensor(domain.effects,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    order_rng=np.random.default_rng(seed); best=math.inf; best_epoch=0; stale=0; logs=[]; skips=0
    for epoch in range(1,epochs+1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('registered budget reached at epoch boundary')
        model.train(); order=order_rng.permutation(len(fit['q'])); losses=[]
        for start_row in range(0,len(order),32):
            ix=order[start_row:start_row+32]
            if not fit['trainable'][ix].any(): skips+=1; continue
            opt.zero_grad(set_to_none=True)
            w=model(fit['tokens'][ix],fit['mask'][ix],fit['support'][ix])
            prior=(w[...,None]*bank[fit['indices'][ix]]).sum(1)
            loss=(((prior-fit['truth'][ix])**2).mean(1)*fit['weights'][ix]).mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        val_mse=None
        if early:
            model.eval()
            with torch.no_grad():
                numer=0.; denom=0.
                for b in range(0,len(valid['q']),32):
                    ix=slice(b,b+32); w=model(valid['tokens'][ix],valid['mask'][ix],valid['support'][ix])
                    prior=(w[...,None]*bank[valid['indices'][ix]]).sum(1)
                    numer+=float((((prior-valid['truth'][ix])**2).mean(1)*valid['weights'][ix]).sum())
                    denom+=float(valid['weights'][ix].sum())
                val_mse=numer/denom
            if val_mse<best: best=val_mse; best_epoch=epoch; stale=0
            else: stale+=1
        logs.append({'epoch':epoch,'train_mse':float(np.mean(losses)) if losses else np.nan,'validation_mse':val_mse,'skipped_batches':skips})
        if early and stale>=10: break
    torch.cuda.synchronize(); elapsed=time.monotonic()-start
    stage_dir.mkdir(parents=True,exist_ok=True)
    atomic_csv(stage_dir/'EPOCHS.csv',pd.DataFrame(logs))
    audit={'seed':seed,'collective':collective,'selected_epoch':best_epoch if early else epochs,'elapsed_seconds':elapsed,
        'n_fit_queries':len(fit['q']),'n_fit_genes':domain.tasks.iloc[fit['q']].gene.nunique(),
        'fit_query_ids_hash':fingerprint(domain.tasks.iloc[fit['q']].task_id),
        'pca_unique_history_ids_hash':fingerprint(domain.memory.iloc[prep['history_rows']].experiment_id),
        'pca_history_rows':len(prep['history_rows']),'pca_components':prep['codes'].shape[1],
        'parameters':sum(p.numel() for p in model.parameters()),
        'effective_parameters':sum(p.numel() for p in model.parameters())-(0 if collective else 64*64),
        'skipped_zero_choice_batches':skips,
        'validation_genes_disjoint': bool(not set(domain.tasks.iloc[fit['q']].gene)&set(domain.tasks.iloc[valid['q']].gene)) if early else None}
    if early and not audit['validation_genes_disjoint']: raise RuntimeError('internal validation leaked genes')
    atomic_json(stage_dir/'FIT_AUDIT.json',audit)
    if not early:
        import joblib
        joblib.dump({k:v for k,v in prep.items() if k!='codes'},stage_dir/'PREPROCESSOR.joblib')
        torch.save({'state_dict':model.state_dict(),'n_features':len(prep['median'])*2+prep['codes'].shape[1],'collective':collective,'audit':audit},stage_dir/'MODEL.pt')
    del bank,fit,valid
    return model,prep,audit


def reuse_model(domain, groups, path, device):
    import joblib
    saved=torch.load(path/'MODEL.pt',map_location=device,weights_only=False)
    expected=fingerprint(domain.tasks.iloc[[g['q'] for g in groups if len(g['ix'])]].task_id)
    if saved['audit']['fit_query_ids_hash'] != expected: raise ValueError('reuse fitting IDs changed')
    prep=joblib.load(path/'PREPROCESSOR.joblib')
    actual_rows=np.unique(np.concatenate([g['ix'] for g in groups if len(g['ix'])]))
    if not np.array_equal(actual_rows,prep['history_rows']): raise ValueError('reuse history rows changed')
    prep['codes']=prep['pca'].transform(domain.effects).astype(np.float32)/prep['pc_scale']
    model=SetScorer(saved['n_features'],saved['collective']).to(device); model.load_state_dict(saved['state_dict'])
    return model,prep,saved['audit']


def predict_weights(model, prep, domain, groups, device):
    packed=pack(domain,groups,prep,device); model.eval(); arrays=[]
    with torch.no_grad():
        for b in range(0,len(packed['q']),32):
            ix=slice(b,b+32)
            arrays.append(model(packed['tokens'][ix],packed['mask'][ix],packed['support'][ix]).cpu().numpy())
    result={}
    for g,w in zip(packed['groups'],np.concatenate(arrays)):
        result[g['q']]=w[:len(g['ix'])].astype(float)
    return result


def cpu_baseline(args):
    from sklearn.ensemble import HistGradientBoostingRegressor
    z=np.load(args.input,allow_pickle=False); x=z['fit_x']; q=z['query_x']; target=z['fit_y']
    missing=~np.isfinite(x); qm=~np.isfinite(q)
    med=np.asarray([np.median(x[np.isfinite(x[:,j]),j]) if np.isfinite(x[:,j]).any() else 0. for j in range(x.shape[1])])
    x=np.where(missing,med,x); q=np.where(qm,med,q); center=x.mean(0); scale=x.std(0); scale[scale<1e-8]=1
    model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_depth=3,min_samples_leaf=20,l2_regularization=10.,random_state=20260930)
    model.fit(np.c_[(x-center)/scale,missing],target)
    np.save(args.output,model.predict(np.c_[(q-center)/scale,qm]))
    import sklearn
    print(json.dumps({'baseline_cpu_version':sklearn.__version__,'fit_pair_rows':len(x),'query_pair_rows':len(q)}),flush=True)


def hgb_weights(domain, fit_groups, query_groups, path, cpu_python):
    fit=[g for g in fit_groups if len(g['ix'])]; query=[g for g in query_groups if len(g['ix'])]
    x=np.concatenate([g['x'] for g in fit]); q=np.concatenate([g['x'] for g in query])
    y=np.concatenate([np.sqrt(np.mean((domain.effects[g['ix']]-domain.truth[g['q']])**2,axis=1)) for g in fit])
    path.mkdir(parents=True,exist_ok=True); np.savez(path/'HGB_INPUT.npz',fit_x=x,query_x=q,fit_y=y)
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    # Original CPU sklearn/numpy live in the registered user site. GPU torch
    # remains isolated; the independent CPU process restores its own site.
    env.pop('PYTHONNOUSERSITE',None)
    subprocess.run([cpu_python,str(Path(__file__).resolve()),'--phase','cpu-baseline','--input',str(path/'HGB_INPUT.npz'),'--output',str(path/'SCORES.npy')],env=env,check=True)
    scores=np.load(path/'SCORES.npy'); result={}; offset=0
    for g in query:
        s=scores[offset:offset+len(g['ix'])]; offset+=len(s)
        a=np.exp(np.clip(-(s-s.min())/max(float(s.std()),1e-8),-20,20)); a/=a.sum()
        result[g['q']]=.5*a+.5*g['s']
    atomic_json(path/'FIT_AUDIT.json',{'fit_pair_rows':len(x),'fit_query_rows':len(fit),'fit_genes':domain.tasks.iloc[[g['q'] for g in fit]].gene.nunique(),
        'fit_query_ids_hash':fingerprint(domain.tasks.iloc[[g['q'] for g in fit]].task_id),'cpu_python':cpu_python,
        'label':'individual-history transfer RMSE; original HGB objective','weighting':'original pair-row weighting; neural objective is gene-balanced query reconstruction'})
    return result


def score_records(domain, groups, weights, builder, seed, outer):
    records=[]; prior_vectors=[]; prior_ids=[]; weight_records=[]
    for g in groups:
        q=g['q']; task=domain.tasks.iloc[q]
        if not len(g['ix']): continue
        w=weights[q]
        if (w < -1e-7).any() or not np.isclose(w.sum(),1,atol=1e-6): raise ValueError('invalid mixture')
        h=domain.effects[g['ix']]; mu=np.sum(w[:,None]*h,0)
        variance=float(w@np.mean((h-mu)**2,axis=1)); bio=float(np.sqrt(np.mean((mu-domain.truth[q])**2)))
        prior_vectors.append(mu); prior_ids.append(str(task.task_id))
        for memory_i,wi in zip(g['ix'],w): weight_records.append({'task_id':task.task_id,'memory_id':domain.memory.iloc[memory_i].experiment_id,'weight':float(wi)})
        for upstream,pred in domain.predictions.items():
            p=pred[q]; distance=float(np.sqrt(np.mean((p-mu)**2)))
            direct_mean=float(w@np.mean((p-h)**2,axis=1))
            if not np.isclose(distance**2+variance,direct_mean,rtol=2e-5,atol=1e-7): raise ValueError('distance identity failed')
            records.append({'domain':domain.name,'upstream':upstream,'builder':builder,'seed':seed,'fold':outer,'task_id':task.task_id,
                'gene':task.gene,'context':task.context,'n_history':len(w),'bio_rmse':bio,'bio_cosine':cosine(mu,domain.truth[q]),
                'risk':math.sqrt(max(0,direct_mean)),'direct_risk':distance,'true_error_rmse':float(np.sqrt(np.mean((p-domain.truth[q])**2))),
                'magnitude_risk':float(np.sqrt(np.mean(p**2))),'support_risk':-math.log1p(g['support']),
                'prior_uncertainty':math.sqrt(max(0,variance)),'history_conflict':g['conflict'],'effective_sources':float(1/(w@w))})
    return records,prior_ids,np.asarray(prior_vectors),weight_records


def metric_values(frame, score_col='risk'):
    ids=frame.task_id.astype(str).to_numpy(); score=frame[score_col].to_numpy(float); error=frame.true_error_rmse.to_numpy(float)
    finite=np.isfinite(score)&np.isfinite(error); ids=ids[finite]; score=score[finite]; error=error[finite]; n=len(error)
    result={'n_tasks':n,'n_clusters':frame.loc[finite,'gene'].nunique()}
    if not n: return result|{'utility20':np.nan,'spearman':np.nan,'aurc':np.nan}
    k=math.ceil(.2*n); descending=np.lexsort((ids,-score)); oracle=np.lexsort((ids,-error)); ascending=np.lexsort((ids,score))
    denom=float(error[oracle[:k]].mean()-error.mean())
    result.update(utility20=float((error[descending[:k]].mean()-error.mean())/denom) if n>=20 and denom>1e-12 else np.nan,
        spearman=float(spearmanr(score,error).statistic) if n>=3 and np.ptp(score)>1e-15 and np.ptp(error)>1e-15 else np.nan,
        aurc=float(np.mean(np.cumsum(error[ascending])/np.arange(1,n+1))),
        high_risk_miss_rate=float(1-len(set(descending[:k])&set(oracle[:k]))/k))
    for coverage in [.1,.2,.5]: result[f'error_at_{int(coverage*100)}']=float(error[ascending[:max(1,math.ceil(coverage*n))]].mean())
    return result


def summarize(records):
    frame=pd.DataFrame(records); rows=[]
    for key,g in frame.groupby(['domain','upstream','builder','seed','fold','context'],sort=True):
        rows.append(dict(zip(['domain','upstream','builder','seed','fold','context'],key))|metric_values(g)|{'bio_rmse':float(g.bio_rmse.mean()),'bio_cosine':float(g.bio_cosine.mean())})
    contexts=pd.DataFrame(rows); macro=[]
    for key,g in contexts.groupby(['domain','upstream','builder','seed','fold'],sort=True):
        valid=g[g.utility20.notna()]
        macro.append(dict(zip(['domain','upstream','builder','seed','fold'],key))|{'valid_strata':len(valid),'planned_strata':len(g),
            **{c:float(valid[c].mean()) if len(valid) else np.nan for c in ['utility20','spearman','aurc','bio_rmse','bio_cosine']}})
    return frame,contexts,pd.DataFrame(macro)


def oracle_diagnostic(domain, groups):
    rows=[]
    for g in groups:
        if not len(g['ix']): continue
        h=domain.effects[g['ix']].astype(float); y=domain.truth[g['q']].astype(float); s=g['s']; k=len(s)
        gram=h@h.T/h.shape[1]; target=h@y/h.shape[1]; constant=float(y@y/h.shape[1])
        def obj(a):
            w=.5*a+.5*s
            return float(w@gram@w-2*w@target+constant)
        def jac(a): return gram@(.5*a+.5*s)-target
        if k==1: a=np.ones(1); success=True; msg='single history'
        else:
            r=minimize(obj,s,method='SLSQP',jac=jac,bounds=[(0.,1.)]*k,
                constraints=[{'type':'eq','fun':lambda a: a.sum()-1,'jac':lambda a:np.ones_like(a)}],
                options={'ftol':1e-10,'maxiter':500})
            a=r.x; success=bool(r.success and abs(a.sum()-1)<1e-6 and a.min()>=-1e-7); msg=r.message
        baseline=float(np.mean((s@h-y)**2))
        rows.append({'domain':domain.name,'task_id':domain.tasks.iloc[g['q']].task_id,'gene':domain.tasks.iloc[g['q']].gene,
            'n_history':k,'support_mean_mse':baseline,'constrained_best_mse':obj(a) if success else np.nan,
            'solver_success':success,'solver_message':str(msg),'constraint':'w=0.5*a+0.5*support; a simplex',
            'diagnostic_only':True,'risk_oracle_claim':False})
    return rows


def run(args):
    if not torch.cuda.is_available(): raise RuntimeError('use verified txpert Torch2.6 environment')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    runtime=RUNTIME/args.run_id; out=OUT/args.run_id
    runtime.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    if (out/'RUN_STATUS.json').exists(): raise FileExistsError('run is immutable; choose a new run-id')
    started=time.monotonic(); deadline=started+args.gpu_hours_cap*3600; device=torch.device(f'cuda:{args.gpu}')
    manifest={'run_id':args.run_id,'phase':args.phase,'pid':os.getpid(),'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'role':'DEV/SEEN_NEW_PUBLICSET','domains':args.domains,'outer_folds':args.folds,'seeds':args.seeds,'code_sha256':file_hash(__file__),
        'new_upstream_training':0,'new_test_truth_reads':0,'gpu_hours_cap':args.gpu_hours_cap,
        'torch':torch.__version__,'source_history_scenario':'rotating context deployment; other-context effects known; shared fits use disjoint genes',
        'MC_history_scenario':'all outer evaluation experiments forbidden globally; inner validation excluded likewise',
        'MC_estimand':'cell-weighted bank; fixed registered query truth; old guide bank retained as data-alignment control',
        'config':{'pca':32,'hidden':64,'support_mix':.5,'lr':.001,'weight_decay':.0001,'batch':32,'max_epochs':100,'patience':10},
        'status':'RUNNING','reuse_run':args.reuse_run,'next_action':'complete fixed prototype then full development by budget'}
    input_paths=[COMMON/'GENE_IDS.json',COMMON/'SOURCE_TASKS.csv',COMMON/'VALIDATION_BIOLOGY_TASKS.csv',
        COMMON/'SOURCE_PUBLIC_EFFECTS.npy',COMMON/'SOURCE_TRUE_EFFECTS.npy',COMMON/'VALIDATION_TRUE_EFFECTS.npy',
        COMMON/'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy',SOURCE_BANK/'manifest.json',
        COMMON/'public_mcfaline_trainval/manifest.json']
    manifest['input_bindings']=[{'path':str(p),'sha256':file_hash(p)} for p in input_paths]
    atomic_json(out/'RUN_MANIFEST.json',manifest)
    records=[]; fits=[]; reuses=[]; history_audit=[]; oracles=[]
    try:
        for name in args.domains:
            domain=load_domain(name)
            atomic_csv(out/f'{name}_QUERY_SPLIT.csv',domain.tasks)
            for outer in args.folds:
                train_rows=np.flatnonzero(domain.tasks.fold.to_numpy()!=outer)
                query_rows=np.flatnonzero(domain.tasks.fold.to_numpy()==outer)
                if args.phase=='prototype' and name=='Source': query_rows=query_rows[domain.tasks.iloc[query_rows].context.eq('K562').to_numpy()]
                if set(domain.tasks.iloc[train_rows].gene)&set(domain.tasks.iloc[query_rows].gene): raise ValueError('outer gene leak')
                forbidden=domain.forbidden(query_rows)
                train_groups=domain.groups(train_rows,forbidden); query_groups=domain.groups(query_rows,forbidden)
                fit_history=np.unique(np.concatenate([g['ix'] for g in train_groups if len(g['ix'])]))
                if set(domain.memory.iloc[fit_history].perturbation_target)&set(domain.tasks.iloc[query_rows].gene): raise ValueError('query gene entered PCA/fit history')
                for g in query_groups:
                    task=domain.tasks.iloc[g['q']]
                    history_audit.append({'domain':name,'outer_fold':outer,'task_id':task.task_id,'gene':task.gene,'context':task.context,'n_history':len(g['ix']),
                        'query_experiment_excluded':('E201::'+str(task.context)+'::'+str(task.condition) if name=='Source' else 'McFaline23::'+str(task.task_id)) not in set(domain.memory.iloc[g['ix']].experiment_id),
                        'inference_scope':f'outer{outer}/context{task.context}' if name=='Source' else f'outer{outer}',
                        'forbidden_evaluation_ids_hash':fingerprint(forbidden) if name!='Source' else fingerprint('E201::'+str(t.context)+'::'+str(t.condition) for t in domain.tasks.iloc[query_rows].itertuples() if t.context==task.context)})
                atomic_csv(out/'HISTORY_SET_AUDIT.csv',pd.DataFrame(history_audit))
                oracles.extend(oracle_diagnostic(domain,query_groups)); atomic_csv(out/'CONSTRAINED_RECONSTRUCTION_DIAGNOSTIC.csv',pd.DataFrame(oracles))
                manual={g['q']:g['s'] for g in query_groups if len(g['ix'])}
                hgb=hgb_weights(domain,train_groups,query_groups,runtime/f'{name}/outer{outer}/B1_HGB',args.cpu_python)
                for builder,weights in [('B0_SupportMean',manual),('B1_HGB',hgb)]:
                    rows,ids,priors,ws=score_records(domain,query_groups,weights,builder,0,outer); records.extend(rows)
                    np.savez(runtime/f'{name}/outer{outer}/{builder}_PRIORS.npz',task_ids=np.asarray(ids),priors=priors)
                    atomic_csv(runtime/f'{name}/outer{outer}/{builder}_WEIGHTS.csv',pd.DataFrame(ws))
                    if builder=='B0_SupportMean':
                        for label,col in [('Magnitude','magnitude_risk'),('NegativeHistorySupport','support_risk')]:
                            records.extend([{**r,'builder':label,'risk':r[col],'direct_risk':r[col]} for r in rows])
                if domain.old_effects is not None:
                    aligned=domain.effects; domain.effects=domain.old_effects
                    old_groups=domain.groups(query_rows,forbidden)
                    rows,_,_,_=score_records(domain,old_groups,{g['q']:g['s'] for g in old_groups if len(g['ix'])},'AlignmentControl_OldGuideMean',0,outer)
                    records.extend(rows); domain.effects=aligned
                genes=sorted(domain.tasks.iloc[train_rows].gene.unique(),key=lambda g:hashlib.sha256(f'PublicSet-inner-v1|{name}|{outer}|{g}'.encode()).hexdigest())
                val_genes=set(genes[:max(1,math.ceil(.2*len(genes)))])
                val_rows=np.asarray([i for i in train_rows if domain.tasks.iloc[i].gene in val_genes])
                inner_rows=np.asarray([i for i in train_rows if domain.tasks.iloc[i].gene not in val_genes])
                inner_forbidden=forbidden|domain.forbidden(val_rows)
                inner_groups=domain.groups(inner_rows,inner_forbidden); val_groups=domain.groups(val_rows,inner_forbidden)
                for builder,collective in [('B2_Pointwise',False),('B3_DeepSets',True)]:
                    for seed in args.seeds:
                        if (time.monotonic()-started)/3600>args.gpu_hours_cap: raise TimeoutError('registered GPU wall-time budget reached')
                        base=runtime/f'{name}/outer{outer}/{builder}/seed{seed}'
                        reuse=None
                        for candidate_run in args.reuse_run:
                            candidate_base=RUNTIME/candidate_run/f'{name}/outer{outer}/{builder}/seed{seed}'
                            candidate=candidate_base/'refit'
                            if (candidate/'MODEL.pt').exists():
                                reuse=candidate; break
                            link=candidate_base/'REUSED_MODEL.json'
                            if link.exists():
                                candidate=Path(json.loads(link.read_text())['model_path'])
                                if (candidate/'MODEL.pt').exists(): reuse=candidate; break
                        if reuse is not None:
                            model,prep,audit=reuse_model(domain,train_groups,reuse,device)
                            reuses.append({'domain':name,'fold':outer,'builder':builder,'seed':seed,'model_path':str(reuse),'new_fits':0})
                            base.mkdir(parents=True,exist_ok=True); atomic_json(base/'REUSED_MODEL.json',reuses[-1])
                        else:
                            early_audit=None
                            expected_inner=fingerprint(domain.tasks.iloc[[g['q'] for g in inner_groups if len(g['ix'])]].task_id)
                            for candidate_run in args.reuse_run:
                                candidate=RUNTIME/candidate_run/f'{name}/outer{outer}/{builder}/seed{seed}/early/FIT_AUDIT.json'
                                if candidate.exists():
                                    value=json.loads(candidate.read_text())
                                    if value['fit_query_ids_hash']==expected_inner:
                                        early_audit=value
                                        reuses.append({'domain':name,'fold':outer,'builder':builder,'seed':seed,'stage':'early_selection','model_path':str(candidate),'new_fits':0})
                                        break
                            if early_audit is None:
                                _,_,early_audit=training_run(domain,inner_groups,val_groups,collective,seed,100,device,True,base/'early',deadline)
                                fits.append({'domain':name,'fold':outer,'builder':builder,'stage':'early',**early_audit})
                                atomic_csv(out/'FIT_LEDGER.csv',pd.DataFrame(fits))
                            model,prep,audit=training_run(domain,train_groups,None,collective,seed,early_audit['selected_epoch'],device,False,base/'refit',deadline)
                            fits.append({'domain':name,'fold':outer,'builder':builder,'stage':'refit',**audit})
                        weights=predict_weights(model,prep,domain,query_groups,device)
                        rows,ids,priors,ws=score_records(domain,query_groups,weights,builder,seed,outer); records.extend(rows)
                        np.savez(base/'PRIORS.npz',task_ids=np.asarray(ids),priors=priors)
                        atomic_csv(base/'WEIGHTS.csv',pd.DataFrame(ws)); atomic_csv(out/'FIT_LEDGER.csv',pd.DataFrame(fits))
                        frame,contexts,macro=summarize(records)
                        atomic_csv(out/'TASK_PREDICTIONS.csv.gz',frame); atomic_csv(out/'CONTEXT_RESULTS.csv',contexts); atomic_csv(out/'MACRO_RESULTS.csv',macro)
                        print(json.dumps({'completed':f'{name}/fold{outer}/{builder}/seed{seed}','selected_epoch':audit['selected_epoch'],
                            'fit_seconds':round(audit['elapsed_seconds'],2),'macro':macro.tail(3).to_dict('records')},default=str),flush=True)
                        del model,prep; torch.cuda.empty_cache()
        frame,contexts,macro=summarize(records)
        atomic_csv(out/'TASK_PREDICTIONS.csv.gz',frame); atomic_csv(out/'CONTEXT_RESULTS.csv',contexts); atomic_csv(out/'MACRO_RESULTS.csv',macro)
        manifest.update(status='COMPLETE',elapsed_seconds=time.monotonic()-started,completed_neural_fit_stages=len(fits),
            neural_fit_seconds=float(sum(f['elapsed_seconds'] for f in fits)),reused_neural_models=len(reuses),
            next_action='paired statistics and fixed nested HGB follow-up; prototype does not select architecture')
        atomic_csv(out/'MODEL_REUSE_LEDGER.csv',pd.DataFrame(reuses))
        atomic_json(out/'RUN_STATUS.json',manifest)
    except Exception as error:
        manifest.update(status='FAILED',error=repr(error),elapsed_seconds=time.monotonic()-started,next_action='diagnose and recover affected stage without overwriting original attempt')
        atomic_json(out/'RUN_STATUS.json',manifest); raise
    finally:
        atomic_json(out/'RUN_MANIFEST.json',manifest)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prototype','full','cpu-baseline','semantic-tests'],default='prototype')
    p.add_argument('--run-id',default='prototype_v1'); p.add_argument('--domains',nargs='+',choices=['Source','McFaline'],default=['McFaline','Source'])
    p.add_argument('--folds',nargs='+',type=int,default=[0]); p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    p.add_argument('--gpu',type=int,default=0); p.add_argument('--gpu-hours-cap',type=float,default=4.)
    p.add_argument('--cpu-python',default='/home/miniconda/bin/python'); p.add_argument('--input',type=Path); p.add_argument('--output',type=Path)
    p.add_argument('--reuse-run',action='append',default=[])
    return p.parse_args()


def semantic_tests():
    torch.manual_seed(20260930); model=SetScorer(7,True)
    x=torch.randn(3,4,7); mask=torch.tensor([[1,1,1,0],[1,1,0,0],[1,0,0,0]],dtype=torch.bool)
    support=mask.float()/mask.sum(1,keepdim=True); w=model(x,mask,support)
    assert torch.allclose(w.sum(1),torch.ones(3)) and torch.all(w>=0)
    order=torch.tensor([2,0,3,1]); reordered=model(x[:,order],mask[:,order],support[:,order])
    assert torch.allclose(reordered[:,torch.argsort(order)],w,atol=1e-7)
    xp=x.clone(); xp[~mask]=1000; assert torch.allclose(model(xp,mask,support),w,atol=1e-7)
    assert w[2,0]==1 and torch.all(w[2,1:]==0)
    # A non-linear head must make relative scores sensitive to context.
    # Check Jacobian at multiple synthetic inputs rather than a single coincidence.
    active=False
    for trial in range(10):
        u=torch.randn(1,3,64); c=torch.randn(1,64,requires_grad=True)
        score=model.head(torch.cat([u,c[:,None].expand_as(u)],-1)).squeeze(-1)
        derivative=torch.autograd.grad(score[0,0]-score[0,1],c)[0]
        active=active or bool(derivative.abs().max()>1e-8)
    assert active,'set context cancels from relative scores'
    h=np.random.default_rng(1).normal(size=(3,10)); p=np.ones(10); weights=np.array([.2,.3,.5]); mu=weights@h
    assert np.isclose(np.mean((p-mu)**2)+weights@np.mean((h-mu)**2,axis=1),weights@np.mean((p-h)**2,axis=1))
    print(json.dumps({'semantic_tests':'PASS','order_invariance':True,'padding_invariance':True,'single_history':True,'nonlinear_context_effect':active}))


if __name__=='__main__':
    args=parse_args()
    if args.phase=='cpu-baseline': cpu_baseline(args)
    elif args.phase=='semantic-tests': semantic_tests()
    else: run(args)
