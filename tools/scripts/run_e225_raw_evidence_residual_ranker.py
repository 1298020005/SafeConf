#!/usr/bin/env python3
"""Matched-budget conditional residual ranking, using no target-calibrated score."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
ERROR = 'error_two_predictor_mean_rmse'
KEYS = ['dataset', 'fold_id', 'setting', 'train_fraction']
SETTINGS = ('context_unseen_row', 'perturbation_unseen_column',
            'random_missing_pair', 'context_and_perturbation_unseen')
INPUT_HASH = 'c84cb0f2b8c36c27b33d62cfbad7e98d2228288a85937e18351ab13d069d7ba0'
FEATURES = ['m', 'd', 'cn', 'scarcity', 'setting']
ARMS = ('M', 'MD', 'MDH', 'MDH_condition')
SEED = 20260922
CANDIDATES = [(0., 100.)] + [(lam, alpha) for lam in (.25, .5, 1.) for alpha in (100., 10., 1.)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(raw):
    required = KEYS + [ERROR, 'task_instance_id', 'baseline_predicted_magnitude',
                       'model_disagreement_rmse', 'context_novelty_scaled', 'training_support_count']
    frame = raw[required].copy().sort_values(KEYS + ['task_instance_id'], kind='stable').reset_index(drop=True)
    for src, dst in [('baseline_predicted_magnitude', 'm'), ('model_disagreement_rmse', 'd')]:
        frame[dst] = frame.groupby(KEYS, observed=True)[src].rank(method='average', pct=True)
    frame['cn'] = frame.context_novelty_scaled / 5.
    frame['scarcity'] = 1. / (1. + frame.training_support_count)
    if not np.isfinite(frame[['m','d','cn','scarcity',ERROR]].to_numpy()).all():
        raise ValueError('Nonfinite allowed inputs or labels')
    if not set(frame.setting).issubset(SETTINGS):
        raise ValueError('Unknown scenario')
    return frame

def design(frame, arm):
    cols = ['m'] if arm == 'M' else ['m', 'd'] if arm == 'MD' else ['m','d','cn','scarcity']
    x = PolynomialFeatures(2, include_bias=False).fit_transform(frame[cols].to_numpy(float))
    if arm == 'MDH_condition':
        onehot = np.column_stack([(frame.setting == s).to_numpy(float) for s in SETTINGS])
        x = np.column_stack([x, onehot, (onehot[:,:,None] * x[:,None,:]).reshape(len(frame), -1)])
    return x

def train_arrays(train):
    target = train.groupby(KEYS, observed=True)[ERROR].rank(method='average', pct=True).to_numpy() - train.m.to_numpy()
    sizes = train.groupby(KEYS, observed=True).dataset.transform('size').to_numpy(float)
    blocks = train[KEYS].drop_duplicates().groupby('dataset').size()
    w = 1. / (sizes * train.dataset.map(blocks).to_numpy(float))
    w *= len(train) / w.sum()
    return target, w

def fit_correction(train, test_features, arm, alpha):
    # test_features comes from an explicit allowlist; no target labels or old score.
    y, weights = train_arrays(train)
    model = Ridge(alpha=alpha).fit(design(train[FEATURES], arm), y, sample_weight=weights)
    correction = model.predict(design(test_features[FEATURES], arm))
    record = {'arm': arm, 'alpha': alpha, 'intercept': float(model.intercept_),
              'coefficients': model.coef_.tolist(), 'n_parameters': len(model.coef_) + 1}
    return correction, record

def top_weights(score, fraction):
    n = len(score)
    k = max(1, int(np.ceil(fraction*n)))
    threshold = np.partition(np.asarray(score), n-k)[n-k]
    selected = np.asarray(score) > threshold
    ties = np.asarray(score) == threshold
    weights = selected.astype(float)
    weights[ties] = (k - selected.sum()) / ties.sum()
    return weights

def metrics(score, y, fraction):
    score, y = np.asarray(score, float), np.asarray(y, float)
    w = top_weights(score, fraction)
    oracle = top_weights(y, fraction)
    avg = y.mean()
    denom = np.dot(oracle, y) / oracle.sum() - avg
    selected = np.dot(w, y)
    remaining_n = len(y) - w.sum()
    return {'utility': float((selected/w.sum()-avg)/denom) if denom > 1e-15 else np.nan,
            'capture': float(selected/y.sum()) if y.sum() > 0 else np.nan,
            'remaining_relative_error': float((y.sum()-selected)/remaining_n/avg)
            if avg > 0 and remaining_n > 0 else np.nan}

def evaluate(frame, score, budgets=(.2,)):
    rows = []
    for key, idx in frame.groupby(KEYS, sort=True).indices.items():
        y, pred = frame.iloc[idx][ERROR].to_numpy(float), np.asarray(score)[idx]
        sy, sp = rankdata(y), rankdata(pred)
        rho = float(np.corrcoef(sy, sp)[0,1]) if np.std(sy)>0 and np.std(sp)>0 else np.nan
        for b in budgets:
            rows.append({**dict(zip(KEYS,key)), 'budget':b, 'n_tasks':len(idx),
                         'spearman':rho, **metrics(pred,y,b)})
    return pd.DataFrame(rows)

def select_inner(train):
    rows=[]
    for heldout in sorted(train.dataset.unique()):
        fit=train.loc[train.dataset.ne(heldout)].reset_index(drop=True)
        valid=train.loc[train.dataset.eq(heldout)].reset_index(drop=True)
        for arm in ARMS:
            corrections={a:fit_correction(fit,valid[FEATURES],arm,a)[0] for a in (1.,10.,100.)}
            for lam,alpha in CANDIDATES:
                pred=valid.m.to_numpy()+lam*corrections[alpha]
                values=evaluate(valid,pred)
                rows.append({'inner_study':heldout,'training_studies':'|'.join(sorted(fit.dataset.unique())),
                             'arm':arm,'lambda':lam,'alpha':alpha,'utility':float(values.utility.mean())})
    table=pd.DataFrame(rows)
    choices={}
    for arm in ARMS:
        a=table[table.arm.eq(arm)].groupby(['lambda','alpha']).utility.mean()
        choices[arm]=max(CANDIDATES,key=lambda c:(a.loc[c],-c[0],c[1]))
    return choices,table

def outer_job(args):
    frame,heldout,output=args
    with threadpool_limits(limits=1):
        train=frame.loc[frame.dataset.ne(heldout)].reset_index(drop=True)
        test=frame.loc[frame.dataset.eq(heldout)].reset_index(drop=True)
        choices,inner=select_inner(train)
        inner['outer_study']=heldout
        scores={'magnitude':test.m.to_numpy(),'disagreement':test.d.to_numpy()}
        chosen=[]
        for arm in ARMS:
            lam,alpha=choices[arm]
            corr,record=fit_correction(train,test[FEATURES],arm,alpha)
            scores[arm]=test.m.to_numpy()+lam*corr
            chosen.append({'outer_study':heldout,'arm':arm,'lambda':lam,'alpha':alpha,
                           'parameters':record,'training_studies':sorted(train.dataset.unique())})
        # Save predictions with task IDs and without target outcomes before evaluation.
        pred=test[KEYS+['task_instance_id']].copy()
        for name,values in scores.items(): pred[name]=values
        path=output/f'PREDICTIONS_{heldout}.csv'
        pred.to_csv(path,index=False)
        prediction_hash=sha(path)
        rows=[]
        for name,values in scores.items():
            table=evaluate(test,values,budgets=(.1,.2,.3))
            table['method']=name
            rows.append(table)
        return pd.concat(rows,ignore_index=True),inner,chosen,{'study':heldout,'file':path.name,'sha256':prediction_hash}

def summarize(fold):
    study=fold.groupby(['dataset','method','budget'])[['utility','capture','spearman','remaining_relative_error']].mean().reset_index()
    ids=sorted(study.dataset.unique())
    draws=np.random.default_rng(SEED).integers(0,len(ids),size=(10000,len(ids)))
    rows=[]
    for budget in (.1,.2,.3):
        for baseline in ('magnitude','M','MD','MDH'):
            b=study[(study.method==baseline)&(study.budget==budget)].set_index('dataset').loc[ids]
            for method in ('M','MD','MDH','MDH_condition'):
                a=study[(study.method==method)&(study.budget==budget)].set_index('dataset').loc[ids]
                for metric in ('utility','capture','spearman','remaining_relative_error'):
                    delta=(a[metric]-b[metric]).to_numpy()
                    boot=delta[draws].mean(axis=1)
                    rows.append({'method':method,'baseline':baseline,'budget':budget,'metric':metric,
                                 'delta':float(delta.mean()),'ci95_lower':float(np.quantile(boot,.025)),
                                 'ci95_upper':float(np.quantile(boot,.975)),
                                 'positive_studies':int((delta>0).sum()),'n_studies':len(ids)})
    return study,pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    args=parser.parse_args()
    if sha(args.input)!=INPUT_HASH: raise ValueError('Input fingerprint changed')
    if (args.output/'RUN_STATUS.json').exists(): raise ValueError('Refusing to replace an existing run')
    start=time.monotonic()
    frame=prepare(pd.read_csv(args.input))
    if len(frame)!=8196 or frame.dataset.nunique()!=4: raise ValueError('Population changed')
    args.output.mkdir(parents=True,exist_ok=True)
    tables=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(outer_job,[(frame,s,args.output) for s in sorted(frame.dataset.unique())]):
            tables.append(result)
            print('Completed',result[3]['study'],flush=True)
    fold=pd.concat([t[0] for t in tables],ignore_index=True)
    inner=pd.concat([t[1] for t in tables],ignore_index=True)
    choices=[c for t in tables for c in t[2]]
    study,intervals=summarize(fold)
    for name,table in [('BLOCK_RESULTS',fold),('INNER_VALIDATION',inner),('STUDY_RESULTS',study),('INTERVALS',intervals)]:
        table.to_csv(args.output/(name+'.csv'),index=False)
    (args.output/'MODEL_CHOICES.json').write_text(json.dumps(choices,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    primary=intervals[(intervals.method=='MDH_condition')&(intervals.budget==.2)&(intervals.metric=='utility')].set_index('baseline')
    passed=bool(primary.loc['magnitude','delta']>=.02 and primary.loc['magnitude','positive_studies']>=3
                and (primary.ci95_lower>0).all())
    status={'experiment':'E225_raw_evidence_residual_ranker','status':'COMPLETE',
            'development_gate':'PASS' if passed else 'NOT_SUPPORTED',
            'evidence_class':'retrospective_development_not_external_confirmation',
            'created_at':datetime.now().astimezone().isoformat(), 'input_sha256':sha(args.input),
            'script_sha256':sha(Path(__file__)),'n_records':len(frame),
            'n_unique_dataset_context_perturbation':1083,'n_studies':4,
            'outer_error_labels_used_for_fitting_or_selection':False,
            'target_calibrated_score_used_as_feature':False,'e208_rows_read':0,
            'all_studies_retained':True,'source_data_modified':False,'workers':args.workers,
            'fits':4*(3*4*3+4),'feature_allowlist':FEATURES,'prediction_files':[t[3] for t in tables],
            'elapsed_seconds':time.monotonic()-start,'primary':primary.reset_index().to_dict(orient='records')}
    if status['input_sha256']!=INPUT_HASH: raise ValueError('Input changed during execution')
    (args.output/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# E225：不使用目标校准分数的条件小模型','',
           '输入是原始可部署证据；训练对象是两个预测器RMSE的均值。新增评分器不读取外层研究的验证或测试误差。',
           '上游预测器原有验证协议保持不变，不宣称整个系统从未用过验证数据。','',
           '| 条件模型相对 | 20%效用差值 | 描述性95%区间 | 正向研究 |',
           '|---|---:|---|---:|']
    for baseline,r in primary.iterrows():
        lines.append(f'| {baseline} | {r.delta:+.5f} | [{r.ci95_lower:+.5f}, {r.ci95_upper:+.5f}] | {int(r.positive_studies)}/4 |')
    lines+=['',f'预定开发门：**{status["development_gate"]}**。',
            '', 'M只用幅度；MD增加分歧；MDH增加历史背景新颖度与支持稀缺；MDH_condition增加场景交互。',
            '同分边界采用等概率选择的精确期望，不使用CSV行顺序破同分。',
            '所有8196条重复评估记录保留，研究是自助重采样单位；不能把这些记录当作独立研究。',
            '4研究与反复开发限制外推；本轮不替代 E208，也没有验证更大历史库的因果收益。','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
