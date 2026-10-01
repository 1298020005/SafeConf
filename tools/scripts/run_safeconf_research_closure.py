#!/usr/bin/env python3
"""October closure: nested biological references and budget-audited risk transfer.

All assets used here are DEV/SEEN. The September confirmation remains immutable;
additional McFaline comparisons are explicitly post-confirmation analyses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual import PublicMemoryStore
from tools.safeconf_continual.contracts import FrozenErrorCDF
from tools.safeconf_continual.research import (
    P, PUBLIC, SEEDS, bootstrap_u20, budget_subset, fit_risk, ids_hash,
    metrics, rank_labels, shuffled_labels, summarize,
)
from tools.scripts import run_dual_memory_txpert_public_biology as tx
from tools.scripts import seal_mcfaline_dual_memory_risk as mc

RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
OUT = tx.STAGE / 'dual_memory_continual/research_closure_20261001'
MC = Path('/home/yyf/runtime_artifacts/safeconf_mcfaline23')
REFS = ('Uniform', 'Manual', 'Learned')
PREFIX = {'Manual': 'CellWeighted', 'Learned': 'LearnedHGBRegularized'}


def save_csv(name, frame):
    tx.atomic_csv(OUT / name, frame)


def annotate(frame, dataset, upstream, version, contract):
    frame = frame.copy().reset_index(drop=True)
    frame['dataset_id'] = dataset
    frame['upstream'] = upstream
    frame['model_version'] = version
    frame['output_contract_id'] = contract
    if 'gene' not in frame:
        frame['gene'] = frame.perturbation.astype(str)
    return frame


def read_tx():
    tasks = pd.read_csv(tx.TASK_PATH)
    tasks = tasks[tasks.analysis_stratum.eq('primary_ge30')].reset_index(drop=True)
    tasks['fold'] = tx.fold_assignment(tasks)
    rows = tasks.source_mean_delta_row.to_numpy(int)
    controls = np.asarray(np.load(tx.VECTOR_ROOT / 'e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy', mmap_mode='r')[rows], float)
    truth = np.asarray(np.load(tx.VECTOR_ROOT / 'e201/evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy', mmap_mode='r')[rows], float) - controls
    memory, effects, memory_controls, manifest = PublicMemoryStore(tx.STORE_ROOT).load()
    eligibility = pd.read_parquet(tx.STORE_ROOT / manifest['eligibility']['path'])
    pairs = tx.build_pairs(tasks, truth, controls, memory, effects, memory_controls, eligibility)
    frames, predictions = {}, {}
    for name, short, series in [('TxPert_GAT', 'txpert_gat', 'e201'), ('TxPert_Exphormer', 'txpert_exphormer', 'e205')]:
        frame = pd.read_csv(tx.STAGE / f'safeconf_v4_development/{short}/FEATURE_TABLE.csv.gz').set_index('task_id').loc[tasks.task_id].reset_index()
        pred = np.asarray(np.load(tx.VECTOR_ROOT / f'{series}/pretruth_vectors/{"E201" if series=="e201" else "E205"}_FAMILY_CENTROIDS.npy', mmap_mode='r')[rows], float)
        ctrl = np.asarray(np.load(tx.VECTOR_ROOT / f'{series}/pretruth_vectors/{"E201" if series=="e201" else "E205"}_CONTROL_CENTROIDS.npy', mmap_mode='r')[rows], float)
        pred = pred - ctrl
        recomputed = tx.rmse_rows(pred, truth)
        if np.max(np.abs(recomputed-frame.true_error_rmse)) > 1e-6:
            raise RuntimeError('prediction/error alignment changed')
        frame['fold'] = tasks.fold
        # Recompute universal evidence on the actual effect vector for each model.
        frame = mc.prediction_features(frame, pred)
        frames[name] = annotate(frame, 'TxPert_E201', name, f'{series}_official_frozen', 'E201_log1p_matched_batch_delta_v1')
        predictions[name] = pred
    return tasks, pairs, effects, frames, predictions, truth


def fit_public_scores(pairs, fit_mask, query_mask, scope, audits):
    fit, query = pairs.loc[fit_mask], pairs.loc[query_mask]
    fit_genes, query_genes = set(fit.gene), set(query.gene)
    if fit_genes & query_genes:
        raise RuntimeError('public supervision crossed biological clusters')
    score = tx.fit_predict_regressor(fit, query, tx.PAIR_FEATURES, fit.transfer_rmse.to_numpy(float), 'hgb')
    audits.append({'scope': scope, 'n_fit_pairs': len(fit), 'n_fit_clusters': len(fit_genes),
                   'n_query_pairs': len(query), 'fit_clusters_hash': ids_hash(fit_genes),
                   'query_clusters_hash': ids_hash(query_genes), 'disjoint': True})
    return score


def prior_arrays(tasks, pairs, effects, scores):
    priors = {r: np.full((len(tasks), effects.shape[1]), np.nan) for r in REFS}
    rows = []
    scored = pairs.assign(transfer_score=scores)
    for q, group in scored.groupby('task_row', sort=True):
        for name, mode in [('Uniform', 'uniform'), ('Manual', 'cells'), ('Learned', 'learned_regularized')]:
            prior, weights, uncertainty, effective = tx.prior_from_scores(group, effects, 'transfer_score', mode)
            priors[name][q] = prior
            rows.append({'task_id': tasks.iloc[q].task_id, 'reference': name,
                         'prior_uncertainty': uncertainty, 'effective_sources': effective,
                         'history_conflict': float(group.source_conflict.mean()),
                         'log_history_support': float(np.log1p(np.expm1(group.log_source_cells).sum()))})
    return priors, pd.DataFrame(rows)


def reference_frame(base, pred, prior, summary, reference):
    frame = base.copy()
    s = summary[summary.reference.eq(reference)].set_index('task_id')
    frame['prior_magnitude'] = np.sqrt(np.mean(prior**2, axis=1))
    frame['prediction_prior_rmse'] = tx.rmse_rows(pred, prior)
    frame['prediction_prior_cosine'] = tx.cosine_rows(pred, prior)
    for c in ('prior_uncertainty', 'effective_sources', 'history_conflict', 'log_history_support'):
        frame[c] = frame.task_id.map(s[c])
    return frame


def build_nested():
    tasks, pairs, effects, bases, predictions, truth = read_tx()
    tx.atomic_csv(RUNTIME / 'TX_TASK_SPLIT.csv', tasks[['task_id', 'gene', 'target', 'fold']])
    audits, biology = [], []
    oof_scores = np.full(len(pairs), np.nan)
    for outer in range(5):
        started = time.monotonic()
        train_pairs, test_pairs = pairs.fold.ne(outer), pairs.fold.eq(outer)
        scores = np.full(len(pairs), np.nan)
        scores[test_pairs] = fit_public_scores(pairs, train_pairs, test_pairs, f'outer{outer}/test', audits)
        oof_scores[test_pairs] = scores[test_pairs]
        train_tasks = tasks[tasks.fold.ne(outer)].copy()
        for inner, (fit_rows, query_rows) in enumerate(GroupKFold(4).split(train_tasks, groups=train_tasks.gene)):
            fit_ids = set(train_tasks.iloc[fit_rows].task_id)
            query_ids = set(train_tasks.iloc[query_rows].task_id)
            f, q = pairs.task_id.isin(fit_ids), pairs.task_id.isin(query_ids)
            if (pairs.loc[f | q, 'fold'] == outer).any():
                raise RuntimeError('outer test entered public inner OOF')
            scores[q] = fit_public_scores(pairs, f, q, f'outer{outer}/inner{inner}', audits)
        if not np.isfinite(scores).all():
            raise RuntimeError('incomplete nested public predictions')
        priors, summary = prior_arrays(tasks, pairs, effects, scores)
        for upstream, base in bases.items():
            for ref in REFS:
                frame = reference_frame(base, predictions[upstream], priors[ref], summary, ref)
                frame.to_parquet(RUNTIME / f'nested_{outer}_{upstream}_{ref}.parquet', index=False)
        for ref in REFS:
            query = tasks.fold.eq(outer).to_numpy()
            for i in np.flatnonzero(query):
                biology.append({'task_id': tasks.iloc[i].task_id, 'gene': tasks.iloc[i].gene,
                    'target': tasks.iloc[i].target, 'fold': outer, 'reference': ref,
                    'effect_rmse': float(tx.rmse_rows(priors[ref][i:i+1], truth[i:i+1])[0]),
                    'effect_cosine': float(tx.cosine_rows(priors[ref][i:i+1], truth[i:i+1])[0])})
        print(json.dumps({'phase':'nested_public','outer_fold':outer,'elapsed_seconds':round(time.monotonic()-started,2)}), flush=True)
    # Independent external risk training may use this entire source OOF cohort.
    priors, summary = prior_arrays(tasks, pairs, effects, oof_scores)
    for ref in REFS:
        full = pd.concat([reference_frame(base, predictions[name], priors[ref], summary, ref)
                          for name, base in bases.items()], ignore_index=True)
        full.to_parquet(RUNTIME / f'source_{ref}.parquet', index=False)
    save_csv('NESTED_ISOLATION_AUDIT.csv', pd.DataFrame(audits))
    save_csv('PUBLIC_BIOLOGY_OOF.csv.gz', pd.DataFrame(biology))
    save_csv('SPLIT_MANIFEST.csv', tasks[['task_id','gene','target','fold']])


def external_query():
    seal = MC / 'test_pretruth_20260930/risk_seal'
    aggregate = MC / 'test_pretruth_20260930/aggregated_predictions'
    frozen = pd.read_csv(seal / 'TEST_RISK_FEATURES.csv.gz')
    errors = pd.read_csv(MC / 'coldstart_confirmation_20260930/TEST_TASK_ERRORS.csv.gz').set_index('task_id')
    frozen['true_error_rmse'] = frozen.task_id.map(errors.true_error_rmse)
    frozen = annotate(frozen, 'McFaline23', 'DecoderOnly', 'decoder_validation_repair_v1', 'McFaline_log1p_delta_512_v1')
    memory_root = Path('/home/yyf/data/safeconf_dual_memory_20260929/public_mcfaline_trainval')
    memory = pd.read_parquet(memory_root / 'public_memory.parquet')
    effects = np.load(memory_root / 'effect_vectors.npy', mmap_mode='r')
    controls = np.load(memory_root / 'control_vectors.npy', mmap_mode='r')
    pred = np.load(aggregate / 'TEST_CALIBRATED_EFFECTS.npy')
    pairs = mc.build_pairs(frozen, np.load(aggregate / 'TEST_CONTROL_STATES.npy'), memory, effects, controls, None)
    frames = {}
    for ref, prefix in PREFIX.items():
        frame = frozen.copy()
        for col in PUBLIC:
            frame[col] = frame[f'{prefix}_{col}']
        frames[ref] = frame
    uniform = np.full((len(frozen), effects.shape[1]), np.nan)
    summary = []
    for q, group in pairs.groupby('task_row', sort=True):
        vectors = np.asarray(effects[group.memory_row.to_numpy(int)],float)
        mean = vectors.mean(axis=0)
        uniform[q] = mean
        summary.append({'task_id': frozen.iloc[q].task_id, 'reference':'Uniform',
            'prior_uncertainty':float(np.sqrt(np.mean((vectors-mean)**2))),
            'effective_sources':float(len(vectors)), 'history_conflict':float(group.source_conflict.mean()),
            'log_history_support':float(np.log1p(np.expm1(group.log_source_cells).sum()))})
    frames['Uniform'] = reference_frame(frozen, pred, uniform, pd.DataFrame(summary), 'Uniform')
    for ref, frame in frames.items():
        if frame[PUBLIC].isna().any().any():
            raise RuntimeError('no-history tasks require separate coverage evaluation')
        frame.to_parquet(RUNTIME / f'external_{ref}.parquet',index=False)
    return frames


def add_predictions(rows, frame, score, line, method, seed=SEEDS[0], **metadata):
    part = frame[['task_id','target','gene','fold','upstream','true_error_rmse']].copy()
    part['risk'] = np.asarray(score,float)
    part['line'], part['method'], part['seed'] = line, method, seed
    for k,v in metadata.items(): part[k] = v
    rows.append(part)


def run_matrix():
    queries_external = external_query()
    lines = [('GAT_to_Exphormer','TxPert_GAT','TxPert_Exphormer'),
             ('Exphormer_to_GAT','TxPert_Exphormer','TxPert_GAT'),
             ('TxPert_to_McFaline',None,'DecoderOnly')]
    records, cdf, null_audit, ledger = [], [], [], []
    for line, source, target in lines:
        folds = range(5) if source else [-1]
        for fold in folds:
            started = time.monotonic()
            fits, queries = {}, {}
            for ref in REFS:
                if source:
                    a = pd.read_parquet(RUNTIME / f'nested_{fold}_{source}_{ref}.parquet')
                    b = pd.read_parquet(RUNTIME / f'nested_{fold}_{target}_{ref}.parquet')
                    fits[ref] = a[a.fold.ne(fold)].reset_index(drop=True)
                    queries[ref] = b[b.fold.eq(fold)].reset_index(drop=True)
                    if set(fits[ref].gene) & set(queries[ref].gene):
                        raise RuntimeError('risk transfer crossed outer clusters')
                else:
                    fits[ref] = pd.read_parquet(RUNTIME / f'source_{ref}.parquet')
                    queries[ref] = queries_external[ref]
            labels, audit = rank_labels(fits['Manual'],f'{line}/outer{fold}')
            cdf.extend(audit)
            for seed in SEEDS:
                manual = queries['Manual']
                for method, score in [('Magnitude',manual.predicted_magnitude),
                        ('ManualPriorMagnitude',manual.prior_magnitude),
                        ('ManualPriorDispersion',manual.prior_uncertainty)]:
                    add_predictions(records,manual,score,line,method,seed)
                for kind in ('ridge','hgb'):
                    fit = fit_risk(fits['Manual'],labels,P,kind,seed)
                    add_predictions(records,manual,fit.predict(manual),line,f'Prediction_{kind}',seed)
                for ref in REFS:
                    train, query = fits[ref], queries[ref]
                    add_predictions(records,query,query.prediction_prior_rmse,line,f'{ref}_DirectRMSE',seed)
                    for kind in ('ridge','hgb'):
                        fit = fit_risk(train,labels,P+PUBLIC,kind,seed)
                        add_predictions(records,query,fit.predict(query),line,f'{ref}_{kind}',seed)
                    if ref!='Uniform':
                        distance = np.sqrt(query.prediction_prior_rmse**2+query.prior_uncertainty**2)
                        add_predictions(records,query,distance,line,f'{ref}_WeightedHistoryDistance',seed)
            for shuffle in range(5):
                null, audit = shuffled_labels(fits['Manual'],labels,SEEDS[0]+shuffle)
                null_audit.append({'line':line,'outer_fold':fold,**audit})
                for ref in ('Manual','Learned'):
                    fit = fit_risk(fits[ref],null,P+PUBLIC,'hgb',SEEDS[0])
                    add_predictions(records,queries[ref],fit.predict(queries[ref]),line,f'{ref}_ShuffledHGB_{shuffle}')
            ledger.append({'line':line,'outer_fold':fold,'source_error_rows':len(fits['Manual']),
                'source_error_clusters':fits['Manual'].gene.nunique(),'source_predictors':fits['Manual'].upstream.nunique(),
                'source_records_hash':ids_hash(fits['Manual'].upstream+'::'+fits['Manual'].task_id),
                'c_risk_error_labels':0,'c_feedback_error_labels':0,
                'c_validation_biological_tasks':542 if source is None else 0,
                'c_validation_upstream_calibration_tasks':542 if source is None else 0,
                'evaluation_role':'POST_CONFIRMATION_SEEN' if source is None else 'DEV_SEEN_NESTED',
                'new_large_upstream_training':False})
            print(json.dumps({'phase':'matrix','line':line,'fold':fold,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
            # Checkpoint completed fold results so interruption loses no experiments.
            save_csv('MATRIX_TASK_PREDICTIONS.csv.gz',pd.concat(records,ignore_index=True))
    predictions = pd.concat(records,ignore_index=True)
    strata, macro = summarize(predictions)
    save_csv('MATRIX_CONTEXT_RESULTS.csv',strata)
    save_csv('MATRIX_MACRO_RESULTS.csv',macro)
    fold_rows=[]
    for key,part in predictions.groupby(['line','method','seed','target','fold']):
        fold_rows.append(dict(zip(['line','method','seed','target','fold'],key))|metrics(part,part.risk.to_numpy()))
    save_csv('MATRIX_FOLD_RESULTS.csv',pd.DataFrame(fold_rows))
    save_csv('CDF_AUDIT.csv',pd.DataFrame(cdf))
    save_csv('SHUFFLE_AUDIT.csv',pd.DataFrame(null_audit))
    save_csv('INFORMATION_BUDGET_LEDGER.csv',pd.DataFrame(ledger))
    print(macro[macro.seed.eq(SEEDS[0])][['line','method','utility20','spearman','aurc']].to_string(index=False),flush=True)


def run_bootstrap(replicates):
    predictions=pd.read_csv(OUT / 'MATRIX_TASK_PREDICTIONS.csv.gz')
    primary=predictions[predictions.seed.eq(SEEDS[0])]
    rows=[]
    for line,part in primary.groupby('line'):
        meta=['task_id','target','gene','fold','upstream','true_error_rmse']
        wide=part.pivot(index=meta,columns='method',values='risk').reset_index()
        for ref in ('Manual','Learned'):
            for a,b in [(f'{ref}_WeightedHistoryDistance','Magnitude'),
                (f'{ref}_hgb',f'{ref}_WeightedHistoryDistance'),
                (f'{ref}_hgb',f'{ref}_DirectRMSE'),
                (f'{ref}_hgb','Prediction_hgb')]+[(f'{ref}_hgb',f'{ref}_ShuffledHGB_{s}') for s in range(5)]:
                result=bootstrap_u20(wide,wide[a].to_numpy(),wide[b].to_numpy(),replicates)
                rows.append({'line':line,'method_a':a,'method_b':b,**result})
        save_csv('PAIRED_CLUSTER_BOOTSTRAP.csv',pd.DataFrame(rows))
        print(json.dumps({'phase':'bootstrap','line':line,'comparisons':len(rows)}),flush=True)


def write_method_budget_ledger():
    predictions=pd.read_csv(OUT/'MATRIX_TASK_PREDICTIONS.csv.gz')
    cdf=pd.read_csv(OUT/'CDF_AUDIT.csv')
    rows=[]
    for (line,fold),part in predictions.groupby(['line','fold']):
        external=line=='TxPert_to_McFaline'
        source='TxPert_GAT' if line=='GAT_to_Exphormer' else 'TxPert_Exphormer'
        fit=pd.read_parquet(RUNTIME/'source_Manual.parquet') if external else pd.read_parquet(RUNTIME/f'nested_{fold}_{source}_Manual.parquet')
        if not external:fit=fit[fit.fold.ne(fold)]
        scope=f'{line}/outer{fold}'
        n_context=fit.target.nunique()
        use=cdf.run_id.eq(scope)
        cdf.loc[use,'n_planned_contexts']=4
        cdf.loc[use,'n_actual_contexts']=n_context
        cdf.loc[use,'context_coverage']=n_context/4
        for method in sorted(part.method.unique()):
            supervised=method.endswith(('_ridge','_hgb')) or '_ShuffledHGB_' in method
            public=method.startswith(('Uniform','Manual','Learned'))
            rows.append({'line':line,'outer_fold':fold,'method':method,'source_error_rows':len(fit) if supervised else 0,
                'source_error_clusters':fit.gene.nunique() if supervised else 0,
                'source_predictors':fit.upstream.nunique() if supervised else 0,
                'source_error_uses':'risk target and training-only CDF' if supervised else 'NONE',
                'source_error_records_hash':ids_hash(fit.upstream+'::'+fit.task_id) if supervised else '',
                'public_evidence_used':public,'public_biology_transfer_learner':method.startswith('Learned'),
                'c_risk_error_labels':0,'c_feedback_error_labels':0,'c_error_CDF_labels':0,
                'c_validation_biology_tasks':542 if external and method.startswith('Learned') else 0,
                'c_validation_upstream_calibration_tasks':542 if external else 0,
                'target_evaluation_role':'POST_CONFIRMATION_SEEN' if external else 'DEV_SEEN_NESTED'})
    save_csv('INFORMATION_BUDGET_LEDGER.csv',pd.DataFrame(rows))
    save_csv('CDF_AUDIT.csv',cdf)


def run_source_scaling():
    records, cdf, diversity = [], [], []
    for ref in ('Manual','Learned'):
        source=pd.read_parquet(RUNTIME/f'source_{ref}.parquet')
        query=pd.read_parquet(RUNTIME/f'external_{ref}.parquet')
        for order in range(5):
            for budget in (.10,.25,.50,.75,1.):
                fit=budget_subset(source,budget,order)
                labels,audit=rank_labels(fit,f'scaling/{ref}/{order}/{budget}',budget)
                cdf.extend(audit)
                for seed in SEEDS:
                    model=fit_risk(fit,labels,P+PUBLIC,'hgb',seed)
                    add_predictions(records,query,model.predict(query),'TxPert_to_McFaline',f'{ref}_hgb',seed,
                        budget=budget,order=order,n_source_clusters=fit.gene.nunique(),
                        n_source_rows=len(fit),n_source_predictors=fit.upstream.nunique())
            print(json.dumps({'phase':'source_scaling','reference':ref,'order':order}),flush=True)
        upstreams=sorted(source.upstream.unique())
        single_scores={}
        for name in upstreams:
            fit=source[source.upstream.eq(name)].reset_index(drop=True)
            labels,audit=rank_labels(fit,f'diversity/{ref}/{name}')
            cdf.extend(audit)
            score=fit_risk(fit,labels,P+PUBLIC,'hgb').predict(query)
            single_scores[name]=score
            add_predictions(diversity,query,score,'TxPert_to_McFaline',f'{ref}/{name}',
                n_source_clusters=fit.gene.nunique(),n_source_rows=len(fit),n_source_predictors=1)
        add_predictions(diversity,query,np.mean(list(single_scores.values()),axis=0),
            'TxPert_to_McFaline',f'{ref}/SeparateRiskAverage',n_source_clusters=source.gene.nunique(),
            n_source_rows=len(source),n_source_predictors=2)
        # One source per exact task: identical task cohort and record budget.
        selected=[]
        for task,part in source.groupby('task_id',sort=True):
            part=part.sort_values('upstream')
            choice=int(hashlib.sha256(f'source-diversity-v1|{task}'.encode()).hexdigest(),16)%len(part)
            selected.append(part.iloc[choice])
        equal=pd.DataFrame(selected).reset_index(drop=True)
        for name,fit in [('PooledEqualRecords',equal),('PooledFullClusterWeight',source)]:
            labels,audit=rank_labels(fit,f'diversity/{ref}/{name}')
            cdf.extend(audit)
            score=fit_risk(fit,labels,P+PUBLIC,'hgb').predict(query)
            add_predictions(diversity,query,score,'TxPert_to_McFaline',f'{ref}/{name}',
                n_source_clusters=fit.gene.nunique(),n_source_rows=len(fit),n_source_predictors=2)
    predictions=pd.concat(records,ignore_index=True)
    save_csv('SOURCE_SCALING_TASK_PREDICTIONS.csv.gz',predictions)
    strata,macro=summarize(predictions)
    save_csv('SOURCE_SCALING_STRATA.csv',strata);save_csv('SOURCE_SCALING_MACRO.csv',macro)
    if diversity:
        d=pd.concat(diversity,ignore_index=True);save_csv('SOURCE_DIVERSITY_PREDICTIONS.csv.gz',d)
        strata,macro=summarize(d)
        save_csv('SOURCE_DIVERSITY_STRATA.csv',strata);save_csv('SOURCE_DIVERSITY_MACRO.csv',macro)
    save_csv('SOURCE_SCALING_CDF_AUDIT.csv',pd.DataFrame(cdf))


def mapped_query_labels(fit,query):
    output=np.full(len(query),np.nan)
    from tools.safeconf_continual.research import CDF_KEYS
    for key,group in fit.groupby(CDF_KEYS):
        mask=np.ones(len(query),bool)
        for column,value in zip(CDF_KEYS,key): mask &= query[column].eq(value).to_numpy()
        if len(group)>=2:
            output[mask]=FrozenErrorCDF.fit(group.true_error_rmse).transform(query.loc[mask,'true_error_rmse'])
    return output


def feedback_kappa(feedback):
    losses={k:[] for k in (10,25,50,100)}
    for train_rows,val_rows in GroupKFold(min(3,feedback.gene.nunique())).split(feedback,groups=feedback.gene):
        fit=feedback.iloc[train_rows].reset_index(drop=True)
        query=feedback.iloc[val_rows].reset_index(drop=True)
        labels,_=rank_labels(fit,'feedback-inner-CDF')
        target=mapped_query_labels(fit,query)
        correction=fit_risk(fit,labels-fit.shared_risk.to_numpy(),P+PUBLIC,'hgb').predict(query,clip=False)
        for kappa in losses:
            final=np.clip(query.shared_risk.to_numpy()+fit.gene.nunique()/(fit.gene.nunique()+kappa)*correction,0,1)
            finite=np.isfinite(target)
            losses[kappa].append(float(np.mean(np.abs(final[finite]-target[finite]))))
    return min(losses,key=lambda k:(np.mean(losses[k]),-k))


def run_feedback():
    frame=pd.read_parquet(RUNTIME/'external_Learned.parquet')
    main=pd.read_csv(OUT/'MATRIX_TASK_PREDICTIONS.csv.gz')
    shared=main[main.line.eq('TxPert_to_McFaline')&main.method.eq('Learned_hgb')&main.seed.eq(SEEDS[0])].set_index('task_id').risk
    frame['shared_risk']=frame.task_id.map(shared)
    if frame.shared_risk.isna().any():raise RuntimeError('missing independent shared predictions')
    ordered=sorted(frame.gene.astype(str).unique(),key=lambda x:hashlib.sha256(f'SafeConf-McFaline-feedback-v1\0{x}'.encode()).hexdigest())
    n_feedback=int(np.ceil(.6*len(ordered))); pool=set(ordered[:n_feedback])
    train_pool=frame[frame.gene.isin(pool)].reset_index(drop=True)
    query=frame[~frame.gene.isin(pool)].reset_index(drop=True)
    assert not set(train_pool.gene)&set(query.gene)
    tx.atomic_json(OUT/'FEEDBACK_SPLIT_CONTRACT.json',{
        'contract':'SafeConf-McFaline-feedback-v1','n_feedback_clusters':n_feedback,
        'n_holdout_clusters':len(ordered)-n_feedback,'n_feedback_tasks':len(train_pool),
        'n_holdout_tasks':len(query),'feedback_cluster_order_hash':ids_hash(ordered[:n_feedback]),
        'holdout_clusters_hash':ids_hash(ordered[n_feedback:]),
        'target_CDF_uses_validation_errors':False,'role':'retrospective post-confirmation SEEN analysis',
        'shared_score':'October nested source supervision; original September shared score separately preserved'})
    records,cdf,ledger=[],[],[]
    # Save every budget prediction before aggregate evaluation.
    for budget in (0.,.10,.25,.50,.75,1.):
        for seed in SEEDS:
            add_predictions(records,query,query.shared_risk,'McFaline_feedback','Shared',seed,budget=budget)
        if budget==0:
            ledger.append({'budget':0,'method':'Shared','c_feedback_error_rows':0,'c_feedback_error_clusters':0,
                'c_validation_risk_error_rows':0,'target_CDF':'NONE','status':'target learners not fitted'})
            continue
        n=int(np.ceil(budget*n_feedback)); permitted=set(ordered[:n])
        fit=train_pool[train_pool.gene.isin(permitted)].reset_index(drop=True)
        labels,audit=rank_labels(fit,f'feedback/{budget}',budget);cdf.extend(audit)
        kappa=feedback_kappa(fit)
        for method,kind,columns in [('TargetOnly_Ridge','ridge',P),('TargetOnly_HGB','hgb',P),
            ('PublicTarget_HGB','hgb',P+PUBLIC),('SharedTarget_HGB','hgb',P+PUBLIC+['shared_risk'])]:
            for seed in SEEDS:
                model=fit_risk(fit,labels,columns,kind,seed)
                add_predictions(records,query,model.predict(query),'McFaline_feedback',method,seed,
                    budget=budget,n_feedback_clusters=n,n_feedback_rows=len(fit))
            ledger.append({'budget':budget,'method':method,'c_feedback_error_rows':len(fit),
                'c_feedback_error_clusters':n,'c_validation_risk_error_rows':0,'target_CDF':'budget-only context ECDF',
                'allowed_feedback_records_hash':ids_hash(fit.task_id),'public_features':bool(set(columns)&set(PUBLIC)),
                'source_error_score':method=='SharedTarget_HGB','c_validation_upstream_calibration_rows':542,
                'c_validation_biology_rows':542 if set(columns)&set(PUBLIC) else 0})
        for seed in SEEDS:
            adapter=fit_risk(fit,labels-fit.shared_risk.to_numpy(),P+PUBLIC,'hgb',seed)
            final=np.clip(query.shared_risk.to_numpy()+n/(n+kappa)*adapter.predict(query,clip=False),0,1)
            add_predictions(records,query,final,'McFaline_feedback','ResidualHGB',seed,budget=budget,
                n_feedback_clusters=n,n_feedback_rows=len(fit),kappa=kappa)
        ledger.append({'budget':budget,'method':'ResidualHGB','c_feedback_error_rows':len(fit),
            'c_feedback_error_clusters':n,'c_validation_risk_error_rows':0,'target_CDF':'budget-only with inner-refitted ECDF',
            'allowed_feedback_records_hash':ids_hash(fit.task_id),'kappa':kappa})
        print(json.dumps({'phase':'strict_feedback','budget':budget,'feedback_clusters':n,'kappa':kappa}),flush=True)
    predictions=pd.concat(records,ignore_index=True)
    save_csv('STRICT_FEEDBACK_TASK_PREDICTIONS.csv.gz',predictions)
    strata,macro=summarize(predictions)
    save_csv('STRICT_FEEDBACK_STRATA.csv',strata);save_csv('STRICT_FEEDBACK_MACRO.csv',macro)
    save_csv('STRICT_FEEDBACK_CDF_AUDIT.csv',pd.DataFrame(cdf))
    save_csv('STRICT_FEEDBACK_INFORMATION_LEDGER.csv',pd.DataFrame(ledger))
    print(macro[macro.seed.eq(SEEDS[0])][['budget','method','utility20','spearman','aurc']].to_string(index=False),flush=True)


def render_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def finish(fig,name,tight=True):
        fig.tight_layout()
        opts={'bbox_inches':'tight'} if tight else {}
        fig.savefig(figures/f'{name}.png',dpi=180,**opts)
        fig.savefig(figures/f'{name}.pdf',**opts)
        plt.close(fig)
    matrix=pd.read_csv(OUT/'MATRIX_MACRO_RESULTS.csv');matrix=matrix[matrix.seed.eq(SEEDS[0])]
    methods=['Magnitude','Prediction_hgb','Manual_WeightedHistoryDistance','Learned_WeightedHistoryDistance','Manual_hgb','Learned_hgb']
    names=['Magnitude','Prediction HGB','Manual history distance','Learned history distance','Manual + source HGB','Learned + source HGB']
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    for ax,(line,g) in zip(axes,matrix.groupby('line',sort=True)):
        values=g.set_index('method').loc[methods].utility20
        ax.barh(np.arange(len(methods)),values,color=['#777777','#999999','#e5b45e','#cc9342','#579bba','#326d94'])
        ax.set_yticks(np.arange(len(methods)),names);ax.invert_yaxis();ax.set_xlim(-.22,1.)
        ax.axvline(0,color='#bbbbbb',linewidth=.8);ax.set_title(line.replace('_',' '));ax.set_xlabel('Utility@20')
        for i,value in enumerate(values):ax.text(value+.01,i,f'{value:.3f}',va='center',fontsize=9)
    fig.suptitle('Same-task comparison; October nested DEV/SEEN analyses',y=1.02)
    finish(fig,'core_information_comparison')
    feedback=pd.read_csv(OUT/'STRICT_FEEDBACK_MACRO.csv');feedback=feedback[feedback.seed.eq(SEEDS[0])]
    fig,ax=plt.subplots(figsize=(7.6,4.6))
    for method,label in [('Shared','Shared'),('TargetOnly_HGB','Target-only HGB'),('PublicTarget_HGB','Public + target HGB'),('SharedTarget_HGB','Shared + target HGB')]:
        g=feedback[feedback.method.eq(method)].sort_values('budget')
        ax.plot(g.budget*100,g.utility20,'o-',label=label)
    ax.set_xlabel('Opened feedback pool (%)');ax.set_ylabel('Utility@20');ax.legend(frameon=False,loc='lower right')
    ax.set_title('Strict feedback-only CDF; permanent evaluation tasks')
    finish(fig,'four_information_feedback',False)
    scaling=pd.read_csv(OUT/'SOURCE_SCALING_MACRO.csv');scaling=scaling[scaling.seed.eq(SEEDS[0])]
    fig,ax=plt.subplots(figsize=(7.6,4.6))
    for method,label in [('Manual_hgb','Manual reference'),('Learned_hgb','Learned reference')]:
        g=scaling[scaling.method.eq(method)].groupby('budget').utility20.agg(['mean','min','max'])
        ax.plot(g.index*100,g['mean'],'o-',label=label)
        ax.fill_between(g.index*100,g['min'],g['max'],alpha=.15)
    ax.set_xlabel('Allowed independent source clusters (%)');ax.set_ylabel('Utility@20');ax.legend(frameon=False)
    ax.set_title('Source supervision scaling: mean and range of five hash orders')
    finish(fig,'source_supervision_scaling',False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['nested','matrix','statistics','scaling','feedback','ledger','figures','all'],default='all')
    parser.add_argument('--bootstrap',type=int,default=5000)
    args=parser.parse_args()
    RUNTIME.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    config={'seeds':SEEDS,'public_inner_folds':4,'public_outer_folds':5,
        'risk_features':P+PUBLIC,'risk_cdf_groups':['dataset_id','output_contract_id','upstream','model_version','target'],
        'primary_aggregation':'equal-context macro of pooled outer-OOF predictions; outer-fold strata also reported',
        'external_role':'previously opened, post-confirmation comparison; original confirmation is unchanged',
        'cluster_training_weights':'equal total weight per biological perturbation cluster',
        'bootstrap_replicates':args.bootstrap,'risk_parameters':{'ridge_alpha':10,'hgb_iter':200,'hgb_learning_rate':.05,'hgb_depth':3,'hgb_min_leaf':20,'hgb_l2':10}}
    config['phase']=args.phase
    config['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tx.atomic_json(OUT/f'EXECUTION_CONFIG_{args.phase}.json',config)
    if args.phase in ('nested','all'):build_nested()
    if args.phase in ('matrix','all'):run_matrix()
    if args.phase in ('ledger','matrix','all'):write_method_budget_ledger()
    if args.phase in ('scaling','all'):run_source_scaling()
    if args.phase in ('feedback','all'):run_feedback()
    if args.phase in ('statistics','all'):run_bootstrap(args.bootstrap)
    if args.phase in ('figures','all'):render_figures()


if __name__=='__main__':main()
