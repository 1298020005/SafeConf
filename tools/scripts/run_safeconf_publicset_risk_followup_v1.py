#!/usr/bin/env python3
"""New DEV PublicSet nested risk follow-up; Source only, context-scoped teachers.

GPU and CPU phases deliberately use separate registered Python environments.
Every completed stage is durable and reusable after an interrupted session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from tools.scripts import run_safeconf_publicset_v1 as pub

OUT = pub.OUT / 'risk_followup_v1'
RUNTIME = pub.RUNTIME / 'risk_followup_v1'
RISK_OUT = OUT / 'registered_universal_v1'
RISK_MODEL_ROOT = RUNTIME / 'registered_universal_v1' / 'risk'
BUILDERS = ('B0_SupportMean', 'B1_HGB', 'B2_Pointwise', 'B3_DeepSets')
CONTEXTS = ('K562', 'RPE1', 'hepg2', 'jurkat')
FIELDS = ('prior_magnitude', 'prior_uncertainty', 'log_history_support',
          'effective_sources', 'history_conflict', 'log_history_count')


def binding(path):
    path = Path(path)
    return {'path': str(path), 'sha256': pub.file_hash(path)}


def read_json(path):
    return json.loads(Path(path).read_text())


def source_domain():
    d = pub.load_domain('Source')
    if len(d.tasks) != 1808 or d.tasks.gene.nunique() != 575:
        raise RuntimeError('registered Source cohort changed')
    return d


def stage_json(path, value):
    pub.atomic_json(path, value)


def teacher_audit(d):
    rows = []
    data = Path('/home/yyf/data/txpert_official_20260802')
    for series, upstream in [('e201', 'TxPert_GAT'), ('e205', 'TxPert_Exphormer')]:
        prefix = series.upper()
        for context in CONTEXTS:
            manifest_path = data / f'cache/E201_blind_{context}/E201_BLIND_VIEW_MANIFEST.json'
            manifest = read_json(manifest_path)
            for seed in (1, 2, 3, 4):
                run_path = data / series / f'formal/{context}/seed_{seed}/{prefix}_RUN_STATUS.json'
                prediction_path = data / series / f'formal/predictions/{context}/seed_{seed}/{prefix}_PREDICTION_RUN.json'
                run, pred = read_json(run_path), read_json(prediction_path)
                view = run['blind_view_manifest']
                ok = (run['status'] == pred['status'] == 'COMPLETE'
                      and run['target'] == pred['target'] == view['target'] == manifest['target'] == context
                      and context not in run['source_contexts']
                      and context not in view['source_contexts']
                      and int(view['n_target_treatments']) == int(manifest['n_target_treatments']) == 0
                      and int(run['target_perturbed_cells_accessed']) == 0
                      and pred['checkpoint_role'] == 'last'
                      and pred['checkpoint_sha256'] == next(x['sha256'] for x in run['checkpoint_files'] if x['role'] == 'last'))
                if not ok:
                    raise RuntimeError(f'invalid context teacher provenance: {series}/{context}/{seed}')
                rows.append({'upstream': upstream, 'context': context, 'teacher_seed': seed,
                             'target_treated_training_rows': 0, 'target_treated_accessed': 0,
                             'training_contexts': '|'.join(run['source_contexts']),
                             'run_status_path': str(run_path), 'run_status_sha256': pub.file_hash(run_path),
                             'prediction_path': str(prediction_path), 'prediction_sha256': pub.file_hash(prediction_path),
                             'checkpoint_sha256': pred['checkpoint_sha256'], 'checkpoint_role': 'last',
                             'view_manifest_sha256': pub.file_hash(manifest_path), 'status': 'PASS'})
    pub.atomic_csv(OUT / 'TEACHER_CONTEXT_PROVENANCE.csv', pd.DataFrame(rows))
    base = pd.read_csv(ROOT/'docs/实验结果/E201_txpert_multitarget_retraining_20260802/tables/E201_PRETRUTH_TASK_BASE.csv').set_index('task_id').loc[d.tasks.task_id]
    native_rows = base.source_mean_delta_row.to_numpy(int)
    native_genes = read_json(pub.SOURCE_BANK/'gene_ids.json')['gene_ids']
    common_genes = read_json(pub.COMMON/'GENE_IDS.json'); lookup = {g:i for i,g in enumerate(native_genes)}
    cols = np.asarray([lookup[g] for g in common_genes]); lineage = []
    for series, upstream in [('e201','TxPert_GAT'),('e205','TxPert_Exphormer')]:
        family_path = data/series/f'pretruth_vectors/{series.upper()}_FAMILY_CENTROIDS.npy'
        control_path = data/series/f'pretruth_vectors/{series.upper()}_CONTROL_CENTROIDS.npy'
        family = np.asarray(np.load(family_path,mmap_mode='r')[native_rows][:,cols],float)
        control = np.asarray(np.load(control_path,mmap_mode='r')[native_rows][:,cols],float)
        derived = (family-control).astype(np.float32)
        if not np.array_equal(derived,d.predictions[upstream]):
            raise RuntimeError('cached common-axis teacher effect differs from native frozen family centroid')
        lineage.append({'upstream':upstream,'family_centroid_binding':binding(family_path),
            'control_binding':binding(control_path),'cached_effect_binding':binding(pub.COMMON/f'SOURCE_{upstream}_PREDICTED_EFFECTS.npy'),
            'exact_common_axis_reproduction':True,'task_row_mapping_hash':pub.fingerprint(base.index)})
    stage_json(OUT/'TEACHER_EFFECT_CACHE_LINEAGE.json',lineage)
    # Cached family effects are the frozen four-seed centroids. They are not new
    # validation exports and every task uses its own context-specific teacher.
    stage_json(OUT / 'INPUT_AUDIT.json', {
        'role': 'NEW_DEV_SEEN_PUBLICSET_CONTEXT_SCOPED', 'n_tasks': len(d.tasks),
        'n_genes': d.tasks.gene.nunique(), 'gene_axis': 2840,
        'teacher_models_checked': len(rows), 'all_context_target_treated_training_rows': 0,
        'source_supervision': 'Other architecture, same context, outer-train genes only',
        'target_errors': 'Evaluation only; never fit labels or CDF',
        'new_upstream_training': 0, 'MC_TEST_reads': 0, 'MC_validation_reads': 0,
        'input_bindings': [binding(pub.COMMON / name) for name in (
            'GENE_IDS.json', 'SOURCE_TASKS.csv', 'SOURCE_PUBLIC_EFFECTS.npy',
            'SOURCE_PUBLIC_CONTROLS.npy', 'SOURCE_TRUE_EFFECTS.npy',
            'SOURCE_TxPert_GAT_PREDICTED_EFFECTS.npy', 'SOURCE_TxPert_Exphormer_PREDICTED_EFFECTS.npy')],
        'code_binding': binding(__file__), 'public_runner_binding': binding(pub.__file__),
    })


def split_rows(d, outer, inner=None):
    train = np.flatnonzero(d.tasks.fold.to_numpy() != outer)
    test = np.flatnonzero(d.tasks.fold.to_numpy() == outer)
    genes = sorted(d.tasks.iloc[train].gene.unique(), key=lambda g: hashlib.sha256(
        f'PublicSet-risk-inner-v1|Source|{outer}|{g}'.encode()).hexdigest())
    mapping = {g: i % 4 for i, g in enumerate(genes)}
    inner_fold = np.asarray([mapping[d.tasks.iloc[i].gene] for i in train])
    if inner is None:
        return train, test, mapping
    fit, query = train[inner_fold != inner], train[inner_fold == inner]
    if set(d.tasks.iloc[fit].gene) & set(d.tasks.iloc[query].gene):
        raise RuntimeError('inner query genes entered training')
    return fit, query, mapping


def validate_groups(d, groups, forbidden_genes):
    hist = np.unique(np.concatenate([g['ix'] for g in groups if len(g['ix'])]))
    if set(d.memory.iloc[hist].perturbation_target) & set(forbidden_genes):
        raise RuntimeError('excluded query genes entered Public training/PCA')
    return hist


def save_priors(d, groups, weights, path, audit):
    path = Path(path)
    records, vectors, qrows, edges = [], [], [], []
    for g in groups:
        q = g['q']
        if not len(g['ix']):
            continue
        w = np.asarray(weights[q], float)
        if np.any(w < -1e-7) or not np.isclose(w.sum(), 1, atol=1e-6):
            raise RuntimeError('invalid public weights')
        h = d.effects[g['ix']]
        mu = np.sum(w[:, None] * h, 0)
        variance = float(w @ np.mean((h - mu) ** 2, axis=1))
        vectors.append(mu); qrows.append(q)
        records.append([np.sqrt(np.mean(mu ** 2)), math.sqrt(max(variance, 0)),
                        math.log1p(g['support']), 1 / (w @ w), g['conflict'], math.log1p(len(g['ix']))])
        for j, weight in zip(g['ix'], w):
            edges.append({'task_id': d.tasks.iloc[q].task_id,
                          'memory_id': d.memory.iloc[j].experiment_id, 'weight': weight})
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / 'PRIORS.tmp.npz'
    np.savez(tmp, qrows=np.asarray(qrows, int), task_ids=d.tasks.iloc[qrows].task_id.to_numpy(str),
             priors=np.asarray(vectors, np.float32), summary=np.asarray(records), fields=np.asarray(FIELDS))
    os.replace(tmp, path / 'PRIORS.npz')
    pub.atomic_csv(path / 'WEIGHTS.csv.gz', pd.DataFrame(edges))
    stage_json(path / 'PREDICTION_AUDIT.json', audit | {
        'n_queries': len(qrows), 'query_ids_hash': pub.fingerprint(d.tasks.iloc[qrows].task_id),
        'prior_binding': binding(path / 'PRIORS.npz'), 'weights_binding': binding(path / 'WEIGHTS.csv.gz')})


def read_priors(path):
    with np.load(Path(path) / 'PRIORS.npz', allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def own_model_save(model, prep, audit, path):
    import joblib
    joblib.dump({k: v for k, v in prep.items() if k != 'codes'}, path / 'PREPROCESSOR.joblib')
    torch.save({'state_dict': model.state_dict(), 'n_features': len(prep['median']) * 2 + prep['codes'].shape[1],
                'collective': model.collective, 'audit': audit}, path / 'MODEL.pt')


def consolidate_nn_ledger():
    rows = []
    for path in sorted((RUNTIME / 'nested').glob('outer*/inner*/B*/seed*/*/FIT_AUDIT.json')):
        if not any('/'+b+'/' in str(path) for b in ('B2_Pointwise','B3_DeepSets')):
            continue
        audit = read_json(path)
        rows.append({'stage_path': str(path.parent), **audit, 'audit_sha256': pub.file_hash(path)})
    pub.atomic_csv(OUT / 'NN_FIT_LEDGER.csv', pd.DataFrame(rows))
    return rows


def gpu(args):
    d = source_domain(); teacher_audit(d)
    if not torch.cuda.is_available():
        raise RuntimeError('registered Torch2.6/CUDA environment required')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    cache = {}; original_fit_preprocessor = pub.fit_preprocessor
    def cached_preprocessor(domain, groups):
        key = pub.fingerprint(domain.tasks.iloc[[g['q'] for g in groups]].task_id)
        if key not in cache:
            cache[key] = original_fit_preprocessor(domain, groups)
        return cache[key]
    pub.fit_preprocessor = cached_preprocessor
    existing = consolidate_nn_ledger()
    spent = sum(r['elapsed_seconds'] for r in existing)
    started = time.monotonic(); deadline = started + max(0, args.gpu_hours_cap * 3600 - spent)
    status = {'status': 'RUNNING', 'pid': os.getpid(), 'phase': 'nested_gpu',
              'started_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'gpu': args.gpu,
              'planned_stages': 240, 'budget_gpu_hours': args.gpu_hours_cap, 'prior_stage_seconds': spent,
              'MC_TEST_reads': 0, 'code_sha256': pub.file_hash(__file__)}
    stage_json(OUT / 'GPU_STATUS.json', status); print(json.dumps(status), flush=True)
    try:
        for outer in args.folds:
            outertrain, outertest, mapping = split_rows(d, outer)
            fold_table = d.tasks.copy(); fold_table['inner_fold'] = fold_table.gene.map(mapping)
            pub.atomic_csv(OUT / f'SPLIT_outer{outer}.csv', fold_table)
            for seed in args.seeds:
                for inner in range(4):
                    fit_rows, query_rows, _ = split_rows(d, outer, inner)
                    groups, query_groups = d.groups(fit_rows, set()), d.groups(query_rows, set())
                    prohibited = set(d.tasks.iloc[np.r_[outertest, query_rows]].gene)
                    histories = validate_groups(d, groups, prohibited)
                    genes = sorted(d.tasks.iloc[fit_rows].gene.unique(), key=lambda g: hashlib.sha256(
                        f'PublicSet-inner-v1|Source|riskouter{outer}|inner{inner}|{g}'.encode()).hexdigest())
                    valgenes = set(genes[:max(1, math.ceil(.2 * len(genes)))])
                    early_rows = np.asarray([q for q in fit_rows if d.tasks.iloc[q].gene not in valgenes])
                    val_rows = np.asarray([q for q in fit_rows if d.tasks.iloc[q].gene in valgenes])
                    earlygroups, valgroups = d.groups(early_rows, set()), d.groups(val_rows, set())
                    validate_groups(d, earlygroups, prohibited | valgenes)
                    for builder, collective in [('B2_Pointwise', False), ('B3_DeepSets', True)]:
                        base = RUNTIME / f'nested/outer{outer}/inner{inner}/{builder}/seed{seed}'
                        if (base / 'PRIORS.npz').exists():
                            continue
                        early = base / 'early'; refit = base / 'refit'
                        early.mkdir(parents=True, exist_ok=True); refit.mkdir(parents=True, exist_ok=True)
                        if (early / 'FIT_AUDIT.json').exists():
                            earlyaudit = read_json(early / 'FIT_AUDIT.json')
                        else:
                            m, p, earlyaudit = pub.training_run(d, earlygroups, valgroups, collective, seed, 100,
                                                               device, True, early, deadline)
                            earlyaudit.update(outer=outer, inner=inner, builder=builder, stage='early',
                                              evaluation_gene_hash=pub.fingerprint(prohibited | valgenes))
                            own_model_save(m, p, earlyaudit, early); stage_json(early / 'FIT_AUDIT.json', earlyaudit)
                            del m, p; consolidate_nn_ledger()
                        if (refit / 'MODEL.pt').exists():
                            model, prep, audit = pub.reuse_model(d, groups, refit, device)
                        else:
                            model, prep, audit = pub.training_run(d, groups, None, collective, seed,
                                                                 earlyaudit['selected_epoch'], device, False, refit, deadline)
                            audit.update(outer=outer, inner=inner, builder=builder, stage='refit',
                                         evaluation_gene_hash=pub.fingerprint(prohibited),
                                         fit_gene_hash=pub.fingerprint(d.tasks.iloc[fit_rows].gene.unique()),
                                         unique_history_gene_disjoint=True)
                            stage_json(refit / 'FIT_AUDIT.json', audit); own_model_save(model, prep, audit, refit)
                        weights = pub.predict_weights(model, prep, d, query_groups, device)
                        save_priors(d, query_groups, weights, base, {
                            'builder': builder, 'seed': seed, 'outer': outer, 'inner': inner,
                            'n_fit_unique_history': len(histories), 'excluded_genes_hash': pub.fingerprint(prohibited),
                            'public_error_supervision': 0, 'source_truth_for_biology_fit_only': True,
                            'model_binding': binding(refit / 'MODEL.pt'), 'audit_binding': binding(refit / 'FIT_AUDIT.json')})
                        ledger = consolidate_nn_ledger()
                        stage_json(OUT / 'GPU_STATUS.json', status | {'completed_stages': len(ledger),
                            'neural_stage_seconds': sum(x['elapsed_seconds'] for x in ledger),
                            'current': f'outer{outer}/inner{inner}/{builder}/seed{seed}'})
                        print(json.dumps({'completed': str(base.relative_to(RUNTIME)),
                                          'completed_stages': len(ledger), 'selected_epoch': audit['selected_epoch'],
                                          'stage_seconds': audit['elapsed_seconds']}), flush=True)
                        del model, prep; torch.cuda.empty_cache()
                if outer == 0 and seed == pub.SEEDS[0]:
                    ledger = consolidate_nn_ledger(); measured = sum(x['elapsed_seconds'] for x in ledger)
                    projection = measured / max(len(ledger), 1) * 240
                    stage_json(OUT / 'FIRST_16_STAGE_TIMING.json', {'completed_stages': len(ledger),
                        'actual_stage_seconds': measured, 'projected_240_stage_seconds': projection,
                        'budget_seconds': args.gpu_hours_cap * 3600,
                        'continue_all_registered_seeds': projection <= args.gpu_hours_cap * 3600})
                    if projection > args.gpu_hours_cap * 3600:
                        raise TimeoutError('16-stage projection exceeds registered GPU budget')
                cache.clear()
        ledger = consolidate_nn_ledger()
        stage_json(OUT / 'GPU_STATUS.json', status | {'status': 'COMPLETE', 'completed_stages': len(ledger),
            'neural_stage_seconds': sum(x['elapsed_seconds'] for x in ledger), 'elapsed_seconds': time.monotonic()-started})
    except Exception as e:
        ledger = consolidate_nn_ledger()
        stage_json(OUT / 'GPU_STATUS.json', status | {'status': 'BUDGET_STOP' if isinstance(e, TimeoutError) else 'FAILED',
            'error': repr(e), 'completed_stages': len(ledger), 'elapsed_seconds': time.monotonic()-started})
        raise


def b1_fit_weights(d, fitgroups, querygroups, path):
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from tools.safeconf_continual.learners import NumericPreprocessor
    start = time.monotonic(); path.mkdir(parents=True, exist_ok=True)
    if (path / 'MODEL.joblib').exists():
        prep, model = joblib.load(path / 'MODEL.joblib')
        audit = read_json(path / 'FIT_AUDIT.json')
        if audit['fit_query_ids_hash'] != pub.fingerprint(d.tasks.iloc[[g['q'] for g in fitgroups]].task_id):
            raise RuntimeError('B1 reuse fit IDs changed')
    else:
        x = np.concatenate([g['x'] for g in fitgroups])
        y = np.concatenate([np.sqrt(np.mean((d.effects[g['ix']] - d.truth[g['q']]) ** 2, axis=1)) for g in fitgroups])
        prep = NumericPreprocessor().fit(x)
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_depth=3,
            min_samples_leaf=20, l2_regularization=10., random_state=20260930)
        model.fit(prep.transform(x), y)
        joblib.dump((prep, model), path / 'MODEL.joblib')
        audit = {'fit_query_ids_hash': pub.fingerprint(d.tasks.iloc[[g['q'] for g in fitgroups]].task_id),
                 'fit_pair_rows': len(x), 'elapsed_seconds': time.monotonic()-start,
                 'objective': 'original individual-history transfer RMSE, pair-row weighting',
                 'model_binding': binding(path / 'MODEL.joblib')}
        stage_json(path / 'FIT_AUDIT.json', audit)
    result = {}
    for g in querygroups:
        score = model.predict(prep.transform(g['x']))
        weight = np.exp(np.clip(-(score-score.min()) / max(score.std(), 1e-8), -20, 20)); weight /= weight.sum()
        result[g['q']] = .5 * weight + .5 * g['s']
    return result, audit


def lookup_outer_model(outer, builder, seed):
    for run in ('full_v2_resume', 'full_v1', 'prototype_v2_cpu_environment_repair'):
        base = pub.RUNTIME / f'{run}/Source/outer{outer}/{builder}/seed{seed}'
        if (base / 'refit/MODEL.pt').exists():
            return base / 'refit'
        for filename in ('MODEL_REUSE.json', 'REUSED_MODEL.json'):
            if (base / filename).exists():
                data = read_json(base / filename)
                for key in ('model_path', 'reuse_model_path', 'source_model_path', 'path'):
                    candidate = Path(data.get(key, ''))
                    if (candidate / 'MODEL.pt').exists():
                        return candidate
    raise FileNotFoundError(f'outer model missing: {outer}/{builder}/{seed}; rerun after full_v2_resume completes')


def gpu_outer(args):
    d = source_domain(); device = torch.device(f'cuda:{args.gpu}'); torch.set_num_threads(4)
    started = time.monotonic()
    records = []
    for outer in args.folds:
        fit, query, _ = split_rows(d, outer)
        fitgroups, querygroups = d.groups(fit, set()), d.groups(query, set())
        for builder in ('B2_Pointwise', 'B3_DeepSets'):
            for seed in args.seeds:
                destination = RUNTIME / f'outer/outer{outer}/{builder}/seed{seed}'
                if (destination / 'PRIORS.npz').exists():
                    continue
                source = lookup_outer_model(outer, builder, seed)
                model, prep, audit = pub.reuse_model(d, fitgroups, source, device)
                weights = pub.predict_weights(model, prep, d, querygroups, device)
                save_priors(d, querygroups, weights, destination, {'model_binding': binding(source/'MODEL.pt'),
                    'preprocessor_binding': binding(source/'PREPROCESSOR.joblib'), 'new_fit': 0,
                    'fit_query_ids_hash': audit['fit_query_ids_hash']})
                records.append({'outer': outer, 'builder': builder, 'seed': seed, 'model_path': str(source), 'new_fit': 0})
                del model, prep; torch.cuda.empty_cache()
    records = []
    for path in sorted((RUNTIME/'outer').glob('outer*/B*/seed*/PREDICTION_AUDIT.json')):
        audit = read_json(path)
        if 'model_binding' in audit:
            records.append({'prediction_path':str(path.parent),'model_path':audit['model_binding']['path'],
                            'model_sha256':audit['model_binding']['sha256'],
                            'preprocessor_sha256':audit['preprocessor_binding']['sha256'],'new_fit':0})
    pub.atomic_csv(OUT / 'OUTER_MODEL_REUSE_LEDGER.csv', pd.DataFrame(records))
    stage_json(OUT/'OUTER_GPU_INFERENCE_RECEIPT.json',{'status':'COMPLETE','pid':os.getpid(),'gpu':args.gpu,
        'new_models':0,'reused_models':len(records),'elapsed_seconds':time.monotonic()-started,
        'MC_TEST_reads':0,'model_reuse_ledger_binding':binding(OUT/'OUTER_MODEL_REUSE_LEDGER.csv')})


def feature_frame(d, prior, upstream, risk_version='registered_universal_v1'):
    from tools.safeconf_continual.research import P, PUBLIC
    q = prior['qrows']; p = d.predictions[upstream][q].astype(float); mu = prior['priors'].astype(float)
    f = d.tasks.iloc[q].copy().reset_index(drop=True).rename(columns={'context': 'target'})
    f['dataset_id'] = 'TxPert_E201'; f['upstream'] = upstream
    f['model_version'] = 'E201_GAT_last_four_seed' if upstream == 'TxPert_GAT' else 'E205_Exphormer_last_four_seed'
    f['output_contract_id'] = 'E201_common2840gene_log1p_delta_v1'
    a = np.abs(p); scale = np.quantile(a, .95, axis=1)
    sparsity = np.mean(a <= 1e-8, axis=1)
    if risk_version == 'legacy_relative_sparsity_v1':
        scale = np.maximum(scale, 1e-12)
        sparsity = np.mean(a <= scale[:, None]*.01, axis=1)
    for name, value in zip(P, (np.sqrt(np.mean(p*p, axis=1)), a.mean(1), p.mean(1), p.std(1), scale,
                              sparsity)): f[name] = value
    for j, name in enumerate(FIELDS): f[name] = prior['summary'][:, j]
    f['prediction_prior_rmse'] = np.sqrt(np.mean((p-mu)**2, axis=1))
    denom = np.linalg.norm(p, axis=1)*np.linalg.norm(mu, axis=1)
    f['prediction_prior_cosine'] = np.divide(np.sum(p*mu, axis=1), denom, out=np.zeros(len(p)), where=denom>1e-12)
    f['true_error_rmse'] = np.sqrt(np.mean((p-d.truth[q].astype(float))**2, axis=1))
    if not np.isfinite(f[P+PUBLIC].to_numpy(float)).all(): raise RuntimeError('nonfinite risk feature')
    return f


def merged_nested(outer, builder, seed):
    pieces = [read_priors(RUNTIME / f'nested/outer{outer}/inner{inner}/{builder}/seed{seed}') for inner in range(4)]
    merged = {k: np.concatenate([p[k] for p in pieces]) for k in ('qrows', 'task_ids', 'priors', 'summary')}
    if len(np.unique(merged['qrows'])) != len(merged['qrows']): raise RuntimeError('duplicate nested OOF query')
    order = np.argsort(merged['qrows']); return {k: v[order] for k, v in merged.items()}


def cpu(args):
    import joblib
    import sklearn
    from tools.safeconf_continual.research import P, PUBLIC, rank_labels, fit_risk, metrics
    d = source_domain(); teacher_audit(d)
    status = {'status': 'RUNNING', 'pid': os.getpid(), 'phase': 'context_scoped_cpu_risk',
              'sklearn': sklearn.__version__, 'numpy': np.__version__, 'MC_TEST_reads': 0,
              'risk_parameters': {'iterations': 200, 'learning_rate': .05, 'depth': 3, 'min_leaf': 20, 'l2': 10},
              'code_sha256': pub.file_hash(__file__), 'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    status['risk_version'] = args.risk_version
    stage_json(RISK_OUT/'CPU_STATUS.json', status); print(json.dumps(status), flush=True)
    start = time.monotonic();waiting=0.
    if args.wait_for_inputs:
        required=[]
        for outer in args.folds:
            for builder in args.builders:
                if builder not in ('B2_Pointwise','B3_DeepSets'):continue
                seeds=pub.SEEDS if builder in ('B2_Pointwise','B3_DeepSets') else (0,)
                for seed in seeds:
                    required.extend(RUNTIME/f'nested/outer{outer}/inner{i}/{builder}/seed{seed}/PRIORS.npz' for i in range(4))
                    required.append(RUNTIME/f'outer/outer{outer}/{builder}/seed{seed}/PRIORS.npz')
        while any(not path.exists() for path in required):
            stage_json(RISK_OUT/'CPU_STATUS.json',status|{'status':'WAITING_FOR_FIXED_PUBLIC_PRIORS',
                'missing_priors':sum(not path.exists() for path in required),'waiting_seconds':waiting})
            waitstart=time.monotonic();time.sleep(5);waiting+=time.monotonic()-waitstart
    for outer in args.folds:
        fit, query, _ = split_rows(d, outer)
        fitgroups, querygroups = d.groups(fit, set()), d.groups(query, set())
        for builder in args.builders:
            seeds = pub.SEEDS if builder in ('B2_Pointwise', 'B3_DeepSets') else (0,)
            for seed in seeds:
                if builder in ('B0_SupportMean', 'B1_HGB'):
                    for inner in range(4):
                        fitrows, queryrows, _ = split_rows(d, outer, inner)
                        tr, qu = d.groups(fitrows, set()), d.groups(queryrows, set())
                        path = RUNTIME / f'nested/outer{outer}/inner{inner}/{builder}/seed0'
                        if (path/'PRIORS.npz').exists(): continue
                        if builder == 'B0_SupportMean':
                            weights, audit = {g['q']: g['s'] for g in qu}, {'new_fit': 0}
                        else:
                            weights, audit = b1_fit_weights(d, tr, qu, path/'fit')
                        save_priors(d, qu, weights, path, audit | {'outer': outer, 'inner': inner,
                            'excluded_query_genes_hash': pub.fingerprint(d.tasks.iloc[np.r_[query, queryrows]].gene.unique())})
                    path = RUNTIME / f'outer/outer{outer}/{builder}/seed0'
                    if not (path/'PRIORS.npz').exists():
                        if builder == 'B0_SupportMean':
                            weights, audit = {g['q']: g['s'] for g in querygroups}, {'new_fit': 0}
                        else:
                            source = None
                            for run in ('full_v2_resume', 'full_v1', 'prototype_v2_cpu_environment_repair'):
                                candidate = pub.RUNTIME / f'{run}/Source/outer{outer}'
                                if (candidate/'B1_HGB_WEIGHTS.csv').exists(): source = candidate; break
                            if source is None: raise FileNotFoundError('outer B1 weights missing; await full_v2_resume')
                            audit = read_json(source/'B1_HGB/FIT_AUDIT.json')
                            if audit['fit_query_ids_hash'] != pub.fingerprint(d.tasks.iloc[fit].task_id):
                                raise RuntimeError('outer B1 scope mismatch')
                            frame = pd.read_csv(source/'B1_HGB_WEIGHTS.csv')
                            if set(frame.task_id) != set(d.tasks.iloc[query].task_id):
                                # Prototype contains only K562. Full resume must provide all contexts.
                                raise RuntimeError('outer B1 reuse query scope incomplete')
                            index = frame.set_index(['task_id', 'memory_id']).weight
                            weights = {g['q']: np.asarray([index.loc[(d.tasks.iloc[g['q']].task_id,
                                d.memory.iloc[i].experiment_id)] for i in g['ix']]) for g in querygroups}
                            audit = {'new_fit': 0, 'weights_binding': binding(source/'B1_HGB_WEIGHTS.csv'),
                                     'source_audit_binding': binding(source/'B1_HGB/FIT_AUDIT.json')}
                        save_priors(d, querygroups, weights, path, audit)
                trprior = merged_nested(outer, builder, seed)
                quprior = read_priors(RUNTIME / f'outer/outer{outer}/{builder}/seed{seed}')
                if set(trprior['qrows']) != set(fit) or set(quprior['qrows']) != set(query):
                    raise RuntimeError('nested/outer features do not cover registered Source fold')
                for source, target in [('TxPert_GAT', 'TxPert_Exphormer'), ('TxPert_Exphormer', 'TxPert_GAT')]:
                    trframe, quframe = feature_frame(d, trprior, source, args.risk_version), feature_frame(d, quprior, target, args.risk_version)
                    for context in CONTEXTS:
                        train = trframe[trframe.target == context].reset_index(drop=True)
                        test = quframe[quframe.target == context].reset_index(drop=True)
                        if set(train.gene) & set(test.gene): raise RuntimeError('risk train query gene overlap')
                        methods = [builder]
                        if builder == 'B0_SupportMean': methods = ['P_only', 'Support_only', builder]
                        for method in methods:
                            riskseeds = (seed,) if builder in ('B2_Pointwise', 'B3_DeepSets') else (pub.SEEDS[0],)
                            for riskseed in riskseeds:
                                path = RISK_MODEL_ROOT / f'{source}_to_{target}/{context}/outer{outer}/{method}/publicseed{seed}/riskseed{riskseed}'
                                if (path/'PREDICTIONS.csv.gz').exists(): continue
                                path.mkdir(parents=True, exist_ok=True); run_id = str(path.relative_to(RUNTIME))
                                labels, cdfa = rank_labels(train, run_id)
                                columns = (P if method == 'P_only' else
                                           P+['log_history_support','log_history_count'] if method == 'Support_only' else P+PUBLIC)
                                fitstart = time.monotonic(); model = fit_risk(train, labels, columns, 'hgb', riskseed)
                                risks = model.predict(test); elapsed = time.monotonic()-fitstart
                                joblib.dump(model, path/'MODEL.joblib')
                                # Full train error ledger makes every CDF and label budget auditable.
                                ledger = train[['dataset_id','output_contract_id','upstream','model_version','target','task_id','gene','fold','true_error_rmse']].copy()
                                ledger['training_rank_label'] = labels; pub.atomic_csv(path/'TRAINING_ERRORS_CDF.csv.gz', ledger)
                                stage_json(path/'CDF_AUDIT.json', cdfa)
                                result = test[['task_id','gene','target','fold','upstream','true_error_rmse']].copy()
                                result['risk'] = risks; result['method'] = method; result['public_seed'] = seed
                                result['risk_seed'] = riskseed; result['source_upstream'] = source
                                pub.atomic_csv(path/'PREDICTIONS.csv.gz', result)
                                stage_json(path/'FIT_AUDIT.json', {'scope': run_id, 'method': method, 'public_seed': seed,
                                    'risk_seed': riskseed, 'source_upstream': source, 'target_upstream': target,
                                    'context': context, 'outer': outer, 'n_train': len(train), 'n_test': len(test),
                                    'fit_gene_hash': pub.fingerprint(train.gene.unique()),
                                    'query_gene_hash': pub.fingerprint(test.gene.unique()),
                                    'training_task_hash': pub.fingerprint(train.task_id), 'feature_columns': columns,
                                    'source_context_exclusive': True, 'source_query_context_teacher_excluded_treated': True,
                                    'target_error_fit_rows': 0, 'elapsed_seconds': elapsed,
                                    'model_binding': binding(path/'MODEL.joblib'),
                                    'training_error_binding': binding(path/'TRAINING_ERRORS_CDF.csv.gz'),
                                    'prediction_binding': binding(path/'PREDICTIONS.csv.gz'), 'risk_version': args.risk_version})
                                print(json.dumps({'completed_risk': run_id, 'fit_seconds': elapsed}), flush=True)
    aggregate()
    stage_json(RISK_OUT/'CPU_STATUS.json', status | {'status': 'COMPLETE_REQUESTED_BUILDERS',
                'builders': args.builders, 'elapsed_seconds': time.monotonic()-start,
                'active_elapsed_seconds':time.monotonic()-start-waiting,'waiting_seconds':waiting})


def aggregate():
    import fcntl
    with open(RISK_MODEL_ROOT.parent/'AGGREGATE.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return aggregate_locked()


def aggregate_locked():
    from tools.safeconf_continual.research import metrics
    frames, fits, cdfs = [], [], []
    for path in sorted(RISK_MODEL_ROOT.glob('*/*/outer*/*/publicseed*/riskseed*/FIT_AUDIT.json')):
        audit = read_json(path); fits.append(audit)
        f = pd.read_csv(path.parent/'PREDICTIONS.csv.gz')
        if audit['method'] in ('P_only', 'Support_only', 'B0_SupportMean', 'B1_HGB'):
            f = pd.concat([f.assign(risk_seed=s, public_seed=s) for s in pub.SEEDS], ignore_index=True)
        elif audit['public_seed'] != audit['risk_seed']:
            continue
        frames.append(f)
        cdfs.extend(read_json(path.parent/'CDF_AUDIT.json'))
    if not frames: return
    frame = pd.concat(frames, ignore_index=True)
    keys = ['source_upstream','upstream','target','fold','method','public_seed','risk_seed']
    rows = [dict(zip(keys, key)) | metrics(part, part.risk.to_numpy()) for key, part in frame.groupby(keys, sort=True)]
    pooledkeys = [c for c in keys if c != 'fold']
    pooled = pd.DataFrame([dict(zip(pooledkeys, key)) | metrics(part, part.risk.to_numpy())
                           for key, part in frame.groupby(pooledkeys, sort=True)])
    met = ['utility20','spearman','aurc','high_risk_miss_rate','error_at_10','error_at_20','error_at_50']
    macrokeys = ['source_upstream','upstream','method','public_seed','risk_seed']
    foldresults = pd.DataFrame(rows)
    macro = foldresults.groupby(macrokeys, as_index=False)[met].mean()
    pub.atomic_csv(RISK_OUT/'TASK_PREDICTIONS.csv.gz', frame)
    pub.atomic_csv(RISK_OUT/'CONTEXT_FOLD_RESULTS.csv', foldresults)
    pub.atomic_csv(RISK_OUT/'POOLED_CONTEXT_RESULTS.csv', pooled)
    pub.atomic_csv(RISK_OUT/'POOLED_CONTEXT_MACRO_RESULTS.csv', pooled.groupby(macrokeys, as_index=False)[met].mean())
    pub.atomic_csv(RISK_OUT/'MACRO_RESULTS.csv', macro)
    pub.atomic_csv(RISK_OUT/'SEED_MEAN_MACRO_RESULTS.csv', macro.groupby(['source_upstream','upstream','method'], as_index=False)[met].mean())
    pub.atomic_csv(RISK_OUT/'RISK_FIT_LEDGER.csv', pd.DataFrame(fits))
    pub.atomic_csv(RISK_OUT/'CDF_LEDGER.csv', pd.DataFrame(cdfs))
    stage_json(RISK_OUT/'RESULT_MANIFEST.json', {'status': 'COMPLETE' if len(frame) == 1808*2*3*6 and len(fits) == 400 else 'PARTIAL',
        'task_prediction_rows': len(frame), 'expected_full_rows': 1808*2*3*6,
        'actual_risk_fits': len(fits), 'scope': 'Source1808/575/2840, two architecture directions, four context teachers',
        'primary_aggregation': 'equal macro of 20 context by outer-fold strata, then mean of three seed metrics',
        'supplemental_aggregation': 'four context macro of pooled outer OOF',
        'target_error_training_records': 0, 'MC_TEST_reads': 0,
        'physical_content_shuffle': 'NOT_RUN; real Public and Support first',
        'bindings': [binding(RISK_OUT/name) for name in ('TASK_PREDICTIONS.csv.gz','CONTEXT_FOLD_RESULTS.csv',
             'POOLED_CONTEXT_RESULTS.csv','MACRO_RESULTS.csv','RISK_FIT_LEDGER.csv','CDF_LEDGER.csv')]})


def paired_statistics(frame, output, runtime, methods, contrasts, bootstrap=5000,
                      orders=(0,), order_column=None):
    """Same 575-gene draws across every context/fold, seed, arm and null order."""
    from tools.safeconf_continual.research import metrics
    output, runtime = Path(output), Path(runtime)
    output.mkdir(parents=True, exist_ok=True); runtime.mkdir(parents=True, exist_ok=True)
    rows, gates, stratarows = [], [], []
    met = ['utility20','spearman','aurc','high_risk_miss_rate','error_at_10','error_at_20','error_at_50']
    safetycols = ['aurc','error_at_10','error_at_20','error_at_50']
    for (source, target), part in frame.groupby(['source_upstream','upstream'], sort=True):
        if set(part.method) != set(methods): raise RuntimeError('all fixed arms required for paired statistics')
        identity = ['task_id','gene','target','fold']
        base = part[identity+['true_error_rmse']].drop_duplicates().sort_values('task_id').reset_index(drop=True)
        if len(base) != 1808 or base.gene.nunique() != 575: raise RuntimeError('paired Source cohort changed')
        scores = np.empty((len(base), len(pub.SEEDS), len(orders), len(methods)))
        for s, seed in enumerate(pub.SEEDS):
            for o, order in enumerate(orders):
                for j, method in enumerate(methods):
                    selected = (part.public_seed == seed)&(part.method == method)
                    if order_column: selected &= part[order_column] == order
                    query = part[selected].set_index('task_id')
                    if len(query) != 1808 or query.index.duplicated().any():
                        raise RuntimeError(f'missing paired seed/order/arm: {seed}/{order}/{method}')
                    if not np.allclose(query.loc[base.task_id].true_error_rmse, base.true_error_rmse, atol=1e-14, rtol=1e-12):
                        raise RuntimeError('paired arm error mismatch')
                    scores[:,s,o,j] = query.loc[base.task_id].risk.to_numpy()
        contexts = base.target.to_numpy(str); folds = base.fold.to_numpy(int)
        error = base.true_error_rmse.to_numpy(float); gene = base.gene.to_numpy(str)
        clusters = sorted(set(gene)); clusterrows = [np.flatnonzero(gene == g) for g in clusters]
        taskids = base.task_id.to_numpy(str); flat_scores = scores.reshape(len(base), -1)
        def all_utility(use, pooled=False):
            values = []
            for fold in ([None] if pooled else range(5)):
                for context in CONTEXTS:
                    keep = contexts[use] == context
                    if fold is not None: keep &= folds[use] == fold
                    ix = use[keep]; ix = ix[np.argsort(taskids[ix], kind='stable')]
                    n = len(ix); k = math.ceil(.2*n)
                    if n < 20: values.append(np.full(flat_scores.shape[1], np.nan)); continue
                    e = error[ix]; denominator = np.sort(e)[-k:].mean()-e.mean()
                    if denominator <= 1e-12: values.append(np.full(flat_scores.shape[1], np.nan)); continue
                    hi = np.argsort(-flat_scores[ix], axis=0, kind='stable')[:k]
                    values.append((e[hi].mean(axis=0)-e.mean())/denominator)
            return np.nanmean(values, axis=0).reshape(len(pub.SEEDS), len(orders), len(methods))
        observed = all_utility(np.arange(len(base)))
        pooledobserved = all_utility(np.arange(len(base)), True)
        rng = np.random.default_rng(pub.SEEDS[0])
        gene_counts = np.asarray([np.bincount(rng.integers(0,len(clusters),len(clusters)),
            minlength=len(clusters)) for _ in range(bootstrap)],dtype=np.int32)
        geneindex = np.searchsorted(np.asarray(clusters),gene)
        def weighted_draws(pooled=False):
            # Repeated copies of a task have identical scores/errors. Integer
            # gene multiplicities exactly reproduce expanded paired draws and
            # avoid sorting the same fixed predictions 5000 times.
            values=[]
            for fold in ([None] if pooled else range(5)):
                for context in CONTEXTS:
                    keep = contexts == context
                    if fold is not None: keep &= folds == fold
                    ix=np.flatnonzero(keep);ix=ix[np.argsort(taskids[ix],kind='stable')]
                    e=error[ix]; high=np.argsort(-flat_scores[ix],axis=0,kind='stable')
                    oracle=np.argsort(-e,kind='stable'); result=np.full((bootstrap,flat_scores.shape[1]),np.nan)
                    for begin in range(0,bootstrap,128):
                        counts=gene_counts[begin:begin+128, geneindex[ix]]
                        n=counts.sum(1); k=np.ceil(.2*n).astype(int); safe_k=np.maximum(k,1)
                        avg=np.divide(counts@e,n,out=np.zeros(len(n)),where=n>0)
                        oc=counts[:,oracle]; prior=np.cumsum(oc,axis=1)-oc
                        used=np.minimum(np.maximum(k[:,None]-prior,0),oc)
                        oraclemean=(used*e[oracle]).sum(1)/safe_k
                        denom=oraclemean-avg
                        hc=counts[:,high]; prior=np.cumsum(hc,axis=1)-hc
                        used=np.minimum(np.maximum(k[:,None,None]-prior,0),hc)
                        topmean=(used*e[high][None]).sum(1)/safe_k[:,None]
                        valid=(n>=20)&(denom>1e-12)
                        result[begin:begin+len(n)][valid]=(topmean[valid]-avg[valid,None])/denom[valid,None]
                    values.append(result)
            return np.nanmean(values,axis=0).reshape(bootstrap,len(pub.SEEDS),len(orders),len(methods))
        draws=weighted_draws();pooleddraws=weighted_draws(True)
        max_bootstrap_difference=0.
        for b in range(min(3,bootstrap)):
            use=np.concatenate([np.tile(clusterrows[j],gene_counts[b,j]) for j in range(len(clusters)) if gene_counts[b,j]])
            direct,pooleddirect=all_utility(use),all_utility(use,True)
            max_bootstrap_difference=max(max_bootstrap_difference,float(np.nanmax(np.abs(draws[b]-direct))),
                                          float(np.nanmax(np.abs(pooleddraws[b]-pooleddirect))))
        if max_bootstrap_difference>1e-12:raise RuntimeError('weighted gene bootstrap differs from exact task expansion')
        np.savez(runtime/f'{source}_to_{target}_JOINT_GENE_BOOTSTRAP.npz', draws=draws,
            pooled_draws=pooleddraws, methods=np.asarray(methods), seeds=np.asarray(pub.SEEDS),
            orders=np.asarray(orders), observed=observed, pooled_observed=pooledobserved,
            gene_multiplicities=gene_counts,genes=np.asarray(clusters),
            expansion_equivalence_max_difference=max_bootstrap_difference)
        metricarray = np.empty((20, len(pub.SEEDS), len(orders), len(methods), len(met)))
        for fold in range(5):
            for ci, context in enumerate(CONTEXTS):
                mask = (base.fold == fold)&(base.target == context)
                f = base[mask]; ix = f.index.to_numpy()
                for s, seed in enumerate(pub.SEEDS):
                    for o, order in enumerate(orders):
                        for j, method in enumerate(methods):
                            measured = metrics(f, scores[ix,s,o,j])
                            metricarray[fold*4+ci,s,o,j] = [measured.get(c,np.nan) for c in met]
        macro = np.nanmean(metricarray, axis=(0,1,2))
        for a, b in contrasts:
            aj, bj = methods.index(a), methods.index(b)
            for aggregation, point, boot in [('20 context by outer-fold macro',observed,draws),
                                             ('four pooled-context macro (supplement)',pooledobserved,pooleddraws)]:
                diff = boot[:,:,:,aj]-boot[:,:,:,bj]
                for s, seed in enumerate(pub.SEEDS):
                    for o, order in enumerate(orders):
                        rows.append({'source_upstream':source,'upstream':target,'candidate':a,'comparator':b,
                            'seed':seed,'null_order':order,'aggregation':aggregation,
                            'delta_utility20':point[s,o,aj]-point[s,o,bj],
                            'ci95_lower':np.nanquantile(diff[:,s,o],.025),'ci95_upper':np.nanquantile(diff[:,s,o],.975),
                            'bootstrap_replicates':bootstrap,'n_tasks':len(base),'n_gene_clusters':len(clusters),
                            'fixed_fit_nominal_interval':True})
                avg = np.nanmean(diff,axis=(1,2))
                row = {'source_upstream':source,'upstream':target,'candidate':a,'comparator':b,
                    'seed':'three_seed_mean','null_order':'all_order_mean','aggregation':aggregation,
                    'delta_utility20':float(np.nanmean(point[:,:,aj]-point[:,:,bj])),
                    'ci95_lower':float(np.nanquantile(avg,.025)),'ci95_upper':float(np.nanquantile(avg,.975)),
                    'bootstrap_replicates':bootstrap,'n_tasks':len(base),'n_gene_clusters':len(clusters),
                    'n_orders':len(orders),'fixed_fit_nominal_interval':True}
                rows.append(row)
                if aggregation.startswith('four'): continue
                strata = []
                for fold in range(5):
                    for ci, context in enumerate(CONTEXTS):
                        av = np.nanmean(metricarray[fold*4+ci,:,:,aj],axis=(0,1))
                        bv = np.nanmean(metricarray[fold*4+ci,:,:,bj],axis=(0,1))
                        values = {c:av[met.index(c)]/bv[met.index(c)]-1 for c in safetycols}
                        values['high_risk_miss_rate'] = av[met.index('high_risk_miss_rate')]-bv[met.index('high_risk_miss_rate')]
                        item = {'fold':fold,'context':context,'delta':float(av[0]-bv[0]),**values}
                        strata.append(item); stratarows.append(row | item)
                valid = [s for s in strata if np.isfinite(s['delta'])]
                gate = row | {'planned_strata':20,'valid_strata':len(valid),'valid_strata_fraction':len(valid)/20,
                    'nonnegative_strata_fraction':sum(s['delta']>=0 for s in valid)/max(len(valid),1),
                    'worst_high_risk_miss_rate_increase':max((s['high_risk_miss_rate'] for s in valid),default=np.nan)}
                for c in safetycols:
                    gate['macro_'+c+'_relative_increase'] = float(macro[aj,met.index(c)]/macro[bj,met.index(c)]-1)
                    gate['worst_'+c+'_relative_increase'] = max((s[c] for s in valid),default=np.nan)
                gate['macro_high_risk_miss_rate_increase'] = float(macro[aj,met.index('high_risk_miss_rate')]-macro[bj,met.index('high_risk_miss_rate')])
                gate['fixed_gate_pass'] = (gate['delta_utility20']>=.005 and len(valid)>=16 and
                    gate['nonnegative_strata_fraction']>=.6 and gate['ci95_lower']>=-.005 and
                    all(gate['macro_'+c+'_relative_increase']<=.05 for c in safetycols) and
                    gate['macro_high_risk_miss_rate_increase']<=.02)
                gates.append(gate)
        pub.atomic_csv(output/'PAIRED_GENE_BOOTSTRAP_CONTRASTS.csv',pd.DataFrame(rows))
        pub.atomic_csv(output/'FIXED_ADOPTION_GATES.csv',pd.DataFrame(gates))
        pub.atomic_csv(output/'CONTRAST_STRATA_TRANSPARENCY.csv',pd.DataFrame(stratarows))
        print(json.dumps({'completed_statistics':f'{source}_to_{target}','bootstrap':bootstrap}),flush=True)
    stage_json(output/'STATISTICS_RECEIPT.json', {'status':'COMPLETE','bootstrap_replicates':bootstrap,
        'scheme':'Identical paired 575-gene draws shared across 20 strata, all methods, three seeds and all fixed null orders',
        'primary':'Equal macro of 20 context by outer-fold metrics; seed metrics averaged, never risk scores',
        'supplement':'Four context macro of pooled outer OOF',
        'safety_gate':'Macro AURC and errors at 10/20/50 <=5% increase; macro miss <=+.02; worst strata disclosed only',
        'adoption_gate':'delta >=.005; >=60% valid strata nonnegative; >=80% strata valid; CI lower >=-.005',
        'training_uncertainty_included':False,'fixed_model_nominal_intervals':True,
        'bootstrap_computation':'Integer gene multiplicities exactly reproduce expanded resamples; first3 draws checked in both aggregations to1e-12',
        'bindings':[binding(output/'PAIRED_GENE_BOOTSTRAP_CONTRASTS.csv'),binding(output/'FIXED_ADOPTION_GATES.csv'),
                    binding(output/'CONTRAST_STRATA_TRANSPARENCY.csv')]})


def statistics(args):
    while args.wait_for_inputs:
        completed=len(list(RISK_MODEL_ROOT.glob('*/*/outer*/*/publicseed*/riskseed*/FIT_AUDIT.json')))
        if completed==400:break
        stage_json(RISK_OUT/'STATISTICS_STATUS.json',{'status':'WAITING_FOR_ALL_FIXED_ARMS','pid':os.getpid(),
            'actual_risk_fits':completed,'expected_risk_fits':400,'MC_TEST_reads':0})
        time.sleep(5)
    start=time.monotonic()
    aggregate()
    frame = pd.read_csv(RISK_OUT/'TASK_PREDICTIONS.csv.gz')
    methods = ['P_only','Support_only','B0_SupportMean','B1_HGB','B2_Pointwise','B3_DeepSets']
    contrasts = [('B0_SupportMean','Support_only'),('B1_HGB','Support_only'),
                 ('B2_Pointwise','P_only'),('B3_DeepSets','P_only'),
                 ('B2_Pointwise','Support_only'),('B3_DeepSets','Support_only'),
                 ('B2_Pointwise','B0_SupportMean'),('B3_DeepSets','B0_SupportMean'),
                 ('B2_Pointwise','B1_HGB'),('B3_DeepSets','B1_HGB'),('B3_DeepSets','B2_Pointwise')]
    paired_statistics(frame,RISK_OUT,RISK_MODEL_ROOT.parent,methods,contrasts,args.bootstrap)
    stage_json(RISK_OUT/'STATISTICS_STATUS.json',{'status':'COMPLETE','pid':os.getpid(),
        'active_elapsed_seconds':time.monotonic()-start,'MC_TEST_reads':0})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['nested-gpu','outer-gpu','cpu-risk','aggregate','statistics'], required=True)
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--seeds', nargs='+', type=int, default=list(pub.SEEDS))
    p.add_argument('--builders', nargs='+', choices=BUILDERS, default=list(BUILDERS))
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--gpu-hours-cap', type=float, default=2.)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--risk-version', choices=['registered_universal_v1','legacy_relative_sparsity_v1'], default='registered_universal_v1')
    p.add_argument('--wait-for-inputs',action='store_true')
    args = p.parse_args()
    global RISK_OUT, RISK_MODEL_ROOT
    if args.risk_version == 'legacy_relative_sparsity_v1':
        RISK_OUT, RISK_MODEL_ROOT = OUT, RUNTIME/'risk'
    RISK_OUT.mkdir(parents=True, exist_ok=True); RISK_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.phase == 'nested-gpu': gpu(args)
    elif args.phase == 'outer-gpu': gpu_outer(args)
    elif args.phase == 'cpu-risk': cpu(args)
    elif args.phase == 'statistics': statistics(args)
    else: aggregate()


if __name__ == '__main__': main()
