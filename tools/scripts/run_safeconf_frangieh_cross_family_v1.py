#!/usr/bin/env python3
"""SEEN retrospective scGPT <-> GEARS risk transfer on native Frangieh512.

Original upstream TRAIN/VAL remain biological history; original upstream TEST
is explicitly re-registered as a retrospective risk-development cohort.  Each
original checkpoint fold is isolated. Source labels and predictors are never
averaged with the target model, and risk train/evaluation are gene-disjoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import h5py
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC, fit_risk, ids_hash, metrics, rank_labels

OLD = Path('/home/yyf/proj/docs/实验结果')
MANIFEST = OLD / 'E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv'
H5AD = Path('/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad')
ASSETS = {
    'scGPT': OLD / 'E106_frangieh_context_scgpt_20260713',
    'GEARS': OLD / 'E107_frangieh_context_gears_20260713',
}
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/frangieh_cross_family_v1'
RUN = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/frangieh_cross_family_v1')
CONTRACT = 'Frangieh_E106_E107_native512_context_control_delta_v1'
SPLIT_PREFIX = 'frangieh-native-risk-v1|'
SEED = 20261002
METHODS = ['Magnitude', 'NegativeHistorySupport', 'DirectHistoryDistance', 'WeightedHistoryDistance',
           'PredictionRidge', 'PredictionHGB', 'PublicRidge', 'PublicHGB']
LEARNED = [('PredictionRidge', P, 'ridge'), ('PredictionHGB', P, 'hgb'),
           ('PublicRidge', P + PUBLIC, 'ridge'), ('PublicHGB', P + PUBLIC, 'hgb')]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def save_csv(name, values):
    pd.DataFrame(values).to_csv(DOC / name, index=False)


def log(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def bind_inputs(manifest):
    with h5py.File(H5AD, 'r') as h:
        genes = [decode(x) for x in h['var/gene_name'][()]]
    assert len(genes) == len(set(genes)) == 512
    write_json(RUN / 'GENE_IDS.json', genes)
    records = [{'path': str(MANIFEST), 'sha256': sha(MANIFEST), 'kind': 'task_split'},
               {'path': str(H5AD), 'sha256': None, 'kind': 'gene_metadata_only',
                'gene_order_sha256': hashlib.sha256('\n'.join(genes).encode()).hexdigest()}]
    for fold in sorted(manifest.fold_id.unique()):
        for name, directory in ASSETS.items():
            for filename in ('predicted_effects.npz', 'true_effects.npz', 'RUN_STATUS.json'):
                p = directory / 'folds' / fold / filename
                records.append({'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size,
                                'kind': filename, 'upstream': name, 'original_fold': fold})
    save_csv('INPUT_HASHES.csv', records)
    return records


def load_fold(fold, manifest):
    frame = manifest[manifest.fold_id.eq(fold)].copy().reset_index(drop=True)
    assert len(frame) == 567
    frame['task_id'] = frame.context.astype(str) + '::' + frame.perturbation.astype(str)
    frame['gene'] = frame.perturbation.str.replace('+ctrl', '', regex=False)
    frame['array_key'] = frame.split + '::' + frame.task_id
    assert not frame.task_id.duplicated().any()
    truth = None
    predictions = {}
    expected = set(frame.array_key)
    for name, directory in ASSETS.items():
        source = directory / 'folds' / fold
        with np.load(source / 'predicted_effects.npz', allow_pickle=False) as z:
            assert set(z.files) == expected
            p = np.stack([z[k] for k in frame.array_key]).astype(float)
        with np.load(source / 'true_effects.npz', allow_pickle=False) as z:
            assert set(z.files) == expected
            y = np.stack([z[k] for k in frame.array_key]).astype(float)
        assert p.shape == y.shape == (567, 512)
        assert np.isfinite(p).all() and np.isfinite(y).all()
        if truth is not None:
            if not np.allclose(truth, y, rtol=1e-6, atol=1e-7):
                raise ValueError('two upstream caches disagree on biological truth')
        else:
            truth = y
        predictions[name] = p
    return frame, truth, predictions


def public_features(frame, truth):
    """Known biology only: original TRAIN/VAL; old TEST never enters the bank."""
    n = len(frame)
    prior = np.full((n, 512), np.nan)
    result = pd.DataFrame(index=frame.index)
    for name in PUBLIC:
        result[name] = np.nan
    result['n_history'] = 0
    result['history_available'] = False
    result['log_history_support'] = 0.0
    result['effective_sources'] = 0.0
    edges = []
    bank = frame.split.isin(['train', 'val']).to_numpy()
    for row in frame.itertuples():
        ix = np.flatnonzero(bank & frame.gene.eq(row.gene).to_numpy()
                            & frame.context.ne(row.context).to_numpy()
                            & frame.task_id.ne(row.task_id).to_numpy())
        if not len(ix):
            continue
        assert frame.iloc[ix].split.isin(['train', 'val']).all()
        assert row.Index not in ix
        vectors = truth[ix]
        support = frame.iloc[ix].n_cells.to_numpy(float)
        assert (support > 0).all()
        weight = support / support.sum()
        mu = weight @ vectors
        dispersion = float(np.sum(weight * np.mean((vectors - mu) ** 2, axis=1)))
        # Preserve the original unweighted bank conflict independently of weights.
        conflict = float(np.sqrt(np.mean((vectors - vectors.mean(axis=0)) ** 2, axis=1)).mean())
        prior[row.Index] = mu
        result.loc[row.Index, ['prior_magnitude', 'prior_uncertainty', 'log_history_support',
                             'effective_sources', 'history_conflict', 'n_history', 'history_available']] = [
            np.sqrt(np.mean(mu ** 2)), np.sqrt(dispersion), np.log1p(support.sum()),
            1 / np.sum(weight ** 2), conflict, len(ix), True]
        for j, w in zip(ix, weight):
            edges.append({'original_fold': row.fold_id, 'query_task_id': row.task_id,
                          'query_split': row.split, 'query_gene': row.gene,
                          'history_task_id': frame.iloc[j].task_id,
                          'history_split': frame.iloc[j].split, 'history_context': frame.iloc[j].context,
                          'n_history_cells': int(frame.iloc[j].n_cells), 'weight': float(w),
                          'query_truth_used_for_history': False})
    return prior, result, edges


def make_features(frame, prediction, prior, public, upstream):
    out = frame.copy()
    absolute = np.abs(prediction)
    out['predicted_magnitude'] = np.sqrt(np.mean(prediction ** 2, axis=1))
    out['prediction_abs_mean'] = absolute.mean(axis=1)
    out['prediction_signed_mean'] = prediction.mean(axis=1)
    out['prediction_std'] = prediction.std(axis=1)
    out['prediction_abs_q95'] = np.quantile(absolute, .95, axis=1)
    out['prediction_sparsity'] = np.mean(absolute <= np.maximum(out.prediction_abs_q95.to_numpy(), 1e-12)[:, None] * .01, axis=1)
    for col in public:
        out[col] = public[col].to_numpy()
    out['prediction_prior_rmse'] = np.sqrt(np.mean((prediction - prior) ** 2, axis=1))
    dot = np.sum(prediction * prior, axis=1)
    norm = np.linalg.norm(prediction, axis=1) * np.linalg.norm(prior, axis=1)
    cosine = np.divide(dot, norm, out=np.zeros(len(out)), where=norm > 1e-12)
    cosine[~out.history_available.to_numpy(bool)] = np.nan
    out['prediction_prior_cosine'] = cosine
    out['weighted_history_distance'] = np.sqrt(out.prediction_prior_rmse ** 2 + out.prior_uncertainty ** 2)
    out['dataset_id'] = 'Frangieh_SEEN_native512'
    out['upstream'] = upstream
    out['model_version'] = upstream + '_' + out.fold_id
    out['output_contract_id'] = CONTRACT
    out['target'] = out.context
    return out


def check_contract(frame, prediction, truth, prior, public):
    test = frame.split.eq('test')
    assert test.sum() == 279 and frame.loc[test, 'gene'].nunique() == 189
    for i in np.flatnonzero(test.to_numpy() & public.history_available.to_numpy(bool)):
        candidate = frame.split.isin(['train', 'val']) & frame.gene.eq(frame.iloc[i].gene) & frame.context.ne(frame.iloc[i].context)
        ix = np.flatnonzero(candidate.to_numpy())
        w = frame.iloc[ix].n_cells.to_numpy(float)
        w /= w.sum()
        expanded = np.sum(w * np.mean((prediction[i] - truth[ix]) ** 2, axis=1))
        decomposed = np.mean((prediction[i] - prior[i]) ** 2) + public.iloc[i].prior_uncertainty ** 2
        if not np.isclose(expanded, decomposed, rtol=1e-10, atol=1e-12):
            raise ValueError('weighted distance identity failed')
        if len(ix) == 1:
            assert public.iloc[i].prior_uncertainty == 0
    no_history = ~public.history_available.to_numpy(bool)
    assert np.isnan(prior[no_history]).all()


def baseline_predict(frame, truth, fit_mask, query_indices, kind):
    if kind == 'ZeroEffect':
        return np.zeros((len(query_indices), 512))
    train = np.flatnonzero(fit_mask)
    global_mean = truth[train].mean(axis=0)
    if kind == 'TrainGlobalMean':
        return np.repeat(global_mean[None], len(query_indices), axis=0)
    answers = []
    for q in query_indices:
        context = frame.iloc[q].context
        local = train[frame.iloc[train].context.to_numpy() == context]
        answers.append(truth[local].mean(axis=0) if len(local) else global_mean)
    return np.stack(answers)


def competence(frame, truth, predictions, folds, reps):
    train = frame.split.eq('train').to_numpy()
    val = np.flatnonzero(frame.split.eq('val').to_numpy())
    train_rows = np.flatnonzero(train)
    kinds = ['ZeroEffect', 'TrainGlobalMean', 'TrainContextMean']
    candidates = {}
    for kind in kinds:
        e = []
        for fold in range(5):
            query = train_rows[frame.iloc[train_rows].risk_fold.to_numpy() == fold]
            if not len(query):
                continue
            fit = train & frame.risk_fold.ne(fold).to_numpy()
            prediction = baseline_predict(frame, truth, fit, query, kind)
            temp = frame.iloc[query][['context', 'gene']].copy()
            temp['error'] = np.sqrt(np.mean((prediction - truth[query]) ** 2, axis=1))
            e.append(temp)
        candidates[kind] = pd.concat(e).groupby('context').error.mean().mean()
    chosen = min(kinds, key=lambda k: (candidates[k], kinds.index(k)))
    b = baseline_predict(frame, truth, train, val, chosen)
    be = np.sqrt(np.mean((b - truth[val]) ** 2, axis=1))
    records, detail = [], []
    genes = sorted(frame.iloc[val].gene.unique())
    gene_idx = pd.Index(genes).get_indexer(frame.iloc[val].gene)
    counts = np.random.default_rng(SEED).multinomial(len(genes), np.ones(len(genes)) / len(genes), size=reps)
    weights = counts[:, gene_idx]
    for model, pred in predictions.items():
        e = np.sqrt(np.mean((pred[val] - truth[val]) ** 2, axis=1))
        v = frame.iloc[val].copy()
        v['model_error'] = e
        v['baseline_error'] = be
        context_result = v.groupby('context')[['model_error', 'baseline_error']].mean()
        gap = context_result.model_error.mean() / context_result.baseline_error.mean() - 1
        bm, mm = [], []
        for context in sorted(v.context.unique()):
            select = v.context.to_numpy() == context
            den = weights[:, select].sum(axis=1)
            bm.append(np.divide(weights[:, select] @ be[select], den, out=np.full(reps, np.nan), where=den > 0))
            mm.append(np.divide(weights[:, select] @ e[select], den, out=np.full(reps, np.nan), where=den > 0))
        draw = np.nanmean(mm, axis=0) / np.nanmean(bm, axis=0) - 1
        lower, upper = np.nanquantile(draw, [.025, .975])
        fraction = float((context_result.model_error / context_result.baseline_error <= 1.02).mean())
        non_worse = float((context_result.model_error <= context_result.baseline_error).mean())
        records.append({'original_fold': str(frame.fold_id.iloc[0]), 'upstream': model,
                        'n_validation_tasks': len(val), 'n_validation_clusters': len(genes),
                        'baseline_selected_on_train_oof': chosen, 'relative_macro_rmse_gap': float(gap),
                        'context_fraction_within_2pct': fraction, 'context_fraction_nonworse': non_worse,
                        'ci95_lower': float(lower), 'ci95_upper': float(upper),
                        'competence_pass': bool(gap <= .02 and non_worse >= .60 and lower <= .02),
                        'all_train_oof_baselines': json.dumps({k: float(x) for k, x in candidates.items()}),
                        'qualification': 'validation_used_for_upstream_early_stopping; retrospective competence; retain all models'})
        for row, er, br in zip(v.itertuples(), e, be):
            detail.append({'original_fold': row.fold_id, 'upstream': model, 'task_id': row.task_id,
                           'gene': row.gene, 'context': row.context, 'model_error': float(er),
                           'baseline_error': float(br), 'baseline': chosen})
    return records, detail


def summarize(predictions):
    records = []
    keys = ['direction', 'original_fold', 'method']
    for key, group in predictions.groupby(keys, sort=True):
        meta = dict(zip(keys, key))
        for scope, subset in [('all_test', group), ('heldout_context', group[group.context.eq(group.heldout_context)]),
                              ('history_available', group[group.history_available]),
                              ('no_history', group[~group.history_available])]:
            for risk_fold, part in subset.groupby('risk_fold'):
                records.append(meta | {'scope': scope, 'risk_fold': risk_fold, 'aggregation': 'risk_fold'} | metrics(part, part.risk.to_numpy()))
            records.append(meta | {'scope': scope, 'risk_fold': -1, 'aggregation': 'original_fold_oof'} | metrics(subset, subset.risk.to_numpy()))
        for context, part in group.groupby('context'):
            records.append(meta | {'scope': 'context:' + context, 'risk_fold': -1, 'aggregation': 'context_oof'} | metrics(part, part.risk.to_numpy()))
    strata = pd.DataFrame(records)
    save_csv('STRATUM_RESULTS.csv', strata)
    cols = ['utility20', 'spearman', 'aurc', 'high_risk_miss_rate', 'error_at_10', 'error_at_20', 'error_at_50']
    macro = strata.groupby(['direction', 'method', 'scope', 'aggregation'], dropna=False)[cols].mean().reset_index()
    sizes = strata.groupby(['direction', 'method', 'scope', 'aggregation'], dropna=False).agg(
        planned_strata=('utility20', 'size'), valid_strata=('utility20', 'count'),
        evaluated_row_uses=('n_tasks', 'sum'), planned_row_uses=('n_planned', 'sum')).reset_index()
    macro = macro.merge(sizes)
    save_csv('MACRO_RESULTS.csv', macro)
    return strata, macro


def weighted_utility(errors, score, ids, weights):
    """Exact multiplicity bootstrap U20, preserving task-ID tie order."""
    n = weights.sum(axis=1)
    k = np.ceil(.2 * n)
    order = np.lexsort((ids, -score))
    oracle = np.lexsort((ids, -errors))
    def top_mean(rank):
        w = weights[:, rank]
        preceding = np.cumsum(w, axis=1) - w
        used = np.minimum(w, np.maximum(k[:, None] - preceding, 0))
        return np.divide(used @ errors[rank], k, out=np.full(len(k), np.nan), where=k > 0)
    mean = np.divide(weights @ errors, n, out=np.full(len(n), np.nan), where=n > 0)
    denominator = top_mean(oracle) - mean
    return np.divide(top_mean(order) - mean, denominator, out=np.full(len(n), np.nan),
                     where=(n >= 20) & (denominator > 1e-12))


def bootstrap(predictions, replicates):
    comparisons = [('PublicHGB', 'Magnitude'), ('PublicHGB', 'PredictionHGB'),
                   ('PublicHGB', 'NegativeHistorySupport'), ('PublicRidge', 'Magnitude'),
                   ('PublicHGB', 'WeightedHistoryDistance'), ('PublicHGB', 'DirectHistoryDistance')]
    outputs = []
    for direction, group in predictions.groupby('direction'):
        index = ['original_fold', 'risk_fold', 'task_id', 'gene', 'context', 'heldout_context']
        wide = group.pivot(index=index, columns='method', values='risk').reset_index()
        truth = group.groupby(index).true_error_rmse.agg(['min', 'max', 'first']).reset_index()
        assert np.allclose(truth['min'], truth['max'])
        wide = wide.merge(truth[index + ['first']].rename(columns={'first': 'true_error_rmse'}))
        genes = sorted(wide.gene.unique())
        lookup = pd.Index(genes)
        counts = np.random.default_rng(SEED).multinomial(len(genes), np.ones(len(genes)) / len(genes), size=replicates)
        for a, b in comparisons:
            for scope in ['all_test', 'heldout_context']:
                selected = wide if scope == 'all_test' else wide[wide.context.eq(wide.heldout_context)]
                selected = selected[np.isfinite(selected[a]) & np.isfinite(selected[b])].copy()
                deltas = []
                points = []
                valid = 0
                for _, part in selected.groupby(['original_fold', 'risk_fold'], sort=True):
                    e = part.true_error_rmse.to_numpy(float)
                    ids = part.task_id.to_numpy(str)
                    w = counts[:, lookup.get_indexer(part.gene)]
                    delta = weighted_utility(e, part[a].to_numpy(), ids, w) - weighted_utility(e, part[b].to_numpy(), ids, w)
                    one = np.ones((1, len(part)), dtype=int)
                    point = weighted_utility(e, part[a].to_numpy(), ids, one)[0] - weighted_utility(e, part[b].to_numpy(), ids, one)[0]
                    if np.isfinite(point):
                        valid += 1
                    points.append(point)
                    deltas.append(delta)
                if not deltas:
                    continue
                draws = np.nanmean(deltas, axis=0)
                ci = np.nanquantile(draws, [.025, .975])
                outputs.append({'direction': direction, 'scope': scope, 'method': a, 'baseline': b,
                                'delta_utility20': float(np.nanmean(points)), 'bootstrap_mean_delta': float(np.nanmean(draws)),
                                'ci95_lower': float(ci[0]), 'ci95_upper': float(ci[1]),
                                'n_paired_row_uses': len(selected), 'n_paired_biological_tasks': selected.task_id.nunique(),
                                'n_paired_gene_clusters': selected.gene.nunique(), 'planned_strata': 15, 'valid_strata': valid,
                                'valid_bootstrap_draws': int(np.isfinite(draws).sum()), 'replicates': replicates,
                                'seed': SEED, 'sampling_unit': 'gene; synchronized across original checkpoints and model rows',
                                'history_comparator_coverage_restricted': b.endswith('HistoryDistance')})
                log(phase='bootstrap', direction=direction, method=a, baseline=b, scope=scope,
                    delta=outputs[-1]['delta_utility20'], lower=float(ci[0]), upper=float(ci[1]))
    save_csv('PAIRED_BOOTSTRAP.csv', outputs)
    return pd.DataFrame(outputs)


def run(replicates, report_only=False):
    started = time.monotonic()
    cpu_start = time.process_time()
    DOC.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'models').mkdir(exist_ok=True)
    if report_only:
        predictions = pd.read_parquet(RUN / 'TASK_PREDICTIONS.parquet')
        _, macro = summarize(predictions)
        intervals = bootstrap(predictions, replicates)
        return finish_report(predictions, macro, intervals, started, cpu_start, report_only=True)
    if (RUN / 'TASK_PREDICTIONS.parquet').exists():
        raise RuntimeError('existing completed predictions protected; use --report-only for report recovery')
    config = {'run_id': 'frangieh_cross_family_v1', 'data_role': 'SEEN_RETROSPECTIVE_RISK_DEVELOPMENT',
              'original_test_role_change': 'old upstream test reused as retrospective risk CV; never new confirmation',
              'native_gene_count': 512, 'output_contract_id': CONTRACT,
              'public_history': 'same-gene other-context original upstream TRAIN or VAL only; no TEST truths',
              'source_labels': 'source-model own RMSE on risk-train only; per-context midrank ECDF',
              'no_ensemble_scores': True, 'no_target_error_for_source_fit': True,
              'split': {'hash_prefix': SPLIT_PREFIX, 'input': 'original perturbation string including +ctrl',
                        'sort': 'SHA256 hex ascending', 'assignment': 'position modulo 5', 'seed': None},
              'risk_features': P + PUBLIC, 'prediction_features': P, 'fit_seed': SEED,
              'ridge_alpha': 10, 'hgb': {'max_iter': 200, 'learning_rate': .05, 'max_depth': 3,
                                        'min_samples_leaf': 20, 'l2_regularization': 10},
              'source_row_weighting': 'equal total weight per perturbation gene',
              'primary_strata': 'direction x original upstream fold x risk fold; all-test and heldout-context separate',
              'bootstrap': {'replicates': replicates, 'seed': SEED, 'unit': 'gene'},
              'expected_fits': 120, 'upstream_training': 0, 'download_bytes': 0,
              'threads': 4, 'pid': os.getpid(), 'started_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}
    write_json(DOC / 'CONFIG.json', config)
    write_json(DOC / 'RUN_STATUS.json', {'status': 'running', 'pid': os.getpid(), 'configuration_sha256': sha(DOC / 'CONFIG.json')})
    manifest = pd.read_csv(MANIFEST)
    labels = sorted(manifest.perturbation.unique(), key=lambda x: hashlib.sha256((SPLIT_PREFIX + x).encode()).hexdigest())
    assignment = {label: i % 5 for i, label in enumerate(labels)}
    assert len(assignment) == 189
    manifest['risk_fold'] = manifest.perturbation.map(assignment).astype(int)
    save_csv('RISK_SPLIT_MANIFEST.csv', manifest)
    bind_inputs(manifest)
    source_ledger, cdf_audit, eligibility, predictions_out = [], [], [], []
    competence_rows, competence_tasks, coverage, integrity = [], [], [], []
    for fold in sorted(manifest.fold_id.unique()):
        frame, truth, predictions = load_fold(fold, manifest)
        prior, public, edges = public_features(frame, truth)
        eligibility.extend(edges)
        cr, ct = competence(frame, truth, predictions, assignment, replicates)
        competence_rows.extend(cr)
        competence_tasks.extend(ct)
        features = {}
        for name, pred in predictions.items():
            check_contract(frame, pred, truth, prior, public)
            f = make_features(frame, pred, prior, public, name)
            # RMSE attaches only after feature construction. The fit selector below
            # reads the source branch; target errors are not passed to any learner.
            f['true_error_rmse'] = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
            features[name] = f
            f.to_parquet(RUN / f'FEATURES_{fold}_{name}.parquet', index=False)
        test_frame = frame[frame.split.eq('test')]
        for context, part in test_frame.groupby('context'):
            p = public.loc[part.index]
            coverage.append({'original_fold': fold, 'context': context, 'n_tasks': len(part),
                             'n_genes': part.gene.nunique(), 'n_history_available': int(p.history_available.sum()),
                             'coverage': float(p.history_available.mean()), 'history_sizes': json.dumps(p.n_history.value_counts().to_dict())})
        for source, target in [('scGPT', 'GEARS'), ('GEARS', 'scGPT')]:
            direction = source + '_to_' + target
            for risk_fold in range(5):
                fit = features[source][features[source].split.eq('test') & features[source].risk_fold.ne(risk_fold)].reset_index(drop=True)
                query = features[target][features[target].split.eq('test') & features[target].risk_fold.eq(risk_fold)].reset_index(drop=True)
                assert not (set(fit.gene) & set(query.gene))
                run_id = f'{fold}/{direction}/risk{risk_fold}'
                labels_y, audit = rank_labels(fit, run_id)
                cdf_audit.extend(audit)
                assert np.isfinite(labels_y).all()
                scores = {'Magnitude': query.predicted_magnitude.to_numpy(),
                          'NegativeHistorySupport': -query.log_history_support.to_numpy(),
                          'DirectHistoryDistance': query.prediction_prior_rmse.to_numpy(),
                          'WeightedHistoryDistance': query.weighted_history_distance.to_numpy()}
                for method, columns, kind in LEARNED:
                    t0 = time.monotonic()
                    model = fit_risk(fit, labels_y, columns, kind, seed=SEED, weighted=True)
                    values = model.predict(query)
                    dest = RUN / 'models' / f'{fold}_{direction}_risk{risk_fold}_{method}.joblib'
                    joblib.dump(model, dest)
                    restored = joblib.load(dest)
                    assert np.array_equal(values, restored.predict(query))
                    scores[method] = values
                    source_ledger.append({'run_id': run_id, 'method': method, 'source_upstream': source,
                                          'target_upstream': target, 'n_source_error_rows': len(fit),
                                          'n_source_error_genes': fit.gene.nunique(), 'n_target_evaluation_rows': len(query),
                                          'n_target_evaluation_genes': query.gene.nunique(),
                                          'source_label_records_hash': ids_hash(fit.task_id),
                                          'risk_eval_records_hash': ids_hash(query.task_id),
                                          'cdf_source_only': True, 'target_error_labels_used_for_fit': 0,
                                          'public_test_truth_uses': 0, 'n_features': len(columns),
                                          'fit_wall_seconds': time.monotonic() - t0, 'model_path': str(dest),
                                          'model_sha256': sha(dest), 'status': 'complete'})
                for method, score in scores.items():
                    keep = ['task_id', 'gene', 'context', 'heldout_context', 'risk_fold', 'setting',
                            'history_available', 'n_history', 'true_error_rmse']
                    part = query[keep].copy()
                    part['original_fold'] = fold
                    part['direction'] = direction
                    part['source_upstream'] = source
                    part['target_upstream'] = target
                    part['method'] = method
                    part['risk'] = score
                    predictions_out.append(part)
                integrity.append({'run_id': run_id, 'disjoint_genes': True, 'no_target_errors_for_fit': True,
                                  'history_from_original_train_val_only': True, 'native_gene_axis': 512,
                                  'distance_identity_passed': True, 'model_roundtrip_passed': True,
                                  'n_source_rows': len(fit), 'n_target_rows': len(query)})
                log(phase='risk_fold_completed', original_fold=fold, direction=direction, risk_fold=risk_fold,
                    source_rows=len(fit), target_rows=len(query),
                    target_utility={m: metrics(query, s)['utility20'] for m, s in scores.items()})
                save_csv('FIT_LEDGER.csv', source_ledger)
                save_csv('CDF_AUDIT.csv', cdf_audit)
    outputs = pd.concat(predictions_out, ignore_index=True)
    outputs.to_parquet(RUN / 'TASK_PREDICTIONS.parquet', index=False)
    outputs.to_csv(DOC / 'TASK_PREDICTIONS.csv.gz', index=False)
    save_csv('HISTORY_ELIGIBILITY.csv', eligibility)
    save_csv('HISTORY_COVERAGE.csv', coverage)
    save_csv('UPSTREAM_COMPETENCE.csv', competence_rows)
    save_csv('UPSTREAM_VALIDATION_TASKS.csv', competence_tasks)
    save_csv('INTEGRITY_CHECKS.csv', integrity)
    _, macro = summarize(outputs)
    intervals = bootstrap(outputs, replicates)
    return finish_report(outputs, macro, intervals, started, cpu_start)


def table(frame, columns):
    text = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    for _, row in frame.iterrows():
        vals = [f'{row[c]:.6f}' if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        text.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(text)


def score_range_diagnostic():
    """Inspect already fitted models, without changing scores or computing a new endpoint."""
    started = time.monotonic()
    ledger = pd.read_csv(DOC / 'FIT_LEDGER.csv')
    feature_cache, rows = {}, []
    for item in ledger.itertuples():
        fold, direction, risk = item.run_id.split('/')
        risk_fold = int(risk.removeprefix('risk'))
        needed = []
        for upstream in (item.source_upstream, item.target_upstream):
            key = (fold, upstream)
            if key not in feature_cache:
                feature_cache[key] = pd.read_parquet(RUN / f'FEATURES_{fold}_{upstream}.parquet')
            needed.append(feature_cache[key])
        fit = needed[0][needed[0].split.eq('test') & needed[0].risk_fold.ne(risk_fold)]
        query = needed[1][needed[1].split.eq('test') & needed[1].risk_fold.eq(risk_fold)]
        model = joblib.load(item.model_path)
        source_raw = model.predict(fit, clip=False)
        target_raw = model.predict(query, clip=False)
        rows.append({'run_id': item.run_id, 'method': item.method, 'direction': direction,
                     'source_raw_min': float(source_raw.min()), 'source_raw_max': float(source_raw.max()),
                     'target_raw_min': float(target_raw.min()), 'target_raw_max': float(target_raw.max()),
                     'target_fraction_clipped_low': float((target_raw < 0).mean()),
                     'target_fraction_clipped_high': float((target_raw > 1).mean()),
                     'target_clipped_unique_scores': len(np.unique(np.clip(target_raw, 0, 1))),
                     'source_prediction_magnitude_median': float(fit.predicted_magnitude.median()),
                     'target_prediction_magnitude_median': float(query.predicted_magnitude.median()),
                     'new_fits': 0, 'new_performance_comparison': False})
    save_csv('RISK_SCORE_RANGE_DIAGNOSTIC.csv', rows)
    summary = pd.DataFrame(rows).groupby(['direction', 'method'])[
        ['target_fraction_clipped_low', 'target_fraction_clipped_high', 'target_clipped_unique_scores']].mean().reset_index()
    save_csv('RISK_SCORE_RANGE_SUMMARY.csv', summary)
    write_json(DOC / 'SCORE_DIAGNOSTIC_COST.json', {'wall_seconds': time.monotonic() - started,
               'new_fits': 0, 'new_performance_comparison': False, 'purpose': 'locate constant Ridge scores without modifying fixed predictions'})
    return summary


def finish_report(predictions, macro, intervals, started, cpu_start, report_only=False):
    fit = pd.read_csv(DOC / 'FIT_LEDGER.csv')
    selected = macro[macro.scope.isin(['all_test', 'heldout_context']) & macro.aggregation.eq('risk_fold')]
    result = {
        'status': 'complete', 'pid': os.getpid(), 'report_only_recovery': report_only,
        'new_fits_this_invocation': 0 if report_only else len(fit), 'total_risk_fits': len(fit),
        'n_prediction_rows': len(predictions), 'unique_biological_tasks': predictions.task_id.nunique(),
        'unique_gene_clusters': predictions.gene.nunique(), 'native_genes': 512,
        'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu_start,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'new_upstream_training': 0, 'gpu_hours': 0, 'download_bytes': 0,
        'result_path': str(RUN / 'TASK_PREDICTIONS.parquet'),
        'result_sha256': sha(RUN / 'TASK_PREDICTIONS.parquet'),
        'finished_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'evidence_role': 'retrospective SEEN native-axis cross-family risk transfer; not new confirmation',
    }
    write_json(DOC / 'RUN_STATUS.json', result)
    write_json(DOC / 'COST.json', result)
    readme = '''# Frangieh native512: retrospective scGPT ↔ GEARS risk transfer

This is a new **SEEN retrospective** analysis of already-opened upstream prediction caches. Old upstream TEST is re-registered as risk-development data, with five gene-grouped risk folds inside each original checkpoint fold. It does not inherit the original confirmation claim. Both directions use the same hash assignment; each source learner receives only that source model's own errors and prediction features. Target errors are evaluation-only for that direction. No E108 ensemble-error/disagreement risk score is reused.

## Contract and interpretation

- Original folds remain separate. 279 original TEST rows / 189 genes per fold; all three upstream folds overlap biologically and are synchronized in gene bootstrap.
- Public history contains only original upstream TRAIN/VAL experiments of the same perturbation in another context. No old TEST truth can become another query's history. Public is support weighted, requires no learner/CDF; quality is unavailable and is not invented from cell counts.
- Native512 is a separate within-study output contract. No common-axis zero filling, no claim of unchanged Source3285 parameter transfer.
- Same task truth is shared across the two model caches and checked numerically. Public features are built before realized errors attach. Input/feature identity fields never enter regressors.
- Training uses equal total weight per gene and a source-only, context-specific training ECDF; predictions are clipped to [0,1] following the existing rank-risk implementation. No target error rank is needed at inference.
- Source errors are from original upstream TEST, unused for upstream fit/early stopping. However most target genes were upstream-train-exposed in another context: this is held-out task plus unseen risk-training gene transfer, not all-genes-unseen upstream prediction.
- One fixed training seed; no hyperparameter search. All 120 model fits are saved. Three original folds are not three independent studies.
- all_test and heldout_context primary summaries macro-average original-fold × risk-fold strata. Other context results are additionally pooled over OOF risk predictions within an original fold. n<20 makes U20 NA.
- Direct history baselines are NA without history. Their pairwise bootstrap uses the common covered subset; compare coverage before interpreting their utility next to all-task scores.
- Native upstream competence uses the best of zero, TRAIN-global mean and TRAIN-context mean, selected using TRAIN gene-OOF, then evaluated on original validation. The validation was previously used for neural early stopping; it is not independent confirmation. All models remain in the report even if they fail competence.

## Main risk results

''' + table(selected, ['direction', 'scope', 'method', 'utility20', 'spearman', 'aurc', 'valid_strata', 'planned_strata'])
    readme += '\n\n## Paired gene bootstrap (5000 draws)\n\n' + table(intervals, ['direction', 'scope', 'method', 'baseline', 'delta_utility20', 'ci95_lower', 'ci95_upper', 'n_paired_gene_clusters'])
    score_range_diagnostic()
    competence_frame = pd.read_csv(DOC / 'UPSTREAM_COMPETENCE.csv')
    readme += '\n\n## Upstream competence and failure localization\n\n' + table(competence_frame,
        ['original_fold', 'upstream', 'relative_macro_rmse_gap', 'ci95_lower', 'ci95_upper', 'competence_pass'])
    readme += '''\n\nAll six native upstream/fold assets failed the predeclared competence rule against the TRAIN-OOF-selected context-mean baseline. They remain a native-axis cross-family stress/sensitivity experiment; improved SafeConf ranking does not promote an ineligible predictor into primary validation.

The source-error HGB improves over prediction-only and magnitude, but the relevant comparison to weighted historical distance is paired on history-covered tasks. Neither direction establishes superiority to this stronger simple rule. Public information and a complicated risk learner therefore have different evidence status.

RISK_SCORE_RANGE_DIAGNOSTIC inspects saved regressors only. In scGPT→GEARS, both Ridge variants produce constant clipped scores in all risk folds: unbounded outputs fall entirely below zero or above one. This localizes a transfer/clip boundary without changing the frozen method, adding unclipped performance comparisons, or retraining.
'''
    readme += '\n\n## Reproduction\n\n`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/scripts/run_safeconf_frangieh_cross_family_v1.py`\n\nExisting completed predictions are protected from rerunning; `--report-only` recalculates statistics without fitting. Models and feature matrices are in the runtime directory recorded in CONFIG/ledger; compressed per-task predictions and all provenance/statistics are attached here.\n'
    (DOC / 'README.md').write_text(readme)
    failure = DOC / 'FAILURE_AND_REPAIR_LOG.md'
    if not failure.exists():
        failure.write_text('# Failure and repair log\n\nNo technical failure or outcome-driven repair occurred in this execution. No method hyperparameter was changed.\n')
    log(phase='complete', **result)


def run_score_repair(replicates=5000):
    """One registered SEEN output repair. No fit, feature or primary-result edits."""
    started, cpu_start = time.monotonic(), time.process_time()
    report = DOC / 'RidgeScoreRepair'
    runtime = RUN / 'RidgeScoreRepair'
    if (runtime / 'TASK_PREDICTIONS.parquet').exists():
        raise RuntimeError('completed score-repair predictions protected')
    report.mkdir(exist_ok=True)
    runtime.mkdir(exist_ok=True)
    ledger = pd.read_csv(DOC / 'FIT_LEDGER.csv')
    protected = [RUN / 'TASK_PREDICTIONS.parquet', DOC / 'TASK_PREDICTIONS.csv.gz',
                 DOC / 'MACRO_RESULTS.csv', DOC / 'STRATUM_RESULTS.csv', DOC / 'PAIRED_BOOTSTRAP.csv',
                 DOC / 'INPUT_HASHES.csv', DOC / 'CONFIG.json', DOC / 'FIT_LEDGER.csv']
    protected += [Path(p) for p in ledger.model_path]
    before = {str(p): sha(p) for p in protected}
    write_json(report / 'CONFIG.json', {
        'version': 'frangieh_native512_RidgeScoreRepair_v1',
        'data_role': 'SEEN_post_primary_technical_output_repair',
        'formula': '0.5 + arctan(raw_ridge_score - 0.5) / pi',
        'parameter_search': False, 'new_fits': 0,
        'upstream_competence_unchanged': True, 'is_calibrated_percentile': False,
        'hypothesis': 'hard clipping destroyed the ordering of extrapolated Ridge scores',
        'model_scope': 'PredictionRidge and PublicRidge; both transfer directions; all original/risk folds',
        'evaluation': 'same target tasks and errors; fixed 5000 synchronized gene draws',
        'seed': SEED, 'replicates': replicates, 'pid': os.getpid(),
        'started_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
    })
    write_json(report / 'RUN_STATUS.json', {'status': 'running', 'pid': os.getpid(), 'new_fits': 0})
    original = pd.read_parquet(RUN / 'TASK_PREDICTIONS.parquet')
    feature_cache, repair_parts, rows = {}, [], []
    for item in ledger[ledger.method.isin(['PredictionRidge', 'PublicRidge'])].itertuples():
        fold, direction, risk = item.run_id.split('/')
        risk_fold = int(risk.removeprefix('risk'))
        needed = []
        for upstream in [item.source_upstream, item.target_upstream]:
            key = (fold, upstream)
            if key not in feature_cache:
                feature_cache[key] = pd.read_parquet(RUN / f'FEATURES_{fold}_{upstream}.parquet')
            needed.append(feature_cache[key])
        source = needed[0][needed[0].split.eq('test') & needed[0].risk_fold.ne(risk_fold)].reset_index(drop=True)
        target = needed[1][needed[1].split.eq('test') & needed[1].risk_fold.eq(risk_fold)].reset_index(drop=True)
        model = joblib.load(item.model_path)
        raw_source, raw_target = model.predict(source, clip=False), model.predict(target, clip=False)
        repaired_source = .5 + np.arctan(raw_source - .5) / np.pi
        repaired_target = .5 + np.arctan(raw_target - .5) / np.pi
        assert np.isfinite(repaired_source).all() and np.isfinite(repaired_target).all()
        assert ((repaired_target > 0) & (repaired_target < 1)).all()
        ids = target.task_id.to_numpy(str)
        assert np.array_equal(np.lexsort((ids, -raw_target)), np.lexsort((ids, -repaired_target)))
        part = original[original.original_fold.eq(fold) & original.direction.eq(direction)
                        & original.risk_fold.eq(risk_fold) & original.method.eq(item.method)].copy()
        assert set(part.task_id) == set(target.task_id)
        raw_map, repaired_map = dict(zip(ids, raw_target)), dict(zip(ids, repaired_target))
        assert np.allclose(part.risk, np.clip(part.task_id.map(raw_map).to_numpy(), 0, 1), rtol=0, atol=0)
        part['original_clipped_risk'] = part.risk
        part['raw_ridge_score'] = part.task_id.map(raw_map)
        part['risk'] = part.task_id.map(repaired_map)
        part['method'] = item.method + '_ArctanRepair'
        repair_parts.append(part)
        rows.append({'run_id': item.run_id, 'method': item.method, 'direction': direction,
                     'model_path': item.model_path, 'model_sha256': sha(item.model_path),
                     'source_raw_min': float(raw_source.min()), 'source_raw_max': float(raw_source.max()),
                     'target_raw_min': float(raw_target.min()), 'target_raw_max': float(raw_target.max()),
                     'source_bounded_min': float(repaired_source.min()), 'source_bounded_max': float(repaired_source.max()),
                     'target_bounded_min': float(repaired_target.min()), 'target_bounded_max': float(repaired_target.max()),
                     'target_original_unique_scores': len(np.unique(np.clip(raw_target, 0, 1))),
                     'target_repaired_unique_scores': len(np.unique(repaired_target)),
                     'target_raw_order_preserved': True, 'new_fits': 0})
    repaired = pd.concat(repair_parts, ignore_index=True)
    output = pd.concat([original, repaired], ignore_index=True)
    output.to_parquet(runtime / 'TASK_PREDICTIONS.parquet', index=False)
    output.to_csv(report / 'TASK_PREDICTIONS.csv.gz', index=False)
    pd.DataFrame(rows).to_csv(report / 'SCORE_REPAIR_LEDGER.csv', index=False)
    # Reuse the exact existing summarizer, redirecting only its output directory.
    saved_doc = globals()['DOC']
    try:
        globals()['DOC'] = report
        _, macro = summarize(output)
    finally:
        globals()['DOC'] = saved_doc
    comparisons = [(m + '_ArctanRepair', b) for m in ['PredictionRidge', 'PublicRidge']
                   for b in [m, 'Magnitude', 'WeightedHistoryDistance']]
    intervals = []
    for direction, group in output.groupby('direction'):
        keys = ['original_fold', 'risk_fold', 'task_id', 'gene', 'context', 'heldout_context']
        wide = group.pivot(index=keys, columns='method', values='risk').reset_index()
        truth = group.groupby(keys).true_error_rmse.agg(['min', 'max', 'first']).reset_index()
        assert np.allclose(truth['min'], truth['max'])
        wide = wide.merge(truth[keys + ['first']].rename(columns={'first': 'true_error_rmse'}))
        genes = sorted(wide.gene.unique())
        lookup = pd.Index(genes)
        counts = np.random.default_rng(SEED).multinomial(len(genes), np.ones(len(genes)) / len(genes), size=replicates)
        for a, b in comparisons:
            for scope in ['all_test', 'heldout_context']:
                part = wide if scope == 'all_test' else wide[wide.context.eq(wide.heldout_context)]
                part = part[np.isfinite(part[a]) & np.isfinite(part[b])]
                draws, points = [], []
                for _, s in part.groupby(['original_fold', 'risk_fold']):
                    e, ids = s.true_error_rmse.to_numpy(), s.task_id.to_numpy(str)
                    weights = counts[:, lookup.get_indexer(s.gene)]
                    draws.append(weighted_utility(e, s[a].to_numpy(), ids, weights)
                                 - weighted_utility(e, s[b].to_numpy(), ids, weights))
                    one = np.ones((1, len(s)), dtype=int)
                    points.append(weighted_utility(e, s[a].to_numpy(), ids, one)[0]
                                  - weighted_utility(e, s[b].to_numpy(), ids, one)[0])
                draw = np.nanmean(draws, axis=0)
                low, high = np.nanquantile(draw, [.025, .975])
                intervals.append({'direction': direction, 'scope': scope, 'method': a, 'baseline': b,
                                  'delta_utility20': float(np.nanmean(points)),
                                  'bootstrap_mean_delta': float(np.nanmean(draw)),
                                  'ci95_lower': float(low), 'ci95_upper': float(high),
                                  'replicates': replicates, 'seed': SEED,
                                  'n_paired_row_uses': len(part), 'n_paired_gene_clusters': part.gene.nunique(),
                                  'valid_strata': int(np.isfinite(points).sum()), 'planned_strata': 15,
                                  'history_coverage_restricted': b == 'WeightedHistoryDistance'})
                log(phase='score_repair_bootstrap', **intervals[-1])
    intervals = pd.DataFrame(intervals)
    intervals.to_csv(report / 'PAIRED_BOOTSTRAP.csv', index=False)
    after = {p: sha(p) for p in before}
    assert before == after
    write_json(report / 'PROTECTED_ORIGINAL_HASHES.json', {'before': before, 'after_match': True})
    selected = macro[macro.scope.isin(['all_test', 'heldout_context']) & macro.aggregation.eq('risk_fold')
                     & macro.method.isin(['PredictionRidge', 'PublicRidge', 'PredictionRidge_ArctanRepair',
                                         'PublicRidge_ArctanRepair', 'Magnitude', 'WeightedHistoryDistance'])]
    summary = '''# One fixed Ridge output repair: SEEN retrospective analysis

## Why and what changed

Saved-model diagnostics found that hard clipping maps every target score to the same endpoint in all 15 scGPT→GEARS folds. The registered repair applies the same fixed, strictly increasing map to both Ridge variants and both model directions:

`bounded_score = 0.5 + arctan(raw_ridge_score - 0.5) / pi`

This retains the Ridge ordering while bounding scores. It is **not** percentile calibration and does not correct distribution shift. No parameter, feature, target, split, prediction task, model, or hyperparameter was refitted/selected. Source-side outputs use the same map for audit. Original 120 model files, original predictions, primary tables and input hashes are verified unchanged.

This is one post-primary **SEEN technical score-repair** version, not external confirmation. All six upstream assets still fail the same competence gate. The repair must not upgrade their qualification.

## Same-task results

''' + table(selected, ['direction', 'scope', 'method', 'utility20', 'spearman', 'aurc'])
    summary += '\n\n## 5000 paired gene-bootstrap comparisons\n\n' + table(intervals,
        ['direction', 'scope', 'method', 'baseline', 'delta_utility20', 'ci95_lower', 'ci95_upper'])
    summary += '''\n\n## Remaining boundary and stopping rule

The map can only recover ordering lost by clipping; it cannot make incorrectly ordered raw extrapolations correct, align conditional source/target error distributions, or calibrate the target model's risk percentiles. A ranking benefit over clipped output alone does not establish superiority to Magnitude or WeightedHistoryDistance. History comparisons use exactly the common covered subset. No additional map, temperature, target or calibration will be searched in this branch.

## Reproduce

`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/scripts/run_safeconf_frangieh_cross_family_v1.py --score-repair`

This command requires the saved 120-fit primary run, reads only its fitted models/features, and refuses to overwrite completed repair predictions. New model fits: **0**.
'''
    (report / 'README.md').write_text(summary)
    result = {'status': 'complete', 'pid': os.getpid(), 'new_fits': 0, 'saved_ridge_models_read': len(rows),
              'new_prediction_rows': len(repaired), 'primary_120_models_and_tables_unchanged': True,
              'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu_start,
              'gpu_hours': 0, 'download_bytes': 0, 'result_sha256': sha(runtime / 'TASK_PREDICTIONS.parquet'),
              'finished_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}
    write_json(report / 'RUN_STATUS.json', result)
    write_json(report / 'COST.json', result)
    log(phase='score_repair_complete', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap-replicates', type=int, default=5000)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--score-repair', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        try:
            if args.score_repair:
                run_score_repair(args.bootstrap_replicates)
            else:
                run(args.bootstrap_replicates, args.report_only)
        except Exception as exc:
            error_doc = DOC / 'RidgeScoreRepair' if args.score_repair else DOC
            error_doc.mkdir(parents=True, exist_ok=True)
            failure = {'status': 'failed', 'pid': os.getpid(), 'exception': repr(exc), 'traceback': traceback.format_exc()}
            write_json(error_doc / 'RUN_STATUS.json', failure)
            with open(error_doc / 'FAILURE_AND_REPAIR_LOG.md', 'a') as f:
                f.write('\n## Technical failure\n\n```\n' + failure['traceback'] + '\n```\n')
            raise


if __name__ == '__main__':
    main()
