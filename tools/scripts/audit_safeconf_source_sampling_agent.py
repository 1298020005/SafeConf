#!/usr/bin/env python3
"""DEV/SEEN source truth and cell sampling audit; never changes primary assets.

Truth is the released, unmasked E201 cell matrix, independently spot checked
against the official sparse H5AD. Sampling is conditional on frozen controls.
Cell iid, independent batch clusters, and a finite observed cell population
are separate assumptions, not interchangeable biological replicate designs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import metrics

BASE = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001'
OUT = BASE / 'measurement_audit_agent'
COMMON = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
DATA = Path('/home/yyf/data')
TX = DATA / 'txpert_official_20260802'
OFFICIAL = TX / 'cache/K562_cross_cell_lines/de_adata_test.h5ad'
E201 = ROOT / 'docs/实验结果/E201_txpert_multitarget_retraining_20260802'
SEEDS = [2026100101, 2026100102, 2026100103, 2026100104, 2026100105]
ROLE = 'DEV_SEEN_SOURCE_CELL_SAMPLING_DIAGNOSTIC'


def h5col(group, key):
    x = group[key]
    if isinstance(x, h5py.Group):
        categories = h5col(x, 'categories')
        codes = np.asarray(x['codes'])
        assert (codes >= 0).all()
        return categories[codes]
    raw = np.asarray(x)
    return np.asarray([v.decode() if isinstance(v, bytes) else v for v in raw]) if raw.dtype.kind in 'OS' else raw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_add(acc, codes, values):
    unique, inverse = np.unique(codes, return_inverse=True)
    reducer = sparse.csr_matrix((np.ones(len(codes)), (inverse, np.arange(len(codes)))),
                               shape=(len(unique), len(codes)))
    acc[unique] += reducer @ values


def correlation(a, b):
    return float(spearmanr(a, b).statistic)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    common_tasks = pd.read_csv(COMMON / 'SOURCE_TASKS.csv')
    full_tasks = pd.read_csv(E201 / 'tables/E201_PRETRUTH_TASK_BASE.csv').set_index('task_id')
    tasks = full_tasks.loc[common_tasks.task_id].reset_index()
    assert len(tasks) == 1808 and tasks.analysis_stratum.eq('primary_ge30').all()
    assert tasks.task_id.tolist() == common_tasks.task_id.tolist()
    genes = json.loads((COMMON / 'GENE_IDS.json').read_text())
    native_genes = json.loads((DATA / 'safeconf_dual_memory_20260929/public_txpert_e201/gene_ids.json').read_text())['gene_ids']
    columns = np.asarray([native_genes.index(g) for g in genes])
    g = len(genes)
    assert g == 2840 and len(native_genes) == 3352
    rows = tasks.source_mean_delta_row.to_numpy(int)
    controls = np.load(TX / 'e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy', mmap_mode='r')[rows][:, columns].astype(float)
    frozen_centroids = np.load(TX / 'e201/evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy', mmap_mode='r')[rows][:, columns]
    frozen_effects = np.load(COMMON / 'SOURCE_TRUE_EFFECTS.npy')
    predictions = {model: np.load(COMMON / f'SOURCE_{model}_PREDICTED_EFFECTS.npy')
                   for model in ['TxPert_GAT', 'TxPert_Exphormer']}
    release = json.loads((E201 / 'E201_TARGET_TRUTH_RELEASE_STATUS.json').read_text())
    n = len(tasks)
    means = np.zeros((n, g))
    sample_means = np.zeros((len(SEEDS), n, g))
    iid_variances = np.zeros(n)
    batch_variances = np.zeros(n)
    audits, sample_manifest, task_stats = [], [], []

    with h5py.File(OFFICIAL, 'r') as source:
        obs_native = source['obs']
        native_context = h5col(obs_native, 'cell_line').astype(str)
        native_condition = h5col(obs_native, 'condition_name').astype(str)
        native_batch = h5col(obs_native, 'batch').astype(str)
        native_control = h5col(obs_native, 'control').astype(bool)
        axis = h5col(source['var'], source['var'].attrs['_index']).astype(str)
        assert axis.tolist() == native_genes
        axis_sha = hashlib.sha256('\n'.join(axis).encode()).hexdigest()
        guide_fields = [str(k) for k in obs_native if 'guide' in str(k).lower() or 'sgrna' in str(k).lower()]
        indptr = np.asarray(source['X']['indptr'])
        for record in release['records']:
            context = record['target']
            path = DATA / record['truth_path'].removeprefix('DATA/')
            manifest_path = DATA / record['truth_manifest_path'].removeprefix('DATA/')
            manifest = json.loads(manifest_path.read_text())
            obs = pd.read_csv(path.parent / 'observations.csv', keep_default_na=False)
            truth = np.load(path, mmap_mode='r')
            assert truth.shape == (record['n_samples'], 3352) and truth.dtype == np.float32 and truth.flags.c_contiguous
            assert path.stat().st_size == record['truth_bytes']
            assert sha(manifest_path) == record['truth_manifest_sha256']
            assert manifest['status'] == 'SEALED_TARGET_TRUTH' and manifest['var_order_sha256'] == axis_sha
            assert obs.row_index.tolist() == list(range(len(obs)))
            native_rows = np.flatnonzero((native_context == context) & ~native_control & np.isin(native_condition, obs.pert_cond_name.unique()))
            assert len(native_rows) == len(obs)
            assert np.array_equal(native_condition[native_rows], obs.pert_cond_name.astype(str))
            assert np.array_equal(native_context[native_rows], obs.cell_type.astype(str))
            assert np.array_equal(native_batch[native_rows], obs.experimental_batch.astype(str))
            # Independent deterministic sparse raw-cell checks, all 3352 genes.
            spot = np.linspace(0, len(obs) - 1, 32, dtype=int)
            raw_residual = 0.0
            for local in spot:
                native = native_rows[local]
                start, stop = int(indptr[native]), int(indptr[native + 1])
                v = np.zeros(3352, np.float32)
                v[np.asarray(source['X']['indices'][start:stop])] = source['X']['data'][start:stop]
                raw_residual = max(raw_residual, float(np.max(np.abs(v - truth[local]))))
            assert raw_residual == 0
            target_rows = np.flatnonzero(tasks.target.eq(context))
            local_tasks = tasks.iloc[target_rows].reset_index(drop=True)
            labels = obs.pert_cond_name.astype(str).str.removeprefix(context + '_').str.removesuffix('_1+1')
            lookup = {c: i for i, c in enumerate(local_tasks.condition)}
            codes = labels.map(lookup).fillna(-1).to_numpy(int)
            primary_rows = np.flatnonzero(codes >= 0)
            counts = np.bincount(codes[primary_rows], minlength=len(local_tasks))
            assert np.array_equal(counts, local_tasks.n_target_cells)
            pairs = pd.MultiIndex.from_arrays([codes[primary_rows], obs.experimental_batch.iloc[primary_rows].astype(str)])
            batch_codes, batch_keys = pd.factorize(pairs, sort=True)
            row_batch_codes = np.full(len(obs), -1, int)
            row_batch_codes[primary_rows] = batch_codes
            batch_task = np.asarray([k[0] for k in batch_keys], int)
            batch_counts = np.bincount(batch_codes)
            sum_values = np.zeros((len(local_tasks), g))
            squared_trace = np.zeros(len(local_tasks))
            batch_sums = np.zeros((len(batch_keys), g))
            chosen = np.zeros((len(SEEDS), len(obs)), bool)
            for task_code, task in local_tasks.iterrows():
                cell_rows = np.flatnonzero(codes == task_code)
                for rep, seed in enumerate(SEEDS):
                    token = f'SafeConfSourceSamplingAgent::{seed}::{task.task_id}'
                    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big'))
                    selected = np.sort(rng.choice(cell_rows, 20, replace=False))
                    chosen[rep, selected] = True
                    sample_manifest.append({'task_id': task.task_id, 'seed': seed,
                                            'release_row_indices': ' '.join(map(str, selected))})
            full_digest = hashlib.sha256()
            stream_digest = hashlib.sha256()
            with path.open('rb') as file:
                full_digest.update(file.read(truth.offset))
            for start in range(0, len(obs), 1024):
                stop = min(start + 1024, len(obs))
                block = np.ascontiguousarray(truth[start:stop])
                payload = block.tobytes()
                full_digest.update(payload)
                stream_digest.update(payload)
                use = np.flatnonzero(codes[start:stop] >= 0)
                if not len(use):
                    continue
                global_use = start + use
                values = block[use][:, columns].astype(float)
                task_codes = codes[global_use]
                assert np.isfinite(values).all()
                grouped_add(sum_values, task_codes, values)
                grouped_add(batch_sums, row_batch_codes[global_use], values)
                squared_trace += np.bincount(task_codes, weights=np.einsum('ij,ij->i', values, values) / g,
                                             minlength=len(local_tasks))
                for rep in range(len(SEEDS)):
                    selected = chosen[rep, global_use]
                    if selected.any():
                        grouped_add(sample_means[rep], target_rows[task_codes[selected]], values[selected])
            assert full_digest.hexdigest() == record['truth_sha256']
            assert stream_digest.hexdigest() == manifest['truth_stream_sha256']
            mean = sum_values / counts[:, None]
            trace_variance = (squared_trace - np.einsum('ij,ij->i', sum_values, sum_values) / g / counts) / (counts - 1)
            assert trace_variance.min() > -1e-12
            trace_variance = np.maximum(trace_variance, 0)
            iid = trace_variance / counts
            batch_residuals = batch_sums - batch_counts[:, None] * mean[batch_task]
            batch_count_by_task = np.bincount(batch_task, minlength=len(local_tasks))
            cluster = np.bincount(batch_task, weights=np.einsum('ij,ij->i', batch_residuals, batch_residuals) / g,
                                  minlength=len(local_tasks)) / counts**2
            cluster *= batch_count_by_task / (batch_count_by_task - 1)
            means[target_rows] = mean
            iid_variances[target_rows] = iid
            batch_variances[target_rows] = cluster
            center_residual = np.max(np.abs(mean.astype(np.float32) - frozen_centroids[target_rows]))
            # The frozen effect intentionally starts with an already float32
            # centroid. Preserve this two-stage rounding in the identity check.
            effect_residual = np.max(np.abs((mean.astype(np.float32).astype(float) - controls[target_rows]).astype(np.float32) - frozen_effects[target_rows]))
            direct_effect_residual = np.max(np.abs((mean - controls[target_rows]).astype(np.float32) - frozen_effects[target_rows]))
            assert center_residual == 0 and effect_residual == 0, (context, center_residual, effect_residual)
            audits.append({'context': context, 'release_rows': len(obs), 'primary_cells': int(counts.sum()),
                           'primary_tasks': len(local_tasks), 'raw_spot_cells_all3352genes': len(spot),
                           'raw_spot_max_abs_residual': raw_residual, 'all_metadata_alignment': True,
                           'truth_file_sha256_verified': full_digest.hexdigest(),
                           'truth_stream_sha256_verified': stream_digest.hexdigest(),
                           'centroid_max_abs_residual_float32': float(center_residual),
                           'common_effect_max_abs_residual_float32': float(effect_residual),
                           'direct_unrounded_mean_effect_max_abs_difference': float(direct_effect_residual)})
            for i, task in local_tasks.iterrows():
                task_stats.append({'task_id': task.task_id, 'target': context, 'gene': task.gene,
                                   'n_target_cells': int(counts[i]), 'n_target_batches': int(batch_count_by_task[i]),
                                   'within_cell_variance_trace_per_gene': trace_variance[i],
                                   'iid_centroid_sampling_mse': iid[i],
                                   'independent_batch_cluster_centroid_mse': cluster[i],
                                   'fixed20_finite_population_expected_mse': trace_variance[i] * (1 / 20 - 1 / counts[i])})
            del batch_sums, batch_residuals
            print(json.dumps({'context': context, 'status': 'truth_moments_verified', **audits[-1]}), flush=True)

    sample_means /= 20
    truth_effects = frozen_effects.astype(float)
    table = pd.DataFrame(task_stats).set_index('task_id').loc[tasks.task_id].reset_index()
    sampling_observed_mse = np.mean((sample_means - means[None])**2, axis=2)
    table['fixed20_observed_mse_mean5'] = sampling_observed_mse.mean(axis=0)
    table['role'] = ROLE
    table.to_csv(OUT / 'SOURCE_CELL_MOMENTS.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'TRUTH_RECOMPUTATION_AUDIT.csv', index=False)
    pd.DataFrame(sample_manifest).to_csv(OUT / 'FIXED20_SELECTED_RELEASE_ROWS.csv.gz', index=False)
    error_rows, errors, summaries = [], {}, []
    for model, pred in predictions.items():
        mse = np.mean((pred - truth_effects)**2, axis=1)
        samples = np.sqrt(np.mean((pred[None] - (sample_means - controls[None]))**2, axis=2))
        variants = {'full_observed': np.sqrt(mse),
                    'iid_mse_subtracted_clipped': np.sqrt(np.maximum(mse - iid_variances, 0)),
                    'batch_cluster_mse_subtracted_clipped': np.sqrt(np.maximum(mse - batch_variances, 0))}
        variants.update({f'fixed20_seed{seed}': samples[rep] for rep, seed in enumerate(SEEDS)})
        errors[model] = variants
        frame = table.copy()
        frame['upstream'] = model
        frame['observed_mse'] = mse
        frame['iid_fraction_of_observed_mse'] = iid_variances / mse
        for mode, e in variants.items():
            frame[mode] = e
        error_rows.append(frame)
        for context, indices in tasks.groupby('target', sort=True).groups.items():
            ix = np.asarray(indices)
            summaries.append({'upstream': model, 'context': context, 'n_tasks': len(ix),
                'observed_mean_mse': float(mse[ix].mean()), 'iid_noise_mean_mse': float(iid_variances[ix].mean()),
                'iid_noise_fraction_of_mean_mse': float(iid_variances[ix].mean() / mse[ix].mean()),
                'batch_cluster_noise_mean_mse': float(batch_variances[ix].mean()),
                'batch_cluster_noise_fraction_of_mean_mse': float(batch_variances[ix].mean() / mse[ix].mean()),
                'iid_subtraction_nonpositive_tasks': int((mse[ix] <= iid_variances[ix]).sum()),
                'batch_subtraction_nonpositive_tasks': int((mse[ix] <= batch_variances[ix]).sum()),
                'full_error_rho_cell_count': correlation(np.sqrt(mse[ix]), tasks.n_target_cells.iloc[ix]),
                'iid_adjusted_error_rho_cell_count': correlation(variants['iid_mse_subtracted_clipped'][ix], tasks.n_target_cells.iloc[ix]),
                'batch_adjusted_error_rho_cell_count': correlation(variants['batch_cluster_mse_subtracted_clipped'][ix], tasks.n_target_cells.iloc[ix]),
                'fixed20_observed_vs_expected_sampling_mse_ratio': float(sampling_observed_mse[:, ix].mean() / table.fixed20_finite_population_expected_mse.iloc[ix].mean()),
                'fixed20_mean_error_rho_cell_count': float(np.mean([correlation(x[ix], tasks.n_target_cells.iloc[ix]) for x in samples])),
                'role': ROLE})
    pd.concat(error_rows, ignore_index=True).to_csv(OUT / 'SOURCE_ERROR_NOISE_DIAGNOSTIC.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'SOURCE_CONTEXT_NOISE_SUMMARY.csv', index=False)
    finish_from_existing()


def finish_from_existing():
    """Complete report from saved moments without rereading cell matrices."""
    tasks = pd.read_csv(OUT / 'SOURCE_CELL_MOMENTS.csv')
    assert len(tasks) == 1808 and tasks.task_id.nunique() == 1808
    n, g = len(tasks), 2840
    audits = pd.read_csv(OUT / 'TRUTH_RECOMPUTATION_AUDIT.csv').to_dict('records')
    assert len(audits) == 4 and all(a['centroid_max_abs_residual_float32'] == 0 and
        a['common_effect_max_abs_residual_float32'] == 0 and a['raw_spot_max_abs_residual'] == 0 for a in audits)
    with h5py.File(OFFICIAL) as source:
        guide_fields = [str(k) for k in source['obs'] if 'guide' in str(k).lower() or 'sgrna' in str(k).lower()]
    predictions = {model: np.load(COMMON / f'SOURCE_{model}_PREDICTED_EFFECTS.npy')
                   for model in ['TxPert_GAT', 'TxPert_Exphormer']}
    frozen_effects = np.load(COMMON / 'SOURCE_TRUE_EFFECTS.npy')
    diagnostic = pd.read_csv(OUT / 'SOURCE_ERROR_NOISE_DIAGNOSTIC.csv')
    summaries = pd.read_csv(OUT / 'SOURCE_CONTEXT_NOISE_SUMMARY.csv')
    errors, precision = {}, {}
    variant_columns = ['full_observed', 'iid_mse_subtracted_clipped',
                       'batch_cluster_mse_subtracted_clipped'] + [f'fixed20_seed{s}' for s in SEEDS]
    for model, pred in predictions.items():
        use = diagnostic.upstream.eq(model)
        part = diagnostic.loc[use].set_index('task_id').loc[tasks.task_id]
        # Match the frozen primary's original float32 reduction exactly;
        # the moment/MSE diagnostics deliberately use float64 arithmetic.
        frozen_rmse = np.sqrt(np.mean((pred - frozen_effects)**2, axis=1))
        rmse64 = np.sqrt(part.observed_mse.to_numpy())
        precision[model] = float(np.max(np.abs(frozen_rmse - rmse64)))
        diagnostic.loc[use, 'observed_rmse_float64'] = np.sqrt(diagnostic.loc[use, 'observed_mse'])
        diagnostic.loc[use, 'full_observed'] = diagnostic.loc[use, 'task_id'].map(dict(zip(tasks.task_id, frozen_rmse)))
        errors[model] = {c: (frozen_rmse.astype(float) if c == 'full_observed' else part[c].to_numpy())
                         for c in variant_columns}
        for context, indices in tasks.groupby('target', sort=True).groups.items():
            ix = np.asarray(indices)
            summaries.loc[summaries.upstream.eq(model) & summaries.context.eq(context), 'full_error_rho_cell_count'] = correlation(frozen_rmse[ix], tasks.n_target_cells.iloc[ix])
    diagnostic.to_csv(OUT / 'SOURCE_ERROR_NOISE_DIAGNOSTIC.csv', index=False)
    summaries.to_csv(OUT / 'SOURCE_CONTEXT_NOISE_SUMMARY.csv', index=False)
    frozen_scores = pd.concat([pd.read_csv(BASE / 'common_gene_axis/results' / p) for p in
        ['MATRIX_TASK_PREDICTIONS.csv.gz', 'SUPPORT_CONTROL_TASK_PREDICTIONS.csv.gz']], ignore_index=True)
    methods = ['Manual_WeightedHistoryDistance', 'Manual_hgb', 'Prediction_hgb', 'PredictionSupport_HGB',
               'NegativeHistorySupport', 'Manual_DirectRMSE', 'Manual_ShuffledHGB_0']
    frozen_scores = frozen_scores[frozen_scores.upstream.isin(predictions) & frozen_scores.method.isin(methods)]
    strata = []
    task_positions = {t: i for i, t in enumerate(tasks.task_id)}
    for key, part in frozen_scores.groupby(['line', 'upstream', 'method', 'seed', 'target'], sort=True):
        ix = np.asarray([task_positions[t] for t in part.task_id])
        model = key[1]
        assert np.max(np.abs(part.true_error_rmse.to_numpy() - errors[model]['full_observed'][ix])) < 1e-8
        for mode, e in errors[model].items():
            revised = part.assign(true_error_rmse=e[ix])
            result = dict(zip(['line', 'upstream', 'method', 'risk_seed', 'context'], key))
            result.update({'truth_diagnostic': mode, 'role': ROLE})
            result.update(metrics(revised, revised.risk.to_numpy()))
            strata.append(result)
    strata = pd.DataFrame(strata)
    strata.to_csv(OUT / 'FROZEN_RISK_DIAGNOSTIC_STRATA.csv', index=False)
    strata.groupby(['line', 'upstream', 'method', 'risk_seed', 'truth_diagnostic'], as_index=False)[
        ['utility20', 'spearman', 'aurc', 'high_risk_miss_rate']].mean().to_csv(OUT / 'FROZEN_RISK_DIAGNOSTIC_MACRO.csv', index=False)
    status = {'status': 'COMPLETE', 'role': ROLE, 'primary_assets_changed': False,
              'n_tasks': n, 'n_perturbation_genes': int(tasks.gene.nunique()), 'n_expression_genes': g,
              'primary_cells': int(tasks.n_target_cells.sum()), 'release_cells_hashed_once': sum(x['release_rows'] for x in audits),
              'official_source_path': str(OFFICIAL), 'official_source_full_file_rehashed': False,
              'official_source_metadata_and_gene_order_verified': True, 'raw_spot_cells': 128,
              'raw_spot_max_abs_residual': max(x['raw_spot_max_abs_residual'] for x in audits),
              'primary_float32_rmse_vs_float64_max_abs_difference': precision,
              'guide_or_sgrna_fields_in_official_obs': guide_fields,
              'fixed20_seeds': SEEDS, 'fixed20_sampling': 'without replacement; SHA256 task/seed -> numpy Generator choice; original frozen controls retained',
              'cell_iid_formula': 'mean_gene(sample_variance_with_ddof1) / n_cells',
              'batch_cluster_formula': 'B/(B-1) * sum_b ||sum_cells_in_b(x-cell_mean)||^2 / (n_cells^2 * n_genes)',
              'finite_population_fixed20_formula': '(1/20 - 1/n_cells) * mean_gene(sample_variance_with_ddof1)',
              'assumptions': ['Cell iid is conditional on the observed condition and fixed control reference.',
                              'Batch cluster variance additionally assumes independent observed batch clusters.',
                              'Guide identity is absent from official obs; biological guide/replicate independence cannot be established.',
                              'MSE subtraction requires an unbiased noisy centroid independent of predictions; it is a diagnostic, not a replacement endpoint.',
                              'Uncertainty of frozen matched controls and model/predictor fitting is excluded.',
                              'Fixed20 checks finite observed cell sampling only and retain shared original controls; they cannot establish a biological noise floor.'],
              'truth_recomputation': audits, 'code_sha256': sha(Path(__file__))}
    (OUT / 'STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'output': str(OUT), 'summary': summaries.to_dict('records')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if '--finish-from-existing' in sys.argv:
        finish_from_existing()
    else:
        main()
