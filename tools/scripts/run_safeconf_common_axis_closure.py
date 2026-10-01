#!/usr/bin/env python3
"""Fixed-rule same-gene-axis reanalysis; October post-confirmation evidence.

Raw native prediction outputs are reused. No parameter search, new upstream
fit or replacement of the September confirmation is performed. Source labels
and all numerical features are recomputed on the registered 2840-gene axis.
Target treated truth is loaded only after risk predictions have been saved.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual import PublicMemoryStore
from tools.safeconf_continual.research import P, PUBLIC, SEEDS, ids_hash, summarize
from tools.scripts import run_safeconf_research_closure as closure
from tools.scripts.prepare_safeconf_common_gene_predictions import h5column

BASE_RUNTIME, BASE_OUT = closure.RUNTIME, closure.OUT
COMMON = BASE_RUNTIME / 'common_gene_axis'
DOC = BASE_OUT / 'common_gene_axis'
ORIGINAL_READ_TX = closure.read_tx
H5 = Path('/home/yyf/data/perturbench_mcfaline23_official/mcfaline23_gxe_processed.h5ad')
SPLIT = H5.parent / 'splits/mcfaline23_gxe_splits/full_covariate_split.csv'


def source_inputs():
    tasks, _, _, frames, _, _ = ORIGINAL_READ_TX()
    genes = json.loads((COMMON / 'GENE_IDS.json').read_text())
    source_genes = json.loads((closure.tx.STORE_ROOT / 'gene_ids.json').read_text())['gene_ids']
    mapping = {g: i for i, g in enumerate(source_genes)}
    columns = np.asarray([mapping[g] for g in genes])
    rows = tasks.source_mean_delta_row.to_numpy(int)
    controls = np.load(closure.tx.VECTOR_ROOT / 'e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy',
                       mmap_mode='r')[rows][:, columns]
    memory, _, _, manifest = PublicMemoryStore(closure.tx.STORE_ROOT).load()
    effects = np.load(COMMON / 'SOURCE_PUBLIC_EFFECTS.npy')
    memory_controls = np.load(COMMON / 'SOURCE_PUBLIC_CONTROLS.npy')
    eligibility = pd.read_parquet(closure.tx.STORE_ROOT / manifest['eligibility']['path'])
    truth = np.load(COMMON / 'SOURCE_TRUE_EFFECTS.npy')
    pairs = closure.tx.build_pairs(tasks, truth, controls, memory, effects, memory_controls, eligibility)
    predictions = {}
    for model, frame in frames.items():
        prediction = np.load(COMMON / f'SOURCE_{model}_PREDICTED_EFFECTS.npy')
        predictions[model] = prediction
        frame = closure.mc.prediction_features(frame, prediction)
        frame['true_error_rmse'] = closure.tx.rmse_rows(prediction, truth)
        frame['output_contract_id'] = 'E201_common2840gene_log1p_delta_v1'
        frames[model] = frame
    return tasks, pairs, effects, frames, predictions, truth


def test_controls():
    # Use the original exact-row control-only implementation, without its
    # optional AnnData import. It never reads test treated X slices.
    source = (ROOT / 'tools/scripts/aggregate_mcfaline_pretruth_predictions.py').read_text()
    tree = ast.parse(source)
    names = ['categorical', 'aggregate_test_controls_only']
    functions = '\n\n'.join(ast.get_source_segment(source, n) for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name in names)
    namespace = {'np': np, 'pd': pd, 'h5py': h5py, 'Path': Path}
    exec(compile(functions, 'original_control_only_helpers', 'exec'), namespace)
    genes = json.loads((COMMON / 'GENE_IDS.json').read_text())
    lookup = namespace['aggregate_test_controls_only'](H5, SPLIT, genes)
    tasks = pd.read_csv(COMMON / 'TEST_TASKS.csv')
    values = np.stack([lookup[(str(t.context), str(t.treatment))] for t in tasks.itertuples()])
    closure.tx.atomic_npy(COMMON / 'TEST_CONTROLS.npy', values.astype(np.float32))
    return values


def target_features():
    if not json.loads((DOC / 'COMPETENCE_COMMON_AXIS.json').read_text())['passes_fixed_competence_rule']:
        raise RuntimeError('same-axis upstream failed preregistered competence; keep as stress line')
    memory, effects, controls, _ = PublicMemoryStore(COMMON / 'public_mcfaline_trainval').load()
    val = pd.read_csv(COMMON / 'VALIDATION_BIOLOGY_TASKS.csv')
    val['fold'] = closure.mc.fold_assignment(val) if hasattr(closure.mc, 'fold_assignment') else val.perturbation.map(
        lambda g: int.from_bytes(hashlib.sha256(f'SafeConf-McFaline-Decoder-repair-v1\0{g}'.encode()).digest()[:8], 'big') % 5)
    val['target'] = val.context
    test = pd.read_csv(COMMON / 'TEST_TASKS.csv').rename(columns={'gene': 'perturbation'})
    test['gene'] = test.perturbation; test['target'] = test.context; test['fold'] = -1
    val_truth = np.load(COMMON / 'VALIDATION_TRUE_EFFECTS.npy')
    val_controls = np.load(COMMON / 'VALIDATION_CONTROLS.npy')
    control = test_controls()
    baseline_lookup = dict(np.load(COMMON / 'TRAIN_STATE_MEAN_EFFECTS.npz'))
    means = np.stack([baseline_lookup[t.context + '::' + t.treatment] for t in test.itertuples()])
    prediction = .25 * (np.load(COMMON / 'TEST_PREDICTED_STATES.npy') - control) + .75 * means
    closure.tx.atomic_npy(COMMON / 'TEST_CALIBRATED_EFFECTS.npy', prediction.astype(np.float32))
    _, priors, _, audit = closure.mc.public_priors(val, test, val_truth, val_controls, control,
                                                 memory, effects, controls)
    frame = closure.mc.add_prior_features(test, prediction, priors, audit)
    frame['true_error_rmse'] = np.nan
    frame = closure.annotate(frame, 'McFaline23', 'DecoderOnly',
        'decoder_frozen_alpha025_consistent_trainmean_common2840_v1', 'McFaline_common2840gene_log1p_delta_v1')
    output = {}
    for ref, prefix in closure.PREFIX.items():
        part = frame.copy()
        for col in PUBLIC: part[col] = part[f'{prefix}_{col}']
        output[ref] = part
    pairs = closure.mc.build_pairs(frame, control, memory, effects, controls, None)
    arrays = np.full((len(frame), effects.shape[1]), np.nan); summaries = []
    for q, group in pairs.groupby('task_row', sort=True):
        vectors = effects[group.memory_row.to_numpy(int)]
        arrays[q] = vectors.mean(axis=0)
        summaries.append({'task_id': frame.iloc[q].task_id, 'reference': 'Uniform',
            'prior_uncertainty': float(np.sqrt(np.mean((vectors - arrays[q]) ** 2))),
            'effective_sources': len(vectors), 'history_conflict': float(group.source_conflict.mean()),
            'log_history_support': float(np.log1p(np.expm1(group.log_source_cells).sum()))})
    output['Uniform'] = closure.reference_frame(frame, prediction, arrays, pd.DataFrame(summaries), 'Uniform')
    for name, part in output.items():
        if part[PUBLIC].isna().any().any(): raise RuntimeError('common-axis history coverage incomplete')
        part.to_parquet(closure.RUNTIME / f'external_{name}.parquet', index=False)
    return output


def open_seen_truth_after_prediction_freeze():
    predictions = closure.OUT / 'MATRIX_TASK_PREDICTIONS.csv.gz'
    score = pd.read_csv(predictions)
    target = score[score.line.eq('TxPert_to_McFaline')]
    if target.risk.isna().any(): raise RuntimeError('common-axis risks were not fully generated')
    snapshot = closure.RUNTIME / 'COMMON_AXIS_RISK_ONLY_FROZEN.csv.gz'
    closure.tx.atomic_csv(snapshot, score.drop(columns='true_error_rmse'))
    closure.tx.atomic_json(DOC / 'COMMON_AXIS_PREDICTION_FREEZE.json', {
        'role': 'SEEN_POST_CONFIRMATION', 'risk_predictions_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'risk_predictions_snapshot': str(snapshot),
        'gene_ids_sha256': hashlib.sha256((COMMON / 'GENE_IDS.json').read_bytes()).hexdigest(),
        'upstream_effect_sha256': hashlib.sha256((COMMON / 'TEST_CALIBRATED_EFFECTS.npy').read_bytes()).hexdigest(),
        'alpha': .25, 'source_error_labels_only': True, 'target_error_labels_for_risk_training': 0,
        'core_parameters': {'hgb_iter': 200, 'learning_rate': .05, 'depth': 3, 'min_leaf': 20, 'l2': 10},
        'new_pristine_confirmation_claim': False, 'test_biology_opened_after_this_freeze': True})
    tasks = pd.read_csv(COMMON / 'TEST_TASKS.csv')
    task_map = {t.task_id: i for i, t in enumerate(tasks.itertuples())}
    sums = np.zeros((len(tasks), 2840), np.float64); counts = np.zeros(len(tasks), int)
    genes = json.loads((COMMON / 'GENE_IDS.json').read_text())
    with h5py.File(H5, 'r') as f:
        ids = h5column(f['obs'], '_index')
        roles = pd.read_csv(SPLIT, header=None, names=['cell', 'role']).set_index('cell').role.reindex(ids).to_numpy()
        control = h5column(f['obs'], 'control').astype(int) == 1
        targets = h5column(f['obs'], 'perturbation').astype(str)
        contexts = h5column(f['obs'], 'cell_type').astype(str)
        treatments = h5column(f['obs'], 'treatment').astype(str)
        selected = np.flatnonzero((roles == 'test') & ~control)
        axis = {g: i for i, g in enumerate(h5column(f['var'], 'gene_name').astype(str))}
        mapping = np.full(len(axis), -1, int)
        for i, gene in enumerate(genes): mapping[axis[gene]] = i
        x = f['X']; pointer = x['indptr'][:]
        for number, row in enumerate(selected, 1):
            key = targets[row] + '::' + contexts[row] + '::' + treatments[row]
            if key not in task_map: raise RuntimeError('test treated row outside frozen query tasks')
            begin, end = int(pointer[row]), int(pointer[row + 1])
            columns = x['indices'][begin:end]; values = x['data'][begin:end]
            projected = mapping[columns]; keep = projected >= 0
            sums[task_map[key], projected[keep]] += values[keep]; counts[task_map[key]] += 1
            if number % 10000 == 0: print(f'Common-axis test biology: {number}/{len(selected)} cells', flush=True)
    if (counts == 0).any(): raise RuntimeError('frozen test task lacks observed biology')
    truth = sums / counts[:, None] - np.load(COMMON / 'TEST_CONTROLS.npy')
    error = closure.tx.rmse_rows(np.load(COMMON / 'TEST_CALIBRATED_EFFECTS.npy'), truth)
    closure.tx.atomic_npy(COMMON / 'TEST_TRUE_EFFECTS.npy', truth.astype(np.float32))
    lookup = dict(zip(tasks.task_id, error))
    query_rows = score.line.eq('TxPert_to_McFaline')
    score.loc[query_rows, 'true_error_rmse'] = score.loc[query_rows, 'task_id'].map(lookup)
    closure.save_csv('MATRIX_TASK_PREDICTIONS.csv.gz', score)
    strata, macro = summarize(score)
    closure.save_csv('MATRIX_CONTEXT_RESULTS.csv', strata); closure.save_csv('MATRIX_MACRO_RESULTS.csv', macro)
    for ref in closure.REFS:
        part = pd.read_parquet(closure.RUNTIME / f'external_{ref}.parquet')
        part['true_error_rmse'] = part.task_id.map(lookup)
        part.to_parquet(closure.RUNTIME / f'external_{ref}.parquet', index=False)
    closure.tx.atomic_json(DOC / 'COMMON_AXIS_TRUTH_EVALUATION.json', {
        'role': 'SEEN_POST_CONFIRMATION', 'n_tasks': len(tasks), 'n_clusters': tasks.gene.nunique(),
        'test_error_used_for_training': False, 'target_treated_cells': int(counts.sum()),
        'truth_sha256': hashlib.sha256((COMMON / 'TEST_TRUE_EFFECTS.npy').read_bytes()).hexdigest()})
    print(macro[macro.seed.eq(SEEDS[0])][['line', 'method', 'utility20', 'spearman']].to_string(index=False), flush=True)


def main():
    global DOC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['fit', 'evaluate', 'statistics', 'feedback', 'all'], default='all')
    parser.add_argument('--result-root', type=Path,
                        help='Fresh derived-result root for a separate reproduction version')
    parser.add_argument('--risk-cache-root', type=Path,
                        help='Fresh risk-feature cache; supply together with --result-root')
    args = parser.parse_args()
    if (args.result_root is None) != (args.risk_cache_root is None):
        raise ValueError('separate reproduction requires both result and risk-cache roots')
    original_doc = DOC
    if args.result_root is not None:
        DOC = args.result_root.resolve()
        DOC.mkdir(parents=True, exist_ok=True)
        gate = original_doc / 'COMPETENCE_COMMON_AXIS.json'
        registration = DOC / gate.name
        if registration.exists() and registration.read_bytes() != gate.read_bytes():
            raise RuntimeError('reproduction attempted to change the registered upstream gate')
        if not registration.exists():
            registration.write_bytes(gate.read_bytes())
    if (DOC / f'RUN_STATUS_{args.phase}.json').exists():
        raise FileExistsError('completed common-axis run is immutable; use a new registered version')
    closure.RUNTIME = args.risk_cache_root.resolve() if args.risk_cache_root else COMMON / 'risk_cache'
    closure.OUT = DOC / 'results'
    closure.RUNTIME.mkdir(parents=True, exist_ok=True); closure.OUT.mkdir(parents=True, exist_ok=True)
    closure.read_tx = source_inputs
    closure.external_query = target_features
    if args.phase in ('fit', 'all'):
        closure.build_nested(); closure.run_matrix()
    if args.phase in ('evaluate', 'all'): open_seen_truth_after_prediction_freeze()
    if args.phase in ('statistics', 'all'):
        closure.run_bootstrap(5000); closure.run_support_controls(); closure.write_method_budget_ledger()
    if args.phase in ('feedback', 'all'):
        closure.run_feedback(); closure.run_feedback_statistics(5000)
    closure.tx.atomic_json(DOC / f'RUN_STATUS_{args.phase}.json', {'status': 'COMPLETE',
        'role': 'SEEN_POST_CONFIRMATION', 'n_genes': 2840, 'phase': args.phase,
        'new_upstream_training': 0, 'new_model_calls': 0,
        'test_prediction_control_rule': 'observed TEST control; TRAIN cell-mean state baseline; fixed alpha025',
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == '__main__': main()
