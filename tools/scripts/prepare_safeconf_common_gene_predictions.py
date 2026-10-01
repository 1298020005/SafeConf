#!/usr/bin/env python3
"""Project existing frozen predictions onto a metadata-selected common axis.

This stage reads predictions and Source DEV/SEEN arrays only. It does not
load McFaline treated expression or calculate any new target errors. The old
512-gene confirmation stays intact; the new contract is explicitly SEEN.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import ids_hash
from tools.scripts import run_safeconf_research_closure as closure

OUT = closure.RUNTIME / 'common_gene_axis'
DOC = closure.OUT / 'common_gene_axis'


def h5column(group, name):
    value = group[name]
    if isinstance(value, h5py.Group):
        cats = value['categories'][:]
        cats = np.asarray([x.decode() if isinstance(x, bytes) else x for x in cats])
        codes = value['codes'][:]
        if (codes < 0).any(): raise RuntimeError('missing prediction metadata category')
        return cats[codes]
    raw = value[:]
    return np.asarray([x.decode() if isinstance(x, bytes) else x for x in raw]) if raw.dtype.kind in 'OS' else raw


def aggregate(directory, genes, name, test_manifest=None):
    files = sorted(directory.glob('*.h5ad'))
    if not files: raise FileNotFoundError(directory)
    sums, counts, keys = {}, {}, {}
    manifest = []
    expected_axis = None
    for number, path in enumerate(files, 1):
        with h5py.File(path, 'r') as f:
            axis = h5column(f['var'], f['var'].attrs['_index']).astype(str)
            if expected_axis is None:
                expected_axis = axis
                index = {g: i for i, g in enumerate(axis)}
                if not set(genes) <= set(index): raise RuntimeError('frozen prediction lacks common genes')
                columns = np.asarray([index[g] for g in genes])
            elif not np.array_equal(axis, expected_axis):
                raise RuntimeError('prediction gene ordering changed between files')
            # Only X (model predictions) is accessed; no potential truth layers.
            values = np.asarray(f['X'], np.float32)[:, columns]
            ci = h5column(f['obs'], '_condition_idx').astype(int)
            gene = h5column(f['obs'], 'condition').astype(str)
            cell = h5column(f['obs'], 'cell_type').astype(str)
            treatment = h5column(f['obs'], 'treatment').astype(str)
            for condition in np.unique(ci):
                use = np.flatnonzero(ci == condition)
                task_keys = set(zip(gene[use], cell[use], treatment[use]))
                if len(task_keys) != 1: raise RuntimeError('prediction condition mixes biological tasks')
                key = next(iter(task_keys))
                if condition in keys and keys[condition] != key:
                    raise RuntimeError('condition index changed biological meaning')
                keys[condition] = key
                sums.setdefault(condition, np.zeros(len(genes), np.float64))
                counts.setdefault(condition, 0)
                sums[condition] += values[use].sum(axis=0, dtype=np.float64)
                counts[condition] += len(use)
            manifest.append({'file': str(path), 'bytes': path.stat().st_size,
                             'mtime_ns': path.stat().st_mtime_ns, 'prediction_rows': len(values)})
        if number % 20 == 0: print(f'{name}: {number}/{len(files)} frozen prediction files', flush=True)
    order = sorted(keys)
    if order != list(range(len(order))): raise RuntimeError('non-contiguous prediction condition indices')
    tasks = pd.DataFrame([{'condition_index': i, 'gene': keys[i][0], 'context': keys[i][1],
                          'treatment': keys[i][2], 'task_id': '::'.join(keys[i]),
                          'n_generated_prediction_cells': counts[i]} for i in order])
    if test_manifest is not None:
        expected = pd.read_csv(test_manifest)
        expected_keys = list(zip(expected.condition.astype(str), expected.cell_type.astype(str), expected.treatment.astype(str)))
        if [keys[i] for i in order] != expected_keys:
            raise RuntimeError('canonical test predictions changed frozen task order')
    array = np.stack([sums[i] / counts[i] for i in order]).astype(np.float32)
    closure.tx.atomic_npy(OUT / f'{name}_PREDICTED_STATES.npy', array)
    closure.tx.atomic_csv(OUT / f'{name}_TASKS.csv', tasks)
    closure.tx.atomic_csv(DOC / f'{name}_PREDICTION_FILE_MANIFEST.csv', pd.DataFrame(manifest))
    return {'n_tasks': len(tasks), 'n_clusters': tasks.gene.nunique(), 'n_prediction_files': len(files),
            'task_order_hash': ids_hash(tasks.task_id), 'prediction_sha256': hashlib.sha256((OUT / f'{name}_PREDICTED_STATES.npy').read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    genes = json.loads((closure.RUNTIME / 'POSSIBLE_COMMON_GENE_IDS.json').read_text())
    source_genes = json.loads((closure.tx.STORE_ROOT / 'gene_ids.json').read_text())['gene_ids']
    positions = {g: i for i, g in enumerate(source_genes)}
    columns = np.asarray([positions[g] for g in genes])
    if len(genes) < 2000 or len(genes) / len(source_genes) < .7:
        raise RuntimeError('common gene axis does not satisfy the fixed alignment minimum')
    contract = {'role': 'POST_CONFIRMATION_SEEN_CONTRACT_AUDIT', 'n_genes': len(genes),
        'gene_ids_hash': ids_hash(genes), 'source_axis_coverage': len(genes)/len(source_genes),
        'selection': 'source output gene IDs intersect raw target prediction gene IDs; metadata only',
        'missing_gene_zero_fill': False, 'native_target_prediction_genes': 15009,
        'checkpoint_or_prediction_generation_changes': False, 'new_upstream_training': 0,
        'new_model_calls': 0, 'frozen_decoder_state_mean_mix_alpha': .25,
        'current_stage_reads_target_treated_expression': False,
        'future_scope': 'same 16 methods and strict label budgets after canonical biological reference preparation',
        'original_512gene_confirmation_replaced': False,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    closure.tx.atomic_json(DOC / 'PREDICTION_CONTRACT_PREREGISTRATION.json', contract)
    (OUT / 'GENE_IDS.json').write_text(json.dumps(genes) + '\n')
    tasks, pairs, effects, bases, predictions, truth = closure.read_tx()
    for model, array in predictions.items():
        closure.tx.atomic_npy(OUT / f'SOURCE_{model}_PREDICTED_EFFECTS.npy', array[:, columns].astype(np.float32))
    closure.tx.atomic_npy(OUT / 'SOURCE_TRUE_EFFECTS.npy', truth[:, columns].astype(np.float32))
    closure.tx.atomic_npy(OUT / 'SOURCE_PUBLIC_EFFECTS.npy', np.asarray(effects[:, columns], np.float32))
    controls = np.load(closure.tx.STORE_ROOT / 'control_vectors.npy', mmap_mode='r')
    closure.tx.atomic_npy(OUT / 'SOURCE_PUBLIC_CONTROLS.npy', np.asarray(controls[:, columns], np.float32))
    closure.tx.atomic_csv(OUT / 'SOURCE_TASKS.csv', tasks[['task_id', 'gene', 'target', 'condition', 'fold']])
    validation = aggregate(closure.MC / 'validation_supervisor_budget15_20260929T1930Z/decoder_validation/evaluation/predicted_anndata', genes, 'VALIDATION')
    test = aggregate(closure.MC / 'test_pretruth_20260930/full_predictions', genes, 'TEST',
        closure.MC / 'test_pretruth_20260930/aggregated_predictions/TEST_TASKS.csv')
    closure.tx.atomic_json(DOC / 'PREDICTION_PROJECTION_STATUS.json', {'status': 'COMPLETE',
        'contract': contract, 'validation': validation, 'test': test,
        'target_treated_expression_read': False, 'new_upstream_training': 0, 'new_model_calls': 0})
    print(json.dumps({'phase': 'canonical_prediction_projection', 'status': 'COMPLETE',
                      'validation': validation, 'test': test}), flush=True)


if __name__ == '__main__':
    main()
