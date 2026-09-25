#!/usr/bin/env python3
"""Rebuild Cui split-half source variance on E256's exact 5000-gene task view.

The existing Cui E256 truth is public, so this is retrospective. Every fold's
H/Q feature uses only its train source pairs. Full-group means are checked
against archived E256 true effects before the feature table is accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


RAW = Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/CuiHacohen2023.h5ad')
MATRIX = Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
INPUT = Path('/home/yyf/safeconf_runtime/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/safeconf_cui_go_nogo_probe/input')


def source_means() -> tuple[dict, dict]:
    obj = ad.read_h5ad(RAW, backed='r')
    try:
        if obj.n_obs != 96034 or obj.n_vars != 31053 or obj.layers.keys():
            raise ValueError('Cui raw matrix changed')
        if any(str(c).startswith('highly_variable') for c in obj.var.columns):
            raise ValueError('unexpected HVG panel; E256 used first 5000 genes')
        c = obj.obs['celltype'].map(lambda x: '' if pd.isna(x) else str(x).strip())
        p = obj.obs['perturbation'].map(lambda x: '' if pd.isna(x) else str(x).strip())
        rng = np.random.default_rng(5201)
        frame = pd.DataFrame({'context': c, 'perturbation': p}, index=obj.obs.index)
        groups = {}
        for (context, perturbation), sub in frame.groupby(['context', 'perturbation'],
                                                            observed=False):
            if not context or perturbation.lower() in ('', 'nan', 'none', 'null'):
                continue
            positions = frame.index.get_indexer(sub.index)
            if len(positions) < 6:
                continue
            if len(positions) > 2200:
                positions = rng.choice(positions, size=2200, replace=False)
            groups[context, perturbation] = positions
        # Select the same first 5000 genes, after original group sampling.
        matrix = obj.X[:, :5000]
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        matrix = matrix.tocsr().astype(np.float32)
        means = {}
        for key, positions in groups.items():
            if len(positions) < 12:
                # E256 accepts >=6 cells, but such a small half is explicitly
                # retained and flagged in the output audit.
                pass
            a = np.asarray(matrix[positions[::2]].mean(axis=0), np.float32).ravel()
            b = np.asarray(matrix[positions[1::2]].mean(axis=0), np.float32).ravel()
            full = np.asarray(matrix[positions].mean(axis=0), np.float32).ravel()
            means[key] = (full, a, b, len(positions))
        # `control` may have capitalization variants; use the same predicate
        # as E256's task builder.
        def is_control(label: str) -> bool:
            val = label.strip().lower()
            return val in ('control', 'ctrl', 'non-targeting', 'non_targeting',
                           'non-target', 'ntc') or val.startswith(('ctrl', 'control'))
        controls = {context: key for (context, pert), key in
                    ((pair, pair) for pair in means) if is_control(pert)}
        effects = {}
        for (context, pert), (full, a, b, n) in means.items():
            if is_control(pert) or context not in controls:
                continue
            ctrl = means[controls[context]]
            effects[context, pert] = (full - ctrl[0], a - ctrl[1], b - ctrl[2], n)
        return effects, {'n_raw_cells': obj.n_obs, 'n_groups': len(groups),
                         'n_effect_pairs': len(effects),
                         'minimum_effect_group_cells': min(v[3] for v in effects.values()),
                         'n_genes': 5000,
                         'original_selection': 'first 5000 genes; min6; cap2200; seed5201'}
    finally:
        obj.file.close()


def verify_archived_effects(effects: dict) -> dict:
    records = pd.read_csv(INPUT / 'PREDICTION_RECORDS.csv',
                          usecols=['context', 'perturbation', 'true_effect_key'])
    subset = records.drop_duplicates(['context', 'perturbation']).sort_values(
        ['context', 'perturbation']).head(30)
    diffs = []
    with np.load(INPUT / 'true_effects.npz', allow_pickle=False) as archived:
        for row in subset.itertuples(index=False):
            key = (str(row.context), str(row.perturbation))
            if key not in effects:
                raise ValueError(f'archived task not reconstructed: {key}')
            diffs.append(float(np.max(np.abs(effects[key][0] -
                                             archived[str(row.true_effect_key)]))))
    maximum = max(diffs)
    if maximum > 2e-4:
        raise ValueError(f'Cui full-group effect mismatch vs E256: {maximum}')
    return {'n_archived_effects_checked': len(diffs),
            'maximum_abs_effect_difference': maximum}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    effects, audit = source_means()
    audit.update(verify_archived_effects(effects))
    table = pd.read_csv(MATRIX, usecols=['dataset_name', 'fold_id', 'split',
                                         'task_key', 'context', 'perturbation',
                                         'predictor_name', 'perturbation_effect_variance'])
    table = table.loc[table.dataset_name.eq('CuiHacohen2023') &
                      table.predictor_name.eq('ContextSimBaseline')]
    rows = []
    for fold, frame in table.groupby('fold_id', sort=True):
        source = frame.loc[frame.split.eq('train')].groupby('perturbation').context.agg(
            lambda v: sorted(set(v.astype(str))))
        target = frame.loc[frame.split.eq('test')]
        for item in target.itertuples(index=False):
            contexts = source.loc[item.perturbation]
            if item.context in contexts or len(contexts) < 2:
                raise ValueError('target context in train history or insufficient source')
            full = np.stack([effects[context, item.perturbation][0]
                             for context in contexts])
            half_a = np.stack([effects[context, item.perturbation][1]
                               for context in contexts])
            half_b = np.stack([effects[context, item.perturbation][2]
                               for context in contexts])
            observed = full.var(axis=0)
            split_noise = ((half_a - half_b) ** 2).mean(axis=0) / 4
            bio = np.maximum(observed - (1 - 1 / len(contexts)) * split_noise, 0)
            reproduced = float(observed.mean())
            rows.append({'fold_id': int(fold), 'task_key': item.task_key,
                         'n_train_source_contexts': len(contexts),
                         'historical_variance_reconstructed': reproduced,
                         'H_bio_approx_variance': float(bio.mean()),
                         'source_split_noise_variance': float(split_noise.mean()),
                         'E256_historical_variance': float(item.perturbation_effect_variance),
                         'median_source_cells_reconstructed': float(np.median(
                             [effects[context, item.perturbation][3]
                              for context in contexts]))})
    out = pd.DataFrame(rows)
    if len(out) != 1253 or out.task_key.duplicated().any():
        raise ValueError('Cui test task inventory changed')
    maximum = float(np.max(np.abs(out.historical_variance_reconstructed -
                                  out.E256_historical_variance)))
    if maximum > 1e-5:
        raise ValueError(f'E256 H feature mismatch: {maximum}')
    audit['max_abs_history_variance_difference_vs_E256'] = maximum
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    status = {'status': 'RETROSPECTIVE_SOURCE_ONLY_SPLIT_HALF_FEATURES',
              'n_tasks': len(out), 'n_folds': out.fold_id.nunique(),
              'audit': audit,
              'limits': [
                  'Split-half variance is a sampling/measurement proxy and may include cell biology.',
                  'The Cui study was selected after E256 outcomes were public; no independent confirmation.',
                  'Original E256 panel may use all contexts for gene choice; this is an old retrospective contract.'
              ]}
    args.output.with_suffix('.status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
