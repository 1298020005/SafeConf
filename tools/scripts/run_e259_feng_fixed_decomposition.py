#!/usr/bin/env python3
"""E259: fixed-label-budget E258 development decomposition, never test truth.

The four train donors provide out-of-fold error labels. The two validation
donors are evaluated only. Split-half source variation is an approximate
sampling/measurement proxy, not a clean technical-noise estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from build_e258_official_dev_view import TRAIN, VALIDATION, TEST
from run_e258_history_validation import gene_bootstrap, metrics, percentile
from run_e258_official_dev_predictor import gene_mask, load_view, sha256, train_only_shrinkage
from run_e258_split_half_noise_audit import load_halves


ROOT = Path('/home/yyf/data/feng2025_candidate')
NAMES = ('M', 'n_sources', 'median_source_cells', 'source_split_noise',
         'H_raw', 'H_bio_approx', 'Z_hist_diagnostic')
GROUPS = {
    'Ridge_M': ('M',),
    'Ridge_M_Q': ('M', 'n_sources', 'median_source_cells', 'source_split_noise'),
    'Ridge_M_Hraw': ('M', 'H_raw'),
    'Ridge_M_Q_Hraw': ('M', 'n_sources', 'median_source_cells',
                       'source_split_noise', 'H_raw'),
    'Ridge_M_Q_Hbio': ('M', 'n_sources', 'median_source_cells',
                       'source_split_noise', 'H_bio_approx'),
    'Ridge_M_Q_Zhist_diagnostic': ('M', 'n_sources', 'median_source_cells',
                                   'source_split_noise', 'Z_hist_diagnostic'),
}


def history(view: dict, halves: np.ndarray) -> tuple[dict, dict, dict]:
    targets = view['task_target'].astype(str)
    donors = view['task_donor'].astype(str)
    effects = view['effect'].astype(np.float32)
    cells = view['task_cells'].astype(int)
    full, split, support = {}, {}, {}
    for target in sorted(set(targets[np.isin(donors, TRAIN)])):
        for donor in TRAIN:
            mask = (targets == target) & (donors == donor)
            if mask.any():
                full[target, donor] = effects[mask].mean(axis=0)
                split[target, donor] = halves[:, mask].mean(axis=1)
                support[target, donor] = int(cells[mask].sum())
    return full, split, support


def source_mean(rows: np.ndarray, view: dict, full: dict,
                permitted: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    targets = view['task_target'].astype(str)
    donors = view['task_donor'].astype(str)
    base = np.zeros((len(rows), len(view['gene'])), dtype=np.float32)
    eligible = np.zeros(len(rows), dtype=bool)
    for j, row in enumerate(rows):
        sources = [full[targets[row], donor] for donor in permitted
                   if donor != donors[row] and (targets[row], donor) in full]
        if len(sources) >= 2:
            base[j] = np.mean(sources, axis=0)
            eligible[j] = True
    return base, eligible


def features(rows: np.ndarray, pred: np.ndarray, view: dict, full: dict,
             split: dict, support: dict, permitted: tuple[str, ...]) -> np.ndarray:
    targets = view['task_target'].astype(str)
    donors = view['task_donor'].astype(str)
    genes = view['gene'].astype(str)
    result = np.empty((len(rows), len(NAMES)), dtype=np.float64)
    for j, row in enumerate(rows):
        target = targets[row]
        source = [donor for donor in permitted
                  if donor != donors[row] and (target, donor) in full]
        if len(source) < 2:
            raise ValueError('task lacks two independent train-only sources')
        keep = genes != target
        values = np.stack([full[target, donor][keep] for donor in source])
        half = np.stack([split[target, donor][:, keep] for donor in source])
        mean = values.mean(axis=0)
        observed_var_gene = values.var(axis=0)
        # The two halves have approximately equal cell counts. Nonlinear
        # normalization and batch mixture make this only a proxy for noise.
        noise_var_gene = ((half[:, 0] - half[:, 1]) ** 2).mean(axis=0) / 4
        bio_var_gene = np.maximum(observed_var_gene -
                                  (1 - 1 / len(source)) * noise_var_gene, 0)
        p = pred[j, keep]
        result[j] = (
            np.sqrt(np.mean(p ** 2)),
            len(source),
            np.median([support[target, donor] for donor in source]),
            np.sqrt(np.mean(noise_var_gene)),
            np.sqrt(np.mean(observed_var_gene)),
            np.sqrt(np.mean(bio_var_gene)),
            np.sqrt(np.mean((p - mean) ** 2 /
                            (noise_var_gene + bio_var_gene + 1e-6))),
        )
    if not np.isfinite(result).all():
        raise ValueError('nonfinite feature')
    return result


def ranked(matrix: np.ndarray, lines: np.ndarray) -> np.ndarray:
    return np.column_stack([percentile(matrix[:, j], lines)
                            for j in range(matrix.shape[1])])


def run(args: argparse.Namespace) -> dict:
    for path in (args.view, args.shrinkage, args.manifest,
                 Path(f'{args.prefix}.u32'), Path(f'{args.original_prefix}.u32')):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    view = load_view(args.view)
    donors = view['task_donor'].astype(str)
    if set(donors) != set(TRAIN + VALIDATION) or set(donors) & set(TEST):
        raise ValueError('test donor targeted effects entered development view')
    manifest = json.loads(args.manifest.read_text())
    halves, half_audit = load_halves(manifest, view['gene'].astype(str).tolist(),
                                    args.prefix, args.original_prefix)
    full, split, support = history(view, halves)
    lines = view['task_line'].astype(str)
    effects = view['effect'].astype(np.float32)
    mask = gene_mask(view)
    train_x, train_y, train_lines, fold_audit = [], [], [], {}
    for held in TRAIN:
        source = tuple(donor for donor in TRAIN if donor != held)
        source_rows = np.flatnonzero(np.isin(donors, source))
        source_base, eligible = source_mean(source_rows, view, full, source)
        source_rows, source_base = source_rows[eligible], source_base[eligible]
        alpha = train_only_shrinkage(source_base, effects[source_rows],
                                     mask[source_rows], lines[source_rows])
        held_rows = np.flatnonzero(donors == held)
        held_base, eligible = source_mean(held_rows, view, full, source)
        held_rows, held_base = held_rows[eligible], held_base[eligible]
        if len(held_rows) < 100:
            raise ValueError(f'insufficient OOF labels for {held}')
        pred = alpha * held_base
        train_x.append(features(held_rows, pred, view, full, split, support, source))
        train_y.append(np.sqrt((((pred - effects[held_rows]) ** 2) *
                                mask[held_rows]).sum(axis=1) /
                               mask[held_rows].sum(axis=1)))
        train_lines.append(lines[held_rows])
        fold_audit[held] = {'n_oof_labels': len(held_rows),
                            'source_donors': list(source), 'alpha': alpha}
    oof_x = np.concatenate(train_x)
    oof_y = np.concatenate(train_y)
    oof_lines = np.concatenate(train_lines)
    val_rows = np.flatnonzero(np.isin(donors, VALIDATION))
    val_base, eligible = source_mean(val_rows, view, full, TRAIN)
    if not eligible.all():
        raise ValueError('validation source coverage changed')
    shrink = json.loads(args.shrinkage.read_text())
    if shrink['view_sha256'] != sha256(args.view):
        raise ValueError('shrinkage input hash mismatch')
    alpha = float(shrink['alpha_fit_from_train_leave_one_donor_out'])
    pred = alpha * val_base
    val_x = features(val_rows, pred, view, full, split, support, TRAIN)
    val_y = np.sqrt(((((pred - effects[val_rows]) ** 2) *
                      mask[val_rows]).sum(axis=1) /
                     mask[val_rows].sum(axis=1)))
    val_lines = lines[val_rows]
    oof_rank = ranked(oof_x, oof_lines)
    val_rank = ranked(val_x, val_lines)
    y_rank = percentile(oof_y, oof_lines)
    score = {'M_unsupervised': val_rank[:, 0]}
    for name, group in GROUPS.items():
        cols = [NAMES.index(feature) for feature in group]
        model = Ridge(alpha=10.0)
        model.fit(oof_rank[:, cols], y_rank)
        score[name] = model.predict(val_rank[:, cols])
    results = {name: metrics(val_y, values, val_lines)
               for name, values in score.items()}
    targets = view['task_target'][val_rows].astype(str)
    pairwise = {}
    for left, right in (('Ridge_M_Q_Hraw', 'Ridge_M_Q'),
                        ('Ridge_M_Q_Hbio', 'Ridge_M_Q'),
                        ('Ridge_M_Q_Zhist_diagnostic', 'Ridge_M_Q')):
        diff = results[left]['macro_utility20'] - results[right]['macro_utility20']
        pairwise[f'{left}_vs_{right}'] = {
            'delta_macro_utility20': diff,
            'gene_cluster_bootstrap_development_only': gene_bootstrap(
                val_y, score[right], score[left], targets, val_lines),
        }
    features_table = pd.DataFrame(val_x, columns=NAMES)
    features_table['line'] = val_lines
    features_table['target'] = targets
    diagnostic = {
        'Zhist_vs_M_spearman_by_line': {
            line: float(pd.Series(features_table.loc[features_table.line.eq(line),
                                                 'Z_hist_diagnostic']).corr(
                features_table.loc[features_table.line.eq(line), 'M'],
                method='spearman'))
            for line in sorted(set(val_lines))
        },
        'Zhist_numerator_is_constant_multiple_of_history_mean': True,
        'Zhist_not_independent_prediction_history_disagreement': True,
    }
    result = {
        'stage': 'E259_FENG_FIXED_MEASUREMENT_VS_BIOLOGICAL_HISTORY_DEV',
        'status': 'RETROSPECTIVE_DEVELOPMENT_ONLY',
        'test_donor_target_truth_loaded': 0,
        'view_sha256': sha256(args.view),
        'split_half_audit': half_audit,
        'n_oof_labels': len(oof_y),
        'n_validation_tasks': len(val_rows),
        'oof_folds': fold_audit,
        'feature_names': NAMES,
        'feature_groups': GROUPS,
        'model': 'Ridge(alpha=10), same OOF error-label budget for all groups',
        'z_hist_epsilon_gene_variance': 1e-6,
        'results': results,
        'pairwise': pairwise,
        'diagnostic': diagnostic,
        'limits': [
            'Validation donors were used in previous development; no independent confirmation.',
            'Split-half differences approximate sampling/measurement variation and may contain biology.',
            'The shrinkage predictor is a multiple of source historical mean; Z_hist is a diagnostic only.',
            'Only two validation donors; gene bootstrap does not measure new-donor uncertainty.',
            'No downstream test release is authorized by this experiment.'
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'n_oof_labels': len(oof_y), 'n_val': len(val_rows),
                      'macro_utility20': {k: v['macro_utility20']
                                          for k, v in results.items()},
                      'deltas': {k: v['delta_macro_utility20']
                                 for k, v in pairwise.items()},
                      'test_target_truth_loaded': 0}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    dev = ROOT / 'raw_dev'
    parser.add_argument('--view', type=Path, default=dev / 'E258_RAW_DEV_VIEW.npz')
    parser.add_argument('--shrinkage', type=Path,
                        default=ROOT / 'raw_dev_models/shrinkage_status.json')
    parser.add_argument('--manifest', type=Path,
                        default=dev / 'E258_RAW_DEV_GROUPS.json')
    parser.add_argument('--prefix', type=Path,
                        default=dev / 'E258_RAW_SPLIT_HALF_GROUP_SUMS')
    parser.add_argument('--original-prefix', type=Path,
                        default=dev / 'E258_RAW_ALLOWED_GROUP_SUMS')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
