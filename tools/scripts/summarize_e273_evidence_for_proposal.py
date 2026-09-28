#!/usr/bin/env python3
"""Describe existing E273 summaries. No training, raw expression, or test access.

Pairs comparisons within the same dataset/predictor/bucket, averages buckets
within predictor then predictors within dataset. Counts are descriptive;
datasets, predictors and overlapping folds are not independent replicates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIRS = [
    ('P_vs_M', 'M', 'P'),
    ('Q_given_P', 'P', 'P_plus_Q'),
    ('H_given_PQ', 'P_plus_Q', 'P_plus_Q_plus_H'),
    ('E_given_PQH', 'P_plus_Q_plus_H', 'P_plus_Q_plus_H_plus_E'),
    ('E_given_PQ', 'P_plus_Q', 'P_plus_Q_plus_E'),
    ('H_given_PQE', 'P_plus_Q_plus_E', 'P_plus_Q_plus_H_plus_E'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory; preserve the previous audit.')
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.repo / 'docs/实验结果/E273_dual_history_review_20260927/CPU_FOLD_RESULTS.csv'
    frame = pd.read_csv(source)
    keys = ['dataset', 'predictor', 'held_bucket']
    if frame.duplicated(keys + ['method']).any():
        raise ValueError('Duplicate method/unit rows')
    wide = frame.pivot(index=keys, columns='method', values='utility20')
    if not np.isfinite(wide.to_numpy()).all():
        raise ValueError('Missing/nonfinite utility: explicitly redefine paired panel')
    cell = wide.groupby(level=['dataset', 'predictor']).mean()
    dataset = cell.groupby(level='dataset').mean()
    fold_delta, cell_delta, dataset_delta, summary = [], [], [], []
    for name, before, after in PAIRS:
        fd = wide[after] - wide[before]
        cd = fd.groupby(level=['dataset', 'predictor']).mean()
        dd = cd.groupby(level='dataset').mean()
        fold_delta.append(fd.rename(name))
        cell_delta.append(cd.rename(name))
        dataset_delta.append(dd.rename(name))
        summary.append(dict(
            comparison=name, before=before, after=after,
            dataset_macro_delta=float(dd.mean()),
            positive_datasets=int((dd > 0).sum()), n_datasets=len(dd),
            positive_dataset_predictor_means=int((cd > 0).sum()), n_dataset_predictor_units=len(cd),
            positive_folds=int((fd > 0).sum()), n_paired_folds=len(fd),
            leave_one_dataset_out_min=min(float(dd.drop(d).mean()) for d in dd.index),
            leave_one_dataset_out_max=max(float(dd.drop(d).mean()) for d in dd.index),
            interpretation='DESCRIPTIVE_REANALYSIS_NOT_INDEPENDENT_CONFIRMATION',
        ))
    pd.DataFrame(summary).to_csv(args.output / 'ADJACENT_INCREMENT_SUMMARY.csv', index=False)
    pd.concat(dataset_delta, axis=1).to_csv(args.output / 'DATASET_INCREMENT.csv')
    pd.concat(cell_delta, axis=1).to_csv(args.output / 'DATASET_PREDICTOR_INCREMENT.csv')
    pd.concat(fold_delta, axis=1).to_csv(args.output / 'PAIRED_FOLD_INCREMENT.csv')
    dataset.to_csv(args.output / 'DATASET_METHOD_UTILITY.csv')
    dataset.mean().rename('macro_utility20').to_csv(args.output / 'METHOD_MACRO.csv')
    old = pd.read_csv(args.repo / 'docs/实验结果/E273_dual_history_review_20260927/CPU_MACRO_SUMMARY.csv')
    for _, row in old.iterrows():
        assert np.isclose(dataset[row['method']].mean(), row['utility20_mean'], atol=1e-12)
    names = ['M', 'P', 'P_plus_Q', 'P_plus_Q_plus_H', 'P_plus_Q_plus_E', 'P_plus_Q_plus_H_plus_E']
    labels = ['M', 'P', 'P+Q', 'P+Q+H', 'P+Q+E', 'P+Q+H+E']
    dd = pd.concat(dataset_delta, axis=1)
    short = ['Cui', 'Frangieh', 'Lara ex vivo', 'Lara in vivo', 'McFarland', 'Santinha', 'sci-Plex3']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    values = dataset[names].mean().to_numpy()
    axes[0].bar(labels, values, color=['#6b7280', '#5572a3', '#347f91', '#58a3a8', '#db9d46', '#df744a'])
    axes[0].set_ylim(0, 0.86)
    axes[0].set_ylabel('Oracle-normalized review utility @20%')
    axes[0].set_title('Existing E273 CPU results: dataset macro')
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.013, f'{value:.3f}', ha='center', fontsize=9)
    x = np.arange(len(dd))
    axes[1].barh(x - .18, dd['H_given_PQ'], height=.34, label='H increment given P+Q', color='#58a3a8')
    axes[1].barh(x + .18, dd['H_given_PQE'], height=.34, label='H increment given P+Q+E', color='#df744a')
    axes[1].set_yticks(x, short)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='black', linewidth=.8)
    axes[1].set_xlabel('Paired utility difference')
    axes[1].set_title('Historical content gain depends on conditioning')
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Development evidence only | Q is incomplete | 7 datasets, 3 predictors, 102 eligible folds', fontsize=11)
    for ext in ['png', 'pdf']:
        fig.savefig(args.output / f'EVIDENCE_INCREMENT.{ext}', dpi=170)
    plt.close(fig)
    source_rel = str(source.relative_to(args.repo))
    manifest = dict(
        date='2026-09-28', status='COMPLETED_DESCRIPTIVE_REANALYSIS',
        source=source_rel, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_rows=len(frame), n_paired_units=len(wide), n_dataset_predictor_units=len(cell), n_datasets=len(dataset),
        training_performed=False, new_predictions_generated=False, raw_expression_read=False,
        new_final_test_truth_read=False, uncertainty_intervals='not computed: clustered and previously inspected data',
        aggregation='paired fold -> mean within dataset/predictor -> mean within dataset -> equal dataset macro',
        corrections=[
            'Original positive_cells counts any positive fold vs M, not positive cell mean.',
            'Seven dataset groups include chemical studies and two related Lara arms; not seven independent genetic studies.',
            'Q lacks complete source-cell and split-half noise controls.',
            'OOF provenance of upstream predictions is not established by risk-learner cold buckets.',
        ],
    )
    (args.output / 'AUDIT.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
