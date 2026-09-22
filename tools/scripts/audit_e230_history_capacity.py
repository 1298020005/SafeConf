#!/usr/bin/env python3
"""Recompute E230 from sealed features; export complete capacity curves and audit."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e230_history_capacity as experiment


def compare(a, b, keys):
    pd.testing.assert_frame_equal(a.sort_values(keys).reset_index(drop=True),
                                 b.sort_values(keys).reset_index(drop=True),
                                 check_exact=False, atol=1e-11, rtol=1e-10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--truth', type=Path, required=True)
    args = p.parse_args(); out = args.result_dir
    state = json.loads((out/'RUN_STATUS.json').read_text())
    seal = json.loads((out/'FEATURE_SEAL.json').read_text())['files']
    if state['status'] != 'COMPLETE':
        raise ValueError('Incomplete run')
    expected = pd.read_csv(out/'RESULTS.csv')
    actual = experiment.evaluate(out, seal, args.truth)
    keys = ['target', 'history_mode', 'fraction', 'seed', 'k_sources', 'scope', 'method', 'budget']
    compare(actual, expected, keys)
    compare(experiment.contrasts(actual), pd.read_csv(out/'CONTRASTS.csv'), ['contrast'])
    records = []
    for item in seal:
        f = pd.read_csv(out/item['file'])
        if any(c in f for c in ('family_rms_error', 'family_centroid_rmse', 'truth', 'mse')):
            raise ValueError('Target outcome in sealed features')
        if f.groupby('task_id')[['predicted_magnitude','family_disagreement']].nunique().to_numpy().max() != 1:
            raise ValueError('Changed upstream predictions')
        matched = f[f.history_mode.eq('matched30')]
        if not matched.n_source_cells.eq(30).all():
            raise ValueError('Unmatched cell budget')
        if not matched.n_source_contexts.eq(matched.k_sources).all():
            raise ValueError('Unmatched context count')
        for _, group in f.groupby(['history_mode','fraction','seed','k_sources']):
            if group.task_id.duplicated().any():
                raise ValueError('Duplicate task in a configuration')
            recomputed = experiment.score_features(group)
            np.testing.assert_allclose(recomputed[list(experiment.SCORES)], group[list(experiment.SCORES)], atol=1e-10)
        for row in item['source_audit']:
            if row['target'] == row['source_context'] or row['target_perturbed_cells_read']:
                raise ValueError('Target experimental expression in source')
        all_groups = f[f.history_mode.eq('fraction')].groupby(['seed','k_sources'])
        for _, group in all_groups:
            counts = group.pivot(index='task_id',columns='fraction',values='n_source_cells')
            if not (np.diff(counts.to_numpy(),axis=1) >= 0).all():
                raise ValueError('Cell counts not nested')
        records.append({'target':item['target'],'rows':len(f),
            'unique_tasks':int(f.task_id.nunique()),
            'matched_primary_tasks':int(matched[matched.analysis_stratum.eq('primary_ge30')].task_id.nunique()),
            'target_source_expression_rows_read':0,
            'max_full_source_parity_error':max(v['max_abs_difference'] for v in item['parity_checks'])})
    a=actual[(actual.scope=='primary') & (actual.budget==.2)]
    curves=a.groupby(['target','history_mode','fraction','k_sources','method'],as_index=False)[
        ['utility','capture','remaining_relative_error','n_tasks']].mean()
    curves.to_csv(out/'CAPACITY_CURVES.csv',index=False)
    full=curves[(curves.history_mode=='fraction') & (curves.fraction==1) & (curves.k_sources==3)]
    wide=full.pivot(index='target',columns='method',values='utility').sort_index()
    wide.to_csv(out/'FULL_CAPACITY_COMPARATORS.csv')
    delta=wide.m_plus_history-wide.m_plus_d
    rng=np.random.default_rng(20260922)
    boot=delta.to_numpy()[rng.integers(0,4,(20000,4))].mean(axis=1)
    # Secondary contrast: neither a replacement for the preregistered primary nor a new success gate.
    secondary={'comparison':'full_history_minus_fixed_m_plus_d','scope':'secondary_development',
        'delta':float(delta.mean()),'ci95_lower':float(np.quantile(boot,.025)),
        'ci95_upper':float(np.quantile(boot,.975)),'positive_targets':int((delta>0).sum())}
    summary={'audited_at':datetime.now().astimezone().isoformat(),
        'experiment_protocol_commit':'116cc9f','execution_script_sha256':state['script_sha256'],
        'metrics_recomputed':True,'scores_recomputed':True,'prediction_fields_fixed':True,
        'source_capacity_counts_nested':True,'matched_cells_equal_30':True,
        'primary_tasks':1808,'all_tasks':2008,'targets':records,
        'secondary_contrast':secondary,'development_gate':state['development_gate'],
        'scope_note':'Retrospective E201 development analysis, four target contexts; not four new independent studies.',
        'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}}
    (out/'RESULT_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
