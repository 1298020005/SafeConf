#!/usr/bin/env python3
"""Metadata-only eligibility audit; it does NOT claim a Q-adjusted result.

The seven existing E256 studies have prediction/error rows, but that table
does not contain source cell counts or split-half uncertainty. Raw-cell
metadata can show whether reconstruction is plausible; it cannot silently
be joined to a different fold/task preprocessing contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import pandas as pd


RAW = Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb')
E256 = Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
STUDIES = (
    ('CuiHacohen2023', 'CuiHacohen2023.h5ad', 'celltype', ()),
    ('Frangieh', 'FrangiehIzar2021_RNA.h5ad', None, ()),
    ('LaraAstiasoHuntly2023_exvivo', 'LaraAstiasoHuntly2023_exvivo.h5ad', 'celltype', ()),
    ('LaraAstiasoHuntly2023_invivo', 'LaraAstiasoHuntly2023_invivo.h5ad', 'celltype', ()),
    ('McFarlandTsherniak2020', 'McFarlandTsherniak2020.h5ad', 'cell_line', ('dose_value', 'time')),
    ('SantinhaPlatt2023', 'SantinhaPlatt2023.h5ad', 'cell_types', ()),
    ('SrivatsanTrapnell2020_sciplex3', 'SrivatsanTrapnell2020_sciplex3.h5ad', 'celltype', ('dose_value', 'time')),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(args: argparse.Namespace) -> list[dict]:
    if args.output.exists():
        raise FileExistsError(args.output)
    table = pd.read_csv(E256, usecols=['dataset_name', 'context', 'perturbation',
                                        'predictor_name', 'split', 'fold_id',
                                        'prediction_l2_norm', 'true_error_rmse',
                                        'perturbation_support_count',
                                        'perturbation_effect_variance'])
    rows = []
    for study, filename, context_col, modifiers in STUDIES:
        path = RAW / filename
        if not path.exists():
            rows.append({'study': study, 'status': 'RAW_MISSING', 'raw_path': str(path)})
            continue
        obj = ad.read_h5ad(path, backed='r')
        try:
            obs = obj.obs
            sub = table.loc[table.dataset_name.eq(study)]
            test = sub.loc[sub.predictor_name.eq('PertMeanPredictor') & sub.split.eq('test')]
            p = obs['perturbation'].astype(str)
            if context_col is None or context_col not in obs:
                repeated = 0
                raw_contexts = 0
                controls = 0
                status = 'CONTEXT_MAPPING_REQUIRED'
            else:
                c = obs[context_col].astype(str)
                good = p.notna() & ~p.isin(('nan', 'control', 'ctrl', 'None')) & \
                    c.notna() & ~c.isin(('nan', 'None'))
                counts = pd.DataFrame({'context': c[good], 'perturbation': p[good]}) \
                    .groupby(['context', 'perturbation'], observed=True).size()
                supported = counts[counts >= 30].reset_index(name='cells')
                repeated = int((supported.groupby('perturbation').context.nunique() >= 2).sum())
                raw_contexts = int(c.nunique())
                controls = int((p.eq('control') | p.eq('ctrl')).sum())
                if repeated == 0:
                    status = 'NO_REPEATED_SUPPORTED_PERTURBATIONS'
                elif modifiers:
                    status = 'DOSE_TIME_CONTRACT_REQUIRED'
                else:
                    status = 'REBUILD_POSSIBLE_NOT_ALIGNED'
            rows.append({
                'study': study, 'status': status, 'raw_path': str(path),
                'raw_cells': int(obj.n_obs), 'raw_genes': int(obj.n_vars),
                'raw_context_field': context_col or '',
                'raw_contexts': raw_contexts,
                'raw_control_cells': controls,
                'raw_perturbations_repeated_in_2plus_contexts_30cells': repeated,
                'additional_task_modifiers': ','.join(modifiers),
                'e256_test_prediction_error_rows': int(len(test)),
                'e256_test_folds': int(test.fold_id.nunique()),
                'e256_test_contexts': int(test.context.nunique()),
                'e256_history_variance_missing_rate':
                    float(test.perturbation_effect_variance.isna().mean()),
                'e256_support_is_source_cell_count': False,
                'exact_fold_safe_Q_available': False,
                'split_half_Q_available_in_E256': False,
                'uniform_M_Q_H_comparison_completed': False,
                'selection_policy': 'all seven historical studies audited, independent of result sign',
            })
        finally:
            obj.file.close()
        print(f'{study}: {rows[-1]["status"]}, repeated={repeated}', flush=True)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    status = {
        'status': 'ELIGIBILITY_ONLY_NOT_BIOLOGICAL_HISTORY_RESULT',
        'e256_source_sha256': sha256(E256),
        'n_studies': len(rows),
        'n_uniform_M_Q_H_results': 0,
        'note': 'Raw h5ad metadata is not automatically the exact cells used by E256 folds; source/train manifests and preprocessing must be aligned before Q reconstruction.',
        'no_test_donor_target_truth_read': True,
        'table': str(args.output),
    }
    args.output.with_suffix('.status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'n_studies': len(rows), 'statuses': frame.status.value_counts().to_dict()},
                     ensure_ascii=False, indent=2), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args())
