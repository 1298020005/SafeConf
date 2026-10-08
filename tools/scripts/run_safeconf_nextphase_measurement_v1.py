#!/usr/bin/env python3
"""Paired DEV evidence for frozen public rules across registered repeat endpoints.

Reuses saved risks and already-opened development errors. No model fitting,
raw expression access, parameter selection or permanent-test access.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import bootstrap_u20, metrics

BASE = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/measurement'
DEFAULT = Path('/home/yyf/runtime_artifacts/safeconf_nextphase_20261008_v1/measurement_paired')
ENDPOINTS = ['true_error_rmse', 'guide_0_rmse', 'guide_1_rmse',
             'plate_0_rmse', 'plate_1_rmse', 'cell_0_rmse', 'cell_1_rmse',
             'WMSE_Welch_control_adaptation']
METHODS = ['Public_R_history', 'NegativeSupport', 'Magnitude']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'RUN_STATUS.json').exists():
        raise FileExistsError('Use a fresh output directory; preserve previous run.')
    started, cpu_started = time.monotonic(), time.process_time()
    risk_path = BASE / 'simple_risk_sensitivity_v1/FROZEN_SIMPLE_RISKS.csv.gz'
    error_path = BASE / 'DEV_REPEAT_TASK_ERRORS.csv.gz'
    dependency_path = BASE / 'simple_risk_sensitivity_v1/HISTORY_DEPENDENCY_AUDIT.csv'
    freeze = json.loads((BASE / 'simple_risk_sensitivity_v1/PREDICTION_FREEZE.json').read_text())
    assert sha(risk_path) == freeze['risk_file_sha256']
    risks = pd.read_csv(risk_path)
    errors = pd.read_csv(error_path).set_index('task_id').loc[risks.task_id].reset_index()
    dependencies = pd.read_csv(dependency_path)
    assert len(risks) == 542 and risks.gene.nunique() == 377
    assert risks.task_id.is_unique and errors.task_id.is_unique
    assert risks.gene.astype(str).equals(errors.gene.astype(str))
    assert risks.context.astype(str).equals(errors.context.astype(str))
    assert len(dependencies) == 542 and (dependencies.forbidden_intersection == 0).all()
    assert np.isfinite(risks[METHODS]).all().all()
    config = {
        'run_id': 'safeconf_nextphase_20261008_v1_measurement_paired',
        'role': 'DEV/SEEN diagnosis, frozen scores; no method selection',
        'n_tasks': 542, 'n_genes': 377,
        'primary_endpoint': 'true_error_rmse',
        'diagnostic_endpoints': ENDPOINTS[1:],
        'primary_comparison': 'Public_R_history minus NegativeSupport',
        'secondary_comparison': 'Public_R_history minus Magnitude',
        'methods': METHODS, 'bootstrap_replicates': 5000, 'bootstrap_seed': 20260930,
        'resampling': 'paired gene clusters, all contexts of a sampled gene together',
        'aggregation': 'context-equal macro; endpoints never pooled together',
        'missingness': 'common finite rows per endpoint, fixed independently of scores',
        'repeat_scope': 'registered guide/plate/cell splits; biological independence not inferred from labels',
        'wmse_scope': 'existing Welch-control adaptation; diagnostic endpoint only',
        'new_model_fits': 0, 'new_gpu_hours': 0, 'new_download_bytes': 0,
        'permanent_test_truth_opened': False,
        'input_hashes': {str(p): sha(p) for p in [risk_path, error_path, dependency_path]},
        'script_sha256': sha(__file__)
    }
    write_json(out / 'CONFIG.json', config)
    write_json(out / 'RUN_STATUS.json', {'status': 'RUNNING', 'completed_endpoints': []})
    old = pd.read_csv(BASE / 'simple_risk_sensitivity_v1/MACRO_METRICS.csv')
    paired, strata, macro, coverage = [], [], [], []
    for number, endpoint in enumerate(ENDPOINTS, 1):
        frame = risks.copy()
        frame['true_error_rmse'] = errors[endpoint].to_numpy(float)
        frame['target'] = frame.context.astype(str)
        valid = np.isfinite(frame.true_error_rmse.to_numpy())
        frame = frame.loc[valid].reset_index(drop=True)
        for context, part in frame.groupby('context', sort=True):
            coverage.append({'endpoint': endpoint, 'context': context,
                             'valid_tasks': len(part), 'valid_genes': part.gene.nunique(),
                             'planned_tasks': int((risks.context == context).sum())})
        for method in METHODS:
            items = []
            for context, part in frame.groupby('context', sort=True):
                result = metrics(part, part[method].to_numpy())
                strata.append({'endpoint': endpoint, 'method': method, 'context': context, **result})
                items.append(result)
            values = {key: float(np.nanmean([item[key] for item in items]))
                      for key in ['utility20', 'aurc', 'spearman', 'high_risk_miss_rate']}
            expected = old[(old.diagnostic_truth == endpoint) & (old.method == method)].iloc[0]
            assert np.isclose(values['utility20'], expected.utility20, atol=1e-10, rtol=0), (endpoint, method)
            macro.append({'endpoint': endpoint, 'method': method, 'n_tasks': len(frame),
                          'n_genes': frame.gene.nunique(), **values})
        for comparator in ['NegativeSupport', 'Magnitude']:
            result = bootstrap_u20(frame, frame.Public_R_history.to_numpy(),
                                   frame[comparator].to_numpy(), 5000, 20260930)
            paired.append({'endpoint': endpoint, 'comparison': 'Public_R_history-minus-' + comparator, **result})
        for name, values in [('PAIRED_COMPARISONS', paired), ('STRATA', strata),
                             ('MACRO_RESULTS', macro), ('COVERAGE', coverage)]:
            pd.DataFrame(values).to_csv(out / (name + '.csv'), index=False, lineterminator='\n')
        write_json(out / 'RUN_STATUS.json', {
            'status': 'COMPLETE' if number == len(ENDPOINTS) else 'RUNNING',
            'completed_endpoints': ENDPOINTS[:number],
            'wall_seconds': time.monotonic() - started,
            'cpu_seconds': time.process_time() - cpu_started,
            'new_model_fits': 0, 'new_gpu_hours': 0,
            'new_download_bytes': 0, 'permanent_test_truth_opened': False,
            'old_point_estimates_reproduced': True,
            'default_changed': False
        })
        print(json.dumps({'endpoint': endpoint, 'completed': number, 'total': len(ENDPOINTS),
                          'latest_comparisons': paired[-2:]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
