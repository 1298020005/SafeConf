"""Read-only reproduction of the postseal Source DEV biological diagnostic.

Supply a fresh --output directory. No model fitting or new random draws.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.scripts import run_dual_memory_txpert_public_biology as tx
from tools.scripts import run_safeconf_continual_runtime_replay_agent as old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    report = old.DOC_BASE / 'continual_runtime_replay/public_biology_refit_v2'
    runtime = old.COMMON.parent / 'continual_public_biology_refit_20261002_v2'
    reg = json.loads((report / 'REGISTRATION.json').read_text())
    seal = json.loads((report / 'PRE_GATE_MODEL_PRIOR_PREDICTION_FREEZE.json').read_text())
    for binding in seal['runtime_artifacts']:
        assert old.sha(binding['path']) == binding['sha256']
    arms = reg['arms']
    scope = pd.read_csv(old.DOC_BASE / 'continual_runtime_replay/actual_v2/FROZEN_TASK_ROLE_SCOPE.csv')
    tasks = scope[scope.initial_known_history].reset_index(drop=True)
    all_tasks = pd.read_csv(old.COMMON / 'risk_cache/TX_TASK_SPLIT.csv')
    source_rows = {task: row for row, task in enumerate(all_tasks.task_id)}
    roles = ['OLD_ANCHOR', 'NEW_TASK_GATE']
    positions = np.flatnonzero(tasks.replay_role.isin(roles).to_numpy())
    evaluation = tasks.iloc[positions].reset_index(drop=True)
    assert len(evaluation) == 680 and evaluation.gene.nunique() == 227
    truth = np.asarray(np.load(old.COMMON / 'SOURCE_TRUE_EFFECTS.npy', mmap_mode='r')
                       [[source_rows[t] for t in evaluation.task_id]], float)
    boot = np.load(old.COMMON.parent / 'continual_component_diagnostic_20261002_v1/JOINT_GENE_BOOTSTRAP_COUNTS.npz', allow_pickle=False)
    assert boot['genes'].tolist() == sorted(evaluation.gene.unique())
    gene_rows = evaluation.gene.map({g: row for row, g in enumerate(boot['genes'].tolist())}).to_numpy(int)
    values = {}
    for arm in arms:
        priors = np.load(runtime / f'{arm}_PRIORS.npy', mmap_mode='r')
        assert priors.shape == (1699, 2840)
        priors = np.asarray(priors[positions], float)
        values[arm] = np.c_[tx.rmse_rows(priors, truth), tx.cosine_rows(priors, truth)]
    macros, strata, contrasts = [], [], []
    for role in roles:
        points, draws = [], []
        for context in old.CONTEXTS:
            use = np.flatnonzero((evaluation.replay_role.eq(role) & evaluation.target.eq(context)).to_numpy())
            assert len(use) >= 20
            weights = boot['counts'][:, gene_rows[use]].astype(float)
            denominator = weights.sum(axis=1)
            point, draw = [], []
            for arm in arms:
                v = values[arm][use]
                point.append(v.mean(axis=0))
                draw.append(weights @ v / denominator[:, None])
                strata.append(dict(role=role, context=context, arm=arm, biological_tasks=len(use),
                                   rmse_mean=v[:, 0].mean(), cosine_mean=v[:, 1].mean()))
            points.append(point)
            draws.append(np.stack(draw, axis=1))
        point, draw = np.mean(points, axis=0), np.mean(draws, axis=0)
        for a, arm in enumerate(arms):
            macros.append(dict(role=role, arm=arm, independent_biological_tasks=int(evaluation.replay_role.eq(role).sum()),
                               contexts=4, rmse_mean=point[a, 0], cosine_mean=point[a, 1]))
        for a, b in [(2, 0), (1, 0), (2, 1)]:
            delta = draw[:, a] - draw[:, b]
            for metric, name in enumerate(['rmse_mean', 'cosine_mean']):
                finite = delta[:, metric][np.isfinite(delta[:, metric])]
                lower, upper = np.quantile(finite, [.025, .975])
                contrasts.append(dict(role=role, arm_a=arms[a], arm_b=arms[b], metric=name,
                                      point_difference=point[a, metric] - point[b, metric],
                                      ci95_lower=lower, ci95_upper=upper, valid_draws=len(finite), total_draws=5000))
    for name, rows in [('ROOT_BIOLOGY_RECONSTRUCTION_MACRO.csv', macros),
                       ('ROOT_BIOLOGY_RECONSTRUCTION_STRATA.csv', strata),
                       ('ROOT_BIOLOGY_RECONSTRUCTION_INTERVALS.csv', contrasts)]:
        pd.DataFrame(rows).to_csv(out / name, index=False, lineterminator='\n')


if __name__ == '__main__':
    main()
